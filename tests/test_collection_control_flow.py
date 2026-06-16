# Feature: odyssey, Task 3.7: Collection control flow unit tests
"""Unit tests for the random trajectory collector's control flow.

These plain ``pytest`` unit tests exercise the two control-flow guarantees of
:func:`training.pipeline.collect_random_trajectories` using a tiny fake
environment (a test double) so no real Gymnasium / Box2D environment is
required:

* **Stop-at-configured-step-count (Req 2.9).** Collection executes exactly
  ``config.env_steps`` environment steps and stops the instant that budget is
  reached.
* **Reset-on-termination (Req 2.8).** When an episode ends (termination or
  truncation), the environment is reset and collection continues toward the
  step budget; the initial reset is not counted in ``result.episodes``.

A handful of supporting assertions confirm uniform full-range action sampling
is sourced from ``action_space.sample`` (Req 2.1), that valid observations are
stored, and that a malformed observation is rejected (counted but not stored).

Validates: Requirements 2.1, 2.8, 2.9
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from agent.config import CollectionConfig, ScopeConfig
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from training.pipeline import collect_random_trajectories

# A valid raw CarRacing-v2 observation: (96, 96, 3) uint8 in [0, 255] (Req 1.2).
VALID_OBS_SHAPE = (96, 96, 3)


def _valid_obs() -> np.ndarray:
    """Return a fresh valid raw observation (mid-range gray)."""
    return np.full(VALID_OBS_SHAPE, 128, dtype=np.uint8)


def _malformed_obs() -> np.ndarray:
    """Return an observation that fails validation (wrong shape)."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


class FakeActionSpace:
    """Minimal stand-in for a Gymnasium ``Box`` action space.

    Records how many times ``sample`` and ``seed`` are called so tests can
    confirm actions are sourced from the action space (Req 2.1).
    """

    def __init__(self) -> None:
        self.sample_count = 0
        self.seed_count = 0
        self.last_seed: int | None = None

    def sample(self) -> np.ndarray:
        self.sample_count += 1
        # Uniform-over-full-range is the space's job; a fixed valid action is
        # sufficient for control-flow tests.
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def seed(self, seed: int) -> None:
        self.seed_count += 1
        self.last_seed = seed


class FakeEnv:
    """Configurable fake Gymnasium environment (test double).

    Args:
        terminate_at: Iterable of 1-based step indices at which ``step`` reports
            ``terminated=True``. For example, ``{3, 6, 9}`` ends an episode on
            the 3rd, 6th, and 9th step.
        truncate_at: Iterable of 1-based step indices at which ``step`` reports
            ``truncated=True``.
        malformed_at: Iterable of 1-based step indices whose ``step`` returns a
            malformed observation (to drive the rejection path).

    Tracks ``reset_count`` and ``step_count`` so tests can assert the collector's
    control flow precisely.
    """

    def __init__(
        self,
        *,
        terminate_at: Iterable[int] = (),
        truncate_at: Iterable[int] = (),
        malformed_at: Iterable[int] = (),
    ) -> None:
        self.action_space = FakeActionSpace()
        self._terminate_at = set(terminate_at)
        self._truncate_at = set(truncate_at)
        self._malformed_at = set(malformed_at)
        self.reset_count = 0
        self.step_count = 0
        self.last_reset_seed: int | None = None

    def reset(self, seed: int | None = None):
        self.reset_count += 1
        self.last_reset_seed = seed
        return _valid_obs(), {}

    def step(self, action):
        self.step_count += 1
        step_index = self.step_count
        terminated = step_index in self._terminate_at
        truncated = step_index in self._truncate_at
        if step_index in self._malformed_at:
            obs = _malformed_obs()
        else:
            obs = _valid_obs()
        reward = 1.0
        info: dict = {}
        return obs, reward, terminated, truncated, info


def _make_buffer() -> ReplayBuffer:
    """A replay buffer with ample capacity for the small budgets used here."""
    return ReplayBuffer(capacity=1000)


def _make_config(env_steps: int) -> CollectionConfig:
    return CollectionConfig(env_steps=env_steps, action_sampling="uniform")


def _preprocessor() -> FramePreprocessor:
    return FramePreprocessor(ScopeConfig(channels=1, sequence_length=50, name="mvp"))


# ---------------------------------------------------------------------------
# Stop-at-configured-step-count (Req 2.9)
# ---------------------------------------------------------------------------

def test_stops_at_configured_step_count() -> None:
    """The collector executes exactly ``config.env_steps`` steps (Req 2.9)."""
    env_steps = 10
    env = FakeEnv()
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert env.step_count == env_steps
    # Every step produced a valid observation, so all were stored.
    assert result.stored == env_steps
    # One action sampled per step from the action space (Req 2.1).
    assert env.action_space.sample_count == env_steps


@pytest.mark.parametrize("env_steps", [1, 2, 5, 25])
def test_step_count_exact_for_various_budgets(env_steps: int) -> None:
    """The exact-stop guarantee holds across several budgets (Req 2.9)."""
    env = FakeEnv()
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert env.step_count == env_steps


def test_no_extra_step_after_terminal_at_budget_end() -> None:
    """A termination on the final step does not trigger an extra reset (Req 2.8, 2.9).

    When the episode ends exactly as the budget is exhausted there is no room to
    continue, so the post-termination reset is skipped and is not counted.
    """
    env_steps = 5
    # Terminate on the very last step of the budget.
    env = FakeEnv(terminate_at={5})
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert env.step_count == env_steps
    # Only the initial reset happened; no post-termination reset at the boundary.
    assert env.reset_count == 1
    assert result.episodes == 0


# ---------------------------------------------------------------------------
# Reset-on-termination (Req 2.8)
# ---------------------------------------------------------------------------

def test_resets_on_termination() -> None:
    """Each mid-budget termination triggers a reset and counts an episode (Req 2.8)."""
    env_steps = 10
    # Terminations at steps 3, 6, 9 -- all strictly before the budget end.
    env = FakeEnv(terminate_at={3, 6, 9})
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert env.step_count == env_steps
    # Initial reset + one reset per termination that occurs before budget end.
    assert env.reset_count == 1 + 3
    # result.episodes counts only the post-termination resets.
    assert result.episodes == 3


def test_truncation_also_triggers_reset() -> None:
    """Truncation ends an episode and triggers a reset just like termination (Req 2.8)."""
    env_steps = 8
    env = FakeEnv(truncate_at={4})
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert env.step_count == env_steps
    assert env.reset_count == 1 + 1
    assert result.episodes == 1


def test_no_resets_when_no_episode_ends() -> None:
    """With no terminations, only the initial reset occurs (Req 2.8)."""
    env_steps = 6
    env = FakeEnv()
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert env.reset_count == 1
    assert result.episodes == 0


# ---------------------------------------------------------------------------
# Storage / rejection bookkeeping (supporting; Req 2.1)
# ---------------------------------------------------------------------------

def test_valid_observations_are_stored() -> None:
    """Validated transitions are added to the buffer (stored increments)."""
    env_steps = 7
    env = FakeEnv()
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.stored == env_steps
    assert result.rejected_observations == 0
    assert len(buffer) == env_steps


def test_malformed_observation_is_rejected_but_step_counts() -> None:
    """A malformed observation counts the step but is not stored (Req 2.9, 1.8)."""
    env_steps = 5
    # Step 3 yields a malformed observation.
    env = FakeEnv(malformed_at={3})
    buffer = _make_buffer()

    result = collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor()
    )

    assert result.env_steps == env_steps
    assert result.rejected_observations == 1
    assert result.stored == env_steps - 1
    assert len(buffer) == env_steps - 1


def test_seed_is_applied_to_reset_and_action_space() -> None:
    """A provided seed seeds the action space and the initial reset (Req 2.1)."""
    env_steps = 3
    env = FakeEnv()
    buffer = _make_buffer()

    collect_random_trajectories(
        env, buffer, _make_config(env_steps), preprocessor=_preprocessor(), seed=42
    )

    assert env.action_space.seed_count == 1
    assert env.action_space.last_seed == 42
    assert env.last_reset_seed == 42
