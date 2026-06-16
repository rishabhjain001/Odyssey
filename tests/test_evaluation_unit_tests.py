# Feature: odyssey, Task 13.3: Unit tests for evaluation recording
"""Unit tests for :func:`training.evaluator.evaluate` evaluation recording.

These plain ``pytest`` unit tests verify three recording behaviours using a
``FakeEnv`` (test double) that plays back a configurable per-step script.
No real Gymnasium / Box2D environment is required.

Test coverage
-------------
1. **Track-completion flag** (Req 11.3): when an episode ends with
   ``terminated=True, truncated=False`` the completion rate is > 0; when it
   ends with ``truncated=True`` the completion rate is 0.
2. **Cumulative-step recording** (Req 11.5): passing ``env_steps_so_far=1000``
   and running 2 × 5-step episodes yields
   ``EvalResult.cumulative_env_steps == 1010``.
3. **Transition appending** (Req 11.6): after ``evaluate()`` the replay buffer
   holds exactly ``episodes × steps_per_episode`` transitions (one per valid
   env step with a valid observation).

Validates: Requirements 11.3, 11.5, 11.6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import torch

from agent.config import ScopeConfig
from agent.replay import ReplayBuffer
from agent.types import EvalResult, LatentState
from training.evaluator import evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid raw CarRacing-v2 obs shape (Req 1.2)
_OBS_SHAPE: Tuple[int, int, int] = (96, 96, 3)

# ScopeConfig used throughout — 1-channel MVP grayscale, sequence_length=50
_SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_obs() -> np.ndarray:
    """Return a fresh valid raw observation (mid-range gray, (96,96,3) uint8)."""
    return np.full(_OBS_SHAPE, 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# FakeEnv
# ---------------------------------------------------------------------------


@dataclass
class _StepScript:
    """Per-step specification for FakeEnv playback.

    Attributes:
        obs:         The observation to return from this step.
        reward:      The reward to return.
        terminated:  The ``terminated`` flag.
        truncated:   The ``truncated`` flag.
    """

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


class FakeEnv:
    """Injectable fake environment driven by a per-step script.

    The script is a list of :class:`_StepScript` objects that are consumed in
    order across *all* episodes.  When the script is exhausted the env raises
    ``StopIteration`` so tests will fail loudly rather than silently hang.

    The ``reset`` method always returns a valid ``(96,96,3) uint8`` observation
    regardless of where in the script we are; it does **not** consume a script
    entry.

    ``action_space`` exposes a ``sample`` method that returns a fixed valid action
    so the evaluator can call it if needed (the evaluator in zero-state mode just
    passes the actor's output — the action space is not strictly required here,
    but including it keeps the interface compatible with any future assertion).
    """

    class _ActionSpace:
        def sample(self) -> np.ndarray:
            return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    def __init__(self, script: List[_StepScript]) -> None:
        self._script = list(script)
        self._cursor = 0
        self.reset_count = 0
        self.step_count = 0
        self.action_space = FakeEnv._ActionSpace()

    def reset(self, **kwargs):
        self.reset_count += 1
        return _valid_obs(), {}

    def step(self, action):
        if self._cursor >= len(self._script):
            raise StopIteration("FakeEnv script exhausted.")
        entry = self._script[self._cursor]
        self._cursor += 1
        self.step_count += 1
        return entry.obs, entry.reward, entry.terminated, entry.truncated, {}


# ---------------------------------------------------------------------------
# Fake actor  (zero-state mode — no RSSM/Encoder needed)
# ---------------------------------------------------------------------------


class _FakeActor:
    """Actor that always returns a fixed action tensor."""

    def __call__(self, state: LatentState) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Script builders
# ---------------------------------------------------------------------------


def _build_script(
    n_episodes: int,
    steps_per_episode: int,
    terminated: bool,
    truncated: bool,
) -> List[_StepScript]:
    """Build a flat script for ``n_episodes`` each of ``steps_per_episode`` steps.

    The last step of every episode carries the provided ``terminated`` /
    ``truncated`` flags; all other steps have both flags False.
    """
    script: List[_StepScript] = []
    for _ in range(n_episodes):
        for step in range(steps_per_episode):
            is_last = step == steps_per_episode - 1
            script.append(
                _StepScript(
                    obs=_valid_obs(),
                    reward=1.0,
                    terminated=terminated if is_last else False,
                    truncated=truncated if is_last else False,
                )
            )
    return script


def _make_buffer(capacity: int = 1000) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity)


# ===========================================================================
# Test group 1 — Track-completion flag (Req 11.3)
# ===========================================================================


class TestTrackCompletionFlag:
    """Verify completion_rate accounting based on the terminated/truncated flags."""

    def test_terminated_episode_contributes_to_completion_rate(self) -> None:
        """terminated=True, truncated=False → completion_rate > 0.

        Convention (from evaluator docstring and Req 11.3):
            track_completed = terminated and not truncated
        """
        script = _build_script(
            n_episodes=2, steps_per_episode=3, terminated=True, truncated=False
        )
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=0,
            episodes=2,
            scope=_SCOPE,
        )

        assert result.completion_rate > 0.0, (
            "Expect completion_rate > 0 when episodes end with terminated=True"
        )
        assert result.completion_rate == pytest.approx(1.0), (
            "Both episodes terminated naturally — completion_rate should be 1.0"
        )

    def test_truncated_episode_gives_zero_completion_rate(self) -> None:
        """truncated=True → completion_rate == 0 (time-limit exit, not track done).

        Requirement 11.3 defines track completion as the environment signalling
        *all track tiles have been visited*, which maps to ``terminated=True`` AND
        ``truncated=False``.  A pure truncation therefore scores 0.
        """
        script = _build_script(
            n_episodes=2, steps_per_episode=3, terminated=False, truncated=True
        )
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=0,
            episodes=2,
            scope=_SCOPE,
        )

        assert result.completion_rate == pytest.approx(0.0), (
            "Expect completion_rate == 0.0 when episodes end only via truncation"
        )

    def test_mixed_completion(self) -> None:
        """One terminated, one truncated → completion_rate == 0.5."""
        # Episode 1: 3 steps, terminates naturally
        # Episode 2: 3 steps, truncated (time limit)
        script = [
            # Episode 1
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=False, truncated=False),
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=False, truncated=False),
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=True, truncated=False),
            # Episode 2
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=False, truncated=False),
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=False, truncated=False),
            _StepScript(obs=_valid_obs(), reward=1.0, terminated=False, truncated=True),
        ]
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=0,
            episodes=2,
            scope=_SCOPE,
        )

        assert result.completion_rate == pytest.approx(0.5), (
            "1 out of 2 episodes completed → completion_rate should be 0.5"
        )


# ===========================================================================
# Test group 2 — Cumulative-step recording (Req 11.5)
# ===========================================================================


class TestCumulativeStepRecording:
    """Verify EvalResult.cumulative_env_steps = env_steps_so_far + steps_this_eval."""

    def test_cumulative_steps_offset_by_env_steps_so_far(self) -> None:
        """Passing env_steps_so_far=1000 with 2 episodes × 5 steps → 1010.

        Req 11.5: cumulative_env_steps records the *total* environment steps
        consumed by Odyssey at the time of the evaluation, i.e.
            prior_steps + steps_taken_this_run
        """
        n_episodes = 2
        steps_per_ep = 5
        env_steps_so_far = 1000

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=env_steps_so_far,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        expected = env_steps_so_far + (n_episodes * steps_per_ep)  # 1010
        assert result.cumulative_env_steps == expected, (
            f"Expected cumulative_env_steps={expected}, got {result.cumulative_env_steps}"
        )

    def test_cumulative_steps_zero_prior(self) -> None:
        """With env_steps_so_far=0, cumulative_env_steps equals steps taken."""
        n_episodes = 3
        steps_per_ep = 4

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=0,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        expected = n_episodes * steps_per_ep  # 12
        assert result.cumulative_env_steps == expected

    def test_cumulative_steps_large_prior(self) -> None:
        """Large prior offset is correctly carried through."""
        env_steps_so_far = 999_999
        n_episodes = 1
        steps_per_ep = 7

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        result = evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=_make_buffer(),
            env_steps_so_far=env_steps_so_far,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        expected = env_steps_so_far + steps_per_ep
        assert result.cumulative_env_steps == expected


# ===========================================================================
# Test group 3 — Transition appending (Req 11.6)
# ===========================================================================


class TestTransitionAppending:
    """Verify evaluate() stores exactly episodes × steps_per_episode transitions."""

    def test_replay_buffer_contains_all_transitions(self) -> None:
        """After evaluate(), replay buffer has episodes × steps_per_episode entries.

        All observations produced by FakeEnv are valid (96,96,3) uint8, so
        every step's transition must be stored (Req 11.6).
        """
        n_episodes = 2
        steps_per_ep = 5

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        buffer = _make_buffer(capacity=1000)

        evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=buffer,
            env_steps_so_far=0,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        expected_transitions = n_episodes * steps_per_ep  # 10
        assert len(buffer) == expected_transitions, (
            f"Expected {expected_transitions} transitions in the buffer, "
            f"got {len(buffer)}"
        )

    def test_replay_buffer_pre_existing_transitions_are_preserved(self) -> None:
        """Transitions already in the buffer before evaluate() are not evicted.

        We pre-fill the buffer with 3 transitions, run 1 episode of 2 steps,
        and expect 5 transitions total (3 pre-existing + 2 new).
        """
        from agent.types import Transition as T_

        n_episodes = 1
        steps_per_ep = 2

        buffer = _make_buffer(capacity=1000)
        # Pre-fill with 3 valid transitions
        for _ in range(3):
            buffer.add(T_(
                observation=_valid_obs(),
                action=np.array([0.0, 0.5, 0.0], dtype=np.float32),
                reward=0.0,
                terminated=False,
            ))
        assert len(buffer) == 3

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=buffer,
            env_steps_so_far=0,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        assert len(buffer) == 3 + n_episodes * steps_per_ep

    def test_replay_buffer_single_episode_single_step(self) -> None:
        """Edge case: 1 episode × 1 step → exactly 1 transition stored."""
        script = _build_script(
            n_episodes=1, steps_per_episode=1, terminated=True, truncated=False
        )
        env = FakeEnv(script)
        buffer = _make_buffer()

        evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=buffer,
            env_steps_so_far=0,
            episodes=1,
            scope=_SCOPE,
        )

        assert len(buffer) == 1

    def test_replay_buffer_multiple_episodes_longer_runs(self) -> None:
        """5 episodes × 8 steps → 40 transitions stored."""
        n_episodes = 5
        steps_per_ep = 8

        script = _build_script(
            n_episodes=n_episodes,
            steps_per_episode=steps_per_ep,
            terminated=True,
            truncated=False,
        )
        env = FakeEnv(script)
        buffer = _make_buffer(capacity=500)

        evaluate(
            actor=_FakeActor(),
            env=env,
            replay_buffer=buffer,
            env_steps_so_far=0,
            episodes=n_episodes,
            scope=_SCOPE,
        )

        expected = n_episodes * steps_per_ep  # 40
        assert len(buffer) == expected


# ---------------------------------------------------------------------------
# Regression test: evaluate() with the REAL Encoder + RSSM + Actor (Req 11.1, 11.6)
# ---------------------------------------------------------------------------
#
# This guards against an integration bug where the evaluator passed the
# Encoder's CategoricalLatent object straight into RSSM.posterior() (which
# expects a flattened tensor embedding). The earlier unit tests used fake
# actors/encoders and so never exercised the real encoder->rssm wiring.


class _RealComponentsFakeEnv:
    """Minimal env that ends each episode after a fixed number of steps."""

    class _ActionSpace:
        def sample(self):
            return np.zeros(3, dtype=np.float32)

        def seed(self, s):  # noqa: ANN001
            pass

    def __init__(self, steps_per_episode: int = 10) -> None:
        self.action_space = self._ActionSpace()
        self._steps_per_episode = steps_per_episode
        self._n = 0

    def reset(self, **kwargs):  # noqa: ANN003
        self._n = 0
        return _valid_obs(), {}

    def step(self, action):  # noqa: ANN001
        self._n += 1
        terminated = self._n >= self._steps_per_episode
        return _valid_obs(), 1.0, terminated, False, {}

    def close(self) -> None:
        pass


def test_evaluate_with_real_encoder_rssm_actor_rssm_mode():
    """evaluate() runs end-to-end with the real Encoder/RSSM/Actor (rssm mode).

    Reproduces the integration path used by train.py: the encoder output must be
    flattened before being handed to RSSM.posterior(). Asserts the run returns a
    valid EvalResult and that transitions were appended to the replay buffer.

    Validates: Requirements 11.1, 11.2, 11.4, 11.5, 11.6
    """
    from agent.encoder import Encoder
    from agent.rssm import RSSM
    from agent.behavior import Actor

    torch.manual_seed(0)
    encoder = Encoder(channels=1)
    rssm = RSSM(action_dim=3)
    actor = Actor()
    buffer = ReplayBuffer(capacity=1000)

    result = evaluate(
        actor=actor,
        env=_RealComponentsFakeEnv(steps_per_episode=10),
        replay_buffer=buffer,
        env_steps_so_far=0,
        episodes=2,
        scope=_SCOPE,
        rssm=rssm,
        encoder=encoder,
    )

    assert isinstance(result, EvalResult)
    # 2 episodes x 10 steps = 20 env steps consumed and stored.
    assert result.cumulative_env_steps == 20
    assert len(buffer) == 20
    # avg_reward is the arithmetic mean of per-episode totals (10.0 each here).
    assert result.avg_reward == pytest.approx(10.0)
    # Each episode ended via terminated (not truncated) => completion_rate 1.0.
    assert 0.0 <= result.completion_rate <= 1.0
