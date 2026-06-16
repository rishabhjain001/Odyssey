# Feature: odyssey, Property 22: Evaluation aggregates are computed correctly
"""Property-based tests verifying that evaluate() computes avg_reward and
completion_rate correctly across varying episode counts, reward sequences, and
termination modes.

Property 22: Evaluation aggregates are computed correctly

    - avg_reward equals the arithmetic mean of per-episode total rewards.
    - completion_rate = completed_episodes / total_episodes ∈ [0.0, 1.0].

A FakeEnv is built that terminates after a configurable number of steps and
emits configurable per-step rewards and terminated/truncated flags per episode.

FakeEnv normalization rule: if both terminated and truncated are False on the
final step the episode would never end, so the FakeEnv forces terminated=True
on the last step. The expected computations in all tests account for this.

Validates: Requirements 11.2, 11.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.config import ScopeConfig
from agent.replay import ReplayBuffer
from agent.types import LatentState
from training.evaluator import evaluate

# Shared scope — any valid scope works; we use MVP
_SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

def _dummy_obs() -> np.ndarray:
    """Return a valid (96, 96, 3) uint8 observation filled with zeros."""
    return np.zeros((96, 96, 3), dtype=np.uint8)


@dataclass
class _EpisodeSpec:
    """Specification for one fake episode."""
    rewards: List[float]    # per-step rewards
    terminated: bool        # requested terminated flag at the final step
    truncated: bool         # requested truncated flag at the final step

    @property
    def effective_terminated(self) -> bool:
        """The terminated flag actually emitted (normalized so episode always ends)."""
        if not self.terminated and not self.truncated:
            return True  # force-close: FakeEnv must emit something
        return self.terminated

    @property
    def effective_truncated(self) -> bool:
        """The truncated flag actually emitted."""
        if not self.terminated and not self.truncated:
            return False
        return self.truncated

    @property
    def is_completed(self) -> bool:
        """Whether this episode counts as completed (terminated=True, not truncated)."""
        return self.effective_terminated and not self.effective_truncated


class _FakeEnv:
    """Fake environment that plays back a sequence of episode specs.

    Each episode runs for len(spec.rewards) steps with the specified reward.
    At the final step the episode's terminated/truncated flags are emitted.
    If neither flag is True the env forces terminated=True to ensure the
    episode always ends (avoids infinite loops in the evaluator).
    """

    def __init__(self, episode_specs: List[_EpisodeSpec]) -> None:
        self._specs = episode_specs
        self._ep_idx = 0
        self._step_idx = 0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Any]:
        if self._ep_idx < len(self._specs):
            self._step_idx = 0
        return _dummy_obs(), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Any]:
        if self._ep_idx >= len(self._specs):
            # Beyond configured episodes — return immediate terminal
            return _dummy_obs(), 0.0, True, False, {}

        spec = self._specs[self._ep_idx]
        reward = spec.rewards[self._step_idx]
        self._step_idx += 1
        is_last_step = self._step_idx >= len(spec.rewards)

        if is_last_step:
            terminated = spec.effective_terminated
            truncated = spec.effective_truncated
            self._ep_idx += 1
            self._step_idx = 0
        else:
            terminated = False
            truncated = False

        return _dummy_obs(), reward, terminated, truncated, {}


class _FakeActor:
    """Fake actor that always returns a zero action for any LatentState."""

    def __call__(self, state: LatentState) -> "torch.Tensor":
        return torch.zeros(1, 3)


def _make_replay_buffer() -> ReplayBuffer:
    """Create a minimal replay buffer for the evaluator to write into."""
    return ReplayBuffer(capacity=100_000)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

@seed(20240623)
@settings(max_examples=20, deadline=None)
@given(
    n_episodes=st.integers(min_value=2, max_value=6),
    rewards_per_ep=st.lists(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=5,
        ),
        min_size=2, max_size=6,
    ),
)
def test_avg_reward_is_arithmetic_mean(
    n_episodes: int,
    rewards_per_ep: list,
) -> None:
    """avg_reward equals the arithmetic mean of per-episode total rewards.

    We construct a FakeEnv where each episode has a known per-step reward list.
    The expected avg_reward is sum(total_ep_rewards) / n_episodes.

    Validates: Requirements 11.2
    """
    # Trim or pad rewards_per_ep to exactly n_episodes
    while len(rewards_per_ep) < n_episodes:
        rewards_per_ep.append([0.0])
    rewards_per_ep = rewards_per_ep[:n_episodes]

    specs = [
        _EpisodeSpec(
            rewards=ep_rewards,
            terminated=True,
            truncated=False,
        )
        for ep_rewards in rewards_per_ep
    ]

    env = _FakeEnv(specs)
    actor = _FakeActor()
    replay = _make_replay_buffer()

    result = evaluate(
        actor=actor,
        env=env,
        replay_buffer=replay,
        env_steps_so_far=0,
        episodes=n_episodes,
        scope=_SCOPE,
    )

    # Compute expected avg reward
    expected_totals = [sum(ep_rewards) for ep_rewards in rewards_per_ep]
    expected_avg = sum(expected_totals) / n_episodes

    assert math.isclose(result.avg_reward, expected_avg, rel_tol=1e-4, abs_tol=1e-4), (
        f"avg_reward={result.avg_reward:.6f}, expected={expected_avg:.6f}\n"
        f"Episode totals: {expected_totals}"
    )


@seed(20240623)
@settings(max_examples=20, deadline=None)
@given(
    n_episodes=st.integers(min_value=2, max_value=6),
    episode_flags=st.lists(
        st.tuples(
            st.booleans(),   # terminated
            st.booleans(),   # truncated
        ),
        min_size=2, max_size=6,
    ),
)
def test_completion_rate_is_fraction_of_terminated_not_truncated(
    n_episodes: int,
    episode_flags: list,
) -> None:
    """completion_rate == completed_episodes / n_episodes, in [0.0, 1.0].

    An episode is "completed" when the effective terminated=True and
    effective truncated=False. The FakeEnv normalizes (False, False) to
    (True, False), which counts as completed. The expected value is
    computed from the effective flags to match what the evaluator observes.

    Validates: Requirements 11.4
    """
    # Trim or pad flags to exactly n_episodes
    while len(episode_flags) < n_episodes:
        episode_flags.append((True, False))
    episode_flags = episode_flags[:n_episodes]

    specs = [
        _EpisodeSpec(
            rewards=[0.0],      # single neutral step per episode
            terminated=term,
            truncated=trunc,
        )
        for (term, trunc) in episode_flags
    ]

    env = _FakeEnv(specs)
    actor = _FakeActor()
    replay = _make_replay_buffer()

    result = evaluate(
        actor=actor,
        env=env,
        replay_buffer=replay,
        env_steps_so_far=0,
        episodes=n_episodes,
        scope=_SCOPE,
    )

    # Expected completion rate: use effective flags from each spec
    expected_completed = sum(1 for spec in specs if spec.is_completed)
    expected_rate = expected_completed / n_episodes

    # Must be in [0.0, 1.0]
    assert 0.0 <= result.completion_rate <= 1.0, (
        f"completion_rate {result.completion_rate} outside [0.0, 1.0]"
    )

    assert math.isclose(result.completion_rate, expected_rate, rel_tol=1e-5, abs_tol=1e-5), (
        f"completion_rate={result.completion_rate:.6f}, expected={expected_rate:.6f}\n"
        f"Episode specs (effective): "
        f"{[(s.effective_terminated, s.effective_truncated) for s in specs]}, "
        f"completed: {expected_completed}"
    )


@seed(20240623)
@settings(max_examples=20, deadline=None)
@given(
    n_episodes=st.integers(min_value=2, max_value=6),
    rewards_per_ep=st.lists(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=3,
        ),
        min_size=2, max_size=6,
    ),
    episode_flags=st.lists(
        st.tuples(st.booleans(), st.booleans()),
        min_size=2, max_size=6,
    ),
)
def test_combined_avg_reward_and_completion_rate(
    n_episodes: int,
    rewards_per_ep: list,
    episode_flags: list,
) -> None:
    """Both avg_reward and completion_rate are correct simultaneously.

    This is the combined property check for Property 22.

    Validates: Requirements 11.2, 11.4
    """
    # Trim or pad to n_episodes
    while len(rewards_per_ep) < n_episodes:
        rewards_per_ep.append([0.0])
    rewards_per_ep = rewards_per_ep[:n_episodes]

    while len(episode_flags) < n_episodes:
        episode_flags.append((True, False))
    episode_flags = episode_flags[:n_episodes]

    specs = [
        _EpisodeSpec(
            rewards=ep_rewards,
            terminated=term,
            truncated=trunc,
        )
        for ep_rewards, (term, trunc) in zip(rewards_per_ep, episode_flags)
    ]

    env = _FakeEnv(specs)
    actor = _FakeActor()
    replay = _make_replay_buffer()

    result = evaluate(
        actor=actor,
        env=env,
        replay_buffer=replay,
        env_steps_so_far=0,
        episodes=n_episodes,
        scope=_SCOPE,
    )

    # Expected avg reward
    expected_totals = [sum(spec.rewards) for spec in specs]
    expected_avg = sum(expected_totals) / n_episodes

    # Expected completion rate using effective flags
    expected_completed = sum(1 for spec in specs if spec.is_completed)
    expected_rate = expected_completed / n_episodes

    # avg_reward
    assert math.isclose(result.avg_reward, expected_avg, rel_tol=1e-4, abs_tol=1e-4), (
        f"avg_reward={result.avg_reward:.6f}, expected={expected_avg:.6f}"
    )

    # completion_rate in [0.0, 1.0]
    assert 0.0 <= result.completion_rate <= 1.0

    # completion_rate value
    assert math.isclose(result.completion_rate, expected_rate, rel_tol=1e-5, abs_tol=1e-5), (
        f"completion_rate={result.completion_rate:.6f}, expected={expected_rate:.6f}"
    )
