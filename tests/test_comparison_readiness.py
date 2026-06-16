# Feature: odyssey, Property 24: Comparisons require both agents to have completed training
"""Property-based tests for Property 24 of the Odyssey design.

**Property 24: Comparisons require both agents to have completed training**

*For any* request for a reward comparison or sample-efficiency comparison issued
before both Odyssey and the PPO_Baseline have completed training, the system
returns an error and produces no comparison.

**Validates: Requirements 14.7**

Implementation under test: ``training.metrics``
  - ``reward_comparison(odyssey_store, ppo_avg_reward, ppo_steps, min_episodes=5)``
    — raises :class:`~agent.errors.ComparisonNotReadyError` when odyssey has fewer
    than ``min_episodes`` eval records OR ``ppo_avg_reward`` is ``None``.
  - ``sample_efficiency_comparison(odyssey_store, ppo_results, step_counts)``
    — raises :class:`~agent.errors.ComparisonNotReadyError` when either series is empty.

Tests:
1. ``test_reward_comparison_raises_when_odyssey_has_too_few_records`` (Hypothesis):
   Vary the number of Odyssey eval records from 0 to min_episodes-1; assert
   ComparisonNotReadyError is raised and no value is returned.

2. ``test_reward_comparison_raises_when_ppo_not_ready`` (plain pytest):
   Pass a fully-populated store but set ppo_avg_reward=None; assert
   ComparisonNotReadyError is raised.

3. ``test_sample_efficiency_raises_when_odyssey_empty`` (plain pytest):
   Odyssey store has no eval records; assert ComparisonNotReadyError is raised.

4. ``test_sample_efficiency_raises_when_ppo_empty`` (plain pytest):
   PPO results list is empty; assert ComparisonNotReadyError is raised.
"""

from __future__ import annotations

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.errors import ComparisonNotReadyError
from training.metrics import MetricsStore, reward_comparison, sample_efficiency_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_with_n_eval_records(n: int) -> MetricsStore:
    """Return a MetricsStore that contains exactly *n* evaluation records."""
    store = MetricsStore()
    for i in range(n):
        store.record_eval(
            env_steps=(i + 1) * 100,
            avg_reward=float(i) * 1.5,
            completion_rate=min(1.0, float(i) * 0.1),
        )
    return store


# ---------------------------------------------------------------------------
# Property 24 – Part 1: reward_comparison raises when Odyssey has too few records
# ---------------------------------------------------------------------------

@seed(20240901)
@settings(max_examples=20, deadline=None)
@given(n_records=st.integers(min_value=0, max_value=4))
def test_reward_comparison_raises_when_odyssey_has_too_few_records(n_records: int):
    """Property 24 (reward_comparison — insufficient Odyssey records):
    For any Odyssey store with fewer than min_episodes (5) evaluation records,
    reward_comparison raises ComparisonNotReadyError and produces no result.

    **Validates: Requirements 14.7**
    """
    # min_episodes default is 5; n_records is in [0, 4], i.e. always too few.
    store = _store_with_n_eval_records(n_records)

    result = None
    with pytest.raises(ComparisonNotReadyError):
        result = reward_comparison(
            odyssey_store=store,
            ppo_avg_reward=42.0,
            ppo_steps=10_000,
        )

    # The function must have raised before returning anything.
    assert result is None, (
        f"reward_comparison should have raised before returning a value "
        f"(got {result!r} with {n_records} eval record(s))"
    )


# ---------------------------------------------------------------------------
# Property 24 – Part 2: reward_comparison raises when PPO is not ready
# ---------------------------------------------------------------------------

def test_reward_comparison_raises_when_ppo_not_ready():
    """Property 24 (reward_comparison — PPO not evaluated):
    When the Odyssey store is fully populated but ppo_avg_reward is None,
    reward_comparison raises ComparisonNotReadyError and produces no result.

    **Validates: Requirements 14.7**
    """
    # Give Odyssey enough records (>= min_episodes=5).
    store = _store_with_n_eval_records(5)

    result = None
    with pytest.raises(ComparisonNotReadyError):
        result = reward_comparison(
            odyssey_store=store,
            ppo_avg_reward=None,   # PPO not yet evaluated
            ppo_steps=None,
        )

    assert result is None, (
        "reward_comparison should have raised before returning a value "
        f"(got {result!r} with ppo_avg_reward=None)"
    )


# ---------------------------------------------------------------------------
# Property 24 – Part 3: sample_efficiency_comparison raises when Odyssey has
#                        no eval records
# ---------------------------------------------------------------------------

def test_sample_efficiency_raises_when_odyssey_empty():
    """Property 24 (sample_efficiency_comparison — empty Odyssey store):
    When the Odyssey store has no evaluation records, sample_efficiency_comparison
    raises ComparisonNotReadyError and produces no result.

    **Validates: Requirements 14.7**
    """
    empty_store = MetricsStore()  # no eval records recorded
    ppo_results = [{"env_steps": 1000, "avg_reward": 30.0}]

    result = None
    with pytest.raises(ComparisonNotReadyError):
        result = sample_efficiency_comparison(
            odyssey_store=empty_store,
            ppo_results=ppo_results,
            step_counts=[500, 1000],
        )

    assert result is None, (
        "sample_efficiency_comparison should have raised before returning a value "
        f"(got {result!r} with empty odyssey_store)"
    )


# ---------------------------------------------------------------------------
# Property 24 – Part 4: sample_efficiency_comparison raises when ppo_results
#                        is empty
# ---------------------------------------------------------------------------

def test_sample_efficiency_raises_when_ppo_empty():
    """Property 24 (sample_efficiency_comparison — empty PPO results):
    When ppo_results is an empty list, sample_efficiency_comparison raises
    ComparisonNotReadyError and produces no result.

    **Validates: Requirements 14.7**
    """
    store = _store_with_n_eval_records(5)  # Odyssey is ready

    result = None
    with pytest.raises(ComparisonNotReadyError):
        result = sample_efficiency_comparison(
            odyssey_store=store,
            ppo_results=[],           # PPO has no results
            step_counts=[500, 1000],
        )

    assert result is None, (
        "sample_efficiency_comparison should have raised before returning a value "
        f"(got {result!r} with empty ppo_results)"
    )
