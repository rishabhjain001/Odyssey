# Feature: odyssey, Property 23: Sample-efficiency comparison reports both agents at each requested step count
"""Property-based tests for Property 23 of the Odyssey design.

**Property 23: Sample-efficiency comparison reports both agents at each requested step count**

*For any* recorded Odyssey and PPO reward-vs-steps series and any configured set of
cumulative Environment_Step counts, when both agents have completed training the
comparison reports, at each requested step count, the average episode reward achieved
by Odyssey and by the PPO_Baseline at that count.

**Validates: Requirements 14.4**

Implementation under test: ``training.metrics.sample_efficiency_comparison``
"""

from __future__ import annotations

from typing import List

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.errors import ComparisonNotReadyError
from training.metrics import MetricsStore, sample_efficiency_comparison


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_finite_float = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

_env_steps = st.integers(min_value=0, max_value=100_000)


@st.composite
def _eval_records(draw, min_size: int = 1) -> List[dict]:
    """Generate a non-empty list of eval records with 'env_steps' and 'avg_reward'."""
    n = draw(st.integers(min_value=min_size, max_value=20))
    records = []
    for _ in range(n):
        records.append(
            {
                "env_steps": draw(_env_steps),
                "avg_reward": draw(_finite_float),
            }
        )
    return records


@st.composite
def _step_counts(draw) -> List[int]:
    """Generate a list of step counts (possibly empty, possibly with duplicates)."""
    n = draw(st.integers(min_value=0, max_value=10))
    return [draw(_env_steps) for _ in range(n)]


def _build_odyssey_store(records: List[dict]) -> MetricsStore:
    """Populate a MetricsStore with the given eval records."""
    store = MetricsStore()
    for rec in records:
        store.record_eval(
            env_steps=rec["env_steps"],
            avg_reward=rec["avg_reward"],
            completion_rate=0.0,  # not used by sample_efficiency_comparison
        )
    return store


def _latest_reward_up_to(records: List[dict], step: int):
    """Reference implementation: last record with env_steps <= step, or None."""
    best = None
    for rec in records:
        if rec["env_steps"] <= step:
            best = float(rec["avg_reward"])
    return best


# ---------------------------------------------------------------------------
# Property 23 – Sample-efficiency comparison reports both agents at each
# requested step count
# ---------------------------------------------------------------------------


@seed(20240702)
@settings(max_examples=20, deadline=None)
@given(
    odyssey_records=_eval_records(min_size=1),
    ppo_records=_eval_records(min_size=1),
    step_counts=_step_counts(),
)
def test_sample_efficiency_comparison_structure_and_values(
    odyssey_records: List[dict],
    ppo_records: List[dict],
    step_counts: List[int],
):
    """Property 23: sample_efficiency_comparison returns exactly len(step_counts)
    entries, each with the correct keys and values matching the latest record
    at or below the requested step count.

    **Validates: Requirements 14.4**
    """
    store = _build_odyssey_store(odyssey_records)
    result = sample_efficiency_comparison(store, ppo_records, step_counts)

    # Assert: returns exactly len(step_counts) entries
    assert len(result) == len(step_counts), (
        f"Expected {len(step_counts)} result entries, got {len(result)}"
    )

    # Assert: each entry has the required keys
    required_keys = {"env_steps", "odyssey_avg_reward", "ppo_avg_reward"}
    for i, entry in enumerate(result):
        assert set(entry.keys()) == required_keys, (
            f"Entry {i} has keys {set(entry.keys())}, expected {required_keys}"
        )

    # Assert: step values and reward values are correct
    for i, (entry, step) in enumerate(zip(result, step_counts)):
        assert entry["env_steps"] == step, (
            f"Entry {i}: env_steps={entry['env_steps']}, expected {step}"
        )

        expected_odyssey = _latest_reward_up_to(odyssey_records, step)
        expected_ppo = _latest_reward_up_to(ppo_records, step)

        assert entry["odyssey_avg_reward"] == expected_odyssey, (
            f"Entry {i} at step={step}: odyssey_avg_reward={entry['odyssey_avg_reward']!r}, "
            f"expected={expected_odyssey!r}"
        )
        assert entry["ppo_avg_reward"] == expected_ppo, (
            f"Entry {i} at step={step}: ppo_avg_reward={entry['ppo_avg_reward']!r}, "
            f"expected={expected_ppo!r}"
        )
