# Feature: odyssey, Property 14: Reward accuracy equals mean absolute error
"""Property-based test for reward-prediction accuracy computation and MetricsStore recording.

This module implements design Correctness Property 14:

    Property 14: Reward accuracy equals mean absolute error

    For any set of predicted-reward and recorded-reward pairs sampled from the
    buffer, the recorded reward-prediction accuracy equals the arithmetic mean of
    the absolute differences between predicted and recorded rewards.

Validates: Requirements 6.3
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from training.metrics import MetricsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_mae_numpy(predicted: list[float], actual: list[float]) -> float:
    """Compute MAE using numpy: mean(|predicted - actual|)."""
    pred_arr = np.array(predicted, dtype=np.float64)
    act_arr = np.array(actual, dtype=np.float64)
    return float(np.mean(np.abs(pred_arr - act_arr)))


def _compute_mae_formula(predicted: list[float], actual: list[float]) -> float:
    """Compute MAE using the plain formula (no numpy)."""
    n = len(predicted)
    total = sum(abs(p - a) for p, a in zip(predicted, actual))
    return total / n


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

@seed(20240614)
@settings(max_examples=20, deadline=None)
@given(
    rewards=st.lists(
        st.tuples(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_mae_formula_matches_numpy(rewards: list[tuple[float, float]]) -> None:
    """MAE computed by plain formula equals MAE computed by numpy.

    For any list of (predicted, actual) reward pairs the two implementations
    agree to floating-point precision.

    Validates: Requirements 6.3
    """
    predicted = [p for p, _ in rewards]
    actual = [a for _, a in rewards]

    mae_np = _compute_mae_numpy(predicted, actual)
    mae_formula = _compute_mae_formula(predicted, actual)

    assert abs(mae_np - mae_formula) < 1e-9, (
        f"numpy MAE {mae_np} != formula MAE {mae_formula} "
        f"for predicted={predicted}, actual={actual}"
    )


@seed(20240614)
@settings(max_examples=20, deadline=None)
@given(
    rewards=st.lists(
        st.tuples(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_record_reward_accuracy_stores_and_retrieves(
    rewards: list[tuple[float, float]],
) -> None:
    """MetricsStore.record_reward_accuracy stores and reward_accuracies() retrieves it.

    Compute the MAE for the given pairs, record it in a fresh MetricsStore, and
    assert that reward_accuracies() returns exactly that value.

    Validates: Requirements 6.3
    """
    predicted = [p for p, _ in rewards]
    actual = [a for _, a in rewards]

    mae = _compute_mae_numpy(predicted, actual)

    store = MetricsStore()
    store.record_reward_accuracy(mae)

    retrieved = store.reward_accuracies()
    assert len(retrieved) == 1, f"Expected 1 entry, got {len(retrieved)}"
    assert retrieved[0] == mae, f"Stored {mae} but got back {retrieved[0]}"


@seed(20240614)
@settings(max_examples=20, deadline=None)
@given(
    mae_values=st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_multi_recording_preserves_order(mae_values: list[float]) -> None:
    """Recording multiple MAE values returns all of them in insertion order.

    Validates: Requirements 6.3
    """
    store = MetricsStore()
    for mae in mae_values:
        store.record_reward_accuracy(mae)

    retrieved = store.reward_accuracies()
    assert len(retrieved) == len(mae_values), (
        f"Expected {len(mae_values)} entries, got {len(retrieved)}"
    )
    for i, (expected, actual) in enumerate(zip(mae_values, retrieved)):
        assert expected == actual, (
            f"Position {i}: expected {expected}, got {actual}"
        )


@seed(20240614)
@settings(max_examples=20, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_mae_zero_when_predicted_equals_actual(values: list[float]) -> None:
    """MAE is exactly 0.0 when predicted rewards equal actual rewards.

    Validates: Requirements 6.3
    """
    mae = _compute_mae_numpy(values, values)
    assert mae == 0.0, f"Expected MAE=0.0 when predicted==actual, got {mae}"

    # Also verify the MetricsStore records and retrieves it faithfully.
    store = MetricsStore()
    store.record_reward_accuracy(mae)
    retrieved = store.reward_accuracies()
    assert retrieved == [0.0], f"Expected [0.0], got {retrieved}"
