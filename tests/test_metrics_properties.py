# Feature: odyssey, Property 13: Recorded metric series round-trip through storage and persistence
"""Property-based tests for Property 13 of the Odyssey design.

**Property 13: Recorded metric series round-trip through storage and persistence**

*For any* recorded series of reconstruction-loss values (and reward-accuracy and
evaluation metrics), retrieving the values returns the same values in recorded
order; and for any recorded metrics set, persisting to file and reloading yields
an equivalent metrics set.

**Validates: Requirements 5.4, 14.5**

Implementation under test: ``training.metrics.MetricsStore``
  - ``record_reconstruction_loss(step, value)`` / ``reconstruction_losses()``
  - ``record_reward_accuracy(mae)`` / ``reward_accuracies()``
  - ``record_eval(env_steps, avg_reward, completion_rate)`` / ``eval_records()``
  - ``record_world_model_losses(step, losses_dict)`` / ``world_model_loss_history()``
  - ``persist(path)`` and ``load_metrics(path)``

Two tests are provided:

1. ``test_reconstruction_loss_in_order`` (pure Hypothesis, no file I/O):
   Records an arbitrary sequence of (step, value) pairs; asserts that
   ``reconstruction_losses()`` returns them in exactly the same order.

2. ``test_metrics_persistence_round_trip`` (plain pytest using ``tmp_path``):
   Records all four series with representative data, persists to a temp file,
   reloads, and asserts all series are equivalent (same values, same order).
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import List, Tuple

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from training.metrics import MetricsStore, load_metrics


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Finite floats only – JSON does not support NaN / Inf.
_finite_float = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

# Non-negative ints for step counters.
_step = st.integers(min_value=0, max_value=10_000)


@st.composite
def _loss_pairs(draw) -> List[Tuple[int, float]]:
    """A list of (step, value) pairs for reconstruction losses."""
    n = draw(st.integers(min_value=0, max_value=50))
    pairs = []
    for _ in range(n):
        s = draw(_step)
        v = draw(_finite_float)
        pairs.append((s, v))
    return pairs


# ---------------------------------------------------------------------------
# Property 13 – Part 1: In-order retrieval (Hypothesis)
# ---------------------------------------------------------------------------


@seed(20240701)
@settings(max_examples=20, deadline=None)
@given(pairs=_loss_pairs())
def test_reconstruction_loss_in_order(pairs: List[Tuple[int, float]]):
    """Property 13 (in-order retrieval): reconstruction_losses() returns
    the recorded (step, value) pairs in exactly the same order.

    **Validates: Requirements 5.4**
    """
    store = MetricsStore()
    for step, value in pairs:
        store.record_reconstruction_loss(step, value)

    retrieved = store.reconstruction_losses()

    assert len(retrieved) == len(pairs), (
        f"Expected {len(pairs)} entries, got {len(retrieved)}"
    )
    for i, ((exp_step, exp_val), (got_step, got_val)) in enumerate(
        zip(pairs, retrieved)
    ):
        assert got_step == exp_step, (
            f"Step mismatch at index {i}: expected {exp_step}, got {got_step}"
        )
        assert got_val == exp_val, (
            f"Value mismatch at index {i}: expected {exp_val}, got {got_val}"
        )


# ---------------------------------------------------------------------------
# Property 13 – Part 2: Persistence round-trip (plain pytest + tmp_path)
# ---------------------------------------------------------------------------

# A module-level temp dir is used so the file is cleaned up by the OS/pytest.
_TMP_DIR = tempfile.mkdtemp(prefix="odyssey_metrics_test_")


def _build_populated_store() -> MetricsStore:
    """Return a MetricsStore with representative data in all four series."""
    store = MetricsStore()

    # Reconstruction losses
    recon_data = [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.25), (10, 0.1)]
    for step, val in recon_data:
        store.record_reconstruction_loss(step, val)

    # Reward accuracies (MAE values)
    for mae in [0.08, 0.06, 0.05, 0.04]:
        store.record_reward_accuracy(mae)

    # Eval records
    eval_data = [
        (500, 12.5, 0.1),
        (1000, 25.0, 0.3),
        (2000, 45.0, 0.6),
    ]
    for env_steps, avg_reward, completion_rate in eval_data:
        store.record_eval(env_steps, avg_reward, completion_rate)

    # World-model loss snapshots
    wm_data = [
        (0, {"recon": 0.5, "kl": 0.1, "reward": 0.05, "discount": 0.02, "total": 0.67}),
        (1, {"recon": 0.4, "kl": 0.09, "reward": 0.04, "discount": 0.02, "total": 0.55}),
        (2, {"recon": 0.3, "kl": 0.08, "reward": 0.03, "discount": 0.01, "total": 0.42}),
    ]
    for step, losses in wm_data:
        store.record_world_model_losses(step, losses)

    return store


def test_metrics_persistence_round_trip(tmp_path):
    """Property 13 (persistence): persisting and reloading yields an
    equivalent MetricsStore with all series intact in the same order.

    **Validates: Requirements 14.5**
    """
    original = _build_populated_store()
    metrics_file = str(tmp_path / "metrics.json")

    # Persist to temp file
    original.persist(metrics_file)
    assert os.path.isfile(metrics_file), "persist() must create the JSON file"

    # Reload
    loaded = load_metrics(metrics_file)

    # --- Reconstruction losses ---
    orig_losses = original.reconstruction_losses()
    loaded_losses = loaded.reconstruction_losses()
    assert len(loaded_losses) == len(orig_losses), (
        f"reconstruction_losses length mismatch: {len(orig_losses)} vs {len(loaded_losses)}"
    )
    for i, ((os_, ov), (ls_, lv)) in enumerate(zip(orig_losses, loaded_losses)):
        assert ls_ == os_, f"reconstruction_losses step mismatch at index {i}"
        assert math.isclose(lv, ov, rel_tol=1e-9), (
            f"reconstruction_losses value mismatch at index {i}: {ov} vs {lv}"
        )

    # --- Reward accuracies ---
    orig_acc = original.reward_accuracies()
    loaded_acc = loaded.reward_accuracies()
    assert len(loaded_acc) == len(orig_acc), (
        f"reward_accuracies length mismatch: {len(orig_acc)} vs {len(loaded_acc)}"
    )
    for i, (ov, lv) in enumerate(zip(orig_acc, loaded_acc)):
        assert math.isclose(lv, ov, rel_tol=1e-9), (
            f"reward_accuracies mismatch at index {i}: {ov} vs {lv}"
        )

    # --- Eval records ---
    orig_eval = original.eval_records()
    loaded_eval = loaded.eval_records()
    assert len(loaded_eval) == len(orig_eval), (
        f"eval_records length mismatch: {len(orig_eval)} vs {len(loaded_eval)}"
    )
    for i, (orig_rec, loaded_rec) in enumerate(zip(orig_eval, loaded_eval)):
        assert loaded_rec["env_steps"] == orig_rec["env_steps"], (
            f"eval_records env_steps mismatch at index {i}"
        )
        assert math.isclose(loaded_rec["avg_reward"], orig_rec["avg_reward"], rel_tol=1e-9), (
            f"eval_records avg_reward mismatch at index {i}"
        )
        assert math.isclose(
            loaded_rec["completion_rate"], orig_rec["completion_rate"], rel_tol=1e-9
        ), f"eval_records completion_rate mismatch at index {i}"

    # --- World-model loss history ---
    orig_wm = original.world_model_loss_history()
    loaded_wm = loaded.world_model_loss_history()
    assert len(loaded_wm) == len(orig_wm), (
        f"world_model_loss_history length mismatch: {len(orig_wm)} vs {len(loaded_wm)}"
    )
    for i, (orig_entry, loaded_entry) in enumerate(zip(orig_wm, loaded_wm)):
        assert loaded_entry["step"] == orig_entry["step"], (
            f"world_model_loss_history step mismatch at index {i}"
        )
        for key in orig_entry:
            if key == "step":
                continue
            assert key in loaded_entry, (
                f"world_model_loss_history missing key '{key}' at index {i}"
            )
            assert math.isclose(loaded_entry[key], orig_entry[key], rel_tol=1e-9), (
                f"world_model_loss_history '{key}' mismatch at index {i}: "
                f"{orig_entry[key]} vs {loaded_entry[key]}"
            )
