# Feature: odyssey, Property 16: World-model parameter updates are all-or-none
"""Property-based tests for :class:`training.world_model_trainer.WorldModelTrainer`.

This module implements design Correctness Property 16:

    Property 16: World-model parameter updates are all-or-none

    For any world-model training step where the computed loss is non-finite, or
    the buffer holds fewer transitions than the configured batch requires, the
    step is skipped, an error is recorded, and the parameters of all five
    world-model components (Encoder, RSSM, Decoder, Reward_Predictor,
    Discount_Predictor) remain unchanged; otherwise all five are updated together
    within the single step.

Three regimes are exercised:

    * Skip on insufficient data (Req 7.8) -- empty buffer → train_step returns
      None and all five component parameter tensors are unchanged.
    * Skip on non-finite loss (Req 7.9) -- ``_compute_losses`` is monkeypatched to
      return a WorldModelLosses with total=NaN → train_step returns None and all
      five component parameter tensors are unchanged.
    * Success updates all five (Req 7.4) -- buffer filled with enough valid
      transitions → train_step returns WorldModelLosses and at least one
      parameter in each of the five components has changed.

Hypothesis is used to vary batch_size (2..4) and the random seed controlling
component initialisation across all three regimes, verifying the invariant holds
for a range of configurations.

**Validates: Requirements 7.4, 7.8, 7.9**
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.config import ScopeConfig, WorldModelConfig
from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.errors import InsufficientDataError
from agent.heads import DiscountPredictor, RewardPredictor
from agent.replay import ReplayBuffer
from agent.rssm import RSSM
from agent.types import Transition, WorldModelLosses
from training.metrics import MetricsStore
from training.world_model_trainer import WorldModelTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sequence length for the MVP scope.
_SEQ_LEN = 50

# Observation shape as stored in the replay buffer (raw 96×96 RGB uint8).
_OBS_SHAPE = (96, 96, 3)

# Action dimensionality for CarRacing.
_ACTION_DIM = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(batch_size: int, seed_val: int = 0) -> WorldModelTrainer:
    """Construct a WorldModelTrainer with freshly initialised components.

    Uses a small, deterministic seed to make hypothesis examples reproducible.
    The checkpoint path is set to a non-existent location so no checkpoint I/O
    happens during tests.
    """
    torch.manual_seed(seed_val)
    encoder = Encoder(channels=1)
    rssm = RSSM(action_dim=_ACTION_DIM)
    decoder = Decoder(channels=1)
    reward_predictor = RewardPredictor()
    discount_predictor = DiscountPredictor()
    replay = ReplayBuffer(capacity=10_000)
    metrics = MetricsStore()
    config = WorldModelConfig(
        batch_size=batch_size,
        checkpoint_interval=100_000,  # effectively disabled for tests
        checkpoint_path="checkpoints/test_world_model.pt",
    )
    scope = ScopeConfig(channels=1, sequence_length=_SEQ_LEN, name="mvp")
    return WorldModelTrainer(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_predictor=reward_predictor,
        discount_predictor=discount_predictor,
        replay_buffer=replay,
        metrics_store=metrics,
        config=config,
        scope=scope,
    )


def _snapshot_params(trainer: WorldModelTrainer) -> Dict[str, Dict[str, torch.Tensor]]:
    """Deep-copy the parameter tensors of all five components.

    Returns a dict mapping component name → state_dict of cloned tensors.
    """
    return {
        "encoder": {k: v.clone() for k, v in trainer.encoder.state_dict().items()},
        "rssm": {k: v.clone() for k, v in trainer.rssm.state_dict().items()},
        "decoder": {k: v.clone() for k, v in trainer.decoder.state_dict().items()},
        "reward_predictor": {
            k: v.clone() for k, v in trainer.reward_predictor.state_dict().items()
        },
        "discount_predictor": {
            k: v.clone() for k, v in trainer.discount_predictor.state_dict().items()
        },
    }


def _assert_params_unchanged(
    trainer: WorldModelTrainer,
    snapshot: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Assert every parameter tensor in all five components equals the snapshot."""
    for component_name, old_state in snapshot.items():
        component = getattr(trainer, component_name)
        new_state = component.state_dict()
        for param_name, old_tensor in old_state.items():
            new_tensor = new_state[param_name]
            assert torch.equal(old_tensor, new_tensor), (
                f"Parameter '{component_name}.{param_name}' changed unexpectedly.\n"
                f"  max abs diff = {(old_tensor - new_tensor).abs().max().item()}"
            )


def _assert_at_least_one_param_changed_per_component(
    trainer: WorldModelTrainer,
    snapshot: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Assert at least one parameter in each component differs from the snapshot."""
    for component_name, old_state in snapshot.items():
        component = getattr(trainer, component_name)
        new_state = component.state_dict()
        changed = False
        for param_name, old_tensor in old_state.items():
            new_tensor = new_state[param_name]
            if not torch.equal(old_tensor, new_tensor):
                changed = True
                break
        assert changed, (
            f"No parameter in component '{component_name}' changed after a "
            "successful train_step, but all five should have been updated (Req 7.4)."
        )


def _fill_buffer_one_episode(replay: ReplayBuffer, n_transitions: int) -> None:
    """Add ``n_transitions`` transitions as a single episode to ``replay``.

    The last transition has ``terminated=True``; all others have
    ``terminated=False``. This guarantees that ``longest_contiguous_run()`` >=
    n_transitions, so ``sample_sequences(batch_size, seq_len=50)`` can succeed.

    Observations are random uint8 (96, 96, 3); actions are random float32 (3,).
    """
    rng = np.random.default_rng(seed=42)
    for i in range(n_transitions):
        terminated = i == n_transitions - 1
        obs = rng.integers(0, 256, _OBS_SHAPE, dtype=np.uint8)
        action = rng.random(_ACTION_DIM).astype(np.float32)
        replay.add(
            Transition(
                observation=obs,
                action=action,
                reward=float(rng.uniform(-1.0, 1.0)),
                terminated=terminated,
            )
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# batch_size in [2, 4] to keep the success test tractable while still varying
# the config. Smaller batch = fewer FLOPs per hypothesis example.
_batch_size_st = st.integers(min_value=2, max_value=4)

# Seed for torch.manual_seed / numpy component initialisation.
_seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property tests: skip on insufficient data (Req 7.8)
# ---------------------------------------------------------------------------


@seed(20240701)
@settings(max_examples=15, deadline=None)
@given(batch_size=_batch_size_st, torch_seed=_seed_st)
def test_property16_skip_on_empty_buffer_all_params_unchanged(
    batch_size: int, torch_seed: int
) -> None:
    """Req 7.8: empty buffer → train_step skips and all five components unchanged.

    For any batch_size in [2..4] and any random component initialisation:
      - Call train_step(0) on a trainer with an empty replay buffer.
      - Assert return value is None (skip).
      - Assert every parameter tensor in all five components equals its pre-call snapshot.

    **Validates: Requirements 7.4, 7.8**
    """
    trainer = _make_trainer(batch_size=batch_size, seed_val=torch_seed)
    # Buffer is empty -- no transitions added.
    snapshot = _snapshot_params(trainer)

    result = trainer.train_step(step=0)

    assert result is None, (
        f"Expected train_step to return None on empty buffer, got {result!r}"
    )
    _assert_params_unchanged(trainer, snapshot)


@seed(20240702)
@settings(max_examples=15, deadline=None)
@given(batch_size=_batch_size_st, torch_seed=_seed_st)
def test_property16_skip_on_insufficient_data_all_params_unchanged(
    batch_size: int, torch_seed: int
) -> None:
    """Req 7.8: buffer with fewer transitions than required → skip, all params unchanged.

    Adds (batch_size - 1) transitions (always fewer than batch_size), so
    InsufficientDataError is guaranteed regardless of seq_len.

    **Validates: Requirements 7.4, 7.8**
    """
    trainer = _make_trainer(batch_size=batch_size, seed_val=torch_seed)
    # Add fewer transitions than the batch_size so the sample must fail.
    rng = np.random.default_rng(42)
    for i in range(max(0, batch_size - 1)):
        trainer.replay_buffer.add(
            Transition(
                observation=rng.integers(0, 256, _OBS_SHAPE, dtype=np.uint8),
                action=rng.random(_ACTION_DIM).astype(np.float32),
                reward=0.0,
                terminated=(i == batch_size - 2),
            )
        )

    snapshot = _snapshot_params(trainer)
    result = trainer.train_step(step=0)

    assert result is None, (
        f"Expected None (skip) with insufficient data, got {result!r}"
    )
    _assert_params_unchanged(trainer, snapshot)


# ---------------------------------------------------------------------------
# Property tests: skip on non-finite loss (Req 7.9)
# ---------------------------------------------------------------------------


def _nan_losses(_batch):
    """Return a WorldModelLosses with total=NaN and a NaN tensor, mimicking
    a non-finite result from _compute_losses."""
    losses_float = WorldModelLosses(
        recon=float("nan"),
        kl=float("nan"),
        reward=float("nan"),
        discount=float("nan"),
        total=float("nan"),
    )
    total_tensor = torch.tensor(float("nan"), requires_grad=False)
    return losses_float, total_tensor


@seed(20240703)
@settings(max_examples=15, deadline=None)
@given(batch_size=_batch_size_st, torch_seed=_seed_st)
def test_property16_skip_on_nan_loss_all_params_unchanged(
    batch_size: int, torch_seed: int
) -> None:
    """Req 7.9: non-finite (NaN) total loss → skip and all five components unchanged.

    Monkeypatches _compute_losses to return a NaN WorldModelLosses so the
    non-finite guard triggers without needing a real forward pass. The buffer
    is filled just enough that sampling does not fail first.

    **Validates: Requirements 7.4, 7.9**
    """
    trainer = _make_trainer(batch_size=batch_size, seed_val=torch_seed)
    # Fill the buffer sufficiently so the InsufficientDataError path is not hit.
    _fill_buffer_one_episode(trainer.replay_buffer, batch_size * _SEQ_LEN + _SEQ_LEN)

    snapshot = _snapshot_params(trainer)

    with patch.object(trainer, "_compute_losses", side_effect=_nan_losses):
        result = trainer.train_step(step=0)

    assert result is None, (
        f"Expected None (skip) on NaN loss, got {result!r}"
    )
    _assert_params_unchanged(trainer, snapshot)


@seed(20240704)
@settings(max_examples=15, deadline=None)
@given(batch_size=_batch_size_st, torch_seed=_seed_st)
def test_property16_skip_on_inf_loss_all_params_unchanged(
    batch_size: int, torch_seed: int
) -> None:
    """Req 7.9: non-finite (Inf) total loss → skip and all five components unchanged.

    Same structure as the NaN variant but uses positive infinity instead.

    **Validates: Requirements 7.4, 7.9**
    """
    trainer = _make_trainer(batch_size=batch_size, seed_val=torch_seed)
    _fill_buffer_one_episode(trainer.replay_buffer, batch_size * _SEQ_LEN + _SEQ_LEN)

    def _inf_losses(_batch):
        losses_float = WorldModelLosses(
            recon=float("inf"),
            kl=float("inf"),
            reward=float("inf"),
            discount=float("inf"),
            total=float("inf"),
        )
        total_tensor = torch.tensor(float("inf"), requires_grad=False)
        return losses_float, total_tensor

    snapshot = _snapshot_params(trainer)

    with patch.object(trainer, "_compute_losses", side_effect=_inf_losses):
        result = trainer.train_step(step=0)

    assert result is None, (
        f"Expected None (skip) on Inf loss, got {result!r}"
    )
    _assert_params_unchanged(trainer, snapshot)


# ---------------------------------------------------------------------------
# Property tests: success updates all five components (Req 7.4)
# ---------------------------------------------------------------------------


@seed(20240705)
@settings(max_examples=5, deadline=None)
@given(batch_size=_batch_size_st, torch_seed=_seed_st)
def test_property16_success_updates_all_five_components(
    batch_size: int, torch_seed: int
) -> None:
    """Req 7.4: successful train_step updates all five world-model components.

    Fills the buffer with enough transitions for ``sample_sequences(batch_size,
    seq_len=50)`` to succeed (one episode of batch_size * 50 + 50 transitions).
    Snapshots all five component state dicts before the step, calls train_step,
    and asserts:
      - Return value is a WorldModelLosses (not None).
      - At least one parameter in each of the five components has changed.

    max_examples is kept small (5) because each example runs a full
    forward+backward pass through all five components, which takes several
    seconds.

    **Validates: Requirements 7.4**
    """
    trainer = _make_trainer(batch_size=batch_size, seed_val=torch_seed)
    # Need at least batch_size non-overlapping windows of length seq_len=50.
    # One long episode of (batch_size * seq_len + seq_len) transitions guarantees
    # enough valid windows.
    n_transitions = batch_size * _SEQ_LEN + _SEQ_LEN
    _fill_buffer_one_episode(trainer.replay_buffer, n_transitions)

    snapshot = _snapshot_params(trainer)

    result = trainer.train_step(step=0)

    assert result is not None, (
        "Expected train_step to return WorldModelLosses on success, got None"
    )
    assert isinstance(result, WorldModelLosses), (
        f"Expected WorldModelLosses, got {type(result).__name__}"
    )
    assert math.isfinite(result.total), (
        f"Expected finite total loss on success, got {result.total!r}"
    )

    _assert_at_least_one_param_changed_per_component(trainer, snapshot)


# ---------------------------------------------------------------------------
# Plain pytest examples for quick, deterministic regression coverage
# ---------------------------------------------------------------------------


def test_example_empty_buffer_returns_none_and_params_unchanged() -> None:
    """Example: empty buffer → None, all five params unchanged. (Req 7.8)"""
    trainer = _make_trainer(batch_size=2, seed_val=0)
    snapshot = _snapshot_params(trainer)
    result = trainer.train_step(step=0)
    assert result is None
    _assert_params_unchanged(trainer, snapshot)


def test_example_nan_loss_returns_none_and_params_unchanged() -> None:
    """Example: NaN loss → None, all five params unchanged. (Req 7.9)"""
    trainer = _make_trainer(batch_size=2, seed_val=1)
    _fill_buffer_one_episode(trainer.replay_buffer, 2 * _SEQ_LEN + _SEQ_LEN)
    snapshot = _snapshot_params(trainer)
    with patch.object(trainer, "_compute_losses", side_effect=_nan_losses):
        result = trainer.train_step(step=0)
    assert result is None
    _assert_params_unchanged(trainer, snapshot)


def test_example_success_updates_all_five_and_returns_losses() -> None:
    """Example: sufficient data → WorldModelLosses returned and all five updated. (Req 7.4)

    Uses batch_size=2 and one episode of 150 transitions to keep the forward
    pass fast while still exercising the real update path.
    """
    trainer = _make_trainer(batch_size=2, seed_val=2)
    _fill_buffer_one_episode(trainer.replay_buffer, 2 * _SEQ_LEN + _SEQ_LEN)
    snapshot = _snapshot_params(trainer)
    result = trainer.train_step(step=0)
    assert result is not None
    assert isinstance(result, WorldModelLosses)
    assert math.isfinite(result.total)
    _assert_at_least_one_param_changed_per_component(trainer, snapshot)


def test_example_metrics_store_records_skip_on_empty_buffer() -> None:
    """Example: MetricsStore records a skip event when the buffer is empty. (Req 7.8)"""
    trainer = _make_trainer(batch_size=2, seed_val=3)
    result = trainer.train_step(step=5)
    assert result is None
    history = trainer.metrics_store.world_model_loss_history()
    assert len(history) == 1, f"Expected 1 entry in loss history, got {len(history)}"
    entry = history[0]
    assert entry["step"] == 5
    # The skip event is encoded with skipped=1.0
    assert entry.get("skipped") == 1.0


def test_example_metrics_store_records_skip_on_nan_loss() -> None:
    """Example: MetricsStore records a skip event when loss is NaN. (Req 7.9)"""
    trainer = _make_trainer(batch_size=2, seed_val=4)
    _fill_buffer_one_episode(trainer.replay_buffer, 2 * _SEQ_LEN + _SEQ_LEN)
    with patch.object(trainer, "_compute_losses", side_effect=_nan_losses):
        result = trainer.train_step(step=7)
    assert result is None
    history = trainer.metrics_store.world_model_loss_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["step"] == 7
    assert entry.get("skipped") == 1.0
