"""Unit tests for the world-model combined objective (Task 9.6).

Two plain pytest tests (no Hypothesis):

1. ``test_world_model_loss_produces_all_four_finite_terms``
   Builds a minimal batch (batch_size=2, seq_len=50) and calls
   ``world_model_loss`` directly.  Asserts all four component losses are finite
   and >= 0 (where applicable) and that ``total`` is their sum within float
   tolerance.

2. ``test_per_step_loss_recording``
   Creates a ``WorldModelTrainer`` backed by a ``ReplayBuffer`` populated with
   200 transitions in a single episode.  Calls ``train_step(step=5)`` and
   asserts exactly one entry in ``MetricsStore.world_model_loss_history()`` with
   the correct step and all required keys finite.

Validates: Requirements 7.3, 7.5
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from agent.config import ScopeConfig, WorldModelConfig
from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.heads import DiscountPredictor, RewardPredictor
from agent.replay import ReplayBuffer
from agent.rssm import RSSM
from agent.types import Transition, WorldModelLosses
from training.losses import world_model_loss
from training.metrics import MetricsStore
from training.world_model_trainer import WorldModelTrainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# MVP scope: channels=1, seq_len=50, name="mvp"
SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")
BATCH_SIZE = 2
SEQ_LEN = 50  # must equal scope.sequence_length for the MVP scope


def _make_transition(terminated: bool = False) -> Transition:
    """Create a single dummy Transition with a valid (96,96,3) uint8 observation."""
    obs = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    return Transition(observation=obs, action=action, reward=1.0, terminated=terminated)


def _make_batch(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> list[list[Transition]]:
    """Build a list of ``batch_size`` sequences, each of ``seq_len`` Transitions.

    Each sequence is a standalone episode: all steps have terminated=False except
    the very last step which has terminated=True.
    """
    batch = []
    for _ in range(batch_size):
        seq = [_make_transition(terminated=False) for _ in range(seq_len - 1)]
        seq.append(_make_transition(terminated=True))
        batch.append(seq)
    return batch


def _make_components(channels: int = 1):
    """Instantiate the five world-model components for the given channel count."""
    encoder = Encoder(channels=channels)
    rssm = RSSM(action_dim=3)
    decoder = Decoder(channels=channels)
    reward_pred = RewardPredictor()
    discount_pred = DiscountPredictor()
    return encoder, rssm, decoder, reward_pred, discount_pred


# ---------------------------------------------------------------------------
# Test 1: world_model_loss produces all four finite loss terms
# ---------------------------------------------------------------------------

def test_world_model_loss_produces_all_four_finite_terms():
    """world_model_loss returns a WorldModelLosses with all four finite terms.

    Builds a minimal 2-sequence × 50-step batch with valid (96,96,3) uint8
    observations and calls world_model_loss directly.  Asserts:
    - The return type is WorldModelLosses.
    - recon is finite and >= 0.
    - kl is finite.
    - reward is finite and >= 0.
    - discount is finite and >= 0.
    - total == recon + kl + reward + discount (within float tolerance).

    Validates: Requirements 7.3, 7.5
    """
    torch.manual_seed(42)
    np.random.seed(42)

    encoder, rssm, decoder, reward_pred, discount_pred = _make_components(channels=1)
    batch = _make_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    result = world_model_loss(
        batch=batch,
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_predictor=reward_pred,
        discount_predictor=discount_pred,
        scope=SCOPE,
    )

    # Return type check
    assert isinstance(result, WorldModelLosses), (
        f"Expected WorldModelLosses, got {type(result)}"
    )

    # Finiteness and sign checks
    assert math.isfinite(result.recon), f"recon is not finite: {result.recon}"
    assert result.recon >= 0.0, f"recon should be >= 0 (MSE), got {result.recon}"

    assert math.isfinite(result.kl), f"kl is not finite: {result.kl}"

    assert math.isfinite(result.reward), f"reward is not finite: {result.reward}"
    assert result.reward >= 0.0, f"reward should be >= 0 (MSE), got {result.reward}"

    assert math.isfinite(result.discount), f"discount is not finite: {result.discount}"
    assert result.discount >= 0.0, f"discount should be >= 0 (BCE), got {result.discount}"

    # total == sum of four terms (within float tolerance)
    expected_total = result.recon + result.kl + result.reward + result.discount
    assert math.isclose(result.total, expected_total, rel_tol=1e-5, abs_tol=1e-7), (
        f"total ({result.total}) != recon+kl+reward+discount ({expected_total})"
    )


# ---------------------------------------------------------------------------
# Test 2: per-step loss recording via WorldModelTrainer
# ---------------------------------------------------------------------------

def test_per_step_loss_recording():
    """WorldModelTrainer.train_step records per-step losses in MetricsStore.

    Populates a ReplayBuffer(capacity=500) with 200 transitions in a single
    episode (terminated only on the last).  Creates a WorldModelTrainer with
    batch_size=2, sequence_length=50, and calls train_step(step=5).  Asserts:
    - MetricsStore.world_model_loss_history() contains exactly one entry.
    - The entry's "step" key equals 5.
    - The entry contains keys "recon", "kl", "reward", "discount", "total".
    - All five values are finite.

    Validates: Requirements 7.3, 7.5
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Build components
    encoder, rssm, decoder, reward_pred, discount_pred = _make_components(channels=1)
    metrics = MetricsStore()

    # Use a temp directory so no checkpoint file is left behind
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "wm.pt")
        wm_config = WorldModelConfig(
            batch_size=BATCH_SIZE,
            checkpoint_interval=10_000,  # large enough to never trigger in this test
            checkpoint_path=ckpt_path,
        )

        # Populate replay buffer: 200 transitions, one episode
        replay = ReplayBuffer(capacity=500)
        for i in range(200):
            terminated = i == 199  # only the last transition ends the episode
            replay.add(_make_transition(terminated=terminated))

        trainer = WorldModelTrainer(
            encoder=encoder,
            rssm=rssm,
            decoder=decoder,
            reward_predictor=reward_pred,
            discount_predictor=discount_pred,
            replay_buffer=replay,
            metrics_store=metrics,
            config=wm_config,
            scope=SCOPE,
        )

        result = trainer.train_step(step=5)

    # train_step should succeed (not return None) given enough data
    assert result is not None, (
        "train_step returned None; the buffer may not have enough data. "
        f"Buffer length: {len(replay)}, longest run: {replay.longest_contiguous_run()}"
    )

    history = metrics.world_model_loss_history()

    # Exactly one entry recorded
    assert len(history) == 1, (
        f"Expected exactly 1 entry in world_model_loss_history(), got {len(history)}"
    )

    entry = history[0]

    # Step key matches
    assert entry["step"] == 5, f"Expected step=5, got step={entry['step']}"

    # Required loss-term keys are present
    required_keys = {"recon", "kl", "reward", "discount", "total"}
    missing = required_keys - set(entry.keys())
    assert not missing, f"Missing keys in recorded entry: {missing}"

    # All values are finite
    for key in required_keys:
        val = entry[key]
        assert math.isfinite(val), (
            f"Loss term '{key}' is not finite in the recorded entry: {val}"
        )
