# Feature: odyssey, Property 17: World-model checkpoints round-trip
"""Property-based test for Property 17 of the Odyssey design.

**Property 17: World-model checkpoints round-trip**

*For any set of world-model parameters*, persisting them to a checkpoint and
loading that checkpoint restores parameters equal to the originals.

**Validates: Requirements 7.6, 7.7**

Implementation under test: ``training/checkpoint.py`` →
``save_checkpoint(...)`` and ``maybe_load_checkpoint(...)``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.encoder import Encoder
from agent.rssm import RSSM
from agent.decoder import Decoder
from agent.heads import RewardPredictor, DiscountPredictor
from training.checkpoint import save_checkpoint, maybe_load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_components():
    """Instantiate one fresh copy of each world-model component."""
    return (
        Encoder(channels=1),
        RSSM(action_dim=3),
        Decoder(channels=1),
        RewardPredictor(),
        DiscountPredictor(),
    )


def _state_dicts_equal(sd1: dict, sd2: dict) -> bool:
    """Return True iff two state_dicts are parameter-for-parameter equal."""
    if sd1.keys() != sd2.keys():
        return False
    for key in sd1:
        if not torch.equal(sd1[key], sd2[key]):
            return False
    return True


def _randomize_components(components, seed_val: int):
    """Fill all parameters of each component with random values from seed_val."""
    rng = torch.Generator()
    rng.manual_seed(seed_val)
    for module in components:
        with torch.no_grad():
            for param in module.parameters():
                param.copy_(torch.empty_like(param).normal_(generator=rng))


# ---------------------------------------------------------------------------
# Unit tests (no hypothesis)
# ---------------------------------------------------------------------------

def test_maybe_load_checkpoint_returns_zero_when_file_missing(tmp_path):
    """maybe_load_checkpoint returns 0 when the checkpoint file does not exist."""
    path = tmp_path / "nonexistent.pt"
    enc, rssm, dec, rew, dis = _make_components()
    result = maybe_load_checkpoint(path, enc, rssm, dec, rew, dis)
    assert result == 0, f"Expected 0 for missing file, got {result}"


def test_step_round_trips(tmp_path):
    """The saved step integer is restored exactly after save/load."""
    path = tmp_path / "ckpt.pt"
    enc, rssm, dec, rew, dis = _make_components()
    save_checkpoint(path, enc, rssm, dec, rew, dis, step=42)

    enc2, rssm2, dec2, rew2, dis2 = _make_components()
    loaded_step = maybe_load_checkpoint(path, enc2, rssm2, dec2, rew2, dis2)
    assert loaded_step == 42, f"Expected step=42, got {loaded_step}"


def test_parameters_round_trip_after_load(tmp_path):
    """After loading, each component's parameters equal those that were saved."""
    path = tmp_path / "ckpt.pt"
    enc, rssm, dec, rew, dis = _make_components()
    _randomize_components([enc, rssm, dec, rew, dis], seed_val=999)

    # Capture original state_dicts before save
    orig_enc = enc.state_dict()
    orig_rssm = rssm.state_dict()
    orig_dec = dec.state_dict()
    orig_rew = rew.state_dict()
    orig_dis = dis.state_dict()

    save_checkpoint(path, enc, rssm, dec, rew, dis, step=1)

    # Load into freshly initialised components
    enc2, rssm2, dec2, rew2, dis2 = _make_components()
    maybe_load_checkpoint(path, enc2, rssm2, dec2, rew2, dis2)

    assert _state_dicts_equal(orig_enc,  enc2.state_dict()),  "Encoder parameters differ after round-trip"
    assert _state_dicts_equal(orig_rssm, rssm2.state_dict()), "RSSM parameters differ after round-trip"
    assert _state_dicts_equal(orig_dec,  dec2.state_dict()),  "Decoder parameters differ after round-trip"
    assert _state_dicts_equal(orig_rew,  rew2.state_dict()),  "RewardPredictor parameters differ after round-trip"
    assert _state_dicts_equal(orig_dis,  dis2.state_dict()),  "DiscountPredictor parameters differ after round-trip"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@st.composite
def _checkpoint_inputs(draw):
    """Generate a (torch_seed, step) pair for property testing."""
    torch_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    step = draw(st.integers(min_value=0, max_value=10_000_000))
    return torch_seed, step


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(inputs=_checkpoint_inputs())
def test_checkpoint_round_trip_property(inputs):
    """Property 17: For any world-model parameters, saving then loading restores
    parameters equal to the originals, and the step integer is preserved exactly.

    **Validates: Requirements 7.6, 7.7**
    """
    torch_seed, step = inputs

    # Build and randomise components
    enc, rssm, dec, rew, dis = _make_components()
    _randomize_components([enc, rssm, dec, rew, dis], seed_val=torch_seed)

    # Capture original state_dicts
    orig_enc  = {k: v.clone() for k, v in enc.state_dict().items()}
    orig_rssm = {k: v.clone() for k, v in rssm.state_dict().items()}
    orig_dec  = {k: v.clone() for k, v in dec.state_dict().items()}
    orig_rew  = {k: v.clone() for k, v in rew.state_dict().items()}
    orig_dis  = {k: v.clone() for k, v in dis.state_dict().items()}

    # Save checkpoint to a self-managed temp directory (not the fixture, so
    # hypothesis can reset state correctly between examples).
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "ckpt.pt"
    try:
        save_checkpoint(path, enc, rssm, dec, rew, dis, step=step)

        # Load into fresh components
        enc2, rssm2, dec2, rew2, dis2 = _make_components()
        loaded_step = maybe_load_checkpoint(path, enc2, rssm2, dec2, rew2, dis2)
    finally:
        # Clean up the temp file and directory
        if path.exists():
            os.remove(path)
        os.rmdir(tmp_dir)

    # Step must round-trip exactly (Req 7.6, 7.7)
    assert loaded_step == step, (
        f"Step integer did not round-trip: saved {step}, loaded {loaded_step}"
    )

    # Every component's parameters must be identical to the originals (Req 7.6, 7.7)
    assert _state_dicts_equal(orig_enc,  enc2.state_dict()),  "Encoder parameters differ after checkpoint round-trip"
    assert _state_dicts_equal(orig_rssm, rssm2.state_dict()), "RSSM parameters differ after checkpoint round-trip"
    assert _state_dicts_equal(orig_dec,  dec2.state_dict()),  "Decoder parameters differ after checkpoint round-trip"
    assert _state_dicts_equal(orig_rew,  rew2.state_dict()),  "RewardPredictor parameters differ after checkpoint round-trip"
    assert _state_dicts_equal(orig_dis,  dis2.state_dict()),  "DiscountPredictor parameters differ after checkpoint round-trip"
