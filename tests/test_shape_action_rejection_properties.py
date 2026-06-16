# Feature: odyssey, Property 10: Shape and action-dimension mismatches are rejected without state change
"""Property-based test for shape and action-dimension rejection (Property 10).

**Property 10: Shape and action-dimension mismatches are rejected without state change**

For any frame whose resolution or channel count does not match the active scope,
the Encoder raises a shape error and produces no latent; and for any action
whose dimensionality does not match the CarRacing action space, the RSSM rejects
the prediction request, returns an invalid-action error, and leaves its
deterministic and stochastic state unchanged.

This test exercises two components against the MVP grayscale scope
(``channels=1``) and the CarRacing action space (``action_dim=3``):

    * :class:`agent.encoder.Encoder` -- ``forward`` must raise
      :class:`agent.errors.FrameShapeError` for frames with a wrong channel
      count, wrong spatial resolution, or wrong rank, and produce no latent
      (the call raises; nothing is returned).
    * :class:`agent.rssm.RSSM` -- ``predict_prior`` must raise
      :class:`agent.errors.InvalidActionError` for actions whose last dimension
      is not ``action_dim``, leaving the passed-in state's ``deter`` and
      ``stoch`` tensors byte-for-byte unchanged (the RSSM is functional and must
      never mutate its input).

Validates: Requirements 3.5, 4.6
"""

from __future__ import annotations

import torch
import pytest
from hypothesis import given, seed, settings, strategies as st

from agent.encoder import Encoder
from agent.errors import FrameShapeError, InvalidActionError
from agent.rssm import RSSM

# The active scope under test: MVP grayscale has a single channel (Req 3.2, 3.3).
SCOPE_CHANNELS = 1
# Fixed input resolution for every scope (Req 3.2, 3.3).
FRAME_RESOLUTION = 64
# CarRacing action space: [steering, accel, brake] (Req 4.6).
ACTION_DIM = 3


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Channel counts that do NOT match the active scope (anything != 1).
_wrong_channels = st.sampled_from([2, 3, 4])

# Spatial resolutions that do NOT match 64x64.
_wrong_resolutions = st.sampled_from([32, 48, 96, 16, 128])

# Action last-dims that do NOT match the action space (anything != 3).
_wrong_action_dims = st.sampled_from([1, 2, 4, 5, 6])

# Batch sizes to vary across (1..4).
_batch_sizes = st.integers(min_value=1, max_value=4)


# ---------------------------------------------------------------------------
# Property 10 -- Encoder: shape/channel mismatches are rejected, no latent
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(
    batch=_batch_sizes,
    channels=_wrong_channels,
    data=st.data(),
)
def test_encoder_rejects_wrong_channel_count(
    batch: int, channels: int, data: st.DataObject
) -> None:
    """A frame whose channel count != active scope is rejected; no latent (Req 3.5).

    Validates: Requirements 3.5
    """
    encoder = Encoder(channels=SCOPE_CHANNELS)
    assert channels != SCOPE_CHANNELS
    frame = torch.rand(batch, channels, FRAME_RESOLUTION, FRAME_RESOLUTION)

    result = None
    with pytest.raises(FrameShapeError):
        # No latent is produced: the call raises before returning anything.
        result = encoder.forward(frame)
    assert result is None, "Encoder produced a latent for a malformed frame."


@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(
    batch=_batch_sizes,
    resolution=_wrong_resolutions,
    data=st.data(),
)
def test_encoder_rejects_wrong_resolution(
    batch: int, resolution: int, data: st.DataObject
) -> None:
    """A frame whose spatial resolution != 64x64 is rejected; no latent (Req 3.5).

    Validates: Requirements 3.5
    """
    encoder = Encoder(channels=SCOPE_CHANNELS)
    assert resolution != FRAME_RESOLUTION
    # Vary whether height, width, or both are wrong.
    mode = data.draw(st.sampled_from(["both", "height", "width"]))
    h = resolution if mode in ("both", "height") else FRAME_RESOLUTION
    w = resolution if mode in ("both", "width") else FRAME_RESOLUTION
    frame = torch.rand(batch, SCOPE_CHANNELS, h, w)

    result = None
    with pytest.raises(FrameShapeError):
        result = encoder.forward(frame)
    assert result is None, "Encoder produced a latent for a wrong-resolution frame."


@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(batch=_batch_sizes, data=st.data())
def test_encoder_rejects_wrong_rank(batch: int, data: st.DataObject) -> None:
    """A frame whose rank is not 4-D is rejected; no latent (Req 3.5).

    Validates: Requirements 3.5
    """
    encoder = Encoder(channels=SCOPE_CHANNELS)
    # Build malformed-rank shapes (rank 2, 3, and 5).
    rank = data.draw(st.sampled_from([2, 3, 5]))
    if rank == 2:
        shape = (FRAME_RESOLUTION, FRAME_RESOLUTION)
    elif rank == 3:
        shape = (SCOPE_CHANNELS, FRAME_RESOLUTION, FRAME_RESOLUTION)
    else:  # rank == 5
        shape = (batch, 1, SCOPE_CHANNELS, FRAME_RESOLUTION, FRAME_RESOLUTION)
    frame = torch.rand(*shape)

    result = None
    with pytest.raises(FrameShapeError):
        result = encoder.forward(frame)
    assert result is None, "Encoder produced a latent for a wrong-rank frame."


# ---------------------------------------------------------------------------
# Property 10 -- RSSM: action-dim mismatch rejected, state unchanged
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(
    batch=_batch_sizes,
    bad_dim=_wrong_action_dims,
    data=st.data(),
)
def test_rssm_rejects_wrong_action_dim_and_leaves_state_unchanged(
    batch: int, bad_dim: int, data: st.DataObject
) -> None:
    """An action whose dim != action_dim is rejected; prev state is unchanged (Req 4.6).

    Validates: Requirements 4.6
    """
    rssm = RSSM(action_dim=ACTION_DIM)
    prev = rssm.initial(batch)

    # Snapshot the previous state so we can prove it is not mutated.
    deter_before = prev.deter.clone()
    stoch_before = prev.stoch.clone()

    assert bad_dim != ACTION_DIM
    # Optionally vary whether the action is rank-1 (just (bad_dim,)) or batched.
    shape_mode = data.draw(st.sampled_from(["batched", "rank1"]))
    if shape_mode == "batched":
        bad_action = torch.rand(batch, bad_dim)
    else:
        bad_action = torch.rand(bad_dim)

    result = None
    with pytest.raises(InvalidActionError):
        # No prior is produced: the call raises before returning anything.
        result = rssm.predict_prior(prev, bad_action)
    assert result is None, "RSSM produced a prior for an invalid action."

    # The functional RSSM must NOT mutate its input state (Req 4.6).
    assert torch.equal(prev.deter, deter_before), "prev.deter was mutated."
    assert torch.equal(prev.stoch, stoch_before), "prev.stoch was mutated."
