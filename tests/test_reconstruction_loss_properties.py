# Feature: odyssey, Property 12: Reconstruction loss is non-negative and zero exactly when frames are identical
"""Property-based tests for :func:`training.losses.reconstruction_loss`.

This module implements design Correctness Property 12:

    Property 12: Reconstruction loss is non-negative and zero exactly when
    frames are identical

    For any pair of equal-shaped frames, ``reconstruction_loss`` is
    non-negative and equals zero if and only if the frames are identical
    (torch.equal). For any pair of frames whose shapes differ,
    ``reconstruction_loss`` raises :class:`~agent.errors.ReconstructionShapeMismatch`
    and does NOT return a float.

Validates: Requirements 5.2, 5.5
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from agent.errors import ReconstructionShapeMismatch
from training.losses import reconstruction_loss

# Representative frame shapes used across properties.
FRAME_SHAPES = [(1, 64, 64), (3, 64, 64), (2, 1, 64, 64)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_frame(shape: tuple[int, ...], torch_seed: int) -> torch.Tensor:
    """Return a random float32 tensor with the given shape."""
    rng = torch.Generator()
    rng.manual_seed(torch_seed)
    return torch.randn(shape, generator=rng, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Property 1: Non-negativity
# ---------------------------------------------------------------------------

@seed(20240612)
@settings(max_examples=20, deadline=None)
@given(
    shape_idx=st.integers(min_value=0, max_value=len(FRAME_SHAPES) - 1),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_reconstruction_loss_non_negative(
    shape_idx: int, seed_a: int, seed_b: int
) -> None:
    """reconstruction_loss is always >= 0.0 for any same-shape frame pair.

    Validates: Requirements 5.2, 5.5
    """
    shape = FRAME_SHAPES[shape_idx]
    input_frame = _random_frame(shape, seed_a)
    recon_frame = _random_frame(shape, seed_b)

    loss = reconstruction_loss(input_frame, recon_frame)

    assert loss >= 0.0, (
        f"reconstruction_loss returned a negative value {loss} "
        f"for shape {shape}"
    )


# ---------------------------------------------------------------------------
# Property 2: Zero iff identical (forward direction — identical → zero)
# ---------------------------------------------------------------------------

@seed(20240612)
@settings(max_examples=20, deadline=None)
@given(
    shape_idx=st.integers(min_value=0, max_value=len(FRAME_SHAPES) - 1),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_reconstruction_loss_zero_for_identical_frames(
    shape_idx: int, torch_seed: int
) -> None:
    """reconstruction_loss == 0.0 when input_frame and recon_frame are the same tensor.

    Passing input_frame as both arguments guarantees torch.equal returns True
    and the MSE must therefore be exactly 0.0.

    Validates: Requirements 5.2, 5.5
    """
    shape = FRAME_SHAPES[shape_idx]
    frame = _random_frame(shape, torch_seed)

    loss = reconstruction_loss(frame, frame)

    assert loss == 0.0, (
        f"Expected reconstruction_loss to be 0.0 for identical frames, "
        f"got {loss} for shape {shape}"
    )


# ---------------------------------------------------------------------------
# Property 3: Zero iff identical (backward direction — non-identical → non-zero)
# ---------------------------------------------------------------------------

@seed(20240612)
@settings(max_examples=20, deadline=None)
@given(
    shape_idx=st.integers(min_value=0, max_value=len(FRAME_SHAPES) - 1),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
    noise_scale=st.floats(min_value=0.1, max_value=1.0),
)
def test_reconstruction_loss_nonzero_for_nonidentical_frames(
    shape_idx: int, seed_a: int, seed_b: int, noise_scale: float
) -> None:
    """reconstruction_loss > 0.0 when frames differ by at least a small delta.

    We construct a definitely-non-identical pair by taking one frame and
    adding deterministic noise scaled by noise_scale, then assume they are
    not equal (the noise ensures this almost surely, but assume guards against
    any degenerate floating-point collapse).

    Validates: Requirements 5.2, 5.5
    """
    shape = FRAME_SHAPES[shape_idx]
    input_frame = _random_frame(shape, seed_a)

    # Build a distinct recon frame: start from a different random base and
    # add a guaranteed nonzero delta to the first element.
    recon_frame = _random_frame(shape, seed_b).clone()
    # Ensure at least one element differs by adding noise to all values.
    noise_gen = torch.Generator()
    noise_gen.manual_seed(seed_a ^ seed_b)
    noise = torch.empty_like(recon_frame).normal_(generator=noise_gen) * noise_scale
    recon_frame = recon_frame + noise

    assume(not torch.equal(input_frame, recon_frame))

    loss = reconstruction_loss(input_frame, recon_frame)

    assert loss > 0.0, (
        f"Expected reconstruction_loss > 0.0 for non-identical frames, "
        f"got {loss} for shape {shape}"
    )


# ---------------------------------------------------------------------------
# Property 4: Shape mismatch raises ReconstructionShapeMismatch
# ---------------------------------------------------------------------------

@seed(20240612)
@settings(max_examples=20, deadline=None)
@given(
    shape_a_idx=st.integers(min_value=0, max_value=len(FRAME_SHAPES) - 1),
    shape_b_idx=st.integers(min_value=0, max_value=len(FRAME_SHAPES) - 1),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_reconstruction_loss_raises_on_shape_mismatch(
    shape_a_idx: int, shape_b_idx: int, seed_a: int, seed_b: int
) -> None:
    """reconstruction_loss raises ReconstructionShapeMismatch for differing shapes.

    Only exercised when the two sampled shape indices are different (ensured
    via hypothesis.assume). Confirms the function raises rather than returning
    a float, satisfying Req 5.5 (mismatched-shape pairs are excluded and an
    error is raised).

    Validates: Requirements 5.2, 5.5
    """
    assume(shape_a_idx != shape_b_idx)

    shape_a = FRAME_SHAPES[shape_a_idx]
    shape_b = FRAME_SHAPES[shape_b_idx]

    input_frame = _random_frame(shape_a, seed_a)
    recon_frame = _random_frame(shape_b, seed_b)

    with pytest.raises(ReconstructionShapeMismatch):
        reconstruction_loss(input_frame, recon_frame)
