# Feature: odyssey, Property 11: Decoder reconstructions are scope-correct and normalized
"""Property-based tests for :class:`agent.decoder.Decoder` scope-correct reconstructions.

This module implements design Correctness Property 11:

    Property 11: Decoder reconstructions are scope-correct and normalized

    For any latent state, the Decoder produces a reconstructed frame of shape
    (channels, 64, 64) matching the active scope, with every pixel value in
    [0.0, 1.0].

The ``Decoder`` uses a final ``Sigmoid`` activation which mathematically
guarantees all outputs lie in ``[0.0, 1.0]``. This test verifies that
guarantee holds across arbitrary latent values and batch sizes for both the
MVP grayscale scope (channels=1) and the RGB scope (channels=3).

Validates: Requirements 5.1, 9.3
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.decoder import Decoder
from agent.types import LatentState

# Fixed latent dimensions as defined by the DreamerV2-style architecture.
DETER_DIM = 256
NUM_CATEGORICALS = 32
NUM_CLASSES = 32


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    channels=st.sampled_from([1, 3]),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_decoder_output_shape_and_pixels_in_range(
    batch: int, channels: int, torch_seed: int
) -> None:
    """Decoder output has shape ``(B, channels, 64, 64)`` and all pixels in ``[0.0, 1.0]``.

    Validates: Requirements 5.1, 9.3
    """
    torch.manual_seed(torch_seed)

    decoder = Decoder(
        channels=channels,
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    decoder.eval()

    # Build a random LatentState; stoch need not be one-hot —
    # the decoder only concatenates after flattening (Req 5.1).
    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        output = decoder(state)

    # --- Shape checks (Req 5.1) ---
    expected_shape = (batch, channels, 64, 64)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    # --- dtype check ---
    assert output.dtype == torch.float32, (
        f"Expected float32 output, got {output.dtype}"
    )

    # --- Pixel-range checks: Sigmoid guarantees [0.0, 1.0] (Req 9.3) ---
    assert torch.all(output >= 0.0), (
        f"Decoder produced pixel values below 0.0: min={output.min().item()}"
    )
    assert torch.all(output <= 1.0), (
        f"Decoder produced pixel values above 1.0: max={output.max().item()}"
    )


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_decoder_grayscale_scope(batch: int, torch_seed: int) -> None:
    """MVP grayscale scope (channels=1) produces single-channel 64×64 frames.

    Validates: Requirements 5.1, 9.3
    """
    torch.manual_seed(torch_seed)

    decoder = Decoder(channels=1)
    decoder.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        output = decoder(state)

    assert output.shape == (batch, 1, 64, 64), (
        f"Grayscale decoder: expected (B, 1, 64, 64), got {output.shape}"
    )
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
        "Grayscale decoder output pixels outside [0.0, 1.0]"
    )


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_decoder_rgb_scope(batch: int, torch_seed: int) -> None:
    """RGB scope (channels=3) produces three-channel 64×64 frames.

    Validates: Requirements 5.1, 9.3
    """
    torch.manual_seed(torch_seed)

    decoder = Decoder(channels=3)
    decoder.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        output = decoder(state)

    assert output.shape == (batch, 3, 64, 64), (
        f"RGB decoder: expected (B, 3, 64, 64), got {output.shape}"
    )
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
        "RGB decoder output pixels outside [0.0, 1.0]"
    )
