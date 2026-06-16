# Feature: odyssey, Property 15: Discount predictions are valid continuation probabilities
"""Property-based tests for :class:`agent.heads.RewardPredictor` and
:class:`agent.heads.DiscountPredictor`.

This module implements design Correctness Property 15:

    Property 15: Discount predictions are valid continuation probabilities

    For any latent state, the Discount_Predictor output is a scalar in
    [0.0, 1.0], and the Reward_Predictor output is a finite scalar.

The ``DiscountPredictor`` applies a Sigmoid activation which mathematically
guarantees all outputs lie in ``[0.0, 1.0]``. The ``RewardPredictor`` applies
no final activation, so its output is unbounded but must be finite.

Validates: Requirements 6.1, 6.2
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.heads import DiscountPredictor, RewardPredictor
from agent.types import LatentState

# Fixed latent dimensions matching the MVP DreamerV2-style architecture.
DETER_DIM = 256
NUM_CATEGORICALS = 32
NUM_CLASSES = 32


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_discount_predictor_output_in_range(batch: int, torch_seed: int) -> None:
    """DiscountPredictor output is shape ``(B,)`` with all values in ``[0.0, 1.0]``.

    The Sigmoid activation inside DiscountPredictor guarantees values lie in
    ``[0.0, 1.0]`` for any finite input. This property verifies the contract
    holds for arbitrary latent states and batch sizes.

    Validates: Requirements 6.2
    """
    torch.manual_seed(torch_seed)

    head = DiscountPredictor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    head.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        output = head(state)

    # --- Shape check: must be 1-D tensor of length B ---
    assert output.shape == (batch,), (
        f"Expected output shape ({batch},), got {output.shape}"
    )

    # --- dtype check ---
    assert output.dtype == torch.float32, (
        f"Expected float32, got {output.dtype}"
    )

    # --- Range checks: Sigmoid guarantees [0.0, 1.0] (Req 6.2) ---
    assert torch.all(output >= 0.0), (
        f"DiscountPredictor produced values below 0.0: min={output.min().item()}"
    )
    assert torch.all(output <= 1.0), (
        f"DiscountPredictor produced values above 1.0: max={output.max().item()}"
    )


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_reward_predictor_output_is_finite(batch: int, torch_seed: int) -> None:
    """RewardPredictor output is shape ``(B,)`` with all values finite.

    The RewardPredictor applies no output activation, so values are unbounded
    but must be finite real numbers for valid reward signals.

    Validates: Requirements 6.1
    """
    torch.manual_seed(torch_seed)

    head = RewardPredictor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    head.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        output = head(state)

    # --- Shape check: must be 1-D tensor of length B ---
    assert output.shape == (batch,), (
        f"Expected output shape ({batch},), got {output.shape}"
    )

    # --- dtype check ---
    assert output.dtype == torch.float32, (
        f"Expected float32, got {output.dtype}"
    )

    # --- Finiteness check: no NaN or Inf (Req 6.1) ---
    assert torch.all(torch.isfinite(output)), (
        f"RewardPredictor produced non-finite values: {output}"
    )


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_both_heads_output_shape_is_batch(batch: int, torch_seed: int) -> None:
    """Both RewardPredictor and DiscountPredictor produce a 1-D ``(B,)`` tensor.

    Verifies the shared shape contract for both heads across varying batch sizes.

    Validates: Requirements 6.1, 6.2
    """
    torch.manual_seed(torch_seed)

    reward_head = RewardPredictor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    discount_head = DiscountPredictor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    reward_head.eval()
    discount_head.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        reward_out = reward_head(state)
        discount_out = discount_head(state)

    assert reward_out.shape == (batch,), (
        f"RewardPredictor: expected ({batch},), got {reward_out.shape}"
    )
    assert discount_out.shape == (batch,), (
        f"DiscountPredictor: expected ({batch},), got {discount_out.shape}"
    )
