# Feature: odyssey, Property 20: Critic produces a single scalar value estimate
"""Property-based tests verifying that the Critic always produces a single finite
scalar value estimate per batch element.

Property 20: Critic produces a single scalar value estimate

    For any LatentState, Critic()(state) produces a (B,) tensor (1D, length B)
    of finite scalars.

Validates: Requirements 10.4
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.behavior import Critic
from agent.types import LatentState

# MVP latent dimensions
DETER_DIM = 256
NUM_CATEGORICALS = 32
NUM_CLASSES = 32


@seed(20240620)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_critic_output_shape_is_1d_batch(batch: int, torch_seed: int) -> None:
    """Critic output is a 1-D tensor of length B (not 2-D or scalar).

    The Critic must produce exactly one scalar per batch element, returned as a
    1-D tensor shaped (B,) — not (B, 1) or a 0-D scalar.

    Validates: Requirements 10.4
    """
    torch.manual_seed(torch_seed)

    critic = Critic(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    critic.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        value = critic(state)

    # Must be exactly 1-D
    assert value.dim() == 1, (
        f"Expected 1-D output, got {value.dim()}-D tensor of shape {value.shape}"
    )
    # Length must equal batch size
    assert value.shape == (batch,), (
        f"Expected shape ({batch},), got {value.shape}"
    )


@seed(20240620)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_critic_output_values_are_finite(batch: int, torch_seed: int) -> None:
    """Critic output contains only finite values (no NaN or Inf).

    The Critic applies no output activation so values are unbounded, but they
    must always be finite real numbers to be valid value estimates.

    Validates: Requirements 10.4
    """
    torch.manual_seed(torch_seed)

    critic = Critic(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    critic.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        value = critic(state)

    assert torch.all(torch.isfinite(value)), (
        f"Critic produced non-finite values: {value}"
    )


@seed(20240620)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_critic_output_dtype_is_float32(batch: int, torch_seed: int) -> None:
    """Critic output dtype is float32.

    Validates: Requirements 10.4
    """
    torch.manual_seed(torch_seed)

    critic = Critic(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    critic.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        value = critic(state)

    assert value.dtype == torch.float32, (
        f"Expected float32, got {value.dtype}"
    )


@seed(20240620)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_critic_scalar_shape_and_finite_combined(batch: int, torch_seed: int) -> None:
    """Combined check: Critic produces (B,) float32 tensor of finite scalars.

    This is the primary property test for Property 20, checking shape, dtype,
    and finiteness together in one assertion.

    Validates: Requirements 10.4
    """
    torch.manual_seed(torch_seed)

    critic = Critic(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    critic.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        value = critic(state)

    # Shape: 1-D tensor of length B
    assert value.shape == (batch,), (
        f"Expected ({batch},), got {value.shape}"
    )
    # dtype
    assert value.dtype == torch.float32

    # All values are finite (no NaN, no Inf)
    assert torch.all(torch.isfinite(value)), (
        f"Non-finite values in critic output: {value}"
    )
