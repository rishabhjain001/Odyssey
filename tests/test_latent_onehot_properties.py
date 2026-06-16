# Feature: odyssey, Property 8: Latent state is always a valid one-hot categorical grid
"""Property-based tests for the discrete one-hot latent invariant.

This module implements design Correctness Property 8:

    Property 8: Latent state is always a valid one-hot categorical grid

    For any frame encoded by the Encoder, and for any latent state maintained by
    the RSSM across prediction steps, the stochastic component has shape
    ``(32, 32)`` and each of the 32 categorical variables is one-hot along its
    32-class dimension (exactly one ``1.0``, the rest ``0.0``).

Two paths are exercised:

    * The **Encoder** path: a random ``(B, 1, 64, 64)`` frame is encoded and the
      resulting ``CategoricalLatent.sample`` must be a valid one-hot grid.
    * The **RSSM** path, *across multiple prediction steps*: starting from
      ``RSSM.initial(B)``, ``predict_prior`` is called repeatedly with a valid
      ``(B, 3)`` action for several steps, and after EACH step the resulting
      ``LatentState.stoch`` must be a valid one-hot grid. ``posterior`` is also
      checked with a random ``(B, 1024)`` embedding.

Validates: Requirements 3.1, 4.2
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.encoder import Encoder
from agent.rssm import RSSM
from agent.types import CategoricalLatent, LatentState

# Latent shape fixed by the MVP configuration: 32 categorical variables, each a
# distribution over 32 classes (Req 3.1, 4.2).
NUM_CATEGORICALS = 32
NUM_CLASSES = 32

# CarRacing action space dimensionality ([steering, accel, brake]).
ACTION_DIM = 3

# RSSM dimensions matching the MVP configuration / implementation defaults.
DETER_DIM = 256
EMBED_DIM = 1024


def _assert_one_hot_grid(stoch: torch.Tensor) -> None:
    """Assert ``stoch`` is a valid one-hot categorical grid.

    The tensor must have shape ``(B, 32, 32)``; along the last (class) dimension
    every one of the 32 categorical variables must contain exactly one entry
    equal to ``1.0`` and the rest equal to ``0.0`` (all values in ``{0, 1}``).
    """
    assert stoch.dim() == 3, f"expected a 3-D (B, 32, 32) grid, got {tuple(stoch.shape)}"
    batch = stoch.shape[0]
    assert stoch.shape == (batch, NUM_CATEGORICALS, NUM_CLASSES), (
        f"stochastic component shape {tuple(stoch.shape)} != "
        f"(B, {NUM_CATEGORICALS}, {NUM_CLASSES})"
    )

    # Values are exactly in {0.0, 1.0}.
    is_zero = stoch == 0.0
    is_one = stoch == 1.0
    assert torch.all(is_zero | is_one), "one-hot grid has values outside {0.0, 1.0}"

    # Exactly one 1.0 per variable along the class dimension.
    per_var_sum = stoch.sum(dim=-1)
    assert torch.all(per_var_sum == 1.0), "each categorical variable must sum to 1.0"
    one_counts = is_one.sum(dim=-1)
    assert torch.all(one_counts == 1), "each categorical variable must have exactly one 1.0"


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_encoder_sample_is_valid_one_hot_grid(batch: int, torch_seed: int) -> None:
    """A frame encoded by the Encoder yields a valid one-hot ``(B,32,32)`` grid.

    Validates: Requirements 3.1, 4.2
    """
    torch.manual_seed(torch_seed)

    encoder = Encoder(
        channels=1, num_categoricals=NUM_CATEGORICALS, num_classes=NUM_CLASSES
    )
    frame = torch.rand(batch, 1, 64, 64)

    latent = encoder(frame)
    assert isinstance(latent, CategoricalLatent)
    _assert_one_hot_grid(latent.sample)


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    steps=st.integers(min_value=3, max_value=8),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_rssm_prior_one_hot_across_prediction_steps(
    batch: int, steps: int, torch_seed: int
) -> None:
    """RSSM ``predict_prior`` keeps a valid one-hot grid across many steps.

    Starting from ``initial(batch)``, prediction is advanced ``steps`` times with
    a valid ``(batch, 3)`` action; after EACH step the stochastic component must
    remain a valid one-hot ``(batch, 32, 32)`` grid.

    Validates: Requirements 3.1, 4.2
    """
    torch.manual_seed(torch_seed)

    rssm = RSSM(
        action_dim=ACTION_DIM,
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    )

    state: LatentState = rssm.initial(batch)
    for _ in range(steps):
        action = torch.randn(batch, ACTION_DIM)
        state = rssm.predict_prior(state, action)
        _assert_one_hot_grid(state.stoch)


@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_rssm_posterior_is_valid_one_hot_grid(batch: int, torch_seed: int) -> None:
    """RSSM ``posterior`` yields a valid one-hot ``(B,32,32)`` grid.

    Validates: Requirements 3.1, 4.2
    """
    torch.manual_seed(torch_seed)

    rssm = RSSM(
        action_dim=ACTION_DIM,
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    )

    deter = torch.randn(batch, DETER_DIM)
    embed = torch.randn(batch, EMBED_DIM)

    state = rssm.posterior(deter, embed)
    assert isinstance(state, LatentState)
    _assert_one_hot_grid(state.stoch)
