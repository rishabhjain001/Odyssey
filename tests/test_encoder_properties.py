# Feature: odyssey, Property 9: Straight-through sampling is one-hot on forward and differentiable to the encoder
"""Property-based tests for :class:`agent.encoder.Encoder` straight-through sampling.

This module implements design Correctness Property 9:

    Property 9: Straight-through sampling is one-hot on forward and
    differentiable to the encoder

    For any encoder logits, the sampled latent is one-hot on the forward pass,
    and after backpropagation a gradient propagates to the encoder logits (the
    straight-through estimator yields a defined, non-detached gradient path).

The straight-through estimator implemented in ``Encoder.sample`` is
``sample = onehot + (probs - probs.detach())``: the forward value is one-hot per
categorical variable, while gradients route to the logits (and hence the
encoder) through the non-detached ``probs`` term.

Validates: Requirements 3.4
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.encoder import Encoder
from agent.types import CategoricalLatent

# Latent shape fixed by the MVP configuration: 32 categorical variables, each a
# distribution over 32 classes (Req 3.1).
NUM_CATEGORICALS = 32
NUM_CLASSES = 32


def _assert_one_hot(sample: torch.Tensor, num_categoricals: int, num_classes: int) -> None:
    """Assert ``sample`` is one-hot per categorical variable along the last dim.

    Each of the ``num_categoricals`` variables must have exactly one entry equal
    to ``1.0`` and the remaining entries equal to ``0.0`` (values in {0, 1}).
    """
    batch = sample.shape[0]
    assert sample.shape == (batch, num_categoricals, num_classes)

    # Values are exactly in {0.0, 1.0}.
    is_zero = sample == 0.0
    is_one = sample == 1.0
    assert torch.all(is_zero | is_one), "straight-through sample has non-{0,1} values"

    # Exactly one 1.0 per variable -> per-variable sum == 1 and one-count == 1.
    per_var_sum = sample.sum(dim=-1)
    assert torch.all(per_var_sum == 1.0), "each categorical variable must sum to 1.0"
    one_counts = is_one.sum(dim=-1)
    assert torch.all(one_counts == 1), "each categorical variable must have exactly one 1.0"


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_sample_is_one_hot_and_differentiable_to_logits(batch: int, torch_seed: int) -> None:
    """``Encoder.sample`` is one-hot on forward and routes gradient to logits.

    Validates: Requirements 3.4
    """
    # Set the seed deterministically from the hypothesis-provided integer so the
    # stochastic OneHotCategorical draw is reproducible within this example.
    torch.manual_seed(torch_seed)

    encoder = Encoder(channels=1, num_categoricals=NUM_CATEGORICALS, num_classes=NUM_CLASSES)

    logits = torch.randn(
        batch, NUM_CATEGORICALS, NUM_CLASSES, requires_grad=True
    )

    sample = encoder.sample(logits)

    # ONE-HOT on the forward pass.
    _assert_one_hot(sample, NUM_CATEGORICALS, NUM_CLASSES)

    # DIFFERENTIABLE to the logits: a scalar derived from the sample must yield a
    # defined, non-zero gradient on the logits via the straight-through path.
    scalar = sample.sum()
    scalar.backward()

    assert logits.grad is not None, "straight-through path produced no gradient on logits"
    assert torch.isfinite(logits.grad).all(), "logits gradient must be finite"
    assert torch.any(logits.grad != 0.0), "straight-through gradient must be non-zero"


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_forward_path_one_hot_and_reaches_encoder_parameters(
    batch: int, torch_seed: int
) -> None:
    """Full ``Encoder.forward`` yields a one-hot sample with gradient to params.

    Validates: Requirements 3.4
    """
    torch.manual_seed(torch_seed)

    encoder = Encoder(channels=1, num_categoricals=NUM_CATEGORICALS, num_classes=NUM_CLASSES)

    frame = torch.rand(batch, 1, 64, 64, requires_grad=True)

    latent = encoder(frame)
    assert isinstance(latent, CategoricalLatent)

    # The sampled latent is one-hot per categorical variable on the forward pass.
    _assert_one_hot(latent.sample, NUM_CATEGORICALS, NUM_CLASSES)

    # Backprop from the sample must reach the encoder parameters through the
    # straight-through path (some parameter has a defined, non-zero gradient).
    encoder.zero_grad(set_to_none=True)
    latent.sample.sum().backward()

    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient via straight-through"
    assert any(torch.any(g != 0.0) for g in grads), (
        "straight-through path produced only zero gradients on encoder parameters"
    )
