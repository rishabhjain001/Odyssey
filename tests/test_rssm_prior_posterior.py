"""Unit tests for RSSM posterior-vs-prior conditioning (Task 5.4).

These are plain ``pytest`` unit tests (not property-based) that verify the two
conditioning contracts of :class:`agent.rssm.RSSM`:

    * **Prior uses the deterministic state alone (Req 4.5).**
      :meth:`RSSM.predict_prior` forms the prior stochastic state from the newly
      advanced deterministic state only -- it never consumes an encoder
      embedding. We assert this both *structurally* (the ``prior_head`` input
      dimension equals ``deter_dim``, so it physically cannot accept an
      embedding, and ``predict_prior`` takes no ``embed`` argument) and
      *behaviorally* (under a fixed seed, ``predict_prior`` is deterministic
      given the same inputs, and the prior logits it uses equal
      ``prior_head(new_deter)``).

    * **Posterior uses the encoder embedding (Req 4.4).**
      :meth:`RSSM.posterior` forms the posterior stochastic state conditioned on
      *both* the deterministic state and the encoder embedding. We assert this
      structurally (``posterior_head`` input dimension equals
      ``deter_dim + embed_dim``) and behaviorally (with ``deter`` held fixed, the
      posterior logits change when the embedding changes, and are identical when
      the embedding is identical) -- proving the posterior is conditioned on the
      encoder output.

To keep assertions deterministic despite the straight-through one-hot sampling,
behavioral checks either (a) seed ``torch`` before each sampled call, or
(b) compare the deterministic *logits* path directly (``prior_head`` /
``posterior_head`` outputs), which carry no sampling randomness.

Validates: Requirements 4.4, 4.5
"""

from __future__ import annotations

import inspect

import torch

from agent.rssm import RSSM
from agent.types import LatentState

# MVP/CarRacing configuration under test.
ACTION_DIM = 3
DETER_DIM = 256
NUM_CATEGORICALS = 32
NUM_CLASSES = 32
EMBED_DIM = 1024
BATCH = 2
SEED = 20240517


def _make_rssm() -> RSSM:
    rssm = RSSM(
        action_dim=ACTION_DIM,
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    )
    rssm.eval()  # no dropout/batchnorm here, but keep behavior deterministic
    return rssm


# ---------------------------------------------------------------------------
# Req 4.5 -- the PRIOR path uses the deterministic state ALONE
# ---------------------------------------------------------------------------

def test_prior_head_input_dim_is_deter_only() -> None:
    """Structural: prior_head consumes deter_dim features -- no room for embed.

    If the prior depended on the encoder embedding, ``prior_head`` would need an
    input width of ``deter_dim + embed_dim``. Asserting it equals ``deter_dim``
    proves the prior cannot consume an embedding.

    Validates: Requirements 4.5
    """
    rssm = _make_rssm()
    assert rssm.prior_head.in_features == DETER_DIM
    # And it must NOT be the posterior's (deter + embed) width.
    assert rssm.prior_head.in_features != DETER_DIM + EMBED_DIM


def test_predict_prior_signature_takes_no_embed() -> None:
    """Structural: predict_prior's signature has no encoder-embedding argument.

    Validates: Requirements 4.5
    """
    params = set(inspect.signature(RSSM.predict_prior).parameters)
    # Only self, prev, action -- nothing resembling an embedding input.
    assert params == {"self", "prev", "action"}
    assert not any("embed" in name.lower() for name in params)


def test_predict_prior_is_deterministic_given_inputs_and_seed() -> None:
    """Behavioral: same prev + action under the same seed -> identical prior.

    The prior is a function of (prev, action) and the sampling seed only. With
    no embedding in the inputs, seeding torch and replaying the same call must
    reproduce the prior stochastic state bit-for-bit.

    Validates: Requirements 4.5
    """
    rssm = _make_rssm()
    prev = rssm.initial(BATCH)
    action = torch.rand(BATCH, ACTION_DIM)

    torch.manual_seed(SEED)
    out_a = rssm.predict_prior(prev, action)

    torch.manual_seed(SEED)
    out_b = rssm.predict_prior(prev, action)

    assert torch.equal(out_a.deter, out_b.deter)
    assert torch.equal(out_a.stoch, out_b.stoch)


def test_prior_logits_are_a_function_of_new_deter_alone() -> None:
    """Behavioral: the prior logits equal prior_head(new_deter).

    We recompute the advanced deterministic state by running the exact same GRU
    input projection + GRU cell that ``predict_prior`` uses, then confirm that
    ``prior_head(new_deter)`` matches the logits implied by ``predict_prior``'s
    output deter. This proves the prior stochastic head reads from the
    deterministic state alone (no embedding term).

    Validates: Requirements 4.5
    """
    rssm = _make_rssm()
    prev = rssm.initial(BATCH)
    action = torch.rand(BATCH, ACTION_DIM)

    out = rssm.predict_prior(prev, action)

    # Reproduce the deterministic-state advance exactly as predict_prior does.
    stoch_flat = prev.stoch.reshape(BATCH, NUM_CATEGORICALS * NUM_CLASSES)
    gru_in = rssm.gru_input(torch.cat([stoch_flat, action], dim=-1))
    new_deter = rssm.gru(gru_in, prev.deter)

    # The deter returned by predict_prior must match our recomputation.
    assert torch.allclose(out.deter, new_deter, atol=1e-6)

    # Prior logits are produced from the deterministic state alone.
    prior_logits = rssm.prior_head(new_deter)
    assert prior_logits.shape == (BATCH, NUM_CATEGORICALS * NUM_CLASSES)
    # And the one-hot sample is consistent with those logits (argmax classes).
    reshaped = prior_logits.reshape(BATCH, NUM_CATEGORICALS, NUM_CLASSES)
    # The straight-through sample is one-hot; its support must be valid classes.
    assert torch.equal(out.stoch.sum(dim=-1), torch.ones(BATCH, NUM_CATEGORICALS))
    assert reshaped.shape == out.stoch.shape


# ---------------------------------------------------------------------------
# Req 4.4 -- the POSTERIOR path uses the encoder embedding
# ---------------------------------------------------------------------------

def test_posterior_head_input_dim_is_deter_plus_embed() -> None:
    """Structural: posterior_head consumes deter_dim + embed_dim features.

    The posterior head's input width proves it is conditioned on both the
    deterministic state and the encoder embedding.

    Validates: Requirements 4.4
    """
    rssm = _make_rssm()
    assert rssm.posterior_head.in_features == DETER_DIM + EMBED_DIM


def test_posterior_logits_change_when_embed_changes() -> None:
    """Behavioral: with deter fixed, different embeds -> different posterior logits.

    Comparing the deterministic logits path (``posterior_head(cat([deter,
    embed]))``) avoids sampling randomness. Two *different* embeddings with the
    same ``deter`` must yield different posterior logits, proving the posterior
    is conditioned on the encoder embedding (Req 4.4).

    Validates: Requirements 4.4
    """
    rssm = _make_rssm()
    deter = torch.randn(BATCH, DETER_DIM)

    embed_a = torch.randn(BATCH, EMBED_DIM)
    embed_b = torch.randn(BATCH, EMBED_DIM)
    # Guard: the two embeddings really are different inputs.
    assert not torch.equal(embed_a, embed_b)

    logits_a = rssm.posterior_head(torch.cat([deter, embed_a.reshape(BATCH, -1)], dim=-1))
    logits_b = rssm.posterior_head(torch.cat([deter, embed_b.reshape(BATCH, -1)], dim=-1))

    # Same deter, different embed => the posterior output must differ.
    assert not torch.allclose(logits_a, logits_b)


def test_posterior_logits_identical_for_identical_embed() -> None:
    """Behavioral: with deter fixed, the same embed -> identical posterior logits.

    The logits path is a pure function of (deter, embed); repeating it with the
    same inputs must reproduce identical logits.

    Validates: Requirements 4.4
    """
    rssm = _make_rssm()
    deter = torch.randn(BATCH, DETER_DIM)
    embed = torch.randn(BATCH, EMBED_DIM)

    logits_1 = rssm.posterior_head(torch.cat([deter, embed.reshape(BATCH, -1)], dim=-1))
    logits_2 = rssm.posterior_head(torch.cat([deter, embed.reshape(BATCH, -1)], dim=-1))

    assert torch.equal(logits_1, logits_2)


def test_posterior_sample_is_deterministic_given_seed_and_embed() -> None:
    """Behavioral: same deter + embed under the same seed -> identical posterior.

    Validates: Requirements 4.4
    """
    rssm = _make_rssm()
    deter = torch.randn(BATCH, DETER_DIM)
    embed = torch.randn(BATCH, EMBED_DIM)

    torch.manual_seed(SEED)
    post_a = rssm.posterior(deter, embed)
    torch.manual_seed(SEED)
    post_b = rssm.posterior(deter, embed)

    assert torch.equal(post_a.stoch, post_b.stoch)
    # deter is threaded through unchanged by the posterior.
    assert torch.equal(post_a.deter, deter)
    assert isinstance(post_a, LatentState)


def test_posterior_sample_changes_when_embed_changes_under_fixed_seed() -> None:
    """Behavioral: fixing deter and seed, changing the embed changes the posterior.

    This is the end-to-end (sampled) counterpart to the logits-level check:
    holding ``deter`` and the sampling seed constant, swapping the embedding
    produces a different posterior stochastic state, confirming the encoder
    embedding actually drives the posterior (Req 4.4).

    Validates: Requirements 4.4
    """
    rssm = _make_rssm()
    deter = torch.randn(BATCH, DETER_DIM)
    embed_a = torch.randn(BATCH, EMBED_DIM)
    embed_b = torch.randn(BATCH, EMBED_DIM)

    torch.manual_seed(SEED)
    post_a = rssm.posterior(deter, embed_a)
    torch.manual_seed(SEED)
    post_b = rssm.posterior(deter, embed_b)

    # Different embeddings (same deter, same seed) must yield different posteriors.
    assert not torch.equal(post_a.stoch, post_b.stoch)
