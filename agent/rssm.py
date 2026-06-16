"""Recurrent State-Space Model (RSSM) for Odyssey's latent dynamics.

The :class:`RSSM` is the world model's transition core. It maintains the
DreamerV2/PlaNet split of the latent state into two parts (Req 4.1, 4.2):

    * a *deterministic* recurrent component ``deter`` -- the hidden state of a
      GRU that summarizes the history of stochastic states and actions; and
    * a *stochastic* categorical component ``stoch`` -- a one-hot
      ``(B, num_categoricals, num_classes)`` grid (32x32 in the MVP config).

Both components travel together inside :class:`~agent.types.LatentState`. The
RSSM is purely *functional* with respect to that state: every method takes the
state it needs as an argument and returns a fresh :class:`LatentState`, never
mutating its inputs and never touching any environment (Req 4.3, 4.4, 4.5).

Three operations are provided:

    * :meth:`RSSM.initial` -- the zero starting state for a batch.
    * :meth:`RSSM.predict_prior` -- advance the GRU on the previous stochastic
      state and an action, then form the *prior* stochastic state from the new
      deterministic state alone (Req 4.1, 4.3, 4.5). Used in imagination, it
      consumes zero environment steps. An action whose dimensionality does not
      match the configured action space is rejected up front with
      :class:`~agent.errors.InvalidActionError`, leaving the passed-in state
      untouched (Req 4.6).
    * :meth:`RSSM.posterior` -- form the *posterior* stochastic state
      conditioned on both the deterministic state and the encoder's embedding of
      the current observation (Req 4.4).

The stochastic state is always sampled as a valid one-hot grid via the same
straight-through estimator used by :class:`agent.encoder.Encoder`, so it is both
a genuine one-hot sample on the forward pass and differentiable on the backward
pass (Req 4.2).

Design references:
    - RSSM / PlaNet (Hafner et al., 2018): the deterministic + stochastic latent
      split and the recurrent transition.
    - DreamerV2 (Hafner et al., 2020): the 32x32 discrete categorical latent and
      the straight-through gradient estimator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from agent.errors import InvalidActionError
from agent.types import LatentState

__all__ = ["RSSM", "straight_through_onehot"]


def straight_through_onehot(logits: Tensor) -> Tensor:
    """Sample a one-hot categorical grid via the straight-through estimator.

    Mirrors :meth:`agent.encoder.Encoder.sample`. For each categorical variable
    a class is sampled over the last (``num_classes``) dimension, yielding a hard
    one-hot vector. To keep the operation differentiable with respect to the
    logits, the straight-through estimator is applied::

        probs   = softmax(logits, dim=-1)
        onehot  = one_hot(sample_index)            # hard, no gradient
        sample  = onehot + (probs - probs.detach())

    The term ``(probs - probs.detach())`` is grouped deliberately: ``probs`` and
    ``probs.detach()`` hold identical values, so their difference is *exactly*
    ``0.0`` and the returned forward value is the one-hot sample bit-for-bit
    (exactly 0/1). On the backward pass the gradient flows through the
    non-detached ``probs`` term into the logits.

    Args:
        logits: Class logits of shape ``(B, num_categoricals, num_classes)``.

    Returns:
        A one-hot sample of shape ``(B, num_categoricals, num_classes)`` that is
        differentiable with respect to ``logits``.
    """
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.OneHotCategorical(probs=probs)
    onehot = dist.sample()
    return onehot + (probs - probs.detach())


class RSSM(nn.Module):
    """Recurrent State-Space Model maintaining deterministic + stochastic state.

    The RSSM holds no environment handle and performs no environment interaction
    in any method (Req 4.3, 4.4, 4.5). State is threaded through
    :class:`~agent.types.LatentState` values.

    Architecture (channel widths are design discretion; the requirements defer
    hyperparameters):

        * GRU input projection -- ``Linear(num_categoricals*num_classes +
          action_dim -> deter_dim)`` followed by an ELU activation, fed to a
          :class:`torch.nn.GRUCell` with hidden size ``deter_dim`` (Req 4.1, 4.3).
        * Prior head -- ``Linear(deter_dim -> num_categoricals*num_classes)``
          producing prior logits from the deterministic state alone (Req 4.5).
        * Posterior head -- ``Linear(deter_dim + embed_dim ->
          num_categoricals*num_classes)`` producing posterior logits conditioned
          on the deterministic state and the encoder embedding (Req 4.4).

    Args:
        action_dim: Dimensionality of the environment action vector. For
            CarRacing this is 3 (``[steering, accel, brake]``). Used to validate
            actions in :meth:`predict_prior` (Req 4.6).
        deter_dim: Size of the deterministic GRU hidden state (Req 4.1).
        num_categoricals: Number of categorical variables in the stochastic
            state (32 for the MVP config).
        num_classes: Number of classes per categorical variable (32 for the MVP
            config).
        embed_dim: Dimensionality of the encoder embedding consumed by
            :meth:`posterior`. This must match the flattened size of whatever the
            world-model trainer feeds in as ``embed`` (Req 4.4).
        hidden_dim: Width of the GRU input projection. Defaults to ``deter_dim``.
    """

    def __init__(
        self,
        action_dim: int = 3,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
        embed_dim: int = 1024,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.deter_dim = int(deter_dim)
        self.num_categoricals = int(num_categoricals)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.deter_dim

        self._stoch_features = self.num_categoricals * self.num_classes

        # Project (flattened prev.stoch + action) to the GRU input size, then
        # step the GRU to advance the deterministic state (Req 4.1, 4.3).
        self.gru_input = nn.Sequential(
            nn.Linear(self._stoch_features + self.action_dim, self.hidden_dim),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)

        # Prior logits from the deterministic state alone (Req 4.5).
        self.prior_head = nn.Linear(self.deter_dim, self._stoch_features)

        # Posterior logits from deterministic state + encoder embedding (Req 4.4).
        self.posterior_head = nn.Linear(
            self.deter_dim + self.embed_dim, self._stoch_features
        )

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def initial(self, batch: int) -> LatentState:
        """Return the zero starting latent state for ``batch`` sequences.

        The deterministic state is an all-zeros ``(batch, deter_dim)`` tensor --
        the conventional GRU initial hidden state. The stochastic state is an
        all-zeros ``(batch, num_categoricals, num_classes)`` grid. Zeros (rather
        than a uniform one-hot sample) are an acceptable and deliberate choice
        for the initial state: before any observation or action is processed
        there is no information to condition a categorical sample on, and the
        first real stochastic state is produced by the first
        :meth:`predict_prior` / :meth:`posterior` call.

        Args:
            batch: Number of independent sequences in the batch.

        Returns:
            A :class:`LatentState` with zero ``deter`` and zero ``stoch``.
        """
        device = self.prior_head.weight.device
        dtype = self.prior_head.weight.dtype
        deter = torch.zeros(batch, self.deter_dim, device=device, dtype=dtype)
        stoch = torch.zeros(
            batch,
            self.num_categoricals,
            self.num_classes,
            device=device,
            dtype=dtype,
        )
        return LatentState(deter=deter, stoch=stoch)

    # ------------------------------------------------------------------
    # Prior transition (imagination / no environment interaction)
    # ------------------------------------------------------------------
    def predict_prior(self, prev: LatentState, action: Tensor) -> LatentState:
        """Advance the GRU and form the prior stochastic state.

        Steps (Req 4.1, 4.3, 4.5):

            1. Validate that ``action``'s last dimension equals
               ``action_dim``. This check happens *first*, before any
               computation, so on mismatch :class:`InvalidActionError` is raised
               and the passed-in ``prev`` state is left completely unchanged
               (the RSSM is functional, so "unchanged" means ``prev`` is never
               mutated). (Req 4.6)
            2. Flatten ``prev.stoch`` ``(B, 32, 32) -> (B, 1024)`` and concat the
               action; project and step the GRU to get the new deterministic
               state.
            3. Form the prior logits from the new deterministic state alone and
               sample a one-hot stochastic state via the straight-through
               estimator.

        No environment interaction occurs, so this is what imagination uses to
        roll the latent state forward at zero environment-step cost.

        Args:
            prev: The previous :class:`LatentState`.
            action: Action tensor of shape ``(B, action_dim)``.

        Returns:
            A new :class:`LatentState` with the advanced ``deter`` and the prior
            ``stoch``.

        Raises:
            InvalidActionError: If ``action.shape[-1] != action_dim``. The
                ``prev`` state is left unchanged and no prior is produced.
        """
        if action.dim() == 0 or action.shape[-1] != self.action_dim:
            observed = tuple(action.shape)
            raise InvalidActionError(
                f"Action dimensionality {observed} does not match the expected "
                f"action space dim {self.action_dim} "
                f"(action.shape[-1] must equal {self.action_dim}). "
                "State left unchanged; no prior produced."
            )

        batch = prev.stoch.shape[0]
        stoch_flat = prev.stoch.reshape(batch, self._stoch_features)
        gru_in = self.gru_input(torch.cat([stoch_flat, action], dim=-1))
        deter = self.gru(gru_in, prev.deter)

        prior_logits = self.prior_head(deter)
        prior_logits = prior_logits.reshape(
            batch, self.num_categoricals, self.num_classes
        )
        prior_stoch = straight_through_onehot(prior_logits)
        return LatentState(deter=deter, stoch=prior_stoch)

    # ------------------------------------------------------------------
    # Posterior (conditioned on encoder embedding)
    # ------------------------------------------------------------------
    def posterior(self, deter: Tensor, embed: Tensor) -> LatentState:
        """Form the posterior stochastic state from ``deter`` and ``embed``.

        The posterior conditions on both the deterministic state and the
        encoder's embedding of the current observation (Req 4.4). The embedding
        is flattened to ``(B, embed_dim)`` if needed, concatenated with
        ``deter``, projected to posterior logits, reshaped to
        ``(B, num_categoricals, num_classes)``, and sampled one-hot via the
        straight-through estimator. No environment interaction occurs.

        Args:
            deter: Deterministic state of shape ``(B, deter_dim)``.
            embed: Encoder embedding of shape ``(B, embed_dim)`` (or any shape
                that flattens to ``(B, embed_dim)``).

        Returns:
            A new :class:`LatentState` carrying ``deter`` and the posterior
            ``stoch``.
        """
        batch = deter.shape[0]
        embed_flat = embed.reshape(batch, -1)
        post_logits = self.posterior_head(torch.cat([deter, embed_flat], dim=-1))
        post_logits = post_logits.reshape(
            batch, self.num_categoricals, self.num_classes
        )
        post_stoch = straight_through_onehot(post_logits)
        return LatentState(deter=deter, stoch=post_stoch)
