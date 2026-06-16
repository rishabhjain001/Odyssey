"""Behavior model for Odyssey: Actor, Critic, imagination, and λ-returns.

This module provides the Actor-Critic policy and the imagination machinery used
for behavior learning entirely inside imagined latent rollouts (Req 10.1–10.6).

Components
----------
* :class:`Actor` -- maps a :class:`~agent.types.LatentState` to a bounded
  continuous action ``[steering ∈ [-1,1], accel ∈ [0,1], brake ∈ [0,1]]``
  (Req 10.3).
* :class:`Critic` -- maps a :class:`~agent.types.LatentState` to a single
  finite scalar value estimate (Req 10.4).
* :func:`imagine` -- unrolls the RSSM prior forward using Actor actions for a
  configured horizon, predicting rewards and discounts at each step, with zero
  environment interaction (Req 10.1, 10.2).
* :func:`lambda_returns` -- computes bootstrapped λ-return targets using a
  standard backward recursion over predicted rewards, discounts, and value
  bootstraps (Req 10.5).

Design references:
    - DreamerV2 (Hafner et al., 2020): imagined actor-critic trained on latent
      rollouts with λ-return targets.
    - Dreamer (Hafner et al., 2019): λ-return computation over imagined
      trajectories with a learned value bootstrap.
    - RSSM / PlaNet (Hafner et al., 2018): prior prediction advancing the
      latent state without environment interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from agent.types import ImaginedRollout, LatentState

__all__ = ["Actor", "Critic", "imagine", "lambda_returns"]

# Shared hidden dimension for Actor and Critic (design discretion).
_HIDDEN_DIM = 256


def _build_feature(state: LatentState) -> Tensor:
    """Concatenate flattened stoch and deter into a single feature vector.

    Args:
        state: A :class:`~agent.types.LatentState` with ``deter`` of shape
            ``(B, deter_dim)`` and ``stoch`` of shape
            ``(B, num_categoricals, num_classes)``.

    Returns:
        A 2-D tensor of shape ``(B, num_categoricals*num_classes + deter_dim)``.
    """
    batch = state.deter.shape[0]
    stoch_flat = state.stoch.reshape(batch, -1)          # (B, 1024) for MVP
    return torch.cat([stoch_flat, state.deter], dim=-1)  # (B, 1280) for MVP


# ---------------------------------------------------------------------------
# Actor (Req 10.3)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Policy network mapping a latent state to a bounded continuous action.

    The output layer is split into three heads to enforce the CarRacing action
    space (Req 10.3):

    * **Steering** ``∈ [-1.0, 1.0]`` -- a :func:`torch.tanh` applied to a
      single linear output.
    * **Acceleration** ``∈ [0.0, 1.0]`` -- a :func:`torch.sigmoid` applied to a
      single linear output.
    * **Brake** ``∈ [0.0, 1.0]`` -- a :func:`torch.sigmoid` applied to a single
      linear output.

    The three bounded outputs are concatenated into an action vector of shape
    ``(B, 3)``.

    Architecture (hidden width = 256, design discretion)::

        Linear(stoch_features + deter_dim → 256) → ELU
        → Linear(256 → 3)  [split: tanh([:1]), sigmoid([1:3])]

    Args:
        deter_dim: Dimensionality of the RSSM deterministic state (default 256).
        num_categoricals: Number of categorical variables (default 32).
        num_classes: Number of classes per categorical variable (default 32).

    Example::

        actor = Actor()
        state = LatentState(deter=torch.zeros(4, 256), stoch=torch.zeros(4, 32, 32))
        action = actor(state)   # shape (4, 3), bounded by construction
    """

    def __init__(
        self,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        stoch_features = num_categoricals * num_classes  # 1024 for MVP
        in_features = stoch_features + deter_dim         # 1280 for MVP

        self.trunk = nn.Sequential(
            nn.Linear(in_features, _HIDDEN_DIM),
            nn.ELU(inplace=True),
        )
        # Single head producing 3 raw logits; bounds applied in forward().
        self.head = nn.Linear(_HIDDEN_DIM, 3)

    def forward(self, state: LatentState) -> Tensor:
        """Map a latent state to a bounded continuous action.

        Args:
            state: A :class:`~agent.types.LatentState` with ``deter`` of shape
                ``(B, deter_dim)`` and ``stoch`` of shape
                ``(B, num_categoricals, num_classes)``.

        Returns:
            Action tensor of shape ``(B, 3)`` where:

            * ``action[:, 0]`` is steering in ``[-1.0, 1.0]`` (tanh),
            * ``action[:, 1]`` is acceleration in ``[0.0, 1.0]`` (sigmoid),
            * ``action[:, 2]`` is brake in ``[0.0, 1.0]`` (sigmoid).

            The bounds are enforced by the activation functions and hold for
            every possible input (Req 10.3).
        """
        features = _build_feature(state)        # (B, 1280)
        hidden = self.trunk(features)            # (B, 256)
        raw = self.head(hidden)                  # (B, 3)

        steering = torch.tanh(raw[:, 0:1])      # (B, 1) ∈ [-1, 1]
        accel = torch.sigmoid(raw[:, 1:2])      # (B, 1) ∈ [0, 1]
        brake = torch.sigmoid(raw[:, 2:3])      # (B, 1) ∈ [0, 1]

        return torch.cat([steering, accel, brake], dim=-1)  # (B, 3) -- Req 10.3


# ---------------------------------------------------------------------------
# Critic (Req 10.4)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """Value network estimating expected return from a latent state (Req 10.4).

    Produces a single finite scalar value estimate per element in the batch.

    Architecture (hidden width = 256, design discretion)::

        Linear(stoch_features + deter_dim → 256) → ELU
        → Linear(256 → 1) → squeeze (B,1) → (B,)

    No output activation is applied; the raw linear output is the value
    estimate. The Critic is trained to regress bootstrapped λ-return targets.

    Args:
        deter_dim: Dimensionality of the RSSM deterministic state (default 256).
        num_categoricals: Number of categorical variables (default 32).
        num_classes: Number of classes per categorical variable (default 32).

    Example::

        critic = Critic()
        state = LatentState(deter=torch.zeros(4, 256), stoch=torch.zeros(4, 32, 32))
        value = critic(state)   # shape (4,), finite scalar per element
    """

    def __init__(
        self,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        stoch_features = num_categoricals * num_classes  # 1024 for MVP
        in_features = stoch_features + deter_dim         # 1280 for MVP

        self.net = nn.Sequential(
            nn.Linear(in_features, _HIDDEN_DIM),
            nn.ELU(inplace=True),
            nn.Linear(_HIDDEN_DIM, 1),
        )

    def forward(self, state: LatentState) -> Tensor:
        """Estimate the value of a latent state (Req 10.4).

        Args:
            state: A :class:`~agent.types.LatentState` with ``deter`` of shape
                ``(B, deter_dim)`` and ``stoch`` of shape
                ``(B, num_categoricals, num_classes)``.

        Returns:
            A 1-D tensor of shape ``(B,)`` containing one finite scalar value
            estimate per element of the batch. The output is a raw linear value
            without bounds.
        """
        features = _build_feature(state)   # (B, 1280)
        out = self.net(features)           # (B, 1)
        return out.squeeze(-1)             # (B,) -- Req 10.4


# ---------------------------------------------------------------------------
# imagine (Req 10.1, 10.2)
# ---------------------------------------------------------------------------

def imagine(
    rssm: "RSSM",  # type: ignore[name-defined]  # noqa: F821
    actor: Actor,
    reward_p: "RewardPredictor",  # type: ignore[name-defined]  # noqa: F821
    discount_p: "DiscountPredictor",  # type: ignore[name-defined]  # noqa: F821
    start: LatentState,
    horizon: int,
) -> ImaginedRollout:
    """Unroll the RSSM prior forward using Actor actions for ``horizon`` steps.

    This function generates an :class:`~agent.types.ImaginedRollout` entirely
    inside the latent space — **no environment interaction occurs** (Req 10.1,
    10.2). Only :meth:`~agent.rssm.RSSM.predict_prior` is used; the real
    CarRacing environment is never called.

    At each imagined step ``t``:

    1. The Actor is queried with the current :class:`~agent.types.LatentState`
       to obtain an action (Req 10.3).
    2. The RSSM prior is advanced via
       :meth:`~agent.rssm.RSSM.predict_prior` to produce the next
       :class:`~agent.types.LatentState` (Req 4.3, 4.5).
    3. The :class:`~agent.heads.RewardPredictor` and
       :class:`~agent.heads.DiscountPredictor` are applied to the *new* state to
       predict the reward and discount at that step.

    The collected states, actions, rewards, and discounts are assembled into an
    :class:`~agent.types.ImaginedRollout`.

    Args:
        rssm: The RSSM world model used to advance the latent state.
        actor: The Actor policy used to select imagined actions.
        reward_p: The reward predictor applied to each new latent state.
        discount_p: The discount predictor applied to each new latent state.
        start: The starting :class:`~agent.types.LatentState` of shape
            ``(B, ...)``.
        horizon: Number of imagination steps ``H`` (must be ≥ 1).

    Returns:
        An :class:`~agent.types.ImaginedRollout` with:

        * ``states``: list of ``H`` :class:`~agent.types.LatentState` objects,
          one per imagined step (the *next* state after each action).
        * ``actions``: Tensor of shape ``(H, B, 3)`` -- the actor actions taken
          at each step.
        * ``rewards``: Tensor of shape ``(H, B)`` -- predicted scalar rewards at
          each step.
        * ``discounts``: Tensor of shape ``(H, B)`` -- predicted continuation
          probabilities at each step.

    Note:
        Only :meth:`~agent.rssm.RSSM.predict_prior` is used (no
        ``env.step``). This is intentional and verifiable: the ``rssm``
        argument's ``predict_prior`` is the only path by which the state
        advances (Req 10.1, 10.2).
    """
    states: list[LatentState] = []
    actions_list: list[Tensor] = []
    rewards_list: list[Tensor] = []
    discounts_list: list[Tensor] = []

    state = start

    for _ in range(horizon):
        # 1. Actor selects action from current state (Req 10.3).
        action = actor(state)                          # (B, 3)

        # 2. Advance latent state using RSSM prior ONLY — zero env interaction
        #    (Req 10.1, 10.2, 4.3, 4.5).
        next_state = rssm.predict_prior(state, action)  # LatentState

        # 3. Predict reward and discount from the new latent state.
        reward = reward_p(next_state)                  # (B,)
        discount = discount_p(next_state)              # (B,)

        states.append(next_state)
        actions_list.append(action)
        rewards_list.append(reward)
        discounts_list.append(discount)

        state = next_state

    # Stack lists into tensors: each element is (B,*) → (H, B, *).
    actions_tensor = torch.stack(actions_list, dim=0)    # (H, B, 3)
    rewards_tensor = torch.stack(rewards_list, dim=0)    # (H, B)
    discounts_tensor = torch.stack(discounts_list, dim=0)  # (H, B)

    return ImaginedRollout(
        states=states,
        actions=actions_tensor,
        rewards=rewards_tensor,
        discounts=discounts_tensor,
    )


# ---------------------------------------------------------------------------
# lambda_returns (Req 10.5)
# ---------------------------------------------------------------------------

def lambda_returns(
    rewards: Tensor,
    discounts: Tensor,
    values: Tensor,
    lam: float,
) -> Tensor:
    """Compute bootstrapped λ-return targets via backward recursion (Req 10.5).

    Implements the standard DreamerV2/Dreamer λ-return recursion:

    .. math::

        G_{H-1} &= r_{H-1} + \\gamma_{H-1} \\cdot V_H \\\\
        G_t     &= r_t + \\gamma_t \\cdot \\bigl[(1-\\lambda) V_{t+1}
                   + \\lambda G_{t+1}\\bigr]

    where :math:`r_t` are the predicted rewards, :math:`\\gamma_t` are the
    predicted discount (continuation) probabilities, :math:`V_t` are value
    estimates (with :math:`V_H` as the terminal bootstrap), and :math:`\\lambda`
    is the mixing parameter.

    Special cases (Property 21):

    * ``lam = 0`` → one-step TD target:
      :math:`G_t = r_t + \\gamma_t \\cdot V_{t+1}` for all ``t``.
    * ``lam = 1`` with all ``discounts = 1`` → plain sum of future rewards
      plus terminal bootstrap: :math:`G_t = \\sum_{k=t}^{H-1} r_k + V_H`.

    The recursion is computed in reverse order (from ``t = H-1`` to ``t = 0``)
    so that each step depends only on the already-computed next return.

    Args:
        rewards: Predicted rewards, shape ``(H, B)``.
        discounts: Predicted continuation probabilities, shape ``(H, B)``.
            Each value should be in ``[0.0, 1.0]``.
        values: Value estimates (including terminal bootstrap), shape
            ``(H+1, B)``. ``values[t]`` is the critic estimate at imagined step
            ``t``; ``values[H]`` is the terminal bootstrap used in
            :math:`G_{H-1}`.
        lam: The λ mixing parameter in ``[0.0, 1.0]``:

            * ``0.0`` → pure one-step TD (no multi-step bootstrapping),
            * ``1.0`` → full Monte Carlo (sum of all future rewards + bootstrap).

    Returns:
        Tensor of shape ``(H, B)`` containing the λ-return target for each
        imagined step and each batch element.

    Note:
        This function is a pure, deterministic computation; it performs no
        gradient manipulation. It is differentiable with respect to all tensor
        inputs through the standard autograd graph.
    """
    H = rewards.shape[0]

    # Pre-allocate output buffer.
    G = torch.zeros_like(rewards)  # (H, B)

    # Terminal step (t = H-1): bootstrap from values[H].
    # G[H-1] = r[H-1] + γ[H-1] * V[H]  (Req 10.5, Property 21)
    G[H - 1] = rewards[H - 1] + discounts[H - 1] * values[H]

    # Backward recursion from t = H-2 down to t = 0.
    # G[t] = r[t] + γ[t] * ((1-λ)*V[t+1] + λ*G[t+1])
    for t in range(H - 2, -1, -1):
        G[t] = (
            rewards[t]
            + discounts[t] * ((1.0 - lam) * values[t + 1] + lam * G[t + 1])
        )

    return G
