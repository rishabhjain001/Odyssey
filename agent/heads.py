"""Reward and Discount prediction heads for Odyssey's world model.

This module provides two lightweight :class:`torch.nn.Module` subclasses that
operate on a combined latent state and produce scalar predictions:

    * :class:`RewardPredictor` -- predicts a scalar reward for a given
      :class:`~agent.types.LatentState` (Req 6.1).
    * :class:`DiscountPredictor` -- predicts an episode-continuation probability
      in ``[0.0, 1.0]`` for a given :class:`~agent.types.LatentState` (Req 6.2).

Both heads consume the *combined* latent state by concatenating the flattened
stochastic component with the deterministic component to form a single feature
vector before passing it through a small MLP:

    ``stoch (B, 32, 32) → flatten → (B, 1024)``
    ``deter (B, 256) → (B, 256)``
    ``feature = cat([stoch_flat, deter], dim=-1) → (B, 1280)``

Architecture (design discretion; hidden width = 256 documented here):

    ``Linear(1280 → 256) → ELU → Linear(256 → 1) → squeeze to (B,)``

For :class:`RewardPredictor` the final output is a raw linear value -- the
reward can be any finite real number (Req 6.1). No activation is applied; the
caller is responsible for finiteness checks.

For :class:`DiscountPredictor` a :class:`torch.nn.Sigmoid` is applied after the
final linear layer, mapping the output into ``[0.0, 1.0]`` and guaranteeing the
continuation-probability contract (Req 6.2).

Both heads share the same constructor signature:

    ``__init__(self, deter_dim=256, num_categoricals=32, num_classes=32)``

where ``deter_dim``, ``num_categoricals``, and ``num_classes`` default to the
MVP values from the configuration and match the shapes produced by the
:class:`~agent.rssm.RSSM`.

Design references:
    - DreamerV2 (Hafner et al., 2020): reward and discount heads operating on
      the combined RSSM latent state to enable imagined-rollout return
      estimation.
    - RSSM / PlaNet (Hafner et al., 2018): deterministic + stochastic latent
      split whose combined representation feeds all world-model output heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from agent.types import LatentState

__all__ = ["RewardPredictor", "DiscountPredictor"]

# Hidden layer width for both heads (design discretion, documented here).
_HIDDEN_DIM = 256


def _build_feature(state: LatentState) -> Tensor:
    """Concatenate flattened stoch and deter into a single feature vector.

    The stochastic component ``stoch`` has shape ``(B, num_categoricals,
    num_classes)`` -- for the MVP this is ``(B, 32, 32)`` = ``(B, 1024)``
    elements. The deterministic component ``deter`` has shape ``(B, deter_dim)``
    -- for the MVP this is ``(B, 256)``. Together they form a ``(B, 1280)``
    feature vector.

    Args:
        state: A :class:`~agent.types.LatentState` with ``deter`` of shape
            ``(B, deter_dim)`` and ``stoch`` of shape
            ``(B, num_categoricals, num_classes)``.

    Returns:
        A 2-D feature tensor of shape ``(B, num_categoricals*num_classes +
        deter_dim)``.
    """
    batch = state.deter.shape[0]
    stoch_flat = state.stoch.reshape(batch, -1)           # (B, 1024)
    return torch.cat([stoch_flat, state.deter], dim=-1)   # (B, 1280)


class RewardPredictor(nn.Module):
    """Predicts a scalar reward from a latent state (Req 6.1).

    Consumes the combined latent state (flattened stoch concatenated with deter)
    and produces a raw, unbounded scalar reward per element in the batch. No
    output activation is applied -- the reward can be any finite real number.
    The caller is responsible for checking finiteness before use.

    Architecture (hidden width = 256, design discretion)::

        Linear(stoch_features + deter_dim → 256) → ELU
        → Linear(256 → 1) → squeeze (B,1) → (B,)

    Args:
        deter_dim: Dimensionality of the RSSM deterministic state (default 256).
        num_categoricals: Number of categorical variables in the stochastic
            state (default 32; MVP configuration).
        num_classes: Number of classes per categorical variable (default 32;
            MVP configuration).

    Example::

        head = RewardPredictor()
        state = LatentState(deter=torch.zeros(4, 256), stoch=torch.zeros(4, 32, 32))
        reward = head(state)   # shape (4,), dtype float32, no bounds
    """

    def __init__(
        self,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        stoch_features = num_categoricals * num_classes   # 1024 for MVP
        in_features = stoch_features + deter_dim          # 1280 for MVP

        self.net = nn.Sequential(
            nn.Linear(in_features, _HIDDEN_DIM),
            nn.ELU(inplace=True),
            nn.Linear(_HIDDEN_DIM, 1),
        )

    def forward(self, state: LatentState) -> Tensor:
        """Predict a scalar reward per latent state (Req 6.1).

        Args:
            state: A :class:`~agent.types.LatentState` with ``deter`` of shape
                ``(B, deter_dim)`` and ``stoch`` of shape
                ``(B, num_categoricals, num_classes)``.

        Returns:
            A 1-D tensor of shape ``(B,)`` containing the predicted scalar
            reward for each element of the batch. The values are raw linear
            outputs -- no activation is applied, so any finite real value is
            possible. The caller should verify finiteness before using the
            predictions.
        """
        features = _build_feature(state)    # (B, 1280)
        out = self.net(features)            # (B, 1)
        return out.squeeze(-1)              # (B,)  -- Req 6.1


class DiscountPredictor(nn.Module):
    """Predicts a continuation probability in [0.0, 1.0] from a latent state (Req 6.2).

    Consumes the combined latent state (flattened stoch concatenated with deter)
    and produces a scalar episode-continuation probability per element in the
    batch. A :class:`torch.nn.Sigmoid` is applied to the final linear output so
    the returned values are guaranteed to lie in ``[0.0, 1.0]``.

    Architecture (hidden width = 256, design discretion)::

        Linear(stoch_features + deter_dim → 256) → ELU
        → Linear(256 → 1) → Sigmoid → squeeze (B,1) → (B,)

    Args:
        deter_dim: Dimensionality of the RSSM deterministic state (default 256).
        num_categoricals: Number of categorical variables in the stochastic
            state (default 32; MVP configuration).
        num_classes: Number of classes per categorical variable (default 32;
            MVP configuration).

    Example::

        head = DiscountPredictor()
        state = LatentState(deter=torch.zeros(4, 256), stoch=torch.zeros(4, 32, 32))
        gamma = head(state)   # shape (4,), all values in [0.0, 1.0]
    """

    def __init__(
        self,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        stoch_features = num_categoricals * num_classes   # 1024 for MVP
        in_features = stoch_features + deter_dim          # 1280 for MVP

        self.net = nn.Sequential(
            nn.Linear(in_features, _HIDDEN_DIM),
            nn.ELU(inplace=True),
            nn.Linear(_HIDDEN_DIM, 1),
            nn.Sigmoid(),                                 # guarantees [0.0, 1.0] -- Req 6.2
        )

    def forward(self, state: LatentState) -> Tensor:
        """Predict a continuation probability per latent state (Req 6.2).

        Args:
            state: A :class:`~agent.types.LatentState` with ``deter`` of shape
                ``(B, deter_dim)`` and ``stoch`` of shape
                ``(B, num_categoricals, num_classes)``.

        Returns:
            A 1-D tensor of shape ``(B,)`` whose values lie in ``[0.0, 1.0]``,
            representing the predicted episode-continuation (discount) probability
            for each element of the batch. The Sigmoid activation applied inside
            the network guarantees this range (Req 6.2).
        """
        features = _build_feature(state)    # (B, 1280)
        out = self.net(features)            # (B, 1)
        return out.squeeze(-1)              # (B,)  -- Req 6.2
