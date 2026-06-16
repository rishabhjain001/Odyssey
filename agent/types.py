"""Shared data types for Odyssey.

This module defines the core dataclasses and type contracts that flow between
Odyssey's components (perception, replay, world model, behavior model,
evaluation, and metrics). They are intentionally lightweight and free of
behavior so they can be imported anywhere without pulling in heavy
dependencies at definition time.

The neural types (``CategoricalLatent``, ``LatentState``, ``ImaginedRollout``,
``WorldModelLosses``) annotate their tensor fields with :class:`torch.Tensor`.
``from __future__ import annotations`` keeps those annotations as strings, so
this module imports cleanly even in environments where ``torch`` is not
installed; ``torch`` is only referenced under ``TYPE_CHECKING``.

References:
    - DreamerV2 (Hafner et al., 2020): 32x32 discrete categorical latent with a
      straight-through gradient estimator.
    - RSSM / PlaNet (Hafner et al., 2018): split of the latent into a
      deterministic recurrent component and a stochastic categorical component.
    - Dreamer (Hafner et al., 2019): lambda-returns over imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    import torch


# ---------------------------------------------------------------------------
# Frame contract
# ---------------------------------------------------------------------------

#: A ``Frame`` is a model-ready observation produced by the
#: ``FramePreprocessor``. Contract (Req 1.3, 1.4, 1.5, 1.6, 9.2, 9.3):
#:
#:   * dtype ``np.float32``
#:   * shape ``(channels, 64, 64)`` where ``channels`` is 1 for the MVP
#:     grayscale scope and 3 for the RGB scope
#:   * every pixel value lies in the closed interval ``[0.0, 1.0]``
#:
#: ``Frame`` is a :func:`typing.NewType` alias over ``np.ndarray`` rather than a
#: runtime class: it documents the contract and aids static checking without
#: imposing a wrapper object or runtime validation cost. Validation of the
#: contract is performed by the preprocessor, not by this alias.
Frame = NewType("Frame", np.ndarray)


# ---------------------------------------------------------------------------
# Replay / environment transition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    """A single stored interaction with the CarRacing environment (Req 2.2).

    Attributes:
        observation: The observation array. This is typically the validated raw
            ``(96, 96, 3)`` observation or its preprocessed ``Frame``.
        action: The continuous control action ``[steering, accel, brake]``.
        reward: The scalar reward received for the transition.
        terminated: The episode-termination signal for the transition.
    """

    observation: np.ndarray
    action: np.ndarray
    reward: float
    terminated: bool


# ---------------------------------------------------------------------------
# Discrete latent representation (DreamerV2-style)
# ---------------------------------------------------------------------------

@dataclass
class CategoricalLatent:
    """Discrete categorical latent produced by the Encoder (Req 3.1).

    Attributes:
        logits: Unnormalized class logits, shape ``(B, 32, 32)`` -- 32
            categorical variables, each over 32 classes.
        sample: One-hot categorical sample, shape ``(B, 32, 32)``; along the
            class dimension each of the 32 variables is one-hot.
    """

    logits: "torch.Tensor"
    sample: "torch.Tensor"


@dataclass
class LatentState:
    """Combined RSSM latent state (Req 4.1, 4.2).

    Attributes:
        deter: Deterministic GRU hidden state, shape ``(B, deter_dim)``.
        stoch: Stochastic categorical state, shape ``(B, 32, 32)`` one-hot.
    """

    deter: "torch.Tensor"
    stoch: "torch.Tensor"


@dataclass
class ImaginedRollout:
    """A rollout imagined entirely inside the latent space (Req 10.1, 10.2).

    Each field is indexed over the imagination horizon ``H``. Generating a
    rollout performs zero interactions with the CarRacing environment.

    Attributes:
        states: Predicted latent states over the horizon, length ``H``.
        actions: Actor actions taken at each imagined step, length ``H``.
        rewards: Predicted rewards along the rollout, length ``H``.
        discounts: Predicted continuation probabilities along the rollout,
            length ``H``.
    """

    states: list["LatentState"]
    actions: "torch.Tensor"
    rewards: "torch.Tensor"
    discounts: "torch.Tensor"


# ---------------------------------------------------------------------------
# Evaluation and training-loss aggregates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    """Aggregated outcome of a real-environment evaluation run (Req 11.2-11.5).

    Attributes:
        avg_reward: Arithmetic mean of total episode rewards across the run.
        completion_rate: Fraction of episodes that completed the track, in
            ``[0.0, 1.0]``.
        cumulative_env_steps: Cumulative environment steps consumed by Odyssey
            at the time of the evaluation run.
    """

    avg_reward: float
    completion_rate: float
    cumulative_env_steps: int


@dataclass(frozen=True)
class WorldModelLosses:
    """The component and total losses for one world-model step (Req 7.5).

    Attributes:
        recon: Reconstruction loss term.
        kl: KL divergence term between prior and posterior stochastic latents.
        reward: Reward-prediction loss term.
        discount: Episode-continuation (discount) prediction loss term.
        total: Combined objective value for the step.
    """

    recon: float
    kl: float
    reward: float
    discount: float
    total: float
