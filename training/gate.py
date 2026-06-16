"""Reconstruction gate for Odyssey.

This module implements the :class:`ReconstructionGate`, which enforces the
project's core staging discipline:

**MVP capabilities precede behavior capabilities (Req 8.1).**

Perception, encoding, reconstruction, and loss recording form the MVP layer.
Behavior training (imagination/actor-critic) and the PPO baseline are built on
top of a confirmed world model and must not start until the
``Reconstruction_Confirmed_Condition`` is met.  The gate is the single,
authoritative arbiter of that condition.

The gate is designed to be consulted centrally by the training orchestrator
(``train.py``, Req 18.5).  Only ``train.py`` calls :meth:`ReconstructionGate.behavior_allowed`;
behavior training and the PPO baseline should never bypass the gate.

Usage::

    from agent.config import GateConfig
    from training.gate import ReconstructionGate

    gate = ReconstructionGate(GateConfig(loss_threshold=0.02, consecutive_steps=100))

    # During the training loop, pass the full reconstruction-loss history:
    if gate.is_confirmed(recon_history):
        ...  # world model is stable

    allowed, err = gate.behavior_allowed(config.behavior.enabled, recon_history)
    if allowed:
        behavior_trainer.step()
    elif err:
        # err contains the GateNotConfirmedError message; already logged.
        ...
"""

from __future__ import annotations

import logging

from agent.config import GateConfig
from agent.errors import GateNotConfirmedError

logger = logging.getLogger(__name__)


class ReconstructionGate:
    """Gate that guards behavior/baseline activation behind a confirmed world model.

    The ``Reconstruction_Confirmed_Condition`` (Req 8.2, 8.3) is met when the
    last ``consecutive_steps`` recorded reconstruction losses are **all** at or
    below ``loss_threshold``.  Until that condition is met:

    * Behavior-model training (imagination / actor-critic) stays disabled
      (Req 8.2).
    * The PPO baseline stays disabled (Req 8.3).
    * Any attempt to enable behavior while the condition is not met is logged as
      a :class:`~agent.errors.GateNotConfirmedError` and the request is denied
      (Req 8.5).

    The gate must be consulted centrally by the training orchestrator so that no
    behavior component can inadvertently bypass it (Req 18.5).

    Args:
        config: A :class:`~agent.config.GateConfig` specifying ``loss_threshold``
            and ``consecutive_steps``.
    """

    def __init__(self, config: GateConfig) -> None:
        self._config = config

    @property
    def config(self) -> GateConfig:
        """The gate's configuration (read-only)."""
        return self._config

    def is_confirmed(self, recon_history: list[float]) -> bool:
        """Return ``True`` iff the reconstruction gate condition is met.

        The condition is met when:

        1. ``recon_history`` contains at least ``consecutive_steps`` entries, AND
        2. every value in ``recon_history[-consecutive_steps:]`` is ≤
           ``loss_threshold``.

        Returns ``False`` when the history is shorter than ``consecutive_steps``
        or when any value in the tail window exceeds the threshold.

        Args:
            recon_history: Ordered list of recorded reconstruction-loss values,
                oldest first.  May be empty.

        Returns:
            ``True`` if the gate condition is currently satisfied.
        """
        steps = self._config.consecutive_steps
        threshold = self._config.loss_threshold

        if len(recon_history) < steps:
            return False

        return all(v <= threshold for v in recon_history[-steps:])

    def behavior_allowed(
        self,
        config_enabled: bool,
        recon_history: list[float],
    ) -> tuple[bool, str | None]:
        """Determine whether behavior training is permitted at this point.

        The decision tree is:

        * ``config_enabled`` is ``False`` → behavior disabled by configuration;
          return ``(False, None)`` silently (Req 8.2, 8.3).
        * ``config_enabled`` is ``True`` AND gate :meth:`is_confirmed` → gate
          open; return ``(True, None)``.
        * ``config_enabled`` is ``True`` AND gate NOT confirmed → gate closed;
          record and log a :class:`~agent.errors.GateNotConfirmedError`, return
          ``(False, <error message>)`` (Req 8.5).

        This method never raises; it catches the error internally, logs it via
        the standard :mod:`logging` module, and returns the error description as
        a string so the caller can decide how to surface it.

        Args:
            config_enabled: Whether the behavior phase is enabled in the active
                :class:`~agent.config.BehaviorConfig`.
            recon_history: Ordered reconstruction-loss history (oldest first).

        Returns:
            A ``(allowed, error_msg)`` tuple.  ``error_msg`` is ``None`` when
            behavior is allowed or silently disabled; it contains the error
            description string when the gate is explicitly blocking an enabled
            request.
        """
        if not config_enabled:
            # Behavior disabled by configuration — silent, no error.
            return (False, None)

        if self.is_confirmed(recon_history):
            # Gate open: world model is confirmed, behavior may proceed.
            return (True, None)

        # Gate closed: config wants behavior but the condition is not met.
        error = GateNotConfirmedError(
            f"Behavior training requested but the reconstruction gate is not confirmed. "
            f"Requires the last {self._config.consecutive_steps} losses to all be "
            f"<= {self._config.loss_threshold} (current history length: {len(recon_history)})."
        )
        error_msg = str(error)
        logger.warning("GateNotConfirmedError: %s", error_msg)
        return (False, error_msg)
