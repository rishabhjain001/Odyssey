"""Random trajectory collection for Odyssey (Req 2.1, 2.8, 2.9).

This module implements the data-collection phase that seeds the replay buffer
with real experience before any world-model training. It drives the
``CarRacing`` environment with actions sampled *uniformly over the full range of
the action space* (Req 2.1), validates every observation through the
:class:`~agent.perception.FramePreprocessor`, stores validated transitions in
the :class:`~agent.replay.ReplayBuffer` (Req 2.2 storage is the buffer's job),
resets the environment whenever an episode ends (Req 2.8), and stops exactly
once the configured number of environment steps has been collected (Req 2.9).

Design choices
--------------

* **Injectable environment.** The collector accepts an already-constructed
  environment object rather than building one internally. The only surface it
  relies on is the Gymnasium API: ``reset(seed=...) -> (obs, info)``,
  ``step(action) -> (obs, reward, terminated, truncated, info)``, and
  ``action_space.sample()`` / ``action_space.seed(seed)``. This keeps the
  collector unit-testable with a tiny fake env (no real Gymnasium / Box2D
  required) and lets the orchestrator inject a configured ``CarRacing-v2``.

* **Uniform full-range actions (Req 2.1).** ``env.action_space.sample()`` draws
  uniformly across the entire action space, which is the cleanest expression of
  "uniform over the full range" for CarRacing's continuous ``Box`` space. The
  action space can be seeded for reproducibility.

* **Counting semantics.** The collector counts *environment steps*: each
  ``env.step`` call is one step, and collection stops the instant the configured
  ``env_steps`` count is reached (Req 2.9). A real env step always counts toward
  this budget, even when its observation is later rejected -- the step has
  already been consumed by the environment. Rejected observations are simply not
  stored (no transition is added), but they do not earn the agent extra steps.

* **Episode-boundary marker.** Each stored :class:`~agent.types.Transition`
  records ``terminated = terminated or truncated``. The replay buffer uses this
  flag purely to mark episode boundaries so that sampled sequences never span a
  reset (Req 2.5); both natural termination and truncation end an episode and
  trigger a reset (Req 2.8), so both must close the run.

* **Error containment.** A rejected observation
  (:class:`~agent.errors.ObservationValidationError`) or a failed store
  (:class:`~agent.errors.BufferStorageError`) is recorded on the returned
  :class:`CollectionResult` and skipped; it never aborts the whole collection
  nor corrupts already-stored transitions, matching Odyssey's error discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from agent.config import CollectionConfig, ScopeConfig
from agent.errors import BufferStorageError, ObservationValidationError
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from agent.types import Transition

# Environment-step budget bounds defined by the requirements (Req 2.9).
MIN_ENV_STEPS = 1
MAX_ENV_STEPS = 100_000_000


@dataclass
class CollectionResult:
    """Outcome of a random trajectory collection run.

    Attributes:
        env_steps: Number of environment steps actually executed. On a normal
            run this equals ``config.env_steps`` (Req 2.9).
        stored: Number of validated transitions added to the replay buffer.
        rejected_observations: Number of environment steps whose observation
            failed validation and was therefore not stored (Req 1.8).
        storage_failures: Number of transitions that could not be stored because
            the buffer rejected them (Req 2.3).
        episodes: Number of episode resets performed after a termination or
            truncation during collection (Req 2.8). The initial reset is not
            counted here.
    """

    env_steps: int = 0
    stored: int = 0
    rejected_observations: int = 0
    storage_failures: int = 0
    episodes: int = 0
    errors: List[str] = field(default_factory=list)


def _unpack_reset(result: Any) -> np.ndarray:
    """Return the observation from a Gymnasium ``reset`` result.

    Gymnasium's ``reset`` returns ``(obs, info)``; older single-value APIs return
    just ``obs``. Both are accepted so the collector works with a real env and a
    minimal fake.
    """
    if isinstance(result, tuple):
        return result[0]
    return result


def _unpack_step(result: Any) -> tuple[np.ndarray, float, bool, bool]:
    """Return ``(obs, reward, terminated, truncated)`` from a ``step`` result.

    Handles the Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
    and the legacy 4-tuple ``(obs, reward, done, info)`` (mapping ``done`` to
    ``terminated``).
    """
    if not isinstance(result, tuple):
        raise TypeError(
            f"env.step must return a tuple, got {type(result).__name__}."
        )
    if len(result) >= 5:
        obs, reward, terminated, truncated = result[0], result[1], result[2], result[3]
    elif len(result) == 4:
        obs, reward, done, _info = result
        terminated, truncated = bool(done), False
    else:
        raise ValueError(
            f"env.step returned a {len(result)}-tuple; expected the Gymnasium "
            "5-tuple (obs, reward, terminated, truncated, info)."
        )
    return obs, float(reward), bool(terminated), bool(truncated)


def _sample_action(env: Any) -> np.ndarray:
    """Sample an action uniformly over the full action space (Req 2.1)."""
    action = env.action_space.sample()
    return np.asarray(action, dtype=np.float32)


def collect_random_trajectories(
    env: Any,
    buffer: ReplayBuffer,
    config: CollectionConfig,
    *,
    preprocessor: Optional[FramePreprocessor] = None,
    seed: Optional[int] = None,
) -> CollectionResult:
    """Collect random trajectories into ``buffer`` (Req 2.1, 2.8, 2.9).

    Drives ``env`` with uniformly sampled actions, validates each observation,
    stores validated transitions, resets on episode end, and stops once exactly
    ``config.env_steps`` environment steps have been executed.

    Args:
        env: An already-constructed environment exposing the Gymnasium API
            (``reset``, ``step``, ``action_space.sample``). Injected so the
            collector is testable without a real Gymnasium environment.
        buffer: The replay buffer that validated transitions are added to.
        config: Collection configuration. ``config.env_steps`` (in
            ``[1, 100_000_000]``, Req 2.9) sets the stop condition.
        preprocessor: Optional :class:`FramePreprocessor` used to validate
            observations before storage. When omitted, a grayscale-scope
            preprocessor is constructed; validation is scope-independent, so the
            default is sufficient for observation validation.
        seed: Optional seed applied to the initial ``env.reset`` and to
            ``env.action_space`` for reproducible action sampling.

    Returns:
        A :class:`CollectionResult` summarizing the run: environment steps
        executed, transitions stored, rejected observations, storage failures,
        and episode resets.

    Raises:
        ValueError: If ``config.env_steps`` is outside ``[1, 100_000_000]``.
    """
    target_steps = int(config.env_steps)
    # Defensive guard mirroring CollectionConfig's documented bounds (Req 2.9).
    if not MIN_ENV_STEPS <= target_steps <= MAX_ENV_STEPS:
        raise ValueError(
            f"config.env_steps must be in [{MIN_ENV_STEPS}, {MAX_ENV_STEPS}], "
            f"got {target_steps}."
        )

    if preprocessor is None:
        # validate_observation is scope-independent (it checks the raw
        # (96, 96, 3) obs), so a grayscale-scope preprocessor suffices.
        preprocessor = FramePreprocessor(ScopeConfig(channels=1, sequence_length=50))

    # Seed action sampling for reproducibility when requested (Req 2.1).
    if seed is not None:
        action_space = getattr(env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(seed)

    result = CollectionResult()

    # Initial reset to obtain the first observation. The initial reset is not an
    # episode-boundary reset, so it is not counted in result.episodes.
    if seed is not None:
        _unpack_reset(env.reset(seed=seed))
    else:
        _unpack_reset(env.reset())

    while result.env_steps < target_steps:
        action = _sample_action(env)
        obs, reward, terminated, truncated = _unpack_step(env.step(action))

        # The environment step has been consumed; it always counts (Req 2.9).
        result.env_steps += 1

        # Validate before storing: a malformed observation is never stored
        # (Req 1.8) but the step it consumed still counts.
        try:
            preprocessor.validate_observation(obs)
        except ObservationValidationError as exc:
            result.rejected_observations += 1
            result.errors.append(f"step {result.env_steps}: rejected observation: {exc}")
        else:
            # Store the raw validated observation per the Transition contract.
            # The boundary marker is `terminated or truncated` so the replay
            # buffer never samples a sequence across a reset (Req 2.5, 2.8).
            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                terminated=bool(terminated or truncated),
            )
            try:
                buffer.add(transition)
                result.stored += 1
            except BufferStorageError as exc:
                result.storage_failures += 1
                result.errors.append(f"step {result.env_steps}: storage failed: {exc}")

        # Reset on episode end and continue toward the step budget (Req 2.8).
        if (terminated or truncated) and result.env_steps < target_steps:
            _unpack_reset(env.reset())
            result.episodes += 1

    return result
