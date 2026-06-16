"""Frame preprocessing and observation validation for Odyssey.

This module implements :class:`FramePreprocessor`, the pure transformation layer
that converts raw CarRacing-v2 observations into model-ready frames, validates
incoming observations, and maps Actor outputs onto the environment action space.

Responsibilities (Requirement 1, Requirement 9):

    * ``validate_observation`` -- reject observations whose shape is not
      ``(96, 96, 3)`` or that contain values outside ``[0, 255]`` so that a
      malformed observation never reaches downstream processing or storage
      (Req 1.8).
    * ``to_frame`` -- resize to 64x64, apply the scope channel policy (MVP
      averages RGB into one grayscale channel; RGB keeps three channels), scale
      ``value / 255`` into ``[0.0, 1.0]``, and return a ``float32`` array of
      shape ``(channels, 64, 64)`` (Req 1.3, 1.4, 1.5, 1.6, 9.2). On failure it
      raises :class:`FrameConversionError`, leaving prior frames unchanged
      (Req 1.9, 9.6).
    * ``map_action`` -- map an Actor output ``[steering, accel, brake]`` onto the
      continuous CarRacing-v2 action space (Req 1.7).

Statelessness and the "prior frames unchanged" guarantee: the preprocessor holds
no stored frames -- frame/transition storage is the replay buffer's
responsibility. Because conversion validates before doing any work and never
mutates shared state, a failed conversion inherently leaves previously stored
frames and transitions unchanged (Req 1.9, 9.6).

Resizing approach: resizing uses :func:`torch.nn.functional.interpolate` with
bilinear interpolation. ``torch`` is already a core dependency (it powers the
world and behavior models), so this adds no new dependency and gives a correct,
well-tested resize. The observation is moved to a float tensor, resized, then
converted back to a NumPy ``float32`` array; no gradient tracking is involved.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agent.config import ScopeConfig
from agent.errors import FrameConversionError, ObservationValidationError

# The contract for a raw CarRacing-v2 observation (Req 1.2).
_RAW_OBS_SHAPE = (96, 96, 3)
# Target spatial resolution for every produced frame (Req 1.3).
_FRAME_SIZE = 64
# Source pixel value bounds (Req 1.2, 1.6).
_PIXEL_MIN = 0
_PIXEL_MAX = 255

# CarRacing-v2 continuous action bounds: [steering, acceleration, brake] with
# steering in [-1, 1] and acceleration/brake in [0, 1] (Req 1.7, 10.3).
_ACTION_LOW = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class FramePreprocessor:
    """Convert raw observations into scope-correct, normalized frames.

    The preprocessor is configured with a :class:`~agent.config.ScopeConfig`
    that fixes the active channel policy: ``channels == 1`` averages the RGB
    channels into a single grayscale channel (the MVP scope), while
    ``channels == 3`` keeps all three color channels (the RGB scope). Both
    scopes resize to exactly 64x64 and scale pixels into ``[0.0, 1.0]``
    (Req 1.3, 1.4, 1.5, 1.6, 9.2).

    The preprocessor is stateless with respect to stored frames; it performs a
    pure transformation per observation (Req 1.9, 9.6).
    """

    def __init__(self, scope: ScopeConfig) -> None:
        """Initialize the preprocessor for an active scope.

        Args:
            scope: The active scope configuration. ``scope.channels`` must be 1
                (grayscale MVP) or 3 (RGB).

        Raises:
            ValueError: If ``scope.channels`` is neither 1 nor 3.
        """
        if scope.channels not in (1, 3):
            raise ValueError(
                f"ScopeConfig.channels must be 1 (grayscale) or 3 (RGB), "
                f"got {scope.channels}."
            )
        self.scope = scope

    def validate_observation(self, obs: np.ndarray) -> None:
        """Validate a raw observation, raising on any contract violation (Req 1.8).

        Rejects an observation whose shape is not ``(96, 96, 3)`` or that
        contains any value outside the inclusive range ``[0, 255]``. Validation
        runs before any processing so a malformed observation never produces a
        frame nor reaches the replay buffer.

        Args:
            obs: The raw observation to validate.

        Raises:
            ObservationValidationError: If ``obs`` is not a NumPy array, has the
                wrong shape, or contains values outside ``[0, 255]``.
        """
        if not isinstance(obs, np.ndarray):
            raise ObservationValidationError(
                f"Observation must be a numpy.ndarray, got {type(obs).__name__}."
            )
        if obs.shape != _RAW_OBS_SHAPE:
            raise ObservationValidationError(
                f"Observation shape must be {_RAW_OBS_SHAPE}, got {obs.shape}."
            )
        # NaN/Inf are not within [0, 255]; np.min/np.max would propagate NaN, so
        # check finiteness explicitly first for a clear error message.
        if not np.all(np.isfinite(obs)):
            raise ObservationValidationError(
                "Observation contains non-finite values; expected integers in [0, 255]."
            )
        obs_min = float(np.min(obs))
        obs_max = float(np.max(obs))
        if obs_min < _PIXEL_MIN or obs_max > _PIXEL_MAX:
            raise ObservationValidationError(
                f"Observation values must lie in [{_PIXEL_MIN}, {_PIXEL_MAX}], "
                f"got range [{obs_min}, {obs_max}]."
            )

    def to_frame(self, obs: np.ndarray) -> np.ndarray:
        """Convert a raw observation into a model-ready frame (Req 1.3-1.6, 9.2).

        Validates the observation (Req 1.8), resizes it to 64x64, applies the
        active scope's channel policy, and scales pixels into ``[0.0, 1.0]``.
        The result is a ``float32`` array of shape ``(channels, 64, 64)`` where
        ``channels`` matches the active scope.

        Because validation runs first and the method mutates no shared state, a
        failure halts processing of this observation while leaving previously
        stored frames unchanged (Req 1.9, 9.6).

        Args:
            obs: The raw ``(96, 96, 3)`` observation with integer-valued pixels
                in ``[0, 255]``.

        Returns:
            A ``float32`` frame of shape ``(channels, 64, 64)`` with values in
            ``[0.0, 1.0]``.

        Raises:
            ObservationValidationError: If the observation fails validation
                (Req 1.8).
            FrameConversionError: If a validated observation cannot be converted
                into a valid frame for the active scope (Req 1.9, 9.6).
        """
        # Validate before any processing so malformed input is never converted
        # and never reaches downstream storage (Req 1.8).
        self.validate_observation(obs)

        try:
            # (H, W, C) uint8/int -> (1, C, H, W) float tensor for interpolate.
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(obs_chw)).unsqueeze(0)

            # Resize to exactly 64x64 with bilinear interpolation (Req 1.3).
            resized = F.interpolate(
                tensor,
                size=(_FRAME_SIZE, _FRAME_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # -> (C, 64, 64)

            # Apply the scope channel policy (Req 1.4, 1.5).
            if self.scope.channels == 1:
                # MVP: average the RGB channels into a single grayscale channel.
                frame = resized.mean(dim=0, keepdim=True)  # (1, 64, 64)
            else:
                # RGB: keep all three channels.
                frame = resized  # (3, 64, 64)

            # Scale from [0, 255] into [0.0, 1.0] (Req 1.6) and clamp to absorb
            # any interpolation overshoot so the output respects its contract.
            frame = (frame / float(_PIXEL_MAX)).clamp(0.0, 1.0)

            result = frame.detach().to(torch.float32).contiguous().numpy()
        except ObservationValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - record and contain (Req 1.9, 9.6)
            raise FrameConversionError(
                f"Failed to convert observation into a "
                f"({self.scope.channels}, {_FRAME_SIZE}, {_FRAME_SIZE}) frame: {exc}"
            ) from exc

        expected_shape = (self.scope.channels, _FRAME_SIZE, _FRAME_SIZE)
        if result.shape != expected_shape:
            raise FrameConversionError(
                f"Converted frame shape {result.shape} does not match the "
                f"active scope's expected shape {expected_shape}."
            )
        return result

    def map_action(self, actor_output: np.ndarray) -> np.ndarray:
        """Map an Actor output onto the CarRacing-v2 action space (Req 1.7).

        The CarRacing-v2 action space is continuous
        ``[steering in [-1, 1], acceleration in [0, 1], brake in [0, 1]]``. The
        Actor is designed to emit values already within these bounds (tanh
        steering, sigmoid accel/brake); this method enforces the contract by
        clipping to the action bounds and returning a contiguous ``float32``
        array of length 3 suitable for ``env.step``.

        Args:
            actor_output: The raw Actor output ``[steering, accel, brake]``.

        Returns:
            A ``float32`` action array of shape ``(3,)`` within the env action
            space.

        Raises:
            ValueError: If ``actor_output`` does not have exactly 3 elements.
        """
        action = np.asarray(actor_output, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(
                f"Actor output must have 3 elements [steering, accel, brake], "
                f"got shape {np.asarray(actor_output).shape}."
            )
        return np.clip(action, _ACTION_LOW, _ACTION_HIGH).astype(np.float32)
