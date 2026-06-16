"""Typed exceptions for Odyssey -- the single shared error taxonomy.

This module is the cohesive, authoritative home for the typed exceptions that
Odyssey's components raise (see the design's Error Handling section). Every
error subclasses :class:`OdysseyError`, allowing callers to catch the whole
family with one ``except`` clause while still discriminating on the specific
subclass. All other components import their errors from here so the taxonomy
stays consistent across perception, replay, encoding, world-model training,
behavior learning, gating, the PPO baseline, comparison, and reporting.

Odyssey's error discipline: errors are recorded and contained, never silently
swallowed, and the failure of one unit of work never corrupts already-valid
state.

The taxonomy:

    * :class:`OdysseyError` -- base class for the whole family.
    * :class:`ObservationValidationError` -- raw observation fails validation
      (Req 1.8).
    * :class:`FrameConversionError` -- a validated observation cannot be turned
      into a valid frame for the active scope (Req 1.9, 9.6).
    * :class:`BufferStorageError` -- a transition cannot be stored in the replay
      buffer (Req 2.3).
    * :class:`InsufficientDataError` -- a sample/training request cannot be
      satisfied for lack of data (Req 2.10, 9.7, 7.8).
    * :class:`FrameShapeError` -- an encoder input frame's shape or channel count
      mismatches the active scope (Req 3.5).
    * :class:`InvalidActionError` -- an action's dimensionality mismatches the
      RSSM's expected action space (Req 4.6).
    * :class:`ReconstructionShapeMismatch` -- a reconstruction does not match the
      target frame in the reconstruction-loss path (Req 5.5).
    * :class:`NonFiniteLossError` -- the world-model trainer computes a
      non-finite loss (Req 7.9).
    * :class:`GateNotConfirmedError` -- the reconstruction gate has not been
      confirmed when behavior learning is attempted (Req 8.5, 18.5).
    * :class:`PPOTrainingError` -- the PPO baseline fails during training
      (Req 13.5).
    * :class:`ComparisonNotReadyError` -- a comparison is requested before both
      agents have been trained (Req 14.7).
    * :class:`ReportInputMissingError` -- the report generator is missing a
      required input (Req 16.7).
"""

from __future__ import annotations


class OdysseyError(Exception):
    """Base class for all typed Odyssey errors."""


class ObservationValidationError(OdysseyError):
    """Raised when a raw observation fails validation (Req 1.8).

    Triggered when an observation's shape is not ``(96, 96, 3)`` or it contains
    values outside the inclusive range ``[0, 255]``. When raised, the offending
    observation is rejected and no downstream processing (including replay-buffer
    storage) is performed.
    """


class FrameConversionError(OdysseyError):
    """Raised when a validated observation cannot be converted to a Frame (Req 1.9, 9.6).

    Triggered when, despite passing validation, an observation cannot be turned
    into a valid ``(channels, 64, 64)`` frame in ``[0.0, 1.0]`` for the active
    scope. When raised, processing of that observation halts and previously
    stored frames are left unchanged.
    """


class BufferStorageError(OdysseyError):
    """Raised when storing a transition in the replay buffer fails (Req 2.3).

    Triggered when an object cannot be stored as a valid
    :class:`~agent.types.Transition`. When raised, the existing transitions in
    the buffer are left unchanged.
    """


class InsufficientDataError(OdysseyError):
    """Raised when a sample or training request cannot be satisfied (Req 2.10, 9.7, 7.8).

    Triggered when the replay buffer cannot satisfy a sample request -- for
    example, it holds fewer transitions than the requested batch size or the
    requested sequence length exceeds the longest contiguous within-episode run
    -- or when the world-model trainer is asked to train without enough data.
    When raised, the buffer is left unchanged.
    """


class FrameShapeError(OdysseyError):
    """Raised when an encoder input frame mismatches the active scope (Req 3.5).

    Triggered by ``Encoder.forward`` when the input frame is not the expected
    ``(B, channels, 64, 64)`` for the active scope (wrong number of dimensions,
    channel count, or spatial resolution). When raised, the encoder produces no
    latent and performs no further processing of the offending frame.
    """


class InvalidActionError(OdysseyError):
    """Raised when an action's dimensionality mismatches the RSSM (Req 4.6).

    Triggered by ``RSSM.predict_prior`` when the provided action does not match
    the expected action dimensionality. When raised, no prior is produced.
    """


class ReconstructionShapeMismatch(OdysseyError):
    """Raised when a reconstruction does not match its target frame (Req 5.5).

    Triggered in the reconstruction-loss path when the decoder's reconstruction
    shape does not match the target frame's shape, so a reconstruction loss
    cannot be computed.
    """


class NonFiniteLossError(OdysseyError):
    """Raised when the world-model trainer computes a non-finite loss (Req 7.9).

    Triggered when a training step produces a loss that is ``NaN`` or infinite.
    When raised, the offending update is not applied so already-valid model
    parameters are not corrupted.
    """


class GateNotConfirmedError(OdysseyError):
    """Raised when the reconstruction gate has not been confirmed (Req 8.5, 18.5).

    Triggered when behavior learning (imagination/actor-critic training) is
    attempted before the world model's reconstruction gate has been confirmed.
    """


class PPOTrainingError(OdysseyError):
    """Raised when the PPO baseline fails during training (Req 13.5).

    Triggered when the model-free PPO baseline encounters a failure in its
    training loop.
    """


class ComparisonNotReadyError(OdysseyError):
    """Raised when a comparison is requested before both agents are trained (Req 14.7).

    Triggered when a head-to-head comparison between the world-model agent and
    the PPO baseline is requested before both agents have completed training.
    """


class ReportInputMissingError(OdysseyError):
    """Raised when the report generator is missing a required input (Req 16.7).

    Triggered when report generation is requested but one or more required
    inputs (metrics, artifacts, or comparison results) are absent.
    """
