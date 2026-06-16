"""Convolutional Encoder mapping a Frame to a discrete categorical latent.

The :class:`Encoder` is the perception front end of Odyssey's world model. It
maps a preprocessed Frame -- shape ``(B, channels, 64, 64)`` with pixels in
``[0.0, 1.0]`` -- into a DreamerV2-style discrete categorical latent: 32
categorical variables, each a distribution over 32 classes, represented as
logits of shape ``(B, 32, 32)`` (Req 3.1).

Design references:
    - DreamerV2 (Hafner et al., 2020): the 32x32 discrete categorical latent and
      the straight-through gradient estimator used to backpropagate through the
      one-hot sample to the encoder logits (Req 3.4).

Scope awareness:
    The encoder is constructed with the active scope's channel count (1 for the
    MVP grayscale scope, 3 for the RGB scope) and the fixed 64x64 input
    resolution. ``forward`` validates every input against this contract and
    raises :class:`FrameShapeError` on any mismatch, producing no latent
    (Req 3.2, 3.3, 3.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from agent.errors import FrameShapeError
from agent.types import CategoricalLatent

# Re-exported so existing ``from agent.encoder import FrameShapeError`` imports
# keep working now that the shared taxonomy lives in ``agent/errors.py``.
__all__ = ["Encoder", "FrameShapeError", "FRAME_RESOLUTION"]

# Fixed input spatial resolution for every scope (Req 3.2, 3.3).
FRAME_RESOLUTION = 64


class Encoder(nn.Module):
    """Convolutional encoder producing a 32x32 categorical latent (Req 3.1).

    A standard DreamerV2-style convolutional stack downsamples a 64x64 frame to
    a 4x4 feature map over four stride-2 convolutions
    (64 -> 32 -> 16 -> 8 -> 4 spatial), which is flattened and projected by a
    linear layer to ``num_categoricals * num_classes`` logits and reshaped to
    ``(B, num_categoricals, num_classes)`` == ``(B, 32, 32)``.

    The exact channel widths below are design discretion (the requirements defer
    hyperparameters); they form a modest, conventional encoder for 64x64 input.

    Args:
        channels: Active-scope input channel count (1 for grayscale MVP, 3 for
            RGB). Validated against every input frame (Req 3.2, 3.3, 3.5).
        num_categoricals: Number of categorical variables in the latent (32).
        num_classes: Number of classes per categorical variable (32).
    """

    def __init__(
        self,
        channels: int,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_categoricals = int(num_categoricals)
        self.num_classes = int(num_classes)

        # Four stride-2 convolutions: 64 -> 32 -> 16 -> 8 -> 4 spatial.
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # 256 channels x 4 x 4 spatial after the conv stack.
        self._conv_out_features = 256 * 4 * 4
        self.to_logits = nn.Linear(
            self._conv_out_features, self.num_categoricals * self.num_classes
        )

    def _validate_frame(self, frame: Tensor) -> None:
        """Validate the input frame against the active-scope contract (Req 3.5).

        Raises:
            FrameShapeError: If the frame is not 4-D ``(B, channels, 64, 64)``,
                its channel count does not match the active scope, or its
                spatial resolution is not 64x64. No latent is produced.
        """
        if frame.dim() != 4:
            raise FrameShapeError(
                f"Expected a 4-D frame (B, {self.channels}, "
                f"{FRAME_RESOLUTION}, {FRAME_RESOLUTION}), got {frame.dim()} dims "
                f"with shape {tuple(frame.shape)}."
            )
        _, c, h, w = frame.shape
        if c != self.channels:
            raise FrameShapeError(
                f"Frame channel count {c} does not match active scope "
                f"channels {self.channels}."
            )
        if h != FRAME_RESOLUTION or w != FRAME_RESOLUTION:
            raise FrameShapeError(
                f"Frame resolution {h}x{w} does not match expected "
                f"{FRAME_RESOLUTION}x{FRAME_RESOLUTION}."
            )

    def forward(self, frame: Tensor) -> CategoricalLatent:
        """Encode a frame into a categorical latent (Req 3.1, 3.2, 3.3, 3.5).

        Args:
            frame: Input frame of shape ``(B, channels, 64, 64)`` with pixels in
                ``[0.0, 1.0]``, where ``channels`` matches the active scope.

        Returns:
            A :class:`CategoricalLatent` whose ``logits`` have shape
            ``(B, 32, 32)`` and whose ``sample`` is the straight-through one-hot
            sample of the same shape.

        Raises:
            FrameShapeError: If the frame's resolution or channel count does not
                match the active scope; no latent is produced (Req 3.5).
        """
        self._validate_frame(frame)

        batch = frame.shape[0]
        features = self.conv(frame)
        features = features.reshape(batch, -1)
        logits = self.to_logits(features)
        logits = logits.reshape(batch, self.num_categoricals, self.num_classes)

        sample = self.sample(logits)
        return CategoricalLatent(logits=logits, sample=sample)

    def sample(self, logits: Tensor) -> Tensor:
        """Draw a one-hot latent via the straight-through estimator (Req 3.4).

        For each of the ``num_categoricals`` variables, a class is sampled from
        the categorical distribution defined by the logits over the last
        (``num_classes``) dimension, yielding a hard one-hot vector. To keep the
        operation differentiable with respect to the encoder logits, the
        straight-through estimator is applied::

            probs   = softmax(logits, dim=-1)
            onehot  = one_hot(sample_index)            # hard, no gradient
            sample  = onehot + (probs - probs.detach())

        The straight-through term ``probs - probs.detach()`` is grouped
        deliberately: ``probs`` and ``probs.detach()`` hold identical values, so
        their elementwise difference is *exactly* ``0.0``, and ``onehot + 0.0``
        therefore returns the one-hot sample bit-for-bit. (Computing
        ``onehot + probs - probs.detach()`` left-to-right would instead form the
        intermediate ``onehot + probs`` and lose floating-point precision on the
        ``1.0`` entries.) Thus the returned tensor is one-hot per variable
        (exactly one ``1.0``, the rest ``0.0``). On the backward pass the
        gradient flows through ``probs`` -- and hence into ``logits`` and the
        encoder -- because the ``probs`` term is not detached.

        Args:
            logits: Class logits of shape ``(B, num_categoricals, num_classes)``.

        Returns:
            A one-hot sample of shape ``(B, num_categoricals, num_classes)`` that
            is differentiable with respect to ``logits``.
        """
        probs = F.softmax(logits, dim=-1)

        # Hard categorical sample over the class dimension -> one-hot. Drawn from
        # the categorical distribution; carries no gradient.
        dist = torch.distributions.OneHotCategorical(probs=probs)
        onehot = dist.sample()

        # Straight-through estimator: forward value is the one-hot sample,
        # gradients route to the logits via ``probs``. Grouping
        # ``(probs - probs.detach())`` keeps that term exactly 0.0, so the
        # forward value is the one-hot sample bit-for-bit (exactly 0/1).
        return onehot + (probs - probs.detach())
