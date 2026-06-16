"""Convolutional Decoder reconstructing a Frame from a LatentState.

The :class:`Decoder` is the generative component of Odyssey's world model. It
maps a :class:`~agent.types.LatentState` -- the combined deterministic +
stochastic RSSM state -- back to a predicted Frame of shape
``(B, channels, 64, 64)`` with all pixels in ``[0.0, 1.0]`` (Req 5.1, 9.3).

Design references:
    - DreamerV2 (Hafner et al., 2020): the decoder is the symmetric counterpart
      of the convolutional encoder, using transposed convolutions to upsample
      from a 4x4 spatial seed back to the 64x64 target resolution.

Architecture overview::

    LatentState.deter  (B, 256)
    LatentState.stoch  (B, 32, 32)  →  flatten  →  (B, 1024)
                                  cat  →  (B, 1280)
                                  Linear  →  (B, 4096)
                                  reshape  →  (B, 256, 4, 4)
                                  ConvTranspose2d(256→128, 4×4, stride=2, pad=1)  →  (B, 128, 8, 8)
                                  ConvTranspose2d(128→64,  4×4, stride=2, pad=1)  →  (B,  64, 16, 16)
                                  ConvTranspose2d( 64→32,  4×4, stride=2, pad=1)  →  (B,  32, 32, 32)
                                  ConvTranspose2d( 32→C,   4×4, stride=2, pad=1)  →  (B,   C, 64, 64)
                                  Sigmoid                                          →  pixels ∈ [0.0, 1.0]

Each ConvTranspose2d uses ``kernel_size=4, stride=2, padding=1``, which exactly
reverses the encoder's stride-2 ``Conv2d`` stack (4→8→16→32→64 spatial, mirroring
the encoder's 64→32→16→8→4 downsampling path).

Scope awareness:
    The decoder is constructed with the active scope's channel count (1 for the
    MVP grayscale scope, 3 for the RGB scope) so the output frame matches the
    expected shape without any branching at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from agent.types import LatentState

__all__ = ["Decoder"]


class Decoder(nn.Module):
    """Transposed-convolutional decoder producing a reconstructed Frame (Req 5.1, 9.3).

    The decoder accepts the combined RSSM latent state (deterministic GRU
    hidden state concatenated with the flattened stochastic categorical state)
    and reconstructs the corresponding frame through a linear projection
    followed by a symmetric transposed-conv upsample stack.

    The final :class:`~torch.nn.Sigmoid` activation guarantees every output
    pixel lies in ``[0.0, 1.0]`` without any explicit clamp (Req 5.1, 9.3).

    Args:
        channels: Active-scope output channel count (1 for grayscale MVP, 3 for
            RGB). Controls the channel count of the final transposed conv layer
            so the output matches the active scope's frame contract.
        deter_dim: Dimensionality of the deterministic GRU hidden state
            (``LatentState.deter``). Default: 256.
        num_categoricals: Number of categorical variables in the stochastic
            state (``LatentState.stoch``). Default: 32.
        num_classes: Number of classes per categorical variable. Default: 32.

    Input feature width:
        ``deter_dim + num_categoricals * num_classes``
        = ``256 + 32 * 32``
        = ``256 + 1024``
        = ``1280`` by default.
    """

    def __init__(
        self,
        channels: int,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.deter_dim = int(deter_dim)
        self.num_categoricals = int(num_categoricals)
        self.num_classes = int(num_classes)

        # Combined latent feature width: deter + flattened stoch.
        # Default: 256 + 32*32 = 1280.
        in_features = self.deter_dim + self.num_categoricals * self.num_classes

        # Project from combined latent → spatial seed (256 channels, 4×4 spatial).
        # 256 * 4 * 4 = 4096.
        self.project = nn.Linear(in_features, 256 * 4 * 4)

        # Transposed-conv upsample stack -- symmetric mirror of the encoder's
        # four stride-2 Conv2d layers.  Each layer doubles the spatial size:
        #   4 → 8 → 16 → 32 → 64
        # Channel widths mirror the encoder in reverse:
        #   256 → 128 → 64 → 32 → channels
        # kernel_size=4, stride=2, padding=1 is the exact inverse of the
        # encoder's Conv2d(kernel_size=4, stride=2, padding=1).
        self.deconv = nn.Sequential(
            # (B, 256, 4, 4)  →  (B, 128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B, 128, 8, 8)  →  (B, 64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B, 64, 16, 16)  →  (B, 32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B, 32, 32, 32)  →  (B, channels, 64, 64)
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
            # Sigmoid enforces [0.0, 1.0] on every output pixel (Req 5.1, 9.3).
            nn.Sigmoid(),
        )

    def forward(self, state: LatentState) -> Tensor:
        """Reconstruct frame ``(B, channels, 64, 64)``, pixels in ``[0.0, 1.0]``. (Req 5.1)

        The method concatenates the deterministic and flattened stochastic
        components of the latent state, projects them to a 4×4 spatial seed,
        then upsamples through four transposed convolutions back to 64×64.
        The final Sigmoid maps all activations to ``[0.0, 1.0]`` (Req 9.3).

        Args:
            state: A :class:`~agent.types.LatentState` with:
                - ``deter``: shape ``(B, deter_dim)``
                - ``stoch``: shape ``(B, num_categoricals, num_classes)``

        Returns:
            Reconstructed frame tensor of shape ``(B, channels, 64, 64)``
            with all pixel values in the closed interval ``[0.0, 1.0]``.
        """
        batch = state.deter.shape[0]

        # Flatten stochastic state: (B, 32, 32) → (B, 1024)
        stoch_flat = state.stoch.reshape(batch, -1)

        # Concatenate deterministic and stochastic components: (B, 1280)
        features = torch.cat([state.deter, stoch_flat], dim=-1)

        # Linear projection and reshape to spatial seed: (B, 1280) → (B, 256, 4, 4)
        x = self.project(features)               # (B, 4096)
        x = x.reshape(batch, 256, 4, 4)         # (B, 256, 4, 4)

        # Transposed-conv upsample: 4 → 8 → 16 → 32 → 64
        x = self.deconv(x)                       # (B, channels, 64, 64)

        return x
