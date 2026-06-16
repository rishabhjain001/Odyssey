"""Checkpoint save/load for the Odyssey world model (Req 7.6, 7.7).

Provides two module-level functions that the :class:`WorldModelTrainer` calls:

    * :func:`save_checkpoint` -- persist all five world-model component state
      dicts plus the current training step to a ``.pt`` file.
    * :func:`maybe_load_checkpoint` -- silently return ``0`` when no checkpoint
      file exists; load and restore component state dicts and return the saved
      step when the file is present and valid.

These are thin wrappers over ``torch.save`` / ``torch.load`` so the trainer
itself stays free of file-I/O logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch

from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.heads import DiscountPredictor, RewardPredictor
from agent.rssm import RSSM

__all__ = ["save_checkpoint", "maybe_load_checkpoint"]

# Key names stored inside the checkpoint dict.
_KEY_STEP = "step"
_KEY_ENCODER = "encoder"
_KEY_RSSM = "rssm"
_KEY_DECODER = "decoder"
_KEY_REWARD = "reward_predictor"
_KEY_DISCOUNT = "discount_predictor"


def save_checkpoint(
    path: Union[str, Path],
    encoder: Encoder,
    rssm: RSSM,
    decoder: Decoder,
    reward_predictor: RewardPredictor,
    discount_predictor: DiscountPredictor,
    step: int,
) -> None:
    """Persist world-model parameters to ``path`` (Req 7.6).

    Creates parent directories if needed.  Overwrites any existing file at
    ``path``.

    Args:
        path: Destination ``.pt`` file path.
        encoder: Encoder module.
        rssm: RSSM module.
        decoder: Decoder module.
        reward_predictor: RewardPredictor head.
        discount_predictor: DiscountPredictor head.
        step: The current training step to embed in the checkpoint.
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    checkpoint = {
        _KEY_STEP: int(step),
        _KEY_ENCODER: encoder.state_dict(),
        _KEY_RSSM: rssm.state_dict(),
        _KEY_DECODER: decoder.state_dict(),
        _KEY_REWARD: reward_predictor.state_dict(),
        _KEY_DISCOUNT: discount_predictor.state_dict(),
    }
    torch.save(checkpoint, str(path))


def maybe_load_checkpoint(
    path: Union[str, Path],
    encoder: Encoder,
    rssm: RSSM,
    decoder: Decoder,
    reward_predictor: RewardPredictor,
    discount_predictor: DiscountPredictor,
) -> int:
    """Load world-model parameters from ``path`` if it exists (Req 7.7).

    Silently returns ``0`` when ``path`` does not exist or is empty, so the
    caller can start training from step 0 without extra conditionals.

    Args:
        path: Path to the ``.pt`` checkpoint file (written by
            :func:`save_checkpoint`).
        encoder: Encoder module (mutated in-place on load).
        rssm: RSSM module (mutated in-place on load).
        decoder: Decoder module (mutated in-place on load).
        reward_predictor: RewardPredictor head (mutated in-place on load).
        discount_predictor: DiscountPredictor head (mutated in-place on load).

    Returns:
        The saved training step (int) if a checkpoint was loaded, or ``0`` if
        no checkpoint was available.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return 0

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint[_KEY_ENCODER])
    rssm.load_state_dict(checkpoint[_KEY_RSSM])
    decoder.load_state_dict(checkpoint[_KEY_DECODER])
    reward_predictor.load_state_dict(checkpoint[_KEY_REWARD])
    discount_predictor.load_state_dict(checkpoint[_KEY_DISCOUNT])
    return int(checkpoint.get(_KEY_STEP, 0))
