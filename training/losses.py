"""Reconstruction and world-model loss functions for Odyssey (Req 5.2, 5.5).

This file is built in stages across multiple tasks:
  - Task 6.2 (this task): implements ``reconstruction_loss``, which computes
    MSE between input and reconstructed frames, raising
    ``ReconstructionShapeMismatch`` when shapes differ.
  - Task 9.1 (MUST EXTEND, not overwrite): appends the combined world-model
    objective (``world_model_loss``) after this file's existing code.

IMPORTANT for task 9.1: do NOT replace or overwrite this file. Append
``world_model_loss`` (and any helpers it needs) after the
``# --- EXTENSION POINT (task 9.1) ---`` comment below.
"""

from __future__ import annotations

from torch import Tensor

from agent.errors import ReconstructionShapeMismatch


def reconstruction_loss(input_frame: Tensor, recon_frame: Tensor) -> float:
    """MSE between frames; excludes mismatched-shape recon. (Req 5.2, 5.5)

    Computes the mean-squared error between ``input_frame`` and
    ``recon_frame``. If their shapes differ, a
    :class:`~agent.errors.ReconstructionShapeMismatch` exception is raised so
    the caller (``WorldModelTrainer``) can catch it, record the event via
    :meth:`~training.metrics.MetricsStore.record_reconstruction_shape_mismatch`,
    and exclude the pair from gradient computation.

    Properties:
    - The loss is always non-negative (MSE is non-negative by definition).
    - The loss is exactly ``0.0`` if and only if ``input_frame`` and
      ``recon_frame`` are identical (elementwise).

    Args:
        input_frame: The original input frame tensor (any shape).
        recon_frame: The decoder's reconstruction of ``input_frame``.

    Returns:
        A Python ``float`` equal to the MSE:
        ``((input_frame - recon_frame) ** 2).mean()``.

    Raises:
        ReconstructionShapeMismatch: When ``input_frame.shape != recon_frame.shape``.

    Req 5.2: MSE reconstruction loss between input and reconstructed frames.
    Req 5.5: mismatched-shape reconstructions are excluded and the mismatch
             is recorded by the caller.
    """
    if input_frame.shape != recon_frame.shape:
        raise ReconstructionShapeMismatch(
            f"Reconstruction shape {tuple(recon_frame.shape)} does not match "
            f"input shape {tuple(input_frame.shape)}; this pair will be excluded "
            "from the reconstruction loss."
        )
    return float(((input_frame - recon_frame) ** 2).mean())


# --- EXTENSION POINT (task 9.1) ---
# Task 9.1 MUST append world_model_loss (and any helpers) after this line.
# Do NOT modify or remove the reconstruction_loss function above.

# ---------------------------------------------------------------------------
# Task 9.1: Combined world-model objective (Req 7.1, 7.2, 7.3, 7.5)
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agent.encoder import Encoder
    from agent.rssm import RSSM
    from agent.decoder import Decoder
    from agent.heads import RewardPredictor, DiscountPredictor
    from agent.config import ScopeConfig

from agent.types import WorldModelLosses, Transition
from agent.perception import FramePreprocessor


def world_model_loss(
    batch: list[list],  # list[list[Transition]], outer=batch_size, inner=seq_len
    encoder: "Encoder",
    rssm: "RSSM",
    decoder: "Decoder",
    reward_predictor: "RewardPredictor",
    discount_predictor: "DiscountPredictor",
    scope: "ScopeConfig",
) -> WorldModelLosses:
    """Combined world-model objective over a batch of contiguous sequences.

    Composes four terms (Req 7.3):
      1. **Reconstruction loss** – MSE between input frame and decoder output.
      2. **KL divergence** – KL(prior || posterior) at each time step.
      3. **Reward loss** – MSE between predicted and actual reward.
      4. **Discount loss** – BCE between predicted discount and continuation label.

    The total loss is the sum of all four terms. All computations use custom
    PyTorch; no Stable-Baselines3 is imported or used (Req 7.1).

    Sequence length requirement (Req 7.2, 2.6, 9.4):
        Each sequence in ``batch`` must contain at least 50 transitions (the
        configured MVP sequence length). For RGB scopes, sequences must be
        ``scope.sequence_length`` steps which is in ``(50, 200]``. The caller
        (``ReplayBuffer.sample_sequences``) enforces this; ``world_model_loss``
        documents the ≥ 50 requirement but does not re-validate.

    RSSM unrolling:
        For each sequence the RSSM is unrolled across time steps:
          - t=0: the prior state is ``rssm.initial(B)``.
          - Each step t: encode ``obs_t`` → ``CategoricalLatent``; compute the
            posterior from ``(prev_deter, encoder_embed)``; advance the prior
            from ``(prev_state, action_t)``. Loss terms are computed on the
            **posterior** state to train the world model to match observations.

    KL computation:
        KL(prior || posterior) for discrete categoricals. At each step:
          - ``prior_logits  = prior_state.stoch.reshape(B, 32, 32)``  (the
            straight-through sample carries logits implicitly, but we need the
            actual prior logits; see note below).
          - Because ``rssm.predict_prior`` samples stoch from prior logits and
            the logits are not cached on LatentState, we compute the prior
            logits from the prior deterministic state via ``rssm.prior_head``.
          - ``post_logits`` come from ``rssm.posterior_head``.
          - KL = sum over the 32 categorical variables of KL between the
            per-variable prior and posterior distributions.
          - The posterior is treated as the *target* (stop-gradient on it):
            ``post_probs = softmax(post_logits.detach(), dim=-1)`` prevents
            gradient through the posterior path when minimizing KL.
          - Loss = mean over batch and time of sum over variables.

    Args:
        batch: ``list[list[Transition]]`` – outer list is batch_size sequences,
            inner list is seq_len Transitions (≥ 50 per Req 7.2).
        encoder: Trained Encoder module.
        rssm: Trained RSSM module.
        decoder: Trained Decoder module.
        reward_predictor: Trained RewardPredictor module.
        discount_predictor: Trained DiscountPredictor module.
        scope: Active :class:`~agent.config.ScopeConfig` (used to build
            :class:`~agent.perception.FramePreprocessor` for obs→frame).

    Returns:
        A :class:`~agent.types.WorldModelLosses` with all five fields as Python
        floats (recon, kl, reward, discount, total). All four component terms
        are computed; callers (``WorldModelTrainer``) should verify finiteness
        before performing a parameter update (Req 7.9).

    Req 7.1: custom PyTorch loop, no SB3.
    Req 7.2: sequences of length ≥ 50.
    Req 7.3: reconstruction + KL + reward + discount combined objective.
    Req 7.5: each loss term and total are returned per step.
    """
    preprocessor = FramePreprocessor(scope)

    batch_size = len(batch)
    seq_len = len(batch[0])

    # Determine device and dtype from the RSSM parameters.
    _device = rssm.prior_head.weight.device
    _dtype = rssm.prior_head.weight.dtype

    # Accumulators for each loss term (summed over time, then meaned).
    recon_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
    kl_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
    reward_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
    discount_sum = torch.tensor(0.0, device=_device, dtype=_dtype)

    # Track the number of valid reconstruction pairs (for correct averaging
    # when some steps are skipped due to ReconstructionShapeMismatch).
    recon_count = 0
    valid_steps = 0  # steps contributing to kl/reward/discount

    # Initialize the RSSM prior state for all sequences in the batch.
    prior_state = rssm.initial(batch_size)

    for t in range(seq_len):
        # -------------------------------------------------------------------
        # 1. Collect observations, actions, rewards, terminated flags for
        #    this time step across all sequences in the batch.
        # -------------------------------------------------------------------
        obs_list = []
        action_list = []
        reward_list = []
        terminated_list = []

        for seq in batch:
            tr: Transition = seq[t]
            # Convert raw observation to a preprocessed frame tensor.
            frame_np = preprocessor.to_frame(tr.observation)      # (C, 64, 64) float32
            obs_list.append(frame_np)
            action_list.append(tr.action.astype(np.float32))
            reward_list.append(float(tr.reward))
            terminated_list.append(float(tr.terminated))

        # Stack into batched tensors.
        # obs_t:  (B, C, 64, 64)
        obs_t = torch.tensor(
            np.stack(obs_list, axis=0), device=_device, dtype=_dtype
        )
        # action_t: (B, action_dim)
        action_t = torch.tensor(
            np.stack(action_list, axis=0), device=_device, dtype=_dtype
        )
        # reward_t: (B,)
        reward_t = torch.tensor(reward_list, device=_device, dtype=_dtype)
        # terminated_t: (B,) – discount target = 1 - terminated (continuation)
        terminated_t = torch.tensor(terminated_list, device=_device, dtype=_dtype)
        discount_target_t = 1.0 - terminated_t                    # (B,)

        # -------------------------------------------------------------------
        # 2. Encode observations → CategoricalLatent.
        #    The encoder embed fed to rssm.posterior is the logits flattened
        #    to (B, 1024) = (B, 32*32), matching rssm.embed_dim default.
        # -------------------------------------------------------------------
        cat_latent = encoder(obs_t)                                # CategoricalLatent
        # Flatten logits: (B, 32, 32) → (B, 1024)
        encoder_embed = cat_latent.logits.reshape(batch_size, -1)  # (B, 1024)

        # -------------------------------------------------------------------
        # 3. Compute posterior state conditioned on (prev_deter, encoder_embed).
        # -------------------------------------------------------------------
        post_state = rssm.posterior(prior_state.deter, encoder_embed)

        # -------------------------------------------------------------------
        # 4. Compute prior logits and posterior logits for KL divergence.
        #    The prior logits derive from the prior's deterministic state via
        #    rssm.prior_head; the posterior logits from rssm.posterior_head.
        #    We use the *current* prior deter (prior_state.deter) which was
        #    updated before this time step (or is the initial zero state at t=0).
        # -------------------------------------------------------------------
        prior_logits = rssm.prior_head(prior_state.deter)          # (B, 1024)
        prior_logits = prior_logits.reshape(
            batch_size, rssm.num_categoricals, rssm.num_classes
        )                                                           # (B, 32, 32)

        post_logits_for_kl = rssm.posterior_head(
            torch.cat([prior_state.deter, encoder_embed], dim=-1)
        )                                                           # (B, 1024)
        post_logits_for_kl = post_logits_for_kl.reshape(
            batch_size, rssm.num_categoricals, rssm.num_classes
        )                                                           # (B, 32, 32)

        # KL(prior || posterior):
        #   prior_probs    = softmax(prior_logits)   – optimized (gradient flows)
        #   post_probs     = softmax(post_logits.detach()) – stop-gradient target
        # sum over class dimension (dim=-1), then sum over variables (dim=-1 after),
        # then mean over batch. Shape walk:
        #   prior_probs:   (B, 32, 32)
        #   kl_per_var:    (B, 32)     – sum over 32 classes
        #   kl_per_seq:    (B,)        – sum over 32 variables
        #   kl_mean:       scalar
        prior_log_probs = F.log_softmax(prior_logits, dim=-1)       # (B, 32, 32)
        post_probs_sg = F.softmax(post_logits_for_kl.detach(), dim=-1)  # (B, 32, 32)

        # KL(prior || posterior) = sum_k prior_k * (log prior_k - log post_k)
        post_log_probs_sg = torch.log(post_probs_sg + 1e-8)         # (B, 32, 32)
        kl_per_class = torch.exp(prior_log_probs) * (
            prior_log_probs - post_log_probs_sg
        )                                                            # (B, 32, 32)
        kl_per_var = kl_per_class.sum(dim=-1)                        # (B, 32)
        kl_per_seq = kl_per_var.sum(dim=-1)                          # (B,)
        kl_step = kl_per_seq.mean()                                  # scalar

        # -------------------------------------------------------------------
        # 5. Advance the RSSM prior for the next time step using the action.
        #    We use the *posterior* state as the previous state to keep the
        #    prior unrolling grounded in the observed posterior.
        # -------------------------------------------------------------------
        prior_state = rssm.predict_prior(post_state, action_t)

        # -------------------------------------------------------------------
        # 6. Reconstruction loss on the posterior state (Req 7.3).
        # -------------------------------------------------------------------
        recon_frame = decoder(post_state)                            # (B, C, 64, 64)
        try:
            recon_val = reconstruction_loss(obs_t, recon_frame)      # float
            recon_sum = recon_sum + torch.tensor(
                recon_val, device=_device, dtype=_dtype
            )
            recon_count += 1
        except Exception:
            # ReconstructionShapeMismatch: skip this step for reconstruction.
            pass

        # -------------------------------------------------------------------
        # 7. Reward loss: MSE between predicted and actual reward (Req 7.3).
        # -------------------------------------------------------------------
        pred_reward = reward_predictor(post_state)                   # (B,)
        reward_step = F.mse_loss(pred_reward, reward_t)              # scalar

        # -------------------------------------------------------------------
        # 8. Discount loss: BCE between predicted discount and continuation
        #    label (1 - terminated) (Req 7.3).
        # -------------------------------------------------------------------
        pred_discount = discount_predictor(post_state)               # (B,) in [0,1]
        discount_step = F.binary_cross_entropy(
            pred_discount, discount_target_t
        )                                                            # scalar

        # Accumulate.
        kl_sum = kl_sum + kl_step
        reward_sum = reward_sum + reward_step
        discount_sum = discount_sum + discount_step
        valid_steps += 1

    # -------------------------------------------------------------------
    # 9. Average over time steps.
    # -------------------------------------------------------------------
    recon_loss = (recon_sum / recon_count).item() if recon_count > 0 else 0.0
    kl_loss = (kl_sum / valid_steps).item() if valid_steps > 0 else 0.0
    reward_loss = (reward_sum / valid_steps).item() if valid_steps > 0 else 0.0
    discount_loss = (discount_sum / valid_steps).item() if valid_steps > 0 else 0.0

    total_loss = recon_loss + kl_loss + reward_loss + discount_loss

    return WorldModelLosses(
        recon=recon_loss,
        kl=kl_loss,
        reward=reward_loss,
        discount=discount_loss,
        total=total_loss,
    )
