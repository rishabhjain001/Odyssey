"""WorldModelTrainer -- all-or-none world-model parameter updates (Req 7.4, 7.5, 7.8, 7.9).

This module provides :class:`WorldModelTrainer`, which runs a single custom
PyTorch training step over all five world-model components (Encoder, RSSM,
Decoder, RewardPredictor, DiscountPredictor) atomically:

    * If the replay buffer cannot satisfy the configured batch request, the step
      is skipped, an error is recorded, and **all five component parameter sets
      are left unchanged** (Req 7.8).
    * If the computed total loss is not finite (NaN or Inf), the step is skipped,
      an error is recorded, and **all five component parameter sets are left
      unchanged** (Req 7.9).
    * Otherwise **all five components are updated together** by a single call to
      ``optimizer.step()`` (Req 7.4).
    * After a successful update the individual loss terms and total are recorded
      in the metrics store (Req 7.5).

A single Adam optimizer is constructed over the concatenated parameter lists of
all five components, so the ``optimizer.step()`` call is the *only* place
parameter writes happen -- guaranteeing the all-or-none property.

Design notes:
    * ``world_model_loss`` from ``training/losses.py`` returns Python floats (all
      values extracted via ``.item()``), which cannot be back-propagated. To
      retain the computational graph, :meth:`WorldModelTrainer._compute_losses`
      mirrors the structure of ``world_model_loss`` but keeps the total loss as
      a live ``torch.Tensor`` for backpropagation, and also returns a
      :class:`~agent.types.WorldModelLosses` of floats for metrics recording.
    * Learning rate 6e-4 is design discretion (well within the DreamerV2 range)
      and is documented here as the chosen default.

References:
    - DreamerV2 (Hafner et al., 2020): world-model training with Adam, combined
      reconstruction + KL + reward + discount objective.
    - Req 7.4: all-or-none update.
    - Req 7.5: per-step loss recording.
    - Req 7.8: skip on insufficient data.
    - Req 7.9: skip on non-finite loss.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from agent.config import ScopeConfig, WorldModelConfig
from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.errors import InsufficientDataError, NonFiniteLossError
from agent.heads import DiscountPredictor, RewardPredictor
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from agent.rssm import RSSM
from agent.types import Transition, WorldModelLosses
from training.checkpoint import maybe_load_checkpoint, save_checkpoint
from training.losses import reconstruction_loss
from training.metrics import MetricsStore

__all__ = ["WorldModelTrainer"]

# Default Adam learning rate -- design discretion (DreamerV2 reference uses 6e-4).
_DEFAULT_LR: float = 6e-4


class WorldModelTrainer:
    """Trains all five world-model components with an all-or-none update.

    The trainer owns a single Adam optimizer over the concatenated parameter
    lists of all five components. A training step either updates all five
    together (after a successful forward pass with a finite loss) or updates
    none of them (on ``InsufficientDataError`` or ``NonFiniteLossError``).

    Args:
        encoder: The convolutional Encoder module.
        rssm: The Recurrent State-Space Model module.
        decoder: The transposed-conv Decoder module.
        reward_predictor: The scalar reward prediction head.
        discount_predictor: The episode-continuation prediction head.
        replay_buffer: The replay buffer from which training batches are sampled.
        metrics_store: The central metrics store for recording per-step losses.
        config: World-model training configuration (batch size, checkpoint
            interval and path).
        scope: Active scope configuration (channels, sequence length). Used to
            preprocess raw observations into frame tensors and to determine the
            sequence length for batch sampling.
        lr: Adam learning rate. Defaults to 6e-4 (design discretion).
    """

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        reward_predictor: RewardPredictor,
        discount_predictor: DiscountPredictor,
        replay_buffer: ReplayBuffer,
        metrics_store: MetricsStore,
        config: WorldModelConfig,
        scope: ScopeConfig,
        lr: float = _DEFAULT_LR,
    ) -> None:
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_predictor = reward_predictor
        self.discount_predictor = discount_predictor
        self.replay_buffer = replay_buffer
        self.metrics_store = metrics_store
        self.config = config
        self.scope = scope

        # Single Adam optimizer over ALL five components (Req 7.4).
        # A single optimizer.step() either updates all five or none.
        all_params = (
            list(encoder.parameters())
            + list(rssm.parameters())
            + list(decoder.parameters())
            + list(reward_predictor.parameters())
            + list(discount_predictor.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    # ------------------------------------------------------------------
    # Checkpoint management (Req 7.6, 7.7)
    # ------------------------------------------------------------------

    def maybe_load_checkpoint(self) -> int:
        """Load world-model parameters from the configured checkpoint if available.

        Delegates to :func:`training.checkpoint.maybe_load_checkpoint`.  Loads
        from ``config.checkpoint_path`` when the file exists and is non-empty,
        then returns the saved step so training can resume from the right
        position.  Returns 0 when no checkpoint is found so that callers can
        start counting from step 0 without extra conditionals.

        Typical usage in a training script::

            trainer = WorldModelTrainer(...)
            start_step = trainer.maybe_load_checkpoint()
            for step in range(start_step, total_steps):
                trainer.train_step(step)

        Returns:
            The saved training step (int) if a checkpoint was loaded, or ``0``
            if no checkpoint was available.

        Req 7.7: load world-model parameters from a checkpoint at startup.
        """
        return maybe_load_checkpoint(
            path=self.config.checkpoint_path,
            encoder=self.encoder,
            rssm=self.rssm,
            decoder=self.decoder,
            reward_predictor=self.reward_predictor,
            discount_predictor=self.discount_predictor,
        )

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def train_step(self, step: int) -> WorldModelLosses | None:
        """Run one world-model training step (Req 7.4, 7.5, 7.8, 7.9).

        Steps:
            1. Sample a batch from the replay buffer
               (``InsufficientDataError`` → skip, record, return ``None``).
            2. Zero gradients on the optimizer.
            3. Compute the combined world-model loss, retaining the total as a
               live tensor for backprop.
            4. Check loss finiteness
               (non-finite → raise/catch ``NonFiniteLossError``, record, return
               ``None``; **no** ``optimizer.step()`` is called).
            5. Back-propagate and call ``optimizer.step()`` to update all five
               components together (Req 7.4).
            6. Record all loss terms and total in the metrics store (Req 7.5).
            7. Return the :class:`~agent.types.WorldModelLosses`.

        Args:
            step: The current training step index (used for metrics recording).

        Returns:
            :class:`~agent.types.WorldModelLosses` on a successful update, or
            ``None`` if the step was skipped.
        """
        # ------------------------------------------------------------------ #
        # 1. Data sufficiency check (Req 7.8)
        # ------------------------------------------------------------------ #
        try:
            batch = self.replay_buffer.sample_sequences(
                batch_size=self.config.batch_size,
                seq_len=self.scope.sequence_length,
            )
        except InsufficientDataError:
            # Record the skip event. MetricsStore.record_world_model_losses casts
            # all values to float, so we encode the skip as a numeric flag (1.0)
            # rather than a string to stay JSON-serializable and float-compatible.
            self.metrics_store.record_world_model_losses(
                step, {"skipped": 1.0, "insufficient_data": 1.0}
            )
            # All five component parameters are left unchanged (Req 7.8).
            return None

        # ------------------------------------------------------------------ #
        # 2. Zero gradients before the forward pass.
        # ------------------------------------------------------------------ #
        self.optimizer.zero_grad()

        # ------------------------------------------------------------------ #
        # 3. Forward pass -- compute losses while retaining the graph.
        # ------------------------------------------------------------------ #
        losses_float, total_tensor = self._compute_losses(batch)

        # ------------------------------------------------------------------ #
        # 4. Finiteness check (Req 7.9)
        # ------------------------------------------------------------------ #
        if not math.isfinite(losses_float.total):
            # Catch internally: do NOT call optimizer.step().
            try:
                raise NonFiniteLossError(
                    f"World-model total loss is not finite ({losses_float.total!r}) "
                    "at step {step}; skipping parameter update."
                )
            except NonFiniteLossError:
                pass  # Caught internally; handled below.
            self.metrics_store.record_world_model_losses(
                step, {"skipped": 1.0, "non_finite_loss": 1.0}
            )
            # All five component parameters are left unchanged (Req 7.9).
            return None

        # ------------------------------------------------------------------ #
        # 5. All-or-none parameter update (Req 7.4)
        # ------------------------------------------------------------------ #
        total_tensor.backward()
        self.optimizer.step()

        # ------------------------------------------------------------------ #
        # 5a. Checkpoint save at configured interval (Req 7.6)
        # ------------------------------------------------------------------ #
        if (step + 1) % self.config.checkpoint_interval == 0:
            save_checkpoint(
                path=self.config.checkpoint_path,
                encoder=self.encoder,
                rssm=self.rssm,
                decoder=self.decoder,
                reward_predictor=self.reward_predictor,
                discount_predictor=self.discount_predictor,
                step=step,
            )

        # ------------------------------------------------------------------ #
        # 6. Record per-step loss terms (Req 7.5)
        # ------------------------------------------------------------------ #
        self.metrics_store.record_world_model_losses(
            step,
            {
                "recon": losses_float.recon,
                "kl": losses_float.kl,
                "reward": losses_float.reward,
                "discount": losses_float.discount,
                "total": losses_float.total,
            },
        )

        # ------------------------------------------------------------------ #
        # 7. Return the float losses summary.
        # ------------------------------------------------------------------ #
        return losses_float

    # ------------------------------------------------------------------
    # Internal loss computation (retains tensor graph for backward)
    # ------------------------------------------------------------------

    def _compute_losses(
        self, batch: list[list[Transition]]
    ) -> tuple[WorldModelLosses, Tensor]:
        """Compute the combined world-model objective over a batch.

        This mirrors the structure of ``training.losses.world_model_loss`` but
        retains the total loss as a live ``torch.Tensor`` with the full
        computational graph so that ``total_tensor.backward()`` can propagate
        gradients to all five components.

        Returns:
            A tuple of:
                * :class:`~agent.types.WorldModelLosses` with Python floats for
                  metrics recording and finiteness checking.
                * The total loss as a ``torch.Tensor`` (with grad) for
                  ``backward()``.
        """
        preprocessor = FramePreprocessor(self.scope)

        batch_size = len(batch)
        seq_len = len(batch[0])

        # Determine device/dtype from the RSSM parameters.
        _device = self.rssm.prior_head.weight.device
        _dtype = self.rssm.prior_head.weight.dtype

        # Tensor accumulators (live tensors, graph retained).
        recon_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
        kl_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
        reward_sum = torch.tensor(0.0, device=_device, dtype=_dtype)
        discount_sum = torch.tensor(0.0, device=_device, dtype=_dtype)

        recon_count = 0
        valid_steps = 0

        # Initialize RSSM prior state.
        prior_state = self.rssm.initial(batch_size)

        for t in range(seq_len):
            obs_list, action_list, reward_list, terminated_list = [], [], [], []
            for seq in batch:
                tr: Transition = seq[t]
                frame_np = preprocessor.to_frame(tr.observation)
                obs_list.append(frame_np)
                action_list.append(tr.action.astype(np.float32))
                reward_list.append(float(tr.reward))
                terminated_list.append(float(tr.terminated))

            obs_t = torch.tensor(
                np.stack(obs_list, axis=0), device=_device, dtype=_dtype
            )
            action_t = torch.tensor(
                np.stack(action_list, axis=0), device=_device, dtype=_dtype
            )
            reward_t = torch.tensor(reward_list, device=_device, dtype=_dtype)
            discount_target_t = 1.0 - torch.tensor(
                terminated_list, device=_device, dtype=_dtype
            )

            # Encode observations.
            cat_latent = self.encoder(obs_t)
            encoder_embed = cat_latent.logits.reshape(batch_size, -1)

            # Posterior state conditioned on (deter, embed).
            post_state = self.rssm.posterior(prior_state.deter, encoder_embed)

            # Prior and posterior logits for KL.
            prior_logits = self.rssm.prior_head(prior_state.deter)
            prior_logits = prior_logits.reshape(
                batch_size, self.rssm.num_categoricals, self.rssm.num_classes
            )
            post_logits_for_kl = self.rssm.posterior_head(
                torch.cat([prior_state.deter, encoder_embed], dim=-1)
            )
            post_logits_for_kl = post_logits_for_kl.reshape(
                batch_size, self.rssm.num_categoricals, self.rssm.num_classes
            )

            # KL(prior || posterior) -- posterior is stop-gradient target.
            prior_log_probs = F.log_softmax(prior_logits, dim=-1)
            post_probs_sg = F.softmax(post_logits_for_kl.detach(), dim=-1)
            post_log_probs_sg = torch.log(post_probs_sg + 1e-8)
            kl_per_class = torch.exp(prior_log_probs) * (
                prior_log_probs - post_log_probs_sg
            )
            kl_step = kl_per_class.sum(dim=-1).sum(dim=-1).mean()

            # Advance prior for next time step using posterior as starting state.
            prior_state = self.rssm.predict_prior(post_state, action_t)

            # Reconstruction loss (as live tensor).
            recon_frame = self.decoder(post_state)
            try:
                # reconstruction_loss returns a float; we need a tensor for grad.
                # Re-compute as MSE tensor directly to keep the graph.
                if obs_t.shape == recon_frame.shape:
                    recon_step = ((obs_t - recon_frame) ** 2).mean()
                    recon_sum = recon_sum + recon_step
                    recon_count += 1
                else:
                    # Shape mismatch -- record and skip (Req 5.5).
                    self.metrics_store.record_reconstruction_shape_mismatch(
                        step,
                        f"obs {tuple(obs_t.shape)} vs recon {tuple(recon_frame.shape)}",
                    )
            except Exception:
                pass

            # Reward loss (MSE, live tensor).
            pred_reward = self.reward_predictor(post_state)
            reward_step = F.mse_loss(pred_reward, reward_t)

            # Discount loss (BCE, live tensor).
            pred_discount = self.discount_predictor(post_state)
            discount_step = F.binary_cross_entropy(pred_discount, discount_target_t)

            kl_sum = kl_sum + kl_step
            reward_sum = reward_sum + reward_step
            discount_sum = discount_sum + discount_step
            valid_steps += 1

        # Average over time steps.
        recon_tensor = (recon_sum / recon_count) if recon_count > 0 else recon_sum
        kl_tensor = (kl_sum / valid_steps) if valid_steps > 0 else kl_sum
        reward_tensor = (reward_sum / valid_steps) if valid_steps > 0 else reward_sum
        discount_tensor = (discount_sum / valid_steps) if valid_steps > 0 else discount_sum

        total_tensor = recon_tensor + kl_tensor + reward_tensor + discount_tensor

        # Float summary for metrics / finiteness check.
        losses_float = WorldModelLosses(
            recon=float(recon_tensor.item()),
            kl=float(kl_tensor.item()),
            reward=float(reward_tensor.item()),
            discount=float(discount_tensor.item()),
            total=float(total_tensor.item()),
        )

        return losses_float, total_tensor
