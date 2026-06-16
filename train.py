"""Odyssey -- single orchestrating entry point for all training phases (Req 18).

This script is the single root ``train.py`` that runs every Odyssey training
phase in the fixed canonical order required by Req 12.1 and 18.2:

    1. Random trajectory collection  (``pipeline.phases.collection``)
    2. World-model training          (``pipeline.phases.world_model``)
    3. Behavior-model training       (``pipeline.phases.behavior``)
    4. Real-environment evaluation   (``pipeline.phases.evaluation``)

Each phase can be independently enabled or disabled in the config file
(Req 18.3, 18.4). Within every iteration world-model training always precedes
behavior training (Req 12.2). Iterations alternate between world-model and
behavior training until ``pipeline.iteration_count`` is reached (Req 12.3).
Evaluation runs at the configured interval ``evaluation.interval_iterations``
(Req 12.4).

The reconstruction gate is consulted **centrally** here before every call to
the behavior trainer and before any PPO baseline execution, so neither can run
while reconstruction is unconfirmed, regardless of config flags (Req 18.5,
8.2–8.5).

Usage::

    python train.py
    python train.py --config configs/odyssey.yaml

The ``--config`` flag defaults to ``configs/odyssey.yaml``.

Design references:
    - Req 12.1–12.4: ordered phases, world-model before behavior, iteration
      count, evaluation interval.
    - Req 18.1–18.5: single entry point, fixed phase order, per-phase toggle,
      central gate enforcement.
    - DreamerV2 (Hafner et al., 2020): alternating world-model/behavior loop
      with imagined rollouts and reconstruction-gated behavior activation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup (before any other imports so all module-level loggers are
# captured by the root handler configured here).
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("odyssey.train")


# ---------------------------------------------------------------------------
# Project imports (no SB3 at module level per the design constraints).
# ---------------------------------------------------------------------------

from agent.behavior import Actor, Critic
from agent.config import load_config
from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.heads import DiscountPredictor, RewardPredictor
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from agent.rssm import RSSM
from training.behavior_trainer import BehaviorTrainer
from training.evaluator import evaluate
from training.gate import ReconstructionGate
from training.metrics import MetricsStore
from training.pipeline import collect_random_trajectories
from training.world_model_trainer import WorldModelTrainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Odyssey world-model RL agent -- full training pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/odyssey.yaml",
        metavar="PATH",
        help="Path to the YAML config file (default: configs/odyssey.yaml).",
    )
    return parser.parse_args()


def _build_components(config):
    """Construct and return all trainable components from ``config``.

    Returns a dict with keys: encoder, rssm, decoder, reward_predictor,
    discount_predictor, actor, critic, replay_buffer, metrics_store,
    preprocessor, world_model_trainer, behavior_trainer, gate.
    """
    # Neural components.
    encoder = Encoder(channels=config.scope.channels)
    rssm = RSSM(action_dim=3)
    decoder = Decoder(channels=config.scope.channels)
    reward_predictor = RewardPredictor()
    discount_predictor = DiscountPredictor()
    actor = Actor()
    critic = Critic()

    # Infrastructure components.
    replay_buffer = ReplayBuffer(capacity=config.replay.capacity)
    metrics_store = MetricsStore()
    preprocessor = FramePreprocessor(config.scope)

    # Training orchestrators.
    world_model_trainer = WorldModelTrainer(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_predictor=reward_predictor,
        discount_predictor=discount_predictor,
        replay_buffer=replay_buffer,
        metrics_store=metrics_store,
        config=config.world_model,
        scope=config.scope,
    )

    behavior_trainer = BehaviorTrainer(
        actor=actor,
        critic=critic,
        rssm=rssm,
        reward_p=reward_predictor,
        discount_p=discount_predictor,
        metrics_store=metrics_store,
        config=config.behavior,
    )

    # Reconstruction gate (Req 18.5).
    gate = ReconstructionGate(config.gate)

    return dict(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_predictor=reward_predictor,
        discount_predictor=discount_predictor,
        actor=actor,
        critic=critic,
        replay_buffer=replay_buffer,
        metrics_store=metrics_store,
        preprocessor=preprocessor,
        world_model_trainer=world_model_trainer,
        behavior_trainer=behavior_trainer,
        gate=gate,
    )


def _make_env():
    """Create and return the CarRacing-v3 Gymnasium environment (Req 1.1).

    An optional per-episode step cap can be supplied through the
    ``ODYSSEY_MAX_EPISODE_STEPS`` environment variable. This keeps evaluation
    episodes short for fast demonstration runs; when unset, the environment's
    default time limit applies.
    """
    import os  # noqa: PLC0415
    import gymnasium as gym  # noqa: PLC0415 -- import here to keep module lean

    max_steps_env = os.environ.get("ODYSSEY_MAX_EPISODE_STEPS")
    if max_steps_env:
        env = gym.make(
            "CarRacing-v3", continuous=True, max_episode_steps=int(max_steps_env)
        )
    else:
        env = gym.make("CarRacing-v3", continuous=True)
    return env


def run_training(config_path: str = "configs/odyssey.yaml") -> MetricsStore:
    """Execute the full Odyssey training pipeline and return the metrics store.

    This is the main orchestration function, extracted so it can be called from
    tests or other scripts without going through ``argparse``.

    Phase order (Req 12.1, 18.2):
        1. Random trajectory collection (once, before the iteration loop).
        2. Per-iteration loop up to ``pipeline.iteration_count``:
           a. World-model training step (if enabled).
           b. Gate check (Req 18.5).
           c. Behavior training step (if gate open AND enabled in config).
           d. Evaluation (if enabled AND interval reached).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The populated :class:`~training.metrics.MetricsStore` for inspection or
        downstream use (e.g. the dashboard, comparison functions).
    """
    logger.info("Loading configuration from: %s", config_path)
    config = load_config(config_path)
    logger.info(
        "Config loaded — scope=%s, channels=%d, seq_len=%d, "
        "iterations=%d, collection_steps=%d",
        config.scope.name,
        config.scope.channels,
        config.scope.sequence_length,
        config.pipeline.iteration_count,
        config.collection.env_steps,
    )

    # ------------------------------------------------------------------ #
    # Build all components.
    # ------------------------------------------------------------------ #
    components = _build_components(config)
    world_model_trainer: WorldModelTrainer = components["world_model_trainer"]
    behavior_trainer: BehaviorTrainer = components["behavior_trainer"]
    replay_buffer: ReplayBuffer = components["replay_buffer"]
    metrics_store: MetricsStore = components["metrics_store"]
    gate: ReconstructionGate = components["gate"]
    actor = components["actor"]
    encoder = components["encoder"]
    rssm = components["rssm"]

    # ------------------------------------------------------------------ #
    # Load checkpoint at startup if available (Req 7.7).
    # ------------------------------------------------------------------ #
    start_step = world_model_trainer.maybe_load_checkpoint()
    if start_step:
        logger.info("Resumed from checkpoint at step %d.", start_step)

    # ------------------------------------------------------------------ #
    # Track cumulative env steps across all phases.
    # ------------------------------------------------------------------ #
    env_steps: int = 0
    world_model_step: int = start_step  # global world-model training step counter

    # ------------------------------------------------------------------ #
    # Phase 1 — Random trajectory collection (Req 12.1, 18.2, 18.3).
    # ------------------------------------------------------------------ #
    if config.pipeline.phases.collection:
        logger.info(
            "Phase 1: Random trajectory collection (%d env steps).",
            config.collection.env_steps,
        )
        env = _make_env()
        try:
            result = collect_random_trajectories(
                env=env,
                buffer=replay_buffer,
                config=config.collection,
            )
            env_steps += result.env_steps
            logger.info(
                "Collection complete — stored=%d, env_steps=%d, "
                "rejected_obs=%d, episodes=%d",
                result.stored,
                result.env_steps,
                result.rejected_observations,
                result.episodes,
            )
        finally:
            env.close()
    else:
        logger.info("Phase 1: Collection disabled in config — skipping.")

    # ------------------------------------------------------------------ #
    # Phase 2 + 3 + 4 — Iterative training loop (Req 12.2, 12.3, 12.4).
    # ------------------------------------------------------------------ #
    iteration_count = config.pipeline.iteration_count
    logger.info(
        "Starting iteration loop: %d iterations, eval every %d iterations.",
        iteration_count,
        config.evaluation.interval_iterations,
    )

    for iteration in range(1, iteration_count + 1):

        # -------------------------------------------------------------- #
        # Phase 2 — World-model training step (Req 12.2).
        # -------------------------------------------------------------- #
        if config.pipeline.phases.world_model:
            losses = world_model_trainer.train_step(world_model_step)
            if losses is not None:
                # Record the reconstruction loss component centrally (Req 5.3).
                metrics_store.record_reconstruction_loss(world_model_step, losses.recon)
                logger.debug(
                    "Iteration %d | WM step %d | recon=%.4f kl=%.4f "
                    "reward=%.4f discount=%.4f total=%.4f",
                    iteration,
                    world_model_step,
                    losses.recon,
                    losses.kl,
                    losses.reward,
                    losses.discount,
                    losses.total,
                )
            else:
                logger.debug(
                    "Iteration %d | WM step %d skipped (insufficient data or "
                    "non-finite loss).",
                    iteration,
                    world_model_step,
                )
            world_model_step += 1
        else:
            logger.debug("Iteration %d: World-model training disabled — skipping.", iteration)

        # -------------------------------------------------------------- #
        # Phase 3 — Behavior training (Req 12.2, 18.5).
        # Consult the gate CENTRALLY before allowing behavior training
        # (Req 18.5). The gate uses the full reconstruction-loss history.
        # -------------------------------------------------------------- #
        recon_history = [v for _, v in metrics_store.reconstruction_losses()]
        behavior_config_enabled = (
            config.behavior.enabled and config.pipeline.phases.behavior
        )
        allowed, err = gate.behavior_allowed(behavior_config_enabled, recon_history)

        if allowed:
            # Gate open and behavior enabled in config — run behavior step.
            # Build a minimal start state from the RSSM initial state.
            start_state = rssm.initial(batch=1)
            result_b = behavior_trainer.train_step(
                start_states=[start_state],
                step=world_model_step,
            )
            if result_b is not None:
                logger.debug(
                    "Iteration %d | Behavior step | actor_loss=%.4f critic_loss=%.4f",
                    iteration,
                    result_b.get("actor_loss", float("nan")),
                    result_b.get("critic_loss", float("nan")),
                )
        elif err:
            # Gate closed — log the GateNotConfirmedError message (already
            # logged inside gate.behavior_allowed, but surface at INFO level
            # on the first occurrence to make it visible).
            logger.warning("Iteration %d: Behavior skipped — gate not confirmed.", iteration)
        else:
            # Behavior disabled by config — skip silently.
            pass

        # -------------------------------------------------------------- #
        # Phase 4 — Real-environment evaluation (Req 12.4).
        # -------------------------------------------------------------- #
        eval_interval = config.evaluation.interval_iterations
        if (
            config.pipeline.phases.evaluation
            and eval_interval > 0
            and iteration % eval_interval == 0
        ):
            logger.info(
                "Iteration %d: Running evaluation (%d episodes).",
                iteration,
                config.evaluation.episodes,
            )
            eval_env = _make_env()
            try:
                eval_result = evaluate(
                    actor=actor,
                    env=eval_env,
                    replay_buffer=replay_buffer,
                    env_steps_so_far=env_steps,
                    episodes=config.evaluation.episodes,
                    scope=config.scope,
                    rssm=rssm,
                    encoder=encoder,
                )
                # Accumulate env steps used during evaluation.
                eval_env_steps = eval_result.cumulative_env_steps - env_steps
                env_steps = eval_result.cumulative_env_steps

                metrics_store.record_eval(
                    env_steps=eval_result.cumulative_env_steps,
                    avg_reward=eval_result.avg_reward,
                    completion_rate=eval_result.completion_rate,
                )
                logger.info(
                    "Evaluation at iteration %d — avg_reward=%.3f "
                    "completion_rate=%.3f cumulative_env_steps=%d",
                    iteration,
                    eval_result.avg_reward,
                    eval_result.completion_rate,
                    eval_result.cumulative_env_steps,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Evaluation at iteration %d failed: %s", iteration, exc
                )
            finally:
                eval_env.close()
        elif not config.pipeline.phases.evaluation:
            pass  # evaluation phase disabled globally

        # Log progress every 10 iterations.
        if iteration % 10 == 0 or iteration == iteration_count:
            logger.info(
                "Progress: iteration %d/%d  cumulative_env_steps=%d",
                iteration,
                iteration_count,
                env_steps,
            )

    # ------------------------------------------------------------------ #
    # Optional PPO baseline (Req 13, 18.5 gate still applies).
    # ------------------------------------------------------------------ #
    if config.baseline.enabled:
        # Gate check: PPO baseline cannot run while reconstruction is unconfirmed
        # (Req 18.5, 8.3). Reuse behavior_allowed with True so the gate logic is
        # applied; if the gate is closed we skip baseline rather than silently
        # running it.
        recon_history_final = [v for _, v in metrics_store.reconstruction_losses()]
        baseline_allowed, baseline_err = gate.behavior_allowed(True, recon_history_final)

        if baseline_allowed:
            logger.info(
                "Running PPO baseline for %d steps (gate confirmed).", env_steps
            )
            # Lazy import: stable_baselines3 is only needed when baseline is
            # enabled (Req design constraint — no SB3 at module level).
            try:
                from baseline.ppo import PPOBaseline  # noqa: PLC0415

                ppo = PPOBaseline(env_id="CarRacing-v3")
                ppo_result = ppo.train(total_timesteps=env_steps)
                logger.info(
                    "PPO baseline finished — avg_reward=%.3f total_steps=%d",
                    ppo_result.avg_reward,
                    ppo_result.total_steps,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("PPO baseline failed: %s", exc)
        else:
            if baseline_err:
                logger.warning(
                    "PPO baseline skipped — gate not confirmed: %s", baseline_err
                )
            else:
                logger.info("PPO baseline disabled by gate (reconstruction unconfirmed).")

    # ------------------------------------------------------------------ #
    # Persist metrics (Req 14.5).
    # ------------------------------------------------------------------ #
    metrics_path = "metrics.json"
    try:
        metrics_store.persist(metrics_path)
        logger.info("Metrics persisted to: %s", metrics_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist metrics: %s", exc)

    logger.info(
        "Training complete — total cumulative_env_steps=%d  "
        "world_model_steps=%d",
        env_steps,
        world_model_step,
    )
    return metrics_store


def main() -> None:
    """Parse arguments and run the training pipeline."""
    args = _parse_args()
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
