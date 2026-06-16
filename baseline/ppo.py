"""PPO baseline for Odyssey (Req 13).

This module wraps Stable-Baselines3 PPO on ``CarRacing-v2`` as the model-free
comparison benchmark.  It is deliberately thin: all heavy lifting is delegated
to SB3, and Odyssey only records the metric it cares about (average episode
reward, Req 13.4) and enforces the error-handling contract (Req 13.5).

Usage example::

    from baseline.ppo import PPOBaseline

    baseline = PPOBaseline(env_id="CarRacing-v2", seed=0)
    result = baseline.train(total_timesteps=100_000)
    print(result.avg_reward, result.total_steps)

    # Retrieve the latest result later:
    metrics = baseline.get_metrics()

Stable-Baselines3 is imported lazily (inside :meth:`PPOBaseline.train`) so
that the module can be imported cleanly in any environment, including those
that do not have SB3 installed.  An ``ImportError`` with a descriptive message
is raised only when training is attempted without SB3.

References
----------
- **Req 13.1** — Implement the PPO_Baseline using Stable-Baselines3.
- **Req 13.2** — The PPO_Baseline trains on CarRacing_Environment.
- **Req 13.3** — Train for a cumulative number of Environment_Steps equal to
  Odyssey's (passed as ``total_timesteps``).
- **Req 13.4** — Record average episode reward after PPO training completes.
- **Req 13.5** — On training failure, record PPOTrainingError and retain prior
  PPO metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.errors import PPOTrainingError

__all__ = ["PPOBaseline", "PPOResult"]

logger = logging.getLogger(__name__)

# Number of episodes to run during post-training evaluation (Req 13.4).
_EVAL_EPISODES: int = 5


@dataclass
class PPOResult:
    """Result of a completed PPO training run (Req 13.4).

    Attributes:
        avg_reward: Average episode reward measured over a short evaluation run
            after training, computed as the arithmetic mean of total per-episode
            rewards.
        total_steps: The number of environment steps for which the PPO agent was
            trained (the ``total_timesteps`` value passed to
            :meth:`PPOBaseline.train`).
    """

    avg_reward: float
    total_steps: int


class PPOBaseline:
    """Model-free PPO baseline wrapping Stable-Baselines3 (Req 13.1).

    Trains a PPO agent on ``env_id`` for a caller-specified environment-step
    budget (Req 13.3) and records the average episode reward (Req 13.4).

    Stable-Baselines3 is imported lazily inside :meth:`train` so the class can
    be instantiated even if SB3 is not installed; the ``ImportError`` is only
    raised when :meth:`train` is called.

    Args:
        env_id: Gymnasium environment identifier.  Defaults to
            ``"CarRacing-v2"`` (Req 13.2).
        seed: Random seed passed to the PPO model and evaluation environment.
    """

    def __init__(self, env_id: str = "CarRacing-v3", seed: int = 0) -> None:
        self._env_id = env_id
        self._seed = seed
        self._result: Optional[PPOResult] = None
        self.training_completed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> PPOResult:
        """Train PPO for ``total_timesteps`` and return a :class:`PPOResult`.

        Trains the SB3 PPO agent for exactly ``total_timesteps`` environment
        steps (Req 13.3), then evaluates the trained policy over
        :data:`_EVAL_EPISODES` episodes and records the arithmetic mean episode
        reward as ``avg_reward`` (Req 13.4).

        On success, the result is stored internally (accessible via
        :meth:`get_metrics`) and ``training_completed`` is set to ``True``.

        On **any** exception during training or evaluation the method raises a
        :class:`~agent.errors.PPOTrainingError` (wrapping the original
        exception), leaves the previously recorded result unchanged (Req 13.5),
        and leaves ``training_completed`` unchanged.

        Args:
            total_timesteps: The number of environment steps to train PPO for.
                This should equal the cumulative environment steps used to train
                Odyssey so the comparison is fair (Req 13.3).

        Returns:
            A :class:`PPOResult` with the average episode reward measured after
            training and the ``total_steps`` budget that was used.

        Raises:
            ImportError: If ``stable_baselines3`` is not installed.  The message
                explains how to install it.
            PPOTrainingError: If training or evaluation fails for any reason
                other than a missing dependency.  The prior result (if any) is
                retained (Req 13.5).
        """
        # Lazy import so the module is importable without SB3 (Req 13.1).
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is required for PPOBaseline.train().  "
                "Install it with: pip install stable-baselines3[extra]"
            ) from exc

        try:
            # ----------------------------------------------------------------
            # Build training environment
            # ----------------------------------------------------------------
            train_env = gym.make(self._env_id)

            # ----------------------------------------------------------------
            # Construct and train the PPO model (Req 13.1, 13.2, 13.3)
            # ----------------------------------------------------------------
            model = PPO(
                "CnnPolicy",
                train_env,
                seed=self._seed,
                verbose=0,
            )
            logger.info(
                "PPOBaseline: starting training on %s for %d steps.",
                self._env_id,
                total_timesteps,
            )
            model.learn(total_timesteps=total_timesteps)
            logger.info("PPOBaseline: training complete.")

            # ----------------------------------------------------------------
            # Post-training evaluation to record avg_reward (Req 13.4)
            # ----------------------------------------------------------------
            avg_reward = self._evaluate_policy(model, total_timesteps)

            # ----------------------------------------------------------------
            # Record successful result (Req 13.4)
            # ----------------------------------------------------------------
            result = PPOResult(avg_reward=avg_reward, total_steps=total_timesteps)
            self._result = result
            self.training_completed = True
            logger.info(
                "PPOBaseline: avg_reward=%.3f over %d evaluation episodes.",
                avg_reward,
                _EVAL_EPISODES,
            )

            train_env.close()
            return result

        except ImportError:
            # Re-raise import errors directly — not a training failure.
            raise
        except Exception as exc:
            # Retain prior result (Req 13.5) and wrap in PPOTrainingError.
            logger.error(
                "PPOBaseline: training failed — retaining prior metrics. Error: %s", exc
            )
            raise PPOTrainingError(
                f"PPO training on {self._env_id!r} failed: {exc}"
            ) from exc

    def get_metrics(self) -> Optional[PPOResult]:
        """Return the latest successfully recorded :class:`PPOResult`, or ``None``.

        Returns ``None`` when :meth:`train` has not yet completed successfully.

        Returns:
            The most recent :class:`PPOResult`, or ``None`` if no successful
            training run has been recorded yet.

        References:
            Req 13.4 — record average episode reward after PPO training.
            Req 13.5 — retain prior metrics on failure.
        """
        return self._result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_policy(self, model: object, total_timesteps: int) -> float:
        """Run a short evaluation to compute average episode reward (Req 13.4).

        Runs :data:`_EVAL_EPISODES` full episodes in a fresh environment using
        the trained ``model`` and returns the arithmetic mean of the total
        per-episode rewards.

        Args:
            model: The trained SB3 PPO model (must expose a ``predict`` method).
            total_timesteps: Passed through to the result; not used here but
                kept for documentation parity.

        Returns:
            Arithmetic mean of total per-episode rewards over
            :data:`_EVAL_EPISODES` episodes.
        """
        try:
            import gymnasium as gym  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required for PPOBaseline evaluation."
            ) from exc

        eval_env = gym.make(self._env_id)
        episode_rewards: list[float] = []

        try:
            for ep in range(_EVAL_EPISODES):
                obs, _info = eval_env.reset(seed=self._seed + ep)
                ep_reward = 0.0
                done = False
                while not done:
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = eval_env.step(action)
                    ep_reward += float(reward)
                    done = bool(terminated) or bool(truncated)
                episode_rewards.append(ep_reward)
                logger.debug(
                    "PPOBaseline eval episode %d/%d: reward=%.3f",
                    ep + 1,
                    _EVAL_EPISODES,
                    ep_reward,
                )
        finally:
            eval_env.close()

        return float(np.mean(episode_rewards))
