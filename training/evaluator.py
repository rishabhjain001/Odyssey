"""Real-environment evaluation for Odyssey (Req 11).

This module implements :func:`evaluate`, which runs the Actor in the
CarRacing environment for a configured number of episodes and returns
aggregated performance metrics, while also storing every resulting transition
back into the replay buffer.

Evaluation overview
-------------------
For each episode the evaluator:

1. Resets the environment and builds an initial latent state.
2. Steps the Actor until the episode terminates or is truncated.
3. Records per-step transitions in the replay buffer (Req 11.6).
4. Records the episode's total reward and whether the track was completed
   (Req 11.3).

After all episodes the evaluator:

* Computes ``avg_reward`` as the arithmetic mean of per-episode totals
  (Req 11.2).
* Computes ``completion_rate`` as ``completed_episodes / total_episodes``
  in ``[0.0, 1.0]`` (Req 11.4).
* Records ``cumulative_env_steps = env_steps_so_far + steps_this_eval``
  (Req 11.5).

Track-completion convention
---------------------------
Gymnasium's CarRacing-v2 sets ``terminated = True`` when all track tiles have
been visited (natural race completion) and ``truncated = True`` when the time
limit fires without visiting all tiles.  Going off-track leads the environment
to end the episode with ``terminated = True`` but with a large accumulated
negative reward; the flag itself is the same as the success case, so we use
the simpler convention sanctioned by the requirement text (Req 11.3):

    ``track_completed = terminated and not truncated``

This treats both off-track termination and successful race completion as
"completed", which is the only meaningful signal CarRacing exposes via the
API.  Users should interpret ``completion_rate`` together with average reward
to understand driving quality.

Latent-state management
-----------------------
The Actor requires a :class:`~agent.types.LatentState` at each step.  The
evaluator supports two modes (selected by keyword arguments):

* **RSSM mode** (``rssm`` and ``encoder`` both provided): maintains a proper
  latent state by running the RSSM prior + encoder posterior at every step.
  This gives the Actor the same latent context it saw during training.
* **Zero-state mode** (default, when ``rssm`` or ``encoder`` is ``None``): uses
  an all-zeros ``LatentState`` (``deter=zeros``, ``stoch=zeros``) at every
  step.  This is suitable for quick smoke testing, before a world model has
  been trained, or when the evaluator is injected with a fake actor in tests.

Observation preprocessing
--------------------------
``FramePreprocessor(scope).to_frame(obs)`` is called on every observation
before passing it to the encoder (or discarding it in zero-state mode).
If ``to_frame`` or ``validate_observation`` raises, the offending observation
is skipped (not stored and not fed to the Actor), and the episode continues
with the unchanged latent state.

The environment is accepted as a plain argument so that any object
implementing the Gymnasium ``step`` / ``reset`` API can be injected for
testing without requiring the real CarRacing environment to be installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Tuple

import numpy as np
import torch

from agent.config import ScopeConfig
from agent.errors import FrameConversionError, ObservationValidationError
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from agent.types import EvalResult, LatentState, Transition

__all__ = ["evaluate"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal protocol types for injectability
# ---------------------------------------------------------------------------

class _EnvLike(Protocol):
    """Structural interface expected from the environment."""

    def reset(self, **kwargs: Any) -> Tuple[Any, Any]: ...

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Any]: ...


class _ActorLike(Protocol):
    """Structural interface expected from the actor.

    Both :class:`~agent.behavior.Actor` (``nn.Module``) and lightweight fake
    actors used in tests satisfy this protocol via ``__call__``, which is how
    PyTorch modules expose ``forward``.
    """

    def __call__(self, state: LatentState) -> "torch.Tensor": ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    actor: Any,
    env: Any,
    replay_buffer: ReplayBuffer,
    env_steps_so_far: int,
    episodes: int,
    scope: ScopeConfig,
    *,
    rssm: Optional[Any] = None,
    encoder: Optional[Any] = None,
) -> EvalResult:
    """Run the Actor in the environment and return aggregated evaluation metrics.

    Executes ``episodes`` full episodes and stores every collected transition
    in ``replay_buffer``.  See module docstring for the track-completion
    convention and latent-state management modes.

    Args:
        actor: The policy to evaluate.  Must be callable with a
            :class:`~agent.types.LatentState` and return a ``torch.Tensor`` of
            shape ``(1, 3)`` (or ``(3,)``).
        env: A Gymnasium-compatible environment instance (injectable for
            tests).  Must implement ``reset(**kwargs) -> (obs, info)`` and
            ``step(action) -> (obs, reward, terminated, truncated, info)``.
        replay_buffer: The :class:`~agent.replay.ReplayBuffer` to append
            collected transitions to.  Transitions from all episodes are stored
            in arrival order (Req 11.6).
        env_steps_so_far: The number of cumulative environment steps consumed
            by Odyssey *before* this evaluation run.  The returned
            ``EvalResult.cumulative_env_steps`` is this value plus the total
            steps taken during evaluation (Req 11.5).
        episodes: Number of episodes to run (Req 11.1).  Must be >= 1.
        scope: Active :class:`~agent.config.ScopeConfig` used to construct the
            :class:`~agent.perception.FramePreprocessor`.
        rssm: (keyword-only) Optional RSSM instance.  When provided together
            with ``encoder``, the latent state is advanced via the RSSM prior
            and posterior at each step.  When ``None``, a zero
            :class:`~agent.types.LatentState` is used throughout.
        encoder: (keyword-only) Optional Encoder instance.  Used together with
            ``rssm`` to compute posterior latent states.  Ignored when
            ``rssm`` is ``None``.

    Returns:
        An :class:`~agent.types.EvalResult` with:

        * ``avg_reward``: Arithmetic mean of total per-episode rewards
          (Req 11.2).
        * ``completion_rate``: Fraction of episodes in ``[0.0, 1.0]`` where
          ``terminated and not truncated`` (Req 11.3, 11.4).
        * ``cumulative_env_steps``: ``env_steps_so_far`` plus the total steps
          taken during this evaluation run (Req 11.5).

    Raises:
        ValueError: If ``episodes < 1``.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}.")

    preprocessor = FramePreprocessor(scope)
    use_rssm = rssm is not None and encoder is not None

    total_steps = 0
    episode_rewards: list[float] = []
    completed_episodes = 0

    for ep_idx in range(episodes):
        # ----------------------------------------------------------------
        # Reset environment
        # ----------------------------------------------------------------
        reset_result = env.reset()
        # Gymnasium's reset returns (obs, info); handle both new and legacy APIs.
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _info = reset_result
        else:
            obs = reset_result

        # Initialise the latent state for this episode.
        if use_rssm:
            latent: LatentState = rssm.initial(1)
        else:
            latent = _zero_latent_state()

        ep_reward = 0.0
        ep_steps = 0

        while True:
            # ----------------------------------------------------------------
            # Preprocess observation for actor / encoder input
            # ----------------------------------------------------------------
            frame_tensor: Optional[torch.Tensor] = None
            raw_obs = obs  # keep raw obs for Transition storage

            try:
                preprocessor.validate_observation(np.asarray(obs))
                frame = preprocessor.to_frame(np.asarray(obs))
                # frame: (channels, 64, 64) float32 ndarray
                frame_tensor = torch.from_numpy(frame).unsqueeze(0)  # (1,C,H,W)
            except (ObservationValidationError, FrameConversionError) as exc:
                logger.warning(
                    "Evaluator: skipping bad observation in episode %d step %d: %s",
                    ep_idx,
                    ep_steps,
                    exc,
                )

            # ----------------------------------------------------------------
            # Advance latent state via RSSM if available
            # ----------------------------------------------------------------
            if use_rssm and frame_tensor is not None:
                with torch.no_grad():
                    # Encode the current frame to get an embedding. The encoder
                    # returns a CategoricalLatent; the RSSM posterior expects a
                    # tensor embedding, so flatten the categorical logits to
                    # (B, num_categoricals * num_classes) == (B, embed_dim).
                    cat_latent = encoder(frame_tensor)
                    embed = cat_latent.logits.reshape(frame_tensor.shape[0], -1)
                    # Advance the deterministic state via the prior (using the
                    # action taken in the previous step, but we advance deter
                    # first, then posterior).  For the very first step we use
                    # the initial latent's deter directly.
                    latent = rssm.posterior(latent.deter, embed)

            # ----------------------------------------------------------------
            # Select action from Actor
            # ----------------------------------------------------------------
            with torch.no_grad():
                action_tensor = actor(latent)
            # action_tensor: (1, 3) or (3,)
            action_np = _tensor_to_action(action_tensor)

            # ----------------------------------------------------------------
            # Step the environment
            # ----------------------------------------------------------------
            step_result = env.step(action_np)
            next_obs, reward, terminated, truncated, _step_info = step_result

            reward = float(reward)
            done = bool(terminated) or bool(truncated)

            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            # ----------------------------------------------------------------
            # Store transition in replay buffer (Req 11.6)
            # ----------------------------------------------------------------
            try:
                preprocessor.validate_observation(np.asarray(raw_obs))
                transition = Transition(
                    observation=np.asarray(raw_obs, dtype=np.uint8),
                    action=action_np.copy(),
                    reward=reward,
                    terminated=bool(terminated),
                )
                replay_buffer.add(transition)
            except (ObservationValidationError, FrameConversionError) as exc:
                logger.warning(
                    "Evaluator: skipping transition storage for bad obs in "
                    "episode %d step %d: %s",
                    ep_idx,
                    ep_steps,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Evaluator: failed to store transition in episode %d step %d: %s",
                    ep_idx,
                    ep_steps,
                    exc,
                )

            # ----------------------------------------------------------------
            # Advance RSSM prior using executed action (for next step)
            # ----------------------------------------------------------------
            if use_rssm and frame_tensor is not None:
                with torch.no_grad():
                    action_t = torch.from_numpy(action_np).unsqueeze(0)  # (1, 3)
                    latent = rssm.predict_prior(latent, action_t)

            obs = next_obs

            if done:
                break

        # ----------------------------------------------------------------
        # End of episode bookkeeping
        # ----------------------------------------------------------------
        episode_rewards.append(ep_reward)

        # Track completion: episode ended naturally (all tiles visited) rather
        # than by time limit.  Convention: terminated=True and not truncated.
        # Note: going off-track also sets terminated=True, so this rate measures
        # "natural episode end" rather than strictly "all tiles visited".
        # Interpret alongside avg_reward for full picture. (Req 11.3)
        track_completed = bool(terminated) and not bool(truncated)
        if track_completed:
            completed_episodes += 1

        logger.debug(
            "Evaluator: episode %d/%d  reward=%.3f  completed=%s  steps=%d",
            ep_idx + 1,
            episodes,
            ep_reward,
            track_completed,
            ep_steps,
        )

    # ----------------------------------------------------------------
    # Aggregate metrics (Req 11.2, 11.4, 11.5)
    # ----------------------------------------------------------------
    avg_reward = float(np.mean(episode_rewards))
    completion_rate = float(completed_episodes / episodes)
    cumulative_env_steps = env_steps_so_far + total_steps

    return EvalResult(
        avg_reward=avg_reward,
        completion_rate=completion_rate,
        cumulative_env_steps=cumulative_env_steps,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _zero_latent_state() -> LatentState:
    """Return an all-zeros LatentState with MVP dimensions ``(1, 256)`` / ``(1, 32, 32)``."""
    deter = torch.zeros(1, 256, dtype=torch.float32)
    stoch = torch.zeros(1, 32, 32, dtype=torch.float32)
    return LatentState(deter=deter, stoch=stoch)


def _tensor_to_action(action_tensor: "torch.Tensor") -> np.ndarray:
    """Convert an Actor output tensor to a ``float32`` NumPy action array of shape ``(3,)``."""
    arr = action_tensor.detach().cpu().numpy()
    return arr.reshape(-1).astype(np.float32)
