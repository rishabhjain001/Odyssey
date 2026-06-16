# Feature: odyssey, Property 25: Phase orchestration runs exactly the enabled phases in fixed order
"""Tests for train.py phase orchestration.

Property 25: Phase orchestration runs exactly the enabled phases in fixed order

For any configuration subset of enabled phases, executing run_training():
  - Runs exactly the enabled phases.
  - Skips disabled phases.
  - Executed phases appear in the fixed canonical order:
      random trajectory collection
      → World_Model training
      → Behavior_Model training (when gate is open)
      → real environment evaluation
  - Within any iteration, World_Model training precedes Behavior_Model training.

Implementation under test: train.py → run_training(config_path)

Test design:
  - Use plain pytest (NOT hypothesis — 4 phases means 16 combos; just cover key cases).
  - Monkeypatch heavy operations so tests run instantly.
  - Track which phases ran via a shared call-log list.

Validates: Requirements 12.2, 18.2, 18.3, 18.4
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.types import EvalResult, WorldModelLosses
from training.pipeline import CollectionResult

# ---------------------------------------------------------------------------
# Helpers – minimal fake objects
# ---------------------------------------------------------------------------

VALID_OBS_SHAPE = (96, 96, 3)


def _valid_obs() -> np.ndarray:
    return np.full(VALID_OBS_SHAPE, 128, dtype=np.uint8)


class _FakeActionSpace:
    def sample(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def seed(self, s: int) -> None:
        pass


class _FakeEnv:
    """Minimal Gymnasium-style fake env that terminates quickly."""

    def __init__(self, steps_per_episode: int = 2) -> None:
        self.action_space = _FakeActionSpace()
        self._steps: int = 0
        self._steps_per_episode = steps_per_episode

    def reset(self, seed: int | None = None, **kwargs: Any):
        self._steps = 0
        return _valid_obs(), {}

    def step(self, action: Any):
        self._steps += 1
        terminated = self._steps >= self._steps_per_episode
        truncated = False
        return _valid_obs(), 1.0, terminated, truncated, {}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Config YAML factory
# ---------------------------------------------------------------------------

def _write_config(
    tmp_path: Path,
    *,
    collection_enabled: bool = True,
    world_model_enabled: bool = True,
    behavior_enabled: bool = False,
    evaluation_enabled: bool = True,
    iteration_count: int = 2,
    eval_interval: int = 1,
) -> str:
    """Write a minimal odyssey YAML config to tmp_path and return its path."""
    config_text = textwrap.dedent(f"""\
        scope:
          name: mvp
          channels: 1
          sequence_length: 50
        latent:
          num_categoricals: 32
          num_classes: 32
        replay:
          capacity: 1000
          min_sequence_length: 50
        collection:
          env_steps: 10
          action_sampling: uniform
        world_model:
          batch_size: 16
          checkpoint_interval: 1000
          checkpoint_path: {str(tmp_path / 'wm.pt').replace(chr(92), '/')}
        gate:
          loss_threshold: 0.02
          consecutive_steps: 100
        behavior:
          enabled: {str(behavior_enabled).lower()}
          imagination_horizon: 5
          lambda: 0.95
        evaluation:
          episodes: 5
          interval_iterations: {eval_interval}
        pipeline:
          iteration_count: {iteration_count}
          phases:
            collection: {str(collection_enabled).lower()}
            world_model: {str(world_model_enabled).lower()}
            behavior: {str(behavior_enabled).lower()}
            evaluation: {str(evaluation_enabled).lower()}
        baseline:
          enabled: false
    """)
    cfg_path = tmp_path / "test_odyssey.yaml"
    cfg_path.write_text(config_text)
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Monkeypatching helpers
# ---------------------------------------------------------------------------

def _fake_collection_result() -> CollectionResult:
    return CollectionResult(env_steps=10, stored=10, rejected_observations=0, storage_failures=0, episodes=0)


def _fake_eval_result(env_steps_so_far: int = 0) -> EvalResult:
    return EvalResult(avg_reward=1.0, completion_rate=0.5, cumulative_env_steps=env_steps_so_far + 5)


def _fake_wm_losses() -> WorldModelLosses:
    return WorldModelLosses(recon=0.1, kl=0.05, reward=0.01, discount=0.01, total=0.17)


# ---------------------------------------------------------------------------
# Test 1: All enabled phases (collection + world-model + evaluation); behavior gated
# ---------------------------------------------------------------------------

def test_all_enabled_phases_run_in_order(tmp_path: Path) -> None:
    """Collection runs once, world-model runs iteration_count times, evaluation
    runs every interval, behavior is skipped (gate never confirmed).

    Validates: Requirements 12.2, 18.2, 18.3
    """
    call_log: List[str] = []
    cfg_path = _write_config(
        tmp_path,
        collection_enabled=True,
        world_model_enabled=True,
        behavior_enabled=False,  # gate also blocks it regardless
        evaluation_enabled=True,
        iteration_count=2,
        eval_interval=1,
    )

    def fake_collect(env, buffer, config, **kwargs):
        call_log.append("collection")
        return _fake_collection_result()

    def fake_wm_train_step(self, step):
        call_log.append("world_model")
        return _fake_wm_losses()

    def fake_evaluate(actor, env, replay_buffer, env_steps_so_far, episodes, scope, **kwargs):
        call_log.append("evaluation")
        return _fake_eval_result(env_steps_so_far)

    def fake_gym_make(env_id, **kwargs):
        return _FakeEnv()

    with (
        patch("training.world_model_trainer.WorldModelTrainer.train_step", fake_wm_train_step),
        patch("train.collect_random_trajectories", fake_collect),
        patch("train.evaluate", fake_evaluate),
        patch("train._make_env", side_effect=_FakeEnv),
    ):
        import train
        metrics = train.run_training(cfg_path)

    # collection must appear exactly once, before any world_model call
    assert call_log.count("collection") == 1, (
        f"Expected collection to run exactly once, got {call_log.count('collection')}"
    )
    assert call_log[0] == "collection", (
        f"collection must be the first phase, got {call_log[0]!r}"
    )

    # world_model must appear iteration_count (2) times
    assert call_log.count("world_model") == 2, (
        f"Expected world_model to run 2 times, got {call_log.count('world_model')}"
    )

    # evaluation must run every iteration (interval=1), so 2 times
    assert call_log.count("evaluation") == 2, (
        f"Expected evaluation to run 2 times (every iteration), got {call_log.count('evaluation')}"
    )

    # behavior was disabled / gated — must not appear
    assert "behavior" not in call_log, "behavior must not run when disabled/gated"

    # Canonical phase order: collection first, then (world_model, evaluation) repeated
    assert call_log[0] == "collection"
    # Remaining entries should only be world_model and evaluation in per-iteration pairs
    iteration_phases = call_log[1:]
    wm_positions = [i for i, p in enumerate(iteration_phases) if p == "world_model"]
    eval_positions = [i for i, p in enumerate(iteration_phases) if p == "evaluation"]
    # world_model must come before evaluation within each iteration
    for wm_pos, ev_pos in zip(wm_positions, eval_positions):
        assert wm_pos < ev_pos, (
            f"world_model (pos {wm_pos}) must precede evaluation (pos {ev_pos})"
        )


# ---------------------------------------------------------------------------
# Test 2: Collection disabled → collection does NOT run
# ---------------------------------------------------------------------------

def test_collection_disabled_does_not_run(tmp_path: Path) -> None:
    """When pipeline.phases.collection=false, collection is never called.

    Validates: Requirements 18.3
    """
    call_log: List[str] = []
    cfg_path = _write_config(
        tmp_path,
        collection_enabled=False,
        world_model_enabled=True,
        behavior_enabled=False,
        evaluation_enabled=True,
        iteration_count=2,
        eval_interval=1,
    )

    def fake_collect(env, buffer, config, **kwargs):
        call_log.append("collection")
        return _fake_collection_result()

    def fake_wm_train_step(self, step):
        call_log.append("world_model")
        return _fake_wm_losses()

    def fake_evaluate(actor, env, replay_buffer, env_steps_so_far, episodes, scope, **kwargs):
        call_log.append("evaluation")
        return _fake_eval_result(env_steps_so_far)

    with (
        patch("train.collect_random_trajectories", fake_collect),
        patch("training.world_model_trainer.WorldModelTrainer.train_step", fake_wm_train_step),
        patch("train.evaluate", fake_evaluate),
        patch("train._make_env", side_effect=_FakeEnv),
    ):
        import train
        train.run_training(cfg_path)

    assert "collection" not in call_log, (
        "collection must NOT run when pipeline.phases.collection=false"
    )
    # Other phases still run
    assert "world_model" in call_log
    assert "evaluation" in call_log


# ---------------------------------------------------------------------------
# Test 3: World model disabled → world model does NOT run
# ---------------------------------------------------------------------------

def test_world_model_disabled_does_not_run(tmp_path: Path) -> None:
    """When pipeline.phases.world_model=false, world-model train_step is never called.

    Validates: Requirements 18.4
    """
    call_log: List[str] = []
    cfg_path = _write_config(
        tmp_path,
        collection_enabled=True,
        world_model_enabled=False,
        behavior_enabled=False,
        evaluation_enabled=True,
        iteration_count=2,
        eval_interval=1,
    )

    def fake_collect(env, buffer, config, **kwargs):
        call_log.append("collection")
        return _fake_collection_result()

    def fake_wm_train_step(self, step):
        call_log.append("world_model")
        return _fake_wm_losses()

    def fake_evaluate(actor, env, replay_buffer, env_steps_so_far, episodes, scope, **kwargs):
        call_log.append("evaluation")
        return _fake_eval_result(env_steps_so_far)

    with (
        patch("train.collect_random_trajectories", fake_collect),
        patch("training.world_model_trainer.WorldModelTrainer.train_step", fake_wm_train_step),
        patch("train.evaluate", fake_evaluate),
        patch("train._make_env", side_effect=_FakeEnv),
    ):
        import train
        train.run_training(cfg_path)

    assert "world_model" not in call_log, (
        "world_model must NOT run when pipeline.phases.world_model=false"
    )
    assert "collection" in call_log
    assert "evaluation" in call_log


# ---------------------------------------------------------------------------
# Test 4: Evaluation disabled → evaluation does NOT run
# ---------------------------------------------------------------------------

def test_evaluation_disabled_does_not_run(tmp_path: Path) -> None:
    """When pipeline.phases.evaluation=false, evaluate() is never called.

    Validates: Requirements 18.4
    """
    call_log: List[str] = []
    cfg_path = _write_config(
        tmp_path,
        collection_enabled=True,
        world_model_enabled=True,
        behavior_enabled=False,
        evaluation_enabled=False,
        iteration_count=2,
        eval_interval=1,
    )

    def fake_collect(env, buffer, config, **kwargs):
        call_log.append("collection")
        return _fake_collection_result()

    def fake_wm_train_step(self, step):
        call_log.append("world_model")
        return _fake_wm_losses()

    def fake_evaluate(actor, env, replay_buffer, env_steps_so_far, episodes, scope, **kwargs):
        call_log.append("evaluation")
        return _fake_eval_result(env_steps_so_far)

    with (
        patch("train.collect_random_trajectories", fake_collect),
        patch("training.world_model_trainer.WorldModelTrainer.train_step", fake_wm_train_step),
        patch("train.evaluate", fake_evaluate),
        patch("train._make_env", side_effect=_FakeEnv),
    ):
        import train
        train.run_training(cfg_path)

    assert "evaluation" not in call_log, (
        "evaluation must NOT run when pipeline.phases.evaluation=false"
    )
    assert "collection" in call_log
    assert "world_model" in call_log


# ---------------------------------------------------------------------------
# Test 5: World-model precedes behavior (within-iteration ordering)
# ---------------------------------------------------------------------------

def test_world_model_precedes_behavior_within_iteration(tmp_path: Path) -> None:
    """Within each iteration, world-model train_step is always called before
    any behavior training attempt.

    Validates: Requirements 12.2, 18.2
    """
    call_log: List[str] = []
    # Enable behavior in config; gate will still block it (no confirmed recon),
    # but we can verify ordering by checking the gate is consulted after WM.
    # To actually see behavior in the log, we also patch the gate to return allowed=True.
    cfg_path = _write_config(
        tmp_path,
        collection_enabled=False,
        world_model_enabled=True,
        behavior_enabled=True,
        evaluation_enabled=False,
        iteration_count=3,
        eval_interval=999,  # effectively disable evaluation
    )

    def fake_wm_train_step(self, step):
        call_log.append("world_model")
        return _fake_wm_losses()

    behavior_train_step_calls = []

    def fake_behavior_train_step(self, start_states, step):
        call_log.append("behavior")
        behavior_train_step_calls.append(step)
        return {"actor_loss": 0.1, "critic_loss": 0.05}

    def fake_gate_behavior_allowed(self, config_enabled, recon_history):
        # Always allow behavior (gate confirmed) so behavior actually runs
        return True, None

    def fake_collect(env, buffer, config, **kwargs):
        call_log.append("collection")
        return _fake_collection_result()

    with (
        patch("train.collect_random_trajectories", fake_collect),
        patch("training.world_model_trainer.WorldModelTrainer.train_step", fake_wm_train_step),
        patch("training.behavior_trainer.BehaviorTrainer.train_step", fake_behavior_train_step),
        patch("training.gate.ReconstructionGate.behavior_allowed", fake_gate_behavior_allowed),
        patch("train._make_env", side_effect=_FakeEnv),
    ):
        import train
        train.run_training(cfg_path)

    # Verify behavior actually ran (gate was opened by our patch)
    assert "behavior" in call_log, (
        "behavior must appear in call_log (gate was patched to allow it)"
    )
    assert "world_model" in call_log

    # Within each iteration, world_model must come before behavior.
    # The call_log should contain repeated (world_model, behavior) pairs.
    # Scan through the log and assert every 'behavior' entry is preceded by 'world_model'.
    for i, phase in enumerate(call_log):
        if phase == "behavior":
            # Look backward for the nearest world_model
            preceding = call_log[:i]
            assert "world_model" in preceding, (
                f"behavior at position {i} has no preceding world_model: {call_log}"
            )
            # Also verify no behavior appears before the first world_model in the sequence
            first_wm = call_log.index("world_model")
            first_bh = call_log.index("behavior")
            assert first_wm < first_bh, (
                f"First world_model (pos {first_wm}) must precede first behavior (pos {first_bh})"
            )
            break  # checked the invariant structurally

    # Detailed per-iteration ordering: reconstruct (wm, bh) pairs
    # In a 3-iteration run with no collection and no eval, we expect exactly
    # 3 world_model and 3 behavior entries alternating: wm, bh, wm, bh, wm, bh
    assert call_log.count("world_model") == 3, (
        f"Expected 3 world_model calls for 3 iterations, got {call_log.count('world_model')}"
    )
    assert call_log.count("behavior") == 3, (
        f"Expected 3 behavior calls for 3 iterations, got {call_log.count('behavior')}"
    )

    # Check strict alternation: wm always at even indices, bh always at odd indices
    for idx, phase in enumerate(call_log):
        if idx % 2 == 0:
            assert phase == "world_model", (
                f"Expected world_model at position {idx}, got {phase!r}: {call_log}"
            )
        else:
            assert phase == "behavior", (
                f"Expected behavior at position {idx}, got {phase!r}: {call_log}"
            )
