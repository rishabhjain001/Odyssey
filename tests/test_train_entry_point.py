# Feature: odyssey, Tasks 15.3 and 15.4: entry-point smoke tests and e2e integration
"""Smoke and integration tests for the train.py orchestrating entry point.

Task 15.3 — Smoke tests (Validates: Requirements 18.1, 7.1, 1.1):
    1. test_train_py_exists           — train.py exists at the project root.
    2. test_train_py_is_importable    — ``import train`` works; ``train.run_training`` is callable.
    3. test_world_model_trainer_has_no_sb3_import
                                      — importing training.world_model_trainer does NOT load
                                        stable_baselines3 into sys.modules.
    4. test_gymnasium_env_constructs  — gymnasium.make("CarRacing-v3", continuous=True)
                                        constructs and closes without error.
                                        (Skipped if Gymnasium or Box2D is unavailable.)

Task 15.4 — End-to-end integration on a tiny config (Validates: Requirements 12.1, 18.2):
    5. test_end_to_end_tiny_config    — run_training on a tiny config completes without error,
                                        returns a MetricsStore, and records world-model loss
                                        history entries.  The heavy Gymnasium env is replaced
                                        with a lightweight fake to keep the test fast.

Design notes
------------
- The fake env used in test 5 implements the subset of the Gymnasium API that
  ``collect_random_trajectories`` and ``evaluate`` rely on: ``reset``, ``step``,
  ``action_space.sample``, ``action_space.seed``, and ``close``.
- gymnasium.make is monkeypatched so the fake env is injected everywhere train.py
  calls ``_make_env`` (which itself calls ``gymnasium.make``).
- The tiny config writes behaviour.enabled=False and baseline.enabled=False so the
  gate is never consulted for behavior; the test validates the
  collection → world-model → evaluation path.
- 50 collection steps gives exactly one episode of 50 transitions, which is the
  minimum required for a sequence sample of length 50.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Absolute path to the project root (one level up from this file's directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _gym_and_box2d_available() -> bool:
    """Return True if gymnasium and Box2D (for CarRacing-v3) are importable."""
    try:
        import gymnasium  # noqa: F401
        import Box2D  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Task 15.3 — Smoke tests
# ---------------------------------------------------------------------------


def test_train_py_exists() -> None:
    """train.py exists at the project root (Req 18.1).

    Validates: Requirements 18.1
    """
    train_py = _PROJECT_ROOT / "train.py"
    assert train_py.is_file(), (
        f"train.py not found at project root: {train_py}"
    )


def test_train_py_is_importable() -> None:
    """``import train`` works and ``train.run_training`` is callable (Req 18.1).

    Validates: Requirements 18.1
    """
    # Ensure the project root is on sys.path so ``import train`` resolves.
    root_str = str(_PROJECT_ROOT)
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        # Remove cached module to get a clean import (idempotent across test runs).
        sys.modules.pop("train", None)
        train_mod = importlib.import_module("train")
        assert callable(getattr(train_mod, "run_training", None)), (
            "train.run_training is not callable after importing train.py"
        )
    finally:
        if inserted:
            sys.path.remove(root_str)
        sys.modules.pop("train", None)


def test_world_model_trainer_has_no_sb3_import() -> None:
    """Importing training.world_model_trainer does NOT load stable_baselines3 (Req 7.1).

    The world-model trainer must use only custom PyTorch — no Stable-Baselines3.

    Validates: Requirements 7.1
    """
    # Remove any previously loaded instance so the import is fresh.
    mods_to_clear = [k for k in sys.modules if "world_model_trainer" in k]
    for m in mods_to_clear:
        sys.modules.pop(m, None)

    # Remove SB3 from sys.modules before the import so we can detect if it appears.
    sb3_mods_before = {k for k in sys.modules if "stable_baselines3" in k}
    for m in sb3_mods_before:
        sys.modules.pop(m, None)

    importlib.import_module("training.world_model_trainer")

    sb3_present = any("stable_baselines3" in k for k in sys.modules)
    assert not sb3_present, (
        "importing training.world_model_trainer should NOT load stable_baselines3, "
        "but found it in sys.modules after import."
    )


@pytest.mark.skipif(
    not _gym_and_box2d_available(),
    reason="gymnasium or Box2D (gymnasium[box2d]) is not installed",
)
def test_gymnasium_env_constructs() -> None:
    """gymnasium.make('CarRacing-v3', continuous=True) constructs and closes (Req 1.1).

    Validates: Requirements 1.1
    """
    import gymnasium as gym  # noqa: PLC0415

    env = gym.make("CarRacing-v3", continuous=True)
    try:
        assert env is not None, "gym.make returned None"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Task 15.4 — End-to-end integration test on a tiny config
# ---------------------------------------------------------------------------


class _FakeActionSpace:
    """Fake Gymnasium action space for CarRacing-v3 (continuous Box)."""

    def sample(self) -> np.ndarray:
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    def seed(self, seed) -> None:  # noqa: ANN001
        pass  # no-op


class _FakeEnv:
    """Fake CarRacing-v3 environment that yields constant observations.

    - Returns a constant mid-grey 96×96×3 uint8 observation on every reset/step.
    - Episodes terminate after exactly 50 steps so that collection accumulates
      50 contiguous transitions (the minimum sequence length for world-model
      training in MVP scope).
    - A new episode's step counter restarts on ``reset()``.
    """

    def __init__(self) -> None:
        self.action_space = _FakeActionSpace()
        self._episode_step = 0

    @staticmethod
    def _obs() -> np.ndarray:
        return np.ones((96, 96, 3), dtype=np.uint8) * 128

    def reset(self, **kwargs):  # noqa: ANN001, ANN201
        self._episode_step = 0
        return self._obs(), {}

    def step(self, action):  # noqa: ANN001, ANN201
        self._episode_step += 1
        terminated = self._episode_step >= 50
        return self._obs(), 1.0, terminated, False, {}

    def close(self) -> None:
        pass


def test_end_to_end_tiny_config(monkeypatch, tmp_path) -> None:
    """Run the full pipeline on a tiny config without error (Req 12.1, 18.2).

    Uses a fake Gymnasium environment (monkeypatched onto gymnasium.make) so no
    real CarRacing-v3 / Box2D install is required.  The test validates that:

    - ``run_training`` returns a :class:`~training.metrics.MetricsStore`.
    - The returned store has at least one world-model loss history entry
      (confirming the world-model training phase executed).

    The config is written to ``tmp_path`` to keep the test hermetic.

    Validates: Requirements 12.1, 18.2
    """
    # ------------------------------------------------------------------ #
    # 1. Build the tiny YAML config.
    # ------------------------------------------------------------------ #
    tiny_config = {
        "scope": {
            "name": "mvp",
            "channels": 1,
            "sequence_length": 50,
        },
        "latent": {
            "num_categoricals": 32,
            "num_classes": 32,
        },
        "replay": {
            "capacity": 500,
            "min_sequence_length": 50,
        },
        "collection": {
            # 50 env steps → exactly one episode (terminates at step 50)
            # → 50 stored transitions → one sequence of length 50 available.
            "env_steps": 50,
            "action_sampling": "uniform",
        },
        "world_model": {
            "batch_size": 1,
            "checkpoint_interval": 9999,    # avoid writing checkpoints during test
            "checkpoint_path": str(tmp_path / "ckpt.pt"),
        },
        "gate": {
            "loss_threshold": 0.02,
            "consecutive_steps": 100,       # gate won't open with only 2 steps
        },
        "behavior": {
            "enabled": False,               # gate also blocks; double-disabled
            "imagination_horizon": 3,
            "lambda": 0.95,
        },
        "evaluation": {
            "episodes": 5,                  # minimum for comparison (Req 14.3)
            "interval_iterations": 1,       # evaluate every iteration
        },
        "pipeline": {
            "iteration_count": 2,
            "phases": {
                "collection": True,
                "world_model": True,
                "behavior": False,          # disabled; gate also blocks it
                "evaluation": True,
            },
        },
        "baseline": {
            "enabled": False,
        },
    }
    config_file = tmp_path / "tiny_odyssey.yaml"
    config_file.write_text(yaml.dump(tiny_config), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 2. Monkeypatch gymnasium.make to return the fake env.
    # ------------------------------------------------------------------ #
    import gymnasium as gym_module  # noqa: PLC0415

    def _fake_make(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return _FakeEnv()

    monkeypatch.setattr(gym_module, "make", _fake_make)

    # Evict any cached ``train`` module so it picks up the monkeypatched
    # gymnasium.make when it is imported below.
    sys.modules.pop("train", None)

    # ------------------------------------------------------------------ #
    # 3. Ensure the project root is on sys.path for ``import train``.
    # ------------------------------------------------------------------ #
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # ------------------------------------------------------------------ #
    # 4. Run training and assert on the returned MetricsStore.
    # ------------------------------------------------------------------ #
    import train as train_mod  # noqa: PLC0415

    metrics = train_mod.run_training(config_path=str(config_file))

    from training.metrics import MetricsStore  # noqa: PLC0415

    # run_training must return a MetricsStore instance.
    assert isinstance(metrics, MetricsStore), (
        f"run_training should return a MetricsStore, got {type(metrics)}"
    )

    # The world-model phase executed: there must be at least one loss-history
    # entry (even a "skipped" entry counts — it confirms the phase ran).
    wm_history = metrics.world_model_loss_history()
    assert len(wm_history) >= 1, (
        "Expected at least one world-model loss history entry (including any "
        f"skipped steps), but got {len(wm_history)}."
    )

    # If reconstruction losses were recorded (world-model step not skipped),
    # each value must be a finite non-negative float.
    for step, val in metrics.reconstruction_losses():
        assert isinstance(val, float), f"Reconstruction loss at step {step} is not a float: {val!r}"
        assert val >= 0.0, f"Reconstruction loss at step {step} is negative: {val}"
        assert val == val, f"Reconstruction loss at step {step} is NaN"  # NaN check

    # Clean up the cached train module so subsequent tests start fresh.
    sys.modules.pop("train", None)
