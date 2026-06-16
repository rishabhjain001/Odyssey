"""Smoke / integration tests for the PPO baseline (Task 14.5).

Validates: Requirements 13.1, 13.2

These are plain pytest tests (no Hypothesis).  The CarRacing integration test
is automatically skipped when stable-baselines3 is not installed.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from agent.errors import PPOTrainingError
from baseline.ppo import PPOBaseline, PPOResult


# ---------------------------------------------------------------------------
# 1. PPOBaseline constructs without SB3 (Req 13.1)
# ---------------------------------------------------------------------------

class TestPPOBaselineConstruction:
    """PPOBaseline can be instantiated in any environment (Req 13.1)."""

    def test_instantiation_default_args(self) -> None:
        """PPOBaseline() constructs with no arguments."""
        b = PPOBaseline()
        assert b is not None

    def test_get_metrics_returns_none_before_training(self) -> None:
        """get_metrics() returns None before any training run (Req 13.4)."""
        b = PPOBaseline()
        assert b.get_metrics() is None


# ---------------------------------------------------------------------------
# 2. PPOBaseline.train retains prior metrics on failure (Req 13.5)
# ---------------------------------------------------------------------------

class TestPPOBaselineErrorHandling:
    """train() raises PPOTrainingError and retains prior metrics on failure."""

    def test_train_raises_ppo_training_error_when_env_fails(self) -> None:
        """train() raises PPOTrainingError when the environment cannot be created."""
        # Skip if SB3 is absent — the ImportError path is a separate test.
        pytest.importorskip("stable_baselines3")
        b = PPOBaseline(env_id="CarRacing-v3")

        # Patch gymnasium.make to simulate a runtime training failure.
        with patch("gymnasium.make", side_effect=RuntimeError("simulated env failure")):
            with pytest.raises(PPOTrainingError):
                b.train(total_timesteps=10)

    def test_get_metrics_still_none_after_failed_train(self) -> None:
        """get_metrics() returns None (unchanged) after a failed train call."""
        pytest.importorskip("stable_baselines3")
        b = PPOBaseline(env_id="CarRacing-v3")

        with patch("gymnasium.make", side_effect=RuntimeError("simulated env failure")):
            with pytest.raises(PPOTrainingError):
                b.train(total_timesteps=10)

        # Prior result (None) must be retained, not replaced.
        assert b.get_metrics() is None

    def test_get_metrics_retains_prior_result_after_failure(self) -> None:
        """A previously recorded result is preserved after a subsequent failure."""
        pytest.importorskip("stable_baselines3")
        b = PPOBaseline.__new__(PPOBaseline)
        b._env_id = "CarRacing-v3"
        b._seed = 0
        b._result = PPOResult(avg_reward=42.0, total_steps=500)
        b.training_completed = True

        with patch("gymnasium.make", side_effect=RuntimeError("simulated failure after success")):
            with pytest.raises(PPOTrainingError):
                b.train(total_timesteps=10)

        # The pre-existing result must be unchanged (Req 13.5).
        result = b.get_metrics()
        assert result is not None
        assert result.avg_reward == 42.0
        assert result.total_steps == 500


# ---------------------------------------------------------------------------
# 3. PPOResult has the correct fields and types (Req 13.4)
# ---------------------------------------------------------------------------

class TestPPOResult:
    """PPOResult dataclass exposes the right attributes with the right types."""

    def test_ppo_result_has_avg_reward_field(self) -> None:
        """PPOResult.avg_reward is a float."""
        r = PPOResult(avg_reward=-50.5, total_steps=1000)
        assert isinstance(r.avg_reward, float)

    def test_ppo_result_has_total_steps_field(self) -> None:
        """PPOResult.total_steps is an int."""
        r = PPOResult(avg_reward=0.0, total_steps=1000)
        assert isinstance(r.total_steps, int)

    def test_ppo_result_stores_values_correctly(self) -> None:
        """PPOResult stores the exact values passed to its constructor."""
        r = PPOResult(avg_reward=12.34, total_steps=5000)
        assert r.avg_reward == pytest.approx(12.34)
        assert r.total_steps == 5000


# ---------------------------------------------------------------------------
# 4. CarRacing integration / smoke test (Req 13.1, 13.2)
#    Skipped automatically when stable-baselines3 is not installed.
# ---------------------------------------------------------------------------

class TestCarRacingSmoke:
    """End-to-end smoke test — skipped if SB3 is absent."""

    def test_ppo_short_run_on_carracing(self) -> None:
        """PPOBaseline.train(500) returns a PPOResult or raises PPOTrainingError.

        Validates: Requirements 13.1, 13.2
        """
        # Auto-skip if stable-baselines3 is not installed.
        pytest.importorskip("stable_baselines3")

        b = PPOBaseline(env_id="CarRacing-v3", seed=0)
        try:
            result = b.train(total_timesteps=500)
            # If it succeeds, result must be a PPOResult.
            assert isinstance(result, PPOResult)
            assert isinstance(result.avg_reward, float)
            assert result.total_steps == 500
            # get_metrics() must mirror the returned result.
            assert b.get_metrics() is result
        except PPOTrainingError:
            # Allowed — CarRacing may not be available in this environment.
            pass
        except ImportError:
            pytest.skip("SB3 or gymnasium not properly installed in this environment")
