"""Unit tests for default configuration values — Req 8.4.

Verifies that the default config file (configs/odyssey.yaml) leaves
Behavior_Model training disabled, and that BehaviorConfig's built-in default
also has enabled=False.

Validates: Requirements 8.4
"""

from pathlib import Path

import pytest

from agent.config import BehaviorConfig, load_config
from training.gate import ReconstructionGate
from agent.config import GateConfig

# Path to the project root config, relative to the repository root.
_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "odyssey.yaml"


@pytest.fixture(scope="module")
def default_config():
    """Load the default odyssey.yaml config once for all tests in this module."""
    return load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Tests against configs/odyssey.yaml
# ---------------------------------------------------------------------------

def test_behavior_enabled_is_false_in_default_config(default_config):
    """config.behavior.enabled must be False in the default config (Req 8.4)."""
    assert default_config.behavior.enabled is False


def test_pipeline_phases_behavior_is_false_in_default_config(default_config):
    """config.pipeline.phases.behavior must be False in the default config."""
    assert default_config.pipeline.phases.behavior is False


def test_baseline_enabled_is_false_in_default_config(default_config):
    """config.baseline.enabled must be False in the default config."""
    assert default_config.baseline.enabled is False


# ---------------------------------------------------------------------------
# Structural default on BehaviorConfig dataclass
# ---------------------------------------------------------------------------

def test_behavior_config_default_enabled_is_false():
    """BehaviorConfig() constructed with no args must have enabled=False (Req 8.4)."""
    cfg = BehaviorConfig()
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# ReconstructionGate silently rejects when behavior is config-disabled
# ---------------------------------------------------------------------------

def test_reconstruction_gate_behavior_not_allowed_when_config_disabled():
    """gate.behavior_allowed(config_enabled=False, recon_history=[]) returns (False, None).

    When behavior is disabled via config the gate must return (False, None) —
    no exception raised, no error message emitted (Req 8.4).
    """
    gate = ReconstructionGate(GateConfig(loss_threshold=0.02, consecutive_steps=100))
    allowed, err = gate.behavior_allowed(config_enabled=False, recon_history=[])
    assert allowed is False
    assert err is None
