"""Smoke tests for Odyssey project layout and configuration.

These plain pytest smoke tests verify that the project scaffolding is in place
and that the scope of the agent (grayscale MVP vs. RGB) is driven entirely by
configuration, so switching scope requires only a config edit and no code
change.

Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5, 9.1, 9.5
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import load_config

# Repository root: tests/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Project layout (Req 17.1)
# ---------------------------------------------------------------------------

REQUIRED_DIRECTORIES = [
    "agent",
    "training",
    "baseline",
    "dashboard",
    "configs",
    "scripts",
    "report",
]


@pytest.mark.parametrize("directory", REQUIRED_DIRECTORIES)
def test_required_directory_exists(directory: str) -> None:
    """Each component directory in the project layout exists (Req 17.1)."""
    path = PROJECT_ROOT / directory
    assert path.is_dir(), f"Required directory '{directory}/' is missing."


# ---------------------------------------------------------------------------
# Required files (Req 17.2, 17.3, 17.4, 17.5)
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "configs/odyssey.yaml",  # central config file (Req 17.2)
    "requirements.txt",      # dependency manifest (Req 17.3)
    ".env.example",          # environment template (Req 17.4)
    "README.md",             # project README (Req 17.5)
]


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_file_exists(filename: str) -> None:
    """Each required project file exists (Req 17.2-17.5)."""
    path = PROJECT_ROOT / filename
    assert path.is_file(), f"Required file '{filename}' is missing."


def test_required_files_are_non_empty() -> None:
    """The required project files have content, not just placeholders."""
    for filename in REQUIRED_FILES:
        path = PROJECT_ROOT / filename
        assert path.stat().st_size > 0, f"Required file '{filename}' is empty."


# ---------------------------------------------------------------------------
# Config-driven scope switching with no code edit (Req 9.1, 9.5)
# ---------------------------------------------------------------------------

def test_default_mvp_config_loads_grayscale_scope() -> None:
    """The default config loads the grayscale MVP scope (Req 9.1)."""
    config = load_config(PROJECT_ROOT / "configs" / "odyssey.yaml")
    assert config.scope.name == "mvp"
    assert config.scope.channels == 1
    assert config.scope.sequence_length == 50


RGB_CONFIG_YAML = """\
scope:
  name: rgb
  channels: 3
  sequence_length: 100
latent:
  num_categoricals: 32
  num_classes: 32
replay:
  capacity: 1000000
  min_sequence_length: 50
collection:
  env_steps: 100000
  action_sampling: uniform
world_model:
  batch_size: 16
  checkpoint_interval: 1000
  checkpoint_path: checkpoints/world_model.pt
gate:
  loss_threshold: 0.02
  consecutive_steps: 100
behavior:
  enabled: false
  imagination_horizon: 15
  lambda: 0.95
evaluation:
  episodes: 10
  interval_iterations: 5
pipeline:
  iteration_count: 100
  phases:
    collection: true
    world_model: true
    behavior: false
    evaluation: true
baseline:
  enabled: false
"""


def test_switching_scope_changes_channels_and_sequence_length(tmp_path: Path) -> None:
    """Switching the scope name in config changes channel count and sequence
    length with no code edit (Req 9.5).

    The same ``load_config`` loader reads both the default grayscale MVP config
    and a temporary RGB config; only the config file differs.
    """
    mvp_config = load_config(PROJECT_ROOT / "configs" / "odyssey.yaml")

    rgb_path = tmp_path / "rgb.yaml"
    rgb_path.write_text(RGB_CONFIG_YAML, encoding="utf-8")
    rgb_config = load_config(rgb_path)

    # The RGB scope reflects the config values without any code change.
    assert rgb_config.scope.name == "rgb"
    assert rgb_config.scope.channels == 3
    assert rgb_config.scope.sequence_length == 100

    # The two scopes differ purely because the config differs.
    assert rgb_config.scope.channels != mvp_config.scope.channels
    assert rgb_config.scope.sequence_length != mvp_config.scope.sequence_length
