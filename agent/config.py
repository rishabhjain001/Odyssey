"""YAML configuration loading and typed configuration objects for Odyssey.

This module loads the central Odyssey configuration file (default
``configs/odyssey.yaml``, Req 17.2) and produces a tree of typed, frozen
dataclasses. Scope-dependent parameters -- the frame channel count and the
training sequence length -- are read entirely from configuration (Req 9.1), so
progressing from the grayscale MVP to RGB requires only editing the config file
and never any code change (Req 9.5).

Usage::

    from agent.config import load_config

    config = load_config()                 # loads configs/odyssey.yaml
    config = load_config("configs/rgb.yaml")
    channels = config.scope.channels       # 1 (mvp) or 3 (rgb), from config
    seq_len = config.scope.sequence_length

The loader validates the numeric bounds named in the requirements so that an
invalid config fails fast with a clear message rather than surfacing as an
obscure error deep in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# The repository-relative default location of the central config file (Req 17.2).
DEFAULT_CONFIG_PATH = "configs/odyssey.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or out of bounds."""


# ---------------------------------------------------------------------------
# Typed configuration sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopeConfig:
    """Active scope parameters, driven entirely by config (Req 9.1, 9.5).

    Attributes:
        channels: Frame channel count -- 1 for the MVP grayscale scope, 3 for
            the RGB scope.
        sequence_length: Training sequence length -- exactly 50 for the MVP, and
            in ``(50, 200]`` for RGB.
        name: Human-readable scope name (``mvp`` or ``rgb``).
    """

    channels: int
    sequence_length: int
    name: str = "mvp"


@dataclass(frozen=True)
class LatentConfig:
    """Discrete categorical latent shape (Req 3.1, 4.2)."""

    num_categoricals: int = 32
    num_classes: int = 32


@dataclass(frozen=True)
class ReplayConfig:
    """Replay buffer configuration (Req 2.7)."""

    capacity: int = 1_000_000
    min_sequence_length: int = 50


@dataclass(frozen=True)
class CollectionConfig:
    """Random trajectory collection configuration (Req 2.1, 2.9)."""

    env_steps: int = 100_000
    action_sampling: str = "uniform"


@dataclass(frozen=True)
class WorldModelConfig:
    """World-model training configuration (Req 7.2, 7.6)."""

    batch_size: int = 16
    checkpoint_interval: int = 1000
    checkpoint_path: str = "checkpoints/world_model.pt"


@dataclass(frozen=True)
class GateConfig:
    """Reconstruction-gate configuration (Reconstruction_Confirmed_Condition)."""

    loss_threshold: float = 0.02
    consecutive_steps: int = 100


@dataclass(frozen=True)
class BehaviorConfig:
    """Behavior-model configuration; disabled by default (Req 8.4, 10.5).

    Note: the YAML key ``lambda`` maps to the ``lam`` field because ``lambda``
    is a reserved Python keyword.
    """

    enabled: bool = False
    imagination_horizon: int = 15
    lam: float = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration (Req 11.1, 14.3)."""

    episodes: int = 10
    interval_iterations: int = 5


@dataclass(frozen=True)
class PipelinePhases:
    """Per-phase enable flags for the orchestrating pipeline (Req 18.3)."""

    collection: bool = True
    world_model: bool = True
    behavior: bool = False
    evaluation: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Iterative pipeline configuration (Req 12.3, 18.2)."""

    iteration_count: int = 100
    phases: PipelinePhases = field(default_factory=PipelinePhases)


@dataclass(frozen=True)
class BaselineConfig:
    """PPO baseline configuration; disabled by default (Req 13)."""

    enabled: bool = False


@dataclass(frozen=True)
class OdysseyConfig:
    """Top-level typed Odyssey configuration object.

    Composes every configuration section. Constructed by :func:`load_config`
    from the central YAML file.
    """

    scope: ScopeConfig
    latent: LatentConfig
    replay: ReplayConfig
    collection: CollectionConfig
    world_model: WorldModelConfig
    gate: GateConfig
    behavior: BehaviorConfig
    evaluation: EvaluationConfig
    pipeline: PipelineConfig
    baseline: BaselineConfig


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return ``data[key]`` as a dict, defaulting to empty when absent."""
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section '{key}' must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _build_scope(raw: Mapping[str, Any]) -> ScopeConfig:
    """Build and validate the active ScopeConfig from raw config (Req 9.1, 9.5)."""
    name = str(raw.get("name", "mvp"))
    if "channels" not in raw:
        raise ConfigError("scope.channels is required (1 for grayscale, 3 for RGB).")
    if "sequence_length" not in raw:
        raise ConfigError("scope.sequence_length is required.")
    channels = int(raw["channels"])
    sequence_length = int(raw["sequence_length"])

    if channels not in (1, 3):
        raise ConfigError(f"scope.channels must be 1 (grayscale) or 3 (RGB), got {channels}.")
    # MVP requires exactly 50; RGB allows (50, 200] (Req 2.6, 9.4).
    if name == "mvp":
        if sequence_length != 50:
            raise ConfigError(
                f"scope.sequence_length must be 50 for the mvp scope, got {sequence_length}."
            )
    else:
        if not 50 < sequence_length <= 200:
            raise ConfigError(
                "scope.sequence_length for non-mvp scope must be in (50, 200], "
                f"got {sequence_length}."
            )
    return ScopeConfig(channels=channels, sequence_length=sequence_length, name=name)


def _build_replay(raw: Mapping[str, Any]) -> ReplayConfig:
    capacity = int(raw.get("capacity", 1_000_000))
    if not 1 <= capacity <= 10_000_000:
        raise ConfigError(f"replay.capacity must be in [1, 10_000_000], got {capacity}.")
    min_seq = int(raw.get("min_sequence_length", 50))
    return ReplayConfig(capacity=capacity, min_sequence_length=min_seq)


def _build_collection(raw: Mapping[str, Any]) -> CollectionConfig:
    env_steps = int(raw.get("env_steps", 100_000))
    if not 1 <= env_steps <= 100_000_000:
        raise ConfigError(f"collection.env_steps must be in [1, 100_000_000], got {env_steps}.")
    return CollectionConfig(
        env_steps=env_steps,
        action_sampling=str(raw.get("action_sampling", "uniform")),
    )


def _build_evaluation(raw: Mapping[str, Any]) -> EvaluationConfig:
    episodes = int(raw.get("episodes", 10))
    # Comparisons require at least 5 evaluation episodes (Req 14.3).
    if episodes < 5:
        raise ConfigError(f"evaluation.episodes must be >= 5 for comparison, got {episodes}.")
    return EvaluationConfig(
        episodes=episodes,
        interval_iterations=int(raw.get("interval_iterations", 5)),
    )


def _build_pipeline(raw: Mapping[str, Any]) -> PipelineConfig:
    phases_raw = raw.get("phases") or {}
    if not isinstance(phases_raw, Mapping):
        raise ConfigError("pipeline.phases must be a mapping.")
    phases = PipelinePhases(
        collection=bool(phases_raw.get("collection", True)),
        world_model=bool(phases_raw.get("world_model", True)),
        behavior=bool(phases_raw.get("behavior", False)),
        evaluation=bool(phases_raw.get("evaluation", True)),
    )
    return PipelineConfig(
        iteration_count=int(raw.get("iteration_count", 100)),
        phases=phases,
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> OdysseyConfig:
    """Load and validate the Odyssey configuration from ``path``.

    Reads the YAML config file (default ``configs/odyssey.yaml``) and returns a
    fully typed :class:`OdysseyConfig`. The active scope's channel count and
    sequence length are read directly from the file (Req 9.1), so switching the
    scope (e.g. ``mvp`` -> ``rgb``) needs only a config edit (Req 9.5).

    Args:
        path: Path to the YAML config file.

    Returns:
        A populated :class:`OdysseyConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If required fields are missing or values are out of bounds.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("Top-level config must be a mapping.")

    latent_raw = _section(raw, "latent")
    world_model_raw = _section(raw, "world_model")
    gate_raw = _section(raw, "gate")
    behavior_raw = _section(raw, "behavior")
    baseline_raw = _section(raw, "baseline")

    return OdysseyConfig(
        scope=_build_scope(_section(raw, "scope")),
        latent=LatentConfig(
            num_categoricals=int(latent_raw.get("num_categoricals", 32)),
            num_classes=int(latent_raw.get("num_classes", 32)),
        ),
        replay=_build_replay(_section(raw, "replay")),
        collection=_build_collection(_section(raw, "collection")),
        world_model=WorldModelConfig(
            batch_size=int(world_model_raw.get("batch_size", 16)),
            checkpoint_interval=int(world_model_raw.get("checkpoint_interval", 1000)),
            checkpoint_path=str(world_model_raw.get("checkpoint_path", "checkpoints/world_model.pt")),
        ),
        gate=GateConfig(
            loss_threshold=float(gate_raw.get("loss_threshold", 0.02)),
            consecutive_steps=int(gate_raw.get("consecutive_steps", 100)),
        ),
        behavior=BehaviorConfig(
            enabled=bool(behavior_raw.get("enabled", False)),
            imagination_horizon=int(behavior_raw.get("imagination_horizon", 15)),
            lam=float(behavior_raw.get("lambda", 0.95)),
        ),
        evaluation=_build_evaluation(_section(raw, "evaluation")),
        pipeline=_build_pipeline(_section(raw, "pipeline")),
        baseline=BaselineConfig(enabled=bool(baseline_raw.get("enabled", False))),
    )
