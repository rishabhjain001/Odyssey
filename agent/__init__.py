"""Odyssey agent components.

Contains the neural components of the world-model agent: Encoder, RSSM,
Decoder, Reward/Discount predictors, and the Actor/Critic behavior model.

Also exposes the typed configuration objects and shared data types so they can
be imported as ``from agent import load_config, Transition`` etc.
"""

from agent.config import (
    BaselineConfig,
    BehaviorConfig,
    CollectionConfig,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    EvaluationConfig,
    GateConfig,
    LatentConfig,
    OdysseyConfig,
    PipelineConfig,
    PipelinePhases,
    ReplayConfig,
    ScopeConfig,
    WorldModelConfig,
    load_config,
)
from agent.types import (
    CategoricalLatent,
    EvalResult,
    Frame,
    ImaginedRollout,
    LatentState,
    Transition,
    WorldModelLosses,
)

__all__ = [
    # config loader + typed config objects
    "load_config",
    "DEFAULT_CONFIG_PATH",
    "ConfigError",
    "OdysseyConfig",
    "ScopeConfig",
    "LatentConfig",
    "ReplayConfig",
    "CollectionConfig",
    "WorldModelConfig",
    "GateConfig",
    "BehaviorConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "PipelinePhases",
    "BaselineConfig",
    # shared data types
    "Frame",
    "Transition",
    "CategoricalLatent",
    "LatentState",
    "ImaginedRollout",
    "EvalResult",
    "WorldModelLosses",
]
