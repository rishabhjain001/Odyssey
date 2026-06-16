# Feature: odyssey, Property 3: Actions are always within the CarRacing action space
"""Property-based tests verifying that Actor and FramePreprocessor always produce
actions within the CarRacing-v2 continuous action bounds.

Property 3: Actions are always within the CarRacing action space

    For any LatentState, Actor()(state) produces a (B, 3) tensor with:
        steering  [:,0] ∈ [-1, 1]   (tanh)
        accel     [:,1] ∈ [ 0, 1]   (sigmoid)
        brake     [:,2] ∈ [ 0, 1]   (sigmoid)

    Also: FramePreprocessor.map_action clips arbitrary float32 inputs to the
    same bounds.

Validates: Requirements 1.7, 10.3
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.behavior import Actor
from agent.config import ScopeConfig
from agent.perception import FramePreprocessor
from agent.types import LatentState

# MVP latent dimensions
DETER_DIM = 256
NUM_CATEGORICALS = 32
NUM_CLASSES = 32

# Shared MVP scope for map_action tests
_MVP_SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")


@seed(20240619)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_actor_steering_in_minus_one_to_one(batch: int, torch_seed: int) -> None:
    """Actor steering output [:,0] lies within [-1, 1] for any LatentState.

    The tanh activation applied to the steering head mathematically constrains
    its output to the open interval (-1, 1), which is a subset of [-1, 1].

    Validates: Requirements 1.7, 10.3
    """
    torch.manual_seed(torch_seed)

    actor = Actor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    actor.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        action = actor(state)

    assert action.shape == (batch, 3), (
        f"Expected action shape ({batch}, 3), got {action.shape}"
    )
    assert torch.all(action[:, 0] >= -1.0), (
        f"Steering below -1: min={action[:, 0].min().item()}"
    )
    assert torch.all(action[:, 0] <= 1.0), (
        f"Steering above 1: max={action[:, 0].max().item()}"
    )


@seed(20240619)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_actor_accel_in_zero_to_one(batch: int, torch_seed: int) -> None:
    """Actor acceleration output [:,1] lies within [0, 1] for any LatentState.

    The sigmoid activation constrains accel to (0, 1), a subset of [0, 1].

    Validates: Requirements 1.7, 10.3
    """
    torch.manual_seed(torch_seed)

    actor = Actor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    actor.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        action = actor(state)

    assert torch.all(action[:, 1] >= 0.0), (
        f"Accel below 0: min={action[:, 1].min().item()}"
    )
    assert torch.all(action[:, 1] <= 1.0), (
        f"Accel above 1: max={action[:, 1].max().item()}"
    )


@seed(20240619)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_actor_brake_in_zero_to_one(batch: int, torch_seed: int) -> None:
    """Actor brake output [:,2] lies within [0, 1] for any LatentState.

    The sigmoid activation constrains brake to (0, 1), a subset of [0, 1].

    Validates: Requirements 1.7, 10.3
    """
    torch.manual_seed(torch_seed)

    actor = Actor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    actor.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        action = actor(state)

    assert torch.all(action[:, 2] >= 0.0), (
        f"Brake below 0: min={action[:, 2].min().item()}"
    )
    assert torch.all(action[:, 2] <= 1.0), (
        f"Brake above 1: max={action[:, 2].max().item()}"
    )


@seed(20240619)
@settings(max_examples=20, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_actor_full_action_shape_and_bounds(batch: int, torch_seed: int) -> None:
    """Actor produces (B, 3) tensor with all three components in bounds simultaneously.

    Combines shape, steering, accel, and brake checks into a single property.

    Validates: Requirements 1.7, 10.3
    """
    torch.manual_seed(torch_seed)

    actor = Actor(
        deter_dim=DETER_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
    )
    actor.eval()

    state = LatentState(
        deter=torch.randn(batch, DETER_DIM),
        stoch=torch.randn(batch, NUM_CATEGORICALS, NUM_CLASSES),
    )

    with torch.no_grad():
        action = actor(state)

    # Shape
    assert action.shape == (batch, 3)

    # Bounds via torch.all (as specified in the task)
    assert torch.all(action[:, 0] >= -1.0) and torch.all(action[:, 0] <= 1.0), (
        "Steering out of [-1, 1]"
    )
    assert torch.all(action[:, 1] >= 0.0) and torch.all(action[:, 1] <= 1.0), (
        "Accel out of [0, 1]"
    )
    assert torch.all(action[:, 2] >= 0.0) and torch.all(action[:, 2] <= 1.0), (
        "Brake out of [0, 1]"
    )


@seed(20240619)
@settings(max_examples=20, deadline=None)
@given(
    steering=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    accel=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    brake=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_frame_preprocessor_map_action_clips_to_env_bounds(
    steering: float,
    accel: float,
    brake: float,
) -> None:
    """FramePreprocessor.map_action clips arbitrary inputs to the CarRacing bounds.

    For any floating-point steering/accel/brake values, map_action must return
    an array with steering ∈ [-1,1], accel ∈ [0,1], brake ∈ [0,1].

    Validates: Requirements 1.7
    """
    preprocessor = FramePreprocessor(_MVP_SCOPE)
    raw = np.array([steering, accel, brake], dtype=np.float32)

    mapped = preprocessor.map_action(raw)

    assert mapped.shape == (3,), f"Expected shape (3,), got {mapped.shape}"
    assert mapped.dtype == np.float32, f"Expected float32, got {mapped.dtype}"

    assert -1.0 <= float(mapped[0]) <= 1.0, (
        f"Steering {mapped[0]} not in [-1, 1]"
    )
    assert 0.0 <= float(mapped[1]) <= 1.0, (
        f"Accel {mapped[1]} not in [0, 1]"
    )
    assert 0.0 <= float(mapped[2]) <= 1.0, (
        f"Brake {mapped[2]} not in [0, 1]"
    )
