# Feature: odyssey, Property 1: Preprocessing produces scope-correct, normalized frames
"""Property-based tests for :class:`agent.perception.FramePreprocessor`.

This module implements design Correctness Property 1:

    Property 1: Preprocessing produces scope-correct, normalized frames

    For any valid 96x96x3 observation and active scope, ``to_frame`` produces a
    ``float32`` array of shape ``(channels, 64, 64)`` where ``channels`` matches
    the active scope (1 for MVP grayscale, 3 for RGB), the spatial dimensions are
    exactly 64x64, and every pixel value lies in ``[0.0, 1.0]``.

Validates: Requirements 1.3, 1.4, 1.5, 1.6, 9.2, 9.3
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from agent.config import ScopeConfig
from agent.perception import FramePreprocessor

# The two active scopes covered by Property 1: MVP grayscale (1 channel) and
# RGB (3 channels). Both must produce 64x64 normalized frames.
MVP_SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")
RGB_SCOPE = ScopeConfig(channels=3, sequence_length=100, name="rgb")

# Raw CarRacing-v2 observation contract: (96, 96, 3) integers in [0, 255].
_valid_observations = hnp.arrays(
    dtype=np.uint8,
    shape=(96, 96, 3),
    elements=st.integers(min_value=0, max_value=255),
)


@pytest.mark.parametrize("scope", [MVP_SCOPE, RGB_SCOPE], ids=["mvp", "rgb"])
@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(obs=_valid_observations)
def test_to_frame_produces_scope_correct_normalized_frames(
    scope: ScopeConfig, obs: np.ndarray
) -> None:
    """``to_frame`` output is float32, shape (channels, 64, 64), values in [0, 1].

    Validates: Requirements 1.3, 1.4, 1.5, 1.6, 9.2, 9.3
    """
    preprocessor = FramePreprocessor(scope)

    frame = preprocessor.to_frame(obs)

    # dtype is float32 (Req 1.6, 9.2).
    assert frame.dtype == np.float32

    # Shape is exactly (scope.channels, 64, 64): channel count matches the
    # active scope (Req 1.4, 1.5) and spatial dims are exactly 64x64 (Req 1.3).
    assert frame.shape == (scope.channels, 64, 64)

    # Every pixel value lies in [0.0, 1.0] (Req 1.6, 9.3).
    assert np.all(frame >= 0.0)
    assert np.all(frame <= 1.0)
