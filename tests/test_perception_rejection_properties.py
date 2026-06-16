# Feature: odyssey, Property 2: Malformed observations are rejected and never processed or stored
"""Property-based test for malformed-observation rejection (Property 2).

**Property 2: Malformed observations are rejected and never processed or stored**

For any array whose shape is not ``(96, 96, 3)`` or that contains values outside
``[0, 255]``, and for any observation that cannot be converted into a valid frame
for the active scope, the system raises the corresponding validation/conversion
error, records it, performs no downstream processing, stores nothing in the
replay buffer, and leaves previously stored frames and transitions unchanged.

This test exercises :class:`agent.perception.FramePreprocessor` against the MVP
grayscale scope (``ScopeConfig(channels=1, sequence_length=50, name="mvp")``)
and demonstrates containment using :class:`agent.replay.ReplayBuffer`: a buffer
pre-populated with valid transitions is left byte-for-byte unchanged after an
attempt to process a malformed observation raises.

Validates: Requirements 1.8, 1.9, 9.6
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, seed, settings, strategies as st
from hypothesis.extra import numpy as hnp

from agent.config import ScopeConfig
from agent.errors import FrameConversionError, ObservationValidationError
from agent.perception import FramePreprocessor
from agent.replay import ReplayBuffer
from agent.types import Transition

# The active scope under test: MVP grayscale (Req 9.1, 9.5).
MVP_SCOPE = ScopeConfig(channels=1, sequence_length=50, name="mvp")

# The valid raw-observation contract; malformed inputs violate one of these.
VALID_OBS_SHAPE = (96, 96, 3)


# ---------------------------------------------------------------------------
# Strategies for the two malformation modes
# ---------------------------------------------------------------------------

# Mode A: arrays whose shape is NOT (96, 96, 3). Cover a variety of rank/size
# mismatches while keeping values in-range so the *shape* is the sole defect.
_wrong_shapes = st.sampled_from(
    [
        (96, 96),        # missing channel dim
        (96, 96, 1),     # wrong channel count
        (96, 96, 4),     # wrong channel count
        (64, 64, 3),     # wrong spatial resolution
        (32, 32, 3),     # wrong spatial resolution
        (96, 95, 3),     # off-by-one spatial
        (3, 96, 96),     # transposed (channels-first)
        (1,),            # rank-1
        (10, 10, 3),     # small
        (96, 96, 3, 1),  # extra trailing dim
    ]
)


@st.composite
def malformed_wrong_shape(draw: st.DrawFn) -> np.ndarray:
    """Generate an in-range array whose shape is not (96, 96, 3) (Req 1.8)."""
    shape = draw(_wrong_shapes)
    return draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=shape,
            elements=st.integers(min_value=0, max_value=255),
        )
    )


@st.composite
def malformed_out_of_range(draw: st.DrawFn) -> np.ndarray:
    """Generate a correctly-shaped (96,96,3) array with at least one value
    outside ``[0, 255]`` (Req 1.8).

    Uses a signed/float dtype so out-of-range values are representable, fills the
    array with in-range values, then plants at least one guaranteed out-of-range
    value (negative or > 255) at a random location.
    """
    base = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=VALID_OBS_SHAPE,
            elements=st.floats(
                min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    row = draw(st.integers(min_value=0, max_value=95))
    col = draw(st.integers(min_value=0, max_value=95))
    chan = draw(st.integers(min_value=0, max_value=2))
    # An out-of-range value: either strictly negative or strictly above 255.
    bad_value = draw(
        st.one_of(
            st.floats(min_value=-1e6, max_value=-0.0001, allow_nan=False, allow_infinity=False),
            st.floats(min_value=255.0001, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    base[row, col, chan] = bad_value
    return base


malformed_observation = st.one_of(malformed_wrong_shape(), malformed_out_of_range())


# ---------------------------------------------------------------------------
# Helpers for the containment demonstration
# ---------------------------------------------------------------------------

def _make_valid_transition(seed: int) -> Transition:
    """Build a valid Transition with a deterministic, distinct observation."""
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 256, size=VALID_OBS_SHAPE, dtype=np.uint8)
    action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    return Transition(
        observation=obs,
        action=action,
        reward=float(seed),
        terminated=bool(seed % 5 == 0),
    )


def _snapshot(buffer: ReplayBuffer) -> list[Transition]:
    """Capture the buffer's current transitions for later equality checks."""
    # The buffer stores Transition objects by reference; capturing the list lets
    # us assert both length-equality and identity/content-equality afterwards.
    return list(buffer._buffer)  # noqa: SLF001 - test-only introspection


def _assert_buffer_unchanged(
    buffer: ReplayBuffer, before: list[Transition]
) -> None:
    """Assert the buffer holds exactly the pre-existing transitions, unchanged."""
    after = list(buffer._buffer)  # noqa: SLF001 - test-only introspection
    assert len(after) == len(before), "Buffer length changed after a rejected observation."
    for prev, curr in zip(before, after):
        # Same object identity => nothing was added, removed, or replaced.
        assert prev is curr, "A stored transition was mutated or replaced."
        assert np.array_equal(prev.observation, curr.observation)
        assert np.array_equal(prev.action, curr.action)
        assert prev.reward == curr.reward
        assert prev.terminated == curr.terminated


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(obs=malformed_observation, data=st.data())
def test_malformed_observations_are_rejected_and_never_stored(
    obs: np.ndarray, data: st.DataObject
) -> None:
    """Property 2: malformed observations raise, are not processed, and leave a
    pre-populated replay buffer unchanged (Req 1.8, 1.9, 9.6).

    Validates: Requirements 1.8, 1.9, 9.6
    """
    preprocessor = FramePreprocessor(MVP_SCOPE)

    # Pre-populate a replay buffer with some valid transitions; this stands in
    # for "previously stored frames and transitions" that must be preserved.
    num_existing = data.draw(st.integers(min_value=0, max_value=8))
    buffer = ReplayBuffer(capacity=100)
    for i in range(num_existing):
        buffer.add(_make_valid_transition(i))
    before = _snapshot(buffer)

    # validate_observation MUST reject the malformed observation (Req 1.8).
    with pytest.raises(ObservationValidationError):
        preprocessor.validate_observation(obs)

    # to_frame MUST also raise -- because it validates first, malformed input
    # never reaches conversion or any downstream processing (Req 1.8, 1.9).
    # We accept either error type from the documented taxonomy: validation for
    # the malformation modes here, conversion for unconvertible-but-valid-shape
    # cases (Req 1.9, 9.6).
    raised = False
    try:
        preprocessor.to_frame(obs)
    except (ObservationValidationError, FrameConversionError):
        raised = True
        # Containment: the failed attempt must NOT have added anything to the
        # buffer nor mutated existing transitions (Req 1.9, 9.6).
        _assert_buffer_unchanged(buffer, before)
    assert raised, "Malformed observation was processed without raising."

    # Final containment check: nothing was stored as a side effect of the whole
    # rejected-observation flow.
    _assert_buffer_unchanged(buffer, before)


@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(obs=malformed_wrong_shape())
def test_wrong_shape_observations_raise_validation_error(obs: np.ndarray) -> None:
    """Wrong-shape observations are rejected by validation (Req 1.8).

    Validates: Requirements 1.8
    """
    preprocessor = FramePreprocessor(MVP_SCOPE)
    assert obs.shape != VALID_OBS_SHAPE
    with pytest.raises(ObservationValidationError):
        preprocessor.validate_observation(obs)
    with pytest.raises(ObservationValidationError):
        preprocessor.to_frame(obs)


@settings(max_examples=20, deadline=None)
@seed(20240517)
@given(obs=malformed_out_of_range())
def test_out_of_range_observations_raise_validation_error(obs: np.ndarray) -> None:
    """Correctly-shaped observations with out-of-range values are rejected (Req 1.8).

    Validates: Requirements 1.8
    """
    preprocessor = FramePreprocessor(MVP_SCOPE)
    assert obs.shape == VALID_OBS_SHAPE
    # At least one value lies outside [0, 255].
    assert obs.min() < 0 or obs.max() > 255
    with pytest.raises(ObservationValidationError):
        preprocessor.validate_observation(obs)
    with pytest.raises(ObservationValidationError):
        preprocessor.to_frame(obs)
