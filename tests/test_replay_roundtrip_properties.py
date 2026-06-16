# Feature: odyssey, Property 4: Replay storage round-trips transition fields
"""Property-based test for Property 4 of the Odyssey design.

**Property 4: Replay storage round-trips transition fields**

*For any* sequence of transitions added to a buffer (within capacity), each
retrieved transition preserves the original observation, action, reward, and
episode-termination signal.

**Validates: Requirements 2.2**

Implementation under test: ``agent.replay.ReplayBuffer`` -- ``add(Transition)``
and the stored transitions.

Retrieval note:
    ``ReplayBuffer`` exposes ``__len__`` publicly but offers no public method
    that returns every stored transition in order (``sample_sequences`` draws
    contiguous within-episode windows with replacement and is governed by the
    ``terminated`` flags, so it is not a faithful round-trip readout). To assert
    a *storage round-trip* of the raw fields, this test reads the buffer's
    internal FIFO deque (``buffer._buffer``), which holds transitions in arrival
    order. This is deliberate introspection of a documented internal attribute
    used solely to verify the stored-field contract of Req 2.2.
"""

from __future__ import annotations

from typing import List

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.replay import ReplayBuffer
from agent.types import Transition


# ---------------------------------------------------------------------------
# Generators: varied but representative transition fields.
# ---------------------------------------------------------------------------

# Observations: small uint8 arrays standing in for raw frames. A storage
# round-trip does not depend on the (96,96,3) shape, so a representative mix of
# small shapes keeps the test fast while still exercising arbitrary arrays.
_OBS_SHAPES = st.sampled_from([(2, 2), (4,), (3, 3, 1), (8,)])


@st.composite
def _observations(draw):
    shape = draw(_OBS_SHAPES)
    return draw(
        st.builds(
            lambda flat, s: np.asarray(flat, dtype=np.uint8).reshape(s),
            st.lists(
                st.integers(min_value=0, max_value=255),
                min_size=int(np.prod(shape)),
                max_size=int(np.prod(shape)),
            ),
            st.just(shape),
        )
    )


# Actions: float arrays of length 3 ([steering, accel, brake]).
_actions = st.builds(
    lambda vals: np.asarray(vals, dtype=np.float32),
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
)

_rewards = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def _transitions(draw) -> Transition:
    return Transition(
        observation=draw(_observations()),
        action=draw(_actions),
        reward=draw(_rewards),
        terminated=draw(st.booleans()),
    )


@st.composite
def _transition_sequences(draw) -> List[Transition]:
    """A non-empty list of transitions to add (kept within buffer capacity)."""
    return draw(st.lists(_transitions(), min_size=1, max_size=40))


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(transitions=_transition_sequences())
def test_replay_storage_round_trips_transition_fields(transitions: List[Transition]):
    """Property 4: stored transitions preserve every field, in order.

    The buffer capacity is set to at least the number of transitions added, so
    no eviction occurs (FIFO eviction is Property 5's concern). Each retrieved
    transition must preserve the original observation, action, reward, and
    episode-termination signal, in arrival order.
    """
    capacity = max(len(transitions), 1)
    buffer = ReplayBuffer(capacity=capacity)

    for t in transitions:
        buffer.add(t)

    # Within capacity, the stored count equals the number added (Req 2.2).
    assert len(buffer) == len(transitions)

    stored = list(buffer._buffer)
    assert len(stored) == len(transitions)

    for original, retrieved in zip(transitions, stored):
        # Observation round-trips exactly (dtype-agnostic element equality).
        assert np.array_equal(retrieved.observation, original.observation), (
            "observation was not preserved on storage round-trip"
        )
        # Action round-trips exactly.
        assert np.array_equal(retrieved.action, original.action), (
            "action was not preserved on storage round-trip"
        )
        # Reward round-trips exactly.
        assert retrieved.reward == original.reward, (
            "reward was not preserved on storage round-trip"
        )
        # Episode-termination signal round-trips exactly.
        assert retrieved.terminated == original.terminated, (
            "terminated flag was not preserved on storage round-trip"
        )
