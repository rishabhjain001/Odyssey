# Feature: odyssey, Property 5: Replay capacity bound with FIFO eviction
"""Property-based test for :class:`agent.replay.ReplayBuffer` capacity/eviction.

This module implements design Correctness Property 5:

    Property 5: Replay capacity bound with FIFO eviction

    For any sequence of additions to a buffer of configured capacity ``C``
    (with ``1 <= C <= 10,000,000``), the stored count never exceeds ``C``, and
    when additions exceed ``C`` the buffer retains exactly the most recent ``C``
    transitions, discarding the oldest first.

Validates: Requirements 2.7
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.replay import ReplayBuffer
from agent.types import Transition


def _make_transition(index: int) -> Transition:
    """Build a transition tagged with ``index`` so retention can be verified.

    The integer ``index`` is encoded in the reward (and mirrored in the
    observation) so that, after a sequence of additions, we can recover exactly
    which transitions the buffer retained and in what order.
    """
    return Transition(
        observation=np.full((2, 2), index, dtype=np.int64),
        action=np.zeros(3, dtype=np.float32),
        reward=float(index),
        terminated=False,
    )


# A modest capacity keeps the structural FIFO/bound property fast to check; the
# bound itself is independent of the absolute capacity value (Req 2.7 permits up
# to 10,000,000, but the invariant holds at small C).
@seed(20240608)
@settings(max_examples=20, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=50),
    num_adds=st.integers(min_value=0, max_value=120),
)
def test_capacity_bound_with_fifo_eviction(capacity: int, num_adds: int) -> None:
    """Stored count never exceeds C; on overflow the most recent C are kept in order.

    Validates: Requirements 2.7
    """
    buffer = ReplayBuffer(capacity)

    for index in range(num_adds):
        buffer.add(_make_transition(index))
        # Invariant after every addition: the stored count never exceeds C.
        assert len(buffer) <= capacity

    expected_count = min(num_adds, capacity)
    assert len(buffer) == expected_count

    # The transitions the buffer retained, identified by their reward tag.
    retained = [int(t.reward) for t in buffer._buffer]

    if num_adds <= capacity:
        # Nothing was evicted: every added transition is retained in order.
        assert retained == list(range(num_adds))
    else:
        # Overflow: exactly the most recent C indices remain, oldest discarded
        # first, preserving insertion order.
        expected_recent = list(range(num_adds - capacity, num_adds))
        assert retained == expected_recent
