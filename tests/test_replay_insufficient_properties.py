# Feature: odyssey, Property 7: Insufficient data raises an error and leaves the buffer unchanged
"""Property-based test for :class:`agent.replay.ReplayBuffer` insufficient-data handling.

This module implements design Correctness Property 7:

    Property 7: Insufficient data raises an error and leaves the buffer unchanged

    For any sample request whose batch size or sequence length exceeds what the
    buffer can currently provide (including a configured sequence length that
    exceeds the longest contiguous within-episode run), the buffer raises an
    insufficient-data error, withholds the batch, and leaves its contents
    unchanged.

Two unsatisfiable regimes are exercised:

    * Case A -- the requested ``seq_len`` exceeds ``longest_contiguous_run()`` so
      no valid window exists (short episodes vs. a large requested length).
    * Case B -- the requested ``batch_size`` exceeds the number of stored
      transitions (including the empty-buffer case).

In both regimes the call must raise :class:`InsufficientDataError`, and the
buffer's length and exact contents (the ``_buffer`` deque) must be identical
before and after the failed call.

Validates: Requirements 2.10, 9.7
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from agent.errors import InsufficientDataError
from agent.replay import ReplayBuffer
from agent.types import Transition

# A capacity comfortably larger than the generated transition counts, so FIFO
# eviction never interferes with the states we construct.
_CAPACITY = 1000


def _make_transition(index: int, terminated: bool) -> Transition:
    """Build a small, uniquely-identifiable transition.

    ``index`` is encoded into the observation/action/reward so that snapshot
    comparison can detect any reordering or mutation, not just length changes.
    """
    return Transition(
        observation=np.full((2, 2), float(index), dtype=np.float32),
        action=np.array([float(index), 0.0, 0.0], dtype=np.float32),
        reward=float(index),
        terminated=terminated,
    )


def _build_buffer(terminated_flags: List[bool]) -> ReplayBuffer:
    """Construct a buffer whose episode structure follows ``terminated_flags``."""
    buffer = ReplayBuffer(_CAPACITY)
    for i, term in enumerate(terminated_flags):
        buffer.add(_make_transition(i, term))
    return buffer


def _assert_unchanged(
    buffer: ReplayBuffer,
    snapshot_len: int,
    snapshot_contents: List[Transition],
) -> None:
    """Assert the buffer length and exact contents are unchanged by identity."""
    assert len(buffer) == snapshot_len
    current = list(buffer._buffer)
    assert len(current) == len(snapshot_contents)
    # The stored Transition objects must be the very same instances, in order.
    for stored, original in zip(current, snapshot_contents):
        assert stored is original


# Episode structure: a list of per-transition ``terminated`` flags. Lengths up
# to 30 give a rich variety of run lengths and boundary placements.
_terminated_flags = st.lists(st.booleans(), min_size=1, max_size=30)


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(
    terminated_flags=_terminated_flags,
    extra_seq=st.integers(min_value=1, max_value=20),
    batch_choice=st.integers(min_value=1, max_value=20),
)
def test_seq_len_exceeds_longest_run_raises_and_preserves_buffer(
    terminated_flags: List[bool], extra_seq: int, batch_choice: int
) -> None:
    """Case A: ``seq_len`` > longest contiguous run is unsatisfiable.

    The buffer is non-empty and the batch size is satisfiable (<= stored count),
    so the failure is driven purely by the sequence length exceeding the longest
    contiguous within-episode run.

    Validates: Requirements 2.10, 9.7
    """
    buffer = _build_buffer(terminated_flags)
    n = len(buffer)
    assume(n >= 1)

    longest = buffer.longest_contiguous_run()
    # Guarantee unsatisfiability via sequence length.
    seq_len = longest + extra_seq
    assert seq_len > longest
    # Keep the batch size satisfiable so seq_len is the sole cause.
    batch_size = ((batch_choice - 1) % n) + 1
    assert batch_size <= n

    snapshot_len = len(buffer)
    snapshot_contents = list(buffer._buffer)

    with pytest.raises(InsufficientDataError):
        buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)

    _assert_unchanged(buffer, snapshot_len, snapshot_contents)


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(
    terminated_flags=st.lists(st.booleans(), min_size=0, max_size=30),
    extra_batch=st.integers(min_value=1, max_value=20),
    seq_choice=st.integers(min_value=1, max_value=10),
)
def test_batch_size_exceeds_stored_count_raises_and_preserves_buffer(
    terminated_flags: List[bool], extra_batch: int, seq_choice: int
) -> None:
    """Case B: ``batch_size`` > stored transition count is unsatisfiable.

    Covers the empty-buffer case (when ``terminated_flags`` is empty) as well as
    partially-filled buffers. The requested batch size always exceeds the number
    of stored transitions, so no batch can be provided.

    Validates: Requirements 2.10, 9.7
    """
    buffer = _build_buffer(terminated_flags)
    n = len(buffer)

    # Guarantee unsatisfiability via batch size.
    batch_size = n + extra_batch
    assert batch_size > n

    snapshot_len = len(buffer)
    snapshot_contents = list(buffer._buffer)

    with pytest.raises(InsufficientDataError):
        buffer.sample_sequences(batch_size=batch_size, seq_len=seq_choice)

    _assert_unchanged(buffer, snapshot_len, snapshot_contents)
