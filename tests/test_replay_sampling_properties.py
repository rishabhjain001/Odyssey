# Feature: odyssey, Property 6: Sampled sequences are contiguous, within-episode, and correct length
"""Property-based test for Property 6 of the Odyssey design.

**Property 6: Sampled sequences are contiguous, within-episode, and correct length**

*For any* buffer state and any valid sample request of sequence length ``L``
(the configured length: exactly 50 for MVP and in ``(50, 200]`` for RGB), every
returned sequence contains exactly ``L`` transitions drawn contiguously from a
single episode in stored order, and contains no episode-termination boundary
except possibly at its final element.

**Validates: Requirements 2.4, 2.5, 2.6, 7.2, 9.4**

Implementation under test: ``agent.replay.ReplayBuffer.sample_sequences``.

Strategy notes:
    * Each transition is tagged with a unique, increasing global index stored in
      its ``reward`` field. This lets the test confirm that a returned sequence
      is *contiguous in stored order* (its indices increase by exactly 1) and
      lies within a *single episode* (no ``terminated`` flag before the final
      element).
    * The buffer is populated with several episodes of varying length whose
      ``terminated`` flags partition it into episodes. At least one episode is
      forced to be long enough for the requested sequence length, so every
      generated scenario exercises the success path of ``sample_sequences``
      (insufficient-data behavior is covered by task 3.5).
    * The property is length-generic: it holds for any valid ``L``. For test
      speed the generated scenarios use a small ``L`` in ``2..10``; a separate
      explicit test exercises ``L == 50`` (the MVP configured length) against a
      known long episode to show the property holds at the configured length.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from agent.replay import ReplayBuffer
from agent.types import Transition


def _build_buffer(ep_lengths: List[int]) -> Tuple[ReplayBuffer, int]:
    """Build a buffer of episodes with the given lengths.

    Each transition's ``reward`` carries its global stored index (0, 1, 2, ...)
    so contiguity and stored order can be verified. The last transition of every
    episode has ``terminated=True``; all others ``terminated=False``.

    Returns the populated buffer and the total number of transitions stored.
    """
    total = sum(ep_lengths)
    # Capacity comfortably exceeds the total so no eviction occurs and the
    # global index equals the stored position (Req 2.7 eviction is out of scope
    # here; covered by Property 5).
    buffer = ReplayBuffer(capacity=max(total, 1))

    global_index = 0
    for length in ep_lengths:
        for step in range(length):
            is_last = step == length - 1
            buffer.add(
                Transition(
                    observation=np.full((4,), global_index, dtype=np.int64),
                    action=np.zeros((3,), dtype=np.float32),
                    reward=float(global_index),
                    terminated=is_last,
                )
            )
            global_index += 1
    return buffer, total


@st.composite
def _sampling_scenarios(draw):
    """Generate (ep_lengths, seq_len, batch_size) on the success path.

    Guarantees feasibility: at least one episode is long enough to contain a
    contiguous window of length ``seq_len`` and the batch size never exceeds the
    number of stored transitions.
    """
    seq_len = draw(st.integers(min_value=2, max_value=10))

    n_eps = draw(st.integers(min_value=1, max_value=5))
    ep_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=15),
            min_size=n_eps,
            max_size=n_eps,
        )
    )

    # Force one episode to be at least seq_len long so a valid window exists.
    long_idx = draw(st.integers(min_value=0, max_value=n_eps - 1))
    ep_lengths[long_idx] = max(ep_lengths[long_idx], seq_len)

    total = sum(ep_lengths)
    # sample_sequences requires the buffer to hold at least batch_size
    # transitions; keep the request on the success path.
    batch_size = draw(st.integers(min_value=1, max_value=min(6, total)))

    return ep_lengths, seq_len, batch_size


def _assert_valid_sequence(seq: List[Transition], seq_len: int) -> None:
    """Assert one returned sequence satisfies Property 6."""
    # Correct length (Req 2.6, 9.4).
    assert len(seq) == seq_len, f"expected length {seq_len}, got {len(seq)}"

    indices = [int(round(t.reward)) for t in seq]

    # Contiguous and in stored order: indices increase by exactly 1 (Req 2.4).
    for earlier, later in zip(indices, indices[1:]):
        assert later - earlier == 1, (
            f"sequence is not contiguous in stored order: {indices}"
        )

    # Within a single episode: no termination boundary except possibly the
    # final element (Req 2.5).
    for position, transition in enumerate(seq):
        if position < seq_len - 1:
            assert not transition.terminated, (
                f"termination boundary at position {position} of {seq_len}; "
                f"sequence crosses an episode boundary (indices {indices})"
            )


@seed(20240607)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=_sampling_scenarios())
def test_sampled_sequences_are_contiguous_within_episode_and_correct_length(scenario):
    """Property 6: every sampled sequence is contiguous, within-episode, length L."""
    ep_lengths, seq_len, batch_size = scenario
    buffer, _total = _build_buffer(ep_lengths)

    # Precondition for the success path: a valid window must exist.
    assert buffer.longest_contiguous_run() >= seq_len

    batch = buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)

    # The batch contains exactly the requested number of sequences.
    assert len(batch) == batch_size

    for seq in batch:
        _assert_valid_sequence(seq, seq_len)


@seed(20240607)
@settings(max_examples=20, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=8))
def test_sampled_sequences_at_mvp_configured_length_50(batch_size):
    """Property 6 at the MVP configured length L == 50 against a known long episode.

    A single episode of 60 transitions guarantees valid windows of length 50,
    matching the MVP scope's configured sequence length (Req 2.6, 7.2, 9.4).
    """
    seq_len = 50
    # One long episode of length 60 (terminated only on the final step), plus a
    # couple of short episodes to confirm windows never cross the boundary.
    ep_lengths = [60, 5, 3]
    buffer, _total = _build_buffer(ep_lengths)

    assert buffer.longest_contiguous_run() >= seq_len

    batch = buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)

    assert len(batch) == batch_size
    for seq in batch:
        _assert_valid_sequence(seq, seq_len)
        # Every window must live inside the single long episode (indices 0..59),
        # so its final index cannot reach the first boundary at index 59 unless
        # it is exactly the episode tail.
        indices = [int(round(t.reward)) for t in seq]
        assert indices[0] >= 0 and indices[-1] <= 59
