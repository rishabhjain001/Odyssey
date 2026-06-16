"""Episode-aware replay buffer for Odyssey (Req 2).

The :class:`ReplayBuffer` is a FIFO ring buffer of :class:`~agent.types.Transition`
records collected from the ``CarRacing`` environment. It stores transitions in
arrival order, evicts the oldest transition once it reaches its configured
capacity (Req 2.7), and samples *contiguous, within-episode* sequences for
world-model training (Req 2.4, 2.5, 2.6, 7.2, 9.4).

Episode boundaries are tracked through each transition's ``terminated`` flag. A
sampled window ``[i, i + seq_len)`` is valid only when it lies inside a single
episode: ``terminated`` may be ``True`` only on the *last* element of the
window, never before it. This guarantees that sampled sequences never span an
episode-termination boundary except at the final step (Req 2.5).

All public methods leave the buffer unchanged on failure: storage failures
preserve existing transitions (Req 2.3), and insufficient-data sample requests
raise :class:`InsufficientDataError` without mutating the buffer (Req 2.10, 9.7).

Sampling note: :meth:`ReplayBuffer.sample_sequences` draws start positions
*with replacement* across the set of valid windows. This keeps sampling
well-defined whenever at least one valid window exists, mirroring the
batch sampling used by DreamerV2-style world-model training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List

from agent.errors import BufferStorageError, InsufficientDataError
from agent.types import Transition

# Capacity bounds defined by the requirements (Req 2.7).
MIN_CAPACITY = 1
MAX_CAPACITY = 10_000_000


class ReplayBuffer:
    """A FIFO, episode-aware replay buffer of transitions (Req 2).

    Args:
        capacity: Maximum number of transitions retained, in
            ``[1, 10_000_000]`` (Req 2.7). Once full, adding a transition
            evicts the oldest one.

    Raises:
        ValueError: If ``capacity`` is outside ``[1, 10_000_000]``.
    """

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError(f"capacity must be an int, got {type(capacity).__name__}.")
        if not MIN_CAPACITY <= capacity <= MAX_CAPACITY:
            raise ValueError(
                f"capacity must be in [{MIN_CAPACITY}, {MAX_CAPACITY}], got {capacity}."
            )
        self._capacity = capacity
        # deque with maxlen gives O(1) FIFO eviction of the oldest element.
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """The configured maximum number of stored transitions (Req 2.7)."""
        return self._capacity

    def __len__(self) -> int:
        """The number of transitions currently stored."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Storage (Req 2.2, 2.3, 2.7)
    # ------------------------------------------------------------------

    def add(self, t: Transition) -> None:
        """Append a transition, evicting the oldest at capacity (Req 2.2, 2.7).

        The buffer behaves as a FIFO ring: when it already holds ``capacity``
        transitions, the oldest transition is discarded to make room for ``t``.

        On failure (for example, an object that is not a valid
        :class:`~agent.types.Transition`), the existing transitions are left
        unchanged and a :class:`BufferStorageError` is raised (Req 2.3).

        Args:
            t: The transition to store.

        Raises:
            BufferStorageError: If ``t`` cannot be stored; the buffer is
                unchanged.
        """
        if not isinstance(t, Transition):
            # Validate before mutating so existing transitions stay intact.
            raise BufferStorageError(
                f"Cannot store object of type {type(t).__name__}; expected Transition."
            )
        try:
            self._buffer.append(t)
        except Exception as exc:  # pragma: no cover - deque.append rarely fails
            raise BufferStorageError(f"Failed to store transition: {exc}") from exc

    # ------------------------------------------------------------------
    # Episode-run analysis (Req 9.7)
    # ------------------------------------------------------------------

    def longest_contiguous_run(self) -> int:
        """Return the length of the longest within-episode contiguous run (Req 9.7).

        A run is a maximal block of consecutive stored transitions that ends
        either at a ``terminated`` transition (inclusive of that transition) or
        at the end of the buffer. A ``terminated`` transition closes the run it
        belongs to, because it may legally be the final element of a sampled
        window.

        Returns:
            The number of transitions in the longest such run, or ``0`` when the
            buffer is empty.
        """
        longest = 0
        current = 0
        for t in self._buffer:
            current += 1
            if t.terminated:
                if current > longest:
                    longest = current
                current = 0
        if current > longest:
            longest = current
        return longest

    # ------------------------------------------------------------------
    # Sampling (Req 2.4, 2.5, 2.6, 2.10, 9.4, 9.7)
    # ------------------------------------------------------------------

    def _valid_start_indices(self, seq_len: int) -> List[int]:
        """Return start indices of valid contiguous within-episode windows.

        A window ``[i, i + seq_len)`` is valid when it fits inside the stored
        transitions and contains no episode-termination boundary except possibly
        at its final element: ``terminated`` must be ``False`` for every index in
        ``[i, i + seq_len - 1)`` (the last index ``i + seq_len - 1`` may be
        ``terminated``).
        """
        n = len(self._buffer)
        if seq_len <= 0 or seq_len > n:
            return []

        transitions = list(self._buffer)

        # next_term[i] = smallest j >= i with transitions[j].terminated, else n.
        next_term = [n] * (n + 1)
        for i in range(n - 1, -1, -1):
            next_term[i] = i if transitions[i].terminated else next_term[i + 1]

        valid: List[int] = []
        last_start = n - seq_len
        for i in range(last_start + 1):
            # The last legal element of the window is at index i + seq_len - 1.
            # The window is valid iff the nearest termination at-or-after i is
            # not strictly before that last element.
            if next_term[i] >= i + seq_len - 1:
                valid.append(i)
        return valid

    def sample_sequences(self, batch_size: int, seq_len: int) -> List[List[Transition]]:
        """Sample contiguous, within-episode transition sequences (Req 2.4-2.6).

        Returns ``batch_size`` sequences, each containing exactly ``seq_len``
        transitions drawn contiguously from a single episode in stored order.
        No returned sequence spans an episode-termination boundary except
        possibly at its final element (Req 2.5). Start positions are drawn with
        replacement from the set of valid windows.

        Args:
            batch_size: Number of sequences to return (must be >= 1).
            seq_len: Length of each contiguous sequence (must be >= 1). This is
                the configured training sequence length (exactly 50 for the MVP
                scope, in ``(50, 200]`` for RGB; Req 2.6, 9.4).

        Returns:
            A list of ``batch_size`` lists, each of ``seq_len`` transitions.

        Raises:
            InsufficientDataError: If the buffer holds fewer transitions than
                ``batch_size``, or if ``seq_len`` exceeds the longest contiguous
                within-episode run (so no valid window exists). The buffer is
                left unchanged (Req 2.10, 9.7).
        """
        if batch_size <= 0:
            raise InsufficientDataError(f"batch_size must be >= 1, got {batch_size}.")
        if seq_len <= 0:
            raise InsufficientDataError(f"seq_len must be >= 1, got {seq_len}.")

        n = len(self._buffer)
        if n < batch_size:
            raise InsufficientDataError(
                f"Replay buffer holds {n} transitions, fewer than the requested "
                f"batch size {batch_size}."
            )

        longest = self.longest_contiguous_run()
        if seq_len > longest:
            raise InsufficientDataError(
                f"Requested sequence length {seq_len} exceeds the longest "
                f"contiguous within-episode run available ({longest})."
            )

        valid_starts = self._valid_start_indices(seq_len)
        if not valid_starts:
            # Defensive: longest >= seq_len should guarantee a valid window.
            raise InsufficientDataError(
                f"No valid contiguous window of length {seq_len} is available."
            )

        transitions = list(self._buffer)
        batch: List[List[Transition]] = []
        for _ in range(batch_size):
            start = random.choice(valid_starts)
            batch.append(transitions[start:start + seq_len])
        return batch
