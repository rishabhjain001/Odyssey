"""Metrics store for Odyssey training (Req 5.3, 5.4, 6.3, 6.4, 11.5, 14.1, 14.2, 14.5, 14.6).

This file is built in stages across multiple tasks:
  - Task 7.2 (this task): implements the core MetricsStore class with reconstruction-loss
    recording, reward-accuracy recording, evaluation recording, world-model loss history,
    and persistence.
  - Task 6.2 (MUST EXTEND, not overwrite): adds the reconstruction-shape-mismatch
    recording hook to MetricsStore by appending new methods/filling the stub below.
  - Task 14.2 (MUST EXTEND, not overwrite): adds reward/sample-efficiency comparison
    functions by appending to this file after the MetricsStore class.

IMPORTANT for later tasks: do NOT replace or overwrite this file. Append new methods
to the MetricsStore class and/or append new module-level functions after it.

Extension points are clearly marked with ``# --- EXTENSION POINT ---`` comments.
"""

from __future__ import annotations

import json
import os
from typing import Any


class MetricsStore:
    """Central store for all Odyssey training metrics.

    Accumulates metrics in memory using plain Python lists. All stored values
    must be JSON-serializable (int, float, str, list, dict) so the store can be
    persisted to and reloaded from a JSON file (Req 14.5).

    Recorded series (all in insertion order):
    - Reconstruction-loss values: ``(step, value)`` pairs (Req 5.3, 14.6).
    - Reward-prediction accuracy: mean-absolute-error values (Req 6.3, 6.4).
    - Evaluation records: env_steps, avg_reward, completion_rate (Req 11.5, 14.1, 14.2).
    - World-model per-step loss terms (Req 7.5).
    - Reconstruction-shape-mismatch events (stub; filled in by task 6.2).

    The class is intentionally designed for easy extension:
    - To add a new metric series: add a ``list`` attribute in ``__init__`` and
      a pair of ``record_*`` / ``*_history`` methods.
    - Task 6.2 should extend ``record_reconstruction_shape_mismatch`` with real logic.
    - Task 14.2 should append comparison methods after this class.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        # Reconstruction-loss series: list of [step, value] (Req 5.3, 14.6).
        self._reconstruction_losses: list[list[int | float]] = []

        # Reward-prediction accuracy (MAE) series: list of float (Req 6.3, 6.4).
        self._reward_accuracies: list[float] = []

        # Evaluation records: list of dicts with env_steps, avg_reward, completion_rate
        # (Req 11.5, 14.1, 14.2).
        self._eval_records: list[dict[str, Any]] = []

        # World-model per-step loss terms: list of dicts (Req 7.5).
        self._world_model_losses: list[dict[str, Any]] = []

        # Reconstruction-shape-mismatch events (stub for task 6.2).
        self._reconstruction_shape_mismatches: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Reconstruction loss (Req 5.3, 5.4, 14.6)
    # ------------------------------------------------------------------ #

    def record_reconstruction_loss(self, step: int, value: float) -> None:
        """Append a reconstruction-loss observation for the given training step.

        Args:
            step: The training step number at which the loss was measured.
            value: The scalar reconstruction-loss value.

        Req 5.3: records at each training step.
        Req 14.6: records reconstruction loss over training.
        """
        self._reconstruction_losses.append([int(step), float(value)])

    def reconstruction_losses(self) -> list[tuple[int, float]]:
        """Return all recorded reconstruction-loss observations in recorded order.

        Returns:
            List of ``(step, value)`` tuples in the order they were recorded.

        Req 5.4: exposes recorded values for retrieval by the Dashboard.
        """
        return [(int(pair[0]), float(pair[1])) for pair in self._reconstruction_losses]

    # ------------------------------------------------------------------ #
    # Reward-prediction accuracy (Req 6.3, 6.4)
    # ------------------------------------------------------------------ #

    def record_reward_accuracy(self, mae: float) -> None:
        """Append a reward-prediction mean-absolute-error measurement.

        The MAE is computed externally over sampled real transitions; this store
        only records the result (Req 6.3).

        Args:
            mae: Mean absolute error between predicted and recorded rewards.

        Req 6.3: records reward-prediction accuracy.
        """
        self._reward_accuracies.append(float(mae))

    def reward_accuracies(self) -> list[float]:
        """Return all recorded reward-prediction MAE values in recorded order.

        Returns:
            List of float MAE values in the order they were recorded.

        Req 6.4: exposes reward-prediction accuracy for retrieval by the Dashboard.
        """
        return list(self._reward_accuracies)

    # ------------------------------------------------------------------ #
    # Evaluation records (Req 11.5, 14.1, 14.2)
    # ------------------------------------------------------------------ #

    def record_eval(
        self,
        env_steps: int,
        avg_reward: float,
        completion_rate: float,
    ) -> None:
        """Append an evaluation record.

        Args:
            env_steps: Cumulative environment steps consumed by Odyssey at the
                time of this evaluation run (Req 11.5).
            avg_reward: Arithmetic mean of per-episode total rewards across the
                evaluated episodes (Req 11.2, 14.1).
            completion_rate: Fraction of evaluated episodes that completed the
                track, in ``[0.0, 1.0]`` (Req 11.4, 14.1).

        Req 11.5: records cumulative env steps at each evaluation.
        Req 14.1: records average episode reward as a function of cumulative steps.
        """
        self._eval_records.append(
            {
                "env_steps": int(env_steps),
                "avg_reward": float(avg_reward),
                "completion_rate": float(completion_rate),
            }
        )

    def eval_records(self) -> list[dict[str, Any]]:
        """Return all evaluation records in recorded order.

        Each record is a dict with keys ``"env_steps"``, ``"avg_reward"``, and
        ``"completion_rate"``.

        Returns:
            List of evaluation-record dicts in the order they were recorded.

        Req 11.5, 14.1, 14.2: exposes evaluation metrics for the dashboard and
        comparison functions.
        """
        return [dict(r) for r in self._eval_records]

    def cumulative_env_steps(self) -> int:
        """Return the cumulative environment steps at the most recent evaluation.

        Returns ``0`` if no evaluation has been recorded yet.

        Returns:
            The ``env_steps`` value from the latest evaluation record, or ``0``
            if none have been recorded.

        Req 11.5.
        """
        if not self._eval_records:
            return 0
        return int(self._eval_records[-1]["env_steps"])

    # ------------------------------------------------------------------ #
    # World-model loss history (Req 7.5)
    # ------------------------------------------------------------------ #

    def record_world_model_losses(self, step: int, losses: dict[str, float]) -> None:
        """Append a per-step snapshot of world-model loss terms.

        Called by ``WorldModelTrainer`` after each training step (Req 7.5).

        Args:
            step: The world-model training step number.
            losses: A dict of named loss values (e.g. ``{"recon": 0.12,
                "kl": 0.05, "reward": 0.02, "discount": 0.01, "total": 0.20}``).
                Values must be JSON-serializable floats.
        """
        entry: dict[str, Any] = {"step": int(step)}
        for key, val in losses.items():
            entry[str(key)] = float(val)
        self._world_model_losses.append(entry)

    def world_model_loss_history(self) -> list[dict[str, Any]]:
        """Return all world-model loss snapshots in recorded order.

        Returns:
            List of dicts, each containing ``"step"`` and the loss-term keys,
            in the order they were recorded.
        """
        return [dict(entry) for entry in self._world_model_losses]

    # ------------------------------------------------------------------ #
    # Reconstruction-shape-mismatch stub (to be extended by task 6.2)
    # ------------------------------------------------------------------ #

    def record_reconstruction_shape_mismatch(self, step: int, details: str) -> None:
        """Record a reconstruction-shape-mismatch event.

        This method is a stub that stores events so that task 6.2 can extend it
        with full ``ReconstructionShapeMismatch`` error logic without replacing
        the MetricsStore. Task 6.2 should add a ``reconstruction_shape_mismatches()``
        retrieval method and wire this into the reconstruction-loss path (Req 5.5).

        Args:
            step: The training step at which the mismatch occurred.
            details: A human-readable description of the mismatch (e.g. the
                shape pair that differed).

        # --- EXTENSION POINT (task 6.2) ---
        # Replace or augment this stub to integrate with ReconstructionShapeMismatch
        # error recording. Add a retrieval method ``reconstruction_shape_mismatches()``
        # returning the list below.
        """
        self._reconstruction_shape_mismatches.append(
            {"step": int(step), "details": str(details)}
        )

    def reconstruction_shape_mismatches(self) -> list[dict]:
        """Return all recorded reconstruction-shape-mismatch events in recorded order.

        Each entry is a dict with keys ``"step"`` (int) and ``"details"`` (str).
        Added by task 6.2 as the retrieval counterpart to
        :meth:`record_reconstruction_shape_mismatch`.

        Returns:
            A shallow copy of the internal mismatch list in insertion order.

        Req 5.5: exposes recorded mismatch events for inspection and the
        dashboard.
        """
        return list(self._reconstruction_shape_mismatches)

    # ------------------------------------------------------------------ #
    # Persistence (Req 14.5)
    # ------------------------------------------------------------------ #

    def persist(self, path: str) -> None:
        """Serialize the metrics store to a JSON file at ``path`` (Req 14.5).

        Creates or overwrites the file at ``path``. Parent directories are
        created if they do not already exist.

        The serialized format is a JSON object containing all metric series in
        their recorded order so that ``load_metrics`` can reconstruct an
        equivalent store.

        Args:
            path: Filesystem path for the output JSON file.

        Req 14.5: persists recorded metrics to a file.
        """
        payload: dict[str, Any] = {
            "reconstruction_losses": self._reconstruction_losses,
            "reward_accuracies": self._reward_accuracies,
            "eval_records": self._eval_records,
            "world_model_losses": self._world_model_losses,
            "reconstruction_shape_mismatches": self._reconstruction_shape_mismatches,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "MetricsStore":
        """Reconstruct a MetricsStore from a deserialized JSON payload.

        Used by ``load_metrics``; not part of the public API.
        """
        store = cls()
        store._reconstruction_losses = payload.get("reconstruction_losses", [])
        store._reward_accuracies = payload.get("reward_accuracies", [])
        store._eval_records = payload.get("eval_records", [])
        store._world_model_losses = payload.get("world_model_losses", [])
        store._reconstruction_shape_mismatches = payload.get(
            "reconstruction_shape_mismatches", []
        )
        return store


# ------------------------------------------------------------------ #
# Module-level factory (Req 14.5)
# ------------------------------------------------------------------ #


def load_metrics(path: str) -> MetricsStore:
    """Load a MetricsStore from a JSON file previously written by ``persist``.

    The loaded store is equivalent to the original: all metric series are
    restored in the same order and with the same values.

    Args:
        path: Filesystem path to the JSON file produced by
            :meth:`MetricsStore.persist`.

    Returns:
        A :class:`MetricsStore` instance whose series match those that were
        persisted.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Req 14.5: reloads persisted metrics to an equivalent set.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return MetricsStore._from_payload(payload)


# --- EXTENSION POINT (task 14.2) ---
# Append reward/sample-efficiency comparison functions after this line.
# Do NOT modify the MetricsStore class above; only append new module-level
# functions (e.g. ``reward_comparison``, ``sample_efficiency_comparison``)
# that accept a MetricsStore (or its eval_records / PPO records) as input.

from agent.errors import ComparisonNotReadyError


def reward_comparison(
    odyssey_store: MetricsStore,
    ppo_avg_reward: float | None,
    ppo_steps: int | None,
    min_episodes: int = 5,
) -> dict:
    """Compare the most-recent average reward of both agents.

    Raises :class:`~agent.errors.ComparisonNotReadyError` if the Odyssey store
    does not yet hold at least ``min_episodes`` evaluation records, or if
    ``ppo_avg_reward`` is ``None`` (PPO baseline has not finished training).

    Args:
        odyssey_store: The :class:`MetricsStore` populated during Odyssey training.
        ppo_avg_reward: Average episode reward achieved by the PPO baseline over
            its evaluation episodes, or ``None`` if PPO has not been evaluated yet.
        ppo_steps: Cumulative environment steps consumed by the PPO baseline, or
            ``None`` if PPO has not been evaluated yet.
        min_episodes: Minimum number of Odyssey evaluation records required
            before a comparison can be produced.  Defaults to ``5``.

    Returns:
        A dict with keys:
        - ``"odyssey_avg_reward"`` (float): mean reward from the most recent
          Odyssey evaluation record.
        - ``"ppo_avg_reward"`` (float): the supplied PPO average reward.
        - ``"odyssey_env_steps"`` (int): cumulative env steps from the most
          recent Odyssey evaluation record.
        - ``"ppo_env_steps"`` (int): the supplied PPO step count.

    Raises:
        ComparisonNotReadyError: If Odyssey has fewer than ``min_episodes``
            evaluation records, or if ``ppo_avg_reward`` is ``None``.

    Req 14.3, 14.7.
    """
    records = odyssey_store.eval_records()
    if len(records) < min_episodes:
        raise ComparisonNotReadyError(
            f"Odyssey has only {len(records)} evaluation record(s); "
            f"at least {min_episodes} are required for a comparison."
        )
    if ppo_avg_reward is None:
        raise ComparisonNotReadyError(
            "PPO baseline has not been evaluated yet (ppo_avg_reward is None)."
        )

    latest = records[-1]
    return {
        "odyssey_avg_reward": float(latest["avg_reward"]),
        "ppo_avg_reward": float(ppo_avg_reward),
        "odyssey_env_steps": int(latest["env_steps"]),
        "ppo_env_steps": int(ppo_steps) if ppo_steps is not None else 0,
    }


def sample_efficiency_comparison(
    odyssey_store: MetricsStore,
    ppo_results: list[dict],
    step_counts: list[int],
) -> list[dict]:
    """Produce a per-step sample-efficiency comparison between Odyssey and PPO.

    For each step count in ``step_counts`` the function finds the latest
    evaluation record (from each agent) whose ``env_steps`` is less than or
    equal to that count, and records the corresponding ``avg_reward`` (or
    ``None`` if no record exists up to that point).

    Args:
        odyssey_store: The :class:`MetricsStore` populated during Odyssey
            training.  Its evaluation records must contain ``"env_steps"`` and
            ``"avg_reward"`` keys.
        ppo_results: Evaluation records for the PPO baseline.  Each element is
            a dict with at least ``"env_steps"`` (int) and ``"avg_reward"``
            (float).
        step_counts: Sorted or unsorted list of cumulative env-step counts at
            which to compare the two agents.

    Returns:
        A list of dicts (one per entry in ``step_counts``) with keys:
        - ``"env_steps"`` (int): the query step count.
        - ``"odyssey_avg_reward"`` (float | None): latest Odyssey reward up to
          that step count, or ``None`` if no record exists yet.
        - ``"ppo_avg_reward"`` (float | None): latest PPO reward up to that
          step count, or ``None`` if no record exists yet.

    Raises:
        ComparisonNotReadyError: If either the Odyssey eval series or
            ``ppo_results`` is empty.

    Req 14.4, 14.7.
    """
    odyssey_records = odyssey_store.eval_records()
    if not odyssey_records:
        raise ComparisonNotReadyError(
            "Odyssey has no evaluation records; cannot produce a sample-efficiency comparison."
        )
    if not ppo_results:
        raise ComparisonNotReadyError(
            "ppo_results is empty; cannot produce a sample-efficiency comparison."
        )

    def _latest_reward(records: list[dict], step: int) -> float | None:
        """Return the avg_reward of the latest record with env_steps <= step."""
        best: float | None = None
        for rec in records:
            if rec["env_steps"] <= step:
                best = float(rec["avg_reward"])
        return best

    result = []
    for step in step_counts:
        result.append(
            {
                "env_steps": int(step),
                "odyssey_avg_reward": _latest_reward(odyssey_records, step),
                "ppo_avg_reward": _latest_reward(ppo_results, step),
            }
        )
    return result
