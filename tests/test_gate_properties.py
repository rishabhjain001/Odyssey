# Feature: odyssey, Property 18: The reconstruction gate governs behavior and baseline activation
"""Property-based tests for :class:`training.gate.ReconstructionGate`.

This module implements design Correctness Property 18:

    Property 18: The reconstruction gate governs behavior and baseline activation

    For any recorded reconstruction-loss history, ``is_confirmed`` is true if
    and only if the last ``consecutive_steps`` recorded losses are all at or
    below the configured threshold. While the condition is not met,
    behavior-model training and PPO-baseline execution are disabled regardless
    of configuration; and when the configuration requests enabling behavior
    training while the condition is not met, the decision is ``disabled``
    together with a recorded/logged gate error.

Validates: Requirements 8.2, 8.3, 8.5, 18.5
"""

from __future__ import annotations

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.config import GateConfig
from agent.errors import GateNotConfirmedError
from training.gate import ReconstructionGate


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# threshold in [0.0, 1.0] — realistic normalised-loss range
threshold_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# consecutive_steps in [1, 20]
steps_strategy = st.integers(min_value=1, max_value=20)

# history values span the full [0.0, 2.0] range so both confirmed and
# non-confirmed states are explored (values above 1.0 can exceed the threshold)
history_value_strategy = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)

# history length in [0, 40]
history_strategy = st.lists(history_value_strategy, min_size=0, max_size=40)


# ---------------------------------------------------------------------------
# Property 1: is_confirmed iff all recent losses <= threshold
# ---------------------------------------------------------------------------

@seed(20240618)
@settings(max_examples=20, deadline=None)
@given(
    threshold=threshold_strategy,
    consecutive_steps=steps_strategy,
    history=history_strategy,
)
def test_is_confirmed_iff_recent_losses_at_or_below_threshold(
    threshold: float,
    consecutive_steps: int,
    history: list[float],
) -> None:
    """is_confirmed == True iff len(history) >= consecutive_steps and all tail values <= threshold.

    Validates: Requirements 8.2, 8.3, 8.5, 18.5
    """
    gate = ReconstructionGate(GateConfig(loss_threshold=threshold, consecutive_steps=consecutive_steps))

    result = gate.is_confirmed(history)

    # Compute expected value directly from the spec definition
    expected = (
        len(history) >= consecutive_steps
        and all(v <= threshold for v in history[-consecutive_steps:])
    )

    assert result == expected, (
        f"is_confirmed={result} but expected={expected} for "
        f"threshold={threshold}, consecutive_steps={consecutive_steps}, "
        f"history_len={len(history)}, tail={history[-consecutive_steps:] if history else []}"
    )


# ---------------------------------------------------------------------------
# Property 2: behavior disabled when not confirmed, regardless of config_enabled
# ---------------------------------------------------------------------------

@seed(20240618)
@settings(max_examples=20, deadline=None)
@given(
    threshold=threshold_strategy,
    consecutive_steps=steps_strategy,
    history=history_strategy,
)
def test_behavior_disabled_when_not_confirmed(
    threshold: float,
    consecutive_steps: int,
    history: list[float],
) -> None:
    """When the gate is not confirmed, behavior_allowed returns (False, ...) for any config_enabled.

    - config_enabled=True, not confirmed  → (False, non-None error_msg)   (Req 8.5)
    - config_enabled=False, not confirmed → (False, None)                 (Req 8.2, 8.3)

    Validates: Requirements 8.2, 8.3, 8.5, 18.5
    """
    gate = ReconstructionGate(GateConfig(loss_threshold=threshold, consecutive_steps=consecutive_steps))

    # Only exercise the not-confirmed branch
    if gate.is_confirmed(history):
        return  # skip confirmed states; covered by Property 3

    # config_enabled=True: must return (False, non-None error_msg) with gate error info
    allowed_on, err_on = gate.behavior_allowed(config_enabled=True, recon_history=history)
    assert allowed_on is False, (
        f"behavior_allowed(True) returned allowed=True when gate not confirmed; "
        f"threshold={threshold}, steps={consecutive_steps}, history_len={len(history)}"
    )
    assert err_on is not None, (
        f"behavior_allowed(True) returned None error_msg when gate not confirmed; "
        f"threshold={threshold}, steps={consecutive_steps}, history_len={len(history)}"
    )
    # The error message should carry GateNotConfirmedError information
    assert isinstance(err_on, str) and len(err_on) > 0, (
        f"Expected a non-empty string error_msg, got {err_on!r}"
    )

    # config_enabled=False: must return (False, None) silently (Req 8.2/8.3)
    allowed_off, err_off = gate.behavior_allowed(config_enabled=False, recon_history=history)
    assert allowed_off is False, (
        f"behavior_allowed(False) returned allowed=True when gate not confirmed; "
        f"threshold={threshold}, steps={consecutive_steps}, history_len={len(history)}"
    )
    assert err_off is None, (
        f"behavior_allowed(False) returned non-None error when config is disabled; "
        f"got {err_off!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: behavior allowed only when config_enabled AND confirmed
# ---------------------------------------------------------------------------

@seed(20240618)
@settings(max_examples=20, deadline=None)
@given(
    threshold=threshold_strategy,
    consecutive_steps=steps_strategy,
    history=history_strategy,
)
def test_behavior_allowed_only_when_config_enabled_and_confirmed(
    threshold: float,
    consecutive_steps: int,
    history: list[float],
) -> None:
    """When the gate is confirmed: (True, None) for config_enabled=True; (False, None) for False.

    Validates: Requirements 8.2, 8.3, 8.5, 18.5
    """
    gate = ReconstructionGate(GateConfig(loss_threshold=threshold, consecutive_steps=consecutive_steps))

    # Only exercise the confirmed branch
    if not gate.is_confirmed(history):
        return  # skip non-confirmed states; covered by Property 2

    # config_enabled=True AND confirmed → gate open
    allowed_on, err_on = gate.behavior_allowed(config_enabled=True, recon_history=history)
    assert allowed_on is True, (
        f"behavior_allowed(True) returned allowed=False when gate confirmed; "
        f"threshold={threshold}, steps={consecutive_steps}, history_len={len(history)}"
    )
    assert err_on is None, (
        f"behavior_allowed(True) returned non-None error when gate confirmed; got {err_on!r}"
    )

    # config_enabled=False AND confirmed → still disabled by config, silently
    allowed_off, err_off = gate.behavior_allowed(config_enabled=False, recon_history=history)
    assert allowed_off is False, (
        f"behavior_allowed(False) returned allowed=True even when gate confirmed; "
        f"threshold={threshold}, steps={consecutive_steps}, history_len={len(history)}"
    )
    assert err_off is None, (
        f"behavior_allowed(False) returned non-None error when config is disabled; "
        f"got {err_off!r}"
    )
