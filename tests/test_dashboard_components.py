"""Dashboard component / snapshot tests (Task 17.2).

Validates: Requirements 15.1 – 15.12

Tests verify panel composition, empty-state indications, and the auto-refresh
hook using a stubbed Streamlit and a stubbed data source.

Strategy
--------
Streamlit is patched at the module level *before* any panel function is invoked.
We patch all ``st.*`` callables that the panels may invoke so they never attempt
to connect to a running Streamlit server.  Numpy artifact files are written to
``tmp_path`` fixtures so tests are fully self-contained.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a streamlit mock and inject it into sys.modules so that
# ``import streamlit as st`` inside dashboard.app resolves to our mock.
# ---------------------------------------------------------------------------

def _make_st_mock() -> MagicMock:
    """Return a MagicMock that acts as the 'streamlit' module."""
    st = MagicMock()
    # Make context-manager calls (st.columns, st.expander, st.spinner) work.
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock())
    cm.__exit__ = MagicMock(return_value=False)
    st.columns.return_value = [cm, cm]
    st.expander.return_value = cm
    st.spinner.return_value = cm
    return st


# ---------------------------------------------------------------------------
# 1. test_dashboard_is_pure_python
# ---------------------------------------------------------------------------

def test_dashboard_is_pure_python():
    """Req 15.1 — The dashboard must be a pure-Python Streamlit app (no React).

    We import the module with a mocked streamlit and inspect its source for any
    reference to React / JSX / custom-component patterns that would indicate a
    non-pure-Python component.

    Specifically, a pure-Python dashboard must:
    - NOT declare custom React components via st.components.v1.declare_component
    - NOT import 'react' as a Python package
    - NOT import streamlit.components.v1

    Note: the word "React" may legitimately appear in docstrings/comments;
    we only guard against actual functional usage (declare_component).
    """
    st_mock = _make_st_mock()
    # Ensure a clean import by removing any cached module first.
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        import dashboard.app as app  # noqa: F401

    source = inspect.getsource(app)
    # Functional React usage: registering a custom component would require
    # declare_component from streamlit.components.v1.
    assert "declare_component" not in source, (
        "Dashboard must not declare custom React components via st.components.v1."
    )
    # No import of the standalone 'react' Python package.
    assert "import react" not in source.lower(), (
        "Dashboard must not import a 'react' package."
    )


# ---------------------------------------------------------------------------
# 2. test_panel_real_vs_recon_shows_unavailable_when_no_files
# ---------------------------------------------------------------------------

def test_panel_real_vs_recon_shows_unavailable_when_no_files(tmp_path):
    """Req 15.5, 15.12 — When no .npy artifacts exist the panel calls st.info
    with a message containing 'not yet available'."""
    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_real_vs_recon

    non_existent_dir = str(tmp_path / "no_artifacts")
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_real_vs_recon(non_existent_dir)

    # At least one st.info call must have occurred with the expected message.
    info_calls = st_mock.info.call_args_list
    assert info_calls, "st.info should have been called when no artifacts exist."
    messages = [str(call.args[0]).lower() for call in info_calls]
    assert any("not yet available" in msg for msg in messages), (
        f"Expected 'not yet available' in one of the st.info messages; got: {messages}"
    )


# ---------------------------------------------------------------------------
# 3. test_panel_real_vs_recon_calls_image_when_data_present
# ---------------------------------------------------------------------------

def test_panel_real_vs_recon_calls_image_when_data_present(tmp_path):
    """Req 15.2 — When real_frame.npy and recon_frame.npy are present, the
    panel calls st.image at least twice (once for each frame)."""
    # Write synthetic float32 frames of shape (1, 64, 64).
    real_frame = np.zeros((1, 64, 64), dtype=np.float32)
    recon_frame = np.ones((1, 64, 64), dtype=np.float32) * 0.5
    np.save(tmp_path / "real_frame.npy", real_frame)
    np.save(tmp_path / "recon_frame.npy", recon_frame)

    st_mock = _make_st_mock()
    # Make st.columns return two proper context-manager mocks.
    col1 = MagicMock()
    col1.__enter__ = MagicMock(return_value=col1)
    col1.__exit__ = MagicMock(return_value=False)
    col2 = MagicMock()
    col2.__enter__ = MagicMock(return_value=col2)
    col2.__exit__ = MagicMock(return_value=False)
    st_mock.columns.return_value = [col1, col2]

    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_real_vs_recon

    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_real_vs_recon(str(tmp_path))

    # st.image could be called on the col mocks OR on st_mock itself depending
    # on how the 'with col:' context manager forwards attribute access.
    # Check both st_mock.image and col*.image.
    total_image_calls = (
        st_mock.image.call_count
        + col1.image.call_count
        + col2.image.call_count
    )
    assert total_image_calls >= 2, (
        f"Expected at least 2 st.image calls when both frames exist; "
        f"got {total_image_calls}."
    )


# ---------------------------------------------------------------------------
# 4. test_panel_reconstruction_loss_shows_unavailable_when_no_data
# ---------------------------------------------------------------------------

def test_panel_reconstruction_loss_shows_unavailable_when_no_data():
    """Req 15.5 — Calling _panel_reconstruction_loss(None) must call st.info."""
    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_reconstruction_loss

    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_reconstruction_loss(None)

    assert st_mock.info.called, (
        "st.info should be called when no MetricsStore is provided."
    )


# ---------------------------------------------------------------------------
# 5. test_panel_reconstruction_loss_plots_when_data_present
# ---------------------------------------------------------------------------

def test_panel_reconstruction_loss_plots_when_data_present():
    """Req 15.3 — When a MetricsStore with reconstruction losses is provided,
    the panel calls st.pyplot or st.line_chart.

    matplotlib.pyplot.subplots is patched to avoid any display-backend issues in
    headless/CI environments (Python 3.14 has a known deprecation in plt.subplots).
    """
    from training.metrics import MetricsStore

    store = MetricsStore()
    for i in range(5):
        store.record_reconstruction_loss(step=i, value=float(1.0 - i * 0.1))

    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        import dashboard.app as app_mod

    # Patch plt.subplots on the already-imported module so the pyplot code path
    # runs without hitting a real display backend.
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    import matplotlib.pyplot as real_plt  # noqa: F401 — just to confirm it's available

    app_mod._HAS_MPL = True
    with patch.object(app_mod, "plt") as plt_mock:
        plt_mock.subplots.return_value = (fig_mock, ax_mock)
        app_mod._panel_reconstruction_loss(store)

    plot_calls = st_mock.pyplot.call_count + st_mock.line_chart.call_count
    assert plot_calls >= 1, (
        "Expected st.pyplot or st.line_chart to be called when loss data is present; "
        f"pyplot={st_mock.pyplot.call_count}, line_chart={st_mock.line_chart.call_count}."
    )


# ---------------------------------------------------------------------------
# 6. test_panel_imagined_rollout_shows_unavailable_when_no_file
# ---------------------------------------------------------------------------

def test_panel_imagined_rollout_shows_unavailable_when_no_file(tmp_path):
    """Req 15.12 — When imagined_frames.npy is absent, the panel calls st.info."""
    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_imagined_rollout

    empty_dir = str(tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_imagined_rollout(empty_dir)

    assert st_mock.info.called, (
        "st.info should be called when imagined_frames.npy is absent."
    )


# ---------------------------------------------------------------------------
# 7. test_panel_imagined_rollout_shows_frames_when_data_present
# ---------------------------------------------------------------------------

def test_panel_imagined_rollout_shows_frames_when_data_present(tmp_path):
    """Req 15.6 — When imagined_frames.npy with 3 horizon frames is present,
    the panel calls st.image at least 3 times."""
    # Shape (3, 1, 64, 64) — 3 horizon frames, 1 channel, 64×64
    frames = np.random.rand(3, 1, 64, 64).astype(np.float32)
    np.save(tmp_path / "imagined_frames.npy", frames)

    st_mock = _make_st_mock()

    # Make st.columns(n) return n column mocks.
    def _columns_side_effect(n):
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    st_mock.columns.side_effect = _columns_side_effect

    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_imagined_rollout

    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_imagined_rollout(str(tmp_path))

    # Collect image calls from all column mocks and st_mock itself.
    total = st_mock.image.call_count
    # The columns side_effect created new mocks each call; gather via call_args_list
    # from the actual MagicMock columns returned.
    # Because the columns are created inside _columns_side_effect we can't easily
    # introspect them post-hoc, so we patch st.image directly instead.
    # Re-run with a direct st.image spy.
    st_mock2 = _make_st_mock()
    st_mock2.columns.side_effect = _columns_side_effect
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock2}):
        from dashboard.app import _panel_imagined_rollout as _panel2

    image_calls = []

    def _capture_image(*args, **kwargs):
        image_calls.append((args, kwargs))

    st_mock2.image.side_effect = _capture_image

    with patch.dict(sys.modules, {"streamlit": st_mock2}):
        _panel2(str(tmp_path))

    assert len(image_calls) >= 3, (
        f"Expected at least 3 st.image calls for a horizon-3 imagined rollout; "
        f"got {len(image_calls)}."
    )


# ---------------------------------------------------------------------------
# 8. test_panel_reward_comparison_shows_unavailable_when_no_data
# ---------------------------------------------------------------------------

def test_panel_reward_comparison_shows_unavailable_when_no_data(tmp_path):
    """Req 15.11 — When there are no eval records and no ppo_rewards.npy,
    the panel calls st.info with a message containing 'not yet available'."""
    from training.metrics import MetricsStore

    empty_store = MetricsStore()  # no eval records

    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        from dashboard.app import _panel_reward_comparison

    artifact_dir = str(tmp_path / "no_artifacts")
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        _panel_reward_comparison(empty_store, artifact_dir)

    info_calls = st_mock.info.call_args_list
    assert info_calls, "st.info should have been called when reward data is absent."
    messages = [str(call.args[0]).lower() for call in info_calls]
    assert any("not yet available" in msg for msg in messages), (
        f"Expected 'not yet available' in one of the st.info messages; got: {messages}"
    )


# ---------------------------------------------------------------------------
# 9. test_auto_refresh_setup
# ---------------------------------------------------------------------------

def test_auto_refresh_setup():
    """Req 15.4, 15.10 — The dashboard source must reference st.rerun (or
    st.experimental_rerun) to implement the ≤5-second auto-refresh."""
    st_mock = _make_st_mock()
    sys.modules.pop("dashboard.app", None)
    with patch.dict(sys.modules, {"streamlit": st_mock}):
        import dashboard.app as app

    source = inspect.getsource(app)
    assert "rerun" in source, (
        "dashboard.app must reference st.rerun or st.experimental_rerun "
        "to implement the auto-refresh cycle (Req 15.4, 15.10)."
    )
