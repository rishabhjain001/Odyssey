"""Odyssey Streamlit Dashboard (Req 15.1 – 15.12).

Pure-Python Streamlit application — no custom React components.

Reads from:
  - ``MetricsStore`` (via a JSON file on disk): reconstruction-loss series and
    episode-reward series for both Odyssey and PPO.
  - Artifact ``.npy`` files written by the training pipeline.

Panels (one per requirement cluster):
  1. Real vs Reconstructed Frame     (Req 15.2, 15.12)
  2. Reconstruction-loss curve       (Req 15.3, 15.4, 15.5)
  3. Imagined-rollout frames         (Req 15.6, 15.12)
  4. UMAP projection of latents      (Req 15.7)
  5. Reward-comparison curves        (Req 15.8, 15.11)
  6. Planned-trajectory overlay      (Req 15.9, 15.12)

Refresh: a ``time.sleep(5); st.rerun()`` loop at the end of ``main()`` keeps all
panels current within 5 seconds of new data arriving (Req 15.4, 15.10).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Optional imports — fail gracefully so the dashboard can still start.
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    import umap  # umap-learn
    _HAS_UMAP = True
except ImportError:  # pragma: no cover
    _HAS_UMAP = False

# Internal imports
from training.metrics import MetricsStore, load_metrics

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
_DEFAULT_METRICS_PATH = os.environ.get("ODYSSEY_METRICS_PATH", "artifacts/metrics.json")
_DEFAULT_ARTIFACT_DIR = os.environ.get("ODYSSEY_ARTIFACT_DIR", "artifacts")

# Artifact file names (relative to the configured artifact directory)
_REAL_FRAME_FILE = "real_frame.npy"        # (C, 64, 64) float32
_RECON_FRAME_FILE = "recon_frame.npy"      # (C, 64, 64) float32
_IMAGINED_FRAMES_FILE = "imagined_frames.npy"  # (H, C, 64, 64) float32
_LATENTS_FILE = "latents.npy"              # (N, 1024) float32, N >= 500
_PLANNED_TRAJ_FILE = "planned_trajectory.npy"  # (H, 2) float32 x,y
_TRACK_FRAME_FILE = "track_frame.npy"      # (C, 64, 64) float32


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_npy(artifact_dir: str, filename: str) -> Optional[np.ndarray]:
    """Load a .npy artifact; return None if the file does not exist."""
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:  # pragma: no cover
        return None


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a float32 frame of shape (C, H, W) to uint8 (H, W, C) for st.image().

    Values are assumed to be in [0.0, 1.0]; clipped and scaled to [0, 255].
    Grayscale (C=1) frames are squeezed to (H, W) for display.
    """
    # (C, H, W) → (H, W, C)
    frame = np.clip(frame, 0.0, 1.0)
    hwc = np.transpose(frame, (1, 2, 0))  # (H, W, C)
    uint8 = (hwc * 255).astype(np.uint8)
    if uint8.shape[2] == 1:
        # Grayscale — remove the trailing channel dimension for st.image
        return uint8[:, :, 0]
    return uint8


def _try_load_metrics(path: str) -> Optional[MetricsStore]:
    """Load metrics from JSON; return None if the file is absent or corrupt."""
    if not os.path.exists(path):
        return None
    try:
        return load_metrics(path)
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Panel 1: Real vs Reconstructed Frame  (Req 15.2, 15.12)
# ---------------------------------------------------------------------------

def _panel_real_vs_recon(artifact_dir: str) -> None:
    st.header("Real vs Reconstructed Frame")

    real = _load_npy(artifact_dir, _REAL_FRAME_FILE)
    recon = _load_npy(artifact_dir, _RECON_FRAME_FILE)

    if real is None or recon is None:
        st.info("Reconstruction data not yet available.")
        return

    col_real, col_recon = st.columns(2)
    with col_real:
        st.subheader("Real Frame")
        st.image(_frame_to_uint8(real), use_container_width=True)
    with col_recon:
        st.subheader("Reconstructed Frame")
        st.image(_frame_to_uint8(recon), use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 2: Reconstruction-loss curve  (Req 15.3, 15.4, 15.5)
# ---------------------------------------------------------------------------

def _panel_reconstruction_loss(store: Optional[MetricsStore]) -> None:
    st.header("Reconstruction-Loss Curve")

    if store is None:
        st.info("Reconstruction data not yet available.")
        return

    losses = store.reconstruction_losses()  # list of (step, value)
    if not losses:
        st.info("Reconstruction data not yet available.")
        return

    steps, values = zip(*losses)

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, values, linewidth=1.5, color="#1f77b4")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reconstruction loss")
        ax.set_title("Reconstruction Loss vs Training Steps")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    else:
        # Fallback: use built-in line chart with a dict/DataFrame-compatible input
        import pandas as pd
        df = pd.DataFrame({"Reconstruction Loss": list(values)}, index=list(steps))
        st.line_chart(df)


# ---------------------------------------------------------------------------
# Panel 3: Imagined-rollout frames  (Req 15.6, 15.12)
# ---------------------------------------------------------------------------

def _panel_imagined_rollout(artifact_dir: str) -> None:
    st.header("Imagined Rollout Frames")

    imagined = _load_npy(artifact_dir, _IMAGINED_FRAMES_FILE)

    if imagined is None:
        st.info("Imagination data not yet available.")
        return

    # Expected shape: (H, C, 64, 64)
    if imagined.ndim != 4:
        st.info("Imagination data not yet available.")
        return

    horizon = imagined.shape[0]
    st.caption(f"Horizon: {horizon} frames (in time order)")

    # Display all horizon frames in a responsive grid (up to 8 per row)
    max_cols = 8
    cols_per_row = min(horizon, max_cols)
    rows = (horizon + cols_per_row - 1) // cols_per_row

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            frame_idx = row_idx * cols_per_row + col_idx
            if frame_idx >= horizon:
                break
            with col:
                st.image(
                    _frame_to_uint8(imagined[frame_idx]),
                    caption=f"t+{frame_idx + 1}",
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Panel 4: UMAP projection  (Req 15.7)
# ---------------------------------------------------------------------------

def _panel_umap(artifact_dir: str) -> None:
    st.header("UMAP Projection of Stochastic Latents")

    latents = _load_npy(artifact_dir, _LATENTS_FILE)

    if latents is None:
        st.info("Latent data not yet available.")
        return

    if latents.ndim != 2 or latents.shape[0] < 500:
        st.info(
            f"Insufficient latent data for UMAP projection "
            f"(need ≥500 samples; have {latents.shape[0] if latents.ndim == 2 else 0})."
        )
        return

    if not _HAS_UMAP:
        st.warning("umap-learn is not installed. Install it with `pip install umap-learn`.")
        return

    if not _HAS_MPL:
        st.warning("matplotlib is not installed. Cannot render UMAP plot.")
        return

    try:
        with st.spinner("Computing UMAP projection…"):
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latents)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.5, c="#1f77b4")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"UMAP (n={latents.shape[0]} latents)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"UMAP projection failed: {exc}")


# ---------------------------------------------------------------------------
# Panel 5: Reward-comparison curves  (Req 15.8, 15.11)
# ---------------------------------------------------------------------------

def _panel_reward_comparison(store: Optional[MetricsStore], artifact_dir: str) -> None:
    st.header("Reward Comparison: Odyssey vs PPO")

    # Load PPO reward series from a separate artifact if present
    ppo_rewards = _load_npy(artifact_dir, "ppo_rewards.npy")  # optional (N,2): steps, rewards

    if store is None:
        st.info("Comparison data not yet available.")
        return

    odyssey_records = store.eval_records()
    if not odyssey_records:
        st.info("Comparison data not yet available.")
        return

    odyssey_steps = [r["env_steps"] for r in odyssey_records]
    odyssey_rewards = [r["avg_reward"] for r in odyssey_records]

    has_ppo = ppo_rewards is not None and ppo_rewards.ndim == 2 and ppo_rewards.shape[1] == 2

    if not has_ppo:
        st.info("Comparison data not yet available.")
        return

    if not _HAS_MPL:
        # Fallback to st.line_chart
        import pandas as pd
        df_ody = pd.DataFrame({"Odyssey": odyssey_rewards}, index=odyssey_steps)
        if has_ppo:
            ppo_s = ppo_rewards[:, 0].tolist()
            ppo_r = ppo_rewards[:, 1].tolist()
            df_ppo = pd.DataFrame({"PPO": ppo_r}, index=[int(s) for s in ppo_s])
            df = df_ody.join(df_ppo, how="outer")
        else:
            df = df_ody
        st.line_chart(df)
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(odyssey_steps, odyssey_rewards, label="Odyssey", linewidth=2, color="#1f77b4")
    if has_ppo:
        ax.plot(
            ppo_rewards[:, 0].tolist(),
            ppo_rewards[:, 1].tolist(),
            label="PPO",
            linewidth=2,
            color="#ff7f0e",
            linestyle="--",
        )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Episode Reward vs Environment Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel 6: Planned-trajectory overlay  (Req 15.9, 15.12)
# ---------------------------------------------------------------------------

def _panel_planned_trajectory(artifact_dir: str) -> None:
    st.header("Planned Trajectory Overlay")

    traj = _load_npy(artifact_dir, _PLANNED_TRAJ_FILE)
    track = _load_npy(artifact_dir, _TRACK_FRAME_FILE)

    if traj is None or track is None:
        st.info("Trajectory data not yet available.")
        return

    if traj.ndim != 2 or traj.shape[1] != 2:
        st.info("Trajectory data not yet available.")
        return

    if not _HAS_MPL:
        st.warning("matplotlib is not installed. Cannot render trajectory overlay.")
        return

    # Build the figure: track frame as background, trajectory as scatter/line
    track_img = _frame_to_uint8(track)  # (H, W) or (H, W, C)

    fig, ax = plt.subplots(figsize=(6, 6))
    if track_img.ndim == 2:
        ax.imshow(track_img, cmap="gray", origin="upper")
    else:
        ax.imshow(track_img, origin="upper")

    # traj is (H, 2) — x,y coordinates in image pixel space
    xs, ys = traj[:, 0], traj[:, 1]
    ax.plot(xs, ys, color="red", linewidth=2, label="Planned trajectory")
    ax.scatter(xs, ys, c=range(len(xs)), cmap="autumn", s=30, zorder=5)
    ax.scatter(xs[0], ys[0], color="lime", s=80, zorder=6, label="Start")
    ax.scatter(xs[-1], ys[-1], color="cyan", s=80, zorder=6, label="End")
    ax.legend(fontsize=8)
    ax.set_title("Planned Trajectory (time-ordered)")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

def _build_sidebar() -> tuple[str, str]:
    """Render sidebar controls; return (metrics_path, artifact_dir)."""
    st.sidebar.title("Odyssey Dashboard")
    st.sidebar.markdown("Configure data sources below.")

    metrics_path = st.sidebar.text_input(
        "Metrics JSON path",
        value=_DEFAULT_METRICS_PATH,
        help="Path to the MetricsStore JSON file produced by training.",
    )
    artifact_dir = st.sidebar.text_input(
        "Artifact directory",
        value=_DEFAULT_ARTIFACT_DIR,
        help=(
            "Directory containing .npy artifact files "
            "(real_frame.npy, recon_frame.npy, etc.). "
            "Override with ODYSSEY_ARTIFACT_DIR env var."
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Dashboard auto-refreshes every 5 seconds.")
    return metrics_path, artifact_dir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Odyssey dashboard and schedule an auto-refresh."""
    st.set_page_config(
        page_title="Odyssey Dashboard",
        page_icon="🚗",
        layout="wide",
    )

    metrics_path, artifact_dir = _build_sidebar()

    st.title("🚗 Odyssey Training Dashboard")
    st.caption(
        "Visualising world-model perception, imagination, and performance. "
        "Panels refresh automatically every 5 seconds."
    )

    # Load metrics store (may be None if not yet available)
    store = _try_load_metrics(metrics_path)

    # ------------------------------------------------------------------
    # Render all panels
    # ------------------------------------------------------------------
    with st.expander("Panel 1 – Real vs Reconstructed Frame", expanded=True):
        _panel_real_vs_recon(artifact_dir)

    with st.expander("Panel 2 – Reconstruction-Loss Curve", expanded=True):
        _panel_reconstruction_loss(store)

    with st.expander("Panel 3 – Imagined Rollout Frames", expanded=True):
        _panel_imagined_rollout(artifact_dir)

    with st.expander("Panel 4 – UMAP Projection of Latents", expanded=True):
        _panel_umap(artifact_dir)

    with st.expander("Panel 5 – Reward Comparison: Odyssey vs PPO", expanded=True):
        _panel_reward_comparison(store, artifact_dir)

    with st.expander("Panel 6 – Planned Trajectory Overlay", expanded=True):
        _panel_planned_trajectory(artifact_dir)

    # ------------------------------------------------------------------
    # Auto-refresh: sleep 5 s then trigger a full rerun (Req 15.4, 15.10)
    # ------------------------------------------------------------------
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
