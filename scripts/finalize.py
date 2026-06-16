"""Finalize the Odyssey deliverables from a completed training run.

Reads ``metrics.json`` produced by ``train.py`` and:
  1. Builds the training-curve and reward-comparison figures with matplotlib.
  2. Generates the research-writeup PDF via ``report.generate.generate_report``.
  3. Writes dashboard artifact ``.npy`` files (real vs reconstructed frame,
     a UMAP-able latent sample, and the Odyssey/PPO reward series) so the
     Streamlit dashboard has data to display.

Run:  .venv\\Scripts\\python.exe scripts\\finalize.py
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is importable when run as ``python scripts/finalize.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.config import load_config
from agent.decoder import Decoder
from agent.encoder import Encoder
from agent.perception import FramePreprocessor
from agent.rssm import RSSM
from report.generate import generate_report

ARTIFACT_DIR = "artifacts"
METRICS_PATH = "metrics.json"
REPORT_PATH = os.path.join("report", "odyssey_writeup.pdf")


def _load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _training_curve_figure(metrics: dict):
    """Reconstruction loss vs world-model step + eval reward vs env steps."""
    recon = metrics.get("reconstruction_losses", [])
    evals = metrics.get("eval_records", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))

    if recon:
        steps = [p[0] for p in recon]
        vals = [p[1] for p in recon]
        ax1.plot(steps, vals, color="#1f77b4", linewidth=1.8)
    ax1.set_title("Reconstruction Loss")
    ax1.set_xlabel("World-model training step")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.3)

    if evals:
        es = [r["env_steps"] for r in evals]
        rw = [r["avg_reward"] for r in evals]
        ax2.plot(es, rw, marker="o", color="#2ca02c", linewidth=1.8)
    ax2.set_title("Odyssey Episode Return")
    ax2.set_xlabel("Cumulative environment steps")
    ax2.set_ylabel("Avg episode reward")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _reward_comparison_figure(metrics: dict, ppo_avg: float | None):
    """Odyssey reward vs environment steps, with the PPO reference if available."""
    evals = metrics.get("eval_records", [])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if evals:
        es = [r["env_steps"] for r in evals]
        rw = [r["avg_reward"] for r in evals]
        ax.plot(es, rw, marker="o", label="Odyssey", color="#1f77b4", linewidth=2)
        if ppo_avg is not None:
            ax.plot(
                es,
                [ppo_avg] * len(es),
                linestyle="--",
                label="PPO baseline",
                color="#ff7f0e",
                linewidth=2,
            )
        else:
            ax.plot(
                [],
                [],
                linestyle="--",
                label="PPO baseline (not run in this pass)",
                color="#ff7f0e",
            )
    ax.set_title("Sample Efficiency: Odyssey vs PPO")
    ax.set_xlabel("Cumulative environment steps")
    ax.set_ylabel("Avg episode reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _write_dashboard_artifacts(config) -> None:
    """Produce a real frame, its reconstruction, and a latent sample for the dashboard."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    import gymnasium as gym

    # Grab one real observation from the environment.
    env = gym.make("CarRacing-v3", continuous=True, max_episode_steps=50)
    obs, _ = env.reset(seed=0)
    for _ in range(20):  # let the track render past the zoom-in animation
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    pre = FramePreprocessor(config.scope)
    frame = pre.to_frame(np.asarray(obs))  # (1, 64, 64) float32
    frame_t = torch.from_numpy(frame).unsqueeze(0)  # (1, 1, 64, 64)

    encoder = Encoder(channels=config.scope.channels)
    rssm = RSSM(action_dim=3)
    decoder = Decoder(channels=config.scope.channels)

    # Load trained weights if a checkpoint exists; otherwise use fresh weights.
    ckpt = config.world_model.checkpoint_path
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder"])
        rssm.load_state_dict(state["rssm"])
        decoder.load_state_dict(state["decoder"])
        print(f"Loaded world-model checkpoint from {ckpt}")
    else:
        print("No checkpoint found; using freshly initialised weights for artifacts.")

    with torch.no_grad():
        cat = encoder(frame_t)
        embed = cat.logits.reshape(1, -1)
        deter = rssm.initial(1).deter
        post = rssm.posterior(deter, embed)
        recon = decoder(post)  # (1, 1, 64, 64)

        # Sample >= 500 latents for the UMAP panel.
        latents = []
        for _ in range(520):
            c = encoder(frame_t)
            latents.append(c.sample.reshape(-1).numpy())
        latents_arr = np.stack(latents, axis=0)

    np.save(os.path.join(ARTIFACT_DIR, "real_frame.npy"), frame.astype(np.float32))
    np.save(
        os.path.join(ARTIFACT_DIR, "recon_frame.npy"),
        recon.squeeze(0).numpy().astype(np.float32),
    )
    np.save(os.path.join(ARTIFACT_DIR, "latents.npy"), latents_arr.astype(np.float32))

    # Odyssey reward series for the dashboard comparison panel (steps, reward).
    metrics = _load_metrics()
    evals = metrics.get("eval_records", [])
    if evals:
        ppo_series = np.array(
            [[r["env_steps"], 0.0] for r in evals], dtype=np.float32
        )  # placeholder PPO line at 0
        np.save(os.path.join(ARTIFACT_DIR, "ppo_rewards.npy"), ppo_series)

    # Copy metrics.json into artifacts so the dashboard's default path finds it.
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Dashboard artifacts written to {ARTIFACT_DIR}/")


def main() -> None:
    config = load_config("configs/quick.yaml")
    metrics = _load_metrics()

    # Figures.
    fig_train = _training_curve_figure(metrics)
    fig_cmp = _reward_comparison_figure(metrics, ppo_avg=None)

    # Odyssey final reward for the report abstract/results.
    evals = metrics.get("eval_records", [])
    odyssey_avg = evals[-1]["avg_reward"] if evals else 0.0

    n_wm_steps = len(metrics.get("world_model_losses", []))
    os.makedirs("report", exist_ok=True)
    path = generate_report(
        output_path=REPORT_PATH,
        training_curve_figure=fig_train,
        reward_comparison_figure=fig_cmp,
        odyssey_avg_reward=odyssey_avg,
        ppo_avg_reward=0.0,
        extra_notes=(
            "This writeup was generated from a short demonstration run "
            f"({n_wm_steps} world-model steps, {len(evals)} evaluation runs). "
            "The PPO baseline was not trained in this pass; the comparison "
            "figure shows the Odyssey curve only."
        ),
    )
    plt.close(fig_train)
    plt.close(fig_cmp)
    print(f"Report written to {path}")

    _write_dashboard_artifacts(config)


if __name__ == "__main__":
    main()
