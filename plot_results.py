#!/usr/bin/env python3
"""
ADiCo vs DiCo — Complete Results Plotting Script
Generates all figures for the presentation and paper.

Usage:
    python plot_results.py --dico-dir /path/to/dico_baselines/navigation \
                           --adico-dir /path/to/adico_baselines/navigation \
                           --out-dir /path/to/figures
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Color palettes
DICO_COLORS = {
    "homogeneous": "#94a3b8",
    "snd_0p3": "#60a5fa",
    "snd_0p6": "#3b82f6",
    "snd_0p8": "#2563eb",
    "snd_1": "#1d4ed8",
    "snd_1p2": "#1e40af",
    "unconstrained": "#64748b",
}
ADICO_COLORS = {
    "snd_0p3": "#34d399",
    "snd_0p6": "#10b981",
    "snd_0p8": "#059669",
    "snd_1": "#047857",
    "snd_1p2": "#065f46",
}
ALPHA_COLORS = {
    "alpha_0": "#94a3b8",
    "alpha_0p5": "#60a5fa",
    "alpha_1p0": "#10b981",
    "alpha_2p0": "#f59e0b",
}
BETA_COLORS = {
    "beta_5p0": "#10b981",
    "beta_0p01": "#f97316",
}

SND_LABELS = {
    "homogeneous": "SND=0",
    "snd_0p3": "SND=0.3",
    "snd_0p6": "SND=0.6",
    "snd_0p8": "SND=0.8",
    "snd_1": "SND=1.0",
    "snd_1p2": "SND=1.2",
    "unconstrained": "Unconstrained",
}

SND_VALUES = {
    "homogeneous": 0.0,
    "snd_0p3": 0.3,
    "snd_0p6": 0.6,
    "snd_0p8": 0.8,
    "snd_1": 1.0,
    "snd_1p2": 1.2,
    "unconstrained": -1.0,
}

FRAMES_PER_ITER = 60000


# ── Data Loading ─────────────────────────────────────────────

def find_scalars_dir(run_dir):
    """Find the scalars/ directory inside a run's nested structure."""
    # Structure: run_dir/ippo_.../ippo_.../scalars/
    candidates = glob.glob(os.path.join(run_dir, "ippo_*", "ippo_*", "scalars"))
    if candidates:
        return candidates[0]
    # Maybe one level deeper
    candidates = glob.glob(os.path.join(run_dir, "*", "ippo_*", "ippo_*", "scalars"))
    if candidates:
        return candidates[0]
    return None


def load_csv(filepath):
    """Load a step,value CSV (no header) and return arrays."""
    if not os.path.exists(filepath):
        return None, None
    data = np.loadtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    steps = data[:, 0].astype(int)
    values = data[:, 1]
    return steps, values


def load_metric_across_seeds(base_dir, snd_key, metric_csv, adico_subdir=None):
    """Load a metric across all seeds for a given SND config.
    
    Returns: frames (1D), values (2D: n_seeds × n_steps)
    """
    if adico_subdir:
        seed_parent = os.path.join(base_dir, snd_key, adico_subdir)
    else:
        seed_parent = os.path.join(base_dir, snd_key)

    if not os.path.exists(seed_parent):
        return None, None

    seed_dirs = sorted(glob.glob(os.path.join(seed_parent, "seed_*")))
    if not seed_dirs:
        return None, None

    all_values = []
    frames = None

    for sd in seed_dirs:
        scalars = find_scalars_dir(sd)
        if scalars is None:
            continue

        # Load frames counter
        f_steps, f_vals = load_csv(os.path.join(scalars, "counters_total_frames.csv"))
        if f_steps is None:
            continue

        # Load metric
        m_steps, m_vals = load_csv(os.path.join(scalars, metric_csv))
        if m_steps is None:
            continue

        # Align on steps
        common_steps = np.intersect1d(f_steps, m_steps)
        if len(common_steps) == 0:
            continue

        f_mask = np.isin(f_steps, common_steps)
        m_mask = np.isin(m_steps, common_steps)

        cur_frames = f_vals[f_mask]
        cur_values = m_vals[m_mask]

        if frames is None:
            frames = cur_frames
            all_values.append(cur_values)
        else:
            # Truncate to common length
            min_len = min(len(frames), len(cur_frames))
            if len(frames) > min_len:
                frames = frames[:min_len]
                all_values = [v[:min_len] for v in all_values]
            all_values.append(cur_values[:min_len])

    if not all_values:
        return None, None

    # Truncate all to same length
    min_len = min(len(v) for v in all_values)
    frames = frames[:min_len]
    values = np.array([v[:min_len] for v in all_values])

    return frames, values


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_mean_std(ax, frames, values, color, label, alpha=0.2, smooth_window=5):
    """Plot mean ± std with smoothing."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    if smooth_window > 1 and len(mean) > smooth_window:
        mean_s = smooth(mean, smooth_window)
        std_s = smooth(std, smooth_window)
        # Truncate frames to match
        offset = (len(mean) - len(mean_s)) // 2
        frames_s = frames[offset:offset + len(mean_s)]
    else:
        mean_s, std_s, frames_s = mean, std, frames

    ax.plot(frames_s, mean_s, color=color, label=label, linewidth=1.5)
    ax.fill_between(frames_s, mean_s - std_s, mean_s + std_s, color=color, alpha=alpha)


# ── Plot Functions ───────────────────────────────────────────

def plot_baseline_reward(dico_dir, out_dir):
    """Fig 1: DiCo baseline reward curves at each SND_des."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for snd_key in ["homogeneous", "snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2", "unconstrained"]:
        frames, values = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv"
        )
        if frames is None:
            print(f"  [skip] {snd_key}: no data")
            continue
        plot_mean_std(ax, frames, values, DICO_COLORS[snd_key], SND_LABELS[snd_key])
        print(f"  {snd_key}: {values.shape[0]} seeds, {values.shape[1]} steps")

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Episode Reward (mean)")
    ax.set_title("DiCo Baseline: Navigation Reward by Diversity Target")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_baseline_reward.png"))
    plt.close(fig)
    print("  → 01_baseline_reward.png")


def plot_baseline_snd(dico_dir, out_dir):
    """Fig 2: DiCo SND tracking — does measured SND match target?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for snd_key in ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]:
        frames, values = load_metric_across_seeds(
            dico_dir, snd_key, "eval_agents_snd.csv"
        )
        if frames is None:
            continue
        plot_mean_std(ax, frames, values, DICO_COLORS[snd_key], SND_LABELS[snd_key])

        # Draw target line
        target = SND_VALUES[snd_key]
        ax.axhline(y=target, color=DICO_COLORS[snd_key], linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Measured SND (eval)")
    ax.set_title("DiCo Baseline: Diversity Tracking")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_baseline_snd.png"))
    plt.close(fig)
    print("  → 02_baseline_snd.png")


def plot_adico_vs_dico_reward(dico_dir, adico_dir, out_dir):
    """Fig 3: ADiCo vs DiCo reward comparison at matched SND_des."""
    snd_keys = ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]

    fig, axes = plt.subplots(1, len(snd_keys), figsize=(4 * len(snd_keys), 5), sharey=True)
    if len(snd_keys) == 1:
        axes = [axes]

    for i, snd_key in enumerate(snd_keys):
        ax = axes[i]

        # DiCo
        frames_d, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv"
        )
        if frames_d is not None:
            plot_mean_std(ax, frames_d, values_d, DICO_COLORS[snd_key], "DiCo")

        # ADiCo
        frames_a, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )
        if frames_a is not None:
            plot_mean_std(ax, frames_a, values_a, ADICO_COLORS[snd_key], "ADiCo")

        ax.set_title(SND_LABELS[snd_key])
        ax.set_xlabel("Frames")
        if i == 0:
            ax.set_ylabel("Episode Reward")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("ADiCo vs DiCo: Reward at Matched Diversity Targets", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_adico_vs_dico_reward.png"), bbox_inches="tight")
    plt.close(fig)
    print("  → 03_adico_vs_dico_reward.png")


def plot_adico_vs_dico_reward_overlay(dico_dir, adico_dir, out_dir):
    """Fig 3b: ADiCo vs DiCo reward on single plot (best SND_des values)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for snd_key in ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]:
        frames_d, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv"
        )
        frames_a, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )
        if frames_d is not None:
            plot_mean_std(ax, frames_d, values_d, DICO_COLORS[snd_key],
                          f"DiCo {SND_LABELS[snd_key]}", alpha=0.1)
        if frames_a is not None:
            plot_mean_std(ax, frames_a, values_a, ADICO_COLORS[snd_key],
                          f"ADiCo {SND_LABELS[snd_key]}", alpha=0.1)

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Episode Reward (mean)")
    ax.set_title("ADiCo vs DiCo: Reward Comparison (all SND targets)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03b_adico_vs_dico_overlay.png"))
    plt.close(fig)
    print("  → 03b_adico_vs_dico_overlay.png")


def plot_snd_robustness(dico_dir, adico_dir, out_dir):
    """Fig 4: Final reward vs SND_des — robustness comparison."""
    snd_keys = ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]
    snd_vals = [SND_VALUES[k] for k in snd_keys]

    dico_means, dico_stds = [], []
    adico_means, adico_stds = [], []

    for snd_key in snd_keys:
        # DiCo: last 10% of training
        _, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv"
        )
        if values_d is not None:
            tail = values_d[:, -max(1, values_d.shape[1] // 10):]
            dico_means.append(np.mean(tail))
            dico_stds.append(np.std(np.mean(tail, axis=1)))
        else:
            dico_means.append(np.nan)
            dico_stds.append(0)

        # ADiCo
        _, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )
        if values_a is not None:
            tail = values_a[:, -max(1, values_a.shape[1] // 10):]
            adico_means.append(np.mean(tail))
            adico_stds.append(np.std(np.mean(tail, axis=1)))
        else:
            adico_means.append(np.nan)
            adico_stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(snd_vals)
    width = 0.03

    ax.errorbar(x - width, dico_means, yerr=dico_stds, fmt="o-", color="#3b82f6",
                label="DiCo", capsize=4, markersize=6, linewidth=2)
    ax.errorbar(x + width, adico_means, yerr=adico_stds, fmt="s-", color="#10b981",
                label="ADiCo", capsize=4, markersize=6, linewidth=2)

    ax.set_xlabel("SND_des (Diversity Target)")
    ax.set_ylabel("Final Reward (last 10% of training)")
    ax.set_title("Robustness to Diversity Target Choice")
    ax.legend()
    ax.set_xticks(snd_vals)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_snd_robustness.png"))
    plt.close(fig)
    print("  → 04_snd_robustness.png")


def plot_alpha_ablation(adico_dir, out_dir):
    """Fig 5: Alpha ablation at SND=1.0."""
    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_dirs = {
        "alpha_0": ("alpha_0_beta_5p0", "α=0 (DiCo)"),
        "alpha_0p5": ("alpha_0p5_beta_5p0", "α=0.5"),
        "alpha_1p0": ("alpha_1p0_beta_5p0", "α=1.0 (default)"),
        "alpha_2p0": ("alpha_2p0_beta_5p0", "α=2.0"),
    }

    for key, (subdir, label) in alpha_dirs.items():
        frames, values = load_metric_across_seeds(
            adico_dir, "snd_1", "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir=subdir
        )
        if frames is not None:
            plot_mean_std(ax, frames, values, ALPHA_COLORS[key], label)
            print(f"  alpha {key}: {values.shape[0]} seeds")
        else:
            print(f"  [skip] alpha {key}: no data")

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Episode Reward (mean)")
    ax.set_title("Ablation: Adaptation Strength α (SND_des = 1.0)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_alpha_ablation.png"))
    plt.close(fig)
    print("  → 05_alpha_ablation.png")


def plot_beta_ablation(adico_dir, out_dir):
    """Fig 6: Beta ablation at SND=1.0 — full ADiCo vs RND-only."""
    fig, ax = plt.subplots(figsize=(10, 6))

    beta_dirs = {
        "beta_5p0": ("alpha_1p0_beta_5p0", "β=5.0 (full ADiCo)"),
        "beta_0p01": ("alpha_1p0_beta_0p01", "β=0.01 (RND-only)"),
    }

    for key, (subdir, label) in beta_dirs.items():
        frames, values = load_metric_across_seeds(
            adico_dir, "snd_1", "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir=subdir
        )
        if frames is not None:
            plot_mean_std(ax, frames, values, BETA_COLORS[key], label)
        else:
            print(f"  [skip] beta {key}: no data")

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Episode Reward (mean)")
    ax.set_title("Ablation: Progress Gate β (SND_des = 1.0, α = 1.0)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_beta_ablation.png"))
    plt.close(fig)
    print("  → 06_beta_ablation.png")


def plot_diversity_weight(adico_dir, out_dir):
    """Fig 7: Mean diversity weight w(o) over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for snd_key in ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]:
        frames, values = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_diversity_weight.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )
        if frames is not None:
            plot_mean_std(ax, frames, values, ADICO_COLORS[snd_key], SND_LABELS[snd_key])

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="w(o)=1 (DiCo)")
    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Mean Diversity Weight w(o)")
    ax.set_title("ADiCo: Diversity Weight Evolution During Training")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_diversity_weight.png"))
    plt.close(fig)
    print("  → 07_diversity_weight.png")


def plot_scaling_ratio(dico_dir, adico_dir, out_dir):
    """Fig 8: Scaling ratio comparison DiCo vs ADiCo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for snd_key in ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]:
        # DiCo
        frames_d, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_scaling_ratio.csv"
        )
        if frames_d is not None:
            plot_mean_std(axes[0], frames_d, values_d, DICO_COLORS[snd_key], SND_LABELS[snd_key])

        # ADiCo
        frames_a, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_scaling_ratio.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )
        if frames_a is not None:
            plot_mean_std(axes[1], frames_a, values_a, ADICO_COLORS[snd_key], SND_LABELS[snd_key])

    axes[0].set_title("DiCo: Scaling Ratio")
    axes[1].set_title("ADiCo: Scaling Ratio")
    for ax in axes:
        ax.set_xlabel("Training Frames")
        ax.set_ylabel("Mean Scaling Ratio")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_scaling_ratio.png"))
    plt.close(fig)
    print("  → 08_scaling_ratio.png")


def plot_eval_snd_comparison(dico_dir, adico_dir, out_dir):
    """Fig 9: Eval SND — DiCo vs ADiCo at matched targets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for snd_key in ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]:
        target = SND_VALUES[snd_key]

        frames_d, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "eval_agents_snd.csv"
        )
        frames_a, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "eval_agents_snd.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )

        if frames_d is not None:
            plot_mean_std(ax, frames_d, values_d, DICO_COLORS[snd_key],
                          f"DiCo {SND_LABELS[snd_key]}", alpha=0.1)
        if frames_a is not None:
            plot_mean_std(ax, frames_a, values_a, ADICO_COLORS[snd_key],
                          f"ADiCo {SND_LABELS[snd_key]}", alpha=0.1)

        ax.axhline(y=target, color=DICO_COLORS[snd_key], linestyle=":", alpha=0.3)

    ax.set_xlabel("Training Frames")
    ax.set_ylabel("Eval SND")
    ax.set_title("Measured Diversity: DiCo vs ADiCo")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_eval_snd_comparison.png"))
    plt.close(fig)
    print("  → 09_eval_snd_comparison.png")


def plot_final_reward_table(dico_dir, adico_dir, out_dir):
    """Generate a text summary of final rewards for easy comparison."""
    lines = ["SND_des | DiCo (mean±std) | ADiCo (mean±std) | Δ"]
    lines.append("-" * 60)

    snd_keys = ["homogeneous", "snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2", "unconstrained"]

    for snd_key in snd_keys:
        _, values_d = load_metric_across_seeds(
            dico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv"
        )
        _, values_a = load_metric_across_seeds(
            adico_dir, snd_key, "collection_agents_reward_episode_reward_mean.csv",
            adico_subdir="alpha_1p0_beta_5p0"
        )

        d_str, a_str, delta_str = "N/A", "N/A", ""
        d_final, a_final = None, None

        if values_d is not None:
            tail = values_d[:, -max(1, values_d.shape[1] // 10):]
            d_final = np.mean(tail, axis=1)
            d_str = f"{np.mean(d_final):.3f} ± {np.std(d_final):.3f}"

        if values_a is not None:
            tail = values_a[:, -max(1, values_a.shape[1] // 10):]
            a_final = np.mean(tail, axis=1)
            a_str = f"{np.mean(a_final):.3f} ± {np.std(a_final):.3f}"

        if d_final is not None and a_final is not None:
            delta = np.mean(a_final) - np.mean(d_final)
            delta_str = f"{delta:+.3f}"

        lines.append(f"{SND_LABELS[snd_key]:15s} | {d_str:20s} | {a_str:20s} | {delta_str}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(out_dir, "final_reward_summary.txt"), "w") as f:
        f.write(summary)
    print("  → final_reward_summary.txt")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ADiCo vs DiCo results")
    parser.add_argument("--dico-dir", required=True, help="Path to dico_baselines/navigation/")
    parser.add_argument("--adico-dir", required=True, help="Path to adico_baselines/navigation/")
    parser.add_argument("--out-dir", default="./figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("ADiCo vs DiCo Results — Generating Figures")
    print("=" * 60)

    print("\n[1/10] Baseline reward curves...")
    plot_baseline_reward(args.dico_dir, args.out_dir)

    print("\n[2/10] Baseline SND tracking...")
    plot_baseline_snd(args.dico_dir, args.out_dir)

    print("\n[3/10] ADiCo vs DiCo reward (subplots)...")
    plot_adico_vs_dico_reward(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n[4/10] ADiCo vs DiCo reward (overlay)...")
    plot_adico_vs_dico_reward_overlay(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n[5/10] SND robustness...")
    plot_snd_robustness(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n[6/10] Alpha ablation...")
    plot_alpha_ablation(args.adico_dir, args.out_dir)

    print("\n[7/10] Beta ablation...")
    plot_beta_ablation(args.adico_dir, args.out_dir)

    print("\n[8/10] Diversity weight evolution...")
    plot_diversity_weight(args.adico_dir, args.out_dir)

    print("\n[9/10] Scaling ratio comparison...")
    plot_scaling_ratio(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n[10/10] Eval SND comparison...")
    plot_eval_snd_comparison(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n[Summary] Final reward table...")
    plot_final_reward_table(args.dico_dir, args.adico_dir, args.out_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
