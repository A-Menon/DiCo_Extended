#!/usr/bin/env python3
"""
DN-DiCo vs ADiCo vs DiCo — Complete Results Plotting Script
Generates all figures for the paper including three-way comparisons.

Usage:
    python plot_dndico_results.py \
        --dico-dir /path/to/dico_baselines/navigation \
        --adico-dir /path/to/adico_baselines/navigation \
        --dndico-dir /path/to/dndico_baselines/navigation \
        --out-dir /path/to/figures
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

DICO_BLUE = "#3b82f6"
ADICO_GREEN = "#10b981"
DNDICO_RED = "#ef4444"

DICO_COLORS = {
    "snd_0p3": "#93c5fd", "snd_0p6": "#60a5fa", "snd_0p8": "#3b82f6",
    "snd_1": "#2563eb", "snd_1p2": "#1d4ed8",
}
ADICO_COLORS = {
    "snd_0p3": "#6ee7b7", "snd_0p6": "#34d399", "snd_0p8": "#10b981",
    "snd_1": "#059669", "snd_1p2": "#047857",
}
DNDICO_COLORS = {
    "snd_0p3": "#fca5a5", "snd_0p6": "#f87171", "snd_0p8": "#ef4444",
    "snd_1": "#dc2626", "snd_1p2": "#b91c1c",
}
DNDICO_ALPHA_COLORS = {
    "alpha_0": "#94a3b8", "alpha_0p25": "#fca5a5",
    "alpha_0p5": "#ef4444", "alpha_1p0": "#dc2626", "alpha_2p0": "#f59e0b",
}
SND_LABELS = {
    "homogeneous": "SND=0", "snd_0p3": "SND=0.3", "snd_0p6": "SND=0.6",
    "snd_0p8": "SND=0.8", "snd_1": "SND=1.0", "snd_1p2": "SND=1.2",
    "unconstrained": "Unconstrained",
}
SND_VALUES = {
    "homogeneous": 0.0, "snd_0p3": 0.3, "snd_0p6": 0.6,
    "snd_0p8": 0.8, "snd_1": 1.0, "snd_1p2": 1.2, "unconstrained": -1.0,
}
SND_SWEEP = ["snd_0p3", "snd_0p6", "snd_0p8", "snd_1", "snd_1p2"]


def find_scalars_dir(run_dir):
    for pattern in ["ippo_*/ippo_*/scalars", "*/ippo_*/ippo_*/scalars"]:
        candidates = glob.glob(os.path.join(run_dir, pattern))
        if candidates:
            return candidates[0]
    return None

def load_csv(filepath):
    if not os.path.exists(filepath):
        return None, None
    data = np.loadtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(int), data[:, 1]

def load_metric(base_dir, snd_key, metric_csv, subdir=None):
    seed_parent = os.path.join(base_dir, snd_key, subdir) if subdir else os.path.join(base_dir, snd_key)
    if not os.path.exists(seed_parent):
        return None, None
    seed_dirs = sorted(glob.glob(os.path.join(seed_parent, "seed_*")))
    if not seed_dirs:
        return None, None
    all_values, frames = [], None
    for sd in seed_dirs:
        scalars = find_scalars_dir(sd)
        if scalars is None:
            continue
        f_steps, f_vals = load_csv(os.path.join(scalars, "counters_total_frames.csv"))
        m_steps, m_vals = load_csv(os.path.join(scalars, metric_csv))
        if f_steps is None or m_steps is None:
            continue
        common = np.intersect1d(f_steps, m_steps)
        if len(common) == 0:
            continue
        cur_frames = f_vals[np.isin(f_steps, common)]
        cur_values = m_vals[np.isin(m_steps, common)]
        if frames is None:
            frames = cur_frames
            all_values.append(cur_values)
        else:
            ml = min(len(frames), len(cur_frames))
            if len(frames) > ml:
                frames = frames[:ml]
                all_values = [v[:ml] for v in all_values]
            all_values.append(cur_values[:ml])
    if not all_values:
        return None, None
    ml = min(len(v) for v in all_values)
    return frames[:ml], np.array([v[:ml] for v in all_values])

def smooth(v, w=5):
    if len(v) < w:
        return v
    return np.convolve(v, np.ones(w)/w, mode="valid")

def plot_ms(ax, frames, values, color, label, alpha=0.2, sw=5):
    mean, std = np.mean(values, axis=0), np.std(values, axis=0)
    if sw > 1 and len(mean) > sw:
        ms, ss = smooth(mean, sw), smooth(std, sw)
        off = (len(mean) - len(ms)) // 2
        fs = frames[off:off+len(ms)]
    else:
        ms, ss, fs = mean, std, frames
    ax.plot(fs, ms, color=color, label=label, linewidth=1.5)
    ax.fill_between(fs, ms-ss, ms+ss, color=color, alpha=alpha)

def get_final(values):
    if values is None:
        return None
    tail = values[:, -max(1, values.shape[1]//10):]
    return np.mean(tail, axis=1)

REWARD_CSV = "collection_agents_reward_episode_reward_mean.csv"

# ── Plots ────────────────────────────────────────────────────

def plot_01(dico_dir, dndico_dir, out_dir):
    """DN-DiCo vs DiCo reward subplots."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for i, sk in enumerate(SND_SWEEP):
        ax = axes[i]
        f, v = load_metric(dico_dir, sk, REWARD_CSV)
        if f is not None: plot_ms(ax, f, v, DICO_BLUE, "DiCo")
        f, v = load_metric(dndico_dir, sk, REWARD_CSV, subdir="alpha_0p5")
        if f is not None: plot_ms(ax, f, v, DNDICO_RED, "DN-DiCo")
        ax.set_title(SND_LABELS[sk]); ax.set_xlabel("Frames")
        if i == 0: ax.set_ylabel("Episode Reward")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("DN-DiCo vs DiCo: Reward at Matched Diversity Targets", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn01_dndico_vs_dico_reward.png"), bbox_inches="tight")
    plt.close(fig); print("  -> dn01")

def plot_02(dico_dir, adico_dir, dndico_dir, out_dir):
    """Three-way robustness."""
    methods = [
        ("DiCo", dico_dir, None, DICO_BLUE, "o"),
        ("ADiCo", adico_dir, "alpha_1p0_beta_5p0", ADICO_GREEN, "s"),
        ("DN-DiCo", dndico_dir, "alpha_0p5", DNDICO_RED, "D"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.025
    for idx, (nm, bd, sd, c, m) in enumerate(methods):
        means, stds = [], []
        for sk in SND_SWEEP:
            _, v = load_metric(bd, sk, REWARD_CSV, subdir=sd)
            fl = get_final(v)
            means.append(np.mean(fl) if fl is not None else np.nan)
            stds.append(np.std(fl) if fl is not None else 0)
        x = np.array([SND_VALUES[k] for k in SND_SWEEP]) + (idx-1)*w
        ax.errorbar(x, means, yerr=stds, fmt=f"{m}-", color=c, label=nm, capsize=4, markersize=6, linewidth=2)
    ax.set_xlabel("SND_des"); ax.set_ylabel("Final Reward (last 10%)")
    ax.set_title("Robustness to Diversity Target Choice"); ax.legend()
    ax.set_xticks([SND_VALUES[k] for k in SND_SWEEP])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn02_three_way_robustness.png"))
    plt.close(fig); print("  -> dn02")

def plot_03(dndico_dir, out_dir):
    """DN-DiCo alpha ablation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, sub, label in [
        ("alpha_0", "alpha_0", r"$\alpha$=0 (DiCo)"),
        ("alpha_0p25", "alpha_0p25", r"$\alpha$=0.25"),
        ("alpha_0p5", "alpha_0p5", r"$\alpha$=0.5 (default)"),
        ("alpha_1p0", "alpha_1p0", r"$\alpha$=1.0"),
        ("alpha_2p0", "alpha_2p0", r"$\alpha$=2.0"),
    ]:
        f, v = load_metric(dndico_dir, "snd_1", REWARD_CSV, subdir=sub)
        if f is not None:
            plot_ms(ax, f, v, DNDICO_ALPHA_COLORS[key], label)
            print(f"    {key}: {v.shape[0]} seeds")
        else:
            print(f"    [skip] {key}")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Episode Reward (mean)")
    ax.set_title(r"DN-DiCo Ablation: $\alpha$ (SND_des = 1.0)"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn03_dndico_alpha_ablation.png"))
    plt.close(fig); print("  -> dn03")

def plot_04(dndico_dir, out_dir):
    """DN-DiCo diversity weight."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for sk in SND_SWEEP:
        f, v = load_metric(dndico_dir, sk, "collection_agents_diversity_weight.csv", subdir="alpha_0p5")
        if f is not None: plot_ms(ax, f, v, DNDICO_COLORS[sk], SND_LABELS[sk])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="w(o)=1 (DiCo)")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Mean Diversity Weight w(o)")
    ax.set_title("DN-DiCo: Diversity Weight Evolution"); ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn04_dndico_diversity_weight.png"))
    plt.close(fig); print("  -> dn04")

def plot_05(dico_dir, adico_dir, dndico_dir, out_dir):
    """Three-way scaling ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, title, bd, sd, cols in [
        (axes[0], "DiCo", dico_dir, None, DICO_COLORS),
        (axes[1], "ADiCo", adico_dir, "alpha_1p0_beta_5p0", ADICO_COLORS),
        (axes[2], "DN-DiCo", dndico_dir, "alpha_0p5", DNDICO_COLORS),
    ]:
        for sk in SND_SWEEP:
            f, v = load_metric(bd, sk, "collection_agents_scaling_ratio.csv", subdir=sd)
            if f is not None: plot_ms(ax, f, v, cols[sk], SND_LABELS[sk])
        ax.set_title(f"{title}: Scaling Ratio"); ax.set_xlabel("Training Frames")
        ax.set_ylabel("Mean Scaling Ratio"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn05_three_way_scaling_ratio.png"))
    plt.close(fig); print("  -> dn05")

def plot_06(dico_dir, adico_dir, dndico_dir, out_dir):
    """Three-way eval SND."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for nm, bd, sd, cols, ls in [
        ("DiCo", dico_dir, None, DICO_COLORS, "-"),
        ("ADiCo", adico_dir, "alpha_1p0_beta_5p0", ADICO_COLORS, "--"),
        ("DN-DiCo", dndico_dir, "alpha_0p5", DNDICO_COLORS, "-."),
    ]:
        for sk in SND_SWEEP:
            f, v = load_metric(bd, sk, "eval_agents_snd.csv", subdir=sd)
            if f is not None:
                mean = np.mean(v, axis=0)
                if len(mean) > 5:
                    ms = smooth(mean, 5); off = (len(mean)-len(ms))//2; fs = f[off:off+len(ms)]
                else:
                    ms, fs = mean, f
                ax.plot(fs, ms, color=cols[sk], label=f"{nm} {SND_LABELS[sk]}", linewidth=1.2, linestyle=ls)
    for sk in SND_SWEEP:
        ax.axhline(y=SND_VALUES[sk], color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Eval SND")
    ax.set_title("Measured Diversity: DiCo vs ADiCo vs DN-DiCo")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn06_three_way_eval_snd.png"))
    plt.close(fig); print("  -> dn06")

def plot_07(dico_dir, dndico_dir, out_dir):
    """Single-panel at SND=1.0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    f, v = load_metric(dico_dir, "snd_1", REWARD_CSV)
    if f is not None: plot_ms(ax, f, v, DICO_BLUE, "DiCo")
    f, v = load_metric(dndico_dir, "snd_1", REWARD_CSV, subdir="alpha_0p5")
    if f is not None: plot_ms(ax, f, v, DNDICO_RED, "DN-DiCo")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Episode Reward (mean)")
    ax.set_title("DN-DiCo vs DiCo at SND=1.0"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn07_dndico_vs_dico_snd1.png"))
    plt.close(fig); print("  -> dn07")

def plot_08(adico_dir, dndico_dir, out_dir):
    """Weight comparison ADiCo vs DN-DiCo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    f, v = load_metric(adico_dir, "snd_1", "collection_agents_diversity_weight.csv", subdir="alpha_1p0_beta_5p0")
    if f is not None: plot_ms(axes[0], f, v, ADICO_GREEN, "ADiCo w(o)")
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("ADiCo: Diversity Weight (SND=1.0)"); axes[0].set_xlabel("Training Frames"); axes[0].set_ylabel("Mean w(o)")
    f, v = load_metric(dndico_dir, "snd_1", "collection_agents_diversity_weight.csv", subdir="alpha_0p5")
    if f is not None: plot_ms(axes[1], f, v, DNDICO_RED, "DN-DiCo w(o)")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("DN-DiCo: Diversity Weight (SND=1.0)"); axes[1].set_xlabel("Training Frames"); axes[1].set_ylabel("Mean w(o)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dn08_weight_comparison.png"))
    plt.close(fig); print("  -> dn08")

def plot_09(dico_dir, adico_dir, dndico_dir, out_dir):
    """Final reward table."""
    lines = [f"{'SND_des':15s} | {'DiCo':22s} | {'ADiCo':22s} | {'DN-DiCo':22s}", "-"*90]
    all_keys = ["homogeneous", *SND_SWEEP, "unconstrained"]
    configs = [("DiCo", dico_dir, None), ("ADiCo", adico_dir, "alpha_1p0_beta_5p0"), ("DN-DiCo", dndico_dir, "alpha_0p5")]
    for sk in all_keys:
        parts = [f"{SND_LABELS[sk]:15s}"]
        for nm, bd, sd in configs:
            _, v = load_metric(bd, sk, REWARD_CSV, subdir=sd)
            fl = get_final(v)
            parts.append(f"{np.mean(fl):.3f} +/- {np.std(fl):.3f}" if fl is not None else "N/A")
        lines.append(" | ".join(f"{p:22s}" if i > 0 else p for i, p in enumerate(parts)))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "dndico_final_reward_summary.txt"), "w") as f:
        f.write(summary)
    print("  -> dndico_final_reward_summary.txt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dico-dir", required=True)
    p.add_argument("--adico-dir", required=True)
    p.add_argument("--dndico-dir", required=True)
    p.add_argument("--out-dir", default="./figures_dndico")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print("="*60 + "\nDN-DiCo vs ADiCo vs DiCo Results\n" + "="*60)
    print("\n[1/9] DN-DiCo vs DiCo reward subplots..."); plot_01(a.dico_dir, a.dndico_dir, a.out_dir)
    print("\n[2/9] Three-way robustness..."); plot_02(a.dico_dir, a.adico_dir, a.dndico_dir, a.out_dir)
    print("\n[3/9] DN-DiCo alpha ablation..."); plot_03(a.dndico_dir, a.out_dir)
    print("\n[4/9] DN-DiCo diversity weight..."); plot_04(a.dndico_dir, a.out_dir)
    print("\n[5/9] Three-way scaling ratio..."); plot_05(a.dico_dir, a.adico_dir, a.dndico_dir, a.out_dir)
    print("\n[6/9] Three-way eval SND..."); plot_06(a.dico_dir, a.adico_dir, a.dndico_dir, a.out_dir)
    print("\n[7/9] DN-DiCo vs DiCo at SND=1.0..."); plot_07(a.dico_dir, a.dndico_dir, a.out_dir)
    print("\n[8/9] Weight comparison..."); plot_08(a.adico_dir, a.dndico_dir, a.out_dir)
    print("\n[9/9] Final reward table..."); plot_09(a.dico_dir, a.adico_dir, a.dndico_dir, a.out_dir)
    print("\n" + "="*60 + f"\nAll figures saved to {a.out_dir}/\n" + "="*60)

if __name__ == "__main__":
    main()
