#!/usr/bin/env python3
"""
All-tasks plotting: DiCo baselines + CADiCo comparison.
Generates baseline replication figures AND CADiCo comparison figures for each task.

Usage:
    python plot_all_tasks.py --base-dir /path/to/logs --out-dir /path/to/figures
"""
import argparse, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.grid":True,
    "grid.alpha":0.3,"axes.spines.top":False,"axes.spines.right":False,"font.size":12,
    "axes.labelsize":13,"axes.titlesize":14,"legend.fontsize":10,"figure.dpi":150})

BLUE="#3b82f6"; RED="#ef4444"

TASKS = {
    "navigation": {
        "algo_glob": "ippo_*/ippo_*/scalars",
        "snd_keys": ["homogeneous","snd_0p3","snd_0p6","snd_0p8","snd_1","snd_1p2","unconstrained"],
        "snd_sweep": ["snd_0p3","snd_0p6","snd_0p8","snd_1","snd_1p2"],
        "cadico_sweep": ["snd_0p3","snd_0p6","snd_0p8","snd_1","snd_1p2"],
        "max_frames": 10e6, "title": "Navigation (IPPO)",
    },
    "dispersion": {
        "algo_glob": "maddpg_*/maddpg_*/scalars",
        "snd_keys": ["homogeneous","snd_1","snd_2","snd_3","snd_6","unconstrained"],
        "snd_sweep": ["snd_1","snd_2","snd_3","snd_6"],
        "cadico_sweep": ["snd_1","snd_2","snd_3","snd_6"],
        "max_frames": 10e6, "title": "Dispersion (MADDPG)",
    },
    "tag": {
        "algo_glob": "ippo_*/ippo_*/scalars",
        "snd_keys": ["homogeneous","snd_0p3","snd_0p6","unconstrained"],
        "snd_sweep": ["snd_0p3","snd_0p6"],
        "cadico_sweep": ["snd_0p3","snd_0p6"],
        "max_frames": 12e6, "title": "Tag (IPPO)",
    },
    "sampling": {
        "algo_glob": "iddpg_*/iddpg_*/scalars",
        "snd_keys": ["homogeneous","snd_1","snd_2","snd_3","snd_5","unconstrained"],
        "snd_sweep": ["snd_1","snd_2","snd_3","snd_5"],
        "cadico_sweep": ["snd_1","snd_2","snd_3","snd_5"],
        "max_frames": 5e6, "title": "Sampling (IDDPG)",
    },
}

SND_LABELS = {
    "homogeneous":"SND=0","snd_0p3":"SND=0.3","snd_0p6":"SND=0.6","snd_0p8":"SND=0.8",
    "snd_1":"SND=1.0","snd_1p2":"SND=1.2","snd_1p5":"SND=1.5","snd_2":"SND=2.0",
    "snd_3":"SND=3.0","snd_4":"SND=4.0","snd_5":"SND=5.0","snd_6":"SND=6.0",
    "unconstrained":"Unconstrained",
}
SND_VALUES = {
    "homogeneous":0.0,"snd_0p3":0.3,"snd_0p6":0.6,"snd_0p8":0.8,
    "snd_1":1.0,"snd_1p2":1.2,"snd_1p5":1.5,"snd_2":2.0,
    "snd_3":3.0,"snd_4":4.0,"snd_5":5.0,"snd_6":6.0,
    "unconstrained":-1.0,
}

def snd_color(sk, method="dico"):
    blues = ["#bfdbfe","#93c5fd","#60a5fa","#3b82f6","#2563eb","#1d4ed8","#1e40af","#1e3a8a"]
    reds  = ["#fecaca","#fca5a5","#f87171","#ef4444","#dc2626","#b91c1c","#991b1b","#7f1d1d"]
    grays = {"homogeneous":"#94a3b8","unconstrained":"#64748b"}
    if sk in grays: return grays[sk]
    palette = blues if method == "dico" else reds
    all_snds = sorted([k for k in SND_VALUES if k not in grays and k in SND_LABELS], key=lambda x: SND_VALUES[x])
    idx = all_snds.index(sk) if sk in all_snds else 0
    return palette[min(idx, len(palette)-1)]

RW = "collection_agents_reward_episode_reward_mean.csv"
RW_TAG_ADV = "collection_adversary_reward_episode_reward_mean.csv"

def find_scalars(run_dir, algo_glob):
    for pattern in [algo_glob, "*/" + algo_glob]:
        c = glob.glob(os.path.join(run_dir, pattern))
        if c: return c[0]
    return None

def load_csv(fp):
    if not os.path.exists(fp): return None, None
    d = np.loadtxt(fp, delimiter=",")
    if d.ndim == 1: d = d.reshape(1,-1)
    return d[:,0].astype(int), d[:,1]

def load_metric(base_dir, sk, metric_csv, algo_glob, sd=None):
    sp = os.path.join(base_dir, sk, sd) if sd else os.path.join(base_dir, sk)
    if not os.path.exists(sp): return None, None
    sds = sorted(glob.glob(os.path.join(sp, "seed_*")))
    if not sds: return None, None
    av, fr = [], None
    for s in sds:
        sc = find_scalars(s, algo_glob)
        if sc is None: continue
        fs, fv = load_csv(os.path.join(sc, "counters_total_frames.csv"))
        ms, mv = load_csv(os.path.join(sc, metric_csv))
        if fs is None or ms is None: continue
        cm = np.intersect1d(fs, ms)
        if len(cm) == 0: continue
        cf, cv = fv[np.isin(fs, cm)], mv[np.isin(ms, cm)]
        if fr is None: fr = cf; av.append(cv)
        else:
            ml = min(len(fr), len(cf))
            if len(fr) > ml: fr = fr[:ml]; av = [v[:ml] for v in av]
            av.append(cv[:ml])
    if not av: return None, None
    ml = min(len(v) for v in av)
    return fr[:ml], np.array([v[:ml] for v in av])

def smooth(v, w=5):
    if len(v) < w: return v
    return np.convolve(v, np.ones(w)/w, mode="valid")

def plot_ms(ax, fr, vl, c, lb, sw=5):
    mn, sd = np.mean(vl, axis=0), np.std(vl, axis=0)
    if sw > 1 and len(mn) > sw:
        ms, ss = smooth(mn, sw), smooth(sd, sw)
        o = (len(mn)-len(ms))//2; fs = fr[o:o+len(ms)]
    else: ms, ss, fs = mn, sd, fr
    ax.plot(fs, ms, color=c, label=lb, linewidth=1.5)
    ax.fill_between(fs, ms-ss, ms+ss, color=c, alpha=0.2)

def get_final(values):
    if values is None: return None
    t = values[:, -max(1, values.shape[1]//10):]
    return np.mean(t, axis=1)

def get_reward_csv(task):
    return RW_TAG_ADV if task == "tag" else RW

def plot_baselines(task, dico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; rw = get_reward_csv(task)
    fig, ax = plt.subplots(figsize=(10, 6))
    for sk in cfg["snd_keys"]:
        f, v = load_metric(dico_dir, sk, rw, algo)
        if f is not None:
            plot_ms(ax, f, v, snd_color(sk, "dico"), SND_LABELS.get(sk, sk))
            print(f"    {sk}: {v.shape[0]} seeds, {v.shape[1]} steps")
        else: print(f"    [skip] {sk}")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Episode Reward (mean)")
    ax.set_title(f"{cfg['title']}: DiCo Baselines"); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{task}_01_dico_baselines.png"))
    plt.close(fig); print(f"  -> {task}_01_dico_baselines.png")

def plot_comparison_subplots(task, dico_dir, cadico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; rw = get_reward_csv(task); sweep = cfg["cadico_sweep"]
    n = len(sweep)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), sharey=True)
    if n == 1: axes = [axes]
    for i, sk in enumerate(sweep):
        ax = axes[i]
        f, v = load_metric(dico_dir, sk, rw, algo)
        if f is not None: plot_ms(ax, f, v, BLUE, "DiCo")
        f, v = load_metric(cadico_dir, sk, rw, algo, sd="alpha_0p3")
        if f is not None: plot_ms(ax, f, v, RED, "CADiCo")
        ax.set_title(SND_LABELS.get(sk, sk)); ax.set_xlabel("Frames")
        if i == 0: ax.set_ylabel("Episode Reward")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{cfg['title']}: CADiCo vs DiCo", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{task}_02_cadico_vs_dico.png"), bbox_inches="tight")
    plt.close(fig); print(f"  -> {task}_02_cadico_vs_dico.png")

def plot_robustness(task, dico_dir, cadico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; rw = get_reward_csv(task); sweep = cfg["cadico_sweep"]
    fig, ax = plt.subplots(figsize=(9, 5.5)); w = 0.02
    for idx, (nm, bd, sd, c, m) in enumerate([("DiCo", dico_dir, None, BLUE, "o"), ("CADiCo", cadico_dir, "alpha_0p3", RED, "D")]):
        means, stds = [], []
        for sk in sweep:
            _, v = load_metric(bd, sk, rw, algo, sd=sd); fl = get_final(v)
            means.append(np.mean(fl) if fl is not None else np.nan)
            stds.append(np.std(fl) if fl is not None else 0)
        x = np.array([SND_VALUES[k] for k in sweep]) + (idx-0.5)*w
        ax.errorbar(x, means, yerr=stds, fmt=f"{m}-", color=c, label=nm, capsize=4, markersize=6, linewidth=2)
    ax.set_xlabel("SND_des"); ax.set_ylabel("Final Reward (last 10%)")
    ax.set_title(f"{cfg['title']}: Robustness"); ax.legend()
    ax.set_xticks([SND_VALUES[k] for k in sweep])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{task}_03_robustness.png"))
    plt.close(fig); print(f"  -> {task}_03_robustness.png")

def plot_scaling_ratio(task, dico_dir, cadico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; sweep = cfg["cadico_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, bd, sd) in zip(axes, [("DiCo", dico_dir, None), ("CADiCo", cadico_dir, "alpha_0p3")]):
        for sk in sweep:
            f, v = load_metric(bd, sk, "collection_agents_scaling_ratio.csv", algo, sd=sd)
            if f is not None: plot_ms(ax, f, v, snd_color(sk, "dico" if sd is None else "cadico"), SND_LABELS.get(sk, sk))
        ax.set_title(f"{title}: Scaling Ratio"); ax.set_xlabel("Frames"); ax.set_ylabel("Mean Scaling Ratio"); ax.legend(fontsize=8)
    fig.suptitle(cfg["title"], y=1.02); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{task}_04_scaling_ratio.png"), bbox_inches="tight")
    plt.close(fig); print(f"  -> {task}_04_scaling_ratio.png")

def plot_eval_snd(task, dico_dir, cadico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; sweep = cfg["cadico_sweep"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for sk in sweep:
        f, v = load_metric(dico_dir, sk, "eval_agents_snd.csv", algo)
        if f is not None:
            mn = np.mean(v, axis=0); ms = smooth(mn, 5) if len(mn) > 5 else mn
            o = (len(mn)-len(ms))//2; fs = f[o:o+len(ms)]
            ax.plot(fs, ms, color=snd_color(sk,"dico"), label=f"DiCo {SND_LABELS.get(sk,sk)}", linewidth=1.2)
        f, v = load_metric(cadico_dir, sk, "eval_agents_snd.csv", algo, sd="alpha_0p3")
        if f is not None:
            mn = np.mean(v, axis=0); ms = smooth(mn, 5) if len(mn) > 5 else mn
            o = (len(mn)-len(ms))//2; fs = f[o:o+len(ms)]
            ax.plot(fs, ms, color=snd_color(sk,"cadico"), label=f"CADiCo {SND_LABELS.get(sk,sk)}", linewidth=1.2, linestyle="--")
        if sk in SND_VALUES and SND_VALUES[sk] > 0:
            ax.axhline(y=SND_VALUES[sk], color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frames"); ax.set_ylabel("Eval SND")
    ax.set_title(f"{cfg['title']}: Measured Diversity"); ax.legend(ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{task}_05_eval_snd.png"))
    plt.close(fig); print(f"  -> {task}_05_eval_snd.png")

def plot_reward_table(task, dico_dir, cadico_dir, cfg, out_dir):
    algo = cfg["algo_glob"]; rw = get_reward_csv(task)
    lines = [f"{'SND_des':15s} | {'DiCo':22s} | {'CADiCo':22s}", "-"*65]
    for sk in cfg["snd_keys"]:
        parts = [f"{SND_LABELS.get(sk,sk):15s}"]
        for nm, bd, sd in [("DiCo", dico_dir, None), ("CADiCo", cadico_dir, "alpha_0p3")]:
            _, v = load_metric(bd, sk, rw, algo, sd=sd); fl = get_final(v)
            parts.append(f"{np.mean(fl):.3f} +/- {np.std(fl):.3f}" if fl is not None else "N/A")
        lines.append(" | ".join(f"{p:22s}" if i > 0 else p for i, p in enumerate(parts)))
    s = "\n".join(lines); print(f"\n{s}")
    with open(os.path.join(out_dir, f"{task}_reward_summary.txt"), "w") as f: f.write(s)
    print(f"  -> {task}_reward_summary.txt")

def plot_cross_task_robustness(base_dir, out_dir):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5*len(TASKS), 5), sharey=False)
    for ax, (task, cfg) in zip(axes, TASKS.items()):
        dico_dir = os.path.join(base_dir, "dico_baselines", task)
        cadico_dir = os.path.join(base_dir, "cadico_baselines", task)
        algo = cfg["algo_glob"]; rw = get_reward_csv(task); sweep = cfg["cadico_sweep"]; w = 0.02
        for idx, (nm, bd, sd, c, m) in enumerate([("DiCo", dico_dir, None, BLUE, "o"), ("CADiCo", cadico_dir, "alpha_0p3", RED, "D")]):
            means, stds = [], []
            for sk in sweep:
                _, v = load_metric(bd, sk, rw, algo, sd=sd); fl = get_final(v)
                means.append(np.mean(fl) if fl is not None else np.nan)
                stds.append(np.std(fl) if fl is not None else 0)
            x = np.array([SND_VALUES[k] for k in sweep]) + (idx-0.5)*w
            ax.errorbar(x, means, yerr=stds, fmt=f"{m}-", color=c, label=nm, capsize=3, markersize=5, linewidth=1.5)
        ax.set_xlabel("SND_des"); ax.set_title(cfg["title"]); ax.legend(fontsize=8)
        ax.set_xticks([SND_VALUES[k] for k in sweep])
        if ax == axes[0]: ax.set_ylabel("Final Reward (last 10%)")
    fig.suptitle("CADiCo vs DiCo: Robustness Across Tasks", y=1.02, fontsize=15)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cross_task_robustness.png"), bbox_inches="tight")
    plt.close(fig); print("  -> cross_task_robustness.png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--out-dir", default="./figures_all")
    p.add_argument("--tasks", nargs="*", default=None)
    a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    tasks_to_plot = a.tasks if a.tasks else list(TASKS.keys())
    print("="*60 + "\nCADiCo vs DiCo: All Tasks\n" + "="*60)
    for task in tasks_to_plot:
        if task not in TASKS: print(f"\n[SKIP] Unknown task: {task}"); continue
        cfg = TASKS[task]
        dico_dir = os.path.join(a.base_dir, "dico_baselines", task)
        cadico_dir = os.path.join(a.base_dir, "cadico_baselines", task)
        if not os.path.exists(dico_dir): print(f"\n[SKIP] No DiCo data for {task}"); continue
        task_out = os.path.join(a.out_dir, task); os.makedirs(task_out, exist_ok=True)
        print(f"\n{'='*60}\n{cfg['title']}\n{'='*60}")
        print(f"\n[1/6] Baseline reward curves..."); plot_baselines(task, dico_dir, cfg, task_out)
        if os.path.exists(cadico_dir):
            print(f"\n[2/6] CADiCo vs DiCo subplots..."); plot_comparison_subplots(task, dico_dir, cadico_dir, cfg, task_out)
            print(f"\n[3/6] Robustness..."); plot_robustness(task, dico_dir, cadico_dir, cfg, task_out)
            print(f"\n[4/6] Scaling ratio..."); plot_scaling_ratio(task, dico_dir, cadico_dir, cfg, task_out)
            print(f"\n[5/6] Eval SND..."); plot_eval_snd(task, dico_dir, cadico_dir, cfg, task_out)
            print(f"\n[6/6] Reward table..."); plot_reward_table(task, dico_dir, cadico_dir, cfg, task_out)
        else:
            print(f"  [No CADiCo data for {task}]")
            print(f"\n[6/6] Reward table (DiCo only)..."); plot_reward_table(task, dico_dir, cadico_dir, cfg, task_out)
    print(f"\n{'='*60}\nCross-Task Summary\n{'='*60}")
    plot_cross_task_robustness(a.base_dir, a.out_dir)
    print(f"\n{'='*60}\nAll figures saved to {a.out_dir}/\n{'='*60}")

if __name__ == "__main__": main()
