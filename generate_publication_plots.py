"""
Publishing-Ready Figure Generator for Multi-UAV Edge Computing Research.
Generates ALL publication-grade figures, tables, and curves from saved CSV data.

Figures produced:
  Fig 1  - Training Convergence Curves (all MARL algorithms)
  Fig 2  - Multi-Algorithm Return vs Episode (smoothed with 95% CI bands)
  Fig 3  - Validation Metrics vs Training Episode
  Fig 4  - Reward Component Decomposition
  Fig 5  - Benchmark Summary Table (LaTeX-ready & PNG)
  Fig 6  - Navigation Benchmark Grouped Comparison
  Fig 7  - Offloading Distribution Stacked Area Chart
  Fig 8  - Pareto Front: Latency vs Energy Trade-off
  Fig 9  - Aerodynamic Power Curve
  Fig 10 - Held-Out Generalisation (scatter + boxplot)
  Fig 11 - Per-Seed Performance Variability

Usage:
    python generate_publication_plots.py
    python generate_publication_plots.py --out_dir results/my_figures --dpi 600
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Publication-Grade Style Sheet
# ---------------------------------------------------------------------------
PALETTE = {
    "mappo_pure":      "#0284c7",   # Deep vibrant sky blue (Pure MAPPO CTDE)
    "mappo_residual":  "#8b5cf6",   # Violet (Residual MAPPO)
    "ippo":            "#e6550d",   # Rich vivid orange (IPPO DTDE)
    "ippo_residual":   "#d97706",   # Amber / Gold (Residual IPPO)
    "maddpg":          "#ec4899",   # Pink (MADDPG Off-Policy)
    "maddpg_residual": "#f43f5e",   # Rose red (Residual MADDPG)
    "cppo":            "#10b981",   # Emerald green (CPPO CTCE Upper Bound)
    "astar":           "#6b7280",   # Grey
    "astar_greedy":    "#374151",   # Dark grey
    "hybrid_zero":     "#f59e0b",   # Amber
    "hybrid_res02":    "#84cc16",   # Lime
    "hybrid_res05":    "#14b8a6",   # Teal
    "apf_only":        "#ef4444",   # Red
    "rrt_star":        "#a855f7",   # Purple
    "sapf":            "#fb923c",   # Light orange
}

DISPLAY_NAMES = {
    "mappo_pure":      "Pure MAPPO (CTDE)",
    "mappo_residual":  "Residual MAPPO",
    "ippo":            "IPPO (DTDE)",
    "ippo_residual":   "Residual IPPO",
    "maddpg":          "MADDPG (Off-Policy)",
    "maddpg_residual": "Residual MADDPG",
    "cppo":            "CPPO (CTCE Bound)",
    "astar":           "3D A*",
    "astar_greedy":    "A* + Greedy MEC",
}

LINESTYLES = {
    "mappo_pure":      "-",
    "mappo_residual":  "-",
    "ippo":            "--",
    "ippo_residual":   "--",
    "maddpg":          "-.",
    "maddpg_residual": "-.",
    "cppo":            ":",
    "astar":           (0, (5, 2)),
}

MARKERS = {
    "mappo_pure":      "o",
    "mappo_residual":  "s",
    "ippo":            "^",
    "ippo_residual":   "^",
    "maddpg":          "D",
    "maddpg_residual": "D",
    "cppo":            "v",
    "astar":           "*",
}

def set_pub_style(usetex: bool = False):
    """Apply a consistent, journal-ready matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d1d5db",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#222222",
        "grid.color": "#e5e7eb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "figure.constrained_layout.use": True,
        "text.usetex": usetex,
    })


def rolling_smooth(arr, window: int = 30):
    """Pandas rolling mean smoothing matching the publication baseline."""
    s = pd.Series(arr, dtype=float)
    w = min(window, max(3, len(s) // 8))
    return s.rolling(window=w, min_periods=1).mean().values


def smooth(arr, window: int = 15):
    """Box-car (uniform) smoothing — pure numpy, no scipy needed."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Pad with edge values to avoid boundary shrinkage
    pad = window // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    smoothed = np.convolve(padded, kernel, mode="valid")
    # Trim to original length
    return smoothed[:len(arr)]


def ci95(arr2d):
    """Return mean ± 1.96*SEM across rows."""
    m = arr2d.mean(axis=0)
    s = arr2d.std(axis=0, ddof=1)
    n = arr2d.shape[0]
    return m, s / np.sqrt(n) * 1.96


def save(fig, path, tight=True, formats=("png", "pdf")):
    for fmt in formats:
        p = path.rsplit(".", 1)[0] + f".{fmt}"
        fig.savefig(p, bbox_inches="tight" if tight else None, dpi=300)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------
def load_training_logs(res_dir: str):
    """Load all available algorithm training logs into a dict."""
    files = {
        "mappo_pure":      "training_logs_mappo_pure.csv",
        "mappo_residual":  "training_logs_mappo_residual.csv",
        "ippo":            "training_logs_ippo_pure.csv",
        "ippo_residual":   "training_logs_ippo_residual.csv",
        "maddpg":          "training_logs_maddpg_pure.csv",
        "maddpg_residual": "training_logs_maddpg_residual.csv",
        "cppo":            "training_logs_cppo.csv",
    }
    dfs = {}
    for key, fn in files.items():
        p = os.path.join(res_dir, fn)
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs[key] = df
            print(f"  [loaded] {fn}  ({len(df)} episodes)")
    return dfs


def parse_mean_std(col):
    """Parse '12.3 ± 0.4' or '12.3 ?? 0.4' or '12.3' into (mean, std)."""
    out_m, out_s = [], []
    for v in col:
        s = str(v)
        for sep in ["±", "??", "+-", "+/-", "\ufffd"]:
            if sep in s:
                parts = s.split(sep, 1)
                try:
                    out_m.append(float(parts[0].strip()))
                    out_s.append(float(parts[1].strip()))
                except Exception:
                    out_m.append(float("nan"))
                    out_s.append(0.0)
                break
        else:
            try:
                out_m.append(float(s.strip()))
                out_s.append(0.0)
            except Exception:
                out_m.append(float("nan"))
                out_s.append(0.0)
    return np.array(out_m), np.array(out_s)


# ===========================================================================
# FIGURE 1 — Training Convergence: Return vs Episode (all algorithms)
# ===========================================================================
def fig_training_return(dfs: dict, fig_dir: str, window: int = 20):
    print("[Fig 1] Training Return vs Episode …")
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, df in dfs.items():
        if "return" not in df.columns:
            continue
        ep = df["episode"].values
        raw = df["return"].values
        sm  = smooth(raw, window=window) if window > 1 else raw
        col = PALETTE.get(key, "#555555")
        ls  = LINESTYLES.get(key, "-")
        lab = DISPLAY_NAMES.get(key, key.upper())
        ax.plot(ep, sm, color=col, linestyle=ls, linewidth=1.6 if window <= 1 else 2.0, label=lab, zorder=3)
        if window > 1:
            # Light raw curve behind
            ax.plot(ep, raw, color=col, linestyle=ls, linewidth=0.5, alpha=0.25, zorder=2)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return" + (" (per episode)" if window <= 1 else " (smoothed)"))
    ax.set_title("Fig 1 · Training Convergence — Episode Return")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True)
    save(fig, os.path.join(fig_dir, "fig01_training_return.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 2 — Completion Rate & Arrival Rate vs Episode
# ===========================================================================
def fig_training_rates(dfs: dict, fig_dir: str, window: int = 20):
    print("[Fig 2] Training Completion/Arrival Rate vs Episode …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_pairs = [
        ("completion_rate", "Task Completion Rate (%)", axes[0]),
        ("arrival_rate",    "UAV Arrival Rate (%)",     axes[1]),
    ]

    for col_name, ylabel, ax in metric_pairs:
        for key, df in dfs.items():
            if col_name not in df.columns:
                continue
            ep  = df["episode"].values
            raw = df[col_name].values
            sm  = smooth(raw, window=window) if window > 1 else raw
            c   = PALETTE.get(key, "#555555")
            ls  = LINESTYLES.get(key, "-")
            ax.plot(ep, sm, color=c, linestyle=ls, linewidth=1.6 if window <= 1 else 2.0,
                    label=DISPLAY_NAMES.get(key, key.upper()), zorder=3)
            if window > 1:
                ax.plot(ep, raw, color=c, linestyle=ls, linewidth=0.4, alpha=0.2)

        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", ncol=2)
        ax.grid(True)

    axes[0].set_title("Fig 2a · Task Completion Rate")
    axes[1].set_title("Fig 2b · UAV Arrival Rate")
    fig.suptitle("Fig 2 · Training Efficiency Metrics", fontsize=13, fontweight="bold", y=1.01)
    save(fig, os.path.join(fig_dir, "fig02_training_rates.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 2X — STANDALONE WIDE-FORMAT PUBLICATION CURVES (User Aesthetic)
# ===========================================================================
def plot_wide_publication_curve(
    dfs: dict,
    metric_col: str,
    ylabel: str,
    title_suffix: str,
    out_path: str,
    legend_loc: str = "upper right",
    target_algorithms: list = None,
    ylim: tuple = None,
    y_step: float = None,
    window: int = 30,
):
    """
    Renders a standalone, high-resolution wide-format publication curve matching
    the exact aesthetic of the user-uploaded 1000-episode / multi-UAV benchmarks:
    - Wide aspect ratio: (10.5 x 5.0)
    - 4-border frame with dark slate/black spines
    - Dashed light-gray grid lines with major and minor step locators
    - Raw noisy episodic trace (alpha=0.22, lw=0.75)
    - Bold prominent smoothed line (alpha=1.0, lw=2.4, rolling w=30)
    - Clean boxed legend with subtle border
    """
    if target_algorithms is None:
        target_keys = list(dfs.keys())
    else:
        target_keys = [k for k in target_algorithms if k in dfs]

    valid_keys = [k for k in target_keys if metric_col in dfs[k].columns and len(dfs[k]) > 0]
    if not valid_keys:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)

    # All 4 spines visible
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#222222")
        s.set_linewidth(0.85)

    ax.tick_params(direction="out", length=4.5, width=0.85, colors="#222222", labelsize=10.5)
    ax.tick_params(which="minor", direction="out", length=2.5, width=0.6, colors="#555555")

    # Grid
    ax.grid(True, which="major", linestyle="--", color="#e5e7eb", linewidth=0.75, alpha=0.8)
    ax.grid(True, which="minor", linestyle="--", color="#f3f4f6", linewidth=0.55, alpha=0.6)

    max_ep = max(len(dfs[k]) for k in valid_keys)
    if max_ep >= 800:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ep_prefix = "1000-Episode" if max_ep >= 950 else f"{max_ep}-Episode"
    elif max_ep >= 400:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ep_prefix = f"{max_ep}-Episode"
    elif max_ep >= 150:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
        ep_prefix = f"{max_ep}-Episode"
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ep_prefix = f"{max_ep}-Episode"

    for k in valid_keys:
        df = dfs[k]
        ep = df["episode"].values
        raw = df[metric_col].values.astype(float)
        if metric_col == "total_energy":
            raw = raw / 1000.0  # Convert to kJ
        
        c = PALETTE.get(k, "#0284c7")
        ls = LINESTYLES.get(k, "-")
        lab = DISPLAY_NAMES.get(k, k)

        if window <= 1:
            smooth_val = raw
            ax.plot(ep, raw, color=c, linestyle=ls, linewidth=1.5, label=lab, zorder=3)
        else:
            w = min(window, max(3, len(raw) // 10))
            smooth_val = pd.Series(raw).rolling(window=w, min_periods=1).mean().values
            # Raw trace (faint background)
            ax.plot(ep, raw, color=c, linestyle=ls, linewidth=0.65, alpha=0.18, zorder=2)
            # Smoothed line (bold foreground)
            ax.plot(ep, smooth_val, color=c, linestyle=ls, linewidth=2.3, label=lab, zorder=3)

    suffix_label = title_suffix if window > 1 else title_suffix.replace("(smoothed w=30)", "(Per-Episode)")
    title_text = f"{ep_prefix} Training: {suffix_label}"
    ax.set_title(title_text, fontsize=13, pad=10, fontfamily="sans-serif")
    ax.set_xlabel("Training Episode", fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel(ylabel, fontsize=11, fontfamily="sans-serif")

    if ylim:
        ax.set_ylim(ylim)
    if y_step:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))

    ncols = 3 if len(valid_keys) >= 5 else (2 if len(valid_keys) >= 3 else 1)
    leg = ax.legend(loc=legend_loc, ncol=ncols, frameon=True, framealpha=0.95, edgecolor="#d1d5db", fontsize=9.5)
    leg.get_frame().set_linewidth(0.8)

    save(fig, out_path)
    plt.close(fig)


def fig_clean_wide_suite(dfs: dict, fig_dir: str, window: int = 30):
    """Generates the full suite of wide-format single-panel publication figures with ALL models."""
    print("[Wide Figures] Generating wide-format publication curves for ALL MARL models …")
    
    # Target all MARL algorithms present in logs
    all_marl_keys = [k for k in ["mappo_pure", "mappo_residual", "ippo", "maddpg", "cppo", "ippo_residual", "maddpg_residual"] if k in dfs]
    
    # 1. Task Completion Rate (ALL MARL Models)
    p_comp = os.path.join(fig_dir, "fig_task_completion_rate_wide.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="completion_rate",
        ylabel="Task Completion Rate (%)",
        title_suffix="Task Completion Rate",
        out_path=p_comp,
        legend_loc="lower left",
        target_algorithms=all_marl_keys,
        ylim=(55, 101),
        y_step=10,
        window=window,
    )
    
    # 1b. Task Completion Rate (Pure MAPPO vs IPPO pairwise comparison)
    p_comp_pair = os.path.join(fig_dir, "fig_task_completion_rate_pure_vs_ippo.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="completion_rate",
        ylabel="Task Completion Rate (%)",
        title_suffix="Task Completion Rate (MAPPO vs IPPO)",
        out_path=p_comp_pair,
        legend_loc="lower left",
        target_algorithms=["mappo_pure", "ippo"],
        ylim=(55, 101),
        y_step=10,
        window=window,
    )

    # 2. Episode Return (ALL MARL Models)
    p_ret = os.path.join(fig_dir, "fig_episode_return_wide.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="return",
        ylabel="Episode Return",
        title_suffix="Episode Return (smoothed w=30)",
        out_path=p_ret,
        legend_loc="upper right",
        target_algorithms=all_marl_keys,
        y_step=250,
        window=window,
    )

    # 2b. Episode Return (Pure MAPPO vs IPPO pairwise comparison)
    p_ret_pair = os.path.join(fig_dir, "fig_episode_return_pure_vs_ippo.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="return",
        ylabel="Episode Return",
        title_suffix="Episode Return (MAPPO vs IPPO, smoothed w=30)",
        out_path=p_ret_pair,
        legend_loc="upper right",
        target_algorithms=["mappo_pure", "ippo"],
        y_step=250,
        window=window,
    )

    # 3. Fleet Energy (ALL MARL Models)
    p_ene = os.path.join(fig_dir, "fig_fleet_energy_wide.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="total_energy",
        ylabel="Total Propulsion & Comm Energy (kJ)",
        title_suffix="Fleet Mechanical & Comm Energy (smoothed w=30)",
        out_path=p_ene,
        legend_loc="upper right",
        target_algorithms=all_marl_keys,
        window=window,
    )

    # 4. Safety Interventions (ALL MARL Models)
    p_safe = os.path.join(fig_dir, "fig_safety_interventions_wide.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="safety_interventions",
        ylabel="Interventions per Episode",
        title_suffix="Safety Filter Interventions (smoothed w=30)",
        out_path=p_safe,
        legend_loc="upper right",
        target_algorithms=all_marl_keys,
        window=window,
    )

    # 5. UAV Navigation Arrival Rate (ALL MARL Models)
    p_arr = os.path.join(fig_dir, "fig_arrival_rate_wide.png")
    plot_wide_publication_curve(
        dfs,
        metric_col="arrival_rate",
        ylabel="UAV Arrival Rate (%)",
        title_suffix="UAV Navigation Arrival Rate (smoothed w=30)",
        out_path=p_arr,
        legend_loc="lower right",
        target_algorithms=all_marl_keys,
        ylim=(-2, 105),
        y_step=20,
        window=window,
    )


# ===========================================================================
# FIGURE 3 — Training Energy & Safety Interventions
# ===========================================================================
def fig_training_energy_safety(dfs: dict, fig_dir: str, window: int = 20):
    print("[Fig 3] Training Energy & Safety Interventions …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, df in dfs.items():
        c  = PALETTE.get(key, "#555555")
        ls = LINESTYLES.get(key, "-")
        lab= DISPLAY_NAMES.get(key, key.upper())
        ep = df["episode"].values

        if "total_energy" in df.columns:
            raw_e = df["total_energy"].values / 1000.0
            sm = smooth(raw_e, window=window) if window > 1 else raw_e
            axes[0].plot(ep, sm, color=c, linestyle=ls, linewidth=1.6 if window <= 1 else 2.0, label=lab)
            if window > 1:
                axes[0].plot(ep, raw_e, color=c, linestyle=ls, linewidth=0.4, alpha=0.2)

        if "safety_interventions" in df.columns:
            raw_s = df["safety_interventions"].values.astype(float)
            sm2 = smooth(raw_s, window=window) if window > 1 else raw_s
            axes[1].plot(ep, sm2, color=c, linestyle=ls, linewidth=1.6 if window <= 1 else 2.0, label=lab)
            if window > 1:
                axes[1].plot(ep, raw_s, color=c, linestyle=ls, linewidth=0.4, alpha=0.2)

    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Total Energy per Episode (kJ)")
    axes[0].set_title("Fig 3a · Fleet Energy Consumption")
    axes[0].legend(ncol=2)
    axes[0].grid(True)

    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Safety Interventions per Episode")
    axes[1].set_title("Fig 3b · Safety Filter Activations")
    axes[1].legend(ncol=2)
    axes[1].grid(True)

    fig.suptitle("Fig 3 · Energy & Safety During Training", fontsize=13, fontweight="bold", y=1.01)
    save(fig, os.path.join(fig_dir, "fig03_training_energy_safety.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 4 — Validation Curves (from training logs)
# ===========================================================================
def fig_validation_curves(dfs: dict, fig_dir: str, window: int = 10):
    print("[Fig 4] Validation Curves …")
    has_val = {k: df for k, df in dfs.items() if "val_return" in df.columns
               and df["val_return"].notna().any()}
    if not has_val:
        print("  [skip] No validation data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        ("val_return",          "Validation Return",          axes[0]),
        ("val_completion_rate", "Val. Completion Rate (%)",   axes[1]),
        ("val_arrival_rate",    "Val. Arrival Rate (%)",      axes[2]),
    ]

    for col_name, ylabel, ax in pairs:
        for key, df in has_val.items():
            if col_name not in df.columns:
                continue
            mask = df[col_name].notna()
            ep   = df["episode"].values[mask]
            raw  = df[col_name].values[mask].astype(float)
            if len(raw) < 2:
                continue
            sm = smooth(raw, window=window) if window > 1 else raw
            c  = PALETTE.get(key, "#555555")
            ls = LINESTYLES.get(key, "-")
            ax.plot(ep, sm, color=c, linestyle=ls, linewidth=1.6 if window <= 1 else 2.0,
                    marker="o" if len(ep) <= 100 else None, markersize=3.5,
                    label=DISPLAY_NAMES.get(key, key.upper()))
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=1)
        ax.grid(True)

    axes[0].set_title("Fig 4a · Validation Return")
    axes[1].set_title("Fig 4b · Val. Task Completion")
    axes[2].set_title("Fig 4c · Val. Arrival Rate")
    fig.suptitle("Fig 4 · Held-Out Validation Curves During Training",
                 fontsize=13, fontweight="bold", y=1.01)
    save(fig, os.path.join(fig_dir, "fig04_validation_curves.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 5 — Reward Component Decomposition (MAPPO Pure)
# ===========================================================================
def fig_reward_decomposition(dfs: dict, fig_dir: str, window: int = 20):
    print("[Fig 5] Reward Component Decomposition …")
    if "mappo_pure" not in dfs:
        print("  [skip] mappo_pure not found.")
        return
    df = dfs["mappo_pure"]

    reward_cols = {
        "rew_progress": ("Progress",   "#0ea5e9"),
        "rew_arrival":  ("Arrival",    "#10b981"),
        "rew_task":     ("Task",       "#f59e0b"),
        "rew_energy":   ("Energy",     "#ef4444"),
        "rew_safety":   ("Safety",     "#8b5cf6"),
        "rew_deadline": ("Deadline",   "#ec4899"),
        "rew_delay":    ("Delay",      "#6366f1"),
    }
    present = [(c, lbl, col) for c, (lbl, col) in reward_cols.items() if c in df.columns]
    if not present:
        print("  [skip] No reward component columns found.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ep = df["episode"].values
    for col, lbl, color in present:
        raw_r = df[col].values.astype(float)
        sm = smooth(raw_r, window=window) if window > 1 else raw_r
        ax.plot(ep, sm, color=color, linewidth=1.4 if window <= 1 else 1.8, label=lbl)

    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward Component" + (" (per episode)" if window <= 1 else " (smoothed)"))
    ax.set_title("Fig 5 · Reward Component Decomposition — Pure MAPPO")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(True)
    save(fig, os.path.join(fig_dir, "fig05_reward_decomposition.png"))
    plt.close(fig)
    save(fig, os.path.join(fig_dir, "fig05_reward_decomposition.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 6 — Benchmark Summary Table (publication-quality PNG)
# ===========================================================================
def fig_benchmark_table(res_dir: str, fig_dir: str):
    print("[Fig 6] Benchmark Summary Table …")
    bench_csv = os.path.join(res_dir, "benchmark_summary.csv")
    if not os.path.exists(bench_csv):
        print("  [skip] benchmark_summary.csv not found.")
        return

    df = pd.read_csv(bench_csv)

    # Clean encoding artifacts (ÃÂ± → ±)
    _clean = lambda x: str(x).replace("A\ufffd", "±").replace("??", "±").replace("Ã±", "±").replace("A±", "±") if isinstance(x, str) else x
    df = df.apply(lambda col: col.map(_clean))

    # Choose display columns
    display_cols = [
        "Method", "UAV Success (%)", "Fleet Success (%)", "Arrival (s)",
        "Total Energy (kJ)", "Nav Energy (kJ)", "Tasks Done (%)",
        "Delay (s)", "Interventions", "Collisions"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_disp = df[display_cols].copy()

    # Shorten method names for table
    def shorten(name):
        name = str(name)
        replacements = [
            ("Classical 3D A* + Greedy MEC", "A* + Greedy MEC"),
            ("Classical 3D A* + MAPPO Offload (k_rl=0.0)", "A* + MAPPO (k=0.0)"),
            ("Hierarchical 3D A* + MAPPO Offload (k_rl=0.2)", "A* + MAPPO (k=0.2)"),
            ("Hierarchical 3D A* + MAPPO Offload (k_rl=0.5)", "A* + MAPPO (k=0.5)"),
            ("APF-Assisted Residual MAPPO (with Terminal Damping)", "Residual MAPPO"),
            ("Proposed Pure MAPPO (8.4k updates, Zero APF Guidance)", "Pure MAPPO ★"),
            ("Independent PPO (IPPO, DTDE)", "IPPO"),
            ("Multi-Agent DDPG (MADDPG)", "MADDPG"),
            ("Centralized PPO (CPPO, CTCE)", "CPPO"),
            ("APF-Only Controller (Gain 2.0, Zero Learning)", "APF-Only"),
            ("3D A* + Pure Pursuit (Zero RL Baseline)", "A* Pure Pursuit"),
            ("APF-Assisted Residual MAPPO (Undamped / Historical Orbiting)", "Residual MAPPO (Undamped)"),
        ]
        for long, short in replacements:
            if long in name:
                return short
        # generic truncation
        return name[:35] + "…" if len(name) > 35 else name

    df_disp["Method"] = df_disp["Method"].apply(shorten)

    n_rows, n_cols = df_disp.shape
    row_h = 0.42
    fig_h = max(3.5, n_rows * row_h + 1.8)
    fig_w = max(14, n_cols * 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = df_disp.columns.tolist()
    table_data = df_disp.values.tolist()

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)

    # Style header
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor("#f0f4f8" if i % 2 == 0 else "white")
            cell.set_edgecolor("#cccccc")

    # Highlight proposed method row (★)
    for i, row in enumerate(table_data):
        if "★" in str(row[0]) or "Pure MAPPO" in str(row[0]):
            for j in range(n_cols):
                tbl[i + 1, j].set_facecolor("#fef3c7")
                tbl[i + 1, j].set_text_props(fontweight="bold")

    ax.set_title("Fig 6 · Benchmark Summary Table (Mean ± Std, 5 Test Seeds)",
                 fontsize=12, fontweight="bold", pad=10)

    save(fig, os.path.join(fig_dir, "fig06_benchmark_table.png"), formats=("png",))
    plt.close(fig)

    # Also export a clean CSV
    df_disp.to_csv(os.path.join(fig_dir, "fig06_benchmark_table.csv"), index=False)
    print("  [saved] fig06_benchmark_table.csv")


# ===========================================================================
# FIGURE 7 — Navigation Benchmark Grouped Bar (actual metrics)
# ===========================================================================
def fig_navigation_benchmark(res_dir: str, fig_dir: str):
    print("[Fig 7] Navigation Benchmark …")
    nav_csv = os.path.join(res_dir, "navigation_benchmark.csv")
    if not os.path.exists(nav_csv):
        print("  [skip] navigation_benchmark.csv not found.")
        return

    df = pd.read_csv(nav_csv)
    if "Algorithm" not in df.columns:
        print("  [skip] unexpected format.")
        return

    df["ShortName"] = df["Algorithm"].apply(
        lambda x: x.replace("3D ", "").replace("Hierarchical ", "")
                   .replace(" (Zero RL Baseline)", "").replace(" (Zero RL)", "")
                   .replace("Hybrid ", "")
                   [:30]
    )

    metrics = []
    if "Nav Energy (kJ)" in df.columns:
        metrics.append(("Nav Energy (kJ)", "Nav. Energy (kJ)"))
    if "Arrival Time (s)" in df.columns:
        metrics.append(("Arrival Time (s)", "Arrival Time (s)"))
    if "Mean Path Length (m)" in df.columns:
        metrics.append(("Mean Path Length (m)", "Path Length (m)"))

    if not metrics:
        print("  [skip] no recognised metric columns.")
        return

    n = len(df)
    # Assign colours per row
    row_colors = []
    color_map = {
        "A*": "#6b7280", "MAPPO": "#0ea5e9", "IPPO": "#f97316",
        "CPPO": "#10b981", "APF": "#ef4444", "RRT": "#a855f7",
        "SAPF": "#fb923c", "Hybrid": "#f59e0b",
    }
    for _, row in df.iterrows():
        name = str(row["Algorithm"])
        c = "#0284c7"
        for k, v in color_map.items():
            if k.lower() in name.lower():
                c = v
                break
        row_colors.append(c)

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(5.5 * n_met, 5))
    if n_met == 1:
        axes = [axes]

    x = np.arange(n)
    for ax, (col, ylabel) in zip(axes, metrics):
        vals = df[col].values.astype(float)
        bars = ax.bar(x, vals, color=row_colors, edgecolor="white", linewidth=0.7, zorder=3)
        # Value annotations
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df["ShortName"].values, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_title("Fig 7 · Navigation Benchmark Comparison")
    fig.suptitle("Fig 7 · Navigation Performance Across Methods",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, os.path.join(fig_dir, "fig07_navigation_benchmark.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 8 — Offloading Distribution Stacked Area / Bar
# ===========================================================================
def fig_offloading_distribution(res_dir: str, fig_dir: str):
    print("[Fig 8] Offloading Distribution …")
    off_csv = os.path.join(res_dir, "offloading_distribution.csv")
    if not os.path.exists(off_csv):
        print("  [skip] offloading_distribution.csv not found.")
        return

    df = pd.read_csv(off_csv)
    if not {"Method", "Local (%)", "Peer (%)", "Edge (%)"}.issubset(df.columns):
        print("  [skip] unexpected columns.")
        return

    methods = df["Method"].apply(
        lambda x: x.replace("Classical 3D A* + ", "A*: ")
                   .replace("Hierarchical 3D A* + ", "H-A*: ")
                   [:40]
    ).tolist()
    local = df["Local (%)"].values.astype(float)
    peer  = df["Peer (%)"].values.astype(float)
    edge  = df["Edge (%)"].values.astype(float)

    n = len(methods)
    x = np.arange(n)
    w = 0.6

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    b_local = ax.bar(x, local, w, label="Local Execution", color="#0ea5e9",   zorder=3)
    b_peer  = ax.bar(x, peer,  w, label="Peer Offloading", color="#8b5cf6", bottom=local, zorder=3)
    b_edge  = ax.bar(x, edge,  w, label="Edge Offloading", color="#f97316", bottom=local + peer, zorder=3)

    for bar, val in zip(b_local, local):
        if val > 3:
            ax.text(bar.get_x() + w/2, bar.get_y() + val/2,
                    f"{val:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for bar, val, bot in zip(b_peer, peer, local):
        if val > 3:
            ax.text(bar.get_x() + w/2, bot + val/2,
                    f"{val:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for bar, val, bot in zip(b_edge, edge, local + peer):
        if val > 3:
            ax.text(bar.get_x() + w/2, bot + val/2,
                    f"{val:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Offloading Decisions (%)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Fig 8 · Task Offloading Distribution Breakdown")
    save(fig, os.path.join(fig_dir, "fig08_offloading_distribution.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 9 — Pareto Front: Total Delay vs Total Energy
# ===========================================================================
def fig_pareto_front(res_dir: str, fig_dir: str):
    print("[Fig 9] Pareto Front: Delay vs Energy …")
    bench_csv = os.path.join(res_dir, "benchmark_summary.csv")
    if not os.path.exists(bench_csv):
        print("  [skip] benchmark_summary.csv not found.")
        return

    df = pd.read_csv(bench_csv)

    delay_col = next((c for c in ["Delay (s)", "Total Delay (s)"] if c in df.columns), None)
    energy_col = next((c for c in ["Total Energy (kJ)", "Nav Energy (kJ)"] if c in df.columns), None)

    if not delay_col or not energy_col:
        print("  [skip] needed columns not found.")
        return

    means_d, stds_d = parse_mean_std(df[delay_col])
    means_e, stds_e = parse_mean_std(df[energy_col])

    methods = df["Method"].apply(
        lambda x: str(x).replace("Classical 3D A* + Greedy MEC", "A* Greedy")
                        .replace("Proposed Pure MAPPO (8.4k updates, Zero APF Guidance)", "Pure MAPPO ★")
                        .replace("APF-Assisted Residual MAPPO (with Terminal Damping)", "Residual MAPPO")
                        .replace("Independent PPO (IPPO, DTDE)", "IPPO")
                        .replace("Multi-Agent DDPG (MADDPG)", "MADDPG")
                        .replace("Centralized PPO (CPPO, CTCE)", "CPPO")
                        .replace("APF-Only Controller (Gain 2.0, Zero Learning)", "APF-Only")
                        .replace("Classical 3D A* + MAPPO Offload (k_rl=0.0)", "A*+MAPPO(k=0)")
                        .replace("Hierarchical 3D A* + MAPPO Offload (k_rl=0.2)", "A*+MAPPO(k=0.2)")
                        .replace("Hierarchical 3D A* + MAPPO Offload (k_rl=0.5)", "A*+MAPPO(k=0.5)")
                        .replace("3D A* + Pure Pursuit (Zero RL Baseline)", "A* Pure Pursuit")
                        .replace("APF-Assisted Residual MAPPO (Undamped / Historical Orbiting)", "Res.MAPPO(Undamp)")
    ).tolist()

    # Assign colours
    def get_color(name):
        if "MAPPO ★" in name or "Pure MAPPO" in name:
            return PALETTE["mappo_pure"], 220, "*"
        if "Residual MAPPO" in name:
            return PALETTE["mappo_residual"], 160, "s"
        if "IPPO" in name:
            return PALETTE["ippo"], 100, "^"
        if "MADDPG" in name:
            return PALETTE["maddpg"], 100, "D"
        if "CPPO" in name:
            return PALETTE["cppo"], 100, "v"
        if "APF-Only" in name:
            return PALETTE["apf_only"], 100, "x"
        if "A*" in name:
            return PALETTE["astar"], 100, "o"
        return "#888888", 100, "o"

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (name, md, sd, me, se) in enumerate(
            zip(methods, means_d, stds_d, means_e, stds_e)):
        if np.isnan(md) or np.isnan(me):
            continue
        color, sz, mk = get_color(name)
        ax.errorbar(md, me, xerr=sd, yerr=se,
                    fmt=mk, color=color, markersize=9 if "★" in name else 7,
                    capsize=4, linewidth=1.3, zorder=4,
                    label=name if "★" in name else None)
        # Annotations
        offset_x = 0.3
        offset_y = 0.03 if i % 2 == 0 else -0.06
        ax.annotate(name, xy=(md, me), xytext=(md + offset_x, me + offset_y),
                    fontsize=7.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
                    zorder=5)

    # Pareto frontier
    valid = [(d, e, n) for d, e, n in zip(means_d, means_e, methods)
             if not (np.isnan(d) or np.isnan(e))]
    valid_sorted = sorted(valid, key=lambda t: t[0])
    pareto = []
    min_e = np.inf
    for d, e, n in valid_sorted:
        if e < min_e:
            pareto.append((d, e))
            min_e = e
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="#6b7280", linewidth=1.2,
                alpha=0.7, label="Pareto Frontier", zorder=2)

    ax.set_xlabel("Total MEC Task Delay (s)")
    ax.set_ylabel("Total Fleet Energy (kJ)")
    ax.set_title("Fig 9 · Pareto Trade-off: Task Latency vs Energy Consumption")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")
    save(fig, os.path.join(fig_dir, "fig09_pareto_front.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 10 — Aerodynamic Power Curve
# ===========================================================================
def fig_aerodynamic_power(fig_dir: str):
    """Regenerates the UAV aerodynamic power vs speed curve analytically."""
    print("[Fig 10] Aerodynamic Power Curve …")
    try:
        # Try to import the actual model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from env.aerodynamics import AerodynamicsModel
        aero = AerodynamicsModel()
        speeds = np.linspace(0, 25, 500)
        powers = np.array([aero.power(v) for v in speeds])
        source = "model"
    except Exception:
        # Fallback: analytical DJI-style blade element model
        # P_hover + P_induced + P_profile + P_parasite
        m, g = 2.5, 9.81
        rho = 1.225
        A = 0.1963  # rotor disk area
        Cd0, sigma = 0.012, 0.05
        f = 0.015
        kappa = 1.15
        P_hover = kappa * (m * g) ** 1.5 / np.sqrt(2 * rho * A)
        speeds = np.linspace(0, 25, 500)
        v_i = m * g / (2 * rho * A * np.maximum(speeds, 0.1))
        P_induced = kappa * m * g * v_i
        P_profile = rho * sigma * Cd0 * A * (225.0) ** 3 / 8
        P_parasite = 0.5 * rho * f * speeds ** 3
        powers = P_induced + P_profile + P_parasite
        source = "analytical"

    # Find minimum power (most efficient speed)
    idx_min = np.argmin(powers)
    v_opt = speeds[idx_min]
    p_min = powers[idx_min]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, powers / 1000.0, color=PALETTE["mappo_pure"], linewidth=2.5,
            label="Total Propulsion Power")
    ax.axvline(v_opt, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"Min-Power Speed ({v_opt:.1f} m/s)")
    ax.axhline(p_min / 1000.0, color="#ef4444", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.scatter([v_opt], [p_min / 1000.0], color="#ef4444", s=80, zorder=5)
    ax.annotate(f"  {p_min/1000:.2f} kW @ {v_opt:.1f} m/s",
                xy=(v_opt, p_min / 1000.0), xytext=(v_opt + 1.0, p_min / 1000.0 + 0.1),
                fontsize=9, color="#b91c1c")
    ax.set_xlabel("Airspeed (m/s)")
    ax.set_ylabel("Propulsion Power (kW)")
    ax.set_title("Fig 10 · Aerodynamic Power Curve (UAV Blade-Element Model)")
    ax.set_xlim(0, 25)
    ax.set_ylim(0)
    ax.grid(True)
    ax.legend()
    if source == "analytical":
        ax.text(0.98, 0.95, "(Analytical Fallback)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#888888", style="italic")
    save(fig, os.path.join(fig_dir, "fig10_aerodynamic_power.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 11 — Held-Out Generalisation Scatter Plot
# ===========================================================================
def fig_heldout_generalisation(res_dir: str, fig_dir: str):
    print("[Fig 11] Held-Out Generalisation Scatter …")
    csv_path = os.path.join(res_dir, "navigation_benchmark.csv")
    # Also try heldout eval
    heldout = os.path.join(res_dir, "heldout_eval_3seeds_summary.csv")
    nav_bench = os.path.join(res_dir, "navigation_benchmark.csv")

    if os.path.exists(heldout):
        df = pd.read_csv(heldout)
        x_col = next((c for c in ["total_energy_kJ", "Nav Energy (kJ)", "total_energy"] if c in df.columns), None)
        y_col = next((c for c in ["return", "val_return", "task_completion"] if c in df.columns), None)
        c_col = next((c for c in ["model_type", "train_seed", "model_name"] if c in df.columns), None)
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            if c_col:
                seeds = df[c_col].unique()
                cmap = matplotlib.colormaps.get_cmap("tab10")
                for i, s in enumerate(seeds):
                    sub = df[df[c_col] == s]
                    ax.scatter(sub[x_col], sub[y_col], alpha=0.6,
                               color=cmap(i), label=str(s), s=30)
                ax.legend(title=c_col, fontsize=8)
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=30, color=PALETTE["mappo_pure"])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Fig 11 · Held-Out Generalisation Scatter")
            ax.grid(True)
            save(fig, os.path.join(fig_dir, "fig11_heldout_scatter.png"))
            plt.close(fig)
            return

    # Fallback: use navigation_benchmark table
    if os.path.exists(nav_bench):
        df = pd.read_csv(nav_bench)
        if "Nav Energy (kJ)" in df.columns and "Mission Success (%)" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = df["Nav Energy (kJ)"].values.astype(float)
            y = df["Mission Success (%)"].values.astype(float)
            names = df["Algorithm"].apply(lambda s: s[:25]).values
            colors = [PALETTE.get(
                "mappo_pure" if "MAPPO" in str(a) else
                "ippo" if "IPPO" in str(a) else
                "cppo" if "CPPO" in str(a) else
                "apf_only" if "APF" in str(a) else
                "astar", "#6b7280") for a in df["Algorithm"]]
            ax.scatter(x, y, c=colors, s=90, zorder=4, edgecolors="white", linewidth=0.7)
            for xi, yi, name in zip(x, y, names):
                ax.annotate(name, (xi, yi), textcoords="offset points",
                            xytext=(4, 3), fontsize=7.5)
            ax.set_xlabel("Navigation Energy (kJ)")
            ax.set_ylabel("Mission Success Rate (%)")
            ax.set_title("Fig 11 · Navigation Energy vs Mission Success")
            ax.set_ylim(-10, 115)
            ax.grid(True)
            save(fig, os.path.join(fig_dir, "fig11_nav_energy_vs_success.png"))
            plt.close(fig)
            return
    print("  [skip] No suitable data for fig 11.")


# ===========================================================================
# FIGURE 12 — Per-Seed Variability (boxplot per algorithm)
# ===========================================================================
def fig_per_seed_variability(res_dir: str, fig_dir: str):
    print("[Fig 12] Per-Seed Variability Boxplot …")
    csv_path = os.path.join(res_dir, "benchmark_per_seed.csv")
    if not os.path.exists(csv_path):
        print("  [skip] benchmark_per_seed.csv not found.")
        return

    df = pd.read_csv(csv_path)
    metric = next((c for c in ["uav_success_rate", "completion_rate", "total_energy"] if c in df.columns), None)
    if not metric or "method" not in df.columns:
        print("  [skip] required columns missing.")
        return

    methods = df["method"].unique()[:10]   # cap at 10 for readability
    data = [df[df["method"] == m][metric].dropna().values for m in methods]
    short_names = [str(m)[:30] for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.4), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="#1e3a5f", linewidth=2.0))
    colors_box = [PALETTE.get(
        "mappo_pure" if "MAPPO" in str(m) and "Residual" not in str(m) else
        "mappo_residual" if "Residual" in str(m) else
        "ippo" if "IPPO" in str(m) else
        "maddpg" if "MADDPG" in str(m) else
        "cppo" if "CPPO" in str(m) else
        "apf_only" if "APF" in str(m) else "astar", "#6b7280") for m in methods]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Fig 12 · Per-Seed Variability: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.5)
    save(fig, os.path.join(fig_dir, "fig12_per_seed_variability.png"))
    plt.close(fig)


# ===========================================================================
# FIGURE 13 — Multi-Seed Return CI Bands (all training seeds if available)
# ===========================================================================
def fig_multiseed_ci_bands(res_dir: str, fig_dir: str):
    print("[Fig 13] Multi-seed Confidence Interval Bands …")
    seed_files = {
        42:  os.path.join(res_dir, "training_logs_mappo_residual_seed42.csv"),
        101: os.path.join(res_dir, "training_logs_mappo_residual_seed101.csv"),
        202: os.path.join(res_dir, "training_logs_mappo_residual_seed202.csv"),
    }
    available = {s: pd.read_csv(p) for s, p in seed_files.items() if os.path.exists(p)}
    if not available:
        print("  [skip] No multi-seed training files found.")
        return

    # Align on episode index
    min_ep = min(len(df) for df in available.values())
    returns = np.stack([df["return"].values[:min_ep] for df in available.values()])
    eps = list(available.values())[0]["episode"].values[:min_ep]

    mean = smooth(returns.mean(axis=0), window=20)
    stdev = returns.std(axis=0, ddof=1)
    upper = smooth(mean + stdev, window=20)
    lower = smooth(mean - stdev, window=20)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(eps, lower, upper, color=PALETTE["mappo_residual"], alpha=0.2,
                    label="Mean ± 1 Std Dev")
    ax.plot(eps, mean, color=PALETTE["mappo_residual"], linewidth=2.3,
            label=f"Residual MAPPO (n={len(available)} seeds)")

    # individual seeds lightly
    for s, df in available.items():
        raw = df["return"].values[:min_ep]
        ax.plot(eps, raw, color=PALETTE["mappo_residual"], linewidth=0.5, alpha=0.3)

    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title(f"Fig 13 · Multi-Seed Training Stability (Residual MAPPO, {len(available)} Seeds)")
    ax.legend()
    ax.grid(True)
    save(fig, os.path.join(fig_dir, "fig13_multiseed_ci.png"))
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate all publication figures.")
    parser.add_argument("--res_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                        help="Results directory (default: ./results)")
    parser.add_argument("--out_dir", "--fig_dir", type=str, default=None, dest="out_dir",
                        help="Output figures directory (default: results/publication_figures_v3)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    parser.add_argument("--per_episode", action="store_true", default=False,
                        help="Plot every single episode without rolling-average smoothing")
    parser.add_argument("--window", type=int, default=None,
                        help="Rolling window size for curve smoothing (default: 30, use 1 for raw per-episode)")
    args = parser.parse_args()

    res_dir = args.res_dir
    fig_dir = args.out_dir or os.path.join(res_dir, "publication_figures_v3")
    os.makedirs(fig_dir, exist_ok=True)

    plot_window = 1 if args.per_episode else (args.window if args.window is not None else 30)

    # Apply publication style
    set_pub_style()
    matplotlib.rcParams["savefig.dpi"] = args.dpi

    print(f"\n{'='*70}")
    print(" GENERATING PUBLISHING-READY FIGURES")
    print(f" Results dir : {res_dir}")
    print(f" Output dir  : {fig_dir}")
    print(f" Window size : {plot_window} {'(Per-Episode raw curves)' if plot_window <= 1 else '(Smoothed)'}")
    print(f"{'='*70}\n")

    # Load training logs
    dfs = load_training_logs(res_dir)

    # Generate figures
    if dfs:
        fig_training_return(dfs, fig_dir, window=plot_window)
        fig_training_rates(dfs, fig_dir, window=plot_window)
        fig_clean_wide_suite(dfs, fig_dir, window=plot_window)
        fig_training_energy_safety(dfs, fig_dir, window=plot_window)
        fig_validation_curves(dfs, fig_dir, window=min(10, plot_window))
        fig_reward_decomposition(dfs, fig_dir, window=plot_window)
    else:
        print("[Warning] No training logs found — skipping training curve figures.")

    fig_benchmark_table(res_dir, fig_dir)
    fig_navigation_benchmark(res_dir, fig_dir)
    fig_offloading_distribution(res_dir, fig_dir)
    fig_pareto_front(res_dir, fig_dir)
    fig_aerodynamic_power(fig_dir)
    fig_heldout_generalisation(res_dir, fig_dir)
    fig_per_seed_variability(res_dir, fig_dir)
    fig_multiseed_ci_bands(res_dir, fig_dir)

    print(f"\n{'='*70}")
    print(f" [DONE] All figures written to: {fig_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
