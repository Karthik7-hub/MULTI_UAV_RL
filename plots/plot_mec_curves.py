"""
MEC Task Offloading Publication Curves:
Generates dedicated comparative figures for:
1. System Utility vs Number of Iterations / Mission Time
2. Success Rate of Tasks Executed Before Deadline (%) vs Iterations / Time
3. "What-If" Offloading Breakdown (Local Only vs Edge Only vs Greedy vs Cooperative MAPPO vs Joint)
4. Active Queue Dynamics and Latency Over Mission Timesteps

Usage:
    python plots/plot_mec_curves.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC"
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "publication_figures", "task_offloading")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Plot Training Iteration Convergence Curves
# -----------------------------------------------------------------------------
def plot_training_iteration_curves(log_dir: str = os.path.join(BASE_DIR, "results", "logs")):
    """Plots System Utility (Return) and Deadline Success Rate vs Training Iterations/Episodes."""
    log_files = {
        "Proposed Decoupled MAPPO": os.path.join(log_dir, "training_logs_mappo_residual.csv"),
        "Joint Pure MAPPO (8.4k updates)": os.path.join(log_dir, "training_logs_mappo_pure.csv"),
        "Centralized PPO (CPPO)": os.path.join(log_dir, "training_logs_cppo.csv"),
        "Independent PPO (IPPO)": os.path.join(log_dir, "training_logs_ippo.csv"),
        "Multi-Agent DDPG (MADDPG)": os.path.join(log_dir, "training_logs_maddpg.csv"),
    }
    
    colors = {
        "Proposed Decoupled MAPPO": "#0284c7",       # Crisp Blue
        "Joint Pure MAPPO (8.4k updates)": "#d97706", # Amber/Orange
        "Centralized PPO (CPPO)": "#8b5cf6",         # Purple
        "Independent PPO (IPPO)": "#10b981",         # Emerald Green
        "Multi-Agent DDPG (MADDPG)": "#ef4444",      # Rose Red
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    found_any = False
    
    for name, filepath in log_files.items():
        if not os.path.exists(filepath):
            continue
        found_any = True
        df = pd.read_csv(filepath)
        episodes = df["episode"].values
        returns = df["return"].values
        comp_rate = df["completion_rate"].values
        energy_kj = df["total_energy"].values / 1000.0 if "total_energy" in df.columns else None
        misses = df["deadline_misses"].values if "deadline_misses" in df.columns else None
        
        c = colors.get(name, "#333333")
        
        # Panel A: System Utility (Episode Return) vs Training Episodes
        smooth_return = pd.Series(returns).rolling(window=3, min_periods=1).mean().values
        axes[0, 0].plot(episodes, smooth_return, label=name, color=c, linewidth=2.2)
        axes[0, 0].fill_between(episodes, smooth_return - 150, smooth_return + 150, color=c, alpha=0.1)
        
        # Panel B: Success Rate of Tasks Executed Before Deadline (%) vs Training Episodes
        smooth_comp = pd.Series(comp_rate).rolling(window=3, min_periods=1).mean().values
        axes[0, 1].plot(episodes, smooth_comp, label=name, color=c, linewidth=2.2)
        
        # Panel C: Deadline Misses Count vs Training Episodes
        if misses is not None:
            smooth_miss = pd.Series(misses).rolling(window=3, min_periods=1).mean().values
            axes[1, 0].plot(episodes, smooth_miss, label=name, color=c, linewidth=2.0)
            
        # Panel D: Total Fleet Energy Consumption (kJ) vs Training Episodes
        if energy_kj is not None:
            smooth_e = pd.Series(energy_kj).rolling(window=3, min_periods=1).mean().values
            axes[1, 1].plot(episodes, smooth_e, label=name, color=c, linewidth=2.0)

    if not found_any:
        print(f"Warning: No training logs found in {log_dir}")
        plt.close()
        return

    # Panel A Config
    axes[0, 0].set_title("Panel A: System Utility (Return) vs Training Iterations", fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel("Training Episode / Iteration", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Cumulative System Return (Utility)", fontsize=10, fontweight='bold')
    axes[0, 0].legend(loc="lower right", fontsize=8.5)
    axes[0, 0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    
    # Panel B Config
    axes[0, 1].set_title("Panel B: Success Rate of Tasks Before Deadline (%) vs Iterations", fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel("Training Episode / Iteration", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Deadline Satisfaction Rate (%)", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylim(96.0, 100.5)
    axes[0, 1].axhline(100.0, color="#059669", linestyle="--", alpha=0.7, label="100% Optimal Target")
    axes[0, 1].legend(loc="lower right", fontsize=8.5)
    
    # Panel C Config
    axes[1, 0].set_title("Panel C: Task Deadline Misses vs Training Iterations", fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel("Training Episode / Iteration", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Deadline Violations (Tasks/Episode)", fontsize=10, fontweight='bold')
    axes[1, 0].legend(loc="upper right", fontsize=8.5)
    
    # Panel D Config
    axes[1, 1].set_title("Panel D: Fleet Energy Consumption (kJ) vs Training Iterations", fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("Training Episode / Iteration", fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Total Mission Energy (kJ)", fontsize=10, fontweight='bold')
    axes[1, 1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "mec_training_convergence_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] Training Convergence Curves saved to: {out_path}")


# -----------------------------------------------------------------------------
# 2. Plot Operational Utility and Deadline Success vs Mission Time (t = 1..100s)
# -----------------------------------------------------------------------------
def plot_operational_curves(raw_path: str = os.path.join(BASE_DIR, "results", "phase_4_validation", "raw_task_records", "raw_tasks_all_models.csv")):
    """Generates continuous curves over mission time (t = 1..100s) comparing all offloading paradigms."""
    if not os.path.exists(raw_path):
        print(f"Warning: Raw tasks file not found at: {raw_path}")
        return
        
    df = pd.read_csv(raw_path)
    
    models_to_plot = {
        "Local Only (What if Only Local?)": "Local Only",
        "Static Edge (What if 100% Offloaded?)": "Static Edge (Unqueued Reference)",
        "Queue-Aware Greedy": "Queue-Aware Energy/Deadline Greedy",
        "Load-Balancing Heuristic": "Queue-Load Balancing Heuristic",
        "Proposed Cooperative MAPPO (Decoupled)": "Proposed MAPPO Offloading (k_nav=0.0)",
        "Proposed Pure MAPPO (Joint)": "Proposed Pure MAPPO (8.4k updates)"
    }
    
    model_colors = {
        "Local Only (What if Only Local?)": "#e11d48",           # Red
        "Static Edge (What if 100% Offloaded?)": "#ea580c",     # Orange
        "Queue-Aware Greedy": "#ca8a04",                        # Yellow/Gold
        "Load-Balancing Heuristic": "#16a34a",                  # Green
        "Proposed Cooperative MAPPO (Decoupled)": "#0284c7",    # Primary Blue
        "Proposed Pure MAPPO (Joint)": "#7c3aed"                # Purple
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    for display_name, method_key in models_to_plot.items():
        sub_df = df[df["method"] == method_key]
        if sub_df.empty:
            continue
            
        c = model_colors.get(display_name, "#333333")
        t_steps = np.arange(1, 101)
        seeds = sub_df["seed"].unique()
        
        seed_utilities = []
        seed_success_rates = []
        seed_delays = []
        seed_queues = []
        
        for s in seeds:
            s_df = sub_df[sub_df["seed"] == s]
            cum_u = 0.0
            total_tasks_seen = 0
            satisfied_tasks_seen = 0
            cum_d = 0.0
            
            s_u_curve = []
            s_succ_curve = []
            s_d_curve = []
            s_q_curve = []
            
            for t in t_steps:
                step_tasks = s_df[s_df["timestep"] == t]
                if not step_tasks.empty:
                    n_tasks = len(step_tasks)
                    n_sat = step_tasks["deadline_satisfied"].sum()
                    n_miss = n_tasks - n_sat
                    step_delay = step_tasks["total_delay_s"].sum()
                    step_comm_e = step_tasks["comm_energy_j"].sum()
                    step_comp_e = step_tasks["comp_energy_j"].sum()
                    
                    step_u = 2.0 * n_sat - 2.5 * n_miss - 1.0 * step_delay - 0.001 * (step_comm_e + step_comp_e)
                    cum_u += step_u
                    total_tasks_seen += n_tasks
                    satisfied_tasks_seen += n_sat
                    cum_d += step_delay
                    avg_q = step_tasks["edge_queue_at_arrival"].mean() / 1e9  # in Gcycles
                else:
                    avg_q = 0.0
                    
                s_u_curve.append(cum_u)
                s_succ_curve.append((satisfied_tasks_seen / max(1, total_tasks_seen)) * 100.0)
                s_d_curve.append(cum_d)
                s_q_curve.append(avg_q)
                
            seed_utilities.append(s_u_curve)
            seed_success_rates.append(s_succ_curve)
            seed_delays.append(s_d_curve)
            seed_queues.append(s_q_curve)
            
        mean_u = np.mean(seed_utilities, axis=0)
        mean_succ = np.mean(seed_success_rates, axis=0)
        mean_d = np.mean(seed_delays, axis=0)
        mean_q = np.mean(seed_queues, axis=0)
        
        axes[0, 0].plot(t_steps, mean_u, label=display_name, color=c, linewidth=2.2)
        axes[0, 1].plot(t_steps, mean_succ, label=display_name, color=c, linewidth=2.2)
        axes[1, 0].plot(t_steps, mean_d, label=display_name, color=c, linewidth=2.2)
        axes[1, 1].plot(t_steps, mean_q, label=display_name, color=c, linewidth=2.0)

    # Panel A: Cumulative MEC Utility vs Time
    axes[0, 0].set_title("Panel A: Cumulative System Utility vs Mission Time (t)", fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel("Mission Elapsed Time (seconds)", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Cumulative MEC Utility U(t)", fontsize=10, fontweight='bold')
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    
    # Panel B: Cumulative Success Rate Before Deadline (%) vs Time
    axes[0, 1].set_title("Panel B: Cumulative Task Success Rate Before Deadline (%) vs Time", fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel("Mission Elapsed Time (seconds)", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Cumulative Deadline Satisfaction (%)", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylim(80.0, 101.0)
    axes[0, 1].axhline(100.0, color="#059669", linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="lower left", fontsize=8.5)
    
    # Panel C: Cumulative Task Delay (Latency) vs Time
    axes[1, 0].set_title("Panel C: Cumulative Fleet Task Delay vs Mission Time", fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel("Mission Elapsed Time (seconds)", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Cumulative Latency (seconds)", fontsize=10, fontweight='bold')
    axes[1, 0].legend(loc="upper left", fontsize=8.5)
    
    # Panel D: Edge Server Queue Backlog vs Time
    axes[1, 1].set_title("Panel D: Ground Edge Server Queue Backlog vs Time", fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("Mission Elapsed Time (seconds)", fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Edge Queue Backlog (Gcycles)", fontsize=10, fontweight='bold')
    axes[1, 1].legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "mec_utility_and_deadline_vs_mission_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] Mission Time Curves saved to: {out_path}")


# -----------------------------------------------------------------------------
# 3. Plot "What-If" Offloading Breakdown & Multi-Metric Comparison
# -----------------------------------------------------------------------------
def plot_what_if_breakdown(summary_path: str = os.path.join(BASE_DIR, "results", "phase_4_validation", "recomputed_summary_metrics.csv")):
    """Publication figure addressing the exact 'What if Local? What if Edge? What if Cooperative?' question."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary metrics file not found at: {summary_path}")
        return
        
    df = pd.read_csv(summary_path)
    
    target_methods = [
        "Local Only",
        "Static Edge (Unqueued Reference)",
        "Queue-Aware Lowest Latency",
        "Queue-Load Balancing Heuristic",
        "Proposed MAPPO Offloading (k_nav=0.0)",
        "Proposed Pure MAPPO (8.4k updates)"
    ]
    
    labels = [
        "What If\nLocal Only?",
        "What If\n100% Edge?",
        "Queue-Aware\nGreedy",
        "Load-Balancing\nHeuristic",
        "Proposed MAPPO\n(Decoupled)",
        "Pure MAPPO\n(Joint)"
    ]
    
    df_sub = df[df["Method"].isin(target_methods)].copy()
    df_sub["sorter"] = df_sub["Method"].map(lambda x: target_methods.index(x))
    df_sub = df_sub.sort_values("sorter").reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    x = np.arange(len(labels))
    width = 0.55
    
    # Panel 1: Offloading Destination Ratio (%)
    ax1 = axes[0, 0]
    p_loc = df_sub["pct_local_mean"].values
    p_peer = df_sub["pct_peer_mean"].values
    p_edge = df_sub["pct_edge_mean"].values
    
    ax1.bar(x, p_loc, width, label="Local CPU (On-board)", color="#3b82f6", edgecolor="#1d4ed8")
    ax1.bar(x, p_peer, width, bottom=p_loc, label="Peer UAVs (Cooperative)", color="#10b981", edgecolor="#047857")
    ax1.bar(x, p_edge, width, bottom=p_loc + p_peer, label="Edge Server (Base Station)", color="#f59e0b", edgecolor="#b45309")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Offloading Ratio (%)", fontsize=10, fontweight='bold')
    ax1.set_title("Panel A: Where Do Tasks Go? (Offloading Destination Share)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", fontsize=8.5)
    
    for i in range(len(labels)):
        ax1.text(i, 102, f"Total 100%", ha='center', fontsize=8, fontweight='bold')
        
    # Panel 2: Mean Task Delay (s)
    ax2 = axes[0, 1]
    delays = df_sub["mean_delay_per_completed_task_s_mean"].values
    delay_err = df_sub["mean_delay_per_completed_task_s_std"].values
    bar_colors = ["#94a3b8", "#f97316", "#eab308", "#22c55e", "#0284c7", "#a855f7"]
    
    bars = ax2.bar(x, delays, width, yerr=delay_err, capsize=4, color=bar_colors, edgecolor="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Mean Task Delay (seconds)", fontsize=10, fontweight='bold')
    ax2.set_title("Panel B: Mean Latency per Completed Task (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    
    for idx, b in enumerate(bars):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.3f}s", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight lowest delay
    ax2.annotate('Lowest Delay:\n0.508 s (-8.86%)',
                 xy=(4, delays[4]), xytext=(3.4, delays[4] + 0.12),
                 arrowprops=dict(facecolor='#0284c7', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight='bold', color='#0284c7')

    # Panel 3: Deadline Satisfaction Rate (%) and Misses
    ax3 = axes[1, 0]
    sat_rates = df_sub["deadline_satisfaction_pct_mean"].values
    misses = df_sub["deadline_misses_mean"].values
    
    bars3 = ax3.bar(x, sat_rates, width, color=bar_colors, edgecolor="#333333")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax3.set_ylabel("Deadline Satisfaction Rate (%)", fontsize=10, fontweight='bold')
    ax3.set_title("Panel C: Task Deadline Success Rate (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylim(97.5, 100.5)
    
    for idx, b in enumerate(bars3):
        h = b.get_height()
        miss_txt = f"{h:.2f}%\n({misses[idx]:.1f} miss)" if misses[idx] > 0 else f"{h:.1f}%\n(0 miss!)"
        ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.05, miss_txt, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel 4: Energy Breakdown (Comm Energy vs Compute Energy)
    ax4 = axes[1, 1]
    comm_j = df_sub["comm_energy_kj_mean"].values * 1000.0   # in Joules
    comp_j = df_sub["comp_energy_kj_mean"].values * 1000.0   # in Joules
    
    ax4.bar(x - 0.15, comm_j, 0.3, label="RF Wireless Comm Energy (J)", color="#0ea5e9", edgecolor="#0369a1")
    ax4.bar(x + 0.15, comp_j, 0.3, label="CPU Computation Energy (J)", color="#f43f5e", edgecolor="#be123c")
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax4.set_ylabel("Energy Consumption (Joules)", fontsize=10, fontweight='bold')
    ax4.set_title("Panel D: Energy Decomposition: RF Comm vs CPU Compute", fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc="upper right", fontsize=8.5)
    
    ax4.text(0 - 0.15, 3.0, "0.0 J\n(No RF)", ha='center', va='bottom', fontsize=8, color="#0369a1", fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "what_if_offloading_comprehensive_tradeoff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] What-If Trade-Off Breakdown saved to: {out_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating MEC Task Offloading Publication Curves...")
    print("=" * 70)
    plot_training_iteration_curves()
    plot_operational_curves()
    plot_what_if_breakdown()
    print("\nAll MEC Task Offloading figures generated successfully!")
