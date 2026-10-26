"""
Publication-Grade Visualizer for Multi-UAV Edge Computing.
Strictly generates high-resolution continuous line curves, confidence bands, and 2D/3D trajectory plots.
Zero bar graphs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.0,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

def plot_trajectory_diagnostic(astar_data: dict, mappo_data: dict, obstacles: list, stations: dict, save_path: str):
    """
    Generates the 4-panel Trajectory Diagnostic Comparison figure matching publication standards:
    - Panel A: 2D Flight Trajectories in Airspace (Symmetric Mission)
    - Panel B: Distance to Designated Station vs Time
    - Panel C: Flight Velocity Profile Over Time
    - Panel D: Fleet Movement Energy Evolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    colors = ['#0284c7', '#8b5cf6', '#d97706', '#059669']  # Blue, Purple, Orange, Green
    reach_tol = 5.0
    num_uavs = 4
    time_steps = np.arange(100)

    # -------------------------------------------------------------------------
    # Panel A: 2D Flight Trajectories
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    # Draw cylindrical obstacles situated in-between start and goal
    for idx, (cx, cy, r, h) in enumerate(obstacles):
        circ = plt.Circle((cx, cy), r, color='#f43f5e', alpha=0.3, label='Obstacle Corridor' if idx == 0 else None)
        ax1.add_patch(circ)
        ax1.text(cx, cy, f"Obs #{idx}\n(r={r:.0f}m)", color='#e11d48', ha='center', va='center', fontsize=8, fontweight='bold')

    # Mission center
    ax1.plot(100.0, 100.0, marker='*', color='#059669', markersize=14, label='Mission Center (100, 100)')

    # Stations with reach tolerance circles
    for j in range(num_uavs):
        g = stations[j]
        st_circ = plt.Circle((g[0], g[1]), reach_tol, color=colors[j], alpha=0.15, linestyle=':', fill=True)
        ax1.add_patch(st_circ)
        ax1.plot(g[0], g[1], marker='^', color=colors[j], markersize=10, label=f"Station #{j}" if j == 0 else None)
        ax1.text(g[0], g[1] + 4.5, f"Station #{j}", color=colors[j], ha='center', fontsize=8, fontweight='bold')

    # 3D A* Paths (dashed)
    for j in range(num_uavs):
        p_a = np.array(astar_data[j]["pos"])
        ax1.plot(p_a[:, 0], p_a[:, 1], color=colors[j], linestyle='--', linewidth=2.0, label=f"A* UAV #{j}" if j == 0 else None)
        ax1.plot(p_a[0, 0], p_a[0, 1], marker='o', color=colors[j], markersize=6)

    # MAPPO Paths (solid)
    for j in range(num_uavs):
        p_m = np.array(mappo_data[j]["pos"])
        ax1.plot(p_m[:, 0], p_m[:, 1], color=colors[j], linestyle='-', linewidth=2.5, label=f"MAPPO UAV #{j}" if j == 0 else None)
        ax1.plot(p_m[-1, 0], p_m[-1, 1], marker='x', color=colors[j], markersize=8, mew=2)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax1.set_title("Panel A: 2D Flight Trajectories in Airspace (Symmetric Mission)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel B: Distance to Designated Station vs Time
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    for j in range(num_uavs):
        d_a = np.array(astar_data[j]["dist_goal"][:100])
        d_m = np.array(mappo_data[j]["dist_goal"][:100])
        ax2.plot(time_steps, d_a, color=colors[j], linestyle='--', alpha=0.6, label=f"A* UAV #{j}" if j == 0 else None)
        ax2.plot(time_steps, d_m, color=colors[j], linestyle='-', linewidth=2.0, label=f"MAPPO UAV #{j}" if j == 0 else None)

    ax2.axhline(reach_tol, color='#10b981', linestyle=':', linewidth=2, label=f'Reach Tolerance ({reach_tol:.0f}m)')
    ax2.set_xlabel("Episode Timestep (t, seconds)", fontsize=11)
    ax2.set_ylabel("Distance to Station (meters)", fontsize=11)
    ax2.set_title("Panel B: Distance to Designated Station vs Time", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel C: Flight Velocity Profile Over Time
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    for j in range(num_uavs):
        v_a = np.array(astar_data[j]["speed"][:100])
        v_m = np.array(mappo_data[j]["speed"][:100])
        ax3.plot(time_steps, v_a, color=colors[j], linestyle='--', alpha=0.5, label=f"A* UAV #{j}" if j == 0 else None)
        ax3.plot(time_steps, v_m, color=colors[j], linestyle='-', linewidth=1.8, label=f"MAPPO UAV #{j}" if j == 0 else None)

    ax3.set_xlabel("Episode Timestep (t, seconds)", fontsize=11)
    ax3.set_ylabel("Flight Speed (m/s)", fontsize=11)
    ax3.set_title("Panel C: Flight Velocity Profile Over Time", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel D: Cumulative Movement Energy Evolution
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    cum_e_a = np.zeros(100)
    cum_e_m = np.zeros(100)
    for j in range(num_uavs):
        cum_e_a += np.cumsum(astar_data[j]["power"][:100])
        cum_e_m += np.cumsum(mappo_data[j]["power"][:100])

    ax4.plot(time_steps, cum_e_a, color='#dc2626', linestyle='--', linewidth=2.5, label=f"3D A* Fleet (Final: {cum_e_a[-1]:.0f} J)")
    ax4.plot(time_steps, cum_e_m, color='#0284c7', linestyle='-', linewidth=2.5, label=f"MAPPO Fleet (Final: {cum_e_m[-1]:.0f} J)")
    ax4.set_xlabel("Episode Timestep (t, seconds)", fontsize=11)
    ax4.set_ylabel("Cumulative Propulsion Energy (Joules)", fontsize=11)
    ax4.set_title("Panel D: Fleet Movement Energy Evolution", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved publication trajectory diagnostic figure: {save_path}")

def plot_training_convergence(csv_path: str, save_path: str):
    """
    Plots multi-line training convergence curves with moving averages and confidence bands.
    """
    if not os.path.exists(csv_path):
        print(f"[Plotter] Warning: {csv_path} not found. Skipping training convergence plot.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    window = max(3, len(df) // 10)

    # (a) Episode Return
    ax = axes[0, 0]
    raw = df["return"]
    smoothed = raw.rolling(window, min_periods=1).mean()
    std = raw.rolling(window, min_periods=1).std().fillna(0)
    ax.plot(df["episode"], raw, color='#93c5fd', alpha=0.5, label='Raw Return')
    ax.plot(df["episode"], smoothed, color='#1d4ed8', linewidth=2.5, label=f'Moving Avg (w={window})')
    ax.fill_between(df["episode"], smoothed - std, smoothed + std, color='#bfdbfe', alpha=0.4)
    ax.set_title("(a) Cumulative Episode Return", fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)

    # (b) Task Completion vs Navigation Arrival Rate (%)
    ax = axes[0, 1]
    comp_rate = (df["completed_tasks"] / np.maximum(1, df["completed_tasks"] + df["deadline_misses"])) * 100.0
    comp_smooth = comp_rate.rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], comp_rate, color='#86efac', alpha=0.3, label='Raw Task Comp %')
    ax.plot(df["episode"], comp_smooth, color='#15803d', linewidth=2.2, label='MEC Task Comp %')
    if "arrival_rate" in df.columns:
        arr_rate = df["arrival_rate"]
        arr_smooth = arr_rate.rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], arr_rate, color='#93c5fd', alpha=0.4, label='Raw Arrival %')
        ax.plot(df["episode"], arr_smooth, color='#0284c7', linewidth=2.5, linestyle='--', label='Nav Arrival %')
    ax.axhline(95.0, color='#e11d48', linestyle=':', label='Target Threshold (95%)')
    ax.set_title("(b) Task Completion vs Nav Arrival (%)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc='center right', fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (c) Fleet Total Energy (kJ)
    ax = axes[1, 0]
    energy_kj = df["total_energy"] / 1000.0
    energy_smooth = energy_kj.rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], energy_kj, color='#fca5a5', alpha=0.5, label='Raw Energy (kJ)')
    ax.plot(df["episode"], energy_smooth, color='#b91c1c', linewidth=2.5, label='Smoothed Energy (kJ)')
    ax.set_title("(c) Fleet Energy Consumption (kJ)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Energy (kJ)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    # (d) Safety Interventions
    ax = axes[1, 1]
    interv = df["safety_interventions"].rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], interv, color='#7c3aed', linewidth=2.5, label='Safety Interventions')
    ax.set_title("(d) Analytical Safety Interventions (Collision Avoidance)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Interventions per Episode")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Figure 1: Joint MAPPO Training Convergence & Multi-Objective Performance", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved training convergence curves figure: {save_path}")

def plot_aerodynamic_power_curve(aero_model, save_path: str):
    """
    Plots the rotary-wing aerodynamic power curve P(v) vs speed.
    """
    speeds, p_tot, p_ind, p_prof, p_para = aero_model.get_power_curve(200)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    ax.plot(speeds, p_tot, color='#1d4ed8', linewidth=2.5, label=r'Total Mechanical Power $P(v)$')
    ax.plot(speeds, p_ind, color='#15803d', linestyle='--', linewidth=1.8, label=r'Induced Power $P_i(v)$ (Translational Lift)')
    ax.plot(speeds, p_prof, color='#d97706', linestyle='-.', linewidth=1.8, label=r'Blade Profile Power $P_0(v)$')
    ax.plot(speeds, p_para, color='#dc2626', linestyle=':', linewidth=1.8, label=r'Parasite Drag Power $P_p(v)$')

    # Markers for hover and cruising minimum
    min_idx = np.argmin(p_tot)
    v_opt = speeds[min_idx]
    p_opt = p_tot[min_idx]

    ax.scatter([0.0], [p_tot[0]], color='#e11d48', s=100, zorder=5, label=f'Hovering: v=0 m/s ({p_tot[0]:.1f} W)')
    ax.scatter([v_opt], [p_opt], color='#059669', s=100, marker='^', zorder=5, label=f'Optimal Cruising: v={v_opt:.1f} m/s ({p_opt:.1f} W)')

    ax.set_title("Figure 3: Continuous Rotary-Wing Aerodynamic Power vs Flight Velocity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Flight Speed v (m/s)", fontsize=11)
    ax.set_ylabel("Mechanical Power (Watts)", fontsize=11)
    ax.set_ylim(0, 220)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved aerodynamic power curve figure: {save_path}")

def plot_multimodel_curves(results: dict, obstacles: list, stations: dict, save_path: str):
    """
    Generates a 4-panel multi-model comparative performance figure comparing:
    - Proposed Joint MAPPO (CTDE)
    - Independent PPO (IPPO)
    - Multi-Agent DDPG (MADDPG)
    - 3D A* Baseline + Greedy MEC
    All continuous curves, trajectories, and Pareto frontiers. ZERO bar graphs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    time_steps = np.arange(100)
    reach_tol = 5.0

    style_cfg = {
        "mappo": {"name": "Proposed Joint MAPPO", "color": "#0284c7", "ls": "-", "lw": 2.5, "marker": "o"},
        "ippo": {"name": "Independent PPO (IPPO)", "color": "#059669", "ls": "-.", "lw": 2.0, "marker": "s"},
        "maddpg": {"name": "Multi-Agent DDPG (MADDPG)", "color": "#8b5cf6", "ls": ":", "lw": 2.2, "marker": "^"},
        "astar": {"name": "3D A* + Greedy MEC", "color": "#dc2626", "ls": "--", "lw": 2.0, "marker": "D"}
    }

    # -------------------------------------------------------------------------
    # Panel A: 2D Multi-Model Trajectory Comparison (UAV 0 & UAV 2 corridors)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    for idx, (cx, cy, r, h) in enumerate(obstacles):
        circ = plt.Circle((cx, cy), r, color='#f43f5e', alpha=0.3, label='Obstacle Corridor' if idx == 0 else None)
        ax1.add_patch(circ)
        ax1.text(cx, cy, f"Obs #{idx}\n(r={r:.0f}m)", color='#e11d48', ha='center', va='center', fontsize=8, fontweight='bold')

    ax1.plot(100.0, 100.0, marker='*', color='#059669', markersize=14, label='Mission Center (100, 100)')

    for j in range(len(stations)):
        g = stations[j]
        st_circ = plt.Circle((g[0], g[1]), reach_tol, color='#64748b', alpha=0.15, linestyle=':', fill=True)
        ax1.add_patch(st_circ)
        ax1.plot(g[0], g[1], marker='^', color='#334155', markersize=8)
        ax1.text(g[0], g[1] + 4.5, f"Station #{j}", color='#334155', ha='center', fontsize=7, fontweight='bold')

    # Plot paths for representative UAV #0 and UAV #2 (facing horizontal obstacles)
    for model_key in ["astar", "maddpg", "ippo", "mappo"]:
        if model_key in results:
            cfg_m = style_cfg[model_key]
            p0 = np.array(results[model_key][0]["pos"])
            p2 = np.array(results[model_key][2]["pos"])
            ax1.plot(p0[:, 0], p0[:, 1], color=cfg_m["color"], linestyle=cfg_m["ls"], linewidth=cfg_m["lw"], label=f"{cfg_m['name']} (UAV #0)")
            ax1.plot(p2[:, 0], p2[:, 1], color=cfg_m["color"], linestyle=cfg_m["ls"], linewidth=cfg_m["lw"] * 0.7, alpha=0.7)
            ax1.plot(p0[0, 0], p0[0, 1], marker='o', color=cfg_m["color"], markersize=5)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("Airspace X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Airspace Y Coordinate (meters)", fontsize=11)
    ax1.set_title("Panel A: 2D Spatial Corridors & Obstacle Avoidance Trajectories", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel B: Cumulative Fleet Propulsion Energy Evolution
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    for model_key in ["astar", "maddpg", "ippo", "mappo"]:
        if model_key in results:
            cfg_m = style_cfg[model_key]
            cum_e = np.zeros(100)
            for j in range(4):
                cum_e += np.cumsum(results[model_key][j]["power"][:100])
            ax2.plot(time_steps, cum_e, color=cfg_m["color"], linestyle=cfg_m["ls"], linewidth=cfg_m["lw"],
                     label=f"{cfg_m['name']} ({cum_e[-1]:.0f} J)")

    ax2.set_xlabel("Mission Timestep (seconds)", fontsize=11)
    ax2.set_ylabel("Cumulative Propulsion Energy (Joules)", fontsize=11)
    ax2.set_title("Panel B: Cumulative Fleet Propulsion Energy vs Horizon", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel C: Mean Fleet Distance to Designated Station vs Time
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    for model_key in ["astar", "maddpg", "ippo", "mappo"]:
        if model_key in results:
            cfg_m = style_cfg[model_key]
            mean_dist = np.mean([np.array(results[model_key][j]["dist_goal"][:100]) for j in range(4)], axis=0)
            ax3.plot(time_steps, mean_dist, color=cfg_m["color"], linestyle=cfg_m["ls"], linewidth=cfg_m["lw"],
                     label=f"{cfg_m['name']}")

    ax3.axhline(reach_tol, color='#10b981', linestyle=':', linewidth=2, label=f'Reach Tolerance ({reach_tol:.0f}m)')
    ax3.set_xlabel("Mission Timestep (seconds)", fontsize=11)
    ax3.set_ylabel("Mean Fleet Distance to Station (meters)", fontsize=11)
    ax3.set_title("Panel C: Station Transit & Convergence Curves", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel D: Multi-Objective Energy vs Latency Trade-Off Frontier
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    energies = []
    latencies = []
    labels = []
    m_colors = []

    for model_key in ["astar", "maddpg", "ippo", "mappo"]:
        if model_key in results and "summary" in results[model_key]:
            s = results[model_key]["summary"]
            e = s["total_energy"]
            lat = s["total_delay"]
            energies.append(e)
            latencies.append(lat)
            labels.append(style_cfg[model_key]["name"])
            m_colors.append(style_cfg[model_key]["color"])

    if len(energies) > 1:
        # Sort by energy for Pareto curve
        order = np.argsort(energies)
        s_e = np.array(energies)[order]
        s_lat = np.array(latencies)[order]

        ax4.plot(s_lat, s_e, color='#475569', linestyle='--', linewidth=1.5, alpha=0.6, label='Efficiency Trade-Off Curve')
        for i, model_key in enumerate(["astar", "maddpg", "ippo", "mappo"]):
            if model_key in results and "summary" in results[model_key]:
                cfg_m = style_cfg[model_key]
                s = results[model_key]["summary"]
                ax4.scatter(s["total_delay"], s["total_energy"], color=cfg_m["color"], s=130, marker=cfg_m["marker"],
                            zorder=6, label=f"{cfg_m['name']} (E={s['total_energy']:.0f}J, D={s['total_delay']:.1f}s)")

    ax4.set_xlabel("Total Offloading Latency / Delay (seconds)", fontsize=11)
    ax4.set_ylabel("Total Fleet Energy Consumption (Joules)", fontsize=11)
    ax4.set_title("Panel D: Energy-Latency Multi-Objective Trade-Off Frontier", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.suptitle("Figure 2: Multi-Model Benchmark Comparison (MAPPO vs IPPO vs MADDPG vs 3D A*)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved publication multi-model comparison curves figure: {save_path}")

def plot_mappo_vs_astar_deepdive(mappo_data: dict, astar_data: dict, obstacles: list, stations: dict, save_path: str):
    """
    Dedicated Pairwise Comparative Deep-Dive: Proposed Joint MAPPO vs 3D A* Baseline.
    Strictly continuous curves, confidence bands, and trajectories. Zero bar graphs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    time_steps = np.arange(100)
    reach_tol = 5.0
    uav_colors = ['#0284c7', '#8b5cf6', '#d97706', '#059669']

    # Panel A: 2D Trajectory Comparison
    ax1 = axes[0, 0]
    for idx, (cx, cy, r, h) in enumerate(obstacles):
        circ = plt.Circle((cx, cy), r, color='#f43f5e', alpha=0.3, label='Obstacle Corridor' if idx == 0 else None)
        ax1.add_patch(circ)
        ax1.text(cx, cy, f"Obs #{idx}\n(r={r:.0f}m)", color='#e11d48', ha='center', va='center', fontsize=8, fontweight='bold')

    ax1.plot(100.0, 100.0, marker='*', color='#059669', markersize=14, label='Mission Center')
    for j in range(4):
        g = stations[j]
        st_circ = plt.Circle((g[0], g[1]), reach_tol, color=uav_colors[j], alpha=0.15, linestyle=':', fill=True)
        ax1.add_patch(st_circ)
        ax1.plot(g[0], g[1], marker='^', color=uav_colors[j], markersize=9)
        # Trajectories
        p_a = np.array(astar_data[j]["pos"])
        p_m = np.array(mappo_data[j]["pos"])
        ax1.plot(p_a[:, 0], p_a[:, 1], color=uav_colors[j], linestyle='--', linewidth=1.8, label=f"3D A* UAV #{j}" if j < 2 else None)
        ax1.plot(p_m[:, 0], p_m[:, 1], color=uav_colors[j], linestyle='-', linewidth=2.4, label=f"MAPPO UAV #{j}" if j < 2 else None)
        ax1.plot(p_m[0, 0], p_m[0, 1], marker='o', color=uav_colors[j], markersize=5)
        ax1.plot(p_m[-1, 0], p_m[-1, 1], marker='x', color=uav_colors[j], markersize=7, mew=2)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("Airspace X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Airspace Y Coordinate (meters)", fontsize=11)
    ax1.set_title("Panel A: 2D Flight Corridors & Obstacle Bypass Trajectories", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Panel B: Distance to Station vs Time
    ax2 = axes[0, 1]
    for j in range(4):
        d_a = np.array(astar_data[j]["dist_goal"][:100])
        d_m = np.array(mappo_data[j]["dist_goal"][:100])
        ax2.plot(time_steps, d_a, color=uav_colors[j], linestyle='--', alpha=0.6, label=f"3D A* UAV #{j}" if j == 0 else None)
        ax2.plot(time_steps, d_m, color=uav_colors[j], linestyle='-', linewidth=2.0, label=f"MAPPO UAV #{j}" if j == 0 else None)

    ax2.axhline(reach_tol, color='#10b981', linestyle=':', linewidth=2, label=f'Reach Tolerance ({reach_tol:.0f}m)')
    ax2.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax2.set_ylabel("Distance to Station (meters)", fontsize=11)
    ax2.set_title("Panel B: Station Transit & Arrival Convergence", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Panel C: Instantaneous Velocity Profiles
    ax3 = axes[1, 0]
    for j in range(4):
        s_a = np.array(astar_data[j]["speed"][:100])
        s_m = np.array(mappo_data[j]["speed"][:100])
        ax3.plot(time_steps, s_a, color=uav_colors[j], linestyle='--', alpha=0.5, label=f"3D A* UAV #{j}" if j == 0 else None)
        ax3.plot(time_steps, s_m, color=uav_colors[j], linestyle='-', linewidth=2.0, label=f"MAPPO UAV #{j}" if j == 0 else None)

    ax3.axhline(10.3, color='#059669', linestyle='-.', linewidth=1.8, label='Optimal Aero Cruise Speed (10.3 m/s)')
    ax3.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax3.set_ylabel("Flight Velocity (m/s)", fontsize=11)
    ax3.set_title("Panel C: Velocity Dynamics & Cruising Efficiency", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', framealpha=0.9)

    # Panel D: Cumulative Propulsion Energy
    ax4 = axes[1, 1]
    cum_aero_a = np.sum([np.cumsum(astar_data[j]["power"][:100]) for j in range(4)], axis=0)
    cum_aero_m = np.sum([np.cumsum(mappo_data[j]["power"][:100]) for j in range(4)], axis=0)
    ax4.plot(time_steps, cum_aero_a, color='#dc2626', linestyle='--', linewidth=2.2, label=f"3D A* Fleet Propulsion ({cum_aero_a[-1]:.0f} J)")
    ax4.plot(time_steps, cum_aero_m, color='#0284c7', linestyle='-', linewidth=2.5, label=f"Proposed MAPPO Fleet Propulsion ({cum_aero_m[-1]:.0f} J)")
    ax4.fill_between(time_steps, cum_aero_m, cum_aero_a, color='#0284c7', alpha=0.1, label='Energy Savings Corridor')

    ax4.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax4.set_ylabel("Cumulative Propulsion Energy (Joules)", fontsize=11)
    ax4.set_title("Panel D: Cumulative Fleet Mechanical Propulsion Energy", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', framealpha=0.9)

    plt.suptitle("Figure 3: Deep-Dive Diagnostic Comparison: Proposed MAPPO vs 3D A* Baseline", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved MAPPO vs A* deep-dive figure: {save_path}")

def plot_mappo_vs_ippo_deepdive(mappo_data: dict, ippo_data: dict, obstacles: list, stations: dict, save_path: str):
    """
    Dedicated Pairwise Comparative Deep-Dive: Proposed Joint MAPPO (CTDE) vs IPPO (DTDE Ablation).
    Demonstrates the value of Centralized Training in multi-agent collision avoidance and offloading.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    time_steps = np.arange(100)
    reach_tol = 5.0

    # Panel A: 2D Spatial Corridors
    ax1 = axes[0, 0]
    for idx, (cx, cy, r, h) in enumerate(obstacles):
        circ = plt.Circle((cx, cy), r, color='#f43f5e', alpha=0.3)
        ax1.add_patch(circ)
        ax1.text(cx, cy, f"Obs #{idx}\n(r={r:.0f}m)", color='#e11d48', ha='center', va='center', fontsize=8, fontweight='bold')

    for j in range(4):
        g = stations[j]
        st_circ = plt.Circle((g[0], g[1]), reach_tol, color='#64748b', alpha=0.15, linestyle=':', fill=True)
        ax1.add_patch(st_circ)
        p_i = np.array(ippo_data[j]["pos"])
        p_m = np.array(mappo_data[j]["pos"])
        ax1.plot(p_i[:, 0], p_i[:, 1], color='#059669', linestyle='-.', linewidth=1.8, label="IPPO (DTDE)" if j == 0 else None)
        ax1.plot(p_m[:, 0], p_m[:, 1], color='#0284c7', linestyle='-', linewidth=2.4, label="Proposed MAPPO (CTDE)" if j == 0 else None)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("Airspace X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Airspace Y Coordinate (meters)", fontsize=11)
    ax1.set_title("Panel A: 2D Navigation Profiles (CTDE Coordination vs DTDE)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: Mean Distance to Station vs Time
    ax2 = axes[0, 1]
    m_dist_m = np.mean([mappo_data[j]["dist_goal"][:100] for j in range(4)], axis=0)
    m_dist_i = np.mean([ippo_data[j]["dist_goal"][:100] for j in range(4)], axis=0)
    s_dist_m = np.std([mappo_data[j]["dist_goal"][:100] for j in range(4)], axis=0)
    s_dist_i = np.std([ippo_data[j]["dist_goal"][:100] for j in range(4)], axis=0)

    ax2.plot(time_steps, m_dist_i, color='#059669', linestyle='-.', linewidth=2.2, label='IPPO (DTDE) Mean Distance')
    ax2.fill_between(time_steps, np.maximum(0, m_dist_i - s_dist_i), m_dist_i + s_dist_i, color='#059669', alpha=0.15)
    ax2.plot(time_steps, m_dist_m, color='#0284c7', linestyle='-', linewidth=2.5, label='Proposed MAPPO (CTDE) Mean Distance')
    ax2.fill_between(time_steps, np.maximum(0, m_dist_m - s_dist_m), m_dist_m + s_dist_m, color='#0284c7', alpha=0.15)

    ax2.axhline(reach_tol, color='#10b981', linestyle=':', linewidth=2, label=f'Reach Tolerance ({reach_tol:.0f}m)')
    ax2.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax2.set_ylabel("Mean Distance to Station (meters)", fontsize=11)
    ax2.set_title("Panel B: Station Convergence & Fleet Coordination Band", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Panel C: Minimum Inter-UAV Separation Distance
    ax3 = axes[1, 0]
    min_sep_m = []
    min_sep_i = []
    for t in range(100):
        # MAPPO pairwise
        d_m_list = []
        d_i_list = []
        for j in range(4):
            for k in range(j + 1, 4):
                d_m_list.append(np.linalg.norm(mappo_data[j]["pos"][t] - mappo_data[k]["pos"][t]))
                d_i_list.append(np.linalg.norm(ippo_data[j]["pos"][t] - ippo_data[k]["pos"][t]))
        min_sep_m.append(min(d_m_list))
        min_sep_i.append(min(d_i_list))

    ax3.plot(time_steps, min_sep_i, color='#059669', linestyle='-.', linewidth=2.0, label='IPPO (DTDE) Min Inter-UAV Sep')
    ax3.plot(time_steps, min_sep_m, color='#0284c7', linestyle='-', linewidth=2.4, label='Proposed MAPPO (CTDE) Min Inter-UAV Sep')
    ax3.axhline(8.0, color='#dc2626', linestyle='--', linewidth=2, label=r'Safety Threshold $d_{safe}=8.0$m')
    ax3.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax3.set_ylabel("Minimum Inter-UAV Distance (meters)", fontsize=11)
    ax3.set_title("Panel C: Fleet Collision Avoidance & Inter-UAV Separation", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right', framealpha=0.9)

    # Panel D: Cumulative Total Energy Comparison
    ax4 = axes[1, 1]
    cum_e_m = np.sum([np.cumsum(mappo_data[j]["power"][:100]) for j in range(4)], axis=0)
    cum_e_i = np.sum([np.cumsum(ippo_data[j]["power"][:100]) for j in range(4)], axis=0)
    ax4.plot(time_steps, cum_e_i, color='#059669', linestyle='-.', linewidth=2.2, label=f"IPPO Total Energy ({cum_e_i[-1]:.0f} J)")
    ax4.plot(time_steps, cum_e_m, color='#0284c7', linestyle='-', linewidth=2.5, label=f"Proposed MAPPO Total Energy ({cum_e_m[-1]:.0f} J)")
    ax4.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax4.set_ylabel("Cumulative Propulsion Energy (Joules)", fontsize=11)
    ax4.set_title("Panel D: Fleet Energy Efficiency Evolution", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', framealpha=0.9)

    plt.suptitle("Figure 4: CTDE Ablation Study: Proposed Joint MAPPO vs Independent PPO (IPPO)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved MAPPO vs IPPO deep-dive figure: {save_path}")

def plot_mappo_vs_maddpg_deepdive(mappo_data: dict, maddpg_data: dict, obstacles: list, stations: dict, save_path: str):
    """
    Dedicated Pairwise Comparative Deep-Dive: Proposed Joint MAPPO vs Multi-Agent DDPG (MADDPG).
    Evaluates On-Policy PPO vs Off-Policy Deterministic Actor-Critic under identical conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    time_steps = np.arange(100)
    reach_tol = 5.0

    # Panel A: 2D Flight Trajectories
    ax1 = axes[0, 0]
    for idx, (cx, cy, r, h) in enumerate(obstacles):
        circ = plt.Circle((cx, cy), r, color='#f43f5e', alpha=0.3)
        ax1.add_patch(circ)
        ax1.text(cx, cy, f"Obs #{idx}\n(r={r:.0f}m)", color='#e11d48', ha='center', va='center', fontsize=8, fontweight='bold')

    for j in range(4):
        g = stations[j]
        st_circ = plt.Circle((g[0], g[1]), reach_tol, color='#64748b', alpha=0.15, linestyle=':', fill=True)
        ax1.add_patch(st_circ)
        p_d = np.array(maddpg_data[j]["pos"])
        p_m = np.array(mappo_data[j]["pos"])
        ax1.plot(p_d[:, 0], p_d[:, 1], color='#8b5cf6', linestyle=':', linewidth=2.0, label="MADDPG" if j == 0 else None)
        ax1.plot(p_m[:, 0], p_m[:, 1], color='#0284c7', linestyle='-', linewidth=2.4, label="Proposed MAPPO" if j == 0 else None)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("Airspace X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Airspace Y Coordinate (meters)", fontsize=11)
    ax1.set_title("Panel A: 2D Spatial Corridors (MAPPO vs MADDPG)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: Mean Distance to Station vs Time
    ax2 = axes[0, 1]
    m_dist_m = np.mean([mappo_data[j]["dist_goal"][:100] for j in range(4)], axis=0)
    m_dist_d = np.mean([maddpg_data[j]["dist_goal"][:100] for j in range(4)], axis=0)
    ax2.plot(time_steps, m_dist_d, color='#8b5cf6', linestyle=':', linewidth=2.2, label='MADDPG Mean Distance')
    ax2.plot(time_steps, m_dist_m, color='#0284c7', linestyle='-', linewidth=2.5, label='Proposed MAPPO Mean Distance')
    ax2.axhline(reach_tol, color='#10b981', linestyle=':', linewidth=2, label=f'Reach Tolerance ({reach_tol:.0f}m)')
    ax2.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax2.set_ylabel("Mean Distance to Station (meters)", fontsize=11)
    ax2.set_title("Panel B: Station Convergence & Policy Stability", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Panel C: Flight Velocity Profile Dynamics
    ax3 = axes[1, 0]
    spd_m = np.mean([mappo_data[j]["speed"][:100] for j in range(4)], axis=0)
    spd_d = np.mean([maddpg_data[j]["speed"][:100] for j in range(4)], axis=0)
    ax3.plot(time_steps, spd_d, color='#8b5cf6', linestyle=':', linewidth=2.0, label='MADDPG Fleet Mean Speed')
    ax3.plot(time_steps, spd_m, color='#0284c7', linestyle='-', linewidth=2.4, label='Proposed MAPPO Fleet Mean Speed')
    ax3.axhline(10.3, color='#059669', linestyle='-.', linewidth=1.8, label='Optimal Aero Cruise Speed (10.3 m/s)')
    ax3.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax3.set_ylabel("Mean Fleet Velocity (m/s)", fontsize=11)
    ax3.set_title("Panel C: Flight Velocity Profile & Aerodynamic Cruise Efficiency", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', framealpha=0.9)

    # Panel D: Cumulative Energy Consumption
    ax4 = axes[1, 1]
    cum_e_m = np.sum([np.cumsum(mappo_data[j]["power"][:100]) for j in range(4)], axis=0)
    cum_e_d = np.sum([np.cumsum(maddpg_data[j]["power"][:100]) for j in range(4)], axis=0)
    ax4.plot(time_steps, cum_e_d, color='#8b5cf6', linestyle=':', linewidth=2.2, label=f"MADDPG Total Energy ({cum_e_d[-1]:.0f} J)")
    ax4.plot(time_steps, cum_e_m, color='#0284c7', linestyle='-', linewidth=2.5, label=f"Proposed MAPPO Total Energy ({cum_e_m[-1]:.0f} J)")
    ax4.set_xlabel("Episode Timestep (seconds)", fontsize=11)
    ax4.set_ylabel("Cumulative Propulsion Energy (Joules)", fontsize=11)
    ax4.set_title("Panel D: Cumulative Fleet Energy Consumption Evolution", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', framealpha=0.9)

    plt.suptitle("Figure 5: Policy Architecture Benchmark: Proposed MAPPO vs MADDPG", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved MAPPO vs MADDPG deep-dive figure: {save_path}")

def plot_offload_distribution_stacked(offload_data: dict, save_path: str):
    """
    Plots stacked horizontal or vertical bar chart of task offloading distribution
    (Local % vs Peer % vs Edge Server %) across evaluated algorithms.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    labels = list(offload_data.keys())
    local_pcts = [offload_data[m].get("local", 0.0) for m in labels]
    peer_pcts = [offload_data[m].get("peer", 0.0) for m in labels]
    edge_pcts = [offload_data[m].get("edge", 0.0) for m in labels]

    y_pos = np.arange(len(labels))
    bar_height = 0.55

    p1 = ax.barh(y_pos, local_pcts, bar_height, color='#0284c7', label='Local UAV Execution')
    p2 = ax.barh(y_pos, peer_pcts, bar_height, left=local_pcts, color='#10b981', label='Cooperative Peer Offload')
    left_edge = [l + p for l, p in zip(local_pcts, peer_pcts)]
    p3 = ax.barh(y_pos, edge_pcts, bar_height, left=left_edge, color='#f59e0b', label='Ground Edge Server Offload')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Offloading Destination Percentage (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title("Figure 6: Computational Task Offloading Distribution Breakdown", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')

    # Add text labels on bars
    for idx in range(len(labels)):
        loc = local_pcts[idx]
        pe = peer_pcts[idx]
        ed = edge_pcts[idx]
        if loc > 6:
            ax.text(loc / 2, idx, f"{loc:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if pe > 6:
            ax.text(loc + pe / 2, idx, f"{pe:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if ed > 6:
            ax.text(loc + pe + ed / 2, idx, f"{ed:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved offload distribution breakdown: {save_path}")

def plot_pareto_latency_energy(results_dict: dict, save_path: str):
    """
    Plots Latency vs Energy Multi-Objective Pareto Trade-Off Curve.
    Strictly adheres to IEEE Transactions on Wireless Communications / Robotics standards:
    - Standard column sizing (6.2 x 4.5 inches)
    - Serif typography (DejaVu Serif / Times New Roman)
    - Empirical Pareto-optimal non-dominated frontier curve
    - Distinct, colorblind-friendly geometric markers and colors
    - Clean compact IEEE legend in upper right
    - Saves both high-res 300 DPI PNG and vector PDF
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.color": "#CBD5E1",
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=300)

    # Standard IEEE publication color palette
    colors = {
        "mappo_residual": "#1D4ED8",  # Royal Blue
        "mappo_pure": "#0284C7",      # Cyan Blue
        "mappo": "#1D4ED8",
        "cppo": "#0D9488",            # Teal
        "ippo": "#059669",            # Emerald
        "maddpg": "#7C3AED",          # Purple
        "astar_greedy": "#DC2626",    # Red
        "astar_latency": "#EA580C",   # Orange
        "local_only": "#64748B",      # Slate
        "edge_only": "#D97706",       # Amber
        "random": "#94A3B8"           # Gray
    }

    markers = {
        "mappo_residual": "*",
        "mappo_pure": "o",
        "mappo": "*",
        "cppo": "s",
        "ippo": "^",
        "maddpg": "D",
        "astar_greedy": "v",
        "astar_latency": "P",
        "local_only": "X",
        "edge_only": "p",
        "random": "h"
    }

    # Extract evaluated points
    pts = []
    for k, v in results_dict.items():
        lat = float(v.get("latency", 0.0))
        ene_kj = float(v.get("energy", 0.0)) / 1000.0
        lbl = v.get("label", k)
        c = colors.get(k, "#334155")
        m = markers.get(k, "o")
        s = 110 if m in ["*", "D"] else 70
        pts.append({"key": k, "lat": lat, "ene": ene_kj, "lbl": lbl, "color": c, "marker": m, "size": s})

    # Compute Pareto frontier (minimizing both latency and energy)
    sorted_pts = sorted(pts, key=lambda p: p["lat"])
    frontier = []
    min_ene = float("inf")
    for p in sorted_pts:
        if p["ene"] < min_ene:
            frontier.append(p)
            min_ene = p["ene"]

    # Draw Pareto-optimal frontier line
    if len(frontier) >= 2:
        fx = [p["lat"] for p in frontier]
        fy = [p["ene"] for p in frontier]
        ax.plot(fx, fy, linestyle="--", color="#1E3A8A", linewidth=1.4, alpha=0.85, zorder=3,
                label=r"Pareto Frontier $\mathcal{P}^*$")

    # Plot policy scatter points
    for p in pts:
        ax.scatter(p["lat"], p["ene"], color=p["color"], s=p["size"], marker=p["marker"],
                   edgecolors="#0F172A", linewidth=0.7, zorder=5, label=p["lbl"])

    # Clean axes styling
    ax.set_xlabel(r"Mean Fleet Task Latency $T_{\mathrm{delay}}$ (s)", fontsize=10, labelpad=6)
    ax.set_ylabel(r"Fleet Cumulative Energy $E_{\mathrm{fleet}}$ (kJ)", fontsize=10, labelpad=6)
    ax.set_title(r"Multi-Objective Pareto Trade-Off: Task Latency vs. Fleet Energy",
                 fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, linestyle=":", alpha=0.55)

    # Clean IEEE Legend inside upper right
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#CBD5E1",
        fontsize=8.0,
        ncol=1
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Vector PDF export
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved clean IEEE Pareto trade-off figure:\n  -> {save_path}\n  -> {pdf_path}")

def plot_ablation_comparison(ablation_dict: dict, save_path: str):
    """
    Plots comparative bar group of ablation studies:
    - Joint MAPPO (Full System)
    - MAPPO Navigation + Fixed Greedy Offloading
    - Classical A* Navigation + MAPPO Learned Offloading
    - MAPPO without Energy Reward Shaping
    - MAPPO without Safety Handling
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    labels = list(ablation_dict.keys())
    returns = [ablation_dict[k]["return"] for k in labels]
    energies_kj = [ablation_dict[k]["energy"] / 1000.0 for k in labels]
    interventions = [ablation_dict[k]["interventions"] for k in labels]

    colors = ['#0284c7', '#0ea5e9', '#059669', '#d97706', '#dc2626']
    x = np.arange(len(labels))

    # Panel 1: Return
    axes[0].bar(x, returns, color=colors[:len(labels)], width=0.55, edgecolor='black', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    axes[0].set_ylabel("Episode Return", fontsize=11)
    axes[0].set_title("(a) Cumulative Episode Return", fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6, axis='y')

    # Panel 2: Total Energy
    axes[1].bar(x, energies_kj, color=colors[:len(labels)], width=0.55, edgecolor='black', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    axes[1].set_ylabel("Fleet Energy (kJ)", fontsize=11)
    axes[1].set_title("(b) Fleet Energy Consumption", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6, axis='y')

    # Panel 3: Safety Interventions
    axes[2].bar(x, interventions, color=colors[:len(labels)], width=0.55, edgecolor='black', linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    axes[2].set_ylabel("Safety Interventions", fontsize=11)
    axes[2].set_title("(c) Analytical Safety Interventions", fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.suptitle("Figure 8: Component Ablation Study on Proposed Joint MAPPO Framework", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved ablation study comparison figure: {save_path}")

def plot_3d_flight_trajectories(eval_results: dict, obstacles: list, stations: dict, save_path: str):
    """
    Generates high-resolution 3D visualization showing:
    - 3D Cylindrical obstacle geometry
    - Designated station locations
    - 3D flight trajectories for A* and MAPPO
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#0284c7', '#8b5cf6', '#d97706', '#059669']

    # Draw Cylinders
    for cx, cy, r, h in obstacles:
        z_grid = np.linspace(20.0, h, 20)
        theta_grid = np.linspace(0, 2 * np.pi, 30)
        theta_mesh, z_mesh = np.meshgrid(theta_grid, z_grid)
        x_mesh = cx + r * np.cos(theta_mesh)
        y_mesh = cy + r * np.sin(theta_mesh)
        ax.plot_surface(x_mesh, y_mesh, z_mesh, color='#f43f5e', alpha=0.25, edgecolor='none')

    # Mission center & Ground Edge Server
    ax.scatter([100.0], [100.0], [0.0], color='#dc2626', s=120, marker='s', label='Ground Edge Server (100, 100, 0)')

    # Stations & Trajectories
    for j in range(4):
        g = stations[j]
        ax.scatter([g[0]], [g[1]], [g[2]], color=colors[j], s=100, marker='^', label=f'Station #{j}' if j == 0 else None)

        if "astar" in eval_results:
            p_a = np.array(eval_results["astar"][j]["pos"])
            ax.plot(p_a[:, 0], p_a[:, 1], p_a[:, 2], color=colors[j], linestyle='--', linewidth=1.8, label=f'3D A* UAV #{j}' if j == 0 else None)
        if "mappo" in eval_results:
            p_m = np.array(eval_results["mappo"][j]["pos"])
            ax.plot(p_m[:, 0], p_m[:, 1], p_m[:, 2], color=colors[j], linestyle='-', linewidth=2.4, label=f'MAPPO UAV #{j}' if j == 0 else None)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_zlim(0, 100)
    ax.set_xlabel("X Position (m)", fontsize=10)
    ax.set_ylabel("Y Position (m)", fontsize=10)
    ax.set_zlabel("Altitude Z (m)", fontsize=10)
    ax.set_title("Figure 9: 3D Airspace Trajectory Visualization with Cylindrical Obstacles", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved 3D flight trajectories figure: {save_path}")

def plot_all_marl_training_convergence(csv_dict: dict, save_path: str):
    """
    Plots multi-algorithm MARL training convergence comparison (MAPPO vs IPPO vs MADDPG vs CPPO):
    - (a) Episode Return
    - (b) Task Completion Rate (%)
    - (c) Fleet Energy Consumption (kJ)
    - (d) Validation Score / Stability
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    palette = {
        "mappo": {"name": "Proposed Joint MAPPO (CTDE)", "color": "#0284c7", "ls": "-", "lw": 2.5},
        "ippo": {"name": "Independent PPO (DTDE)", "color": "#059669", "ls": "-.", "lw": 2.0},
        "maddpg": {"name": "Multi-Agent DDPG (Off-Policy)", "color": "#8b5cf6", "ls": ":", "lw": 2.0},
        "cppo": {"name": "Centralized PPO (CTCE Upper Bound)", "color": "#dc2626", "ls": "--", "lw": 2.0}
    }

    loaded_any = False
    for key, path in csv_dict.items():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            loaded_any = True
            cfg = palette.get(key, {"name": key.upper(), "color": "#334155", "ls": "-", "lw": 1.8})
            w = max(3, len(df) // 10)

            # (a) Episode Return
            ax1 = axes[0, 0]
            smooth_ret = df["return"].rolling(w, min_periods=1).mean()
            ax1.plot(df["episode"], smooth_ret, color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"], label=cfg["name"])

            # (b) Task Completion Rate
            ax2 = axes[0, 1]
            comp = (df["completed_tasks"] / np.maximum(1, df["completed_tasks"] + df["deadline_misses"])) * 100.0
            smooth_comp = comp.rolling(w, min_periods=1).mean()
            ax2.plot(df["episode"], smooth_comp, color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"], label=cfg["name"])

            # (c) Fleet Energy (kJ)
            ax3 = axes[1, 0]
            energy_kj = df["total_energy"] / 1000.0
            smooth_e = energy_kj.rolling(w, min_periods=1).mean()
            ax3.plot(df["episode"], smooth_e, color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"], label=cfg["name"])

            # (d) Validation Score
            ax4 = axes[1, 1]
            if "val_return" in df.columns and df["val_return"].notna().sum() > 0:
                valid_val = df.dropna(subset=["val_return"])
                ax4.plot(valid_val["episode"], valid_val["val_return"], color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"], marker="o", markersize=4, label=cfg["name"])
        except Exception as e:
            print(f"[Plotter] Warning: Failed to process {path}: {e}")

    if not loaded_any:
        print("[Plotter] Warning: No training logs found for multi-algorithm plot.")
        plt.close()
        return

    axes[0, 0].set_title("(a) Cumulative Episode Return vs Episode", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Training Episode", fontsize=10)
    axes[0, 0].set_ylabel("Episode Return", fontsize=10)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    axes[0, 1].set_title("(b) Task Completion Rate (%) vs Episode", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Training Episode", fontsize=10)
    axes[0, 1].set_ylabel("Completion Rate (%)", fontsize=10)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    axes[1, 0].set_title("(c) Fleet Mechanical & Comm Energy (kJ) vs Episode", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Training Episode", fontsize=10)
    axes[1, 0].set_ylabel("Total Energy (kJ)", fontsize=10)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    axes[1, 1].set_title("(d) Held-Out Validation Return vs Episode", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Training Episode", fontsize=10)
    axes[1, 1].set_ylabel("Validation Return", fontsize=10)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Multi-Agent Reinforcement Learning Training Convergence Comparison", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved MARL training convergence figure:\n  -> {save_path}\n  -> {pdf_path}")


