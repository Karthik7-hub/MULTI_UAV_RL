"""
Standalone Publication Figure Regenerator.
Regenerates all figures, Pareto trade-off curves, offloading distributions,
and diagnostic trajectory plots from saved CSV logs and benchmark results
WITHOUT retraining any neural networks.

Usage:
    python plot_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from env.aerodynamics import AerodynamicsModel
from plots.plot_curves import (
    plot_training_convergence,
    plot_all_marl_training_convergence,
    plot_aerodynamic_power_curve,
    plot_offload_distribution_stacked,
    plot_pareto_latency_energy,
    plot_ablation_comparison
)

def regenerate_all_figures():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 75)
    print(" REGENERATING ALL PUBLICATION FIGURES FROM SAVED RAW EXPERIMENTAL DATA")
    print("=" * 75)

    # 1. Aerodynamic Power Curve
    pwr_path = os.path.join(fig_dir, "aerodynamic_power_curve.png")
    plot_aerodynamic_power_curve(AerodynamicsModel(), pwr_path)

    # 2. Training Convergence Curves
    log_mappo = os.path.join(res_dir, "training_logs_mappo.csv")
    if not os.path.exists(log_mappo):
        log_mappo = os.path.join(res_dir, "training_logs.csv")
    if os.path.exists(log_mappo):
        conv_path = os.path.join(fig_dir, "training_convergence_curves.png")
        plot_training_convergence(log_mappo, conv_path)

    # 2b. Multi-Algorithm MARL Comparison Curves
    csv_dict = {
        "mappo": os.path.join(res_dir, "training_logs_mappo_pure.csv") if os.path.exists(os.path.join(res_dir, "training_logs_mappo_pure.csv")) else log_mappo,
        "ippo": os.path.join(res_dir, "training_logs_ippo_pure.csv") if os.path.exists(os.path.join(res_dir, "training_logs_ippo_pure.csv")) else os.path.join(res_dir, "logs", "training_logs_ippo.csv"),
        "maddpg": os.path.join(res_dir, "training_logs_maddpg_pure.csv") if os.path.exists(os.path.join(res_dir, "training_logs_maddpg_pure.csv")) else os.path.join(res_dir, "logs", "training_logs_maddpg.csv"),
        "cppo": os.path.join(res_dir, "training_logs_cppo.csv") if os.path.exists(os.path.join(res_dir, "training_logs_cppo.csv")) else os.path.join(res_dir, "logs", "training_logs_cppo.csv")
    }
    marl_conv_path = os.path.join(fig_dir, "marl_training_convergence_comparison.png")
    plot_all_marl_training_convergence(csv_dict, marl_conv_path)

    # 3. Benchmark Summary & Pareto Curve
    bench_csv = os.path.join(res_dir, "benchmark_summary.csv")
    if not os.path.exists(bench_csv):
        bench_csv = os.path.join(res_dir, "benchmarks", "benchmark_summary.csv")
    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        print(f"[Loaded] Benchmark summary from {bench_csv}")

        def _parse_val(val_str, default=0.0):
            if pd.isna(val_str):
                return default
            s = str(val_str)
            for sep in ["±", "\ufffd", "+-", "+/-"]:
                if sep in s:
                    s = s.split(sep)[0]
            try:
                return float(s.strip())
            except Exception:
                return default

        # Extract data for Pareto Curve
        pareto_data = {}
        for _, row in df_bench.iterrows():
            m_name = str(row["Method"])
            lat = _parse_val(row.get("Delay (s)", row.get("Total Delay (s)", row.get("Total Latency / Delay (s)", 20.0))))
            # Check if columns indicate kJ or J
            if "Total Energy (kJ)" in row:
                ene_val = _parse_val(row["Total Energy (kJ)"]) * 1000.0
            else:
                ene_val = _parse_val(row.get("Total Energy (J)", 50000.0))

            k = m_name.lower().replace(" ", "_").replace("*", "").replace("+", "").replace("-", "_")[:12]
            pareto_data[k] = {
                "latency": lat,
                "energy": ene_val,
                "label": m_name
            }

        pareto_path = os.path.join(fig_dir, "pareto_latency_energy_tradeoff.png")
        plot_pareto_latency_energy(pareto_data, pareto_path)

    # 4. Offloading Distribution Breakdown
    offload_csv = os.path.join(res_dir, "offloading_distribution.csv")
    if not os.path.exists(offload_csv):
        offload_csv = os.path.join(res_dir, "benchmarks", "offloading_distribution.csv")
    if os.path.exists(offload_csv):
        df_off = pd.read_csv(offload_csv)
        offload_dict = {}
        for _, row in df_off.iterrows():
            offload_dict[row["Method"]] = {
                "local": float(row["Local (%)"]),
                "peer": float(row["Peer (%)"]),
                "edge": float(row["Edge (%)"])
            }
        offload_path = os.path.join(fig_dir, "offloading_distribution_breakdown.png")
        plot_offload_distribution_stacked(offload_dict, offload_path)

    # 5. Ablation Study Breakdown
    ablation_csv = os.path.join(res_dir, "ablation_summary.csv")
    if not os.path.exists(ablation_csv):
        ablation_csv = os.path.join(res_dir, "benchmarks", "ablation_summary.csv")
    if os.path.exists(ablation_csv):
        df_abl = pd.read_csv(ablation_csv)
        ablation_dict = {}
        for _, row in df_abl.iterrows():
            ablation_dict[row["Ablation"]] = {
                "return": float(row["Return"]),
                "energy": float(row["Total Energy (J)"]),
                "interventions": float(row["Safety Interventions"])
            }
        abl_path = os.path.join(fig_dir, "ablation_study_comparison.png")
        plot_ablation_comparison(ablation_dict, abl_path)

    # 6. 3D UAV Trajectory Comparison (PNG & Vector PDF)
    traj_csv = os.path.join(res_dir, "trajectory_records.csv")
    if not os.path.exists(traj_csv):
        traj_csv = os.path.join(res_dir, "metropolis_trajectory_records.csv")
    nav_csv = os.path.join(res_dir, "navigation_benchmark.csv")
    if not os.path.exists(nav_csv):
        nav_csv = os.path.join(res_dir, "benchmarks", "navigation_benchmark.csv")
    if os.path.exists(traj_csv) and os.path.exists(nav_csv):
        from plots.render_publication_figures import plot_publication_3d_trajectory_benchmark
        df_traj = pd.read_csv(traj_csv)
        df_nav = pd.read_csv(nav_csv)
        cfg = MasterConfig()
        env = MultiUAVEnv(cfg)
        png_3d = os.path.join(fig_dir, "3d_uav_trajectory_comparison.png")
        pdf_3d = os.path.join(fig_dir, "3d_uav_trajectory_comparison.pdf")
        plot_publication_3d_trajectory_benchmark(
            trajectories_df=df_traj,
            nav_metrics_df=df_nav,
            obstacles=cfg.airspace.obstacles,
            stations=env.stations,
            start_positions=env.positions,
            save_png_path=png_3d,
            save_pdf_path=pdf_3d,
            focal_uav_id=0
        )

    print(f"\n[Complete] All figures generated in: {fig_dir}")

if __name__ == "__main__":
    regenerate_all_figures()
