"""
Experiment A: 3D UAV Path Planning & Trajectory Comparison Benchmark.
Simulates and benchmarks all navigation methods under identical cylindrical obstacle geometry:
1. 3D A* (Classical Geometric Detour)
2. 3D RRT* (Sampling-Based Asymptotic Optimal Tree)
3. 3D SAPF (Sequential Adaptive Potential Field with Tangential Circulation)
4. Proposed Joint MAPPO (CTDE Multi-Agent DRL)
5. Independent PPO (IPPO, DTDE)
6. Multi-Agent DDPG (MADDPG)
7. Centralized PPO (CPPO, CTCE)

Outputs:
- results/trajectory_records.csv (step-by-step 3D flight coordinates)
- results/navigation_benchmark.csv (path length, clearance, planning time, success rate)
- results/figures/3d_uav_trajectory_comparison.png (300 DPI publication visualizer)
- results/figures/3d_uav_trajectory_comparison.pdf (Vector PDF)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import MasterConfig, get_preset_config
from env.multi_uav_env import MultiUAVEnv
from utils.device import DeviceManager
from models.astar import AStarBaseline, HybridAStarMAPPONavigator, AStarPurePursuitWithMAPPOOffload
from models.rrt_star import RRTStarBaseline
from models.apf import SAPFBaseline
from models.mappo import MAPPOAgent
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent
from env.safety_filter import compute_min_clearance
from plots.render_publication_figures import plot_publication_3d_trajectory_benchmark

def measure_policy_inference_latency_ms(agent, obs, state, dev, num_warmup: int = 5, num_reps: int = 50) -> float:
    """Accurately measures neural policy forward pass latency with device synchronization and warm-up."""
    is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
    for _ in range(num_warmup):
        with torch.no_grad():
            if hasattr(agent, "get_actions"):
                if isinstance(agent, CPPOAgent):
                    _ = agent.get_actions(state, deterministic=True)
                elif isinstance(agent, MADDPGAgent):
                    _ = agent.get_actions(obs, explore=False)
                else:
                    _ = agent.get_actions(obs, deterministic=True)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_reps):
        with torch.no_grad():
            if hasattr(agent, "get_actions"):
                if isinstance(agent, CPPOAgent):
                    _ = agent.get_actions(state, deterministic=True)
                elif isinstance(agent, MADDPGAgent):
                    _ = agent.get_actions(obs, explore=False)
                else:
                    _ = agent.get_actions(obs, deterministic=True)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0 / num_reps
    return float(elapsed)

def run_navigation_suite(
    seed: int = 101,
    device: str = "auto",
    preset: str = "benchmark",
    num_uavs: int = 4,
    output_dir: str = None,
    strict_ckpts: bool = False
) -> pd.DataFrame:
    if num_uavs != 4:
        raise ValueError(
            f"Unsupported fleet size num_uavs={num_uavs}. Canonical Two-Stream MAPPO strictly supports N=4 "
            f"(3 peer channels, 5 offload actions, 52-dim critic state). Variable fleet sizes (N in {{6, 8, 12}}) "
            f"require permutation-invariant network architectures and are currently unsupported by neural models."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(output_dir) if output_dir else os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(base_dir, "results", "checkpoints")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Experiment": f"3D Navigation Benchmark (Preset: {preset}, UAVs: {num_uavs})"})

    cfg = get_preset_config(preset=preset, num_uavs=num_uavs)
    env = MultiUAVEnv(cfg)
    num_uavs = cfg.physics.num_uavs
    obstacles = cfg.airspace.obstacles
    stations = env.stations

    # 1. Initialize Planners & Navigators
    agents = {}
    planning_times_ms = {}

    # A* Planner
    astar = AStarBaseline(env)
    agents["astar"] = astar

    # RRT* Planner
    rrt_star = RRTStarBaseline(env)
    agents["rrt_star"] = rrt_star

    # SAPF (Adaptive Potential Field)
    sapf = SAPFBaseline(env)
    agents["sapf"] = sapf

    # 1. Initialize Planners & Navigators
    agents = {}
    agent_loaded = {}

    # A* Planner
    astar = AStarBaseline(env)
    agents["astar"] = astar

    # RRT* Planner
    rrt_star = RRTStarBaseline(env)
    agents["rrt_star"] = rrt_star

    # SAPF (Adaptive Potential Field)
    sapf = SAPFBaseline(env)
    agents["sapf"] = sapf

    # Pure MAPPO (Variant A: No hardcoded APF in actor)
    mappo_pure = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded = False
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_pure_agent.pt"), os.path.join(ckpt_dir, "final_mappo_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_mappo_agent.pt"), os.path.join(ckpt_dir, "final_mappo_agent.pt")]:
        if os.path.exists(ckpt):
            try:
                mappo_pure.load_checkpoint(ckpt)
                print(f"[Loaded] Pure MAPPO from {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"[Warning] Failed loading {ckpt}: {e}")
    agent_loaded["mappo_pure"] = loaded
    agent_loaded["mappo"] = loaded
    if strict_ckpts and not loaded:
        raise RuntimeError("Missing/incompatible checkpoint for mappo_pure under strict_ckpts=True")
    agents["mappo_pure"] = mappo_pure
    agents["mappo"] = mappo_pure  # Backward-compatible alias

    # APF-Assisted Residual MAPPO (Variant B: Nominal APF + learned residual)
    mappo_residual = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="residual", device=dev)
    loaded = False
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_residual_agent.pt"), os.path.join(ckpt_dir, "final_mappo_residual_agent.pt")]:
        if os.path.exists(ckpt):
            try:
                mappo_residual.load_checkpoint(ckpt)
                print(f"[Loaded] Residual MAPPO from {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"[Warning] Failed loading {ckpt}: {e}")
    agent_loaded["mappo_residual"] = loaded
    if strict_ckpts and not loaded:
        raise RuntimeError("Missing/incompatible checkpoint for mappo_residual under strict_ckpts=True")
    agents["mappo_residual"] = mappo_residual

    # IPPO
    ippo = IPPOAgent(env.obs_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded = False
    for ckpt in [os.path.join(ckpt_dir, "best_ippo_pure_agent.pt"), os.path.join(ckpt_dir, "final_ippo_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_ippo_agent.pt"), os.path.join(ckpt_dir, "final_ippo_agent.pt")]:
        if os.path.exists(ckpt):
            try:
                ippo.load_checkpoint(ckpt)
                print(f"[Loaded] IPPO from {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"[Warning] Failed loading {ckpt}: {e}")
    agent_loaded["ippo"] = loaded
    if strict_ckpts and not loaded:
        raise RuntimeError("Missing/incompatible checkpoint for ippo under strict_ckpts=True")
    agents["ippo"] = ippo

    # MADDPG
    maddpg = MADDPGAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded = False
    for ckpt in [os.path.join(ckpt_dir, "best_maddpg_pure_agent.pt"), os.path.join(ckpt_dir, "final_maddpg_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_maddpg_agent.pt"), os.path.join(ckpt_dir, "final_maddpg_agent.pt")]:
        if os.path.exists(ckpt):
            try:
                maddpg.load_checkpoint(ckpt)
                print(f"[Loaded] MADDPG from {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"[Warning] Failed loading {ckpt}: {e}")
    agent_loaded["maddpg"] = loaded
    if strict_ckpts and not loaded:
        raise RuntimeError("Missing/incompatible checkpoint for maddpg under strict_ckpts=True")
    agents["maddpg"] = maddpg

    # CPPO
    cppo = CPPOAgent(env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, device=dev)
    loaded = False
    for ckpt in [os.path.join(ckpt_dir, "best_cppo_agent.pt"), os.path.join(ckpt_dir, "final_cppo_agent.pt")]:
        if os.path.exists(ckpt):
            try:
                cppo.load_checkpoint(ckpt)
                print(f"[Loaded] CPPO from {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"[Warning] Failed loading {ckpt}: {e}")
    agent_loaded["cppo"] = loaded
    if strict_ckpts and not loaded:
        raise RuntimeError("Missing/incompatible checkpoint for cppo under strict_ckpts=True")
    agents["cppo"] = cppo

    # Pure APF Controller (Ablation: Nominal gain 2.0, zero learning)
    agents["apf_only"] = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="apf_only", device=dev)

    # 3D A* + Pure Pursuit (No RL Ablation: 100% Classical corner-cutting tracking)
    agents["astar_pure_pursuit"] = HybridAStarMAPPONavigator(env, mappo_agent=None, residual_weight=0.0)

    # Hierarchical A* + MAPPO Offload (Residual weight = 0.0: Movement 100% A* + Pure Pursuit, Offloading from MAPPO)
    agents["hybrid_astar_mappo_zero"] = HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.0)
    agents["hybrid_astar_mappo"] = agents["hybrid_astar_mappo_zero"]

    # Hierarchical A* + MAPPO Residual (Residual weight = 0.2: Learned RL residual movement active)
    agents["hybrid_astar_mappo_res02"] = HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.2)

    # 2. Simulate & Record Trajectories
    print("\nSimulating physical trajectories under genuine aerodynamics and obstacle geometry...")
    trajectory_rows = []
    nav_summary_rows = []

    algo_order = [
        ("3D A* Baseline", "astar"),
        ("3D A* + Pure Pursuit (Zero RL)", "astar_pure_pursuit"),
        ("Hybrid A* + MAPPO Offload (w_res=0.0)", "hybrid_astar_mappo_zero"),
        ("Hybrid A* + MAPPO Residual (w_res=0.2)", "hybrid_astar_mappo_res02"),
        ("3D RRT* (Optimal Tree)", "rrt_star"),
        ("3D SAPF (Adaptive Potential Field)", "sapf"),
        ("APF-Only Controller (Zero Learning)", "apf_only"),
        ("Centralized PPO (CPPO)", "cppo"),
        ("Independent PPO (IPPO)", "ippo"),
        ("Multi-Agent DDPG (MADDPG)", "maddpg"),
        ("Proposed Pure MAPPO (No APF)", "mappo_pure"),
        ("APF-Assisted Residual MAPPO", "mappo_residual")
    ]

    initial_positions = {}

    for label, key in algo_order:
        agent = agents[key]
        obs, state = env.reset(seed=seed)
        if len(initial_positions) == 0:
            initial_positions = {j: np.copy(env.positions[j]) for j in range(num_uavs)}

        if hasattr(agent, "reset"):
            agent.reset()

        hist_pos = {j: [np.copy(env.positions[j])] for j in range(num_uavs)}
        hist_clearance = {j: [] for j in range(num_uavs)}
        arrival_steps = {j: None for j in range(num_uavs)}
        total_interventions = 0

        # Measure planning time if available or microbenchmark inference latency
        t_plan_start = time.perf_counter()
        if key in ["astar", "rrt_star", "sapf", "hybrid_astar_mappo", "hybrid_astar_mappo_zero", "hybrid_astar_mappo_res02", "astar_pure_pursuit"]:
            _ = agent.select_actions(obs)
            total_plan_time = (time.perf_counter() - t_plan_start) * 1000.0 / num_uavs
        else:
            total_plan_time = measure_policy_inference_latency_ms(agent, obs, state, dev)

        # Record initial timestep (t=0)
        for j in range(num_uavs):
            p = env.positions[j]
            d_g = float(np.linalg.norm(p - stations[j]))
            min_c = compute_min_clearance(p, obstacles, margin=2.0)
            trajectory_rows.append({
                "algorithm": key,
                "label": label,
                "scenario_seed": seed,
                "timestep": 0,
                "uav_id": j,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "speed": 0.0,
                "dist_to_goal": d_g,
                "min_obs_clearance": min_c,
                "cumulative_energy": 0.0
            })

        cum_energy = 0.0
        total_proposed_collisions = 0
        total_actual_collisions = 0
        last_info = {}

        for t in range(1, cfg.mappo.episode_length + 1):
            if key in ["astar", "rrt_star", "sapf", "hybrid_astar_mappo", "hybrid_astar_mappo_zero", "hybrid_astar_mappo_res02", "astar_pure_pursuit"]:
                actions = agent.select_actions(obs)
            elif key == "cppo":
                actions, _, _, _ = agent.get_actions(state, deterministic=True)
            elif key in ["mappo", "mappo_pure", "mappo_residual", "ippo", "apf_only"]:
                actions, _, _, _ = agent.get_actions(obs, deterministic=True)
            elif key == "maddpg":
                actions, _ = agent.get_actions(obs, explore=False)

            obs, state, _, done, info = env.step(actions)
            last_info = info
            cum_energy += info["step_energy"]
            total_interventions += info.get("safety_interventions", 0)
            total_proposed_collisions += info.get("proposed_collisions", 0)
            total_actual_collisions += info.get("actual_collisions", 0)

            for j in range(num_uavs):
                p = np.copy(env.positions[j])
                spd = float(np.linalg.norm(env.velocities[j]))
                d_g = float(np.linalg.norm(p - stations[j]))
                min_c = compute_min_clearance(p, obstacles, margin=2.0)
                hist_pos[j].append(p)
                hist_clearance[j].append(min_c)

                if d_g <= cfg.physics.reach_tolerance and arrival_steps[j] is None:
                    arrival_steps[j] = t

                trajectory_rows.append({
                    "algorithm": key,
                    "label": label,
                    "scenario_seed": seed,
                    "timestep": t,
                    "uav_id": j,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "speed": spd,
                    "dist_to_goal": d_g,
                    "min_obs_clearance": min_c,
                    "cumulative_energy": float(cum_energy)
                })

            if done:
                break

        # Calculate navigation benchmark metrics across fleet
        path_lengths = [
            float(last_info["path_length_to_goal"][j]) if (arrival_steps[j] is not None and "path_length_to_goal" in last_info)
            else float(np.sum(np.linalg.norm(np.diff(np.array(hist_pos[j]), axis=0), axis=1)))
            for j in range(num_uavs)
        ]
        mean_path_len = float(np.mean(path_lengths))
        all_clearances = [min(hist_clearance[j]) for j in range(num_uavs)] if hist_clearance[0] else [0.0]
        min_fleet_clearance = float(min(all_clearances))
        arrivals = [arrival_steps[j] if arrival_steps[j] is not None else cfg.mappo.episode_length for j in range(num_uavs)]
        mean_arrival_time = float(np.mean(arrivals))
        success_rate = (sum([1 for a in arrival_steps.values() if a is not None]) / num_uavs) * 100.0

        nav_energy_kj = float(last_info.get("fleet_energy_to_goal", cum_energy) / 1000.0)
        service_energy_kj = float(last_info.get("fleet_service_energy", 0.0) / 1000.0)
        total_energy_kj = float(cum_energy / 1000.0)

        display_label = label
        if key in agent_loaded and not agent_loaded[key]:
            display_label += " [UNTRAINED]"

        nav_summary_rows.append({
            "Algorithm": display_label,
            "Key": key,
            "Mean Path Length (m)": round(mean_path_len, 2),
            "UAV Mean Path (m)": round(mean_path_len, 2),
            "Fleet Total Path (m)": round(float(np.sum(path_lengths)), 2),
            "Arrival Time (s)": round(mean_arrival_time, 1),
            "Nav Energy (kJ)": round(nav_energy_kj, 2),
            "Service Energy (kJ)": round(service_energy_kj, 2),
            "Total Energy (kJ)": round(total_energy_kj, 2),
            "Min Clearance (m)": round(min_fleet_clearance, 2),
            "Planning Time (ms)": round(total_plan_time, 1),
            "Safety Interventions": total_interventions,
            "Actual Collisions": total_actual_collisions,
            "Mission Success (%)": round(success_rate, 1)
        })

    # Save CSV Records
    df_traj = pd.DataFrame(trajectory_rows)
    df_nav = pd.DataFrame(nav_summary_rows)

    traj_csv = os.path.join(res_dir, "trajectory_records.csv")
    nav_csv = os.path.join(res_dir, "navigation_benchmark.csv")
    df_traj.to_csv(traj_csv, index=False)
    df_nav.to_csv(nav_csv, index=False)
    print(f"\n[Saved] Trajectory records CSV: {traj_csv}")
    print(f"[Saved] Navigation benchmark CSV: {nav_csv}")

    print("\n" + "=" * 95)
    print("EXPERIMENT A: 3D UAV PATH PLANNING & NAVIGATION BENCHMARK TABLE:")
    print("=" * 95)
    print(df_nav[["Algorithm", "Mean Path Length (m)", "Arrival Time (s)", "Min Clearance (m)", "Planning Time (ms)", "Safety Interventions", "Mission Success (%)"]].to_string(index=False))
    print("=" * 95)

    # Generate Publication Figures (PNG at 300 DPI and Vector PDF)
    png_path = os.path.join(fig_dir, "3d_uav_trajectory_comparison.png")
    pdf_path = os.path.join(fig_dir, "3d_uav_trajectory_comparison.pdf")

    plot_publication_3d_trajectory_benchmark(
        trajectories_df=df_traj,
        nav_metrics_df=df_nav,
        obstacles=obstacles,
        stations=stations,
        start_positions=initial_positions,
        save_png_path=png_path,
        save_pdf_path=pdf_path,
        focal_uav_id=0
    )

    return df_nav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Navigation Benchmark Suite")
    parser.add_argument("--seed", type=int, default=101, help="Evaluation scenario seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--preset", type=str, default="benchmark", choices=["benchmark", "metropolis", "chokepoint", "canyon"], help="Airspace scenario preset")
    parser.add_argument("--num_uavs", type=int, default=4, help="Fleet size")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save benchmark CSV and figures")
    parser.add_argument("--strict_ckpts", action="store_true", help="Fail closed if any neural model checkpoint is missing")
    args = parser.parse_args()

    run_navigation_suite(seed=args.seed, device=args.device, preset=args.preset, num_uavs=args.num_uavs, output_dir=args.output_dir, strict_ckpts=args.strict_ckpts)

