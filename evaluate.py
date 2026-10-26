"""
Comprehensive Evaluation and Benchmarking Suite for Multi-UAV Edge Computing.
Evaluates learning algorithms and classical baselines across held-out test scenarios:
1. Proposed Joint MAPPO (CTDE with dual-head Actor & Centralized Critic)
2. Centralized PPO (CPPO, CTCE)
3. Independent PPO (IPPO, Decentralized DTDE baseline)
4. Multi-Agent DDPG (MADDPG, Continuous Centralized Q-Critic baseline)
5. 3D A* + Energy- and Deadline-Aware Greedy MEC
6. 3D A* + Lowest-Latency Greedy MEC
7. 3D A* + Local-Only Execution
8. 3D A* + Edge-Only Execution
9. 3D A* + Random Feasible Offloading

Outputs:
- Multi-seed benchmark summary (results/benchmark_summary.csv)
- Offloading distribution breakdown (results/offloading_distribution.csv)
- Full publication figures in results/figures/
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def get_file_sha256(path: str) -> str:
    """Computes SHA256 hex digest for a file."""
    if not os.path.isfile(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from utils.device import DeviceManager
from models.mappo import MAPPOAgent
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent
from models.astar import AStarBaseline, HybridAStarMAPPONavigator
from models.baselines import (
    AStarJointBaseline,
    EnergyDeadlineAwareGreedyPolicy,
    LowestLatencyPolicy,
    LocalOnlyPolicy,
    EdgeOnlyPolicy,
    RandomFeasiblePolicy
)
from plots.plot_curves import (
    plot_trajectory_diagnostic,
    plot_multimodel_curves,
    plot_mappo_vs_astar_deepdive,
    plot_mappo_vs_ippo_deepdive,
    plot_mappo_vs_maddpg_deepdive,
    plot_offload_distribution_stacked,
    plot_pareto_latency_energy,
    plot_3d_flight_trajectories
)

def evaluate_episode(env: MultiUAVEnv, model_key: str, agent: Any, seed: int = 101) -> Dict[str, Any]:
    obs, state = env.reset(seed=seed)
    num_uavs = env.num_uavs
    cfg = env.cfg

    hist = {j: {"pos": [], "speed": [], "dist_goal": [], "power": []} for j in range(num_uavs)}
    step_trajectories = []
    total_energy = 0.0
    total_prop_energy = 0.0
    total_comm_energy = 0.0
    total_comp_energy = 0.0
    total_completed = 0
    total_missed = 0
    total_delay = 0.0
    total_interventions = 0
    total_proposed_collisions = 0
    total_actual_collisions = 0
    offload_counts = {"local": 0, "peer": 0, "edge": 0}
    last_info = {}

    if hasattr(agent, "reset"):
        agent.reset()

    # Record initial state (t=0)
    for j in range(num_uavs):
        pos = np.copy(env.positions[j])
        spd = float(np.linalg.norm(env.velocities[j]))
        d_g = float(np.linalg.norm(pos - env.stations[j]))
        pwr = float(env.aero.power(spd))
        hist[j]["pos"].append(pos)
        hist[j]["speed"].append(spd)
        hist[j]["dist_goal"].append(d_g)
        hist[j]["power"].append(pwr)

    for t in range(1, cfg.mappo.episode_length + 1):
        if hasattr(agent, "select_actions"):
            actions = agent.select_actions(obs)
        elif model_key == "cppo":
            actions, _, _, _ = agent.get_actions(state, deterministic=True)
        elif model_key == "maddpg":
            actions, _ = agent.get_actions(obs, explore=False)
        elif hasattr(agent, "get_actions"):
            actions, _, _, _ = agent.get_actions(obs, deterministic=True)
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        obs, state, _, done, info = env.step(actions)
        last_info = info
        total_energy += info["step_energy"]
        total_prop_energy += info["propulsion_energy"]
        total_comm_energy += info.get("comm_energy", 0.0)
        total_comp_energy += info.get("comp_energy", 0.0)
        total_completed += info["tasks_completed"]
        total_missed += info["deadline_misses"]
        total_delay += info["total_delay"]
        total_interventions += info.get("safety_interventions", 0)
        total_proposed_collisions += info.get("proposed_collisions", 0)
        total_actual_collisions += info.get("actual_collisions", 0)

        offload_counts["edge"] += info.get("offload_edge", 0)
        offload_counts["peer"] += info.get("offload_peer", 0)
        offload_counts["local"] += info.get("offload_local", 0)

        for j in range(num_uavs):
            pos = np.copy(env.positions[j])
            spd = float(np.linalg.norm(env.velocities[j]))
            d_g = float(np.linalg.norm(pos - env.stations[j]))
            pwr = float(env.aero.power(spd))
            hist[j]["pos"].append(pos)
            hist[j]["speed"].append(spd)
            hist[j]["dist_goal"].append(d_g)
            hist[j]["power"].append(pwr)

            step_trajectories.append({
                "timestep": t,
                "uav_id": j,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "speed": spd,
                "dist_goal": d_g,
                "arrived": bool(info["arrival_steps"].get(j) is not None),
                "step_energy": float(info["step_energy"]),
            })

        if done:
            break

    total_tasks = max(1, total_completed + total_missed)
    comp_rate = (total_completed / total_tasks) * 100.0
    tot_decisions = max(1, offload_counts["local"] + offload_counts["peer"] + offload_counts["edge"])

    arrival_steps = last_info.get("arrival_steps", {})
    arrivals = [arrival_steps.get(j) for j in range(num_uavs)]
    successful_arrivals = sum(1 for a in arrivals if a is not None)
    uav_success_rate = (successful_arrivals / num_uavs) * 100.0
    fleet_success = 100.0 if (successful_arrivals == num_uavs) else 0.0

    # Path length calculation from actual recorded step positions
    uav_nav_paths = []
    uav_total_paths = []
    final_goal_errors = []
    for j in range(num_uavs):
        p_arr = np.array(hist[j]["pos"])
        arr_step = arrival_steps.get(j)
        if arr_step is not None and len(p_arr) > 1:
            nav_p = float(np.sum(np.linalg.norm(np.diff(p_arr[:arr_step + 1], axis=0), axis=1)))
        elif len(p_arr) > 1:
            nav_p = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1)))
        else:
            nav_p = 0.0
        tot_p = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
        uav_nav_paths.append(nav_p)
        uav_total_paths.append(tot_p)
        final_goal_errors.append(float(np.linalg.norm(p_arr[-1] - env.stations[j])))

    fleet_nav_path = float(sum(uav_nav_paths))
    fleet_total_path = float(sum(uav_total_paths))
    mean_uav_nav_path = float(np.mean(uav_nav_paths))
    mean_uav_total_path = float(np.mean(uav_total_paths))
    mean_goal_error = float(np.mean(final_goal_errors))

    # Arrival time
    arrived_times = [a * cfg.physics.delta_t for a in arrivals if a is not None]
    if len(arrived_times) > 0:
        mean_arrival_time = float(np.mean(arrived_times))
        fleet_arrival_time = float(np.max(arrived_times)) if fleet_success == 100.0 else float(cfg.mappo.episode_length * cfg.physics.delta_t)
    else:
        mean_arrival_time = float(cfg.mappo.episode_length * cfg.physics.delta_t)
        fleet_arrival_time = float(cfg.mappo.episode_length * cfg.physics.delta_t)

    fleet_energy_to_goal = float(last_info.get("fleet_energy_to_goal", total_energy))
    fleet_service_energy = float(last_info.get("fleet_service_energy", 0.0))

    hist["raw_steps"] = step_trajectories
    hist["summary"] = {
        "total_energy": float(total_energy),
        "fleet_energy_to_goal": float(fleet_energy_to_goal),
        "fleet_service_energy": float(fleet_service_energy),
        "propulsion_energy": float(total_prop_energy),
        "comm_energy": float(total_comm_energy),
        "comp_energy": float(total_comp_energy),
        "completed_tasks": int(total_completed),
        "deadline_misses": int(total_missed),
        "completion_rate": float(comp_rate),
        "total_delay": float(total_delay),
        "mean_delay_per_task": float(total_delay / total_tasks),
        "energy_per_task": float(total_energy / max(1, total_completed)),
        "nav_path_length": float(fleet_nav_path),
        "total_path_length": float(fleet_total_path),
        "fleet_nav_path": float(fleet_nav_path),
        "fleet_total_path": float(fleet_total_path),
        "mean_uav_nav_path": float(mean_uav_nav_path),
        "mean_uav_total_path": float(mean_uav_total_path),
        "interventions": int(total_interventions),
        "proposed_collisions": int(total_proposed_collisions),
        "actual_collisions": int(total_actual_collisions),
        "uav_success_rate": float(uav_success_rate),
        "fleet_success": float(fleet_success),
        "mean_goal_error": float(mean_goal_error),
        "mean_arrival_time": float(mean_arrival_time),
        "fleet_arrival_time": float(fleet_arrival_time),
        "successful_uavs": int(successful_arrivals),
        "pct_local": float((offload_counts["local"] / tot_decisions) * 100.0),
        "pct_peer": float((offload_counts["peer"] / tot_decisions) * 100.0),
        "pct_edge": float((offload_counts["edge"] / tot_decisions) * 100.0)
    }
    return hist

def run_evaluation(
    test_seeds: List[int] = [101, 102, 103, 104, 105],
    device: str = "auto",
    output_dir: str = "results",
    strict_ckpts: bool = True,
    save_plots: bool = True
) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(output_dir)
    ckpt_dir = os.path.join(base_dir, "results", "checkpoints")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    if save_plots:
        os.makedirs(fig_dir, exist_ok=True)

    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, extra_config={"Mode": "Comprehensive Scientific Evaluation", "Output Dir": res_dir, "Test Seeds": str(test_seeds)})

    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    num_uavs = cfg.physics.num_uavs

    # Record Checkpoint Provenance Hashes
    ckpt_manifest = {}
    for key, fn in [
        ("mappo_pure", "best_mappo_pure_agent.pt"),
        ("mappo_residual", "best_mappo_residual_agent.pt"),
        ("ippo", "best_ippo_pure_agent.pt"),
        ("maddpg", "best_maddpg_pure_agent.pt"),
        ("cppo", "best_cppo_agent.pt"),
    ]:
        p = os.path.join(ckpt_dir, fn)
        ckpt_manifest[key] = {
            "filename": fn,
            "path": p,
            "exists": os.path.exists(p),
            "sha256": get_file_sha256(p)
        }

    # Initialize Agents
    agents = {}

    # 1. Pure MAPPO (Variant A: 100% Neural Output, No Injected APF)
    mappo_pure = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded_pure = False
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_pure_agent.pt"), os.path.join(ckpt_dir, "final_mappo_pure_agent.pt")]:
        if os.path.exists(ckpt):
            mappo_pure.load_checkpoint(ckpt)
            print(f"[Loaded Checkpoint] Pure MAPPO from {ckpt}")
            loaded_pure = True
            break
    if not loaded_pure and strict_ckpts:
        raise FileNotFoundError(f"Missing required checkpoint for Pure MAPPO in {ckpt_dir}!")
    agents["mappo_pure"] = mappo_pure
    agents["mappo"] = mappo_pure  # Backward-compatible alias

    # 2. APF-Assisted Residual MAPPO (Variant B: Nominal APF 2.0 + Learned Policy 0.4)
    mappo_residual = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="residual", device=dev)
    loaded_residual = False
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_residual_agent.pt"), os.path.join(ckpt_dir, "final_mappo_residual_agent.pt")]:
        if os.path.exists(ckpt):
            mappo_residual.load_checkpoint(ckpt)
            print(f"[Loaded Checkpoint] Residual MAPPO from {ckpt}")
            loaded_residual = True
            break
    if not loaded_residual and strict_ckpts:
        raise FileNotFoundError(f"Missing required checkpoint for Residual MAPPO in {ckpt_dir}!")
    agents["mappo_residual"] = mappo_residual

    # 2b. Undamped Residual MAPPO (Historical comparison ablation demonstrating post-arrival limit cycles)
    mappo_residual_undamped = MAPPOAgent(
        env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim,
        policy_variant="residual", enable_terminal_damping=False, device=dev
    )
    if loaded_residual:
        mappo_residual_undamped.load_checkpoint(os.path.join(ckpt_dir, "best_mappo_residual_agent.pt"))
    agents["mappo_residual_undamped"] = mappo_residual_undamped

    # 3. APF-Only Controller (Variant C: Pure APF Guidance Alone, Zero Learned Contribution)
    mappo_apf_only = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="apf_only", device=dev)
    if loaded_residual:
        # Load weights into apf_only agent (policy_variant is preserved as apf_only)
        mappo_apf_only.load_checkpoint(os.path.join(ckpt_dir, "best_mappo_residual_agent.pt"))
        print("[Initialized] APF-Only Guidance Controller (Gain 2.0, Zero Learned Policy)")
    agents["mappo_apf_only"] = mappo_apf_only

    # 4. Independent PPO (IPPO, DTDE)
    ippo = IPPOAgent(env.obs_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded_ippo = False
    for ckpt in [os.path.join(ckpt_dir, "best_ippo_pure_agent.pt"), os.path.join(ckpt_dir, "final_ippo_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_ippo_agent.pt")]:
        if os.path.exists(ckpt):
            ippo.load_checkpoint(ckpt)
            print(f"[Loaded Checkpoint] IPPO from {ckpt}")
            loaded_ippo = True
            break
    if not loaded_ippo and strict_ckpts:
        raise FileNotFoundError(f"Missing required checkpoint for IPPO in {ckpt_dir}!")
    agents["ippo"] = ippo

    # 5. Multi-Agent DDPG (MADDPG)
    maddpg = MADDPGAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    loaded_maddpg = False
    for ckpt in [os.path.join(ckpt_dir, "best_maddpg_pure_agent.pt"), os.path.join(ckpt_dir, "final_maddpg_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_maddpg_agent.pt")]:
        if os.path.exists(ckpt):
            maddpg.load_checkpoint(ckpt)
            print(f"[Loaded Checkpoint] MADDPG from {ckpt}")
            loaded_maddpg = True
            break
    if not loaded_maddpg and strict_ckpts:
        raise FileNotFoundError(f"Missing required checkpoint for MADDPG in {ckpt_dir}!")
    agents["maddpg"] = maddpg

    # 6. Centralized PPO (CPPO)
    cppo = CPPOAgent(env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, device=dev)
    loaded_cppo = False
    for ckpt in [os.path.join(ckpt_dir, "best_cppo_agent.pt"), os.path.join(ckpt_dir, "final_cppo_agent.pt")]:
        if os.path.exists(ckpt):
            cppo.load_checkpoint(ckpt)
            print(f"[Loaded Checkpoint] CPPO from {ckpt}")
            loaded_cppo = True
            break
    if not loaded_cppo and strict_ckpts:
        raise FileNotFoundError(f"Missing required checkpoint for CPPO in {ckpt_dir}!")
    agents["cppo"] = cppo

    # 7. Classical & Heuristic Baselines and Matched Ablations
    agents["astar_edge"] = AStarJointBaseline(env, EdgeOnlyPolicy())
    agents["astar_local"] = AStarJointBaseline(env, LocalOnlyPolicy())
    agents["astar_random"] = AStarJointBaseline(env, RandomFeasiblePolicy())
    agents["astar_greedy"] = AStarJointBaseline(env, EnergyDeadlineAwareGreedyPolicy(queue_aware=True))
    agents["astar_latency"] = AStarJointBaseline(env, LowestLatencyPolicy(queue_aware=True))
    agents["astar_pure_pursuit"] = HybridAStarMAPPONavigator(env, mappo_agent=None, residual_weight=0.0)
    agents["hybrid_astar_mappo_zero"] = HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.0)
    agents["hybrid_astar_mappo_res02"] = HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.2, apply_terminal_damping=True)
    agents["hybrid_astar_mappo_res05"] = HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.5, apply_terminal_damping=True)
    agents["hybrid_astar_mappo"] = agents["hybrid_astar_mappo_zero"]
    agents["local_only"] = agents["astar_local"]
    agents["edge_only"] = agents["astar_edge"]
    agents["random"] = agents["astar_random"]
    agents["astar"] = agents["astar_edge"]

    # Model evaluation suite
    methods_meta = [
        ("3D A* + Edge MEC (Primary Classical Benchmark)", "astar_edge"),
        ("3D A* + Pure Pursuit (Zero RL Baseline)", "astar_pure_pursuit"),
        ("Hierarchical 3D A* + Pure Pursuit + MAPPO (Offload Only, k_rl=0.0)", "hybrid_astar_mappo_zero"),
        ("Hierarchical 3D A* + Pure Pursuit + MAPPO (Causal Residual k_rl=0.2)", "hybrid_astar_mappo_res02"),
        ("Hierarchical 3D A* + Pure Pursuit + MAPPO (Causal Residual k_rl=0.5)", "hybrid_astar_mappo_res05"),
        ("APF-Only Controller (Gain 2.0, Zero Learning)", "mappo_apf_only"),
        ("APF-Assisted Residual MAPPO (with Terminal Damping)", "mappo_residual"),
        ("APF-Assisted Residual MAPPO (Undamped / Historical Orbiting)", "mappo_residual_undamped"),
        ("Proposed Pure MAPPO (8.4k updates, Zero APF Guidance)", "mappo_pure"),
        ("Independent PPO (IPPO, DTDE)", "ippo"),
        ("Multi-Agent DDPG (MADDPG)", "maddpg"),
        ("Centralized PPO (CPPO, CTCE)", "cppo"),
    ]

    print(f"\nRunning standardized multi-seed evaluation across {len(test_seeds)} seeds: {test_seeds}")
    detailed_records = []
    first_seed_results = {}
    aggregated_metrics = {}
    all_raw_trajectories = []

    for label, key in methods_meta:
        agent = agents[key]
        seed_stats = []

        for s_idx, s in enumerate(test_seeds):
            ep_hist = evaluate_episode(env, key, agent, seed=s)
            s_sum = ep_hist["summary"]
            s_sum["method"] = label
            s_sum["key"] = key
            s_sum["seed"] = s
            seed_stats.append(s_sum)
            detailed_records.append(s_sum)

            for step_row in ep_hist["raw_steps"]:
                step_row["method"] = label
                step_row["key"] = key
                step_row["seed"] = s
                all_raw_trajectories.append(step_row)

            if s_idx == 0:
                first_seed_results[key] = ep_hist

        df_seeds = pd.DataFrame(seed_stats)
        aggregated_metrics[key] = {
            "Method": label,
            "UAV Success (%)": f"{df_seeds['uav_success_rate'].mean():.1f} ± {df_seeds['uav_success_rate'].std():.1f}",
            "Fleet Success (%)": f"{df_seeds['fleet_success'].mean():.1f} ± {df_seeds['fleet_success'].std():.1f}",
            "Arrival (s)": f"{df_seeds['mean_arrival_time'].mean():.1f} ± {df_seeds['mean_arrival_time'].std():.1f}",
            "Goal Error (m)": f"{df_seeds['mean_goal_error'].mean():.2f} ± {df_seeds['mean_goal_error'].std():.2f}",
            "Nav Energy (kJ)": f"{df_seeds['fleet_energy_to_goal'].mean()/1000.0:.1f} ± {df_seeds['fleet_energy_to_goal'].std()/1000.0:.1f}",
            "Service Energy (kJ)": f"{df_seeds['fleet_service_energy'].mean()/1000.0:.1f} ± {df_seeds['fleet_service_energy'].std()/1000.0:.1f}",
            "Total Energy (kJ)": f"{df_seeds['total_energy'].mean()/1000.0:.1f} ± {df_seeds['total_energy'].std()/1000.0:.1f}",
            "UAV Nav Path (m)": f"{df_seeds['mean_uav_nav_path'].mean():.1f} ± {df_seeds['mean_uav_nav_path'].std():.1f}",
            "Fleet Nav Path (m)": f"{df_seeds['fleet_nav_path'].mean():.1f} ± {df_seeds['fleet_nav_path'].std():.1f}",
            "UAV Total Path (m)": f"{df_seeds['mean_uav_total_path'].mean():.1f} ± {df_seeds['mean_uav_total_path'].std():.1f}",
            "Fleet Total Path (m)": f"{df_seeds['fleet_total_path'].mean():.1f} ± {df_seeds['fleet_total_path'].std():.1f}",
            "Interventions": f"{df_seeds['interventions'].mean():.1f} ± {df_seeds['interventions'].std():.1f}",
            "Collisions": f"{df_seeds['actual_collisions'].mean():.1f} ± {df_seeds['actual_collisions'].std():.1f}",
            "Tasks Done (%)": f"{df_seeds['completion_rate'].mean():.1f} ± {df_seeds['completion_rate'].std():.1f}",
            "Delay (s)": f"{df_seeds['total_delay'].mean():.1f} ± {df_seeds['total_delay'].std():.1f}",
            "Raw_Total_Energy_J": float(df_seeds['total_energy'].mean()),
            "Raw_Nav_Energy_J": float(df_seeds['fleet_energy_to_goal'].mean()),
            "Raw_Total_Delay_s": float(df_seeds['total_delay'].mean()),
            "Pct_Local": float(df_seeds['pct_local'].mean()),
            "Pct_Peer": float(df_seeds['pct_peer'].mean()),
            "Pct_Edge": float(df_seeds['pct_edge'].mean()),
        }

    summary_rows = []
    offload_rows = []
    pareto_dict = {}

    for label, key in methods_meta:
        m = aggregated_metrics[key]
        summary_rows.append({
            "Method": m["Method"],
            "UAV Success (%)": m["UAV Success (%)"],
            "Fleet Success (%)": m["Fleet Success (%)"],
            "Arrival (s)": m["Arrival (s)"],
            "Goal Error (m)": m["Goal Error (m)"],
            "Nav Energy (kJ)": m["Nav Energy (kJ)"],
            "Service Energy (kJ)": m["Service Energy (kJ)"],
            "Total Energy (kJ)": m["Total Energy (kJ)"],
            "UAV Nav Path (m)": m["UAV Nav Path (m)"],
            "Fleet Nav Path (m)": m["Fleet Nav Path (m)"],
            "UAV Total Path (m)": m["UAV Total Path (m)"],
            "Fleet Total Path (m)": m["Fleet Total Path (m)"],
            "Interventions": m["Interventions"],
            "Collisions": m["Collisions"],
            "Tasks Done (%)": m["Tasks Done (%)"],
            "Delay (s)": m["Delay (s)"]
        })
        offload_rows.append({
            "Method": label,
            "Local (%)": m["Pct_Local"],
            "Peer (%)": m["Pct_Peer"],
            "Edge (%)": m["Pct_Edge"]
        })
        pareto_dict[key] = {
            "latency": m["Raw_Total_Delay_s"],
            "energy": m["Raw_Total_Energy_J"],
            "label": label.split("(")[0].strip()
        }

    df_summary = pd.DataFrame(summary_rows)
    df_offload = pd.DataFrame(offload_rows)
    df_detailed = pd.DataFrame(detailed_records)
    df_raw_traj = pd.DataFrame(all_raw_trajectories)

    print("\n" + "=" * 115)
    print("STANDARDIZED MULTI-SEED SCIENTIFIC BENCHMARK SUMMARY TABLE:")
    print("=" * 115)
    print(df_summary.to_string(index=False))
    print("=" * 115)

    summary_csv = os.path.join(res_dir, "benchmark_summary.csv")
    offload_csv = os.path.join(res_dir, "offloading_distribution.csv")
    detailed_csv = os.path.join(res_dir, "benchmark_detailed.csv")
    raw_traj_csv = os.path.join(res_dir, "raw_evaluation_trajectories.csv")
    manifest_json = os.path.join(res_dir, "evaluation_manifest.json")

    df_summary.to_csv(summary_csv, index=False)
    df_offload.to_csv(offload_csv, index=False)
    df_detailed.to_csv(detailed_csv, index=False)
    per_seed_csv = os.path.join(res_dir, "benchmark_per_seed.csv")
    df_detailed.to_csv(per_seed_csv, index=False)
    df_raw_traj.to_csv(raw_traj_csv, index=False)
    raw_traj_dir = os.path.join(res_dir, "raw_trajectories")
    if os.path.exists(raw_traj_dir):
        df_raw_traj.to_csv(os.path.join(raw_traj_dir, "raw_evaluation_trajectories.csv"), index=False)

    # Save complete evaluation manifest
    git_commit = "UNKNOWN"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir).decode("utf-8").strip()
    except Exception:
        pass

    manifest_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "test_seeds": test_seeds,
        "output_dir": res_dir,
        "checkpoints": ckpt_manifest,
        "methods_evaluated": [m[0] for m in methods_meta],
        "metrics_summary": df_summary.to_dict(orient="records"),
        "airspace": {
            "bounds": [cfg.airspace.x_min, cfg.airspace.x_max, cfg.airspace.y_min, cfg.airspace.y_max, cfg.airspace.z_min, cfg.airspace.z_max],
            "obstacles": cfg.airspace.obstacles,
            "edge_server": cfg.airspace.edge_server_pos
        },
        "physics": {
            "reach_tolerance": cfg.physics.reach_tolerance,
            "d_safe": cfg.physics.d_safe,
            "obstacle_margin": cfg.physics.obstacle_margin,
            "v_max": cfg.physics.v_max,
            "delta_t": cfg.physics.delta_t,
            "episode_length": cfg.mappo.episode_length
        }
    }
    with open(manifest_json, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"\n[Saved] Summary table: {summary_csv}")
    print(f"[Saved] Offload breakdown: {offload_csv}")
    print(f"[Saved] Detailed records: {detailed_csv}")
    print(f"[Saved] Raw step trajectories: {raw_traj_csv}")
    print(f"[Saved] Evaluation manifest: {manifest_json}")

    # Generate Publication Figures if requested
    if save_plots:
        try:
            diag_path = os.path.join(fig_dir, "trajectory_diagnostic_comparison.png")
            plot_trajectory_diagnostic(
                astar_data=first_seed_results["astar_edge"],
                mappo_data=first_seed_results.get("mappo_residual", first_seed_results["mappo_pure"]),
                obstacles=cfg.airspace.obstacles,
                stations=env.stations,
                save_path=diag_path
            )

            multi_curve_path = os.path.join(fig_dir, "multimodel_performance_curves.png")
            plot_multimodel_curves(
                results={
                    "mappo": first_seed_results.get("mappo_residual", first_seed_results["mappo_pure"]),
                    "ippo": first_seed_results["ippo"],
                    "maddpg": first_seed_results["maddpg"],
                    "astar": first_seed_results["astar_edge"]
                },
                obstacles=cfg.airspace.obstacles,
                stations=env.stations,
                save_path=multi_curve_path
            )

            # Figure 6: Offloading Distribution
            offload_dict = {row["Method"]: {"local": row["Local (%)"], "peer": row["Peer (%)"], "edge": row["Edge (%)"]}
                            for _, row in df_offload.iterrows()}
            plot_offload_distribution_stacked(offload_dict, os.path.join(fig_dir, "offloading_distribution_breakdown.png"))

            # Figure 7: Pareto Trade-off
            plot_pareto_latency_energy(pareto_dict, os.path.join(fig_dir, "pareto_latency_energy_tradeoff.png"))

            # Figure 9: 3D Trajectories
            plot_3d_flight_trajectories(
                eval_results={"mappo": first_seed_results.get("mappo_residual", first_seed_results["mappo_pure"]), "astar": first_seed_results["astar_edge"]},
                obstacles=cfg.airspace.obstacles,
                stations=env.stations,
                save_path=os.path.join(fig_dir, "3d_flight_trajectories.png")
            )
        except Exception as e:
            print(f"[Warning] Figure generation encountered an issue: {e}")

    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Scientific Evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of held-out test seeds")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save audit evaluation results")
    parser.add_argument("--strict_ckpts", action="store_true", default=True, help="Enforce strict checkpoint presence")
    parser.add_argument("--no_strict", dest="strict_ckpts", action="store_false", help="Permit missing checkpoints")
    parser.add_argument("--no_plots", dest="save_plots", action="store_false", default=True, help="Disable plot generation")
    args = parser.parse_args()

    seeds = [101 + i for i in range(args.num_seeds)]
    run_evaluation(test_seeds=seeds, device=args.device, output_dir=args.output_dir, strict_ckpts=args.strict_ckpts, save_plots=args.save_plots)
