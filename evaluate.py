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
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from utils.device import DeviceManager
from models.mappo import MAPPOAgent
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent
from models.astar import AStarBaseline
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

    for t in range(cfg.mappo.episode_length):
        for j in range(num_uavs):
            pos = np.copy(env.positions[j])
            spd = float(np.linalg.norm(env.velocities[j]))
            d_g = float(np.linalg.norm(pos - env.stations[j]))
            pwr = float(env.aero.power(spd))
            hist[j]["pos"].append(pos)
            hist[j]["speed"].append(spd)
            hist[j]["dist_goal"].append(d_g)
            hist[j]["power"].append(pwr)

        if model_key in ["astar", "astar_greedy", "astar_latency", "local_only", "edge_only", "random"]:
            actions = agent.select_actions(obs)
        elif model_key == "cppo":
            actions, _, _, _ = agent.get_actions(state, deterministic=True)
        elif model_key in ["mappo", "mappo_pure", "mappo_residual", "ippo"]:
            actions, _, _, _ = agent.get_actions(obs, deterministic=True)
        elif model_key == "maddpg":
            actions, _ = agent.get_actions(obs, explore=False)
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

        if done:
            break

    total_tasks = max(1, total_completed + total_missed)
    comp_rate = (total_completed / total_tasks) * 100.0
    total_path = sum([np.sum(np.linalg.norm(np.diff(np.array(hist[j]["pos"]), axis=0), axis=1)) for j in range(num_uavs)])
    tot_decisions = max(1, offload_counts["local"] + offload_counts["peer"] + offload_counts["edge"])

    fleet_energy_to_goal = float(last_info.get("fleet_energy_to_goal", total_energy))
    fleet_service_energy = float(last_info.get("fleet_service_energy", 0.0))
    fleet_path_to_goal = float(last_info.get("fleet_path_length_to_goal", total_path))
    arrival_steps = last_info.get("arrival_steps", {})
    arrivals = [arrival_steps.get(j) for j in range(num_uavs)]
    successful_arrivals = sum(1 for a in arrivals if a is not None)
    mission_success_rate = (successful_arrivals / num_uavs) * 100.0
    mean_arrival_time = float(np.mean([a * cfg.physics.delta_t if a is not None else cfg.mappo.episode_length * cfg.physics.delta_t for a in arrivals]))

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
        "total_path_length": float(total_path),
        "fleet_path_to_goal": float(fleet_path_to_goal),
        "interventions": int(total_interventions),
        "proposed_collisions": int(total_proposed_collisions),
        "actual_collisions": int(total_actual_collisions),
        "mission_success_rate": float(mission_success_rate),
        "mean_arrival_time": float(mean_arrival_time),
        "pct_local": float((offload_counts["local"] / tot_decisions) * 100.0),
        "pct_peer": float((offload_counts["peer"] / tot_decisions) * 100.0),
        "pct_edge": float((offload_counts["edge"] / tot_decisions) * 100.0)
    }
    return hist

def run_evaluation(test_seeds: List[int] = [101, 102, 103, 104, 105], device: str = "auto") -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, extra_config={"Mode": "Comprehensive Benchmark Evaluation", "Test Seeds": str(test_seeds)})

    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    num_uavs = cfg.physics.num_uavs

    # Initialize Agents
    agents = {}

    # 1. Pure MAPPO (Variant A: No hardcoded APF in actor)
    mappo_pure = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_pure_agent.pt"), os.path.join(ckpt_dir, "final_mappo_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_mappo_agent.pt"), os.path.join(ckpt_dir, "final_mappo_agent.pt")]:
        if os.path.exists(ckpt):
            mappo_pure.load_checkpoint(ckpt)
            print(f"[Loaded] Pure MAPPO from {ckpt}")
            break
    agents["mappo_pure"] = mappo_pure
    agents["mappo"] = mappo_pure  # Backward-compatible alias

    # 2. APF-Assisted Residual MAPPO (Variant B: Nominal APF + learned residual)
    mappo_residual = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="residual", device=dev)
    for ckpt in [os.path.join(ckpt_dir, "best_mappo_residual_agent.pt"), os.path.join(ckpt_dir, "final_mappo_residual_agent.pt")]:
        if os.path.exists(ckpt):
            mappo_residual.load_checkpoint(ckpt)
            print(f"[Loaded] Residual MAPPO from {ckpt}")
            break
    agents["mappo_residual"] = mappo_residual

    # 3. IPPO
    ippo = IPPOAgent(env.obs_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    for ckpt in [os.path.join(ckpt_dir, "best_ippo_pure_agent.pt"), os.path.join(ckpt_dir, "final_ippo_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_ippo_agent.pt"), os.path.join(ckpt_dir, "final_ippo_agent.pt")]:
        if os.path.exists(ckpt):
            ippo.load_checkpoint(ckpt)
            print(f"[Loaded] IPPO from {ckpt}")
            break
    agents["ippo"] = ippo

    # 4. MADDPG
    maddpg = MADDPGAgent(env.obs_dim, env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    for ckpt in [os.path.join(ckpt_dir, "best_maddpg_pure_agent.pt"), os.path.join(ckpt_dir, "final_maddpg_pure_agent.pt"),
                 os.path.join(ckpt_dir, "best_maddpg_agent.pt"), os.path.join(ckpt_dir, "final_maddpg_agent.pt")]:
        if os.path.exists(ckpt):
            maddpg.load_checkpoint(ckpt)
            print(f"[Loaded] MADDPG from {ckpt}")
            break
    agents["maddpg"] = maddpg

    # 5. CPPO
    cppo = CPPOAgent(env.state_dim, num_uavs, num_uavs + 1, cfg.mappo.hidden_dim, device=dev)
    for ckpt in [os.path.join(ckpt_dir, "best_cppo_agent.pt"), os.path.join(ckpt_dir, "final_cppo_agent.pt")]:
        if os.path.exists(ckpt):
            cppo.load_checkpoint(ckpt)
            print(f"[Loaded] CPPO from {ckpt}")
            break
    agents["cppo"] = cppo

    # 6. Classical & Heuristic Baselines
    agents["astar_greedy"] = AStarJointBaseline(env, EnergyDeadlineAwareGreedyPolicy())
    agents["astar_latency"] = AStarJointBaseline(env, LowestLatencyPolicy())
    agents["local_only"] = AStarJointBaseline(env, LocalOnlyPolicy())
    agents["edge_only"] = AStarJointBaseline(env, EdgeOnlyPolicy())
    agents["random"] = AStarJointBaseline(env, RandomFeasiblePolicy())
    agents["astar"] = agents["astar_greedy"]  # Legacy alias

    # Model metadata for clean reporting
    methods_meta = [
        ("3D A* + Energy-Deadline MEC (Classical Benchmark)", "astar_greedy"),
        ("3D A* + Lowest-Latency MEC", "astar_latency"),
        ("3D A* + Local-Only Execution", "local_only"),
        ("3D A* + Edge-Only Execution", "edge_only"),
        ("3D A* + Random Feasible MEC", "random"),
        ("Centralized PPO (CPPO, CTCE)", "cppo"),
        ("Independent PPO (IPPO, DTDE)", "ippo"),
        ("Multi-Agent DDPG (MADDPG)", "maddpg"),
        ("Proposed Pure MAPPO (No APF Guidance)", "mappo_pure"),
        ("APF-Assisted Residual MAPPO (Nominal+Residual)", "mappo_residual")
    ]

    print("\nRunning multi-seed held-out evaluations across", len(test_seeds), "test scenarios...")
    detailed_records = []
    first_seed_results = {}
    aggregated_metrics = {}

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

            if s_idx == 0:
                first_seed_results[key] = ep_hist

        # Compute mean and standard deviation across seeds
        df_seeds = pd.DataFrame(seed_stats)
        aggregated_metrics[key] = {
            "Method": label,
            "Success Rate (%)": f"{df_seeds['mission_success_rate'].mean():.1f} ± {df_seeds['mission_success_rate'].std():.1f}",
            "Arrival Time (s)": f"{df_seeds['mean_arrival_time'].mean():.1f} ± {df_seeds['mean_arrival_time'].std():.1f}",
            "Nav Energy (kJ)": f"{df_seeds['fleet_energy_to_goal'].mean()/1000.0:.1f} ± {df_seeds['fleet_energy_to_goal'].std()/1000.0:.1f}",
            "Service Energy (kJ)": f"{df_seeds['fleet_service_energy'].mean()/1000.0:.1f} ± {df_seeds['fleet_service_energy'].std()/1000.0:.1f}",
            "Total Energy (kJ)": f"{df_seeds['total_energy'].mean()/1000.0:.1f} ± {df_seeds['total_energy'].std()/1000.0:.1f}",
            "Completion Rate (%)": f"{df_seeds['completion_rate'].mean():.1f} ± {df_seeds['completion_rate'].std():.1f}",
            "Total Delay (s)": f"{df_seeds['total_delay'].mean():.1f} ± {df_seeds['total_delay'].std():.1f}",
            "Energy / Task (J)": f"{df_seeds['energy_per_task'].mean():.1f} ± {df_seeds['energy_per_task'].std():.1f}",
            "Path Length (m)": f"{df_seeds['total_path_length'].mean():.1f} ± {df_seeds['total_path_length'].std():.1f}",
            "Safety Interventions": f"{df_seeds['interventions'].mean():.1f} ± {df_seeds['interventions'].std():.1f}",
            "Actual Collisions": f"{df_seeds['actual_collisions'].mean():.1f} ± {df_seeds['actual_collisions'].std():.1f}",
            "Raw_Total_Energy_J": float(df_seeds['total_energy'].mean()),
            "Raw_Nav_Energy_J": float(df_seeds['fleet_energy_to_goal'].mean()),
            "Raw_Total_Delay_s": float(df_seeds['total_delay'].mean()),
            "Pct_Local": float(df_seeds['pct_local'].mean()),
            "Pct_Peer": float(df_seeds['pct_peer'].mean()),
            "Pct_Edge": float(df_seeds['pct_edge'].mean()),
        }

    # Format Benchmark Summary Table
    summary_rows = []
    offload_rows = []
    pareto_dict = {}

    for label, key in methods_meta:
        m = aggregated_metrics[key]
        summary_rows.append({
            "Method": m["Method"],
            "Success (%)": m["Success Rate (%)"],
            "Arrival (s)": m["Arrival Time (s)"],
            "Nav Energy (kJ)": m["Nav Energy (kJ)"],
            "Service Energy (kJ)": m["Service Energy (kJ)"],
            "Total Energy (kJ)": m["Total Energy (kJ)"],
            "Tasks Done (%)": m["Completion Rate (%)"],
            "Delay (s)": m["Total Delay (s)"],
            "Path Length (m)": m["Path Length (m)"],
            "Interventions": m["Safety Interventions"],
            "Collisions": m["Actual Collisions"]
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

    print("\n" + "=" * 90)
    print("FINAL MULTI-SEED SCIENTIFIC BENCHMARK SUMMARY TABLE:")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    summary_csv = os.path.join(res_dir, "benchmark_summary.csv")
    offload_csv = os.path.join(res_dir, "offloading_distribution.csv")
    detailed_csv = os.path.join(res_dir, "benchmark_detailed.csv")

    df_summary.to_csv(summary_csv, index=False)
    df_offload.to_csv(offload_csv, index=False)
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\n[Saved] Summary table: {summary_csv}")
    print(f"[Saved] Offload breakdown: {offload_csv}")
    print(f"[Saved] Detailed records: {detailed_csv}")

    # Generate Publication Figures
    if "mappo" not in first_seed_results:
        first_seed_results["mappo"] = first_seed_results.get("mappo_residual", first_seed_results.get("mappo_pure"))

    diag_path = os.path.join(fig_dir, "trajectory_diagnostic_comparison.png")
    plot_trajectory_diagnostic(
        astar_data=first_seed_results["astar_greedy"],
        mappo_data=first_seed_results["mappo"],
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=diag_path
    )

    multi_curve_path = os.path.join(fig_dir, "multimodel_performance_curves.png")
    plot_multimodel_curves(
        results={
            "mappo": first_seed_results["mappo"],
            "ippo": first_seed_results["ippo"],
            "maddpg": first_seed_results["maddpg"],
            "astar": first_seed_results["astar_greedy"]
        },
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=multi_curve_path
    )

    plot_mappo_vs_astar_deepdive(
        mappo_data=first_seed_results["mappo"],
        astar_data=first_seed_results["astar_greedy"],
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=os.path.join(fig_dir, "mappo_vs_astar_diagnostic.png")
    )

    plot_mappo_vs_ippo_deepdive(
        mappo_data=first_seed_results["mappo"],
        ippo_data=first_seed_results["ippo"],
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=os.path.join(fig_dir, "mappo_vs_ippo_ablation.png")
    )

    plot_mappo_vs_maddpg_deepdive(
        mappo_data=first_seed_results["mappo"],
        maddpg_data=first_seed_results["maddpg"],
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=os.path.join(fig_dir, "mappo_vs_maddpg_comparison.png")
    )

    # Figure 6: Offloading Distribution
    offload_dict = {row["Method"]: {"local": row["Local (%)"], "peer": row["Peer (%)"], "edge": row["Edge (%)"]}
                    for _, row in df_offload.iterrows()}
    plot_offload_distribution_stacked(offload_dict, os.path.join(fig_dir, "offloading_distribution_breakdown.png"))

    # Figure 7: Pareto Trade-off
    plot_pareto_latency_energy(pareto_dict, os.path.join(fig_dir, "pareto_latency_energy_tradeoff.png"))

    # Figure 9: 3D Trajectories
    plot_3d_flight_trajectories(
        eval_results={"mappo": first_seed_results["mappo"], "astar": first_seed_results["astar_greedy"]},
        obstacles=cfg.airspace.obstacles,
        stations=env.stations,
        save_path=os.path.join(fig_dir, "3d_flight_trajectories.png")
    )

    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Scientific Evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of held-out test seeds")
    args = parser.parse_args()

    seeds = [101 + i for i in range(args.num_seeds)]
    run_evaluation(test_seeds=seeds, device=args.device)
