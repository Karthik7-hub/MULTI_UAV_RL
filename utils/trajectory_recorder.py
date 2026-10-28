"""
Deterministic Trajectory Recorder for Metropolis 3D Airspace Scenario.
Provides reproducible flight trajectory recording for publication figure derivation
and automated test validation without relying on ephemeral scratch scripts.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import get_preset_config
from env.multi_uav_env import MultiUAVEnv
from env.safety_filter import compute_min_clearance
from utils.device import DeviceManager
from models.astar import AStarBaseline, HybridAStarMAPPONavigator
from models.rrt_star import RRTStarBaseline
from models.apf import SAPFBaseline
from models.mappo import MAPPOAgent

def record_metropolis_trajectories(
    output_csv: Optional[str] = None,
    output_summary: Optional[str] = None,
    seed: int = 101,
    max_steps: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes reproducible physical trajectories across baseline and learning planners
    under Metropolis airspace geometry and records raw timestep coordinates.
    """
    dev = DeviceManager.select_device("cpu")
    cfg = get_preset_config(preset="metropolis", num_uavs=4)
    if max_steps is not None:
        cfg.mappo.episode_length = max_steps

    env = MultiUAVEnv(cfg)
    obstacles = cfg.airspace.obstacles
    stations = env.stations
    ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")

    # Initialize RL agents
    mappo_residual = MAPPOAgent(env.obs_dim, env.state_dim, 4, 5, cfg.mappo.hidden_dim, policy_variant="residual", device=dev)
    res_ckpt = os.path.join(ckpt_dir, "best_mappo_residual_agent.pt")
    if os.path.exists(res_ckpt):
        try:
            mappo_residual.load_checkpoint(res_ckpt)
        except Exception as e:
            print(f"[Notice] Could not load residual checkpoint: {e}")

    mappo_pure = MAPPOAgent(env.obs_dim, env.state_dim, 4, 5, cfg.mappo.hidden_dim, policy_variant="pure", device=dev)
    pure_ckpt = os.path.join(ckpt_dir, "best_mappo_pure_agent.pt")
    if os.path.exists(pure_ckpt):
        try:
            mappo_pure.load_checkpoint(pure_ckpt)
        except Exception as e:
            print(f"[Notice] Could not load pure checkpoint: {e}")

    agents = {
        "hybrid_astar_mappo": HybridAStarMAPPONavigator(env, mappo_agent=mappo_residual, residual_weight=0.0),
        "astar": AStarBaseline(env),
        "sapf": SAPFBaseline(env),
        "rrt_star": RRTStarBaseline(env),
        "mappo_residual": mappo_residual,
        "mappo_pure": mappo_pure,
    }

    labels = {
        "hybrid_astar_mappo": "Hierarchical A* + MAPPO (Hybrid, w_res=0.0)",
        "astar": "3D A* Baseline",
        "sapf": "3D SAPF (Adaptive Potential Field)",
        "rrt_star": "3D RRT* (Optimal Tree)",
        "mappo_residual": "APF-Assisted Residual MAPPO",
        "mappo_pure": "Proposed Pure MAPPO (No Prior)"
    }

    trajectory_records = []
    summary_records = []

    for key, agent in agents.items():
        label = labels[key]
        obs, state = env.reset(seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()

        hist_pos = {j: [np.copy(env.positions[j])] for j in range(4)}
        hist_clearance = {j: [] for j in range(4)}
        arrival_steps = {j: None for j in range(4)}
        cum_energy = 0.0
        total_interventions = 0

        # Record t=0
        for j in range(4):
            p = np.copy(env.positions[j])
            d_g = float(np.linalg.norm(p - stations[j]))
            min_c = compute_min_clearance(p, obstacles, margin=2.0)
            hist_clearance[j].append(min_c)
            trajectory_records.append({
                "scenario": "metropolis",
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

        t_start = time.perf_counter()
        for t in range(1, cfg.mappo.episode_length + 1):
            if key in ["hybrid_astar_mappo", "astar", "sapf", "rrt_star"]:
                actions = agent.select_actions(obs)
            elif key in ["mappo_pure", "mappo_residual"]:
                actions, _, _, _ = agent.get_actions(obs, deterministic=True)

            obs, state, _, done, info = env.step(actions)
            cum_energy += info["step_energy"]
            total_interventions += info.get("safety_interventions", 0)

            for j in range(4):
                p = np.copy(env.positions[j])
                spd = float(np.linalg.norm(env.velocities[j]))
                d_g = float(np.linalg.norm(p - stations[j]))
                min_c = compute_min_clearance(p, obstacles, margin=2.0)
                hist_pos[j].append(p)
                hist_clearance[j].append(min_c)

                if d_g <= cfg.physics.reach_tolerance and arrival_steps[j] is None:
                    arrival_steps[j] = t

                trajectory_records.append({
                    "scenario": "metropolis",
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

        plan_time_ms = (time.perf_counter() - t_start) * 1000.0 / 4.0
        path_lens = [
            float(info["path_length_to_goal"][j]) if (arrival_steps[j] is not None and "path_length_to_goal" in info)
            else float(np.sum(np.linalg.norm(np.diff(np.array(hist_pos[j][:arrival_steps[j]+1 if arrival_steps[j] is not None else None]), axis=0), axis=1)))
            for j in range(4)
        ]
        mean_path = float(np.mean(path_lens))
        all_clear = [min(hist_clearance[j]) for j in range(4)]
        min_clear = float(min(all_clear))
        arrivals = [arrival_steps[j] if arrival_steps[j] is not None else cfg.mappo.episode_length for j in range(4)]
        mean_arr = float(np.mean(arrivals))
        success_pct = (sum(1 for a in arrival_steps.values() if a is not None) / 4.0) * 100.0

        summary_records.append({
            "algorithm": key,
            "label": label,
            "mean_path_m": round(mean_path, 2),
            "mean_arrival_s": round(mean_arr, 1),
            "min_clearance_m": round(min_clear, 2),
            "safety_interventions": total_interventions,
            "success_rate_pct": round(success_pct, 1),
            "planning_time_ms": round(plan_time_ms, 2)
        })

    out_csv = output_csv or os.path.join(BASE_DIR, "results", "metropolis_trajectory_records.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_traj = pd.DataFrame(trajectory_records)
    df_traj.to_csv(out_csv, index=False)

    out_sum = output_summary or os.path.join(BASE_DIR, "results", "metropolis_evaluation_summary.csv")
    os.makedirs(os.path.dirname(out_sum), exist_ok=True)
    df_sum = pd.DataFrame(summary_records)
    df_sum.to_csv(out_sum, index=False)

    return df_traj, df_sum

if __name__ == "__main__":
    record_metropolis_trajectories(seed=101)
