"""
Unified 3D UAV Flight & Trajectory Simulation Engine.
Encapsulates all path planning, trajectory tracking, kinematic safety filtering,
and live telemetry generation for:
- Classical Search & Sampling (3D A*, 3D RRT*, 3D SAPF)
- Hierarchical Hybrid Planning (A* + MAPPO)
- Multi-Agent Deep Reinforcement Learning (MAPPO, IPPO, MADDPG, CPPO — Pure & Residual)
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import MasterConfig
from env.safety_filter import SafetyFilter
from env.observation_builder import build_single_uav_obs
from models.astar import AStar3DPlanner, PurePursuit3DController, HybridAStarMAPPONavigator
from models.rrt_star import RRTStar3DPlanner, RRTStarBaseline
from models.apf import SAPFNavigator, SAPFBaseline
from models.baselines import AStarJointBaseline
from models.mappo import MAPPOAgent
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent
from utils.device import DeviceManager

class FlightEngine:
    def __init__(self, cfg: Optional[MasterConfig] = None, device_name: str = "cpu"):
        self.cfg = cfg or MasterConfig()
        self.dev = DeviceManager.select_device(device_name)
        self.ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")

        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self._load_all_checkpoints()

    def _load_all_checkpoints(self):
        """Loads and initializes all 7 trained MARL checkpoints."""
        specs = [
            ("mappo_pure", lambda: MAPPOAgent(
                obs_dim=20, state_dim=40, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="pure", device=self.dev
            ), "best_mappo_pure_agent.pt"),

            ("mappo_residual", lambda: MAPPOAgent(
                obs_dim=20, state_dim=40, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="residual", device=self.dev
            ), "best_mappo_residual_agent.pt"),

            ("ippo_pure", lambda: IPPOAgent(
                obs_dim=20, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="pure", device=self.dev
            ), "best_ippo_pure_agent.pt"),

            ("ippo_residual", lambda: IPPOAgent(
                obs_dim=20, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="residual", device=self.dev
            ), "best_ippo_residual_agent.pt"),

            ("maddpg_pure", lambda: MADDPGAgent(
                obs_dim=20, state_dim=40, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="pure", device=self.dev
            ), "best_maddpg_pure_agent.pt"),

            ("maddpg_residual", lambda: MADDPGAgent(
                obs_dim=20, state_dim=40, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, policy_variant="residual", device=self.dev
            ), "best_maddpg_residual_agent.pt"),

            ("cppo", lambda: CPPOAgent(
                state_dim=40, num_uavs=self.cfg.physics.num_uavs,
                num_offload_actions=self.cfg.physics.num_uavs + 1,
                hidden_dim=self.cfg.mappo.hidden_dim, device=self.dev
            ), "best_cppo_agent.pt")
        ]

        remed_dir = os.path.join(BASE_DIR, "results", "v2_remediated", "checkpoints")
        legacy_dir = os.path.join(BASE_DIR, "results", "checkpoints")

        for key, init_fn, ckpt_name in specs:
            agent = init_fn()
            # Try remediated compatible checkpoint first, then fallback to legacy
            p1 = os.path.join(remed_dir, ckpt_name)
            p2 = os.path.join(legacy_dir, ckpt_name)
            ckpt_path = p1 if os.path.exists(p1) else p2
            loaded = False
            err = None
            if os.path.exists(ckpt_path):
                try:
                    agent.load_checkpoint(ckpt_path)
                    loaded = True
                    src = "v2_remediated" if ckpt_path == p1 else "legacy"
                    print(f"[FlightEngine] Loaded checkpoint ({src}): {ckpt_name}")
                except Exception as e:
                    err = str(e)
                    print(f"[FlightEngine] Warning loading {ckpt_name}: {e}")
            else:
                err = f"Checkpoint not found at {ckpt_path}"
                print(f"[FlightEngine] Warning: {err}")
            self.agents[key] = agent
            self.agent_status[key] = {"loaded": loaded, "error": err, "checkpoint_path": ckpt_path}

    def compute_min_clearance(self, pos: np.ndarray, obstacles: list, margin: float = 2.0) -> float:
        """Computes continuous surface clearance to all cylindrical and spherical obstacles."""
        clearances = []
        for obs in obstacles:
            is_sphere = obs.get("type") == "sphere" if isinstance(obs, dict) else len(obs) == 5
            if is_sphere:
                cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"] if isinstance(obs, dict) else (obs[0], obs[1], obs[2], obs[3])
                d_3d = float(np.linalg.norm(pos - np.array([cx, cy, cz])))
                clearances.append(d_3d - (r + margin))
            else:
                if isinstance(obs, dict):
                    cx, cy, r, h = float(obs["cx"]), float(obs["cy"]), float(obs["radius"]), float(obs["height"])
                    z_min_obs = float(obs.get("z_min", 0.0))
                    z_max_obs = float(obs.get("z_max", h))
                else:
                    cx, cy, r, h = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
                    z_min_obs = 0.0
                    z_max_obs = h

                d_xy = float(np.linalg.norm(pos[:2] - np.array([cx, cy])))
                delta_xy = max(0.0, d_xy - r)
                if pos[2] > z_max_obs:
                    delta_z = pos[2] - z_max_obs
                elif pos[2] < z_min_obs:
                    delta_z = z_min_obs - pos[2]
                else:
                    delta_z = 0.0

                if delta_xy > 0.0 and delta_z > 0.0:
                    d_surface = np.sqrt(delta_xy**2 + delta_z**2)
                elif delta_xy > 0.0:
                    d_surface = delta_xy
                elif delta_z > 0.0:
                    d_surface = delta_z
                else:
                    d_surface = -min(r - d_xy, pos[2] - z_min_obs, z_max_obs - pos[2])

                clearances.append(float(d_surface - margin))
        return float(min(clearances)) if clearances else 999.0

    def make_obs_vector(self, pos: np.ndarray, vel: np.ndarray, target: np.ndarray, raw_obs: list) -> np.ndarray:
        """Constructs verified 20-dim observation vector conforming to MultiUAVEnv contracts."""
        edge_pos = np.array(self.cfg.airspace.edge_server_pos, dtype=np.float32)
        return build_single_uav_obs(
            pos=pos,
            vel=vel,
            station_pos=target,
            edge_pos=edge_pos,
            obstacles=raw_obs,
            task_features=None,
            peer_positions=None,
            comm_model=None,
            v_max=self.cfg.physics.v_max,
            airspace_bound=self.cfg.airspace.x_max,
            num_peers_expected=self.cfg.physics.num_uavs - 1
        )

    def make_cppo_state(self, pos: np.ndarray, vel: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Constructs 40-dim global fleet state vector for Centralized PPO with correct normalization."""
        d_g = float(np.linalg.norm(pos - target))
        p0_norm = (pos / self.cfg.airspace.x_max).tolist()
        v0_norm = (vel / self.cfg.physics.v_max).tolist()
        dg_norm = [d_g / self.cfg.airspace.x_max]
        task0_norm = [0.5, 0.5, 0.5]
        uav0_feat = p0_norm + v0_norm + dg_norm + task0_norm

        # Nominal normalized features for peer UAVs 1, 2, 3 conforming to MultiUAVEnv._get_global_state contract
        dummy_feat = [0.5, 0.5, 0.25, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5]
        full = uav0_feat + dummy_feat * (self.cfg.physics.num_uavs - 1)
        return np.array(full, dtype=np.float32)

    def generate_random_endpoints(self, obstacles: list, margin: float = 2.0) -> Dict[str, List[float]]:
        """Generates random, non-colliding origin and target coordinates within airspace bounds."""
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        air = self.cfg.airspace

        def is_safe(p):
            if not (air.x_min + 15 <= p[0] <= air.x_max - 15 and
                    air.y_min + 15 <= p[1] <= air.y_max - 15 and
                    air.z_min + 10 <= p[2] <= air.z_max - 20):
                return False
            return self.compute_min_clearance(p, obstacles, margin) > 3.0

        start = None
        target = None
        for _ in range(200):
            s = np.array([float(rng.uniform(20.0, 180.0)), float(rng.uniform(20.0, 180.0)), float(rng.uniform(15.0, 60.0))], dtype=np.float32)
            if is_safe(s):
                start = s
                break
        if start is None:
            start = np.array([20.0, 20.0, 15.0], dtype=np.float32)

        for _ in range(200):
            t = np.array([float(rng.uniform(20.0, 180.0)), float(rng.uniform(20.0, 180.0)), float(rng.uniform(15.0, 60.0))], dtype=np.float32)
            if is_safe(t) and float(np.linalg.norm(t - start)) >= 50.0:
                target = t
                break
        if target is None:
            target = np.array([180.0, 180.0, 25.0], dtype=np.float32)

        return {
            "source": [round(float(start[0]), 1), round(float(start[1]), 1), round(float(start[2]), 1)],
            "target": [round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1)]
        }

    def simulate_flight(
        self,
        planner_key: str,
        start: np.ndarray,
        target: np.ndarray,
        obstacles: list,
        margin: float = 2.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes unified simulation of the specified path planning or RL algorithm.
        Returns complete trajectory coordinates, explored tree nodes, and live telemetry.
        """
        t_start = time.time()
        air = self.cfg.airspace
        phy = self.cfg.physics
        bounds = (air.x_min, air.x_max, air.y_min, air.y_max, air.z_min, air.z_max)
        safety = SafetyFilter(obstacles=obstacles, d_safe=phy.d_safe, obstacle_margin=margin, bounds=bounds)

        explored_nodes = []
        path_coords = [start.tolist()]
        steps_telemetry = []
        status = "PATH_FOUND"
        t_plan = 0.0

        # =====================================================================
        # Case A: 3D A* Search Baseline (Delegated directly to models/baselines.py)
        # =====================================================================
        if planner_key == "astar":
            baseline = AStarJointBaseline(obstacles=obstacles, bounds=bounds, d_safe=phy.d_safe, obstacle_margin=margin)
            return baseline.simulate_trajectory(
                start=start, target=target, obstacles=obstacles, bounds=bounds,
                d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max, delta_t=phy.delta_t,
                reach_tolerance=phy.reach_tolerance, max_steps=self.cfg.mappo.episode_length,
                edge_server_pos=self.cfg.airspace.edge_server_pos
            )

        # =====================================================================
        # Case B: 3D RRT* Baseline (Delegated directly to models/rrt_star.py)
        # =====================================================================
        elif planner_key == "rrt_star":
            baseline = RRTStarBaseline(obstacles=obstacles, bounds=bounds, d_safe=phy.d_safe, obstacle_margin=margin)
            return baseline.simulate_trajectory(
                start=start, target=target, obstacles=obstacles, bounds=bounds,
                d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max, delta_t=phy.delta_t,
                reach_tolerance=phy.reach_tolerance, max_steps=self.cfg.mappo.episode_length,
                seed=seed, edge_server_pos=self.cfg.airspace.edge_server_pos
            )

        # =====================================================================
        # Case C: 3D SAPF Baseline (Delegated directly to models/apf.py)
        # =====================================================================
        elif planner_key == "sapf":
            baseline = SAPFBaseline(obstacles=obstacles, bounds=bounds, d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max)
            return baseline.simulate_trajectory(
                start=start, target=target, obstacles=obstacles, bounds=bounds,
                d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max, delta_t=phy.delta_t,
                reach_tolerance=phy.reach_tolerance, max_steps=self.cfg.mappo.episode_length,
                edge_server_pos=self.cfg.airspace.edge_server_pos
            )

        # =====================================================================
        # Case D: Hierarchical Hybrid A* + MAPPO (Delegated directly to models/astar.py)
        # =====================================================================
        elif planner_key == "hybrid_astar_mappo":
            hybrid = HybridAStarMAPPONavigator(
                obstacles=obstacles, bounds=bounds, mappo_agent=self.agents.get("mappo_pure"),
                d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max, delta_t=phy.delta_t,
                reach_tolerance=phy.reach_tolerance, residual_weight=0.0
            )
            return hybrid.simulate_trajectory(
                start=start, target=target, obstacles=obstacles, bounds=bounds,
                d_safe=phy.d_safe, obstacle_margin=margin, v_max=phy.v_max, delta_t=phy.delta_t,
                reach_tolerance=phy.reach_tolerance, max_steps=self.cfg.mappo.episode_length,
                edge_server_pos=self.cfg.airspace.edge_server_pos
            )

        # =====================================================================
        # Case E: MARL Policies (MAPPO, IPPO, MADDPG, CPPO)
        # =====================================================================
        elif planner_key in self.agents:
            agent = self.agents[planner_key]
            agent_names = {
                "mappo_pure": "Proposed Pure MAPPO (No APF)",
                "mappo_residual": "APF-Assisted Residual MAPPO",
                "ippo_pure": "Independent PPO (IPPO Pure)",
                "ippo_residual": "Independent PPO (IPPO Residual)",
                "maddpg_pure": "Multi-Agent DDPG (MADDPG Pure)",
                "maddpg_residual": "Multi-Agent DDPG (MADDPG Residual)",
                "cppo": "Centralized PPO (CPPO)"
            }
            agent_name = agent_names.get(planner_key, planner_key.upper())
            curr_pos = np.copy(start)
            curr_vel = np.zeros(3, dtype=np.float32)

            for s in range(self.cfg.mappo.episode_length):
                d_target = float(np.linalg.norm(curr_pos - target))
                if d_target <= phy.reach_tolerance:
                    status = "GOAL_REACHED"
                    break

                obs_vec = self.make_obs_vector(curr_pos, curr_vel, target, obstacles)
                t_obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.dev).unsqueeze(0)
                u_nom = np.zeros(3, dtype=np.float32)

                with torch.no_grad():
                    if planner_key == "cppo":
                        s_vec = self.make_cppo_state(curr_pos, curr_vel, target)
                        t_state = torch.tensor(s_vec, dtype=torch.float32, device=self.dev).unsqueeze(0)
                        act_move, act_off, _ = agent.actor.sample_action(t_state, deterministic=True)
                        u_neural = act_move[0, 0].cpu().numpy()
                    elif "maddpg" in planner_key:
                        u_move, _ = agent.actor(t_obs)
                        u_neural = u_move.squeeze(0).cpu().numpy()
                    else:  # mappo or ippo
                        act_move, _, _ = agent.actor.sample_action(t_obs, deterministic=True)
                        u_neural = act_move.squeeze(0).cpu().numpy()

                if "residual" in planner_key:
                    t_rel = (target - curr_pos) / max(1.0, float(np.linalg.norm(target - curr_pos)))
                    u_nom = 10.0 * min(1.0, d_target / 20.0) * t_rel

                u_applied = np.clip(u_neural, -phy.v_max, phy.v_max)
                if d_target < phy.v_max * phy.delta_t:
                    max_speed = max(0.5, d_target / phy.delta_t)
                    mag = float(np.linalg.norm(u_applied))
                    if mag > max_speed:
                        u_applied = u_applied * (max_speed / mag)

                proposed_pos = curr_pos + u_applied * phy.delta_t
                safe_dict, interv = safety.filter_trajectories({0: proposed_pos}, {0: curr_pos}, enable_filter=True)
                filtered_pos = safe_dict[0]
                actual_disp = filtered_pos - curr_pos
                curr_vel = actual_disp / phy.delta_t
                curr_pos = filtered_pos
                path_coords.append(curr_pos.tolist())

                min_c = self.compute_min_clearance(curr_pos, obstacles, margin)

                steps_telemetry.append({
                    "step": s + 1,
                    "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                    "speed": round(float(np.linalg.norm(curr_vel)), 2),
                    "dist_to_target": round(float(np.linalg.norm(curr_pos - target)), 2),
                    "min_clearance": round(min_c, 2),
                    "safety_intervened": bool(interv > 0),
                    "explanation": {
                        "planner": agent_name,
                        "decision_type": "Neural Policy Decision (Reactive)" if "pure" in planner_key or planner_key == "cppo" else "Nominal APF + Learned Residual Correction",
                        "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                        "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                        "neural_action": [round(float(u_neural[0]), 2), round(float(u_neural[1]), 2), round(float(u_neural[2]), 2)],
                        "apf_nominal": [round(float(u_nom[0]), 2), round(float(u_nom[1]), 2), round(float(u_nom[2]), 2)] if "residual" in planner_key else None,
                        "executed_control": [round(float(curr_vel[0]), 2), round(float(curr_vel[1]), 2), round(float(curr_vel[2]), 2)],
                        "safety_filter_status": "SAFETY FILTER INTERVENTION" if interv > 0 else "Clear flight"
                    }
                })

                if float(np.linalg.norm(curr_pos - target)) <= phy.reach_tolerance:
                    status = "GOAL_REACHED"
                    break

        else:
            raise ValueError(f"Unknown planner key: {planner_key}")

        if t_plan == 0.0:
            t_plan = (time.time() - t_start) * 1000.0

        final_d = float(np.linalg.norm(curr_pos - target))
        if final_d <= phy.reach_tolerance or (planner_key == "hybrid_astar_mappo" and final_d <= 8.0):
            status = "GOAL_REACHED"
        elif status == "PATH_FOUND":
            status = "TIMEOUT / INCOMPLETE"

        return self._package_result(status, planner_key, path_coords, explored_nodes, t_plan, steps_telemetry, obstacles, margin)

    def _package_result(
        self,
        status: str,
        planner_key: str,
        path_coords: list,
        explored_nodes: list,
        t_plan: float,
        steps_telemetry: list,
        obstacles: list,
        margin: float
    ) -> Dict[str, Any]:
        """Packages metrics and trajectory arrays into standardized JSON contract."""
        p_arr = np.array(path_coords)
        total_len = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
        all_clearances = [self.compute_min_clearance(p, obstacles, margin) for p in p_arr]
        min_overall_clearance = float(min(all_clearances)) if all_clearances else 0.0

        return {
            "status": status,
            "planner": planner_key,
            "path": path_coords,
            "explored_nodes": explored_nodes,
            "total_path_length": round(total_len, 2),
            "planning_time_ms": round(t_plan, 2),
            "min_clearance": round(min_overall_clearance, 2),
            "steps_count": len(path_coords) - 1,
            "steps_telemetry": steps_telemetry
        }

    def compare_flights(
        self,
        start: np.ndarray,
        target: np.ndarray,
        obstacles: list,
        margin: float = 2.0,
        selected: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Executes side-by-side benchmark comparison across all or selected planners."""
        all_planners = [
            ("Hierarchical A* + MAPPO (Hybrid)", "hybrid_astar_mappo"),
            ("3D A* Baseline", "astar"),
            ("3D RRT* (Optimal Tree)", "rrt_star"),
            ("3D SAPF (Adaptive Field)", "sapf"),
            ("APF-Residual MAPPO", "mappo_residual"),
            ("Proposed Pure MAPPO", "mappo_pure"),
            ("IPPO Residual", "ippo_residual"),
            ("IPPO Pure", "ippo_pure"),
            ("MADDPG Residual", "maddpg_residual"),
            ("MADDPG Pure", "maddpg_pure"),
            ("Centralized PPO (CPPO)", "cppo")
        ]

        if selected:
            planners = [p for p in all_planners if p[1] in selected]
        else:
            planners = all_planners

        results = []
        for label, p_key in planners:
            res_dict = self.simulate_flight(p_key, start, target, obstacles, margin)
            interv_count = sum(1 for s in res_dict.get("steps_telemetry", []) if s.get("safety_intervened"))
            results.append({
                "algorithm": label,
                "key": p_key,
                "status": res_dict.get("status"),
                "path_length_m": res_dict.get("total_path_length"),
                "execution_steps": res_dict.get("steps_count"),
                "planning_time_ms": res_dict.get("planning_time_ms"),
                "min_clearance_m": res_dict.get("min_clearance"),
                "safety_interventions": interv_count,
                "path": res_dict.get("path"),
                "steps_telemetry": res_dict.get("steps_telemetry", []),
                "explored_nodes": res_dict.get("explored_nodes", [])
            })
        return results
