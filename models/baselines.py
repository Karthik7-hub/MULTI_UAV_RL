"""
Comprehensive Heuristic & Classical Baselines for Multi-UAV Edge Computing.
Implements:
1. A* + Energy- and Deadline-Aware Greedy MEC Policy (Strong Classical Baseline)
2. Lowest-Latency Greedy Policy
3. Nearest-Feasible Destination Policy
4. Local-Only Execution Policy
5. Edge-Only Offloading Policy
6. Random Feasible Offloading Policy
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from models.astar import AStar3DPlanner, PurePursuit3DController
from env.safety_filter import SafetyFilter, compute_min_clearance
from env.observation_builder import build_single_uav_obs

def compute_min_clearance(pos: np.ndarray, obstacles: list, margin: float = 2.0) -> float:
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
    return float(min(clearances)) if clearances else 50.0

class BaseOffloadingHeuristic:
    """Interface for rule-based task offloading policies."""
    def select_offload_action(self, uav_id: int, env) -> int:
        raise NotImplementedError

class LocalOnlyPolicy(BaseOffloadingHeuristic):
    """Executes 100% of tasks on the local UAV CPU."""
    def select_offload_action(self, uav_id: int, env) -> int:
        return 0

class EdgeOnlyPolicy(BaseOffloadingHeuristic):
    """Offloads 100% of tasks to the Ground Edge Server."""
    def select_offload_action(self, uav_id: int, env) -> int:
        return env.num_uavs

class RandomFeasiblePolicy(BaseOffloadingHeuristic):
    """Random uniform selection among all valid offload destinations (Lower Bound)."""
    def select_offload_action(self, uav_id: int, env) -> int:
        return int(np.random.randint(0, env.num_uavs + 1))

class NearestFeasiblePolicy(BaseOffloadingHeuristic):
    """Selects the geometrically nearest available computation node (Local vs Peers vs Edge)."""
    def select_offload_action(self, uav_id: int, env) -> int:
        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)
        
        # Local distance is 0.0
        d_local = 0.0
        d_edge = float(np.linalg.norm(pos - edge_pos))
        
        peer_dists = []
        for p_offset in range(1, env.num_uavs):
            peer_idx = (uav_id + p_offset) % env.num_uavs
            d_p = float(np.linalg.norm(pos - env.positions[peer_idx]))
            peer_dists.append((d_p, p_offset))
            
        peer_dists.sort(key=lambda x: x[0])
        best_peer_dist, best_peer_offset = peer_dists[0]
        
        # Heuristic threshold: if closest peer is very close (< 25m), consider peer, else local if edge is far
        # Direct geometric nearest:
        # Local is distance 0, but if we compare remote candidates:
        if d_edge < best_peer_dist:
            return env.num_uavs
        else:
            return best_peer_offset

class LowestLatencyPolicy(BaseOffloadingHeuristic):
    """Greedily selects the destination minimizing estimated end-to-end task completion latency."""
    def __init__(self, queue_aware: bool = True):
        self.queue_aware = queue_aware

    def select_offload_action(self, uav_id: int, env) -> int:
        task = env.current_tasks[uav_id]
        if task is None:
            return 0

        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)

        # 1. Local estimated latency (including local queue backlog if queue_aware)
        q_local = env.comp.local_queues.get(uav_id, 0.0) if self.queue_aware else 0.0
        t_local = (q_local + task.cycles) / env.cfg.comp.f_local

        # 2. Peer estimated latency (scan all peers)
        best_peer_offset = 1
        min_peer_t = float("inf")
        for p_offset in range(1, env.num_uavs):
            peer_idx = (uav_id + p_offset) % env.num_uavs
            d_p = float(np.linalg.norm(pos - env.positions[peer_idx]))
            r_p = env.comm.data_rate(d_p, num_concurrent=1)
            t_tx = task.data_size / max(1e3, r_p)
            q_peer = env.comp.peer_queues.get(peer_idx, 0.0) if self.queue_aware else 0.0
            t_comp = (q_peer + task.cycles) / env.cfg.comp.f_peer
            t_peer = t_tx + t_comp
            if t_peer < min_peer_t:
                min_peer_t = t_peer
                best_peer_offset = p_offset

        # 3. Edge server estimated latency (including transmission contention & edge queue backlog)
        d_edge = float(np.linalg.norm(pos - edge_pos))
        r_edge = env.comm.data_rate(d_edge, num_concurrent=env.num_uavs)
        t_edge_tx = task.data_size / max(1e3, r_edge)
        bg_load = getattr(env.comp, "edge_background_load", 2)
        eff_f_edge = env.cfg.comp.f_edge / max(1, bg_load + 1)
        q_edge = env.comp.edge_queue if self.queue_aware else 0.0
        t_edge_comp = (q_edge + task.cycles) / eff_f_edge
        t_edge = t_edge_tx + t_edge_comp

        candidates = [t_local, min_peer_t, t_edge]
        min_idx = int(np.argmin(candidates))
        if min_idx == 0:
            return 0
        elif min_idx == 1:
            return best_peer_offset
        else:
            return env.num_uavs

class EnergyDeadlineAwareGreedyPolicy(BaseOffloadingHeuristic):
    """
    # =========================================================================
    # PURPOSE:
    # Principled multi-objective classical offloading heuristic.
    # Evaluates each possible offloading candidate k in {0=Local, 1..N-1=Peers, N=Edge}
    # and selects the destination that minimizes a composite multi-objective cost J(k).
    #
    # MATHEMATICAL FORMULATION:
    # J(k) = T_total(k) + w_E * E(k) + lambda_penalty * I( T_total(k) > T_max )
    #
    # Incorporates realistic queue backlog and multi-tenant slicing dynamics.
    # =========================================================================
    """
    def __init__(self, weight_energy: float = 0.005, deadline_penalty: float = 10.0, queue_aware: bool = True):
        self.weight_energy = weight_energy
        self.deadline_penalty = deadline_penalty
        self.queue_aware = queue_aware

    def select_offload_action(self, uav_id: int, env) -> int:
        task = env.current_tasks[uav_id]
        if task is None:
            return 0

        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)
        deadline = task.deadline

        choices = []  # tuples of (cost, action_idx)

        # 1. Local execution
        q_local = env.comp.local_queues.get(uav_id, 0.0) if self.queue_aware else 0.0
        t_local = (q_local + task.cycles) / env.cfg.comp.f_local
        e_local = env.cfg.comp.kappa * (env.cfg.comp.f_local ** 2) * task.cycles
        miss_local = 1.0 if t_local > deadline else 0.0
        cost_local = t_local + self.weight_energy * e_local + self.deadline_penalty * miss_local
        choices.append((cost_local, 0))

        # 2. Peer UAV execution (all peers)
        for p_offset in range(1, env.num_uavs):
            peer_idx = (uav_id + p_offset) % env.num_uavs
            d_p = float(np.linalg.norm(pos - env.positions[peer_idx]))
            r_p = env.comm.data_rate(d_p, num_concurrent=1)
            t_tx = task.data_size / max(1e3, r_p)
            e_tx = env.comm.tx_power * t_tx
            q_peer = env.comp.peer_queues.get(peer_idx, 0.0) if self.queue_aware else 0.0
            t_comp = (q_peer + task.cycles) / env.cfg.comp.f_peer
            t_total = t_tx + t_comp
            miss_peer = 1.0 if t_total > deadline else 0.0
            cost_peer = t_total + self.weight_energy * e_tx + self.deadline_penalty * miss_peer
            choices.append((cost_peer, p_offset))

        # 3. Edge Server execution (including queue backlog and multi-tenant slicing)
        d_edge = float(np.linalg.norm(pos - edge_pos))
        r_edge = env.comm.data_rate(d_edge, num_concurrent=env.num_uavs)
        t_tx_edge = task.data_size / max(1e3, r_edge)
        e_tx_edge = env.comm.tx_power * t_tx_edge
        bg_load = getattr(env.comp, "edge_background_load", 2)
        eff_f_edge = env.cfg.comp.f_edge / max(1, bg_load + 1)
        q_edge = env.comp.edge_queue if self.queue_aware else 0.0
        t_comp_edge = (q_edge + task.cycles) / eff_f_edge
        t_edge_total = t_tx_edge + t_comp_edge
        miss_edge = 1.0 if t_edge_total > deadline else 0.0
        cost_edge = t_edge_total + self.weight_energy * e_tx_edge + self.deadline_penalty * miss_edge
        choices.append((cost_edge, env.num_uavs))

        # Select destination minimizing composite cost
        choices.sort(key=lambda x: x[0])
        return choices[0][1]

class LoadBalancingHeuristic(BaseOffloadingHeuristic):
    """
    Decentralized Queue-Load Balancing Heuristic for Multi-Tier MEC Networks.

    Decision Rule:
    Evaluates composite offloading cost J_k for each destination k in {0=Local, 1..N-1=Peers, N=Edge}:
        J_k = T_tx(k) + W_k + T_exec(k) + alpha_load * (Q_k / C_k)
    where:
    - W_k = Q_k / f_k is queue waiting time
    - T_tx(k) = D / R_k is wireless transmission delay (0 for local)
    - T_exec(k) = C / f_k is computational execution time
    - (Q_k / C_k) penalizes cumulative backlog normalized by processor service capacity C_k = f_k * delta_t.
    """
    def __init__(self, load_weight: float = 0.1):
        self.load_weight = load_weight

    def select_offload_action(self, uav_id: int, env) -> int:
        task = env.current_tasks[uav_id]
        if task is None:
            return 0

        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)
        candidates = []  # (cost, action_idx)

        # 1. Local CPU evaluation
        f_loc = env.cfg.comp.f_local
        q_loc = env.comp.local_queues.get(uav_id, 0.0)
        t_loc_wait = q_loc / f_loc
        t_loc_exec = task.cycles / f_loc
        cost_loc = t_loc_wait + t_loc_exec + self.load_weight * (q_loc / max(1e6, f_loc))
        candidates.append((cost_loc, 0))

        # 2. Peer UAVs evaluation
        f_peer = env.cfg.comp.f_peer
        for p_offset in range(1, env.num_uavs):
            peer_idx = (uav_id + p_offset) % env.num_uavs
            d_p = float(np.linalg.norm(pos - env.positions[peer_idx]))
            r_p = env.comm.data_rate(d_p, num_concurrent=1)
            t_tx = task.data_size / max(1e3, r_p)
            q_p = env.comp.peer_queues.get(peer_idx, 0.0)
            t_p_wait = q_p / f_peer
            t_p_exec = task.cycles / f_peer
            cost_p = t_tx + t_p_wait + t_p_exec + self.load_weight * (q_p / max(1e6, f_peer))
            candidates.append((cost_p, p_offset))

        # 3. Ground Edge Server evaluation
        bg_load = getattr(env.comp, "edge_background_load", 2)
        eff_f_edge = env.cfg.comp.f_edge / max(1, bg_load + 1)
        d_edge = float(np.linalg.norm(pos - edge_pos))
        r_edge = env.comm.data_rate(d_edge, num_concurrent=env.num_uavs)
        t_tx_edge = task.data_size / max(1e3, r_edge)
        q_edge = env.comp.edge_queue
        t_edge_wait = q_edge / eff_f_edge
        t_edge_exec = task.cycles / eff_f_edge
        cost_edge = t_tx_edge + t_edge_wait + t_edge_exec + self.load_weight * (q_edge / max(1e6, eff_f_edge))
        candidates.append((cost_edge, env.num_uavs))

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else 0

class AStarJointBaseline:
    """
    Modular 3D A* Path Planner paired with any pluggable MEC offloading policy.
    Supports:
    1. Multi-UAV Gym Environment step-by-step evaluation (evaluate.py).
    2. Autonomous point-to-point 3D trajectory simulation (WebGL visualizer).
    """
    def __init__(
        self,
        env: Optional[Any] = None,
        offload_policy: Optional[BaseOffloadingHeuristic] = None,
        obstacles: Optional[list] = None,
        bounds: Tuple[float, float, float, float, float, float] = (0.0, 200.0, 0.0, 200.0, 0.0, 100.0),
        grid_res: float = 5.0,
        d_safe: float = 3.0,
        obstacle_margin: float = 2.0
    ):
        self.env = env
        self.offload_policy = offload_policy or EnergyDeadlineAwareGreedyPolicy()
        self.bounds = bounds
        self.grid_res = grid_res
        self.d_safe = d_safe
        self.obstacle_margin = obstacle_margin

        if env is not None:
            raw_obs = env.cfg.airspace.obstacles
            self.bounds = (
                env.cfg.airspace.x_min, env.cfg.airspace.x_max,
                env.cfg.airspace.y_min, env.cfg.airspace.y_max,
                env.cfg.airspace.z_min, env.cfg.airspace.z_max
            )
            self.d_safe = env.cfg.physics.d_safe
            self.obstacle_margin = env.cfg.physics.obstacle_margin
        else:
            raw_obs = obstacles or []

        self.planner = AStar3DPlanner(
            obstacles=raw_obs,
            bounds=self.bounds,
            grid_res=self.grid_res,
            d_safe=self.d_safe,
            obstacle_margin=self.obstacle_margin
        )
        self.cached_paths = {}
        self.current_wp_idx = {}

    def reset(self):
        self.cached_paths = {}
        self.current_wp_idx = {}

    def select_actions(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        actions = {}
        p_cfg = self.env.cfg.physics

        for j in range(self.env.num_uavs):
            pos = self.env.positions[j]
            goal = self.env.stations[j]

            if j not in self.cached_paths:
                self.cached_paths[j] = self.planner.plan_path(pos, goal)
                self.current_wp_idx[j] = 1

            path = self.cached_paths[j]
            wp_idx = self.current_wp_idx[j]
            curr_target = path[min(wp_idx, len(path) - 1)]

            d_target = float(np.linalg.norm(curr_target - pos))
            d_final = float(np.linalg.norm(goal - pos))

            if d_target < 4.0 and wp_idx < len(path) - 1:
                self.current_wp_idx[j] += 1
                curr_target = path[self.current_wp_idx[j]]
                d_target = float(np.linalg.norm(curr_target - pos))

            # 3D Navigation Action: Steer along planned waypoint corridor with inter-UAV cooperative separation
            if d_final > p_cfg.reach_tolerance:
                dir_vec = (curr_target - pos) / max(1e-6, d_target)
                step_dist = min(d_target, p_cfg.v_max * p_cfg.delta_t)
                disp = dir_vec * step_dist

                # Inter-UAV cooperative separation to prevent multi-agent collisions
                repulsion = np.zeros(3, dtype=np.float32)
                for k in range(self.env.num_uavs):
                    if k != j:
                        p_k = self.env.positions[k]
                        d_peer = float(np.linalg.norm(pos - p_k))
                        if d_peer < self.d_safe + 4.0:
                            rep_dir = (pos - p_k) / max(1e-4, d_peer)
                            repulsion += rep_dir * float(self.d_safe + 4.0 - d_peer) * 0.8
                if np.linalg.norm(repulsion) > 0.05:
                    disp = disp + repulsion
                    mag = float(np.linalg.norm(disp))
                    if mag > p_cfg.v_max * p_cfg.delta_t:
                        disp = disp * (p_cfg.v_max * p_cfg.delta_t / mag)
            else:
                # Arrived at station: hold position (deadband stop)
                disp = np.zeros(3, dtype=np.float32)

            # Task Offloading Action from designated heuristic
            offload_act = self.offload_policy.select_offload_action(j, self.env)

            actions[j] = {"move": disp.astype(np.float32), "offload": offload_act}

        return actions

    def simulate_trajectory(
        self,
        start: np.ndarray,
        target: np.ndarray,
        obstacles: Optional[list] = None,
        bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        d_safe: float = 3.0,
        obstacle_margin: float = 2.0,
        v_max: float = 15.0,
        delta_t: float = 1.0,
        reach_tolerance: float = 5.0,
        max_steps: int = 100,
        lookahead_dist: float = 10.0,
        decel_dist: float = 15.0,
        edge_server_pos: Tuple[float, float, float] = (100.0, 100.0, 0.0),
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes complete 3D A* search, trajectory tracking, and live telemetry generation.
        Returns standardized JSON contract dictionary for WebGL visualizer.
        """
        start = np.asarray(start, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        use_bounds = bounds or self.bounds
        use_obstacles = obstacles if obstacles is not None else self.planner.obstacles

        planner = AStar3DPlanner(
            obstacles=use_obstacles,
            bounds=use_bounds,
            grid_res=self.grid_res,
            d_safe=d_safe,
            obstacle_margin=obstacle_margin
        )

        path_3d, t_plan, exp_list = planner.plan_path_with_details(start, target)
        explored_nodes = [[float(p[0]), float(p[1]), float(p[2])] for p in exp_list[:600]]
        path_coords = [start.tolist()]
        steps_telemetry = []

        if path_3d is None or len(path_3d) < 2:
            return {
                "status": "NO_PATH_FOUND",
                "planner": "astar",
                "path": path_coords,
                "explored_nodes": explored_nodes,
                "total_path_length": 0.0,
                "planning_time_ms": round(t_plan, 2),
                "min_clearance": compute_min_clearance(start, use_obstacles, obstacle_margin),
                "steps_count": 0,
                "steps_telemetry": []
            }

        tracker = PurePursuit3DController(
            waypoints=path_3d,
            v_max=v_max,
            lookahead_dist=lookahead_dist,
            acceptance_radius=reach_tolerance,
            decel_dist=decel_dist
        )
        safety = SafetyFilter(
            obstacles=use_obstacles,
            d_safe=d_safe,
            obstacle_margin=obstacle_margin,
            bounds=use_bounds
        )

        curr_pos = np.copy(start)
        cum_dist = 0.0
        status = "PATH_FOUND"

        for s in range(max_steps):
            d_target = float(np.linalg.norm(curr_pos - target))
            if d_target <= reach_tolerance:
                status = "GOAL_REACHED"
                break

            next_pos, v_cmd, wp_idx, trk_status = tracker.step(curr_pos, dt=delta_t)
            safe_dict, interv = safety.filter_trajectories({0: next_pos}, {0: curr_pos}, enable_filter=True)
            actual_pos = safe_dict[0]
            actual_disp = actual_pos - curr_pos
            actual_speed = float(np.linalg.norm(actual_disp)) / delta_t
            cum_dist += float(np.linalg.norm(actual_disp))
            curr_pos = actual_pos
            path_coords.append(curr_pos.tolist())

            min_c = compute_min_clearance(curr_pos, use_obstacles, obstacle_margin)

            obs_vec = build_single_uav_obs(
                pos=curr_pos,
                vel=actual_disp / delta_t,
                station_pos=target,
                edge_pos=np.array(edge_server_pos, dtype=np.float32),
                obstacles=use_obstacles,
                v_max=v_max,
                airspace_bound=use_bounds[1]
            )

            steps_telemetry.append({
                "step": s + 1,
                "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                "speed": round(actual_speed, 2),
                "dist_to_target": round(d_target, 2),
                "min_clearance": round(min_c, 2),
                "safety_intervened": bool(interv > 0),
                "explanation": {
                    "planner": "3D A* Baseline",
                    "decision_type": "Deterministic 3D Grid Search + Waypoint Lookahead Tracking",
                    "g_cost": round(cum_dist, 2),
                    "h_cost": round(d_target, 2),
                    "f_cost": round(cum_dist + d_target, 2),
                    "direction_chosen": f"Waypoint #{wp_idx + 1} / {len(path_3d)}",
                    "explored_nodes_count": len(exp_list),
                    "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                    "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                    "neural_action": [round(float(v_cmd[0]), 2), round(float(v_cmd[1]), 2), round(float(v_cmd[2]), 2)],
                    "executed_control": [round(float(actual_disp[0]), 2), round(float(actual_disp[1]), 2), round(float(actual_disp[2]), 2)]
                }
            })

            if trk_status == "GOAL_REACHED" or float(np.linalg.norm(curr_pos - target)) <= reach_tolerance:
                status = "GOAL_REACHED"
                break

        p_arr = np.array(path_coords)
        total_len = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
        all_clearances = [compute_min_clearance(p, use_obstacles, obstacle_margin) for p in p_arr]
        min_overall_clearance = float(min(all_clearances)) if all_clearances else 0.0

        return {
            "status": status,
            "planner": "astar",
            "path": path_coords,
            "explored_nodes": explored_nodes,
            "total_path_length": round(total_len, 2),
            "planning_time_ms": round(t_plan, 2),
            "min_clearance": round(min_overall_clearance, 2),
            "steps_count": len(path_coords) - 1,
            "steps_telemetry": steps_telemetry
        }
