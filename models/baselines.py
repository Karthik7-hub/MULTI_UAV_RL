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
from typing import Dict, Any, List, Optional
from models.astar import AStar3DPlanner

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
    def select_offload_action(self, uav_id: int, env) -> int:
        task = env.current_tasks[uav_id]
        if task is None:
            return 0

        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)

        # 1. Local estimated latency
        t_local = task.cycles / env.cfg.comp.f_local

        # 2. Peer estimated latency (scan all peers)
        best_peer_offset = 1
        min_peer_t = float("inf")
        for p_offset in range(1, env.num_uavs):
            peer_idx = (uav_id + p_offset) % env.num_uavs
            d_p = float(np.linalg.norm(pos - env.positions[peer_idx]))
            r_p = env.comm.data_rate(d_p, num_concurrent=1)
            t_tx = task.data_size / max(1e3, r_p)
            t_comp = task.cycles / env.cfg.comp.f_peer
            t_peer = t_tx + t_comp
            if t_peer < min_peer_t:
                min_peer_t = t_peer
                best_peer_offset = p_offset

        # 3. Edge server estimated latency
        d_edge = float(np.linalg.norm(pos - edge_pos))
        r_edge = env.comm.data_rate(d_edge, num_concurrent=1)
        t_edge_tx = task.data_size / max(1e3, r_edge)
        t_edge_comp = task.cycles / env.cfg.comp.f_edge
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
    # 1. Candidate Latencies & Energies:
    #    - Local Node (k=0):
    #        T_local = C / f_local [seconds]
    #        E_local = kappa * (f_local)^2 * C [Joules]
    #    - Peer UAVs (k in 1..N-1):
    #        T_peer = (D / R_peer(d_p)) + (C / f_peer) [seconds]
    #        E_peer = P_tx * (D / R_peer(d_p)) [Joules]
    #    - Ground Edge Server (k=N):
    #        T_edge = (D / R_edge(d_e)) + (C / f_edge) [seconds]
    #        E_edge = P_tx * (D / R_edge(d_e)) [Joules]
    #
    # 2. Penalty for Deadline Violation:
    #    I(condition) = 1.0 if T_total > T_max, else 0.0.
    #    If the task cannot finish before deadline T_max, a heavy penalty (10.0) is added.
    #
    # 3. Parameters:
    #    - weight_energy (w_E) = 0.005: Trade-off factor converting Joules to seconds-equivalent.
    #    - deadline_penalty (lambda_penalty) = 10.0: Penalty for missing latency deadline.
    # =========================================================================
    """
    def __init__(self, weight_energy: float = 0.005, deadline_penalty: float = 10.0):
        self.weight_energy = weight_energy
        self.deadline_penalty = deadline_penalty

    def select_offload_action(self, uav_id: int, env) -> int:
        task = env.current_tasks[uav_id]
        if task is None:
            return 0

        pos = env.positions[uav_id]
        edge_pos = np.array(env.cfg.airspace.edge_server_pos)
        deadline = task.deadline

        choices = []  # tuples of (cost, action_idx)

        # 1. Local execution
        t_local = task.cycles / env.cfg.comp.f_local
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
            t_comp = task.cycles / env.cfg.comp.f_peer
            t_total = t_tx + t_comp
            miss_peer = 1.0 if t_total > deadline else 0.0
            cost_peer = t_total + self.weight_energy * e_tx + self.deadline_penalty * miss_peer
            choices.append((cost_peer, p_offset))

        # 3. Edge Server execution
        d_edge = float(np.linalg.norm(pos - edge_pos))
        r_edge = env.comm.data_rate(d_edge, num_concurrent=1)
        t_tx_edge = task.data_size / max(1e3, r_edge)
        e_tx_edge = env.comm.tx_power * t_tx_edge
        t_comp_edge = task.cycles / env.cfg.comp.f_edge
        t_edge_total = t_tx_edge + t_comp_edge
        miss_edge = 1.0 if t_edge_total > deadline else 0.0
        cost_edge = t_edge_total + self.weight_energy * e_tx_edge + self.deadline_penalty * miss_edge
        choices.append((cost_edge, env.num_uavs))

        # Select destination minimizing composite cost
        choices.sort(key=lambda x: x[0])
        return choices[0][1]

class AStarJointBaseline:
    """
    Modular 3D A* Path Planner paired with any pluggable MEC offloading policy.
    Provides standard path planning while allowing flexible baseline comparisons.
    """
    def __init__(self, env, offload_policy: Optional[BaseOffloadingHeuristic] = None):
        self.env = env
        self.planner = AStar3DPlanner(
            obstacles=env.cfg.airspace.obstacles,
            grid_res=5.0,
            d_safe=env.cfg.physics.d_safe
        )
        self.offload_policy = offload_policy or EnergyDeadlineAwareGreedyPolicy()
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

            # 3D Navigation Action: Steer along planned waypoint corridor
            if d_final > p_cfg.reach_tolerance:
                dir_vec = (curr_target - pos) / max(1e-6, d_target)
                step_dist = min(d_target, p_cfg.v_max * p_cfg.delta_t)
                disp = dir_vec * step_dist
            else:
                # Arrived at station: hold position (deadband stop)
                disp = np.zeros(3, dtype=np.float32)

            # Task Offloading Action from designated heuristic
            offload_act = self.offload_policy.select_offload_action(j, self.env)

            actions[j] = {"move": disp.astype(np.float32), "offload": offload_act}

        return actions
