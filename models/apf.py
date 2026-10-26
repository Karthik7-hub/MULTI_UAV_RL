"""
Sequential Adaptive Potential Field (SAPF / APF) 3D Navigation for Multi-UAV Systems.
Features:
- Attractive potential field directed toward designated mission stations
- Nonlinear cylindrical obstacle repulsive field with finite height handling
- Tangential circulation field preventing local minima / dead-end entrapment
- Dynamic inter-UAV separation repulsive forces
- Step-by-step physical displacement output integrated into MultiUAVEnv
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from env.safety_filter import SafetyFilter
from env.observation_builder import build_single_uav_obs
from env.safety_filter import compute_min_clearance

class SAPFNavigator:
    def __init__(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        k_att: float = 1.0,
        k_rep: float = 80.0,
        k_tangent: float = 45.0,
        k_peer: float = 40.0,
        d_0: float = 25.0,
        d_safe: float = 8.0,
        obstacle_margin: float = 2.0,
        v_max: float = 15.0
    ):
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_tangent = k_tangent
        self.k_peer = k_peer
        self.d_0 = d_0
        self.d_safe = d_safe
        self.eff_margin = (d_safe / 2.0) + obstacle_margin
        self.v_max = v_max

    def compute_force(
        self,
        uav_id: int,
        pos: np.ndarray,
        goal: np.ndarray,
        peer_positions: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        # =========================================================================
        # PURPOSE:
        # Calculates the net artificial potential field (APF) force vector guiding UAV j.
        #
        # EQUATIONS:
        # F_net = F_att + F_rep + F_tangent + F_peer
        #
        # 1. Attractive Force toward Designated Goal Station:
        #    F_att = k_att * (goal - pos)
        #    Pulls the UAV smoothly toward its assigned station.
        #
        # 2. Nonlinear Cylindrical Obstacle Repulsive Force:
        #    When distance to cylinder surface d_surf < d_0:
        #    F_rep = k_rep * (1/d_surf - 1/d_0) * (1 / d_surf^2) * n_out
        #    Pushes the UAV radially outward away from the cylinder core.
        #
        # 3. Tangential Circulation Field (Prevents Local Minima & Deadlocks):
        #    F_tangent = k_tangent * (1 / d_surf) * n_perp
        #    Where n_perp is perpendicular to n_out in the direction pointing toward the goal.
        #    Imparts a vortex-like circulation around the cylinder so the UAV slides
        #    effortlessly around obstacles instead of getting trapped in a head-on deadlock!
        #
        # 4. Inter-UAV Separation Repulsive Force:
        #    F_peer = k_peer * (1/d_peer - 1/d_safe_sep) * (1 / d_peer^2) * u_peer
        #    Prevents mid-air collisions between adjacent UAVs.
        # =========================================================================
        """
        # 1. Attractive Force toward Station
        vec_goal = goal - pos
        d_goal = float(np.linalg.norm(vec_goal))
        if d_goal > 1e-4:
            u_goal = vec_goal / d_goal
            # Attenuate attraction within 4m of station for stable hover
            scale_att = np.clip((d_goal - 2.0) / 10.0, 0.0, 1.0)
            f_att = self.k_att * u_goal * min(d_goal, 12.0) * scale_att
        else:
            f_att = np.zeros(3, dtype=np.float32)

        # 2. Obstacle Repulsive & Tangential Circulation Forces
        f_rep = np.zeros(3, dtype=np.float32)
        min_clearance = float("inf")

        for obs in self.obstacles:
            if isinstance(obs, dict):
                cx, cy, r, h = float(obs["cx"]), float(obs["cy"]), float(obs["radius"]), float(obs["height"])
            else:
                cx, cy, r, h = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
            eff_r = r + self.eff_margin
            eff_h = h + self.eff_margin
            d_xy = float(np.linalg.norm(pos[:2] - np.array([cx, cy])))
            d_surf = d_xy - eff_r

            if d_surf < min_clearance:
                min_clearance = d_surf

            # Active repulsion if within influence radius and below top height
            if d_surf < self.d_0 and pos[2] <= eff_h:
                if d_xy > 1e-4:
                    n_out = (pos[:2] - np.array([cx, cy])) / d_xy
                else:
                    n_out = np.array([1.0, 0.0])

                # Repulsive magnitude
                d_eff = max(0.5, d_surf)
                rep_mag = self.k_rep * (1.0 / d_eff - 1.0 / self.d_0) * (1.0 / (d_eff ** 2))
                f_rep[:2] += rep_mag * n_out

                # Adaptive Tangential Circulation Field to avoid local minima
                # Compute perpendicular vector that aligns favorably with goal direction
                n_perp = np.array([-n_out[1], n_out[0]])
                if np.dot(n_perp, vec_goal[:2]) < 0:
                    n_perp = -n_perp
                f_rep[:2] += self.k_tangent * (1.0 / d_eff) * n_perp

                # Vertical evasion force if near cylinder top
                if pos[2] > (h - 10.0):
                    f_rep[2] += 5.0 * (1.0 / d_eff)

        # 3. Inter-UAV Separation Repulsion
        f_peer = np.zeros(3, dtype=np.float32)
        for other_id, other_pos in peer_positions.items():
            if other_id != uav_id:
                d_p = float(np.linalg.norm(pos - other_pos))
                if d_p < self.d_safe * 1.8:
                    d_p_eff = max(0.5, d_p)
                    u_p = (pos - other_pos) / d_p_eff
                    f_peer += self.k_peer * (1.0 / d_p_eff - 1.0 / (self.d_safe * 1.8)) * (1.0 / (d_p_eff ** 2)) * u_p

        f_net = f_att + f_rep + f_peer
        return f_net, min_clearance

class SAPFBaseline:
    """
    # =========================================================================
    # PURPOSE:
    # Sequential Adaptive Potential Field (SAPF) Multi-UAV Navigation Baseline.
    # Evaluates artificial attractive, repulsive, and circulation potential forces
    # at every discrete simulation timestep to compute real-time flight vectors,
    # paired with classical greedy MEC task offloading.
    # =========================================================================
    """
class SAPFBaseline:
    """
    Sequential Adaptive Potential Field (SAPF / APF) 3D Navigation Baseline.
    Supports:
    1. Multi-UAV Gym Environment step-by-step evaluation (evaluate.py).
    2. Autonomous point-to-point 3D trajectory simulation (WebGL visualizer).
    """
    def __init__(
        self,
        env: Optional[Any] = None,
        offload_policy=None,
        obstacles: Optional[list] = None,
        k_att: float = 1.2,
        k_rep: float = 80.0,
        k_tangent: float = 45.0,
        d_0: float = 25.0,
        d_safe: float = 3.0,
        obstacle_margin: float = 2.0,
        v_max: float = 15.0,
        bounds: Tuple[float, float, float, float, float, float] = (0.0, 200.0, 0.0, 200.0, 0.0, 100.0)
    ):
        self.env = env
        from models.baselines import EnergyDeadlineAwareGreedyPolicy
        self.offload_policy = offload_policy or EnergyDeadlineAwareGreedyPolicy()
        self.bounds = bounds
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_tangent = k_tangent
        self.d_0 = d_0
        self.d_safe = d_safe
        self.obstacle_margin = obstacle_margin
        self.v_max = v_max

        if env is not None:
            raw_obs = env.cfg.airspace.obstacles
            self.d_safe = env.cfg.physics.d_safe
            self.obstacle_margin = env.cfg.physics.obstacle_margin
            self.v_max = env.cfg.physics.v_max
        else:
            raw_obs = obstacles or []

        self.navigator = SAPFNavigator(
            obstacles=raw_obs,
            k_att=self.k_att,
            k_rep=self.k_rep,
            k_tangent=self.k_tangent,
            d_0=self.d_0,
            d_safe=self.d_safe,
            obstacle_margin=self.obstacle_margin,
            v_max=self.v_max
        )
        self.planning_times = {}

    def reset(self):
        self.planning_times = {}

    def select_actions(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        actions = {}
        p_cfg = self.env.cfg.physics

        for j in range(self.env.num_uavs):
            t0 = time.time()
            pos = self.env.positions[j]
            goal = self.env.stations[j]

            f_net, min_clear = self.navigator.compute_force(
                uav_id=j,
                pos=pos,
                goal=goal,
                peer_positions=self.env.positions
            )
            self.planning_times[j] = (time.time() - t0) * 1000.0

            d_final = float(np.linalg.norm(goal - pos))
            if d_final > p_cfg.reach_tolerance:
                f_norm = float(np.linalg.norm(f_net))
                if f_norm > 1e-4:
                    disp = (f_net / f_norm) * min(f_norm, p_cfg.v_max * p_cfg.delta_t)
                else:
                    disp = np.zeros(3, dtype=np.float32)
            else:
                disp = np.zeros(3, dtype=np.float32)

            offload_act = self.offload_policy.select_offload_action(j, self.env)
            actions[j] = {"move": disp.astype(np.float32), "offload": offload_act}

        return actions

    def simulate_trajectory(
        self,
        start: np.ndarray,
        target: np.ndarray,
        obstacles: Optional[list] = None,
        bounds: Optional[Tuple[float, float, float, float, float, float]] = (0.0, 200.0, 0.0, 200.0, 0.0, 100.0),
        d_safe: float = 3.0,
        obstacle_margin: float = 2.0,
        v_max: float = 15.0,
        delta_t: float = 1.0,
        reach_tolerance: float = 5.0,
        max_steps: int = 100,
        edge_server_pos: Tuple[float, float, float] = (100.0, 100.0, 0.0),
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes real-time artificial potential field navigation, kinematic projection filtering,
        and live telemetry generation. Returns standardized JSON contract dictionary for WebGL.
        """
        start = np.asarray(start, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        use_bounds = bounds or (0.0, 200.0, 0.0, 200.0, 0.0, 100.0)
        use_obstacles = obstacles if obstacles is not None else self.navigator.obstacles

        nav = SAPFNavigator(
            obstacles=use_obstacles,
            k_att=self.k_att,
            k_rep=self.k_rep,
            k_tangent=self.k_tangent,
            d_0=self.d_0,
            d_safe=d_safe,
            obstacle_margin=obstacle_margin,
            v_max=v_max
        )
        safety = SafetyFilter(
            obstacles=use_obstacles,
            d_safe=d_safe,
            obstacle_margin=obstacle_margin,
            bounds=use_bounds
        )

        curr_pos = np.copy(start)
        path_coords = [curr_pos.tolist()]
        steps_telemetry = []
        status = "PATH_FOUND"
        t0 = time.time()

        for s in range(max_steps):
            d_target = float(np.linalg.norm(curr_pos - target))
            if d_target <= reach_tolerance:
                status = "GOAL_REACHED"
                break

            f_net, min_c = nav.compute_force(0, curr_pos, target, {})
            u_applied = np.clip(f_net, -v_max, v_max)

            # Terminal smooth deceleration near goal
            if d_target < v_max * delta_t:
                max_speed = max(0.5, d_target / delta_t)
                mag = float(np.linalg.norm(u_applied))
                if mag > max_speed:
                    u_applied = u_applied * (max_speed / mag)

            proposed_pos = curr_pos + u_applied * delta_t
            safe_dict, interv = safety.filter_trajectories({0: proposed_pos}, {0: curr_pos}, enable_filter=True)
            filtered_pos = safe_dict[0]
            actual_disp = filtered_pos - curr_pos
            curr_vel = actual_disp / delta_t
            curr_pos = filtered_pos
            path_coords.append(curr_pos.tolist())

            min_c = compute_min_clearance(curr_pos, use_obstacles, obstacle_margin)

            obs_vec = build_single_uav_obs(
                pos=curr_pos,
                vel=curr_vel,
                station_pos=target,
                edge_pos=np.array(edge_server_pos, dtype=np.float32),
                obstacles=use_obstacles,
                v_max=v_max,
                airspace_bound=use_bounds[1]
            )

            steps_telemetry.append({
                "step": s + 1,
                "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                "speed": round(float(np.linalg.norm(curr_vel)), 2),
                "dist_to_target": round(float(np.linalg.norm(curr_pos - target)), 2),
                "min_clearance": round(min_c, 2),
                "safety_intervened": bool(interv > 0),
                "explanation": {
                    "planner": "3D SAPF Baseline",
                    "decision_type": "Potential Field (Attraction + Nonlinear Repulsion + Tangential Circulation)",
                    "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                    "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                    "neural_action": [round(float(f_net[0]), 2), round(float(f_net[1]), 2), round(float(f_net[2]), 2)],
                    "executed_control": [round(float(curr_vel[0]), 2), round(float(curr_vel[1]), 2), round(float(curr_vel[2]), 2)],
                    "safety_filter_status": "SAFETY FILTER ACTIVE" if interv > 0 else "Clear flight"
                }
            })

            if float(np.linalg.norm(curr_pos - target)) <= reach_tolerance:
                status = "GOAL_REACHED"
                break

        t_plan = (time.time() - t0) * 1000.0
        p_arr = np.array(path_coords)
        total_len = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
        all_clearances = [compute_min_clearance(p, use_obstacles, obstacle_margin) for p in p_arr]
        min_overall_clearance = float(min(all_clearances)) if all_clearances else 0.0

        return {
            "status": status,
            "planner": "sapf",
            "path": path_coords,
            "explored_nodes": [],
            "total_path_length": round(total_len, 2),
            "planning_time_ms": round(t_plan, 2),
            "min_clearance": round(min_overall_clearance, 2),
            "steps_count": len(path_coords) - 1,
            "steps_telemetry": steps_telemetry
        }
