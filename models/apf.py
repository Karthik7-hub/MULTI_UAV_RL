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
from typing import Dict, List, Tuple, Any

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

        for cx, cy, r, h in self.obstacles:
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
    """Sequential Adaptive Potential Field Baseline with greedy MEC offloading."""
    def __init__(self, env, offload_policy=None):
        self.env = env
        self.navigator = SAPFNavigator(
            obstacles=env.cfg.airspace.obstacles,
            d_safe=env.cfg.physics.d_safe,
            obstacle_margin=env.cfg.physics.obstacle_margin,
            v_max=env.cfg.physics.v_max
        )
        from models.baselines import EnergyDeadlineAwareGreedyPolicy
        self.offload_policy = offload_policy or EnergyDeadlineAwareGreedyPolicy()
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
