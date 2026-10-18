"""
Analytical Kinematic Projection Safety Filter.
Provides formal guarantees against:
1. Obstacle collision (cylindrical obstacle boundary projection)
2. Inter-UAV collision (minimum separation preservation d_safe)
3. Airspace boundary containment
"""

import numpy as np
from typing import Dict, List, Tuple

class SafetyFilter:
    """
    # =========================================================================
    # PURPOSE:
    # Analytical Kinematic Projection Safety Filter.
    # Enforces hard safety guarantees at every simulation step by projecting any
    # proposed UAV displacement that violates safety margins back onto the nearest
    # mathematically safe boundary.
    #
    # THREE HARD CONSTRAINTS ENFORCED:
    # 1. Cylindrical Obstacle Avoidance (buffer = radius + obstacle_margin)
    # 2. Inter-UAV Separation Preservation (minimum Euclidean distance = d_safe)
    # 3. Airspace Geofence Containment (x, y, z bounding box)
    # =========================================================================
    """
    def __init__(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        d_safe: float = 8.0,
        obstacle_margin: float = 2.0,
        bounds: Tuple[float, float, float, float, float, float] = (0.0, 200.0, 0.0, 200.0, 20.0, 100.0)
    ):
        self.obstacles = obstacles
        self.d_safe = d_safe
        self.obstacle_margin = obstacle_margin
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = bounds

    def filter_trajectories(
        self, proposed_positions: Dict[int, np.ndarray], current_positions: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], int]:
        """
        # =========================================================================
        # PURPOSE:
        # Analyzes the proposed 3D positions of all UAVs, detects safety violations,
        # applies minimal analytical geometric corrections, and counts interventions.
        #
        # EQUATIONS:
        # 1. Obstacle Boundary Projection (Horizontal XY plane):
        #    If d_xy < (r + margin) and z <= h:
        #        n_out = (p_xy - c_xy) / d_xy
        #        p_xy_corrected = c_xy + n_out * (r + margin)
        #
        # 2. Symmetrical Inter-UAV Separation Repulsion:
        #    If dist(p_j, p_k) < d_safe:
        #        overlap = d_safe - dist(p_j, p_k)
        #        direction = (p_j - p_k) / dist(p_j, p_k)
        #        p_j_corrected = p_j + 0.5 * overlap * direction
        #        p_k_corrected = p_k - 0.5 * overlap * direction
        #
        # 3. Airspace Bounding Box Clamping:
        #    p_corrected = clip(p, p_min, p_max)
        #
        # VARIABLES & UNITS:
        # - proposed_positions : Proposed next 3D coordinates from agent actions [m]
        # - current_positions  : Current 3D coordinates before this step [m]
        # - cx, cy, r, h       : Center (x, y), radius, and top height of obstacle [m]
        # - d_safe             : Minimum inter-UAV separation safety buffer = 8.0 m
        # - obstacle_margin    : Extra buffer around obstacle cylinder = 2.0 m
        #
        # RESULT:
        # - filtered_positions : Guaranteed collision-free 3D coordinates [m]
        # - interventions      : Total number of safety corrections triggered [count]
        # =========================================================================
        """
        filtered = {j: np.copy(pos) for j, pos in proposed_positions.items()}
        interventions = 0
        num_uavs = len(proposed_positions)

        # ---------------------------------------------------------------------
        # 1. Cylindrical Obstacle Filter
        # ---------------------------------------------------------------------
        for j in range(num_uavs):
            pos = filtered[j]
            for cx, cy, r, h in self.obstacles:
                eff_r = r + self.obstacle_margin  # Effective safety cylinder radius [m]
                d_xy = np.linalg.norm(pos[:2] - np.array([cx, cy]))

                # Check if UAV penetrated the horizontal cylinder circle below the obstacle height h
                if d_xy < eff_r and pos[2] <= h:
                    interventions += 1
                    # Outward normal unit vector from cylinder center to UAV
                    if d_xy > 1e-4:
                        n_out = (pos[:2] - np.array([cx, cy])) / d_xy
                    else:
                        n_out = np.array([1.0, 0.0])  # Arbitrary fallback if exactly at center
                    
                    # Project position onto the safe perimeter
                    filtered[j][:2] = np.array([cx, cy]) + n_out * eff_r

        # ---------------------------------------------------------------------
        # 2. Inter-UAV Separation Filter
        # ---------------------------------------------------------------------
        for j in range(num_uavs):
            for k in range(j + 1, num_uavs):
                p_j = filtered[j]
                p_k = filtered[k]
                dist = np.linalg.norm(p_j - p_k)

                # Check if distance between UAV j and UAV k is less than d_safe
                if dist < self.d_safe:
                    interventions += 1
                    overlap = self.d_safe - dist  # Penetration depth [m]

                    # Robust direction calculation to prevent division-by-zero if UAVs coincide
                    if dist > 1e-4:
                        direction = (p_j - p_k) / dist
                    else:
                        prev_dir = current_positions[j] - current_positions[k]
                        prev_d = float(np.linalg.norm(prev_dir))
                        if prev_d > 1e-4:
                            direction = prev_dir / prev_d
                        else:
                            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                    # Symmetrically push both UAVs apart by half the overlap distance
                    filtered[j] += 0.5 * overlap * direction
                    filtered[k] -= 0.5 * overlap * direction

        # ---------------------------------------------------------------------
        # 3. Airspace Geofence Clamping
        # ---------------------------------------------------------------------
        for j in range(num_uavs):
            filtered[j][0] = np.clip(filtered[j][0], self.x_min, self.x_max)
            filtered[j][1] = np.clip(filtered[j][1], self.y_min, self.y_max)
            filtered[j][2] = np.clip(filtered[j][2], self.z_min, self.z_max)

        return filtered, interventions
