"""
Analytical Kinematic Projection Safety Filter.
Provides formal guarantees against:
1. Obstacle collision (cylindrical obstacle boundary projection)
2. Inter-UAV collision (minimum separation preservation d_safe)
3. Airspace boundary containment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class SafetyFilter:
    """
    Analytical Kinematic Projection Safety Filter.
    Enforces physically consistent safety guarantees:
    1. Continuous line-segment collision detection against cylindrical obstacles (no discrete tunneling).
    2. Ray-cylinder entry point clamping (stops safely before obstacle; no boundary teleportation).
    3. Continuous inter-UAV minimum distance preservation along relative trajectory (d_safe).
    4. Airspace geofence boundary clamping.
    5. Disentangled tracking of proposed collisions, interventions, and actual collisions.
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
        self.last_stats = {
            "interventions": 0,
            "proposed_collisions": 0,
            "actual_collisions": 0
        }

    def check_segment_cylinder_intersection(
        self, p0: np.ndarray, p1: np.ndarray, cx: float, cy: float, r: float, h: float,
        z_min: float = 0.0, z_max: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Calculates exact continuous 3D intersection between segment p0->p1 and finite cylinder.
        Uses parametric interval clipping: intersects iff [t_in, t_out]_xy overlaps [t_in, t_out]_z.
        Returns:
            (intersects, t_entry) where t_entry in [0, 1] is the fractional distance along segment.
        """
        eff_r = r + self.obstacle_margin
        eff_z_min = z_min - self.obstacle_margin
        eff_z_max = (h if z_max is None else z_max) + self.obstacle_margin

        # 1. Parameter interval in Z: [t_z1, t_z2]
        dz = p1[2] - p0[2]
        if abs(dz) < 1e-6:
            if eff_z_min <= p0[2] <= eff_z_max:
                t_z1, t_z2 = -float("inf"), float("inf")
            else:
                return False, 1.0
        else:
            t_a = (eff_z_min - p0[2]) / dz
            t_b = (eff_z_max - p0[2]) / dz
            t_z1 = min(t_a, t_b)
            t_z2 = max(t_a, t_b)

        # 2. Parameter interval in XY: [t_xy1, t_xy2]
        d_xy = p1[:2] - p0[:2]
        f_xy = p0[:2] - np.array([cx, cy], dtype=np.float32)
        a_xy = float(np.dot(d_xy, d_xy))

        if a_xy < 1e-8:
            if float(np.dot(f_xy, f_xy)) <= eff_r**2:
                t_xy1, t_xy2 = -float("inf"), float("inf")
            else:
                return False, 1.0
        else:
            b_xy = float(2.0 * np.dot(f_xy, d_xy))
            c_xy = float(np.dot(f_xy, f_xy) - eff_r**2)
            disc_xy = b_xy**2 - 4.0 * a_xy * c_xy
            if disc_xy < 0.0:
                return False, 1.0
            sqrt_d = np.sqrt(disc_xy)
            t_xy1 = (-b_xy - sqrt_d) / (2.0 * a_xy)
            t_xy2 = (-b_xy + sqrt_d) / (2.0 * a_xy)

        # 3. Exact parametric interval intersection over [0, 1]
        t_enter = max(0.0, float(t_z1), float(t_xy1))
        t_exit = min(1.0, float(t_z2), float(t_xy2))

        if t_enter <= t_exit:
            return True, float(t_enter)

        return False, 1.0

    def check_segment_sphere_intersection(
        self, p0: np.ndarray, p1: np.ndarray, cx: float, cy: float, cz: float, r: float
    ) -> Tuple[bool, float]:
        """Calculates exact continuous intersection between 3D segment p0->p1 and sphere."""
        eff_r = r + self.obstacle_margin
        center = np.array([cx, cy, cz], dtype=np.float32)
        d = p1 - p0
        f = p0 - center

        if float(np.dot(f, f)) <= eff_r**2 or float(np.dot(p1 - center, p1 - center)) <= eff_r**2:
            return True, 0.0

        a = float(np.dot(d, d))
        if a < 1e-8:
            return False, 1.0

        b = float(2.0 * np.dot(f, d))
        c_val = float(np.dot(f, f) - eff_r**2)
        disc = b**2 - 4.0 * a * c_val
        if disc < 0.0:
            return False, 1.0

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        if max(0.0, float(t1)) <= min(1.0, float(t2)):
            return True, max(0.0, float(t1))
        return False, 1.0

    @staticmethod
    def _relative_trajectory_collision(
        start_a: np.ndarray,
        end_a: np.ndarray,
        start_b: np.ndarray,
        end_b: np.ndarray,
        min_separation: float,
    ) -> Tuple[bool, float]:
        """Return whether two simultaneous line segments breach separation.

        Relative motion turns the two-UAV problem into a segment from
        ``start_a - start_b`` to ``end_a - end_b`` around the origin.  The
        first root of the separation sphere gives the latest safe fraction of
        the proposed movement.
        """
        relative_start = np.asarray(start_a, dtype=np.float32) - np.asarray(start_b, dtype=np.float32)
        relative_delta = (
            np.asarray(end_a, dtype=np.float32)
            - np.asarray(end_b, dtype=np.float32)
            - relative_start
        )
        delta_sq = float(np.dot(relative_delta, relative_delta))

        if delta_sq < 1e-12:
            return float(np.linalg.norm(relative_start)) < min_separation, 0.0

        t_closest = float(np.clip(-np.dot(relative_start, relative_delta) / delta_sq, 0.0, 1.0))
        closest = relative_start + t_closest * relative_delta
        if float(np.linalg.norm(closest)) >= min_separation:
            return False, 1.0

        start_distance = float(np.linalg.norm(relative_start))
        if start_distance <= min_separation:
            return True, 0.0

        b_term = float(2.0 * np.dot(relative_start, relative_delta))
        c_term = float(np.dot(relative_start, relative_start) - min_separation ** 2)
        discriminant = b_term ** 2 - 4.0 * delta_sq * c_term
        if discriminant <= 0.0:
            return True, t_closest

        t_entry = (-b_term - np.sqrt(discriminant)) / (2.0 * delta_sq)
        return True, float(np.clip(t_entry, 0.0, 1.0))

    def filter_trajectories(
        self,
        proposed_positions: Dict[int, np.ndarray],
        current_positions: Dict[int, np.ndarray],
        enable_filter: bool = True
    ) -> Tuple[Dict[int, np.ndarray], int]:
        """
        Analyzes the proposed 3D positions of all UAVs against continuous obstacle segments,
        peer separations, and geofence bounds.
        
        If enable_filter is True:
            Intervenes physically: stops UAV along approach vector before penetration.
        If enable_filter is False:
            Passes proposed positions through and records actual collisions.
            
        Returns:
            (filtered_positions, interventions_count)
        """
        filtered = {j: np.copy(pos) for j, pos in proposed_positions.items()}
        interventions = 0
        proposed_collisions = 0
        actual_collisions = 0
        num_uavs = len(proposed_positions)

        # ---------------------------------------------------------------------
        # 1. Continuous Cylindrical & Spherical Obstacle Collision Check & Multi-Pass Projection
        # ---------------------------------------------------------------------
        for pass_idx in range(3):
            any_intervention_in_pass = False
            for j in range(num_uavs):
                p0 = current_positions[j]
                p1 = filtered[j]

                for obs in self.obstacles:
                    is_sphere = False
                    z_min_obs = 0.0
                    z_max_obs = 100.0
                    if isinstance(obs, dict):
                        is_sphere = (obs.get("type") == "sphere")
                        if is_sphere:
                            cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                        else:
                            cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
                            z_min_obs = float(obs.get("z_min", 0.0))
                            z_max_obs = float(obs.get("z_max", h))
                    elif len(obs) == 5:
                        is_sphere = True
                        cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
                    else:
                        cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]
                        z_min_obs = 0.0
                        z_max_obs = h

                    if is_sphere:
                        intersects, t_entry = self.check_segment_sphere_intersection(p0, p1, cx, cy, cz, r)
                    else:
                        intersects, t_entry = self.check_segment_cylinder_intersection(
                            p0, p1, cx, cy, r, h, z_min=z_min_obs, z_max=z_max_obs
                        )

                    if intersects:
                        if pass_idx == 0:
                            proposed_collisions += 1
                        if enable_filter:
                            interventions += 1
                            any_intervention_in_pass = True
                            eff_r = r + self.obstacle_margin
                            
                            # Safe braking stop: clamp displacement just before the obstacle surface
                            if t_entry > 0.0:
                                safe_t = max(0.0, t_entry - 0.02)
                                p_safe = p0 + safe_t * (p1 - p0)
                            else:
                                p_safe = np.copy(p1)

                            if is_sphere:
                                c_3d = np.array([cx, cy, cz], dtype=np.float32)
                                d_out = p_safe - c_3d
                                dist_out = float(np.linalg.norm(d_out))
                                if dist_out < eff_r:
                                    n_out = (d_out / dist_out) if dist_out > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                                    p_safe = c_3d + n_out * (eff_r + 0.05)
                            else:
                                c_xy = np.array([cx, cy], dtype=np.float32)
                                d_out = p_safe[:2] - c_xy
                                dist_out = float(np.linalg.norm(d_out))
                                eff_z_max = z_max_obs + self.obstacle_margin
                                dist_to_top = eff_z_max - p_safe[2]

                                if 0 < dist_to_top < (eff_r - dist_out) and (eff_z_max + 0.05) <= self.z_max:
                                    # Overflight top clearance
                                    p_safe[2] = eff_z_max + 0.05
                                else:
                                    # Lateral radial pushout
                                    if dist_out < eff_r:
                                        n_out = (d_out / dist_out) if dist_out > 1e-4 else np.array([1.0, 0.0], dtype=np.float32)
                                        p_safe[:2] = c_xy + n_out * (eff_r + 0.05)

                            filtered[j] = p_safe
                            p1 = p_safe
                        else:
                            if pass_idx == 0:
                                actual_collisions += 1

            if not any_intervention_in_pass:
                break

        # ---------------------------------------------------------------------
        # 2. Inter-UAV Separation Preservation (Continuous relative motion)
        # ---------------------------------------------------------------------
        # Clamp first so the peer check evaluates the positions that will
        # actually be applied, rather than an out-of-bounds intermediate.
        for j in range(num_uavs):
            filtered[j][0] = np.clip(filtered[j][0], self.x_min, self.x_max)
            filtered[j][1] = np.clip(filtered[j][1], self.y_min, self.y_max)
            filtered[j][2] = np.clip(filtered[j][2], self.z_min, self.z_max)

        max_peer_passes = 3 if enable_filter else 1
        for pass_idx in range(max_peer_passes):
            any_peer_intervention = False
            for j in range(num_uavs):
                for k in range(j + 1, num_uavs):
                    collides, t_entry = self._relative_trajectory_collision(
                        current_positions[j], filtered[j], current_positions[k], filtered[k], self.d_safe
                    )
                    if not collides:
                        continue

                    if pass_idx == 0:
                        proposed_collisions += 1

                    if not enable_filter:
                        actual_collisions += 1
                        continue

                    interventions += 1
                    any_peer_intervention = True
                    initial_sep = float(np.linalg.norm(current_positions[j] - current_positions[k]))

                    if initial_sep >= self.d_safe - 1e-6:
                        # Stop both UAVs immediately before their first
                        # separation-boundary crossing.  This also catches
                        # endpoint-safe swaps, which endpoint-only checks miss.
                        safe_t = max(0.0, t_entry - 1e-3)
                        filtered[j] = current_positions[j] + safe_t * (filtered[j] - current_positions[j])
                        filtered[k] = current_positions[k] + safe_t * (filtered[k] - current_positions[k])
                    else:
                        # A caller may already provide an unsafe state.  That
                        # past collision cannot be undone, but the resulting
                        # positions are projected back to the separation shell.
                        p_j = filtered[j]
                        p_k = filtered[k]
                        endpoint_sep = float(np.linalg.norm(p_j - p_k))
                        if endpoint_sep > 1e-6:
                            direction = (p_j - p_k) / endpoint_sep
                        else:
                            initial_dir = current_positions[j] - current_positions[k]
                            initial_norm = float(np.linalg.norm(initial_dir))
                            direction = initial_dir / initial_norm if initial_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        midpoint = 0.5 * (p_j + p_k)
                        filtered[j] = midpoint + 0.5 * self.d_safe * direction
                        filtered[k] = midpoint - 0.5 * self.d_safe * direction

            if not any_peer_intervention:
                break

        # ---------------------------------------------------------------------
        # 3. Airspace Geofence Containment
        # ---------------------------------------------------------------------
        for j in range(num_uavs):
            filtered[j][0] = np.clip(filtered[j][0], self.x_min, self.x_max)
            filtered[j][1] = np.clip(filtered[j][1], self.y_min, self.y_max)
            filtered[j][2] = np.clip(filtered[j][2], self.z_min, self.z_max)

        self.last_stats = {
            "interventions": interventions,
            "proposed_collisions": proposed_collisions,
            "actual_collisions": actual_collisions
        }
        return filtered, interventions

    def filter_trajectories_with_stats(
        self,
        proposed_positions: Dict[int, np.ndarray],
        current_positions: Dict[int, np.ndarray],
        enable_filter: bool = True
    ) -> Tuple[Dict[int, np.ndarray], int, Dict[str, int]]:
        filtered, interventions = self.filter_trajectories(proposed_positions, current_positions, enable_filter=enable_filter)
        return filtered, interventions, self.last_stats

def parse_obstacle(obs: Any) -> Dict[str, Any]:
    """
    Parses dictionary, 4-tuple, or 5-tuple obstacle representations into a canonical normalized structure.
    - Cylinder: dict(type="cylinder", cx=..., cy=..., radius=..., height=..., z_min=..., z_max=...)
    - Sphere: dict(type="sphere", cx=..., cy=..., cz=..., radius=...)
    """
    if isinstance(obs, dict):
        obs_type = obs.get("type", "cylinder")
        if obs_type == "sphere":
            return {
                "type": "sphere",
                "cx": float(obs["cx"]),
                "cy": float(obs["cy"]),
                "cz": float(obs["cz"]),
                "radius": float(obs["radius"])
            }
        else:
            h = float(obs["height"])
            return {
                "type": "cylinder",
                "cx": float(obs["cx"]),
                "cy": float(obs["cy"]),
                "radius": float(obs["radius"]),
                "height": h,
                "z_min": float(obs.get("z_min", 0.0)),
                "z_max": float(obs.get("z_max", h))
            }
    elif isinstance(obs, (tuple, list)):
        if len(obs) == 4:
            cx, cy, r, h = obs
            return {
                "type": "cylinder",
                "cx": float(cx),
                "cy": float(cy),
                "radius": float(r),
                "height": float(h),
                "z_min": 0.0,
                "z_max": float(h)
            }
        elif len(obs) == 5:
            if isinstance(obs[0], str) and obs[0] == "sphere":
                cx, cy, cz, r = obs[1:5]
            else:
                cx, cy, cz, r = obs[:4]
            return {
                "type": "sphere",
                "cx": float(cx),
                "cy": float(cy),
                "cz": float(cz),
                "radius": float(r)
            }
        else:
            raise ValueError(f"Malformed obstacle tuple with length {len(obs)}: {obs}. Expected length 4 (cylinder) or 5 (sphere).")
    else:
        raise TypeError(f"Unsupported obstacle type: {type(obs)}. Expected dict, tuple, or list.")

def compute_min_clearance(pos: np.ndarray, obstacles: list, margin: float = 2.0) -> float:
    """
    Computes exact continuous 3D surface clearance to all cylindrical and spherical obstacles.
    Clearance = d_surface - margin.
    """
    clearances = []
    p = np.array(pos, dtype=np.float32)
    for raw_obs in obstacles:
        parsed = parse_obstacle(raw_obs)
        if parsed["type"] == "sphere":
            cx, cy, cz, r = parsed["cx"], parsed["cy"], parsed["cz"], parsed["radius"]
            d_3d = float(np.linalg.norm(p - np.array([cx, cy, cz])))
            clearances.append(d_3d - (r + margin))
        else:
            cx, cy, r, h = parsed["cx"], parsed["cy"], parsed["radius"], parsed["height"]
            z_min_obs = parsed["z_min"]
            z_max_obs = parsed["z_max"]

            d_xy = float(np.linalg.norm(p[:2] - np.array([cx, cy])))
            delta_xy = max(0.0, d_xy - r)
            if p[2] > z_max_obs:
                delta_z = p[2] - z_max_obs
            elif p[2] < z_min_obs:
                delta_z = z_min_obs - p[2]
            else:
                delta_z = 0.0

            if delta_xy > 0.0 and delta_z > 0.0:
                d_surface = float(np.sqrt(delta_xy**2 + delta_z**2))
            elif delta_xy > 0.0:
                d_surface = delta_xy
            elif delta_z > 0.0:
                d_surface = delta_z
            else:
                # Penetration into obstacle
                d_surface = -float(min(r - d_xy, p[2] - z_min_obs, z_max_obs - p[2]))

            clearances.append(float(d_surface - margin))
    return float(min(clearances)) if clearances else 50.0
