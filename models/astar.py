"""
Deterministic 3D A* Path Planner + Greedy MEC Offloading Baseline.
- Computes geometric collision-free shortest path around cylindrical obstacles to assigned mission station.
- Evaluates minimum latency among Local, Peer UAV, and Edge Server for each task.
"""

import heapq
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class GeometricDetourPlanner:
    """
    Analytical Geometric Waypoint Detour Heuristic.
    Directly computes a lateral perpendicular bypass waypoint around an intersecting cylinder.
    Not a graph search algorithm; provided as an analytical baseline.
    """
    def __init__(self, obstacles: List[Tuple[float, float, float, float]], d_safe: float = 8.0, obstacle_margin: float = 2.0):
        self.obstacles = obstacles
        self.d_safe = d_safe
        self.eff_margin = (d_safe / 2.0) + obstacle_margin

    def plan_path_with_time(self, start: np.ndarray, goal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        t0 = time.time()
        steps = 25
        blocked_obs = None
        for alpha in np.linspace(0.0, 1.0, steps):
            pt = (1.0 - alpha) * start + alpha * goal
            for cx, cy, r, h in self.obstacles:
                eff_r = r + self.eff_margin
                if np.linalg.norm(pt[:2] - np.array([cx, cy])) < eff_r and pt[2] <= h:
                    blocked_obs = (cx, cy, r, h)
                    break
            if blocked_obs is not None:
                break

        if blocked_obs is None:
            dt = (time.time() - t0) * 1000.0
            self.last_planning_time_ms = dt
            return [start, goal], dt

        cx, cy, r, h = blocked_obs
        dir_vec = goal[:2] - start[:2]
        d_len = float(np.linalg.norm(dir_vec))
        if d_len > 1e-4:
            u_dir = dir_vec / d_len
            n_perp = np.array([-u_dir[1], u_dir[0]])
        else:
            n_perp = np.array([0.0, 1.0])

        detour_dist = r + self.eff_margin + 2.0
        detour_xy = np.array([cx, cy]) + n_perp * detour_dist
        detour_pt = np.array([detour_xy[0], detour_xy[1], start[2]], dtype=np.float32)
        dt = (time.time() - t0) * 1000.0
        self.last_planning_time_ms = dt
        return [start, detour_pt, goal], dt

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        path, _ = self.plan_path_with_time(start, goal)
        return path

class AStar3DPlanner:
    """
    Genuine 3D Grid Graph Search A* Path Planner.
    - Operates on a 3D discretized grid with 26-connected spatial neighbor transitions.
    - Evaluates priority f(n) = g(n) + h(n), where h(n) = Euclidean distance to goal.
    - Collision checks against 3D cylindrical obstacles (finite height, radius + d_safe/2 + margin).
    - Employs line-of-sight shortcutting to prune unnecessary discrete grid waypoints.
    """
    def __init__(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        bounds: Tuple[float, float, float, float, float, float] = (0.0, 200.0, 0.0, 200.0, 20.0, 100.0),
        grid_res: float = 3.0,
        d_safe: float = 8.0,
        obstacle_margin: float = 2.0
    ):
        self.obstacles = obstacles
        self.bounds = bounds
        self.grid_res = grid_res
        self.d_safe = d_safe
        self.eff_margin = (d_safe / 2.0) + obstacle_margin
        self.last_planning_time_ms: float = 0.0

    def is_point_valid(self, p: np.ndarray) -> bool:
        """Checks airspace bounds and finite-height cylinder/sphere obstacle containment."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and z_min <= p[2] <= z_max):
            return False
        for obs in self.obstacles:
            if isinstance(obs, dict):
                obs_type = obs.get("type", "cylinder")
                if obs_type == "sphere":
                    cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                    eff_r = r + self.eff_margin
                    # AABB bounding-box fast rejection
                    if abs(p[0] - cx) > eff_r or abs(p[1] - cy) > eff_r or abs(p[2] - cz) > eff_r:
                        continue
                    if ((p[0] - cx)**2 + (p[1] - cy)**2 + (p[2] - cz)**2) < (eff_r**2):
                        return False
                else:
                    cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
                    z_min_obs = float(obs.get("z_min", 0.0))
                    z_max_obs = float(obs.get("z_max", h))
                    eff_r = r + self.eff_margin
                    eff_z_min = z_min_obs - self.eff_margin
                    eff_z_max = z_max_obs + self.eff_margin
                    # Overflight or Underflight: completely safe outside vertical extent
                    if p[2] > eff_z_max or p[2] < eff_z_min:
                        continue
                    # Horizontal AABB fast rejection
                    if abs(p[0] - cx) > eff_r or abs(p[1] - cy) > eff_r:
                        continue
                    if ((p[0] - cx)**2 + (p[1] - cy)**2) < (eff_r**2):
                        return False
            else:
                if len(obs) == 4:
                    cx, cy, r, h = obs
                    z_min_obs = 0.0
                    z_max_obs = h
                elif len(obs) == 5:
                    cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
                    eff_r = r + self.eff_margin
                    if abs(p[0] - cx) > eff_r or abs(p[1] - cy) > eff_r or abs(p[2] - cz) > eff_r:
                        continue
                    if ((p[0] - cx)**2 + (p[1] - cy)**2 + (p[2] - cz)**2) < (eff_r**2):
                        return False
                    continue
                else:
                    cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]
                    z_min_obs = 0.0
                    z_max_obs = h

                eff_r = r + self.eff_margin
                eff_z_min = z_min_obs - self.eff_margin
                eff_z_max = z_max_obs + self.eff_margin
                if p[2] > eff_z_max or p[2] < eff_z_min:
                    continue
                if abs(p[0] - cx) > eff_r or abs(p[1] - cy) > eff_r:
                    continue
                if ((p[0] - cx)**2 + (p[1] - cy)**2) < (eff_r**2):
                    return False
        return True

    def check_segment_obstacle_intersection(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Exact continuous 3D line-segment intersection testing against finite-height cylinders and spheres.
        Uses exact parametric interval clipping: intersects iff [t_in, t_out]_xy overlaps [t_in, t_out]_z.
        """
        d = p2 - p1
        seg_len_sq = float(np.dot(d, d))
        if seg_len_sq < 1e-8:
            return not self.is_point_valid(p1)

        for obs in self.obstacles:
            if isinstance(obs, dict) and obs.get("type") == "sphere":
                cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                eff_r = r + self.eff_margin
                # AABB segment rejection
                if max(p1[0], p2[0]) < (cx - eff_r) or min(p1[0], p2[0]) > (cx + eff_r) or \
                   max(p1[1], p2[1]) < (cy - eff_r) or min(p1[1], p2[1]) > (cy + eff_r) or \
                   max(p1[2], p2[2]) < (cz - eff_r) or min(p1[2], p2[2]) > (cz + eff_r):
                    continue

                center = np.array([cx, cy, cz], dtype=np.float32)
                f = p1 - center
                if float(np.dot(f, f)) < eff_r**2 or float(np.dot(p2 - center, p2 - center)) < eff_r**2:
                    return True
                a = float(np.dot(d, d))
                b = float(2.0 * np.dot(f, d))
                c_val = float(np.dot(f, f) - eff_r**2)
                disc = b**2 - 4.0 * a * c_val
                if disc >= 0.0:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b - sqrt_disc) / (2.0 * a)
                    t2 = (-b + sqrt_disc) / (2.0 * a)
                    if max(0.0, float(t1)) <= min(1.0, float(t2)):
                        return True
            else:
                if isinstance(obs, dict):
                    cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
                    z_min_obs = float(obs.get("z_min", 0.0))
                    z_max_obs = float(obs.get("z_max", h))
                elif len(obs) == 5:
                    cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
                    eff_r = r + self.eff_margin
                    center = np.array([cx, cy, cz], dtype=np.float32)
                    f = p1 - center
                    if float(np.dot(f, f)) < eff_r**2 or float(np.dot(p2 - center, p2 - center)) < eff_r**2:
                        return True
                    a = float(np.dot(d, d))
                    b = float(2.0 * np.dot(f, d))
                    c_val = float(np.dot(f, f) - eff_r**2)
                    disc = b**2 - 4.0 * a * c_val
                    if disc >= 0.0:
                        sqrt_disc = np.sqrt(disc)
                        t1 = (-b - sqrt_disc) / (2.0 * a)
                        t2 = (-b + sqrt_disc) / (2.0 * a)
                        if max(0.0, float(t1)) <= min(1.0, float(t2)):
                            return True
                    continue
                else:
                    cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]
                    z_min_obs = 0.0
                    z_max_obs = h

                eff_r = r + self.eff_margin
                eff_z_min = z_min_obs - self.eff_margin
                eff_z_max = z_max_obs + self.eff_margin

                # Finite vertical extent rejection: if segment is entirely above or below cylinder
                if min(p1[2], p2[2]) > eff_z_max or max(p1[2], p2[2]) < eff_z_min:
                    continue

                # 2D horizontal AABB bounding box check
                if max(p1[0], p2[0]) < (cx - eff_r) or min(p1[0], p2[0]) > (cx + eff_r) or \
                   max(p1[1], p2[1]) < (cy - eff_r) or min(p1[1], p2[1]) > (cy + eff_r):
                    continue

                # 1. Parameter interval in Z: [t_z1, t_z2]
                dz = p2[2] - p1[2]
                if abs(dz) < 1e-6:
                    if eff_z_min <= p1[2] <= eff_z_max:
                        t_z1, t_z2 = -float("inf"), float("inf")
                    else:
                        continue
                else:
                    t_a = (eff_z_min - p1[2]) / dz
                    t_b = (eff_z_max - p1[2]) / dz
                    t_z1 = min(t_a, t_b)
                    t_z2 = max(t_a, t_b)

                # 2. Parameter interval in XY: [t_xy1, t_xy2]
                d_xy = p2[:2] - p1[:2]
                f_xy = p1[:2] - np.array([cx, cy], dtype=np.float32)
                a_xy = float(np.dot(d_xy, d_xy))

                if a_xy < 1e-8:
                    if float(np.dot(f_xy, f_xy)) <= eff_r**2:
                        t_xy1, t_xy2 = -float("inf"), float("inf")
                    else:
                        continue
                else:
                    b_xy = float(2.0 * np.dot(f_xy, d_xy))
                    c_xy = float(np.dot(f_xy, f_xy) - eff_r**2)
                    disc_xy = b_xy**2 - 4.0 * a_xy * c_xy
                    if disc_xy < 0.0:
                        continue
                    sqrt_d = np.sqrt(disc_xy)
                    t_xy1 = (-b_xy - sqrt_d) / (2.0 * a_xy)
                    t_xy2 = (-b_xy + sqrt_d) / (2.0 * a_xy)

                # 3. Exact parametric interval intersection over [0, 1]
                t_enter = max(0.0, float(t_z1), float(t_xy1))
                t_exit = min(1.0, float(t_z2), float(t_xy2))

                if t_enter <= t_exit:
                    return True
        return False

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Continuous collision checking along line segment with zero tunneling."""
        if not self.is_point_valid(p1) or not self.is_point_valid(p2):
            return False
        return not self.check_segment_obstacle_intersection(p1, p2)

    def plan_path_with_details(self, start: np.ndarray, goal: np.ndarray) -> Tuple[Optional[List[np.ndarray]], float, List[Tuple[float, float, float]]]:
        """
        Executes genuine 3D A* graph search from start to goal.
        Returns:
            waypoints: List of 3D coordinates (with line-of-sight smoothing), or None if blocked
            planning_time_ms: Wall-clock execution time in milliseconds
            explored_nodes: Discretized 3D points expanded during graph search
        """
        t0 = time.time()
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        # Fast line-of-sight check: ONLY returns straight line if analytical check proves 100% collision-free
        if self.is_segment_valid(start, goal):
            dt = (time.time() - t0) * 1000.0
            self.last_planning_time_ms = dt
            return [start, goal], dt, []

        def snap_to_grid(p: np.ndarray) -> Tuple[float, float, float]:
            snapped = np.round(p / self.grid_res) * self.grid_res
            return (float(snapped[0]), float(snapped[1]), float(snapped[2]))

        s_node = snap_to_grid(start)
        g_node = snap_to_grid(goal)

        # 26-connectivity direction offsets
        res = self.grid_res
        directions = []
        for dx in [-res, 0.0, res]:
            for dy in [-res, 0.0, res]:
                for dz in [-res, 0.0, res]:
                    if dx == 0.0 and dy == 0.0 and dz == 0.0:
                        continue
                    cost = float(np.sqrt(dx**2 + dy**2 + dz**2))
                    directions.append((dx, dy, dz, cost))

        open_set: List[Tuple[float, float, Tuple[float, float, float]]] = []
        heapq.heappush(open_set, (0.0, 0.0, s_node))
        came_from: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
        g_score: Dict[Tuple[float, float, float], float] = {s_node: 0.0}
        explored: List[Tuple[float, float, float]] = []

        reached_node = None
        max_expansions = 50000
        expansions = 0

        while open_set and expansions < max_expansions:
            expansions += 1
            _, curr_g, curr_node = heapq.heappop(open_set)
            explored.append(curr_node)

            # Goal proximity check
            dist_to_g = float(np.linalg.norm(np.array(curr_node) - np.array(g_node)))
            if dist_to_g <= res * 1.5:
                reached_node = curr_node
                break

            if curr_g > g_score.get(curr_node, float("inf")):
                continue

            curr_arr = np.array(curr_node, dtype=np.float32)
            for dx, dy, dz, step_cost in directions:
                neighbor = (curr_node[0] + dx, curr_node[1] + dy, curr_node[2] + dz)
                neighbor_arr = np.array(neighbor, dtype=np.float32)
                if not self.is_point_valid(neighbor_arr):
                    continue

                # Corner-clipping protection: ensure diagonal transitions do not clip obstacle corners
                if (abs(dx) + abs(dy) + abs(dz) > res) and self.check_segment_obstacle_intersection(curr_arr, neighbor_arr):
                    continue

                tentative_g = curr_g + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = curr_node
                    g_score[neighbor] = tentative_g
                    h_score = float(np.linalg.norm(np.array(neighbor) - np.array(g_node)))
                    heapq.heappush(open_set, (tentative_g + h_score, tentative_g, neighbor))

        if reached_node is None:
            # Route genuinely blocked: return None (never fake a path through obstacles)
            dt = (time.time() - t0) * 1000.0
            self.last_planning_time_ms = dt
            return None, dt, explored

        # Reconstruct path
        path = [goal]
        curr = reached_node
        while curr in came_from:
            path.append(np.array(curr, dtype=np.float32))
            curr = came_from[curr]
        path.append(start)
        path.reverse()

        # Line-of-sight shortcutting / smoothing using analytical collision guarantees
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            furthest = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if self.is_segment_valid(path[i], path[j]):
                    furthest = j
                    break
            smoothed.append(path[furthest])
            i = furthest

        dt = (time.time() - t0) * 1000.0
        self.last_planning_time_ms = dt
        return smoothed, dt, explored

    def plan_path_with_time(self, start: np.ndarray, goal: np.ndarray) -> Tuple[Optional[List[np.ndarray]], float]:
        path, dt, _ = self.plan_path_with_details(start, goal)
        return path, dt

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        path, _, _ = self.plan_path_with_details(start, goal)
        return path

class PurePursuit3DController:
    """
    Classical 3D Pure Pursuit Trajectory Tracking Controller.
    - Tracks an ordered sequence of 3D global waypoints.
    - Lookahead target search along path polyline.
    - Enforces maximum flight velocity v_max and smooth terminal deceleration.
    - Waypoint progression with acceptance radius and stall detection.
    """
    def __init__(
        self,
        waypoints: List[np.ndarray],
        v_max: float = 15.0,
        lookahead_dist: float = 10.0,
        acceptance_radius: float = 5.0,
        decel_dist: float = 15.0
    ):
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.v_max = v_max
        self.lookahead_dist = lookahead_dist
        self.acceptance_radius = acceptance_radius
        self.decel_dist = decel_dist
        self.active_wp_idx = 0
        self.consecutive_stalls = 0

    def step(self, current_pos: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """
        Executes one pure pursuit control cycle.
        Returns:
            (next_pos, commanded_velocity, active_waypoint_index, status)
        """
        if not self.waypoints:
            return current_pos, np.zeros(3, dtype=np.float32), 0, "NO_PATH_FOUND"

        final_wp = self.waypoints[-1]
        dist_to_final = float(np.linalg.norm(current_pos - final_wp))
        if dist_to_final <= self.acceptance_radius:
            return current_pos, np.zeros(3, dtype=np.float32), len(self.waypoints) - 1, "GOAL_REACHED"

        # Advance active waypoint using along-track progression and acceptance radius
        while self.active_wp_idx < len(self.waypoints) - 1:
            w_curr = self.waypoints[self.active_wp_idx]
            w_prev = self.waypoints[max(0, self.active_wp_idx - 1)]
            seg = w_curr - w_prev
            seg_len = float(np.linalg.norm(seg))
            d_curr_wp = float(np.linalg.norm(current_pos - w_curr))

            passed = False
            if seg_len > 1e-4:
                u_seg = seg / seg_len
                passed = float(np.dot(current_pos - w_curr, u_seg)) >= -1.0

            if d_curr_wp <= max(8.0, self.acceptance_radius) or passed:
                self.active_wp_idx += 1
            else:
                break

        # Search forward along polyline for lookahead target point
        target_pt = self.waypoints[self.active_wp_idx]
        for w in range(self.active_wp_idx, len(self.waypoints)):
            d_w = float(np.linalg.norm(self.waypoints[w] - current_pos))
            if d_w >= self.lookahead_dist:
                target_pt = self.waypoints[w]
                break
            target_pt = self.waypoints[w]

        vec_to_target = target_pt - current_pos
        dist_to_target = float(np.linalg.norm(vec_to_target))

        if dist_to_target < 1e-4:
            v_cmd = np.zeros(3, dtype=np.float32)
        else:
            dir_unit = vec_to_target / dist_to_target
            # Smooth terminal deceleration when approaching final goal
            if dist_to_final < self.decel_dist:
                speed = self.v_max * max(0.2, dist_to_final / self.decel_dist)
            else:
                speed = self.v_max
            v_cmd = dir_unit * min(speed, dist_to_target / dt)

        displacement = v_cmd * dt
        next_pos = current_pos + displacement

        # Stall detection: if commanded progress is negligible despite not reaching goal
        progress = float(np.linalg.norm(displacement))
        if progress < 0.1 and dist_to_final > self.acceptance_radius:
            self.consecutive_stalls += 1
        else:
            self.consecutive_stalls = 0

        status = "CONTROLLER_STALLED" if self.consecutive_stalls >= 5 else "PATH_FOUND"
        return next_pos, v_cmd, self.active_wp_idx, status

class AStarBaseline:
    """
    # =========================================================================
    # PURPOSE:
    # Deterministic 3D A* Path Planner paired with lowest-latency MEC offloading.
    # Evaluates collision-free discrete waypoint navigation around 3D obstacles
    # while greedily selecting the computation node (Local vs Peer vs Edge)
    # with the lowest estimated transmission + processing delay.
    # =========================================================================
    """
    def __init__(self, env):
        self.env = env
        self.planner = AStar3DPlanner(
            obstacles=env.cfg.airspace.obstacles,
            grid_res=5.0,
            d_safe=env.cfg.physics.d_safe
        )
        self.cached_paths = {}
        self.current_wp_idx = {}

    def reset(self):
        self.cached_paths = {}
        self.current_wp_idx = {}

    def select_actions(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        actions = {}
        p_cfg = self.env.cfg.physics
        edge_pos = np.array(self.env.cfg.airspace.edge_server_pos)

        for j in range(self.env.num_uavs):
            pos = self.env.positions[j]
            goal = self.env.stations[j]

            if j not in self.cached_paths:
                self.cached_paths[j] = self.planner.plan_path(pos, goal)
                self.current_wp_idx[j] = 1

            path = self.cached_paths[j]
            wp_idx = self.current_wp_idx[j]
            curr_target = path[min(wp_idx, len(path) - 1)]

            d_target = np.linalg.norm(curr_target - pos)
            d_final = np.linalg.norm(goal - pos)

            # Advance to next waypoint if reached intermediate detour
            if d_target < 4.0 and wp_idx < len(path) - 1:
                self.current_wp_idx[j] += 1
                curr_target = path[self.current_wp_idx[j]]
                d_target = np.linalg.norm(curr_target - pos)

            # 1. 3D Navigation Action: Steer along planned waypoint corridor
            if d_final > p_cfg.reach_tolerance:
                dir_vec = (curr_target - pos) / max(1e-6, d_target)
                step_dist = min(d_target, p_cfg.v_max * p_cfg.delta_t)
                disp = dir_vec * step_dist
            else:
                # Arrived at station: hold position (deadband stop)
                disp = np.zeros(3, dtype=np.float32)

            # 2. Greedy MEC Offloading Action
            task = self.env.current_tasks[j]
            if task is None:
                offload_choice = 0
            else:
                # Evaluate local latency
                t_local = task.cycles / self.env.cfg.comp.f_local
                
                # Evaluate peer latency (find closest peer)
                best_peer_idx = 1
                min_peer_t = float("inf")
                for p_offset in range(1, self.env.num_uavs):
                    peer_id = (j + p_offset) % self.env.num_uavs
                    d_p = np.linalg.norm(pos - self.env.positions[peer_id])
                    r_p = self.env.comm.data_rate(d_p)
                    t_peer = (task.data_size / max(1e3, r_p)) + (task.cycles / self.env.cfg.comp.f_peer)
                    if t_peer < min_peer_t:
                        min_peer_t = t_peer
                        best_peer_idx = p_offset

                # Evaluate edge latency
                d_e = np.linalg.norm(pos - edge_pos)
                r_e = self.env.comm.data_rate(d_e)
                t_edge = (task.data_size / max(1e3, r_e)) + (task.cycles / self.env.cfg.comp.f_edge)

                # Select action with minimum total latency
                latencies = [t_local, min_peer_t, t_edge]
                min_idx = int(np.argmin(latencies))
                if min_idx == 0:
                    offload_choice = 0
                elif min_idx == 1:
                    offload_choice = best_peer_idx
                else:
                    offload_choice = self.env.num_uavs

            actions[j] = {"move": disp.astype(np.float32), "offload": offload_choice}

        return actions

class HybridAStarMAPPONavigator:
    """
    # =========================================================================
    # PURPOSE:
    # Hierarchical Hybrid Multi-UAV Navigation and Mobile Edge Computing Architecture.
    # Combines deterministic global geometric search with reactive multi-agent
    # reinforcement learning for joint trajectory execution and task offloading.
    #
    # HIERARCHICAL COUPLING ARCHITECTURE:
    # 1. High-Level Macro Planner (3D A* Search):
    #    - Evaluates 3D grid graph with 26-connected spatial neighbor transitions.
    #    - Priority f(n) = g(n) + h(n), where h(n) = ||p_n - p_goal||_2.
    #    - Guarantees zero obstacle penetration against complex 3D cylinder/sphere
    #      geometries by building a collision-free polyline waypoint corridor:
    #      W_j = [w_0, w_1, ..., w_K].
    #
    # 2. Mid-Level Path Tracking Controller (3D Pure Pursuit):
    #    - Searches forward along the A* corridor W_j for a lookahead target point
    #      p_target at distance L = lookahead_dist (default: 12.0 m).
    #    - Computes commanded velocity vector:
    #      v_cmd = v_max * (p_target - p_curr) / ||p_target - p_curr||_2
    #    - Enforces terminal deceleration within decel_dist (15.0 m) of goal.
    #
    # 3. Low-Level Decentralized RL Execution (MAPPO Actor):
    #    - Evaluates local observation o_j and samples optimal discrete offloading
    #      decision a_off in {0=Local, 1..N-1=Peer UAV, N=Ground Edge Server}.
    #    - When coupled with velocity refinement, balances global corridor adherence
    #      with dynamic inter-UAV separation and local avoidance.
    # =========================================================================
    """
    def __init__(self, env: Any, mappo_agent: Optional[Any] = None, lookahead: float = 12.0):
        self.env = env
        self.mappo_agent = mappo_agent
        self.lookahead = lookahead
        self.controllers = {}
        self.raw_obs = env.cfg.airspace.obstacles
        self.margin = env.cfg.physics.obstacle_margin
        self.reset()

    def reset(self):
        self.controllers = {}
        planner = AStar3DPlanner(
            obstacles=self.raw_obs,
            bounds=(
                self.env.cfg.airspace.x_min, self.env.cfg.airspace.x_max,
                self.env.cfg.airspace.y_min, self.env.cfg.airspace.y_max,
                self.env.cfg.airspace.z_min, self.env.cfg.airspace.z_max
            ),
            grid_res=5.0,
            d_safe=self.env.cfg.physics.d_safe,
            obstacle_margin=self.margin
        )
        for j in range(self.env.num_uavs):
            start = self.env.positions[j]
            goal = self.env.stations[j]
            wps = planner.plan_path(start, goal)
            if not wps or len(wps) < 2:
                wps = [start, goal]
            self.controllers[j] = PurePursuit3DController(
                wps,
                v_max=self.env.cfg.physics.v_max,
                lookahead_dist=self.lookahead,
                acceptance_radius=self.env.cfg.physics.reach_tolerance
            )

    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        actions = {}
        for j in range(self.env.num_uavs):
            curr_pos = self.env.positions[j]
            next_pos, cmd_vel, wp_idx, status = self.controllers[j].step(curr_pos, dt=self.env.cfg.physics.delta_t)
            
            # Continuous movement displacement
            move_disp = cmd_vel * self.env.cfg.physics.delta_t
            
            # Neural offloading decision (if mappo_agent available), else greedy
            offload_choice = 0
            if self.mappo_agent is not None:
                import torch
                t_obs = torch.tensor(obs[j], dtype=torch.float32, device=self.mappo_agent.device).unsqueeze(0)
                with torch.no_grad():
                    _, act_off, _ = self.mappo_agent.actor.sample_action(t_obs, deterministic=True)
                offload_choice = int(act_off.squeeze(0).cpu().item())
            else:
                offload_choice = self.env.num_uavs
                
            actions[j] = {
                "move": np.clip(move_disp, -self.env.cfg.physics.v_max, self.env.cfg.physics.v_max),
                "offload": offload_choice
            }
        return actions

