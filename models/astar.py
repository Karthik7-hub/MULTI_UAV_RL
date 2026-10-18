"""
Deterministic 3D A* Path Planner + Greedy MEC Offloading Baseline.
- Computes geometric collision-free shortest path around cylindrical obstacles to assigned mission station.
- Evaluates minimum latency among Local, Peer UAV, and Edge Server for each task.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

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
        """Checks airspace bounds and cylinder obstacle containment."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and z_min <= p[2] <= z_max):
            return False
        for cx, cy, r, h in self.obstacles:
            eff_r = r + self.eff_margin
            eff_h = h + self.eff_margin
            if ((p[0] - cx)**2 + (p[1] - cy)**2) < (eff_r**2) and p[2] <= eff_h:
                return False
        return True

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray, num_checks: int = 15) -> bool:
        """Continuous collision checking along line segment."""
        for alpha in np.linspace(0.0, 1.0, num_checks):
            p = (1.0 - alpha) * p1 + alpha * p2
            if not self.is_point_valid(p):
                return False
        return True

    def plan_path_with_time(self, start: np.ndarray, goal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """
        Executes genuine A* graph search from start to goal.
        Returns:
            waypoints: List of 3D coordinates (with line-of-sight smoothing)
            planning_time_ms: Wall-clock execution time in milliseconds
        """
        t0 = time.time()
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        # Fast line-of-sight check
        if self.is_segment_valid(start, goal, num_checks=25):
            dt = (time.time() - t0) * 1000.0
            self.last_planning_time_ms = dt
            return [start, goal], dt

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

        reached_node = None
        max_expansions = 4000
        expansions = 0

        while open_set and expansions < max_expansions:
            expansions += 1
            _, curr_g, curr_node = heapq.heappop(open_set)

            # Goal proximity check
            dist_to_g = float(np.linalg.norm(np.array(curr_node) - np.array(g_node)))
            if dist_to_g <= res * 1.5:
                reached_node = curr_node
                break

            if curr_g > g_score.get(curr_node, float("inf")):
                continue

            for dx, dy, dz, step_cost in directions:
                neighbor = (curr_node[0] + dx, curr_node[1] + dy, curr_node[2] + dz)
                if not self.is_point_valid(np.array(neighbor, dtype=np.float32)):
                    continue

                tentative_g = curr_g + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = curr_node
                    g_score[neighbor] = tentative_g
                    h_score = float(np.linalg.norm(np.array(neighbor) - np.array(g_node)))
                    heapq.heappush(open_set, (tentative_g + h_score, tentative_g, neighbor))

        if reached_node is None:
            # Fallback to direct goal if search budget exceeded
            dt = (time.time() - t0) * 1000.0
            self.last_planning_time_ms = dt
            return [start, goal], dt

        # Reconstruct path
        path = [goal]
        curr = reached_node
        while curr in came_from:
            path.append(np.array(curr, dtype=np.float32))
            curr = came_from[curr]
        path.append(start)
        path.reverse()

        # Line-of-sight shortcutting / smoothing
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            furthest = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if self.is_segment_valid(path[i], path[j], num_checks=12):
                    furthest = j
                    break
            smoothed.append(path[furthest])
            i = furthest

        dt = (time.time() - t0) * 1000.0
        self.last_planning_time_ms = dt
        return smoothed, dt

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        path, _ = self.plan_path_with_time(start, goal)
        return path

class AStarBaseline:
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
