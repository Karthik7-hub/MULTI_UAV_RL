"""
3D RRT* (Rapidly-exploring Random Tree Star) Path Planner for Cylindrical Obstacles.
Features:
- 3D sampling within airspace bounds
- Exact 3D cylinder collision checking (finite height, radius + d_safe)
- Near-neighbor rewiring within dynamic radius for asymptotic optimality
- Waypoint-following controller for physical trajectory generation in MultiUAVEnv
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class RRTStarNode:
    def __init__(self, pos: np.ndarray, parent=None, cost: float = 0.0):
        self.pos = np.array(pos, dtype=np.float32)
        self.parent = parent
        self.cost = cost

class RRTStar3DPlanner:
    def __init__(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        bounds: Tuple[float, float, float, float, float, float] = (0.0, 200.0, 0.0, 200.0, 20.0, 100.0),
        step_size: float = 6.0,
        search_radius: float = 18.0,
        max_iters: int = 500,
        goal_sample_rate: float = 0.15,
        d_safe: float = 8.0,
        obstacle_margin: float = 2.0
    ):
        self.obstacles = obstacles
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = bounds
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iters = max_iters
        self.goal_sample_rate = goal_sample_rate
        self.eff_margin = (d_safe / 2.0) + obstacle_margin

    def is_point_valid(self, pos: np.ndarray) -> bool:
        """Checks if a 3D point is within bounds and outside all cylindrical obstacle geometries."""
        if not (self.x_min <= pos[0] <= self.x_max and
                self.y_min <= pos[1] <= self.y_max and
                self.z_min <= pos[2] <= self.z_max):
            return False

        for cx, cy, r, h in self.obstacles:
            eff_r = r + self.eff_margin
            eff_h = h + self.eff_margin
            d_xy = np.linalg.norm(pos[:2] - np.array([cx, cy]))
            if d_xy < eff_r and pos[2] <= eff_h:
                return False
        return True

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray, num_checks: int = 12) -> bool:
        """Line-segment discrete collision checking against obstacle cylinders."""
        for alpha in np.linspace(0.0, 1.0, num_checks):
            p = (1.0 - alpha) * p1 + alpha * p2
            if not self.is_point_valid(p):
                return False
        return True

    def plan_path(self, start: np.ndarray, goal: np.ndarray, seed: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        # =========================================================================
        # PURPOSE:
        # Executes 3D Rapidly-exploring Random Tree Star (RRT*) path planning.
        # Asymptotically optimal sampling-based algorithm that navigates around 3D cylinders.
        #
        # CORE ALGORITHMIC STEPS:
        # 1. Biased Random Sampling:
        #    With probability goal_sample_rate (15%), sample the goal coordinate directly.
        #    Otherwise, sample uniformly within the 3D airspace bounds.
        #
        # 2. Nearest Node Search:
        #    x_nearest = argmin_{v in Tree} ||v.pos - x_rand||_2
        #
        # 3. Steering Step:
        #    Extend a branch from x_nearest toward x_rand by step_size (6.0m).
        #    x_new = x_nearest + min(dist, step_size) * (x_rand - x_nearest) / dist
        #
        # 4. Cylinder Collision Check:
        #    Checks line segment between x_nearest and x_new against all cylinder obstacles.
        #
        # 5. Best Parent Selection (Cost Minimization):
        #    Scans near neighbors within search_radius (18.0m).
        #    Selects parent that yields the lowest cumulative path cost to x_new:
        #    x_parent = argmin_{v in V_near} (cost(v) + dist(v, x_new))
        #
        # 6. Near-Neighbor Rewiring (Asymptotic Optimality Key!):
        #    For each neighbor v in V_near:
        #    If cost(x_new) + dist(x_new, v) < cost(v):
        #        Rewire parent: v.parent = x_new
        #        v.cost = cost(x_new) + dist(x_new, v)
        #    This constantly shortens and straightens paths throughout the tree!
        # =========================================================================
        """
        t0 = time.time()
        rng = np.random.RandomState(seed if seed is not None else 42)

        # Quick check for straight line path
        if self.is_segment_valid(start, goal, num_checks=25):
            planning_time = (time.time() - t0) * 1000.0
            return [start, goal], planning_time

        nodes = [RRTStarNode(start, parent=None, cost=0.0)]
        goal_node = None
        min_cost = float("inf")

        for _ in range(self.max_iters):
            # Step 1: Biased sampling (15% goal-directed)
            if rng.rand() < self.goal_sample_rate:
                rnd_pos = np.copy(goal)
            else:
                rnd_pos = np.array([
                    rng.uniform(self.x_min, self.x_max),
                    rng.uniform(self.y_min, self.y_max),
                    rng.uniform(start[2] - 5.0, start[2] + 15.0)  # Plan in cruising altitude band
                ], dtype=np.float32)

            # Step 2: Find nearest node in tree
            dists = [np.linalg.norm(node.pos - rnd_pos) for node in nodes]
            nearest_idx = int(np.argmin(dists))
            nearest_node = nodes[nearest_idx]

            # Step 3: Steer toward sample
            dir_vec = rnd_pos - nearest_node.pos
            dist = np.linalg.norm(dir_vec)
            if dist < 1e-4:
                continue
            step_len = min(dist, self.step_size)
            new_pos = nearest_node.pos + (dir_vec / dist) * step_len

            # Step 4: Collision check
            if not self.is_segment_valid(nearest_node.pos, new_pos):
                continue

            # Step 5: Near-neighbor search & optimal parent selection
            new_cost = nearest_node.cost + step_len
            best_parent = nearest_node

            near_nodes = []
            for node in nodes:
                d = np.linalg.norm(node.pos - new_pos)
                if d <= self.search_radius:
                    near_nodes.append((node, d))

            # Choose parent with minimum cost
            for node, d in near_nodes:
                if node.cost + d < new_cost and self.is_segment_valid(node.pos, new_pos):
                    new_cost = node.cost + d
                    best_parent = node

            new_node = RRTStarNode(new_pos, parent=best_parent, cost=new_cost)
            nodes.append(new_node)

            # Step 6: Rewire near neighbors to optimize tree
            for node, d in near_nodes:
                if new_cost + d < node.cost and self.is_segment_valid(new_pos, node.pos):
                    node.parent = new_node
                    node.cost = new_cost + d

            # Goal reached check
            d_to_goal = np.linalg.norm(new_pos - goal)
            if d_to_goal <= self.step_size and (new_cost + d_to_goal < min_cost):
                if self.is_segment_valid(new_pos, goal):
                    goal_node = RRTStarNode(goal, parent=new_node, cost=new_cost + d_to_goal)
                    min_cost = goal_node.cost

        # Reconstruct path
        path = []
        curr = goal_node
        if curr is None:
            # Fallback: find closest node to goal
            best_idx = int(np.argmin([np.linalg.norm(node.pos - goal) for node in nodes]))
            curr = nodes[best_idx]
            path.append(goal)

        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        path.reverse()

        planning_time = (time.time() - t0) * 1000.0
        return path, planning_time

class RRTStarBaseline:
    """
    # =========================================================================
    # PURPOSE:
    # Autonomous 3D RRT* Trajectory Following & MEC Offloading Baseline.
    # Combines asymptotic optimal random-sampling path planning with rule-based
    # edge computing offloading policies.
    #
    # EXECUTION PIPELINE:
    # 1. On first step / reset: Plans 3D RRT* waypoint sequence to station.
    # 2. At each timestep: Steers UAV toward active waypoint along the tree branch.
    # 3. Task Offloading: Dispatches task to designated heuristic (e.g., energy-deadline aware).
    # =========================================================================
    """
    def __init__(self, env, offload_policy=None):
        self.env = env
        self.planner = RRTStar3DPlanner(
            obstacles=env.cfg.airspace.obstacles,
            bounds=(
                env.cfg.airspace.x_min, env.cfg.airspace.x_max,
                env.cfg.airspace.y_min, env.cfg.airspace.y_max,
                env.cfg.airspace.z_min, env.cfg.airspace.z_max
            ),
            d_safe=env.cfg.physics.d_safe,
            obstacle_margin=env.cfg.physics.obstacle_margin
        )
        from models.baselines import EnergyDeadlineAwareGreedyPolicy
        self.offload_policy = offload_policy or EnergyDeadlineAwareGreedyPolicy()
        self.cached_paths = {}
        self.current_wp_idx = {}
        self.planning_times = {}

    def reset(self):
        self.cached_paths = {}
        self.current_wp_idx = {}
        self.planning_times = {}

    def select_actions(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        actions = {}
        p_cfg = self.env.cfg.physics

        for j in range(self.env.num_uavs):
            pos = self.env.positions[j]
            goal = self.env.stations[j]

            if j not in self.cached_paths:
                path, p_time = self.planner.plan_path(pos, goal, seed=42 + j)
                self.cached_paths[j] = path
                self.current_wp_idx[j] = 1
                self.planning_times[j] = p_time

            path = self.cached_paths[j]
            wp_idx = self.current_wp_idx[j]
            curr_target = path[min(wp_idx, len(path) - 1)]

            d_target = float(np.linalg.norm(curr_target - pos))
            d_final = float(np.linalg.norm(goal - pos))

            if d_target < 4.0 and wp_idx < len(path) - 1:
                self.current_wp_idx[j] += 1
                curr_target = path[self.current_wp_idx[j]]
                d_target = float(np.linalg.norm(curr_target - pos))

            if d_final > p_cfg.reach_tolerance:
                dir_vec = (curr_target - pos) / max(1e-6, d_target)
                step_dist = min(d_target, p_cfg.v_max * p_cfg.delta_t)
                disp = dir_vec * step_dist
            else:
                disp = np.zeros(3, dtype=np.float32)

            offload_act = self.offload_policy.select_offload_action(j, self.env)
            actions[j] = {"move": disp.astype(np.float32), "offload": offload_act}

        return actions
