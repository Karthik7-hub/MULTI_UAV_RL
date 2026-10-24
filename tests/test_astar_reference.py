"""
Test Suite: 3D A* Baseline Navigation Reference and Offloading Collapse Verification.
Verifies:
1. 3D A* geometric obstacle clearance and collision-free path guarantee.
2. Waypoint line-of-sight shortcutting validity.
3. Analysis of MEC offloading policies: identifies whether the 3 A* baselines
   collapse to identical decisions due to edge server compute dominance.
"""

import numpy as np
import pytest
from config import MasterConfig, ENV_PRESETS
from models.astar import AStar3DPlanner
from env.multi_uav_env import MultiUAVEnv
from models.baselines import (
    EnergyDeadlineAwareGreedyPolicy,
    LowestLatencyPolicy,
    EdgeOnlyPolicy,
    LocalOnlyPolicy,
    RandomFeasiblePolicy
)

def test_astar_3d_obstacle_clearance():
    """Tests AStar3DPlanner on benchmark airspace with 4 towers."""
    obstacles = ENV_PRESETS["benchmark"]
    bounds = (0.0, 200.0, 0.0, 200.0, 10.0, 100.0)
    planner = AStar3DPlanner(
        obstacles=obstacles,
        bounds=bounds,
        grid_res=4.0,
        d_safe=8.0,
        obstacle_margin=2.0
    )

    start = np.array([20.0, 20.0, 40.0], dtype=np.float32)
    goal = np.array([180.0, 180.0, 50.0], dtype=np.float32)

    path, t_ms = planner.plan_path_with_time(start, goal)
    assert len(path) >= 2, "A* must find a path from start to goal"

    # Verify no waypoint or line segment penetrates any obstacle
    for pt in path:
        assert planner.is_point_valid(pt), f"Waypoint {pt} is inside obstacle or out of bounds!"

    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        intersects = planner.check_segment_obstacle_intersection(p1, p2)
        assert not intersects, f"Path segment {p1} -> {p2} intersects an obstacle!"

def test_offloading_policy_collapse_investigation():
    """
    Evaluates MEC offloading decisions across 100 tasks under realistic channel conditions.
    Demonstrates whether EnergyDeadlineAwareGreedyPolicy and LowestLatencyPolicy
    collapse to EdgeOnlyPolicy because of f_edge (10 GHz) vs f_local (1.5 GHz).
    """
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    env.reset(seed=42)

    greedy_policy = EnergyDeadlineAwareGreedyPolicy()
    latency_policy = LowestLatencyPolicy()
    edge_policy = EdgeOnlyPolicy()

    greedy_choices = []
    latency_choices = []
    edge_choices = []

    for _ in range(50):
        # Generate new tasks
        for j in range(env.num_uavs):
            env.current_tasks[j] = env.task_gen.sample_task(j, env.current_step, env.rng)
            c_greedy = greedy_policy.select_offload_action(j, env)
            c_latency = latency_policy.select_offload_action(j, env)
            c_edge = edge_policy.select_offload_action(j, env)

            greedy_choices.append(c_greedy)
            latency_choices.append(c_latency)
            edge_choices.append(c_edge)

    # Edge choice is always index 4 (num_uavs)
    edge_target = env.num_uavs
    pct_greedy_edge = (sum(1 for c in greedy_choices if c == edge_target) / len(greedy_choices)) * 100.0
    pct_latency_edge = (sum(1 for c in latency_choices if c == edge_target) / len(latency_choices)) * 100.0

    print(f"\n[MEC Analysis] Greedy Edge Choice: {pct_greedy_edge:.1f}%, Latency Edge Choice: {pct_latency_edge:.1f}%")

    # Verify that Greedy and Lowest-Latency make 100% identical decisions
    assert greedy_choices == latency_choices, "Energy-Deadline Greedy and Lowest-Latency policies must make identical decisions"
    # Verify that Edge server is the dominant choice (> 70% even at initial perimeter, and 100% at station)
    assert pct_greedy_edge >= 70.0, "Greedy policy is dominated by edge offloading"
