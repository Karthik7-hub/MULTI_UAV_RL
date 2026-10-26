"""Regression coverage for configuration and runtime integration defects."""

import numpy as np

from config import MasterConfig, get_preset_config
from env.comms_compute import Task
from env.multi_uav_env import MultiUAVEnv
from env.safety_filter import SafetyFilter
from train import build_arg_parser


def test_documented_policy_variant_cli_alias_is_available():
    args = build_arg_parser().parse_args(["--model", "mappo", "--policy_variant", "residual"])
    assert args.variant == "residual"


def test_canyon_preset_has_safe_start_and_station_rings():
    env = MultiUAVEnv(get_preset_config("canyon"))
    assert env.verify_station_feasibility()
    for pos in env.positions.values():
        assert env._is_pos_outside_obstacles(pos, margin=env.cfg.physics.obstacle_margin)


def test_environment_preserves_configured_edge_load_and_receiver_peer_queue():
    cfg = MasterConfig()
    cfg.comp.edge_background_load = 7
    env = MultiUAVEnv(cfg)
    assert env.comp.edge_background_load == 7

    env.current_tasks = {j: None for j in range(env.num_uavs)}
    env.current_tasks[0] = Task(
        uav_id=0,
        cycles=1.0e9,
        data_size=1.0e6,
        deadline=2.0,
        arrival_time=0.0,
    )
    actions = {
        j: {"move": np.zeros(3, dtype=np.float32), "offload": 0}
        for j in range(env.num_uavs)
    }
    actions[0]["offload"] = 1
    env.step(actions)

    assert env.comp.peer_queues[1] == 1.0e9
    assert 0 not in env.comp.peer_queues


def test_safety_filter_blocks_swept_peer_swap():
    safety = SafetyFilter(obstacles=[], d_safe=8.0)
    current = {
        0: np.array([80.0, 100.0, 50.0], dtype=np.float32),
        1: np.array([120.0, 100.0, 50.0], dtype=np.float32),
    }
    proposed = {
        0: np.array([120.0, 100.0, 50.0], dtype=np.float32),
        1: np.array([80.0, 100.0, 50.0], dtype=np.float32),
    }

    filtered, interventions, stats = safety.filter_trajectories_with_stats(proposed, current)
    assert interventions >= 1
    assert stats["proposed_collisions"] >= 1
    assert stats["actual_collisions"] == 0

    rel_start = current[0] - current[1]
    rel_end = filtered[0] - filtered[1]
    rel_delta = rel_end - rel_start
    t_min = np.clip(-np.dot(rel_start, rel_delta) / np.dot(rel_delta, rel_delta), 0.0, 1.0)
    assert np.linalg.norm(rel_start + t_min * rel_delta) >= safety.d_safe - 1e-4
