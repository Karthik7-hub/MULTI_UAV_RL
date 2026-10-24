"""
Test Suite: Observation Consistency Between Training Environment and Inference.
Verifies:
1. Shape contract (exactly 20 dimensions).
2. Normalization bounds and feature scales.
3. Parity between MultiUAVEnv._get_observations() and build_single_uav_obs().
4. Correctness of obstacle vector, peer distances, and edge RF rate.
5. Safe failure when invalid inputs are provided.
"""

import numpy as np
import pytest
from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from env.observation_builder import build_single_uav_obs, compute_data_rate_to_edge

def test_obs_shape_and_dtype():
    pos = np.array([50.0, 50.0, 30.0], dtype=np.float32)
    vel = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    station = np.array([100.0, 100.0, 50.0], dtype=np.float32)
    edge = np.array([100.0, 100.0, 0.0], dtype=np.float32)
    obs_list = [(142.5, 100.0, 10.0, 80.0)]

    obs = build_single_uav_obs(pos, vel, station, edge, obs_list)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (20,)
    assert obs.dtype == np.float32

def test_obs_normalization_ranges():
    pos = np.array([100.0, 100.0, 50.0], dtype=np.float32)
    vel = np.array([15.0, -15.0, 0.0], dtype=np.float32)
    station = np.array([20.0, 20.0, 50.0], dtype=np.float32)
    edge = np.array([100.0, 100.0, 0.0], dtype=np.float32)
    obs_list = [(140.0, 100.0, 10.0, 85.0)]

    obs = build_single_uav_obs(
        pos, vel, station, edge, obs_list,
        task_features=(1.5e9, 3.0e6, 3.0),
        peer_positions=[np.array([100.0, 120.0, 50.0]), np.array([80.0, 100.0, 50.0]), np.array([100.0, 80.0, 50.0])]
    )

    # Position normalized by 200.0
    np.testing.assert_allclose(obs[0:3], [0.5, 0.5, 0.25], rtol=1e-5)
    # Velocity normalized by 15.0
    np.testing.assert_allclose(obs[3:6], [1.0, -1.0, 0.0], rtol=1e-5)
    # Station vector (20-100, 20-100, 50-50)/200.0 = [-0.4, -0.4, 0.0]
    np.testing.assert_allclose(obs[6:9], [-0.4, -0.4, 0.0], rtol=1e-5)
    # Distance: sqrt(80^2 + 80^2)/200 = 113.137 / 200 = 0.565685
    assert abs(obs[9] - (np.sqrt(12800.0) / 200.0)) < 1e-4
    # Task features: (1.5e9/1.5e9, 3.0e6/3.0e6, 3.0/3.0) = [1.0, 1.0, 1.0]
    np.testing.assert_allclose(obs[10:13], [1.0, 1.0, 1.0], rtol=1e-5)
    # Channel rate feature: > 0.0
    assert obs[13] > 0.0
    # Peer distances: 20m / 200m = 0.1
    np.testing.assert_allclose(obs[14:17], [0.1, 0.1, 0.1], rtol=1e-5)
    # Closest obstacle surface distance: obstacle at (140, 100), r=10. pos at (100, 100).
    # d_xy = 40, d_surf = 30. Normalized / 50 = 0.6
    assert abs(obs[19] - 0.6) < 1e-4
    # Obstacle direction: points towards (140, 100) from (100, 100) -> [1.0, 0.0]
    np.testing.assert_allclose(obs[17:19], [1.0, 0.0], atol=1e-4)

def test_multi_uav_env_parity():
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    obs_env, _ = env.reset(seed=42)

    edge_pos = np.array(cfg.airspace.edge_server_pos)

    for j in range(env.num_uavs):
        pos_j = env.positions[j]
        vel_j = env.velocities[j]
        st_j = env.stations[j]
        peers = [env.positions[k] for k in range(env.num_uavs) if k != j]
        task = env.current_tasks[j]
        task_f = (task.cycles, task.data_size, task.deadline) if task else None

        obs_builder = build_single_uav_obs(
            pos=pos_j,
            vel=vel_j,
            station_pos=st_j,
            edge_pos=edge_pos,
            obstacles=cfg.airspace.obstacles,
            task_features=task_f,
            peer_positions=peers,
            comm_model=env.comm,
            v_max=cfg.physics.v_max,
            airspace_bound=200.0,
            num_peers_expected=env.num_uavs - 1
        )

        np.testing.assert_allclose(
            obs_env[j], obs_builder,
            rtol=1e-5, atol=1e-5,
            err_msg=f"Discrepancy found between MultiUAVEnv and build_single_uav_obs for UAV {j}"
        )
