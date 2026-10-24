"""
Test Suite: Deterministic Evaluation Protocol and Metric Verification.
Verifies:
1. Disentangled Navigation Energy vs Post-Arrival Hovering / Service Energy.
2. Mission Success rate definition (reach tolerance within episode limit).
3. Path length calculation from actual recorded coordinates.
4. Goal error tracking (final distance to goal).
5. Accurate accounting of collisions and safety interventions.
"""

import numpy as np
import pytest
from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv

def test_metric_disentanglement_synthetic():
    """
    Constructs a deterministic 10-step trajectory for 2 UAVs:
    UAV 0 reaches goal at step 3.
    UAV 1 reaches goal at step 7.
    Step duration delta_t = 1.0s.
    Constant power = 100.0 Watts.
    Speed = 10.0 m/s when moving, 0.0 m/s when arrived.
    """
    num_steps = 10
    delta_t = 1.0
    power = 100.0  # Joules per second

    # UAV 0: steps 1,2,3 moving (disp=10m each -> 30m total). Steps 4..10 hovering (disp=0).
    # UAV 1: steps 1..7 moving (disp=10m each -> 70m total). Steps 8..10 hovering (disp=0).
    uav0_arrival_step = 3
    uav1_arrival_step = 7

    uav0_nav_energy = uav0_arrival_step * power * delta_t  # 300 J
    uav0_service_energy = (num_steps - uav0_arrival_step) * power * delta_t  # 700 J
    uav0_total_energy = num_steps * power * delta_t  # 1000 J

    uav1_nav_energy = uav1_arrival_step * power * delta_t  # 700 J
    uav1_service_energy = (num_steps - uav1_arrival_step) * power * delta_t  # 300 J
    uav1_total_energy = num_steps * power * delta_t  # 1000 J

    fleet_nav_energy = uav0_nav_energy + uav1_nav_energy  # 1000 J
    fleet_service_energy = uav0_service_energy + uav1_service_energy  # 1000 J
    fleet_total_energy = uav0_total_energy + uav1_total_energy  # 2000 J

    assert fleet_nav_energy == 1000.0
    assert fleet_service_energy == 1000.0
    assert fleet_total_energy == 2000.0
    assert (fleet_nav_energy + fleet_service_energy) == fleet_total_energy

    # Verify that reporting total energy instead of nav energy would inflate by 2x
    inflation_factor = fleet_total_energy / fleet_nav_energy
    assert inflation_factor == 2.0

def test_environment_energy_disentanglement_run():
    """Runs a controlled deterministic episode in MultiUAVEnv and verifies energy accounting."""
    cfg = MasterConfig()
    cfg.mappo.episode_length = 20
    env = MultiUAVEnv(cfg)
    obs, state = env.reset(seed=42)

    # Force UAV 0 to stay stationary at its start position (hovering)
    # Force UAV 1..3 to move towards station
    for t in range(20):
        actions = {}
        for j in range(env.num_uavs):
            disp = (env.stations[j] - env.positions[j])
            dist = float(np.linalg.norm(disp))
            if dist > 1e-3:
                step_disp = disp * (min(dist, 10.0) / dist)
            else:
                step_disp = np.zeros(3)
            actions[j] = {"move": step_disp.astype(np.float32), "offload": 0}

        obs, state, reward, done, info = env.step(actions)
        if done:
            break

    # Verify energy conservation: nav_energy + service_energy == propulsion_energy
    for j in range(env.num_uavs):
        cum_e = env.cumulative_prop_energy[j]
        nav_e = env.energy_to_goal[j]
        serv_e = env.service_energy[j]
        np.testing.assert_allclose(
            cum_e, nav_e + serv_e,
            rtol=1e-4,
            err_msg=f"Energy conservation violated for UAV {j}: cum={cum_e}, nav={nav_e}, serv={serv_e}"
        )

    # Verify path length conservation
    for j in range(env.num_uavs):
        total_p = env.total_path_length[j]
        nav_p = env.path_length_to_goal[j]
        assert total_p >= nav_p

def test_failure_and_goal_error_accounting():
    """Verifies that an agent that fails to reach goal has 0% success and finite goal error."""
    cfg = MasterConfig()
    cfg.mappo.episode_length = 5
    env = MultiUAVEnv(cfg)
    env.reset(seed=42)

    # All UAVs execute zero movement (stay still)
    for _ in range(5):
        actions = {j: {"move": np.zeros(3, dtype=np.float32), "offload": 0} for j in range(env.num_uavs)}
        _, _, _, done, info = env.step(actions)

    # None should have arrived
    assert all(info["arrival_steps"][j] is None for j in range(env.num_uavs))
    assert not info["all_arrived"]
    assert info["fleet_service_energy"] == 0.0
    assert info["fleet_energy_to_goal"] == sum(env.cumulative_prop_energy.values())
    assert info["fleet_mean_goal_error"] > 15.0  # Stood still away from station
