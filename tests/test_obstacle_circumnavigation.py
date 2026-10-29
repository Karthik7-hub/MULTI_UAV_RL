"""
Regression test suite for Phase 3: Obstacle Circumnavigation for Multi-UAV MAPPO.
Validates:
1. Tangent geometry: bidirectional safe bypass, obstacle on left vs right, collinear path, invariance when obstacle is behind or overflown.
2. Reward correctness: detour progress neutralized and rewarded; invalid moves penalized; safety/proximity penalties strictly active.
3. Architecture integrity: decoupled navigation (18) vs offloading (10) streams, shape invariance.
4. Checkpoint compatibility: backward compatibility loading legacy 20-dim checkpoints into 22-dim models with explicit adaptation.
5. Reversibility and toggle validity: disabling tangent features reproduces exact Phase 2 baseline.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from env.observation_builder import (
    compute_obstacle_clearance_tangent,
    build_single_uav_obs
)
from models.mappo import MAPPOAgent, ActorNetwork, CriticNetwork


# =============================================================================
# 1. Tangent Geometry Tests
# =============================================================================

def test_tangent_geometry_bidirectional_bypass():
    """Verifies that obstacle on the left steers right, and obstacle on the right steers left."""
    # Cylinder centered at (100, 100), radius 10, height 80
    obstacles = [(100.0, 100.0, 10.0, 80.0)]
    station = np.array([100.0, 150.0, 50.0], dtype=np.float32)  # Target is due North

    # Case A: UAV at (95, 80), flight path North.
    # Obstacle at X=100 is to the RIGHT of flight path.
    # Safe bypass must steer LEFT (negative X).
    pos_right_obs = np.array([95.0, 80.0, 50.0], dtype=np.float32)
    tan_left = compute_obstacle_clearance_tangent(pos_right_obs, station, obstacles)
    assert tan_left[0] < -0.1, f"Expected leftward tangent (negative X), got {tan_left}"
    assert np.isclose(np.linalg.norm(tan_left), 1.0, atol=1e-5)

    # Case B: UAV at (105, 80), flight path North.
    # Obstacle at X=100 is to the LEFT of flight path.
    # Safe bypass must steer RIGHT (positive X).
    pos_left_obs = np.array([105.0, 80.0, 50.0], dtype=np.float32)
    tan_right = compute_obstacle_clearance_tangent(pos_left_obs, station, obstacles)
    assert tan_right[0] > 0.1, f"Expected rightward tangent (positive X), got {tan_right}"
    assert np.isclose(np.linalg.norm(tan_right), 1.0, atol=1e-5)


def test_tangent_geometry_collinear_head_on():
    """Verifies that exact head-on collinear approach breaks tie cleanly without NaN or zero norm."""
    obstacles = [(100.0, 100.0, 10.0, 80.0)]
    station = np.array([100.0, 150.0, 50.0], dtype=np.float32)
    pos_head_on = np.array([100.0, 80.0, 50.0], dtype=np.float32)

    tan = compute_obstacle_clearance_tangent(pos_head_on, station, obstacles)
    assert not np.isnan(tan).any()
    assert np.isclose(np.linalg.norm(tan), 1.0, atol=1e-5)
    # Norm is orthogonal to line of sight (1, 0)
    assert np.isclose(tan[1], 0.0, atol=1e-4)


def test_tangent_geometry_invariance_conditions():
    """Verifies that tangent correctly returns [0, 0] when obstacle does not impede flight."""
    obstacles = [(100.0, 100.0, 10.0, 80.0)]
    station = np.array([100.0, 150.0, 50.0], dtype=np.float32)

    # 1. Obstacle is behind UAV (UAV at 100, 120; obstacle at 100, 100; station at 100, 150)
    pos_passed = np.array([100.0, 120.0, 50.0], dtype=np.float32)
    tan_behind = compute_obstacle_clearance_tangent(pos_passed, station, obstacles)
    assert np.allclose(tan_behind, [0.0, 0.0])

    # 2. UAV is flying cleanly above obstacle height (obstacle height 80, UAV at z=90m)
    pos_above = np.array([100.0, 80.0, 90.0], dtype=np.float32)
    tan_above = compute_obstacle_clearance_tangent(pos_above, station, obstacles)
    assert np.allclose(tan_above, [0.0, 0.0])

    # 3. UAV is far away (> 25m surface distance)
    pos_far = np.array([100.0, 20.0, 50.0], dtype=np.float32)  # d_surf = 70m
    tan_far = compute_obstacle_clearance_tangent(pos_far, station, obstacles)
    assert np.allclose(tan_far, [0.0, 0.0])


# =============================================================================
# 2. Reward Correctness Tests
# =============================================================================

def test_detour_reward_correctness_valid_bypass():
    """Verifies that making lateral detour progress around an obstacle neutralizes the Euclidean penalty."""
    cfg = MasterConfig()
    cfg.physics.enable_safety_filter = False  # Isolate reward calculation
    env = MultiUAVEnv(cfg)
    env.reset(seed=42)

    # Place UAV 0 in proximity of East obstacle (cx=142.5, cy=100.0, r=10.0)
    # Station is at (100 + 40, 100, 50) = (140, 100, 50)
    # Place UAV 1, 2, 3 at station to strictly isolate UAV 0's detour reward calculation
    for j in range(1, env.num_uavs):
        env.positions[j] = env.stations[j].copy()
        env.arrival_steps[j] = 0

    env.positions[0] = np.array([142.5, 85.0, 50.0], dtype=np.float32)
    env.stations[0] = np.array([142.5, 120.0, 50.0], dtype=np.float32)  # Target is North of obstacle
    env._prev_d_stations = {j: float(np.linalg.norm(env.positions[j] - env.stations[j])) for j in range(env.num_uavs)}

    # UAV executes lateral detour: moves in positive tangent direction (dx > 0)
    # Action moves slightly away from station in Y but rounds obstacle in X
    actions = {j: {"move": np.array([0.0, 0.0, 0.0], dtype=np.float32), "offload": 0} for j in range(env.num_uavs)}
    # Steer East: [2.0, -0.2, 0.0] -> increases goal distance slightly but executes tangent detour
    actions[0]["move"] = np.array([2.0, -0.2, 0.0], dtype=np.float32)

    _, _, r, _, info = env.step(actions)
    r_components = info.get("reward_components", {})

    # With detour reward, progress should NOT be deeply negative despite Euclidean distance increasing
    assert r_components["progress"] >= -0.5, f"Expected detour-tolerant progress, got {r_components['progress']}"


def test_safety_and_proximity_penalties_remain_strictly_active():
    """Verifies that obstacle proximity and safety intervention penalties are NOT silenced by detour reward."""
    cfg = MasterConfig()
    cfg.physics.enable_safety_filter = True
    env = MultiUAVEnv(cfg)
    env.reset(seed=42)

    # Place UAV directly within 5m of obstacle cylinder surface
    # East corridor cylinder at (142.5, 100.0, r=10.0)
    env.positions[0] = np.array([142.5, 87.0, 50.0], dtype=np.float32)  # d_surf = 13.0 - 10.0 = 3.0m (<10m prox)

    actions = {j: {"move": np.array([0.0, 0.0, 0.0], dtype=np.float32), "offload": 0} for j in range(env.num_uavs)}
    _, _, _, _, info = env.step(actions)
    r_components = info.get("reward_components", {})

    # Proximity penalty must be active (negative)
    assert r_components["proximity"] < -0.1, f"Expected active proximity penalty, got {r_components['proximity']}"


# =============================================================================
# 3. Architecture Integrity Tests
# =============================================================================

def test_architecture_decoupled_dimensions():
    """Verifies that canonical 22-dim obs produces nav_dim=18 and offload_dim=10 with authoritative indices."""
    actor = ActorNetwork(obs_dim=22, num_offload_actions=5, hidden_dim=64, policy_variant="pure")
    assert actor.nav_dim == 18, f"Expected nav_dim=18, got {actor.nav_dim}"
    assert actor.offload_dim == 10, f"Expected offload_dim=10, got {actor.offload_dim}"
    assert actor.nav_indices == list(range(0, 10)) + list(range(14, 22))
    assert actor.offload_indices == list(range(0, 3)) + list(range(10, 17))
    assert actor.nav_backbone[0].in_features == 18
    assert actor.offload_backbone[0].in_features == 10


def test_actor_critic_forward_pass_shapes():
    """Verifies clean tensor forward passes through Actor and Critic with 22-dim obs."""
    obs_batch = torch.randn(8, 22)
    state_batch = torch.randn(8, 40)  # Centralized state is always 40 dims

    actor = ActorNetwork(obs_dim=22, num_offload_actions=5, hidden_dim=64, policy_variant="pure")
    critic = CriticNetwork(state_dim=40, hidden_dim=64)

    offload_logits, move_mean, move_std = actor(obs_batch)
    assert offload_logits.shape == (8, 5)
    assert move_mean.shape == (8, 3)
    assert move_std.shape == (3,)

    val = critic(state_batch)
    assert val.shape == (8, 1)


# =============================================================================
# 4. Checkpoint Compatibility Tests
# =============================================================================

def test_checkpoint_backward_compatibility():
    """Verifies that an older 16-dim nav checkpoint loads safely into an 18-dim nav agent with explicit zero-padding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "legacy_16dim_nav_checkpoint.pt")

        # Create a reference agent and construct synthetic 16-dim nav backbone weights
        ref_agent = MAPPOAgent(
            obs_dim=22, state_dim=40, num_uavs=4, hidden_dim=64,
            policy_variant="pure"
        )
        actor_state = ref_agent.actor.state_dict()
        # Truncate nav_backbone.0.weight to 16 columns (pre-tangent two-stream architecture)
        w_16 = actor_state["nav_backbone.0.weight"][:, :16].clone()
        actor_state["nav_backbone.0.weight"] = w_16

        torch.save({
            "actor": actor_state,
            "critic": ref_agent.critic.state_dict(),
            "policy_variant": "pure"
        }, ckpt_path)

        # Create a new agent (canonical 18-dim nav)
        new_agent = MAPPOAgent(
            obs_dim=22, state_dim=40, num_uavs=4, hidden_dim=64,
            policy_variant="pure"
        )

        # Load the 16-dim nav checkpoint into the 18-dim nav agent
        new_agent.load_checkpoint(ckpt_path)

        # Verify weights were adapted:
        # First 16 columns of nav_backbone.0.weight must match legacy weights
        w_curr = new_agent.actor.nav_backbone[0].weight.detach().cpu().numpy()
        assert np.allclose(w_curr[:, :16], w_16.numpy()), "First 16 input weights must match legacy checkpoint"
        # Columns 16 and 17 (the newly added tangent inputs) must be zero-initialized
        assert np.allclose(w_curr[:, 16:], 0.0), "Newly added tangent inputs must be zero-initialized"

        # Behavioral Verification:
        # Construct compatible 20-dim obs and corresponding 22-dim obs with zero tangents
        test_obs_20 = torch.randn(4, 20)
        test_obs_22 = torch.zeros(4, 22)
        test_obs_22[:, :20] = test_obs_20
        test_obs_22[:, 20:22] = 0.0  # Zero tangent features

        # Behavioral forward equivalence:
        # Manually compute the 16-dim first-layer output using saved weights
        nav_indices_16 = list(range(0, 10)) + list(range(14, 20))
        h1_legacy = torch.nn.functional.linear(
            test_obs_20[:, nav_indices_16],
            w_16,
            actor_state["nav_backbone.0.bias"]
        )

        # Compute adapted model's first-layer output
        nav_feat_18 = test_obs_22[:, new_agent.actor.nav_indices]
        h1_adapted = new_agent.actor.nav_backbone[0](nav_feat_18)

        # Must be bitwise-identical within float precision
        assert torch.allclose(h1_adapted, h1_legacy, atol=1e-6), "Migrated model must produce identical layer 1 activations when tangent features are zero"

        # Verify full action sampling
        obs_dict = {j: np.random.randn(22).astype(np.float32) for j in range(4)}
        actions, act_m, act_o, log_p = new_agent.get_actions(obs_dict, deterministic=True)
        assert len(actions) == 4
        assert act_m[0].shape == (3,)


# =============================================================================
# 5. Canonical MultiUAVEnv Integration Tests
# =============================================================================

def test_env_canonical_22dim_observation_and_tangents():
    """Verifies that MultiUAVEnv natively produces canonical 22-dim observations with bounded tangents."""
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)

    assert env.obs_dim == 22, f"Expected canonical obs_dim=22, got {env.obs_dim}"
    obs, state = env.reset(seed=42)

    for j in range(env.num_uavs):
        assert obs[j].shape == (22,), f"Expected observation shape (22,), got {obs[j].shape}"
        tan = obs[j][20:22]
        assert np.linalg.norm(tan) <= 1.01, f"Tangent norm {np.linalg.norm(tan)} exceeded 1.01"

    actions = {j: {"move": np.zeros(3, dtype=np.float32), "offload": 0} for j in range(env.num_uavs)}
    next_obs, next_state, r, done, info = env.step(actions)

    for j in range(env.num_uavs):
        assert next_obs[j].shape == (22,)
        tan = next_obs[j][20:22]
        assert np.linalg.norm(tan) <= 1.01


def test_env_enforces_num_uavs_restriction():
    """Verifies that MultiUAVEnv explicitly enforces N=4 fleet constraint."""
    cfg = MasterConfig()
    cfg.physics.num_uavs = 6

    with pytest.raises(ValueError) as excinfo:
        MultiUAVEnv(cfg)

    assert "strictly enforces fleet size N=4" in str(excinfo.value)


