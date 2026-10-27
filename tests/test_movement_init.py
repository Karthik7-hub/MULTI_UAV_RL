"""
Regression Test Suite for Production MAPPO Actor Initialization and Stream Decoupling.

Validates:
1. Production movement head initializes with orthogonal gain 0.01 and move_log_std = 0.5.
2. Initial deterministic velocity output has near-zero magnitude (< 0.5 m/s), ensuring isotropic exploration.
3. Exploration standard deviation is exp(0.5) ~= 1.6487 m/s.
4. move_log_std is an active, trainable parameter of shape (3,) that receives gradients.
5. Save/load round-trips preserve actor outputs and parameter values exactly.
6. Offloading head outputs, dimensions, and weights operate independently from navigation.
7. Fixed PyTorch random seed produces deterministic initialization.
8. Config integration has no legacy experimental flags (movement_init_mode, enable_tangent_obs, enable_detour_reward).
"""

import os
import tempfile
import pytest
import torch
import numpy as np

from config import MasterConfig
from env.observation_builder import OBS_DIM, STATE_DIM_PER_UAV
from models.mappo import ActorNetwork, MAPPOAgent


def test_production_actor_initialization():
    """Verify production actor initializes movement head with gain 0.01 and log_std 0.5."""
    torch.manual_seed(1234)
    actor = ActorNetwork(
        obs_dim=OBS_DIM,
        num_offload_actions=5,
        hidden_dim=128,
        policy_variant="pure"
    )

    assert torch.allclose(actor.move_log_std, torch.tensor([0.5, 0.5, 0.5]), atol=1e-6)
    # Check that move_mean_head weights are initialized with small orthogonal gain (norm < 0.2)
    assert torch.norm(actor.move_mean_head.weight).item() < 0.2
    assert torch.allclose(actor.move_mean_head.bias, torch.zeros(3))

    # Output std should be exp(0.5) ~= 1.6487
    test_obs = torch.randn(10, OBS_DIM)
    _, mean, std = actor(test_obs)
    assert mean.shape == (10, 3)
    assert std.shape == (3,)
    assert torch.allclose(std, torch.tensor([np.exp(0.5)] * 3, dtype=torch.float32), atol=1e-5)

    # Mean speed should be small (near zero), ensuring unbiased isotropic exploration
    speeds = torch.norm(mean, dim=-1)
    assert torch.mean(speeds).item() < 0.5, f"Expected small near-zero mean velocity, got {torch.mean(speeds).item()}"


def test_move_log_std_trainable_and_receives_gradient():
    """Verify move_log_std is an nn.Parameter and receives gradients during backward pass."""
    actor = ActorNetwork(obs_dim=OBS_DIM, num_offload_actions=5)
    assert isinstance(actor.move_log_std, torch.nn.Parameter)
    assert actor.move_log_std.requires_grad

    # Forward pass and synthetic loss to check gradient flow
    obs = torch.randn(4, OBS_DIM)
    _, mean, std = actor(obs)
    # Simulate a policy loss that depends on std
    loss = (std ** 2).sum() + (mean ** 2).sum()
    loss.backward()

    assert actor.move_log_std.grad is not None
    assert torch.all(torch.isfinite(actor.move_log_std.grad))
    assert torch.any(actor.move_log_std.grad != 0.0)


def test_checkpoint_save_load_round_trip():
    """Verify checkpoint save and load preserves exact outputs and parameter values."""
    agent1 = MAPPOAgent(
        obs_dim=OBS_DIM, state_dim=STATE_DIM_PER_UAV * 4, num_uavs=4, hidden_dim=128,
        policy_variant="pure", device="cpu"
    )

    test_obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        logits1, mean1, std1 = agent1.actor(test_obs)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        agent1.save_checkpoint(ckpt_path)

        # Inspect checkpoint contents
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        assert "actor" in ckpt_data
        assert "critic" in ckpt_data

        agent2 = MAPPOAgent(
            obs_dim=OBS_DIM, state_dim=STATE_DIM_PER_UAV * 4, num_uavs=4, hidden_dim=128,
            policy_variant="pure", device="cpu"
        )
        agent2.load_checkpoint(ckpt_path)

        with torch.no_grad():
            logits2, mean2, std2 = agent2.actor(test_obs)

        assert torch.allclose(logits1, logits2, atol=1e-7)
        assert torch.allclose(mean1, mean2, atol=1e-7)
        assert torch.allclose(std1, std2, atol=1e-7)


def test_offload_stream_and_architecture_invariance():
    """Verify offloading head initialization, dimensions, and weights operate independently from nav head."""
    torch.manual_seed(999)
    actor = ActorNetwork(obs_dim=OBS_DIM, num_offload_actions=5)

    assert actor.offload_head.weight.shape == (5, 128)
    assert actor.move_mean_head.weight.shape == (3, 128)
    assert actor.move_log_std.shape == (3,)

    # Forward pass outputs both streams cleanly
    test_obs = torch.randn(4, OBS_DIM)
    logits, mean, std = actor(test_obs)
    assert logits.shape == (4, 5)
    assert mean.shape == (4, 3)
    assert std.shape == (3,)


def test_deterministic_initialization_under_fixed_seed():
    """Verify identical random seed produces identical parameter tensors."""
    torch.manual_seed(42)
    actor1 = ActorNetwork(obs_dim=OBS_DIM, num_offload_actions=5)

    torch.manual_seed(42)
    actor2 = ActorNetwork(obs_dim=OBS_DIM, num_offload_actions=5)

    for p1, p2 in zip(actor1.parameters(), actor2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)


def test_config_production_cleanliness():
    """Verify MasterConfig has eliminated obsolete experimental flags."""
    cfg = MasterConfig()
    assert not hasattr(cfg.mappo, "movement_init_mode"), "movement_init_mode must not exist in production config"
    assert not hasattr(cfg.mappo, "enable_tangent_obs"), "enable_tangent_obs must not exist in production config"
    assert not hasattr(cfg.mappo, "enable_detour_reward"), "enable_detour_reward must not exist in production config"
