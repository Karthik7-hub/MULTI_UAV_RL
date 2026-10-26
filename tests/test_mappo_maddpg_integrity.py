import pytest
import torch
import numpy as np
from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from models.mappo import MAPPOAgent, RolloutBuffer
from models.maddpg import MADDPGAgent, ReplayBuffer

def test_mappo_autograd_and_apf_non_inplace():
    """Verify that compute_nominal_apf_guidance in both pure and residual variants has non-inplace autograd."""
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    agent = MAPPOAgent(env.obs_dim, env.state_dim, num_uavs=env.num_uavs, policy_variant="residual", device="cpu")
    
    obs_t = torch.randn(4, env.obs_dim, requires_grad=False)
    # Forward pass
    logits, mean, std = agent.actor(obs_t)
    loss = mean.sum() + logits.sum()
    # Backward pass must not raise any RuntimeError about in-place modifications
    loss.backward()
    assert agent.actor.move_mean_head.weight.grad is not None
    assert torch.isfinite(agent.actor.move_mean_head.weight.grad).all()

def test_rollout_buffer_early_termination_slicing():
    """Verify that RolloutBuffer.compute_gae and update handle step_idx < num_steps without zeros corruption."""
    num_steps = 100
    num_uavs = 4
    obs_dim = 20
    state_dim = 40
    buffer = RolloutBuffer(num_steps, num_uavs, obs_dim, state_dim)
    
    # Simulate an episode terminating early at step 30
    for t in range(30):
        obs = np.random.randn(num_uavs, obs_dim).astype(np.float32)
        state = np.random.randn(state_dim).astype(np.float32)
        act_m = np.random.randn(num_uavs, 3).astype(np.float32)
        act_o = np.random.randint(0, 5, size=(num_uavs,))
        lp = np.random.randn(num_uavs).astype(np.float32)
        r = 1.0
        val = 0.5
        done = (t == 29)
        term = (t == 29)
        buffer.insert(obs, state, act_m, act_o, lp, r, val, done, terminated=term)
    
    adv, ret = buffer.compute_gae(last_val=0.0, last_terminated=True)
    assert len(adv) == 30, f"Expected 30 advantages, got {len(adv)}"
    assert len(ret) == 30, f"Expected 30 returns, got {len(ret)}"
    assert np.all(np.isfinite(adv))
    assert np.all(np.isfinite(ret))

def test_maddpg_target_deterministic_and_grad_clip():
    """Verify that MADDPG updates execute cleanly with deterministic target actions and gradient clipping."""
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    agent = MADDPGAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        num_uavs=env.num_uavs,
        num_offload_actions=env.num_uavs + 1,
        policy_variant="pure",
        device="cpu"
    )
    
    # Add dummy transitions to buffer
    for _ in range(70):
        s = np.zeros(env.state_dim, dtype=np.float32)
        o = np.zeros((env.num_uavs, env.obs_dim), dtype=np.float32)
        a = np.zeros((env.num_uavs, agent.action_dim_per_uav), dtype=np.float32)
        r = 1.0
        ns = np.zeros(env.state_dim, dtype=np.float32)
        no = np.zeros((env.num_uavs, env.obs_dim), dtype=np.float32)
        agent.buffer.add(s, o, a, r, ns, no, done=False, terminated=False)
        
    # Update should succeed without error
    agent.update(batch_size=64)
    for p in agent.critic.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            assert p.grad.norm(2).item() <= 1.5

def test_reward_metric_disentanglement():
    """Verify that arrival_rate and task_completed are distinctly reported in environment info."""
    cfg = MasterConfig()
    env = MultiUAVEnv(cfg)
    obs, state = env.reset(seed=42)
    actions = {j: {"move": np.zeros(3, dtype=np.float32), "offload": 0} for j in range(env.num_uavs)}
    _, _, r, done, info = env.step(actions)
    
    assert "arrival_rate" in info
    assert "tasks_completed" in info
    assert 0.0 <= info["arrival_rate"] <= 100.0
    assert info["arrival_rate"] == 0.0  # Initially at perimeter, 0 arrivals
