"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Baseline.
Reference:
    R. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,"
    NeurIPS 2017.
Features:
- Decentralized Deterministic Actor with continuous 3D velocity displacement and Gumbel-Softmax discrete offloading.
- Centralized Critic Q_j(S, a_1, ..., a_N) conditioned on global state and joint actions of all UAVs.
- Target Actor and Critic networks with Polyak soft updates.
- Off-policy Experience Replay Buffer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class MADDPGActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        max_disp: float = 15.0,
        policy_variant: str = "pure"  # "pure" or "residual"
    ):
        super().__init__()
        self.max_disp = max_disp
        self.policy_variant = policy_variant.lower()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.move_head = nn.Linear(hidden_dim, 3)
        self.offload_head = nn.Linear(hidden_dim, num_offload_actions)

    def compute_nominal_apf_guidance(self, obs: torch.Tensor) -> torch.Tensor:
        station_vec = obs[:, 6:9]
        dist = torch.norm(station_vec, dim=-1, keepdim=True) * 200.0
        dir_to_station = station_vec / (torch.norm(station_vec, dim=-1, keepdim=True) + 1e-6)
        guidance = dir_to_station * torch.clamp((dist - 3.5) / 10.0, 0.0, 1.0)

        obs_vec_xy = obs[:, -3:-1]
        obs_dist = obs[:, -1:] * 50.0
        obs_dot = (dir_to_station[:, :2] * obs_vec_xy).sum(dim=-1, keepdim=True)
        
        # 2D cross-product to detect whether obstacle is on left or right of flight path:
        # cross > 0 means obstacle is to the LEFT -> steer RIGHT (-tangent)
        # cross < 0 means obstacle is to the RIGHT -> steer LEFT (+tangent)
        cross_2d = dir_to_station[:, 0:1] * obs_vec_xy[:, 1:2] - dir_to_station[:, 1:2] * obs_vec_xy[:, 0:1]
        sign_turn = torch.where(cross_2d >= 0.0, -torch.ones_like(cross_2d), torch.ones_like(cross_2d))
        tangent_xy = torch.stack([-obs_vec_xy[:, 1], obs_vec_xy[:, 0]], dim=-1) * sign_turn

        avoid_weight = torch.clamp((25.0 - obs_dist) / 20.0, 0.0, 1.8) * torch.clamp(obs_dot, 0.0, 1.0)
        avoid_vector = 1.0 * tangent_xy - 0.6 * obs_vec_xy

        # Add vertical climbing component when obstacle directly blocks horizontal path
        curr_alt = obs[:, 2:3] * 200.0
        can_climb = torch.where(dir_to_station[:, 2:3] < -0.1, torch.zeros_like(curr_alt), torch.clamp((75.0 - curr_alt) / 20.0, 0.0, 1.0))
        climb_z = torch.clamp((25.0 - obs_dist) / 20.0, 0.0, 1.0) * torch.clamp(obs_dot, 0.0, 1.0) * 0.8 * can_climb

        combined = guidance.clone()
        combined[:, :2] = combined[:, :2] + avoid_weight * avoid_vector
        combined[:, 2:3] = combined[:, 2:3] + climb_z
        return combined

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        learned_move = self.move_head(feat)

        if self.policy_variant == "residual":
            nominal_guidance = self.compute_nominal_apf_guidance(obs)
            raw_move = learned_move + 1.5 * nominal_guidance
        else:
            raw_move = learned_move

        move = torch.tanh(raw_move) * self.max_disp
        offload_logits = self.offload_head(feat)
        return move, offload_logits

class MADDPGCritic(nn.Module):
    def __init__(self, state_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + total_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, obs_dim: int, num_uavs: int, action_dim_per_uav: int):
        self.capacity = capacity
        self.num_uavs = num_uavs
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs_buf = np.zeros((capacity, num_uavs, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, num_uavs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, num_uavs, action_dim_per_uav), dtype=np.float32)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, obs, act, reward, next_state, next_obs, done):
        self.state_buf[self.idx] = state
        self.next_state_buf[self.idx] = next_state
        for j in range(self.num_uavs):
            self.obs_buf[self.idx, j] = obs[j]
            self.next_obs_buf[self.idx, j] = next_obs[j]
            self.act_buf[self.idx, j] = act[j]
        self.reward_buf[self.idx] = reward
        self.done_buf[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.state_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.act_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.reward_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.next_state_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=device)
        )

class MADDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_uavs: int = 4,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        noise_std: float = 0.1,
        buffer_capacity: int = 50000,
        policy_variant: str = "pure",
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_uavs = num_uavs
        self.num_offload_actions = num_offload_actions
        self.action_dim_per_uav = 3 + num_offload_actions
        self.total_action_dim = num_uavs * self.action_dim_per_uav
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.policy_variant = policy_variant.lower()

        from utils.device import DeviceManager
        if isinstance(device, str):
            self.device = DeviceManager.select_device(device)
        else:
            self.device = device

        # Actor and Critic + Targets
        self.actor = MADDPGActor(obs_dim, num_offload_actions, hidden_dim, policy_variant=self.policy_variant).to(self.device)
        self.actor_target = MADDPGActor(obs_dim, num_offload_actions, hidden_dim, policy_variant=self.policy_variant).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = MADDPGCritic(state_dim, self.total_action_dim, hidden_dim).to(self.device)
        self.critic_target = MADDPGCritic(state_dim, self.total_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, obs_dim, num_uavs, self.action_dim_per_uav)

    def get_actions(self, obs_dict: Dict[int, np.ndarray], explore: bool = True):
        actions = {}
        act_vectors = {}

        self.actor.eval()
        with torch.no_grad():
            for j in range(self.num_uavs):
                t_o = torch.tensor(obs_dict[j], dtype=torch.float32, device=self.device).unsqueeze(0)
                move, offload_logits = self.actor(t_o)

                m_np = move.squeeze(0).cpu().numpy()
                if explore:
                    m_np += np.random.normal(0, self.noise_std, size=3)
                    m_np = np.clip(m_np, -15.0, 15.0)

                # Gumbel-Softmax or argmax
                if explore:
                    offload_prob = F.gumbel_softmax(offload_logits, tau=1.0, hard=True).squeeze(0)
                    o_int = int(torch.argmax(offload_prob).item())
                    o_vec = offload_prob.cpu().numpy()
                else:
                    o_int = int(torch.argmax(offload_logits, dim=-1).item())
                    o_vec = np.zeros(self.num_offload_actions, dtype=np.float32)
                    o_vec[o_int] = 1.0

                actions[j] = {"move": m_np.astype(np.float32), "offload": o_int}
                act_vectors[j] = np.concatenate([m_np, o_vec]).astype(np.float32)

        return actions, act_vectors

    def update(self, batch_size: int = 64):
        """
        # =========================================================================
        # PURPOSE:
        # Executes off-policy mini-batch gradient updates for MADDPG.
        #
        # EQUATIONS (Lowe et al., NeurIPS 2017):
        # 1. Temporal-Difference (TD) Target for Centralized Q-Critic:
        #    y = r + gamma * (1 - done) * Q_target(s', a'_1, ..., a'_N)
        #    Where a'_k = mu_target(o'_k) is predicted by the target actor networks.
        #
        # 2. Centralized Critic MSE Loss:
        #    L(Q) = (1 / |B|) * sum( (Q(s, a_1, ..., a_N) - y)^2 )
        #
        # 3. Deterministic Policy Gradient (DPG) for Actor:
        #    grad_theta_j J = (1 / |B|) * sum( grad_theta_j mu_j(o_j) * grad_a_j Q(s, a_1, ..., a_N) )
        #    Maximized by minimizing: L(mu) = -E[ Q(s, a_1, ..., a_N) ]
        #
        # 4. Polyak Target Network Soft Updates:
        #    theta_target <- tau * theta + (1 - tau) * theta_target  (tau = 0.01)
        #    Slowly tracks online networks to maintain stationary Q-learning targets.
        # =========================================================================
        """
        if self.buffer.size < batch_size:
            return

        states, obs, actions, rewards, next_states, next_obs, dones = self.buffer.sample(batch_size, self.device)

        # ---------------------------------------------------------------------
        # 1. Update Centralized Q-Critic
        # ---------------------------------------------------------------------
        with torch.no_grad():
            next_acts = []
            for j in range(self.num_uavs):
                n_m, n_o = self.actor_target(next_obs[:, j])
                n_o_prob = F.gumbel_softmax(n_o, tau=1.0, hard=True)
                next_acts.append(torch.cat([n_m, n_o_prob], dim=-1))
            flat_next_acts = torch.cat(next_acts, dim=-1)
            target_q = self.critic_target(next_states, flat_next_acts).squeeze(-1)
            # TD Target y
            y = rewards + (1.0 - dones) * self.gamma * target_q

        flat_acts = actions.view(batch_size, -1)
        current_q = self.critic(states, flat_acts).squeeze(-1)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------------------------------------------------------------
        # 2. Update Decentralized Actor (Deterministic Policy Gradient)
        # ---------------------------------------------------------------------
        pred_acts = []
        for j in range(self.num_uavs):
            p_m, p_o = self.actor(obs[:, j])
            p_o_prob = F.gumbel_softmax(p_o, tau=1.0, hard=True)
            pred_acts.append(torch.cat([p_m, p_o_prob], dim=-1))
        flat_pred_acts = torch.cat(pred_acts, dim=-1)
        actor_loss = -self.critic(states, flat_pred_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------------------------------------------------------------
        # 3. Polyak Soft Target Updates
        # ---------------------------------------------------------------------
        for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "policy_variant": self.policy_variant,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "policy_variant" in ckpt:
            self.policy_variant = ckpt["policy_variant"]
            self.actor.policy_variant = self.policy_variant
