"""
Independent Proximal Policy Optimization (IPPO) Baseline.
Unlike MAPPO, IPPO uses Decentralized Training and Decentralized Execution (DTDE):
- Each UAV has its own local Critic V_j(o_j) taking only local observation o_j (no global state).
- Serves as the primary MARL ablation baseline to evaluate the benefit of Centralized Training (CTDE).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Any, Tuple

from models.mappo import ActorNetwork, RolloutBuffer

class LocalCriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class IPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        num_uavs: int = 4,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        policy_variant: str = "pure",
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.policy_variant = policy_variant.lower()

        from utils.device import DeviceManager
        if isinstance(device, str):
            self.device = DeviceManager.select_device(device)
        else:
            self.device = device

        # Decentralized Actor (parameter-shared or individual)
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            num_offload_actions=num_offload_actions,
            hidden_dim=hidden_dim,
            policy_variant=self.policy_variant
        ).to(self.device)
        # Decentralized Local Critic: observes ONLY local observation o_j
        self.critic = LocalCriticNetwork(obs_dim, hidden_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def get_actions(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = False):
        actions = {}
        act_move_dict = {}
        act_offload_dict = {}
        log_probs = {}

        self.actor.eval()
        with torch.no_grad():
            for j in range(self.num_uavs):
                t_obs = torch.tensor(obs_dict[j], dtype=torch.float32, device=self.device).unsqueeze(0)
                move, offload, log_p = self.actor.sample_action(t_obs, deterministic=deterministic)

                m_np = move.squeeze(0).cpu().numpy()
                o_int = int(offload.squeeze(0).item())
                actions[j] = {"move": m_np, "offload": o_int}
                act_move_dict[j] = m_np
                act_offload_dict[j] = o_int
                log_probs[j] = float(log_p.squeeze(0).item())

        return actions, act_move_dict, act_offload_dict, log_probs

    def get_value(self, obs_dict: Dict[int, np.ndarray]) -> float:
        """Computes mean value across local observations."""
        self.critic.eval()
        with torch.no_grad():
            vals = []
            for j in range(self.num_uavs):
                t_o = torch.tensor(obs_dict[j], dtype=torch.float32, device=self.device).unsqueeze(0)
                v = self.critic(t_o)
                vals.append(float(v.squeeze(0).item()))
            return float(np.mean(vals))

    def update(
        self,
        buffer: RolloutBuffer,
        next_obs: Optional[Dict[int, np.ndarray]] = None,
        next_state: Optional[np.ndarray] = None,
        last_terminated: bool = False,
        ppo_epochs: int = 5,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        # =========================================================================
        # PURPOSE:
        # Executes Independent PPO (IPPO) updates under the DTDE paradigm.
        #
        # KEY DIFFERENCE FROM MAPPO (CTDE):
        # - In MAPPO, the Critic V_phi(s) evaluates the true global environmental state s.
        # - In IPPO, the Critic V_phi(o_j) evaluates ONLY the local observation o_j.
        # - Because peer UAVs are concurrently updating their policies, the environment appears
        #   non-stationary to each individual UAV's critic, representing the standard DTDE baseline.
        # =========================================================================
        """
        # Calculate advantages using mean local value estimates
        if next_obs is not None:
            last_val = self.get_value(next_obs)
        else:
            last_val = self.get_value({j: buffer.obs_buf[-1, j] for j in range(self.num_uavs)})
        adv, ret = buffer.compute_gae(last_val, last_terminated=last_terminated, gamma=gamma, gae_lambda=gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_obs = torch.tensor(buffer.obs_buf, dtype=torch.float32, device=self.device).view(-1, self.obs_dim)
        t_act_move = torch.tensor(buffer.act_move_buf, dtype=torch.float32, device=self.device).view(-1, 3)
        t_act_offload = torch.tensor(buffer.act_offload_buf, dtype=torch.int64, device=self.device).view(-1)
        t_old_log_probs = torch.tensor(buffer.log_prob_buf, dtype=torch.float32, device=self.device).view(-1)

        t_advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        flat_adv = t_advantages.unsqueeze(1).repeat(1, self.num_uavs).view(-1)
        flat_returns = torch.tensor(ret, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, self.num_uavs).view(-1)

        self.actor.train()
        self.critic.train()

        num_samples = len(flat_adv)
        indices = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                b_idx = indices[start:end]

                # Actor update with PPO-Clip
                new_log_p, entropy = self.actor.evaluate_actions(t_obs[b_idx], t_act_move[b_idx], t_act_offload[b_idx])
                ratio = torch.exp(new_log_p - t_old_log_probs[b_idx])
                surr1 = ratio * flat_adv[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * flat_adv[b_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Local Critic update using local observation (DTDE)
                val_pred = self.critic(t_obs[b_idx]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(val_pred, flat_returns[b_idx])

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
