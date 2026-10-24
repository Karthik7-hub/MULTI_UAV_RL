"""
Centralized Proximal Policy Optimization (CPPO) Baseline.
Implements Centralized Training with Centralized Execution (CTCE):
- Centralized Actor takes global state and predicts joint actions for all UAVs simultaneously.
- Centralized Critic takes global state and predicts V(s).
Provides the upper-bound centralized coordination comparison against CTDE (MAPPO) and DTDE (IPPO).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple

from utils.device import DeviceManager

class CentralizedActorNetwork(nn.Module):
    def __init__(self, state_dim: int, num_uavs: int = 4, num_offload_actions: int = 5, hidden_dim: int = 128, max_disp: float = 15.0):
        super().__init__()
        self.num_uavs = num_uavs
        self.num_offload_actions = num_offload_actions
        self.max_disp = max_disp

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        # Joint continuous movement head for all UAVs: (num_uavs * 3)
        self.move_mean_head = nn.Linear(hidden_dim, num_uavs * 3)
        self.move_log_std = nn.Parameter(torch.zeros(num_uavs * 3) - 0.5)

        # Joint discrete offload heads: list of linear layers for each UAV
        self.offload_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_offload_actions) for _ in range(num_uavs)
        ])

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        # Continuous moves: (batch, num_uavs, 3)
        raw_move = self.move_mean_head(features).view(-1, self.num_uavs, 3)
        move_mean = torch.tanh(raw_move) * self.max_disp
        move_std = torch.exp(torch.clamp(self.move_log_std, -2.0, 0.5)).view(-1, self.num_uavs, 3)

        # Offload logits: (batch, num_uavs, num_offload_actions)
        logits_list = [head(features).unsqueeze(1) for head in self.offload_heads]
        offload_logits = torch.cat(logits_list, dim=1)

        return offload_logits, move_mean, move_std

    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, mean, std = self.forward(state)
        # logits shape: (batch, num_uavs, num_actions)
        dist_offload = Categorical(logits=logits)
        dist_move = Normal(mean, std)

        if deterministic:
            act_offload = torch.argmax(logits, dim=-1)
            act_move = mean
        else:
            act_offload = dist_offload.sample()
            act_move = dist_move.sample()

        log_prob_offload = dist_offload.log_prob(act_offload).sum(dim=-1)
        log_prob_move = dist_move.log_prob(act_move).sum(dim=[-1, -2])
        total_log_prob = log_prob_offload + log_prob_move

        return act_move, act_offload, total_log_prob

    def evaluate_actions(self, state: torch.Tensor, act_move: torch.Tensor, act_offload: torch.Tensor):
        logits, mean, std = self.forward(state)
        dist_offload = Categorical(logits=logits)
        dist_move = Normal(mean, std)

        log_prob_offload = dist_offload.log_prob(act_offload).sum(dim=-1)
        log_prob_move = dist_move.log_prob(act_move).sum(dim=[-1, -2])
        total_log_prob = log_prob_offload + log_prob_move

        entropy = dist_offload.entropy().sum(dim=-1) + dist_move.entropy().sum(dim=[-1, -2])
        return total_log_prob, entropy

class CentralizedCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class CPPOAgent:
    """Centralized PPO Agent for Multi-UAV Joint Navigation and Offloading."""
    def __init__(
        self,
        state_dim: int,
        num_uavs: int = 4,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto"
    ):
        self.state_dim = state_dim
        self.num_uavs = num_uavs
        self.num_offload_actions = num_offload_actions
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = DeviceManager.select_device(device) if isinstance(device, str) else device

        self.actor = CentralizedActorNetwork(state_dim, num_uavs, num_offload_actions, hidden_dim).to(self.device)
        self.critic = CentralizedCriticNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def get_actions(self, state: np.ndarray, deterministic: bool = False):
        self.actor.eval()
        with torch.no_grad():
            t_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            move, offload, log_p = self.actor.sample_action(t_s, deterministic=deterministic)

            m_np = move.squeeze(0).cpu().numpy()          # (num_uavs, 3)
            o_np = offload.squeeze(0).cpu().numpy()        # (num_uavs,)
            lp_float = float(log_p.squeeze(0).item())

            actions = {j: {"move": m_np[j], "offload": int(o_np[j])} for j in range(self.num_uavs)}
            act_move_dict = {j: m_np[j] for j in range(self.num_uavs)}
            act_offload_dict = {j: int(o_np[j]) for j in range(self.num_uavs)}

        return actions, act_move_dict, act_offload_dict, lp_float

    def get_value(self, state: np.ndarray) -> float:
        self.critic.eval()
        with torch.no_grad():
            t_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            v = self.critic(t_s)
            return float(v.squeeze(0).item())

    def update(
        self,
        states: np.ndarray,
        act_moves: np.ndarray,
        act_offloads: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        ppo_epochs: int = 5,
        batch_size: int = 64
    ):
        """
        # =========================================================================
        # PURPOSE:
        # Executes Centralized PPO (CPPO) updates under the CTCE paradigm.
        #
        # ARCHITECTURAL DISTINCTION:
        # - Centralized Training & Centralized Execution (CTCE):
        #   A single monolithic actor receives the entire global state s (dimension 40)
        #   and directly outputs actions for ALL N=4 UAVs at the same time.
        # - Advantage: Full global information awareness; zero coordination mismatch.
        # - Limitation: Unrealistic for real-world swarms where individual UAVs cannot maintain
        #   instantaneous zero-latency communication links to a central ground station.
        #   Serves as an empirical performance upper-bound benchmark for MAPPO.
        #
        # MATHEMATICAL FORMULATION:
        # 1. Joint Policy Factorization:
        #    pi_theta(a | s) = prod_{j=1}^N pi_theta^{move}(a_{j,move} | s) * pi_theta^{offload}(a_{j,offload} | s)
        #    log pi_theta(a | s) = sum_{j=1}^N [ log pi_theta^{move} + log pi_theta^{offload} ]
        #
        # 2. Probability Ratio:
        #    r(theta) = exp( log pi_theta(a | s) - log pi_{theta_old}(a | s) )
        #
        # 3. Clipped Surrogate Joint Actor Objective:
        #    L^{CLIP}(theta) = - E [ min( r(theta) hat{A}, clip(r(theta), 1 - eps, 1 + eps) hat{A} ) ]
        #                      - c_ent * H(pi_theta)
        #
        # 4. Centralized Value Function Loss:
        #    L^V(phi) = E [ ( V_phi(s) - R_target )^2 ]
        # =========================================================================
        """
        t_states = torch.tensor(states, dtype=torch.float32, device=self.device)
        t_act_moves = torch.tensor(act_moves, dtype=torch.float32, device=self.device)
        t_act_offloads = torch.tensor(act_offloads, dtype=torch.int64, device=self.device)
        t_old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        t_adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        self.actor.train()
        self.critic.train()

        num_samples = len(t_states)
        indices = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                b_idx = indices[start:end]

                new_lp, entropy = self.actor.evaluate_actions(
                    t_states[b_idx], t_act_moves[b_idx], t_act_offloads[b_idx]
                )
                ratio = torch.exp(new_lp - t_old_lp[b_idx])
                surr1 = ratio * t_adv[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * t_adv[b_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                val_pred = self.critic(t_states[b_idx]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(val_pred, t_returns[b_idx])

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
