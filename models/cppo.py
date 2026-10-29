"""
Centralized Proximal Policy Optimization (CPPO) Baseline.
Implements Centralized Training with Centralized Execution (CTCE):
- Centralized Actor takes global state and predicts joint actions for all UAVs simultaneously.
- Centralized Critic takes global state and predicts V(s).
Provides the upper-bound centralized coordination comparison against CTDE (MAPPO) and DTDE (IPPO).
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple

from utils.device import DeviceManager

class CentralizedActorNetwork(nn.Module):
    def __init__(self, state_dim: int = 52, num_uavs: int = 4, num_offload_actions: int = 5, hidden_dim: int = 128, max_disp: float = 15.0):
        super().__init__()
        self.state_dim = state_dim
        self.num_uavs = num_uavs
        self.num_offload_actions = num_offload_actions
        self.max_disp = max_disp

        # Two-Stream Decoupled Architecture for Centralized Actor:
        # Canonical State: 52 dims (13 per UAV: pos(3), vel(3), rel_station(3), dist(1), task(3))
        # Nav features: pos, vel, rel_station, dist for all 4 UAVs (10 * 4 = 40 dims)
        # Offload features: pos, dist, task for all 4 UAVs (7 * 4 = 28 dims)
        self.nav_indices = []
        self.offload_indices = []
        if state_dim == 52 and num_uavs == 4:
            for j in range(num_uavs):
                self.nav_indices.extend([j * 13 + k for k in range(10)])
                self.offload_indices.extend([j * 13 + 0, j * 13 + 1, j * 13 + 2, j * 13 + 9, j * 13 + 10, j * 13 + 11, j * 13 + 12])
        elif state_dim == 40 and num_uavs == 4:
            for j in range(num_uavs):
                self.nav_indices.extend([j * 10 + k for k in range(7)])
                self.offload_indices.extend([j * 10 + 0, j * 10 + 1, j * 10 + 2, j * 10 + 6, j * 10 + 7, j * 10 + 8, j * 10 + 9])
        else:
            self.nav_indices = list(range(state_dim))
            self.offload_indices = list(range(state_dim))
        
        self.nav_dim = len(self.nav_indices)
        self.offload_dim = len(self.offload_indices)

        self.nav_backbone = nn.Sequential(
            nn.Linear(self.nav_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.move_mean_head = nn.Linear(hidden_dim, num_uavs * 3)
        self.move_log_std = nn.Parameter(torch.zeros(num_uavs * 3) - 0.2)

        self.offload_backbone = nn.Sequential(
            nn.Linear(self.offload_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.offload_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_offload_actions) for _ in range(num_uavs)
        ])
        self.bounded_gaussian: bool = True
        self._last_raw_move = None

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.shape[-1] == self.state_dim:
            nav_feat = state[..., self.nav_indices]
            offload_feat = state[..., self.offload_indices]
        else:
            nav_feat = state
            offload_feat = state

        nav_h = self.nav_backbone(nav_feat)
        offload_h = self.offload_backbone(offload_feat)

        raw_move = self.move_mean_head(nav_h).view(-1, self.num_uavs, 3)
        self._last_raw_move = raw_move
        move_mean = torch.tanh(raw_move) * self.max_disp
        move_std = torch.exp(torch.clamp(self.move_log_std, -2.0, 0.5)).view(-1, self.num_uavs, 3)

        logits_list = [head(offload_h).unsqueeze(1) for head in self.offload_heads]
        offload_logits = torch.cat(logits_list, dim=1)

        return offload_logits, move_mean, move_std

    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, mean, std = self.forward(state)
        dist_offload = Categorical(logits=logits)
        raw_move = self._last_raw_move if self._last_raw_move is not None else mean

        if self.bounded_gaussian:
            dist_u = Normal(raw_move, std)
            if deterministic:
                act_offload = torch.argmax(logits, dim=-1)
                act_move = mean
                u = torch.clamp(act_move / self.max_disp, -0.999999, 0.999999).atanh()
                log_prob_u = dist_u.log_prob(u).sum(dim=[-1, -2])
                log_prob_move = log_prob_u - (torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6) + math.log(self.max_disp)).sum(dim=[-1, -2])
            else:
                act_offload = dist_offload.sample()
                u = dist_u.sample()
                act_move = torch.tanh(u) * self.max_disp
                log_prob_u = dist_u.log_prob(u).sum(dim=[-1, -2])
                log_prob_move = log_prob_u - (torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6) + math.log(self.max_disp)).sum(dim=[-1, -2])
        else:
            dist_move = Normal(mean, std)
            if deterministic:
                act_offload = torch.argmax(logits, dim=-1)
                act_move = mean
            else:
                act_offload = dist_offload.sample()
                act_move = dist_move.sample()
            log_prob_move = dist_move.log_prob(act_move).sum(dim=[-1, -2])

        log_prob_offload = dist_offload.log_prob(act_offload).sum(dim=-1)
        total_log_prob = log_prob_offload + log_prob_move
        return act_move, act_offload, total_log_prob

    def evaluate_actions(self, state: torch.Tensor, act_move: torch.Tensor, act_offload: torch.Tensor):
        logits, mean, std = self.forward(state)
        dist_offload = Categorical(logits=logits)
        raw_move = self._last_raw_move if self._last_raw_move is not None else mean

        log_prob_offload = dist_offload.log_prob(act_offload).sum(dim=-1)
        if self.bounded_gaussian:
            u = torch.clamp(act_move / self.max_disp, -0.999999, 0.999999).atanh()
            dist_u = Normal(raw_move, std)
            log_prob_u = dist_u.log_prob(u).sum(dim=[-1, -2])
            log_prob_move = log_prob_u - (torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6) + math.log(self.max_disp)).sum(dim=[-1, -2])
            entropy = dist_offload.entropy().sum(dim=-1) + dist_u.entropy().sum(dim=[-1, -2])
        else:
            dist_move = Normal(mean, std)
            log_prob_move = dist_move.log_prob(act_move).sum(dim=[-1, -2])
            entropy = dist_offload.entropy().sum(dim=-1) + dist_move.entropy().sum(dim=[-1, -2])

        total_log_prob = log_prob_offload + log_prob_move
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
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CPPO checkpoint file not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        actor_state = ckpt["actor"]
        if "nav_backbone.0.weight" in actor_state:
            ckpt_nav_in = actor_state["nav_backbone.0.weight"].shape[1]
            curr_nav_in = self.actor.nav_backbone[0].weight.shape[1]
            if ckpt_nav_in != curr_nav_in:
                raise RuntimeError(
                    f"Incompatible CPPO checkpoint: {path} was trained with obsolete state representation "
                    f"(nav_in={ckpt_nav_in}, derived from obsolete 40-dim state without 3D goal direction). "
                    f"The corrected CPPO architecture requires canonical nav_in={curr_nav_in} (from 52-dim state with 3D goal direction). "
                    f"To preserve research reproducibility, this checkpoint cannot be used for inference. Please retrain CPPO."
                )
        elif "backbone.0.weight" in actor_state:
            raise RuntimeError(
                f"Incompatible historical CPPO checkpoint: {path} was saved with an obsolete single-stream backbone. "
                f"Direct loading is unsupported; please retrain with canonical 52-dim state."
            )
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
