"""
Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and Decentralized Execution (CTDE).
Simultaneously optimizes:
- Discrete task offloading target (Categorical distribution)
- Continuous 3D velocity displacement (Gaussian distribution)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_offload_actions: int, hidden_dim: int = 128, max_disp: float = 15.0):
        super().__init__()
        self.max_disp = max_disp
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        # Discrete head for offloading decision
        self.offload_head = nn.Linear(hidden_dim, num_offload_actions)
        # Continuous head for 3D displacement
        self.move_mean_head = nn.Linear(hidden_dim, 3)
        self.move_log_std = nn.Parameter(torch.zeros(3) - 0.5)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        offload_logits = self.offload_head(features)

        # Inductive goal attractor guidance toward designated station
        station_vec = obs[:, 6:9]
        dist = torch.norm(station_vec, dim=-1, keepdim=True) * 200.0
        dir_to_station = station_vec / (torch.norm(station_vec, dim=-1, keepdim=True) + 1e-6)
        # Suppress attractor velocity when close to station to enable stable hovering (prevents orbiting)
        guidance = dir_to_station * torch.clamp((dist - 3.5) / 10.0, 0.0, 1.0)

        # Lateral obstacle avoidance prior using dynamic slice (last 3 features independent of num_uavs)
        obs_vec_xy = obs[:, -3:-1]
        obs_dist = obs[:, -1:] * 50.0
        obs_dot = (dir_to_station[:, :2] * obs_vec_xy).sum(dim=-1, keepdim=True)
        perp_xy = torch.stack([-obs_vec_xy[:, 1], obs_vec_xy[:, 0]], dim=-1)
        avoid_weight = torch.clamp((25.0 - obs_dist) / 20.0, 0.0, 1.5) * torch.clamp(obs_dot, 0.0, 1.0)

        combined_guidance = guidance.clone()
        combined_guidance[:, :2] = combined_guidance[:, :2] + avoid_weight * perp_xy

        raw_move = self.move_mean_head(features) + 1.5 * combined_guidance
        move_mean = torch.tanh(raw_move) * self.max_disp
        move_std = torch.exp(torch.clamp(self.move_log_std, -2.0, 0.5))
        return offload_logits, move_mean, move_std

    def sample_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits, mean, std = self.forward(obs)
        dist_offload = Categorical(logits=logits)
        dist_move = Normal(mean, std)

        if deterministic:
            act_offload = torch.argmax(logits, dim=-1)
            act_move = mean
        else:
            act_offload = dist_offload.sample()
            act_move = dist_move.sample()

        log_prob_offload = dist_offload.log_prob(act_offload)
        log_prob_move = dist_move.log_prob(act_move).sum(dim=-1)
        total_log_prob = log_prob_offload + log_prob_move

        return act_move, act_offload, total_log_prob

    def evaluate_actions(self, obs: torch.Tensor, act_move: torch.Tensor, act_offload: torch.Tensor):
        logits, mean, std = self.forward(obs)
        dist_offload = Categorical(logits=logits)
        dist_move = Normal(mean, std)

        log_prob_offload = dist_offload.log_prob(act_offload)
        log_prob_move = dist_move.log_prob(act_move).sum(dim=-1)
        total_log_prob = log_prob_offload + log_prob_move

        entropy = dist_offload.entropy() + dist_move.entropy().sum(dim=-1)
        return total_log_prob, entropy

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class RolloutBuffer:
    def __init__(self, num_steps: int, num_uavs: int, obs_dim: int, state_dim: int):
        self.num_steps = num_steps
        self.num_uavs = num_uavs
        self.obs_buf = np.zeros((num_steps, num_uavs, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros((num_steps, state_dim), dtype=np.float32)
        self.act_move_buf = np.zeros((num_steps, num_uavs, 3), dtype=np.float32)
        self.act_offload_buf = np.zeros((num_steps, num_uavs), dtype=np.int64)
        self.log_prob_buf = np.zeros((num_steps, num_uavs), dtype=np.float32)
        self.reward_buf = np.zeros((num_steps,), dtype=np.float32)
        self.value_buf = np.zeros((num_steps,), dtype=np.float32)
        self.done_buf = np.zeros((num_steps,), dtype=np.float32)
        self.step_idx = 0

    def insert(self, obs, state, act_move, act_offload, log_prob, reward, value, done):
        t = self.step_idx
        for j in range(self.num_uavs):
            self.obs_buf[t, j] = obs[j]
            self.act_move_buf[t, j] = act_move[j]
            self.act_offload_buf[t, j] = act_offload[j]
            self.log_prob_buf[t, j] = log_prob[j]
        self.state_buf[t] = state
        self.reward_buf[t] = reward
        self.value_buf[t] = value
        self.done_buf[t] = float(done)
        self.step_idx += 1

    def compute_gae(self, last_val: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        # =========================================================================
        # PURPOSE:
        # Calculates Generalized Advantage Estimation (GAE) and TD(lambda) Returns.
        # Balances bias vs variance in policy gradient estimation.
        #
        # EQUATIONS (Schulman et al., ICLR 2016):
        # 1. Temporal Difference (TD) Residual Error:
        #    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        #
        # 2. Generalized Advantage Estimate:
        #    A_t^{GAE} = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
        #    Computed recursively backwards:
        #    A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
        #
        # 3. Discounted Target Return (for Critic Training):
        #    R_t = A_t^{GAE} + V(s_t)
        #
        # VARIABLES & UNITS:
        # - gamma      : Temporal discount factor = 0.99 (values future rewards)
        # - gae_lambda : GAE bias-variance trade-off parameter = 0.95
        #                (lambda=0: low variance/high bias TD(0); lambda=1: high variance/unbiased Monte Carlo)
        # - V(s_t)     : Centralized critic value prediction at state s_t
        # - r_t        : Scalar step reward received at timestep t
        # - advantages : How much better action a_t was compared to average state value V(s_t)
        # - returns    : Target values for fitting the Centralized Critic network
        # =========================================================================
        """
        advantages = np.zeros(self.num_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            next_val = last_val if t == self.num_steps - 1 else self.value_buf[t + 1]
            next_non_terminal = 1.0 - (1.0 if t == self.num_steps - 1 else self.done_buf[t])
            # TD error delta_t
            delta = self.reward_buf[t] + gamma * next_val * next_non_terminal - self.value_buf[t]
            # Recursive accumulation of exponentially discounted TD errors
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.value_buf
        return advantages, returns

class MAPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_uavs: int = 4,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_uavs = num_uavs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        from utils.device import DeviceManager
        if isinstance(device, str):
            self.device = DeviceManager.select_device(device)
        else:
            self.device = device

        self.actor = ActorNetwork(obs_dim, num_offload_actions, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

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

    def get_value(self, state: np.ndarray) -> float:
        self.critic.eval()
        with torch.no_grad():
            t_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            v = self.critic(t_s)
            return float(v.squeeze(0).item())

    def update(self, buffer: RolloutBuffer, ppo_epochs: int = 5, batch_size: int = 64, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        # =========================================================================
        # PURPOSE:
        # Executes mini-batch PPO policy and value network updates on collected rollout buffer.
        #
        # EQUATIONS:
        # 1. Advantage Normalization:
        #    A_norm = (A - mean(A)) / (std(A) + 1e-8)
        #
        # 2. Probability Ratio:
        #    r_t(theta) = exp( log_pi_theta(a_t | o_t) - log_pi_old(a_t | o_t) )
        #               = pi_theta(a_t | o_t) / pi_old(a_t | o_t)
        #
        # 3. Clipped Surrogate Policy Objective (PPO-Clip):
        #    surr1 = r_t(theta) * A_norm
        #    surr2 = clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_norm
        #    L^{CLIP}(theta) = -E[ min(surr1, surr2) ] - c_entropy * H(pi_theta)
        #    - Clipping prevents destructively large policy updates!
        #
        # 4. Centralized Critic MSE Value Loss:
        #    L^{VF}(phi) = E[ (V_phi(s_t) - R_t)^2 ]
        #    - Fits the centralized critic to predict the true expected target return R_t.
        # =========================================================================
        """
        last_val = self.get_value(buffer.state_buf[-1])
        adv, ret = buffer.compute_gae(last_val, gamma, gae_lambda)

        # Normalize advantages across rollout for stable gradient magnitudes
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_states = torch.tensor(buffer.state_buf, dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(ret, dtype=torch.float32, device=self.device)
        t_advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)

        # Flatten observations and actions across all N UAVs
        t_obs = torch.tensor(buffer.obs_buf, dtype=torch.float32, device=self.device).view(-1, self.obs_dim)
        t_act_move = torch.tensor(buffer.act_move_buf, dtype=torch.float32, device=self.device).view(-1, 3)
        t_act_offload = torch.tensor(buffer.act_offload_buf, dtype=torch.int64, device=self.device).view(-1)
        t_old_log_probs = torch.tensor(buffer.log_prob_buf, dtype=torch.float32, device=self.device).view(-1)
        flat_adv = t_advantages.unsqueeze(1).repeat(1, self.num_uavs).view(-1)

        self.actor.train()
        self.critic.train()

        num_samples = len(flat_adv)
        indices = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                b_idx = indices[start:end]

                # Evaluate current policy on historical actions
                new_log_p, entropy = self.actor.evaluate_actions(t_obs[b_idx], t_act_move[b_idx], t_act_offload[b_idx])
                # Compute PPO importance sampling probability ratio r_t(theta)
                ratio = torch.exp(new_log_p - t_old_log_probs[b_idx])
                surr1 = ratio * flat_adv[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * flat_adv[b_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

            # Critic MSE regression update on global states
            state_idx = np.arange(len(t_states))
            np.random.shuffle(state_idx)
            for s_start in range(0, len(t_states), batch_size):
                s_end = s_start + batch_size
                s_b = state_idx[s_start:s_end]
                val_pred = self.critic(t_states[s_b]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(val_pred, t_returns[s_b])

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
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
