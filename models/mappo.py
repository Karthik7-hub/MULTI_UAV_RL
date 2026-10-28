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
from typing import Dict, Tuple, Optional, Any

from env.observation_builder import (
    OBS_DIM,
    NAV_FEATURE_INDICES,
    NAV_DIM,
    OFFLOAD_FEATURE_INDICES,
    OFFLOAD_DIM
)

class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        max_disp: float = 15.0,
        policy_variant: str = "pure",  # "pure" (Variant A) or "residual" (Variant B)
        enable_terminal_damping: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_disp = max_disp
        self.policy_variant = policy_variant.lower()
        self.enable_terminal_damping = enable_terminal_damping

        # Authoritative Two-Stream Decoupled Slices:
        self.nav_indices = NAV_FEATURE_INDICES
        self.nav_dim = NAV_DIM  # 18 dims: spatial kinematics, stations, peers, obstacles, tangent
        self.offload_indices = OFFLOAD_FEATURE_INDICES
        self.offload_dim = OFFLOAD_DIM  # 10 dims: coordinates, task attributes, channel rate, peers

        # 1. Dedicated 3D Spatial Navigation Stream
        self.nav_backbone = nn.Sequential(
            nn.Linear(self.nav_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        # Production Movement Head Initialization:
        # Orthogonal gain = 0.01 for near-zero initial mean velocity;
        # Log std = 0.5 (sigma = 1.65) for balanced isotropic directional exploration P(v_|| > 0) = 50.4%
        self.move_mean_head = nn.Linear(hidden_dim, 3)
        self.move_log_std = nn.Parameter(torch.full((3,), 0.5, dtype=torch.float32))

        # 2. Dedicated Task Offloading & MEC Stream
        self.offload_backbone = nn.Sequential(
            nn.Linear(self.offload_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.offload_head = nn.Linear(hidden_dim, num_offload_actions)

        # Standard orthogonal layer initialization for hidden representations
        for m in self.nav_backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                nn.init.constant_(m.bias, 0.0)
        for m in self.offload_backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.move_mean_head.weight, gain=0.01)
        nn.init.constant_(self.move_mean_head.bias, 0.0)
        nn.init.orthogonal_(self.offload_head.weight, gain=0.01)
        nn.init.constant_(self.offload_head.bias, 0.0)
        self.freeze_offload: bool = False

    def compute_nominal_apf_guidance(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes verified nominal APF controller vector with 2D cross-product directional steering."""
        station_vec = obs[:, 6:9]
        dist = torch.norm(station_vec, dim=-1, keepdim=True) * 200.0
        dir_to_station = station_vec / (torch.norm(station_vec, dim=-1, keepdim=True) + 1e-6)
        # Suppress attractor velocity when close to station to enable stable hovering (prevents orbiting)
        guidance = dir_to_station * torch.clamp((dist - 3.5) / 10.0, 0.0, 1.0)

        # Lateral obstacle avoidance prior using dynamic slice (last 3 features independent of num_uavs)
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

        # Add vertical climbing component when obstacle directly blocks horizontal path,
        # but permit descent if the goal is below and avoid over-climbing above obstacles
        curr_alt = obs[:, 2:3] * 200.0
        can_climb = torch.where(dir_to_station[:, 2:3] < -0.1, torch.zeros_like(curr_alt), torch.clamp((75.0 - curr_alt) / 20.0, 0.0, 1.0))
        climb_z = torch.clamp((25.0 - obs_dist) / 20.0, 0.0, 1.0) * torch.clamp(obs_dot, 0.0, 1.0) * 0.8 * can_climb

        combined_xy = guidance[:, :2] + avoid_weight * avoid_vector
        combined_z = guidance[:, 2:3] + climb_z
        return torch.cat([combined_xy, combined_z], dim=-1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nav_feat = obs[..., self.nav_indices]
        offload_feat = obs[..., self.offload_indices]

        nav_features = self.nav_backbone(nav_feat)
        offload_features = self.offload_backbone(offload_feat)

        offload_logits = self.offload_head(offload_features)
        learned_move = self.move_mean_head(nav_features)

        if self.policy_variant == "residual":
            # Variant B: Residual RL (Nominal APF + Learned Policy Correction)
            nominal_guidance = self.compute_nominal_apf_guidance(obs)
            if self.enable_terminal_damping:
                station_vec = obs[:, 6:9]
                dist = torch.norm(station_vec, dim=-1, keepdim=True) * 200.0
                rl_terminal_damping = torch.clamp((dist - 3.5) / 10.0, 0.0, 1.0)
                raw_move = (0.4 * rl_terminal_damping) * learned_move + 2.0 * nominal_guidance
            else:
                raw_move = 0.4 * learned_move + 2.0 * nominal_guidance
        elif self.policy_variant == "apf_only":
            # Variant C: APF Baseline (Zero Learned Guidance, Gain 2.0 APF Alone)
            nominal_guidance = self.compute_nominal_apf_guidance(obs)
            raw_move = 2.0 * nominal_guidance
        else:
            # Variant A: Pure MAPPO (100% Neural Policy Output, Zero Injected Guidance)
            raw_move = learned_move

        move_mean = torch.tanh(raw_move) * self.max_disp
        move_std = torch.exp(torch.clamp(self.move_log_std, -1.5, 1.5))
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
        if self.freeze_offload:
            total_log_prob = log_prob_move
        else:
            total_log_prob = log_prob_offload + log_prob_move

        return act_move, act_offload, total_log_prob

    def evaluate_actions(self, obs: torch.Tensor, act_move: torch.Tensor, act_offload: torch.Tensor):
        logits, mean, std = self.forward(obs)
        dist_offload = Categorical(logits=logits)
        dist_move = Normal(mean, std)

        log_prob_offload = dist_offload.log_prob(act_offload)
        log_prob_move = dist_move.log_prob(act_move).sum(dim=-1)
        if self.freeze_offload:
            total_log_prob = log_prob_move
            entropy = dist_move.entropy().sum(dim=-1)
        else:
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
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=float(np.sqrt(2)))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

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
        self.term_buf = np.zeros((num_steps,), dtype=np.float32)
        self.step_idx = 0

    def insert(self, obs, state, act_move, act_offload, log_prob, reward, value, done, terminated: bool = False):
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
        self.term_buf[t] = float(terminated)
        self.step_idx += 1

    def compute_gae(self, last_val: float, last_terminated: bool = False, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        # =========================================================================
        # MATHEMATICAL DERIVATION: GENERALIZED ADVANTAGE ESTIMATION (GAE)
        # =========================================================================
        # 1. Temporal Difference (TD) Residual at step t:
        #    delta_t^V = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        #    where:
        #    - r_t: Team scalar reward obtained at step t.
        #    - gamma in [0, 1]: Future return discount factor (default: 0.99).
        #    - V(s_{t+1}): Centralized Critic value estimate for the next global state.
        #    - d_t: 1.0 if episode genuinely terminated (arrival or collision), 0.0 otherwise.
        #           NOTE: If an episode reaches episode_length without terminal failure
        #           (truncated), d_t=0 and we bootstrap with last_val to avoid truncation bias.
        #    - V(s_t): Value predicted by Centralized Critic at current global state s_t.
        #
        # 2. Generalized Advantage Estimator A_t^{GAE(gamma, lambda)}:
        #    A_t = delta_t^V + (gamma * lambda) * (1 - d_t) * A_{t+1}
        #    where lambda in [0, 1] balances variance vs bias:
        #    - lambda = 0: A_t = delta_t^V (lowest variance, highest TD bias)
        #    - lambda = 1: A_t = sum_{l=0}^inf gamma^l r_{t+l} - V(s_t) (unbiased Monte Carlo, high variance)
        #    - lambda = 0.95: Schulman et al. optimal trade-off parameter.
        #
        # 3. Target Returns for Critic MSE Training:
        #    R_t = A_t + V(s_t)  (equivalent to empirical lambda-return)
        # =========================================================================
        """
        actual_steps = max(1, self.step_idx)
        advantages = np.zeros(actual_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(actual_steps)):
            if t == actual_steps - 1:
                next_val = last_val
                # If truncated by time limit, bootstrap next_val! Only zero out on genuine termination.
                next_non_terminal = 1.0 - float(last_terminated)
            else:
                next_val = self.value_buf[t + 1]
                next_non_terminal = 1.0 - self.term_buf[t]

            # TD error delta_t
            delta = self.reward_buf[t] + gamma * next_val * next_non_terminal - self.value_buf[t]
            # Recursive accumulation of exponentially discounted TD errors
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.value_buf[:actual_steps]
        return advantages, returns

class MAPPOAgent:
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        state_dim: int = 40,
        num_uavs: int = 4,
        num_offload_actions: int = 5,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_variant: str = "pure",
        enable_terminal_damping: bool = True,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_uavs = num_uavs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.policy_variant = policy_variant.lower()
        self.enable_terminal_damping = enable_terminal_damping

        from utils.device import DeviceManager
        if isinstance(device, str):
            self.device = DeviceManager.select_device(device)
        else:
            self.device = device

        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            num_offload_actions=num_offload_actions,
            hidden_dim=hidden_dim,
            policy_variant=self.policy_variant,
            enable_terminal_damping=self.enable_terminal_damping
        ).to(self.device)
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

    def freeze_offloading(self, freeze: bool = True):
        """Freezes offloading stream and actor head to isolate movement learning during fixed-offload ablation."""
        self.actor.freeze_offload = freeze
        for p in self.actor.offload_backbone.parameters():
            p.requires_grad = not freeze
        for p in self.actor.offload_head.parameters():
            p.requires_grad = not freeze

    def update(
        self,
        buffer: RolloutBuffer,
        next_state: Optional[np.ndarray] = None,
        last_terminated: bool = False,
        ppo_epochs: int = 5,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        # =========================================================================
        # MATHEMATICAL DERIVATION: PPO CLIPPED SURROGATE POLICY & VALUE UPDATES
        # =========================================================================
        # 1. Advantage Normalization:
        #    hat{A}_t = (A_t - mu_A) / (sigma_A + 1e-8)
        #    Ensures balanced gradients regardless of reward scaling across curriculum stages.
        #
        # 2. Probability Ratio r_t(theta):
        #    r_t(theta) = pi_theta(a_t | o_t) / pi_{theta_old}(a_t | o_t)
        #               = exp( log pi_theta(a_t | o_t) - log pi_{theta_old}(a_t | o_t) )
        #    Measures how much more or less likely the updated policy would choose action a_t.
        #
        # 3. Clipped Surrogate Actor Loss L^{CLIP}(theta):
        #    L^{CLIP}(theta) = - E_t [ min( r_t(theta) hat{A}_t,
        #                                  clip(r_t(theta), 1 - eps, 1 + eps) hat{A}_t ) ]
        #                      - c_ent * H(pi_theta( . | o_t))
        #    where:
        #    - eps = 0.2: PPO trust-region clipping threshold.
        #    - H: Shannon action entropy to incentivize exploratory behavior.
        #    - c_ent = 0.01: Entropy coefficient.
        #
        # 4. Centralized Critic MSE Loss L^V(phi):
        #    L^V(phi) = E_{t, s_t} [ ( V_phi(s_t) - R_t )^2 ]
        #    Trained via Adam regression to match target returns R_t across global states.
        # =========================================================================
        """
        valid_len = max(1, buffer.step_idx)
        if next_state is not None:
            last_val = self.get_value(next_state)
        else:
            last_val = self.get_value(buffer.state_buf[valid_len - 1])
        adv, ret = buffer.compute_gae(last_val, last_terminated=last_terminated, gamma=gamma, gae_lambda=gae_lambda)

        # Normalize advantages across rollout for stable gradient magnitudes
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_states = torch.tensor(buffer.state_buf[:valid_len], dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(ret, dtype=torch.float32, device=self.device)
        t_advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)

        # Flatten observations and actions across all N UAVs
        t_obs = torch.tensor(buffer.obs_buf[:valid_len], dtype=torch.float32, device=self.device).view(-1, self.obs_dim)
        t_act_move = torch.tensor(buffer.act_move_buf[:valid_len], dtype=torch.float32, device=self.device).view(-1, 3)
        t_act_offload = torch.tensor(buffer.act_offload_buf[:valid_len], dtype=torch.int64, device=self.device).view(-1)
        t_old_log_probs = torch.tensor(buffer.log_prob_buf[:valid_len], dtype=torch.float32, device=self.device).view(-1)
        flat_adv = t_advantages.unsqueeze(1).repeat(1, self.num_uavs).view(-1)
        val_old = torch.tensor(buffer.value_buf[:valid_len], dtype=torch.float32, device=self.device)

        self.actor.train()
        self.critic.train()

        num_samples = len(flat_adv)
        indices = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            early_break = False
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

                # Approximate KL divergence guard against destructive policy collapse
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-10)).mean()
                    if approx_kl.item() > 0.05:
                        early_break = True
                        break

            # Critic Clipped Value Loss regression update on global states
            state_idx = np.arange(len(t_states))
            np.random.shuffle(state_idx)
            for s_start in range(0, len(t_states), batch_size):
                s_end = s_start + batch_size
                s_b = state_idx[s_start:s_end]
                val_pred = self.critic(t_states[s_b]).squeeze(-1)

                # PPO Value Clipping to prevent critic destabilization
                val_pred_clipped = val_old[s_b] + torch.clamp(val_pred - val_old[s_b], -self.clip_eps * 20.0, self.clip_eps * 20.0)
                vf_loss1 = (val_pred - t_returns[s_b]) ** 2
                vf_loss2 = (val_pred_clipped - t_returns[s_b]) ** 2
                critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

            if early_break:
                break

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "policy_variant": self.policy_variant,
            "obs_dim": getattr(self, "obs_dim", OBS_DIM),
            "nav_dim": getattr(self.actor, "nav_dim", NAV_DIM),
            "offload_dim": getattr(self.actor, "offload_dim", OFFLOAD_DIM),
        }, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        actor_state = dict(ckpt["actor"])

        # Guard against incompatible obsolete single-stream checkpoints
        if "backbone.0.weight" in actor_state and "nav_backbone.0.weight" not in actor_state:
            raise RuntimeError(
                f"Incompatible historical checkpoint architecture in {path}. "
                "This checkpoint was saved with an obsolete single-stream network ('backbone.*') "
                "prior to the Two-Stream Decoupled Architecture refactoring. "
                "Direct loading is unsupported; please use retrained Two-Stream checkpoints."
            )

        # Backward compatibility adaptation: 16-dim nav input into canonical 18-dim nav input
        if "nav_backbone.0.weight" in actor_state:
            w_ckpt = actor_state["nav_backbone.0.weight"]
            w_curr = self.actor.nav_backbone[0].weight
            if w_ckpt.shape != w_curr.shape:
                if w_ckpt.shape[0] == w_curr.shape[0] and w_ckpt.shape[1] == 16 and w_curr.shape[1] == 18:
                    adapted_w = torch.zeros_like(w_curr)
                    adapted_w[:, :16] = w_ckpt
                    adapted_w[:, 16:] = 0.0
                    actor_state["nav_backbone.0.weight"] = adapted_w
                elif w_ckpt.shape[0] == w_curr.shape[0] and w_ckpt.shape[1] == 18 and w_curr.shape[1] == 16:
                    actor_state["nav_backbone.0.weight"] = w_ckpt[:, :16]

        self.actor.load_state_dict(actor_state)

        critic_state = dict(ckpt["critic"])
        if "net.0.weight" in critic_state:
            w_c_ckpt = critic_state["net.0.weight"]
            w_c_curr = self.critic.net[0].weight
            if w_c_ckpt.shape != w_c_curr.shape:
                if w_c_ckpt.shape[0] == w_c_curr.shape[0] and w_c_ckpt.shape[1] == 40 and w_c_curr.shape[1] == 52:
                    adapted_cw = torch.zeros_like(w_c_curr)
                    for uav_idx in range(self.num_uavs):
                        src = uav_idx * 10
                        dst = uav_idx * 13
                        adapted_cw[:, dst:dst+6] = w_c_ckpt[:, src:src+6]
                        # rel_goal columns [dst+6:dst+9] are zero-initialized
                        adapted_cw[:, dst+9] = w_c_ckpt[:, src+6]
                        adapted_cw[:, dst+10:dst+13] = w_c_ckpt[:, src+7:src+10]
                    critic_state["net.0.weight"] = adapted_cw
                elif w_c_ckpt.shape[0] == w_c_curr.shape[0] and w_c_ckpt.shape[1] == 52 and w_c_curr.shape[1] == 40:
                    adapted_cw = torch.zeros_like(w_c_curr)
                    for uav_idx in range(self.num_uavs):
                        src = uav_idx * 13
                        dst = uav_idx * 10
                        adapted_cw[:, dst:dst+6] = w_c_ckpt[:, src:src+6]
                        adapted_cw[:, dst+6] = w_c_ckpt[:, src+9]
                        adapted_cw[:, dst+7:dst+10] = w_c_ckpt[:, src+10:src+13]
                    critic_state["net.0.weight"] = adapted_cw

        self.critic.load_state_dict(critic_state)
        if "actor_opt" in ckpt:
            try:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
            except Exception:
                pass  # Optimizer restart on dimension adaptation
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "policy_variant" in ckpt:
            if self.policy_variant != "apf_only":
                self.policy_variant = ckpt["policy_variant"]
                self.actor.policy_variant = self.policy_variant
