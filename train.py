"""
Unified Training Pipeline for Multi-UAV Deep Reinforcement Learning.
Features:
- Hardware Device Management (Auto, CUDA, CPU) with startup diagnostics
- Throughput benchmarking (env steps/sec, updates/sec, wall-clock time)
- Validation scenarios with separate best-validation and final checkpoints
- Algorithms supported: MAPPO (CTDE), IPPO (DTDE), MADDPG (Off-Policy), CPPO (CTCE)
- Systematic ablation studies (no-energy, no-safety, fixed-greedy-offload)
- Full reproducibility with per-seed logging and configuration JSON exports
"""

import os
import sys
import time
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from config import MasterConfig
from env.multi_uav_env import MultiUAVEnv
from utils.device import DeviceManager
from models.mappo import MAPPOAgent, RolloutBuffer
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent
from models.baselines import EnergyDeadlineAwareGreedyPolicy
from plots.plot_curves import plot_training_convergence, plot_aerodynamic_power_curve

def evaluate_validation(env: MultiUAVEnv, agent: Any, model_type: str, val_seeds: List[int] = [1001, 1002, 1003]) -> Dict[str, float]:
    """Evaluates agent deterministically on held-out validation scenarios."""
    val_returns = []
    val_comp_rates = []
    val_energies = []
    val_interventions = []

    for s in val_seeds:
        obs, state = env.reset(seed=s)
        ep_ret = 0.0
        ep_comp = 0
        ep_miss = 0
        ep_energy = 0.0
        ep_interv = 0

        for _ in range(env.cfg.mappo.episode_length):
            if model_type == "cppo":
                actions, _, _, _ = agent.get_actions(state, deterministic=True)
            elif model_type in ["mappo", "ippo"]:
                actions, _, _, _ = agent.get_actions(obs, deterministic=True)
            elif model_type == "maddpg":
                actions, _ = agent.get_actions(obs, explore=False)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            obs, state, r, done, info = env.step(actions)
            ep_ret += r
            ep_comp += info["tasks_completed"]
            ep_miss += info["deadline_misses"]
            ep_energy += info["step_energy"]
            ep_interv += info["safety_interventions"]
            if done:
                break

        total_tasks = max(1, ep_comp + ep_miss)
        val_returns.append(ep_ret)
        val_comp_rates.append((ep_comp / total_tasks) * 100.0)
        val_energies.append(ep_energy)
        val_interventions.append(ep_interv)

    return {
        "val_return": float(np.mean(val_returns)),
        "val_comp_rate": float(np.mean(val_comp_rates)),
        "val_energy": float(np.mean(val_energies)),
        "val_interventions": float(np.mean(val_interventions))
    }

def train_mappo(
    num_episodes: int = 80,
    seed: int = 42,
    device: str = "auto",
    ablation: Optional[str] = None,
    eval_freq: int = 10,
    save_tag: Optional[str] = None
) -> str:
    dev = DeviceManager.select_device(device)
    tag = save_tag or (f"mappo_{ablation}" if ablation else "mappo")
    DeviceManager.print_diagnostics(
        dev, seed=seed,
        extra_config={"Algorithm": "Proposed Joint MAPPO", "Ablation": str(ablation), "Episodes": num_episodes}
    )

    cfg = MasterConfig()
    cfg.mappo.num_episodes = num_episodes
    env = MultiUAVEnv(cfg)

    agent = MAPPOAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        num_uavs=env.num_uavs,
        num_offload_actions=env.num_uavs + 1,
        hidden_dim=cfg.mappo.hidden_dim,
        actor_lr=cfg.mappo.actor_lr,
        critic_lr=cfg.mappo.critic_lr,
        clip_eps=cfg.mappo.clip_epsilon,
        entropy_coef=cfg.mappo.entropy_coef,
        value_coef=cfg.mappo.value_loss_coef,
        max_grad_norm=cfg.mappo.max_grad_norm,
        device=dev
    )

    buffer = RolloutBuffer(
        num_steps=cfg.mappo.episode_length,
        num_uavs=env.num_uavs,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim
    )

    greedy_offloader = EnergyDeadlineAwareGreedyPolicy() if ablation == "fixed_offload" else None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f"best_{tag}_agent.pt")
    final_ckpt_path = os.path.join(ckpt_dir, f"final_{tag}_agent.pt")
    csv_log_path = os.path.join(res_dir, f"training_logs_{tag}.csv")
    config_json_path = os.path.join(res_dir, f"config_{tag}_seed{seed}.json")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Save full configuration JSON for reproducibility
    config_dict = {
        "model": tag,
        "algorithm": "MAPPO",
        "ablation": ablation,
        "seed": seed,
        "device": dev.type,
        "num_episodes": num_episodes,
        "episode_length": cfg.mappo.episode_length,
        "hidden_dim": cfg.mappo.hidden_dim,
        "actor_lr": cfg.mappo.actor_lr,
        "critic_lr": cfg.mappo.critic_lr,
        "gamma": cfg.mappo.gamma,
        "gae_lambda": cfg.mappo.gae_lambda,
        "clip_eps": cfg.mappo.clip_epsilon,
        "batch_size": cfg.mappo.batch_size,
        "ppo_epochs": cfg.mappo.ppo_epochs
    }
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    best_val_return = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    for ep in range(1, num_episodes + 1):
        t_env_start = time.time()
        obs, state = env.reset(seed=seed + ep)
        buffer.step_idx = 0

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0

        for t in range(cfg.mappo.episode_length):
            actions, act_move, act_offload, log_probs = agent.get_actions(obs, deterministic=False)

            # Ablation override for fixed heuristic offloading
            if ablation == "fixed_offload":
                for j in range(env.num_uavs):
                    actions[j]["offload"] = greedy_offloader.select_offload_action(j, env)

            val = agent.get_value(state)
            next_obs, next_state, r, done, info = env.step(actions)

            # Ablations on reward shaping
            if ablation == "no_energy":
                r -= info["reward_components"]["energy"]  # Remove energy penalty
            elif ablation == "no_safety":
                r -= info["reward_components"]["safety"]  # Remove safety penalty

            buffer.insert(obs, state, act_move, act_offload, log_probs, r, val, done)

            obs = next_obs
            state = next_state
            ep_reward += r
            ep_completed += info["tasks_completed"]
            ep_missed += info["deadline_misses"]
            ep_energy += info["step_energy"]
            ep_interventions += info["safety_interventions"]

            if done:
                break

        total_env_time += (time.time() - t_env_start)

        t_up_start = time.time()
        agent.update(
            buffer,
            ppo_epochs=cfg.mappo.ppo_epochs,
            batch_size=cfg.mappo.batch_size,
            gamma=cfg.mappo.gamma,
            gae_lambda=cfg.mappo.gae_lambda
        )
        total_update_time += (time.time() - t_up_start)

        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0

        # Periodic Validation on held-out seeds
        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan, "val_energy": np.nan, "val_interventions": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "mappo")
            if val_stats["val_return"] > best_val_return:
                best_val_return = val_stats["val_return"]
                agent.save_checkpoint(best_ckpt_path)

        cum_env_steps = ep * cfg.mappo.episode_length
        env_steps_per_sec = cum_env_steps / max(1e-4, total_env_time)
        updates_per_sec = ep / max(1e-4, total_update_time)

        metrics.append({
            "episode": ep,
            "return": ep_reward,
            "completed_tasks": ep_completed,
            "deadline_misses": ep_missed,
            "completion_rate": comp_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "val_energy": val_stats["val_energy"],
            "val_interventions": val_stats["val_interventions"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            print(f"[{tag.upper()}] Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Done: {ep_completed:3d} ({comp_rate:5.1f}%) | Energy: {ep_energy/1000.0:6.1f} kJ | Safety: {ep_interventions} | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt_path)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(csv_log_path, index=False)
    if tag == "mappo":
        df_metrics.to_csv(os.path.join(res_dir, "training_logs.csv"), index=False)
        plot_training_convergence(csv_log_path, os.path.join(fig_dir, "training_convergence_curves.png"))
        plot_aerodynamic_power_curve(env.aero, os.path.join(fig_dir, "aerodynamic_power_curve.png"))

    print(f"[{tag.upper()} Finished] Best Checkpoint: {best_ckpt_path} | Final: {final_ckpt_path} | Log: {csv_log_path}")
    return best_ckpt_path

def train_ippo(num_episodes: int = 80, seed: int = 42, device: str = "auto", eval_freq: int = 10) -> str:
    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Algorithm": "Independent PPO (IPPO)", "Episodes": num_episodes})

    cfg = MasterConfig()
    cfg.mappo.num_episodes = num_episodes
    env = MultiUAVEnv(cfg)

    agent = IPPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.num_uavs,
        num_offload_actions=env.num_uavs + 1,
        hidden_dim=cfg.mappo.hidden_dim,
        actor_lr=cfg.mappo.actor_lr,
        critic_lr=cfg.mappo.critic_lr,
        clip_eps=cfg.mappo.clip_epsilon,
        entropy_coef=cfg.mappo.entropy_coef,
        max_grad_norm=cfg.mappo.max_grad_norm,
        device=dev
    )

    buffer = RolloutBuffer(
        num_steps=cfg.mappo.episode_length,
        num_uavs=env.num_uavs,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, "best_ippo_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, "final_ippo_agent.pt")
    csv_log = os.path.join(res_dir, "training_logs_ippo.csv")

    best_val_return = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    for ep in range(1, num_episodes + 1):
        t_env_start = time.time()
        obs, state = env.reset(seed=seed + ep)
        buffer.step_idx = 0

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0

        for t in range(cfg.mappo.episode_length):
            actions, act_move, act_offload, log_probs = agent.get_actions(obs, deterministic=False)
            val = agent.get_value(obs)

            next_obs, next_state, r, done, info = env.step(actions)
            buffer.insert(obs, state, act_move, act_offload, log_probs, r, val, done)

            obs = next_obs
            state = next_state
            ep_reward += r
            ep_completed += info["tasks_completed"]
            ep_missed += info["deadline_misses"]
            ep_energy += info["step_energy"]
            ep_interventions += info["safety_interventions"]
            if done:
                break

        total_env_time += (time.time() - t_env_start)

        t_up_start = time.time()
        agent.update(
            buffer,
            ppo_epochs=cfg.mappo.ppo_epochs,
            batch_size=cfg.mappo.batch_size,
            gamma=cfg.mappo.gamma,
            gae_lambda=cfg.mappo.gae_lambda
        )
        total_update_time += (time.time() - t_up_start)

        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "ippo")
            if val_stats["val_return"] > best_val_return:
                best_val_return = val_stats["val_return"]
                agent.save_checkpoint(best_ckpt)

        cum_env_steps = ep * cfg.mappo.episode_length
        env_steps_per_sec = cum_env_steps / max(1e-4, total_env_time)
        updates_per_sec = ep / max(1e-4, total_update_time)

        metrics.append({
            "episode": ep,
            "return": ep_reward,
            "completed_tasks": ep_completed,
            "deadline_misses": ep_missed,
            "completion_rate": comp_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            print(f"[IPPO]  Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Done: {ep_completed:3d} ({comp_rate:5.1f}%) | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[IPPO Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_maddpg(num_episodes: int = 80, seed: int = 42, device: str = "auto", eval_freq: int = 10) -> str:
    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Algorithm": "Multi-Agent DDPG (MADDPG)", "Episodes": num_episodes})

    cfg = MasterConfig()
    cfg.mappo.num_episodes = num_episodes
    env = MultiUAVEnv(cfg)

    agent = MADDPGAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        num_uavs=env.num_uavs,
        num_offload_actions=env.num_uavs + 1,
        hidden_dim=cfg.mappo.hidden_dim,
        actor_lr=cfg.mappo.actor_lr,
        critic_lr=cfg.mappo.critic_lr,
        gamma=cfg.mappo.gamma,
        tau=0.01,
        noise_std=0.15,
        device=dev
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, "best_maddpg_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, "final_maddpg_agent.pt")
    csv_log = os.path.join(res_dir, "training_logs_maddpg.csv")

    best_val_return = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    for ep in range(1, num_episodes + 1):
        t_env_start = time.time()
        obs, state = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0

        for t in range(cfg.mappo.episode_length):
            actions, act_vecs = agent.get_actions(obs, explore=True)
            next_obs, next_state, r, done, info = env.step(actions)

            agent.buffer.add(state, obs, act_vecs, r, next_state, next_obs, done)

            t_up_start = time.time()
            agent.update(batch_size=cfg.mappo.batch_size)
            total_update_time += (time.time() - t_up_start)

            obs = next_obs
            state = next_state
            ep_reward += r
            ep_completed += info["tasks_completed"]
            ep_missed += info["deadline_misses"]
            ep_energy += info["step_energy"]
            ep_interventions += info["safety_interventions"]
            if done:
                break

        total_env_time += (time.time() - t_env_start)
        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "maddpg")
            if val_stats["val_return"] > best_val_return:
                best_val_return = val_stats["val_return"]
                agent.save_checkpoint(best_ckpt)

        cum_env_steps = ep * cfg.mappo.episode_length
        env_steps_per_sec = cum_env_steps / max(1e-4, total_env_time)
        updates_per_sec = (ep * cfg.mappo.episode_length) / max(1e-4, total_update_time)

        metrics.append({
            "episode": ep,
            "return": ep_reward,
            "completed_tasks": ep_completed,
            "deadline_misses": ep_missed,
            "completion_rate": comp_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            print(f"[MADDPG] Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Done: {ep_completed:3d} ({comp_rate:5.1f}%) | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[MADDPG Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_cppo(num_episodes: int = 80, seed: int = 42, device: str = "auto", eval_freq: int = 10) -> str:
    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Algorithm": "Centralized PPO (CPPO, CTCE)", "Episodes": num_episodes})

    cfg = MasterConfig()
    cfg.mappo.num_episodes = num_episodes
    env = MultiUAVEnv(cfg)

    agent = CPPOAgent(
        state_dim=env.state_dim,
        num_uavs=env.num_uavs,
        num_offload_actions=env.num_uavs + 1,
        hidden_dim=cfg.mappo.hidden_dim,
        actor_lr=cfg.mappo.actor_lr,
        critic_lr=cfg.mappo.critic_lr,
        clip_eps=cfg.mappo.clip_epsilon,
        entropy_coef=cfg.mappo.entropy_coef,
        max_grad_norm=cfg.mappo.max_grad_norm,
        device=dev
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, "best_cppo_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, "final_cppo_agent.pt")
    csv_log = os.path.join(res_dir, "training_logs_cppo.csv")

    best_val_return = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    ep_len = cfg.mappo.episode_length
    for ep in range(1, num_episodes + 1):
        t_env_start = time.time()
        obs, state = env.reset(seed=seed + ep)

        states_buf = np.zeros((ep_len, env.state_dim), dtype=np.float32)
        act_move_buf = np.zeros((ep_len, env.num_uavs, 3), dtype=np.float32)
        act_offload_buf = np.zeros((ep_len, env.num_uavs), dtype=np.int64)
        log_prob_buf = np.zeros((ep_len,), dtype=np.float32)
        reward_buf = np.zeros((ep_len,), dtype=np.float32)
        value_buf = np.zeros((ep_len,), dtype=np.float32)
        done_buf = np.zeros((ep_len,), dtype=np.float32)

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0

        for t in range(ep_len):
            actions, act_m, act_o, lp = agent.get_actions(state, deterministic=False)
            val = agent.get_value(state)

            states_buf[t] = state
            act_move_buf[t] = np.stack([act_m[j] for j in range(env.num_uavs)])
            act_offload_buf[t] = np.array([act_o[j] for j in range(env.num_uavs)], dtype=np.int64)
            log_prob_buf[t] = lp
            value_buf[t] = val

            next_obs, next_state, r, done, info = env.step(actions)
            reward_buf[t] = r
            done_buf[t] = float(done)

            obs = next_obs
            state = next_state
            ep_reward += r
            ep_completed += info["tasks_completed"]
            ep_missed += info["deadline_misses"]
            ep_energy += info["step_energy"]
            ep_interventions += info["safety_interventions"]
            if done:
                break

        total_env_time += (time.time() - t_env_start)

        # GAE computation
        last_val = agent.get_value(state)
        advantages = np.zeros(ep_len, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(ep_len)):
            next_v = last_val if t == ep_len - 1 else value_buf[t + 1]
            non_term = 1.0 - (1.0 if t == ep_len - 1 else done_buf[t])
            delta = reward_buf[t] + cfg.mappo.gamma * next_v * non_term - value_buf[t]
            last_gae = delta + cfg.mappo.gamma * cfg.mappo.gae_lambda * non_term * last_gae
            advantages[t] = last_gae
        returns = advantages + value_buf

        t_up_start = time.time()
        agent.update(states_buf, act_move_buf, act_offload_buf, log_prob_buf, returns, advantages,
                     ppo_epochs=cfg.mappo.ppo_epochs, batch_size=cfg.mappo.batch_size)
        total_update_time += (time.time() - t_up_start)

        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "cppo")
            if val_stats["val_return"] > best_val_return:
                best_val_return = val_stats["val_return"]
                agent.save_checkpoint(best_ckpt)

        cum_env_steps = ep * cfg.mappo.episode_length
        env_steps_per_sec = cum_env_steps / max(1e-4, total_env_time)
        updates_per_sec = ep / max(1e-4, total_update_time)

        metrics.append({
            "episode": ep,
            "return": ep_reward,
            "completed_tasks": ep_completed,
            "deadline_misses": ep_missed,
            "completion_rate": comp_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            print(f"[CPPO]  Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Done: {ep_completed:3d} ({comp_rate:5.1f}%) | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[CPPO Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_all(num_episodes: int = 80, seed: int = 42, device: str = "auto"):
    """Trains all four primary learning algorithms under identical seed and episode budgets."""
    print("\n" + "=" * 80)
    print(f" TRAINING COMPLETE MARL COMPARISON SUITE (Episodes: {num_episodes}, Seed: {seed})")
    print("=" * 80)
    train_mappo(num_episodes=num_episodes, seed=seed, device=device)
    train_ippo(num_episodes=num_episodes, seed=seed, device=device)
    train_maddpg(num_episodes=num_episodes, seed=seed, device=device)
    train_cppo(num_episodes=num_episodes, seed=seed, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Trainer for Multi-UAV Edge Computing")
    parser.add_argument("--model", type=str, default="all", choices=["all", "mappo", "ippo", "maddpg", "cppo", "ablation"],
                        help="Model to train")
    parser.add_argument("--ablation", type=str, default=None, choices=["no_energy", "no_safety", "fixed_offload"],
                        help="Ablation variant for MAPPO")
    parser.add_argument("--episodes", type=int, default=80, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    if args.model == "all":
        train_all(num_episodes=args.episodes, seed=args.seed, device=args.device)
    elif args.model == "mappo":
        train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device, ablation=args.ablation)
    elif args.model == "ippo":
        train_ippo(num_episodes=args.episodes, seed=args.seed, device=args.device)
    elif args.model == "maddpg":
        train_maddpg(num_episodes=args.episodes, seed=args.seed, device=args.device)
    elif args.model == "cppo":
        train_cppo(num_episodes=args.episodes, seed=args.seed, device=args.device)
    elif args.model == "ablation":
        train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device, ablation=args.ablation or "no_energy")
