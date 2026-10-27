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
from plots.plot_curves import (
    plot_training_convergence,
    plot_aerodynamic_power_curve,
    plot_all_marl_training_convergence
)

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            pass
        def write(self, s, *args, **kwargs):
            print(s)

def evaluate_validation(env: MultiUAVEnv, agent: Any, model_type: str, val_seeds: List[int] = [1001, 1002, 1003, 1004]) -> Dict[str, float]:
    """Evaluates agent deterministically on held-out randomized 3D validation scenarios."""
    val_returns = []
    val_comp_rates = []
    val_energies = []
    val_interventions = []
    val_arrivals = []

    for s in val_seeds:
        scen = "nav_general" if s % 2 == 1 else "randomized"
        obs, state = env.reset(seed=s, scenario_mode=scen)
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
        arr_cnt = sum(1 for a in info.get("arrival_steps", {}).values() if a is not None)
        val_arrivals.append((arr_cnt / env.num_uavs) * 100.0)

    avg_arrival = float(np.mean(val_arrivals))
    avg_ret = float(np.mean(val_returns))
    avg_interv = float(np.mean(val_interventions))
    mean_final_err = float(info.get("fleet_mean_goal_error", 50.0))
    # Checkpoint selection metric prioritizes navigation arrival and goal proximity:
    if avg_arrival > 0.0:
        val_score = 1000.0 + avg_arrival * 100.0 - 0.5 * mean_final_err - 2.0 * avg_interv
    else:
        # Strictly rank by how close the agent gets to the destination, never by task return
        val_score = -mean_final_err - 2.0 * avg_interv

    return {
        "val_return": avg_ret,
        "val_comp_rate": float(np.mean(val_comp_rates)),
        "val_energy": float(np.mean(val_energies)),
        "val_interventions": avg_interv,
        "val_arrival_rate": avg_arrival,
        "val_mean_final_dist": mean_final_err,
        "val_score": val_score
    }

def train_mappo(
    num_episodes: int = 80,
    seed: int = 42,
    device: str = "auto",
    policy_variant: str = "pure",
    ablation: Optional[str] = None,
    eval_freq: int = 10,
    save_tag: Optional[str] = None,
    curriculum: bool = False,
    output_dir: Optional[str] = None
) -> str:
    dev = DeviceManager.select_device(device)
    tag = save_tag or (f"mappo_{policy_variant}_{ablation}" if ablation else f"mappo_{policy_variant}")

    cfg = MasterConfig()
    cfg.mappo.num_episodes = num_episodes

    DeviceManager.print_diagnostics(
        dev, seed=seed,
        extra_config={
            "Algorithm": "Proposed Joint MAPPO",
            "Variant": policy_variant,
            "Ablation": str(ablation),
            "Episodes": num_episodes,
            "Curriculum": curriculum
        }
    )

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
        policy_variant=policy_variant,
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
    res_dir = output_dir or os.path.join(base_dir, "results")
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
        "policy_variant": policy_variant,
        "ablation": ablation,
        "curriculum": curriculum,
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

    best_val_score = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    pbar = tqdm(range(1, num_episodes + 1), desc=f"[{tag.upper()}]", unit="ep", ncols=115, dynamic_ncols=True)
    for ep in pbar:
        t_env_start = time.time()
        if curriculum:
            if ep <= int(num_episodes * 0.50):
                c_ratio = float(ep) / float(max(1, int(num_episodes * 0.50)))
                scen_mode = "randomized"
            else:
                c_ratio = 1.0
                scen_mode = "nav_general"
        else:
            c_ratio = 1.0
            scen_mode = "randomized"
        obs, state = env.reset(seed=seed + ep, scenario_mode=scen_mode, curriculum_ratio=c_ratio)
        buffer.step_idx = 0

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0
        last_terminated = False
        next_state = state

        for t in range(cfg.mappo.episode_length):
            actions, act_move, act_offload, log_probs = agent.get_actions(obs, deterministic=False)

            # Ablation override for fixed heuristic offloading
            if ablation == "fixed_offload":
                for j in range(env.num_uavs):
                    actions[j]["offload"] = greedy_offloader.select_offload_action(j, env)

            val = agent.get_value(state)
            next_obs, next_state, r, done, info = env.step(actions)
            last_terminated = bool(info.get("terminated", False))

            # Ablations on reward shaping
            if ablation == "no_energy":
                r -= info["reward_components"]["energy"]  # Remove energy penalty
            elif ablation == "no_safety":
                r -= info["reward_components"]["safety"]  # Remove safety penalty

            buffer.insert(obs, state, act_move, act_offload, log_probs, r, val, done, terminated=last_terminated)

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
            next_state=next_state,
            last_terminated=last_terminated,
            ppo_epochs=cfg.mappo.ppo_epochs,
            batch_size=cfg.mappo.batch_size,
            gamma=cfg.mappo.gamma,
            gae_lambda=cfg.mappo.gae_lambda
        )
        total_update_time += (time.time() - t_up_start)

        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0
        arr_rate = float(sum(1 for a in info.get("arrival_steps", {}).values() if a is not None) / env.num_uavs * 100.0)

        # Periodic Validation on held-out seeds
        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan, "val_energy": np.nan, "val_interventions": np.nan, "val_arrival_rate": np.nan, "val_score": np.nan}
        if ep % eval_freq == 0:
            val_stats = evaluate_validation(env, agent, "mappo")
            if ep >= 10 and val_stats["val_score"] > best_val_score:
                best_val_score = val_stats["val_score"]
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
            "arrival_rate": arr_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "val_arrival_rate": val_stats["val_arrival_rate"],
            "val_energy": val_stats["val_energy"],
            "val_interventions": val_stats["val_interventions"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        pbar.set_postfix({
            "Ret": f"{ep_reward:6.1f}",
            "Arr": f"{arr_rate:4.1f}%",
            "Tasks": f"{comp_rate:4.1f}%",
            "Energy": f"{ep_energy/1000.0:4.1f}kJ",
            "Steps/s": f"{env_steps_per_sec:.0f}"
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            pbar.write(f"[{tag.upper()}] Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Arrivals: {arr_rate:5.1f}% | Tasks: {comp_rate:5.1f}% | Energy: {ep_energy/1000.0:6.1f} kJ | Safety: {ep_interventions} | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt_path)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(csv_log_path, index=False)
    if tag in ["mappo", "mappo_pure"]:
        df_metrics.to_csv(os.path.join(res_dir, "training_logs.csv"), index=False)
        if "plot_training_convergence" in globals():
            try:
                plot_training_convergence(csv_log_path, os.path.join(fig_dir, "training_convergence_curves.png"))
                plot_aerodynamic_power_curve(env.aero, os.path.join(fig_dir, "aerodynamic_power_curve.png"))
            except Exception as e:
                print(f"[Warning] Failed to generate convergence plot: {e}")

    print(f"[{tag.upper()} Finished] Best Checkpoint: {best_ckpt_path} | Final: {final_ckpt_path} | Log: {csv_log_path}")
    return best_ckpt_path

def train_ippo(num_episodes: int = 80, seed: int = 42, device: str = "auto", policy_variant: str = "pure", eval_freq: int = 10, curriculum: bool = True, output_dir: Optional[str] = None) -> str:
    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Algorithm": "Independent PPO (IPPO)", "Variant": policy_variant, "Episodes": num_episodes})

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
        policy_variant=policy_variant,
        device=dev
    )

    buffer = RolloutBuffer(
        num_steps=cfg.mappo.episode_length,
        num_uavs=env.num_uavs,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = output_dir or os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, f"best_ippo_{policy_variant}_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, f"final_ippo_{policy_variant}_agent.pt")
    csv_log = os.path.join(res_dir, f"training_logs_ippo_{policy_variant}.csv")

    best_val_score = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    pbar = tqdm(range(1, num_episodes + 1), desc="[IPPO]", unit="ep", ncols=115, dynamic_ncols=True)
    for ep in pbar:
        t_env_start = time.time()
        if curriculum:
            if ep <= int(num_episodes * 0.50):
                c_ratio = float(ep) / float(max(1, int(num_episodes * 0.50)))
                scen_mode = "randomized"
            else:
                c_ratio = 1.0
                scen_mode = "nav_general"
        else:
            c_ratio = 1.0
            scen_mode = "randomized"
        obs, state = env.reset(seed=seed + ep, scenario_mode=scen_mode, curriculum_ratio=c_ratio)
        buffer.step_idx = 0

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0
        last_terminated = False
        next_state = state
        next_obs = obs

        for t in range(cfg.mappo.episode_length):
            actions, act_move, act_offload, log_probs = agent.get_actions(obs, deterministic=False)
            val = agent.get_value(obs)

            next_obs, next_state, r, done, info = env.step(actions)
            last_terminated = bool(info.get("terminated", False))
            buffer.insert(obs, state, act_move, act_offload, log_probs, r, val, done, terminated=last_terminated)

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
            next_obs=next_obs,
            next_state=next_state,
            last_terminated=last_terminated,
            ppo_epochs=cfg.mappo.ppo_epochs,
            batch_size=cfg.mappo.batch_size,
            gamma=cfg.mappo.gamma,
            gae_lambda=cfg.mappo.gae_lambda
        )
        total_update_time += (time.time() - t_up_start)

        total_tasks = max(1, ep_completed + ep_missed)
        comp_rate = (ep_completed / total_tasks) * 100.0
        arr_rate = float(sum(1 for a in info.get("arrival_steps", {}).values() if a is not None) / env.num_uavs * 100.0)

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "ippo")
            if val_stats.get("val_score", -float("inf")) > best_val_score:
                best_val_score = val_stats["val_score"]
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
            "arrival_rate": arr_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        pbar.set_postfix({
            "Ret": f"{ep_reward:6.1f}",
            "Arr": f"{arr_rate:4.1f}%",
            "Tasks": f"{comp_rate:4.1f}%",
            "Energy": f"{ep_energy/1000.0:4.1f}kJ",
            "Steps/s": f"{env_steps_per_sec:.0f}"
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            pbar.write(f"[IPPO]  Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Arrivals: {arr_rate:5.1f}% | Tasks: {comp_rate:5.1f}% | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    if policy_variant == "pure":
        compat_path = os.path.join(ckpt_dir, "best_ippo_agent.pt")
        compat_final = os.path.join(ckpt_dir, "final_ippo_agent.pt")
        import shutil
        if os.path.exists(best_ckpt):
            shutil.copyfile(best_ckpt, compat_path)
        shutil.copyfile(final_ckpt, compat_final)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[IPPO Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_maddpg(num_episodes: int = 80, seed: int = 42, device: str = "auto", policy_variant: str = "pure", eval_freq: int = 10, curriculum: bool = True, output_dir: Optional[str] = None) -> str:
    dev = DeviceManager.select_device(device)
    DeviceManager.print_diagnostics(dev, seed=seed, extra_config={"Algorithm": "Multi-Agent DDPG (MADDPG)", "Variant": policy_variant, "Episodes": num_episodes})

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
        policy_variant=policy_variant,
        device=dev
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = output_dir or os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, f"best_maddpg_{policy_variant}_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, f"final_maddpg_{policy_variant}_agent.pt")
    csv_log = os.path.join(res_dir, f"training_logs_maddpg_{policy_variant}.csv")

    best_val_score = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    pbar = tqdm(range(1, num_episodes + 1), desc="[MADDPG]", unit="ep", ncols=115, dynamic_ncols=True)
    for ep in pbar:
        t_env_start = time.time()
        if curriculum:
            if ep <= int(num_episodes * 0.50):
                c_ratio = float(ep) / float(max(1, int(num_episodes * 0.50)))
                scen_mode = "randomized"
            else:
                c_ratio = 1.0
                scen_mode = "nav_general"
        else:
            c_ratio = 1.0
            scen_mode = "randomized"
        obs, state = env.reset(seed=seed + ep, scenario_mode=scen_mode, curriculum_ratio=c_ratio)

        ep_reward = 0.0
        ep_completed = 0
        ep_missed = 0
        ep_energy = 0.0
        ep_interventions = 0

        for t in range(cfg.mappo.episode_length):
            actions, act_vecs = agent.get_actions(obs, explore=True)
            next_obs, next_state, r, done, info = env.step(actions)
            last_terminated = bool(info.get("terminated", False))

            agent.buffer.add(state, obs, act_vecs, r, next_state, next_obs, done, terminated=last_terminated)

            t_up_start = time.time()
            if agent.buffer.size >= 500:
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
        arr_rate = float(sum(1 for a in info.get("arrival_steps", {}).values() if a is not None) / env.num_uavs * 100.0)

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "maddpg")
            if val_stats.get("val_score", -float("inf")) > best_val_score:
                best_val_score = val_stats["val_score"]
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
            "arrival_rate": arr_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        pbar.set_postfix({
            "Ret": f"{ep_reward:6.1f}",
            "Arr": f"{arr_rate:4.1f}%",
            "Tasks": f"{comp_rate:4.1f}%",
            "Energy": f"{ep_energy/1000.0:4.1f}kJ",
            "Steps/s": f"{env_steps_per_sec:.0f}"
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            pbar.write(f"[MADDPG] Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Arrivals: {arr_rate:5.1f}% | Tasks: {comp_rate:5.1f}% | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    if policy_variant == "pure":
        compat_path = os.path.join(ckpt_dir, "best_maddpg_agent.pt")
        compat_final = os.path.join(ckpt_dir, "final_maddpg_agent.pt")
        import shutil
        if os.path.exists(best_ckpt):
            shutil.copyfile(best_ckpt, compat_path)
        shutil.copyfile(final_ckpt, compat_final)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[MADDPG Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_cppo(num_episodes: int = 80, seed: int = 42, device: str = "auto", eval_freq: int = 10, curriculum: bool = True, output_dir: Optional[str] = None) -> str:
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
    res_dir = output_dir or os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(res_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, "best_cppo_agent.pt")
    final_ckpt = os.path.join(ckpt_dir, "final_cppo_agent.pt")
    csv_log = os.path.join(res_dir, "training_logs_cppo.csv")

    best_val_score = -float("inf")
    metrics = []
    total_env_time = 0.0
    total_update_time = 0.0
    start_total_time = time.time()

    ep_len = cfg.mappo.episode_length
    pbar = tqdm(range(1, num_episodes + 1), desc="[CPPO]", unit="ep", ncols=115, dynamic_ncols=True)
    for ep in pbar:
        t_env_start = time.time()
        if curriculum:
            if ep <= int(num_episodes * 0.50):
                c_ratio = float(ep) / float(max(1, int(num_episodes * 0.50)))
                scen_mode = "randomized"
            else:
                c_ratio = 1.0
                scen_mode = "nav_general"
        else:
            c_ratio = 1.0
            scen_mode = "randomized"
        obs, state = env.reset(seed=seed + ep, scenario_mode=scen_mode, curriculum_ratio=c_ratio)

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
        arr_rate = float(sum(1 for a in info.get("arrival_steps", {}).values() if a is not None) / env.num_uavs * 100.0)

        val_stats = {"val_return": np.nan, "val_comp_rate": np.nan}
        if ep % eval_freq == 0 or ep == 1:
            val_stats = evaluate_validation(env, agent, "cppo")
            if val_stats.get("val_score", -float("inf")) > best_val_score:
                best_val_score = val_stats["val_score"]
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
            "arrival_rate": arr_rate,
            "total_energy": ep_energy,
            "safety_interventions": ep_interventions,
            "val_return": val_stats["val_return"],
            "val_completion_rate": val_stats["val_comp_rate"],
            "env_steps_per_sec": env_steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "wall_clock_sec": time.time() - start_total_time
        })

        pbar.set_postfix({
            "Ret": f"{ep_reward:6.1f}",
            "Arr": f"{arr_rate:4.1f}%",
            "Tasks": f"{comp_rate:4.1f}%",
            "Energy": f"{ep_energy/1000.0:4.1f}kJ",
            "Steps/s": f"{env_steps_per_sec:.0f}"
        })

        if ep % 10 == 0 or ep == 1:
            val_str = f"ValRet: {val_stats['val_return']:6.1f} | " if not np.isnan(val_stats["val_return"]) else ""
            pbar.write(f"[CPPO]  Ep {ep:3d}/{num_episodes} | Return: {ep_reward:7.2f} | {val_str}Arrivals: {arr_rate:5.1f}% | Tasks: {comp_rate:5.1f}% | Energy: {ep_energy/1000.0:6.1f} kJ | Steps/s: {env_steps_per_sec:.0f}")

    agent.save_checkpoint(final_ckpt)
    pd.DataFrame(metrics).to_csv(csv_log, index=False)
    print(f"[CPPO Finished] Best Checkpoint: {best_ckpt} | Final: {final_ckpt} | Log: {csv_log}")
    return best_ckpt

def train_all(num_episodes: int = 500, seed: int = 42, device: str = "auto", policy_variant: str = "pure", curriculum: bool = True, output_dir: Optional[str] = None):
    """Trains all four primary learning algorithms under identical seed and episode budgets with Phase 2 spatial randomization."""
    print("\n" + "=" * 80)
    print(f" TRAINING COMPLETE MARL COMPARISON SUITE (Episodes: {num_episodes}, Seed: {seed}, Variant: {policy_variant}, Curriculum: {curriculum})")
    print("=" * 80)
    train_mappo(num_episodes=num_episodes, seed=seed, device=device, policy_variant=policy_variant, curriculum=curriculum, output_dir=output_dir)
    train_ippo(num_episodes=num_episodes, seed=seed, device=device, policy_variant=policy_variant, curriculum=curriculum, output_dir=output_dir)
    train_maddpg(num_episodes=num_episodes, seed=seed, device=device, policy_variant=policy_variant, curriculum=curriculum, output_dir=output_dir)
    train_cppo(num_episodes=num_episodes, seed=seed, device=device, curriculum=curriculum, output_dir=output_dir)

    # Automatically plot comparative convergence curve across all four algorithms
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = output_dir or os.path.join(base_dir, "results")
    fig_dir = os.path.join(res_dir, "figures")
    csv_dict = {
        "mappo": os.path.join(res_dir, f"training_logs_mappo_{policy_variant}.csv" if os.path.exists(os.path.join(res_dir, f"training_logs_mappo_{policy_variant}.csv")) else os.path.join(res_dir, "training_logs_mappo.csv")),
        "ippo": os.path.join(res_dir, f"training_logs_ippo_{policy_variant}.csv" if os.path.exists(os.path.join(res_dir, f"training_logs_ippo_{policy_variant}.csv")) else os.path.join(res_dir, "training_logs_ippo.csv")),
        "maddpg": os.path.join(res_dir, f"training_logs_maddpg_{policy_variant}.csv" if os.path.exists(os.path.join(res_dir, f"training_logs_maddpg_{policy_variant}.csv")) else os.path.join(res_dir, "training_logs_maddpg.csv")),
        "cppo": os.path.join(res_dir, "training_logs_cppo.csv")
    }
    if "plot_all_marl_training_convergence" in globals():
        try:
            plot_all_marl_training_convergence(csv_dict, os.path.join(fig_dir, "marl_training_convergence_comparison.png"))
            print(f"[Suite Finished] Generated multi-algorithm convergence curve: {os.path.join(fig_dir, 'marl_training_convergence_comparison.png')}")
        except Exception as e:
            print(f"[Warning] Failed to generate multi-algorithm comparison plot: {e}")

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training entry points.

    Keeping parser construction separate from execution lets callers validate
    supported commands without importing side effects or starting training.
    """
    parser = argparse.ArgumentParser(description="Multi-Model Trainer for Multi-UAV Edge Computing")
    parser.add_argument("--model", type=str, default="all", choices=["all", "mappo", "ippo", "maddpg", "cppo", "ablation"],
                        help="Model to train")
    parser.add_argument("--variant", "--policy_variant", dest="variant", type=str, default="pure", choices=["pure", "residual"],
                        help="Policy variant for navigation: pure (pure neural) or residual (APF nominal + learned delta)")
    parser.add_argument("--ablation", type=str, default=None, choices=["no_energy", "no_safety", "fixed_offload"],
                        help="Ablation variant for MAPPO")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Enable curriculum learning schedule")
    parser.add_argument("--output_dir", type=str, default=None, help="Quarantined output directory for results")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.model == "all":
        train_all(num_episodes=args.episodes, seed=args.seed, device=args.device, policy_variant=args.variant, curriculum=args.curriculum, output_dir=args.output_dir)
    elif args.model == "mappo":
        train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device, policy_variant=args.variant, ablation=args.ablation, curriculum=args.curriculum, output_dir=args.output_dir)
    elif args.model == "ippo":
        train_ippo(num_episodes=args.episodes, seed=args.seed, device=args.device, policy_variant=args.variant, curriculum=args.curriculum, output_dir=args.output_dir)
    elif args.model == "maddpg":
        train_maddpg(num_episodes=args.episodes, seed=args.seed, device=args.device, policy_variant=args.variant, curriculum=args.curriculum, output_dir=args.output_dir)
    elif args.model == "cppo":
        train_cppo(num_episodes=args.episodes, seed=args.seed, device=args.device, curriculum=args.curriculum, output_dir=args.output_dir)
    elif args.model == "ablation":
        train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device, policy_variant=args.variant, ablation=args.ablation or "no_energy", curriculum=args.curriculum, output_dir=args.output_dir)
