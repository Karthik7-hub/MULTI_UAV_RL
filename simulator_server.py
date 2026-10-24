"""
3D UAV Path-Planning Research Simulator - Backend Server
Exposes REST API endpoints for:
1. World geometry and obstacle editing
2. Classical path planning (3D A*, RRT*, SAPF)
3. Reinforcement learning navigation (Pure MAPPO, APF-Residual MAPPO)
4. Step-by-step decision inspection telemetry
5. Side-by-side comparative benchmarking
"""

import os
import sys
import time
import json
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

# Ensure research_v2 root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import MasterConfig, ENV_PRESETS
from env.multi_uav_env import MultiUAVEnv
from env.safety_filter import SafetyFilter
from env.observation_builder import build_single_uav_obs
from models.astar import AStar3DPlanner, PurePursuit3DController
from models.mappo import MAPPOAgent
from utils.device import DeviceManager

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "simulator", "static"))

# Load global configuration and environment
cfg = MasterConfig()
dev = DeviceManager.select_device("cpu")
ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")

# Initialize and cache agents with correct 20-dim observation
pure_agent = MAPPOAgent(
    obs_dim=20,
    state_dim=40,
    num_uavs=cfg.physics.num_uavs,
    hidden_dim=cfg.mappo.hidden_dim,
    policy_variant="pure",
    device=dev
)
pure_ckpt = os.path.join(ckpt_dir, "best_mappo_pure_agent.pt")
pure_agent_loaded = False
pure_load_error = None
if os.path.exists(pure_ckpt):
    try:
        pure_agent.load_checkpoint(pure_ckpt)
        pure_agent_loaded = True
        print("[Simulator] Loaded best_mappo_pure_agent.pt")
    except Exception as e:
        pure_load_error = str(e)
        print("[Simulator] CRITICAL: Could not load pure agent checkpoint:", e)
else:
    pure_load_error = f"Checkpoint not found at {pure_ckpt}"
    print(f"[Simulator] WARNING: {pure_load_error}")

residual_agent = MAPPOAgent(
    obs_dim=20,
    state_dim=40,
    num_uavs=cfg.physics.num_uavs,
    hidden_dim=cfg.mappo.hidden_dim,
    policy_variant="residual",
    device=dev
)
residual_ckpt = os.path.join(ckpt_dir, "best_mappo_residual_agent.pt")
residual_agent_loaded = False
residual_load_error = None
if os.path.exists(residual_ckpt):
    try:
        residual_agent.load_checkpoint(residual_ckpt)
        residual_agent_loaded = True
        print("[Simulator] Loaded best_mappo_residual_agent.pt")
    except Exception as e:
        residual_load_error = str(e)
        print("[Simulator] CRITICAL: Could not load residual agent checkpoint:", e)
else:
    residual_load_error = f"Checkpoint not found at {residual_ckpt}"
    print(f"[Simulator] WARNING: {residual_load_error}")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/world", methods=["GET"])
def get_world():
    """Returns the default airspace configuration, obstacles, and default source/target."""
    return jsonify({
        "bounds": {
            "x_min": cfg.airspace.x_min, "x_max": cfg.airspace.x_max,
            "y_min": cfg.airspace.y_min, "y_max": cfg.airspace.y_max,
            "z_min": cfg.airspace.z_min, "z_max": cfg.airspace.z_max
        },
        "obstacles": [
            {"id": i, "type": "cylinder", "cx": obs[0], "cy": obs[1], "radius": obs[2], "height": obs[3]}
            for i, obs in enumerate(cfg.airspace.obstacles)
        ],
        "default_source": [20.0, 20.0, 10.0],
        "default_target": [180.0, 180.0, 10.0],
        "safety_margin": 2.0,
        "uav_radius": cfg.physics.d_safe / 2.0,
        "v_max": cfg.physics.v_max
    })

@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns all available airspace scenario obstacle presets."""
    preset_data = {}
    for name, obs_list in ENV_PRESETS.items():
        preset_data[name] = [
            {"id": i, "type": "cylinder", "cx": o[0], "cy": o[1], "radius": o[2], "height": o[3], "enabled": True}
            for i, o in enumerate(obs_list)
        ]
    return jsonify(preset_data)

def compute_min_clearance(pos: np.ndarray, obstacles: list, margin: float = 2.0) -> float:
    """Computes exact continuous 3D surface clearance to finite-height cylinders and spheres."""
    clearances = []
    for obs in obstacles:
        is_sphere = obs.get("type") == "sphere" if isinstance(obs, dict) else len(obs) == 5
        if is_sphere:
            cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"] if isinstance(obs, dict) else (obs[0], obs[1], obs[2], obs[3])
            d_3d = float(np.linalg.norm(pos - np.array([cx, cy, cz])))
            clearances.append(d_3d - (r + margin))
        else:
            if isinstance(obs, dict):
                cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
                z_min_obs = float(obs.get("z_min", 0.0))
                z_max_obs = float(obs.get("z_max", h))
            else:
                cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]
                z_min_obs = 0.0
                z_max_obs = h

            d_xy = float(np.linalg.norm(pos[:2] - np.array([cx, cy])))
            delta_xy = max(0.0, d_xy - r)
            if pos[2] > z_max_obs:
                delta_z = pos[2] - z_max_obs
            elif pos[2] < z_min_obs:
                delta_z = z_min_obs - pos[2]
            else:
                delta_z = 0.0

            if delta_xy > 0.0 and delta_z > 0.0:
                d_surface = np.sqrt(delta_xy**2 + delta_z**2)
            elif delta_xy > 0.0:
                d_surface = delta_xy
            elif delta_z > 0.0:
                d_surface = delta_z
            else:
                # Inside cylinder
                d_surface = -min(r - d_xy, pos[2] - z_min_obs, z_max_obs - pos[2])

            clearance = d_surface - margin
            clearances.append(float(clearance))
    return float(min(clearances)) if clearances else 999.0

def make_obs_vector(pos: np.ndarray, vel: np.ndarray, target: np.ndarray, raw_obs: list) -> np.ndarray:
    """Builds the verified 20-dim observation vector matching MultiUAVEnv contracts using shared builder."""
    edge_pos = np.array(cfg.airspace.edge_server_pos, dtype=np.float32)
    return build_single_uav_obs(
        pos=pos,
        vel=vel,
        station_pos=target,
        edge_pos=edge_pos,
        obstacles=raw_obs,
        task_features=None,
        peer_positions=None,
        comm_model=None,
        v_max=cfg.physics.v_max,
        airspace_bound=cfg.airspace.x_max,
        num_peers_expected=cfg.physics.num_uavs - 1
    )

@app.route("/api/plan", methods=["POST"])
def plan_path():
    """
    Executes selected path planning algorithm from start to target.
    Returns path waypoints, genuine explored nodes (for A*), and step-by-step Decision Inspector telemetry.
    """
    data = request.json or {}
    planner_key = data.get("planner", "astar")
    start = np.array(data.get("start", [20.0, 20.0, 10.0]), dtype=np.float32)
    target = np.array(data.get("target", [180.0, 180.0, 10.0]), dtype=np.float32)
    raw_obs = data.get("obstacles", [])
    margin = float(data.get("margin", 2.0))

    print(f"\n[SERVER LIVE INFERENCE] Received request for planner: '{planner_key}'")
    print(f"  Start: {start.tolist()} -> Target: {target.tolist()} | Active Obstacles: {len(raw_obs)}")

    t_start = time.time()
    steps_telemetry = []
    explored_nodes = []
    status = "SUCCESS"

    if planner_key == "astar":
        # 3D A* Search + Pure Pursuit Trajectory Tracking (Classical Baseline)
        bounds = (cfg.airspace.x_min, cfg.airspace.x_max, cfg.airspace.y_min, cfg.airspace.y_max, cfg.airspace.z_min, cfg.airspace.z_max)
        planner = AStar3DPlanner(
            obstacles=raw_obs,
            bounds=bounds,
            grid_res=5.0,
            d_safe=cfg.physics.d_safe,
            obstacle_margin=margin
        )
        path_3d, t_plan, explored_nodes_list = planner.plan_path_with_details(start, target)
        explored_nodes = [[float(p[0]), float(p[1]), float(p[2])] for p in explored_nodes_list[:500]]

        if path_3d is None or len(path_3d) < 2:
            status = "NO_PATH_FOUND"
            path_coords = []
            print(f"  -> 3D A* Planner: Path blocked / unreachable after exploring {len(explored_nodes_list)} nodes in {t_plan:.2f}ms")
        else:
            status = "PATH_FOUND"
            tracker = PurePursuit3DController(
                waypoints=path_3d,
                v_max=cfg.physics.v_max,
                lookahead_dist=10.0,
                acceptance_radius=cfg.physics.reach_tolerance,
                decel_dist=15.0
            )
            safety = SafetyFilter(obstacles=raw_obs, d_safe=cfg.physics.d_safe, obstacle_margin=margin, bounds=bounds)

            curr_pos = np.copy(start)
            path_coords = [curr_pos.tolist()]
            cum_dist = 0.0

            for s in range(cfg.mappo.episode_length):
                d_final = float(np.linalg.norm(curr_pos - target))
                if d_final <= cfg.physics.reach_tolerance:
                    status = "GOAL_REACHED"
                    break

                next_pos, v_cmd, active_wp_idx, tracker_status = tracker.step(curr_pos, dt=cfg.physics.delta_t)
                safe_dict, interv = safety.filter_trajectories({0: next_pos}, {0: curr_pos}, enable_filter=True)
                actual_pos = safe_dict[0]
                actual_disp = actual_pos - curr_pos
                actual_speed = float(np.linalg.norm(actual_disp)) / cfg.physics.delta_t
                cum_dist += float(np.linalg.norm(actual_disp))
                curr_pos = actual_pos
                path_coords.append(curr_pos.tolist())

                d_target = float(np.linalg.norm(curr_pos - target))
                min_c = compute_min_clearance(curr_pos, raw_obs, margin)
                obs_vec = make_obs_vector(curr_pos, v_cmd, target, raw_obs)

                steps_telemetry.append({
                    "step": s + 1,
                    "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                    "speed": round(actual_speed, 2),
                    "dist_to_target": round(d_target, 2),
                    "min_clearance": round(min_c, 2),
                    "safety_intervened": bool(interv > 0),
                    "explanation": {
                        "planner": "3D A* + Pure Pursuit",
                        "decision_type": "Deterministic 3D Search + Waypoint Lookahead Tracking",
                        "g_cost": round(cum_dist, 2),
                        "h_cost": round(d_target, 2),
                        "f_cost": round(cum_dist + d_target, 2),
                        "direction_chosen": f"Active Waypoint #{active_wp_idx + 1} / {len(path_3d)}",
                        "explored_nodes_count": len(explored_nodes_list),
                        "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                        "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                        "neural_action": [round(float(v_cmd[0]), 2), round(float(v_cmd[1]), 2), round(float(v_cmd[2]), 2)],
                        "executed_control": [round(float(actual_disp[0]), 2), round(float(actual_disp[1]), 2), round(float(actual_disp[2]), 2)]
                    }
                })

                if tracker_status == "GOAL_REACHED" or d_target <= cfg.physics.reach_tolerance:
                    status = "GOAL_REACHED"
                    break
                if tracker_status == "CONTROLLER_STALLED":
                    status = "CONTROLLER_STALLED"
                    break

            if status not in ["GOAL_REACHED", "CONTROLLER_STALLED"]:
                status = "GOAL_REACHED" if float(np.linalg.norm(curr_pos - target)) <= cfg.physics.reach_tolerance else "TIMEOUT / INCOMPLETE"

    elif planner_key == "hybrid_astar_mappo":
        # Hierarchical A* + MAPPO (Global Waypoints + Local Neural Tracking & Task Offloading)
        bounds = (cfg.airspace.x_min, cfg.airspace.x_max, cfg.airspace.y_min, cfg.airspace.y_max, cfg.airspace.z_min, cfg.airspace.z_max)
        planner = AStar3DPlanner(
            obstacles=raw_obs,
            bounds=bounds,
            grid_res=5.0,
            d_safe=cfg.physics.d_safe,
            obstacle_margin=margin
        )
        path_3d, t_plan, explored_nodes_list = planner.plan_path_with_details(start, target)
        explored_nodes = [[float(p[0]), float(p[1]), float(p[2])] for p in explored_nodes_list[:500]]

        if path_3d is None or len(path_3d) < 2:
            status = "NO_PATH_FOUND"
            path_coords = []
            print(f"  -> Hierarchical Hybrid: A* found no global path after exploring {len(explored_nodes_list)} nodes.")
        else:
            status = "PATH_FOUND"
            safety = SafetyFilter(obstacles=raw_obs, d_safe=cfg.physics.d_safe, obstacle_margin=margin, bounds=bounds)
            tracker = PurePursuit3DController(
                waypoints=path_3d,
                v_max=cfg.physics.v_max,
                lookahead_dist=12.0,
                acceptance_radius=8.0,
                decel_dist=15.0
            )
            if not pure_agent_loaded:
                return jsonify({
                    "status": "CHECKPOINT_LOAD_FAILED",
                    "error": f"Hybrid planner requires Pure MAPPO checkpoint: {pure_load_error}. Silent random execution blocked."
                }), 500

            curr_pos = np.copy(start)
            curr_vel = np.zeros(3, dtype=np.float32)
            path_coords = [curr_pos.tolist()]
            total_interventions = 0
            consecutive_stalls = 0

            for s in range(cfg.mappo.episode_length):
                d_final = float(np.linalg.norm(curr_pos - target))
                if d_final <= 8.0:
                    status = "GOAL_REACHED"
                    break

                next_pos, v_cmd, active_wp_idx, tracker_status = tracker.step(curr_pos, dt=cfg.physics.delta_t)
                subgoal = np.array(path_3d[min(active_wp_idx, len(path_3d) - 1)], dtype=np.float32)

                # Subgoal conditioning: feed active A* waypoint as local target into MAPPO
                obs_vec = make_obs_vector(curr_pos, curr_vel, subgoal, raw_obs)
                t_obs = torch.tensor(obs_vec, dtype=torch.float32, device=dev).unsqueeze(0)

                with torch.no_grad():
                    act_move, act_offload, _ = pure_agent.actor.sample_action(t_obs, deterministic=True)
                u_neural = act_move.squeeze(0).cpu().numpy()
                offload_choice = int(act_offload.squeeze(0).cpu().item())

                # Hierarchical coupling: A* global corridor velocity + local MAPPO policy refinement
                u_applied = np.clip(0.85 * v_cmd + 0.15 * u_neural, -cfg.physics.v_max, cfg.physics.v_max)
                if d_final < cfg.physics.v_max * cfg.physics.delta_t:
                    max_step_speed = max(0.5, d_final / cfg.physics.delta_t)
                    speed_mag = float(np.linalg.norm(u_applied))
                    if speed_mag > max_step_speed:
                        u_applied = u_applied * (max_step_speed / speed_mag)

                proposed_pos = curr_pos + u_applied * cfg.physics.delta_t

                filtered_dict, interv = safety.filter_trajectories({0: proposed_pos}, {0: curr_pos}, enable_filter=True)
                filtered_pos = filtered_dict[0]
                actual_disp = filtered_pos - curr_pos

                curr_vel = actual_disp / cfg.physics.delta_t
                curr_pos = filtered_pos
                path_coords.append(curr_pos.tolist())
                total_interventions += interv

                min_c = compute_min_clearance(curr_pos, raw_obs, margin)
                offload_name = "Local On-Board" if offload_choice == 0 else (f"Peer UAV #{offload_choice}" if offload_choice <= 3 else "Ground Edge Server")

                steps_telemetry.append({
                    "step": s + 1,
                    "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                    "speed": round(float(np.linalg.norm(curr_vel)), 2),
                    "dist_to_target": round(float(np.linalg.norm(curr_pos - target)), 2),
                    "min_clearance": round(min_c, 2),
                    "safety_intervened": bool(interv > 0),
                    "explanation": {
                        "planner": "Hierarchical A* + MAPPO",
                        "decision_type": f"Global A* Subgoal #{active_wp_idx}/{len(path_3d) - 1} + Local Policy",
                        "active_subgoal": [round(float(subgoal[0]), 2), round(float(subgoal[1]), 2), round(float(subgoal[2]), 2)],
                        "offload_decision": offload_name,
                        "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                        "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                        "neural_action": [round(float(u_neural[0]), 2), round(float(u_neural[1]), 2), round(float(u_neural[2]), 2)],
                        "executed_control": [round(float(curr_vel[0]), 2), round(float(curr_vel[1]), 2), round(float(curr_vel[2]), 2)],
                        "safety_filter_status": "INTERVENTION TRIGGERED (Continuous boundary deceleration)" if interv > 0 else "Clear flight"
                    }
                })

                if float(np.linalg.norm(curr_pos - target)) <= 8.0:
                    status = "GOAL_REACHED"
                    break

            if status != "GOAL_REACHED":
                status = "GOAL_REACHED" if float(np.linalg.norm(curr_pos - target)) <= 8.0 else ("CONTROLLER_STALLED" if consecutive_stalls >= 5 else "TIMEOUT / INCOMPLETE")

    elif planner_key in ["mappo_pure", "mappo_residual"]:
        if planner_key == "mappo_pure" and not pure_agent_loaded:
            return jsonify({
                "status": "CHECKPOINT_LOAD_FAILED",
                "error": f"MAPPO Pure checkpoint not loaded: {pure_load_error}. Silent random execution blocked."
            }), 500
        if planner_key == "mappo_residual" and not residual_agent_loaded:
            return jsonify({
                "status": "CHECKPOINT_LOAD_FAILED",
                "error": f"MAPPO Residual checkpoint not loaded: {residual_load_error}. Silent random execution blocked."
            }), 500

        agent = pure_agent if planner_key == "mappo_pure" else residual_agent
        agent_name = "Pure MAPPO (No Global Guidance)" if planner_key == "mappo_pure" else "APF-Assisted Residual MAPPO (Ablation)"
        bounds = (cfg.airspace.x_min, cfg.airspace.x_max, cfg.airspace.y_min, cfg.airspace.y_max, cfg.airspace.z_min, cfg.airspace.z_max)
        safety = SafetyFilter(obstacles=raw_obs, d_safe=cfg.physics.d_safe, obstacle_margin=margin, bounds=bounds)

        curr_pos = np.copy(start)
        curr_vel = np.zeros(3, dtype=np.float32)
        path_coords = [curr_pos.tolist()]
        total_interventions = 0
        consecutive_stalls = 0
        status = "PATH_FOUND"

        prev_d_target = float(np.linalg.norm(curr_pos - target))
        for s in range(cfg.mappo.episode_length):
            d_target = float(np.linalg.norm(curr_pos - target))
            if d_target <= cfg.physics.reach_tolerance:
                status = "GOAL_REACHED"
                break

            obs_vec = make_obs_vector(curr_pos, curr_vel, target, raw_obs)
            t_obs = torch.tensor(obs_vec, dtype=torch.float32, device=dev).unsqueeze(0)

            with torch.no_grad():
                act_move, act_offload, _ = agent.actor.sample_action(t_obs, deterministic=True)
            u_neural = act_move.squeeze(0).cpu().numpy()
            offload_choice = int(act_offload.squeeze(0).cpu().item())

            # APF nominal component (if residual ablation)
            u_nom = np.zeros(3, dtype=np.float32)
            if planner_key == "mappo_residual":
                t_rel = (target - curr_pos) / max(1.0, float(np.linalg.norm(target - curr_pos)))
                # Proportional deceleration near goal avoids limit-cycle overshooting
                if d_target < 20.0:
                    f_att = 10.0 * max(0.1, d_target / 20.0) * t_rel
                else:
                    f_att = 10.0 * t_rel

                f_rep = np.zeros(3, dtype=np.float32)
                for obs_item in raw_obs:
                    is_sphere = obs_item.get("type") == "sphere" if isinstance(obs_item, dict) else len(obs_item) == 5
                    if is_sphere:
                        cx, cy, cz, r = obs_item["cx"], obs_item["cy"], obs_item["cz"], obs_item["radius"] if isinstance(obs_item, dict) else (obs_item[0], obs_item[1], obs_item[2], obs_item[3])
                        d_3d = float(np.linalg.norm(curr_pos - np.array([cx, cy, cz]))) - (r + margin)
                        if 0 < d_3d < 25.0:
                            n_3d = (curr_pos - np.array([cx, cy, cz])) / max(1e-4, float(np.linalg.norm(curr_pos - np.array([cx, cy, cz]))))
                            cross_3d = np.cross(t_rel, n_3d)
                            t_tan_3d = np.cross(n_3d, cross_3d / float(np.linalg.norm(cross_3d))) if float(np.linalg.norm(cross_3d)) > 1e-4 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
                            f_rep += (8.0 / (d_3d + 1e-3)) * n_3d + (12.0 / (d_3d + 1.0)) * t_tan_3d
                    else:
                        cx, cy, r, h = obs_item["cx"], obs_item["cy"], obs_item["radius"], obs_item["height"] if isinstance(obs_item, dict) else (obs_item[0], obs_item[1], obs_item[2], obs_item[3])
                        d_c = np.linalg.norm(curr_pos[:2] - np.array([cx, cy])) - (r + margin)
                        if 0 < d_c < 25.0 and curr_pos[2] <= h + 5.0:
                            n_c = (curr_pos[:2] - np.array([cx, cy])) / max(1e-4, np.linalg.norm(curr_pos[:2] - np.array([cx, cy])))
                            cross_2d = t_rel[0] * n_c[1] - t_rel[1] * n_c[0]
                            sign_turn = 1.0 if cross_2d >= 0 else -1.0
                            t_tan = np.array([-n_c[1], n_c[0]], dtype=np.float32) * sign_turn
                            f_rep[:2] += (8.0 / (d_c + 1e-3)) * n_c + (12.0 / (d_c + 1.0)) * t_tan
                u_nom = f_att + f_rep

            # Note: For residual agent, u_neural already incorporates nominal APF inside ActorNetwork
            u_raw = u_neural
            u_applied = np.clip(u_raw, -cfg.physics.v_max, cfg.physics.v_max)

            # Smooth terminal deceleration when near the goal (prevents 15m step overshoot)
            if d_target < cfg.physics.v_max * cfg.physics.delta_t:
                max_step_speed = max(0.5, d_target / cfg.physics.delta_t)
                speed_mag = float(np.linalg.norm(u_applied))
                if speed_mag > max_step_speed:
                    u_applied = u_applied * (max_step_speed / speed_mag)

            proposed_pos = curr_pos + u_applied * cfg.physics.delta_t

            filtered_dict, interv = safety.filter_trajectories({0: proposed_pos}, {0: curr_pos}, enable_filter=True)
            filtered_pos = filtered_dict[0]
            actual_disp = filtered_pos - curr_pos
            step_progress = float(np.linalg.norm(actual_disp))

            # Detect controller stalling or boundary pinning
            is_pinned_to_wall = (
                filtered_pos[0] >= cfg.airspace.x_max - 0.5 or filtered_pos[0] <= cfg.airspace.x_min + 0.5 or
                filtered_pos[1] >= cfg.airspace.y_max - 0.5 or filtered_pos[1] <= cfg.airspace.y_min + 0.5
            )
            dist_change = abs(d_target - prev_d_target)
            if (step_progress < 0.25 and dist_change < 0.2) and d_target > 8.0:
                consecutive_stalls += 1
            elif is_pinned_to_wall:
                consecutive_stalls += 1
            else:
                consecutive_stalls = 0
            prev_d_target = d_target

            curr_vel = actual_disp / cfg.physics.delta_t
            curr_pos = filtered_pos
            path_coords.append(curr_pos.tolist())
            total_interventions += interv

            min_c = compute_min_clearance(curr_pos, raw_obs, margin)
            steps_telemetry.append({
                "step": s + 1,
                "pos": [round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2), round(float(curr_pos[2]), 2)],
                "speed": round(float(np.linalg.norm(curr_vel)), 2),
                "dist_to_target": round(float(np.linalg.norm(curr_pos - target)), 2),
                "min_clearance": round(min_c, 2),
                "safety_intervened": bool(interv > 0),
                "explanation": {
                    "planner": agent_name,
                    "decision_type": "Neural Policy Decision (Reactive)" if planner_key == "mappo_pure" else "Nominal APF + Learned Residual Correction",
                    "observation_target_rel": [round(float(obs_vec[6]), 2), round(float(obs_vec[7]), 2), round(float(obs_vec[8]), 2)],
                    "observation_obstacle_rel": [round(float(obs_vec[17]), 2), round(float(obs_vec[18]), 2), round(float(obs_vec[19]), 2)],
                    "neural_action": [round(float(u_neural[0]), 2), round(float(u_neural[1]), 2), round(float(u_neural[2]), 2)],
                    "apf_nominal": [round(float(u_nom[0]), 2), round(float(u_nom[1]), 2), round(float(u_nom[2]), 2)] if planner_key == "mappo_residual" else None,
                    "executed_control": [round(float(curr_vel[0]), 2), round(float(curr_vel[1]), 2), round(float(curr_vel[2]), 2)],
                    "safety_filter_status": "COLLISION AVOIDANCE ACTIVE" if interv > 0 else "Clear flight"
                }
            })

            if consecutive_stalls >= 5:
                status = "COLLISION_AVOIDANCE_BLOCKED" if is_pinned_to_wall else "CONTROLLER_STALLED"
                break

        t_plan = (time.time() - t_start) * 1000.0
        final_d = float(np.linalg.norm(curr_pos - target))
        if final_d <= 8.0:
            status = "GOAL_REACHED"
        elif status == "PATH_FOUND":
            status = "TIMEOUT / INCOMPLETE"
        print(f"  -> {agent_name}: Evaluated {len(path_coords)} neural policy steps | Status: {status} | Interventions: {total_interventions} | Final Dist: {final_d:.2f}m")

    # Compute trajectory total length
    p_arr = np.array(path_coords)
    total_len = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
    all_clearances = [compute_min_clearance(p, raw_obs, margin) for p in p_arr]
    min_overall_clearance = float(min(all_clearances)) if all_clearances else 0.0

    return jsonify({
        "status": status,
        "planner": planner_key,
        "path": path_coords,
        "explored_nodes": explored_nodes,
        "total_path_length": round(total_len, 2),
        "planning_time_ms": round(t_plan, 2),
        "min_clearance": round(min_overall_clearance, 2),
        "steps_count": len(path_coords) - 1,
        "steps_telemetry": steps_telemetry
    })

@app.route("/api/compare", methods=["POST"])
def compare_planners():
    """
    Executes 3D A*, Pure MAPPO, and Residual MAPPO on identical world layout
    and returns a side-by-side scientific research comparison.
    """
    data = request.json or {}
    start = data.get("start", [20.0, 20.0, 30.0])
    target = data.get("target", [180.0, 180.0, 30.0])
    raw_obs = data.get("obstacles", [])
    margin = float(data.get("margin", 2.0))

    planners = [
        ("Hierarchical A* + MAPPO", "hybrid_astar_mappo"),
        ("3D A* Baseline", "astar"),
        ("APF-Residual MAPPO", "mappo_residual"),
        ("Pure MAPPO", "mappo_pure")
    ]

    results = []
    for label, p_key in planners:
        sub_req = {"planner": p_key, "start": start, "target": target, "obstacles": raw_obs, "margin": margin}
        with app.test_request_context(json=sub_req):
            res_json = plan_path().get_json()
            interv_count = sum(1 for s in res_json.get("steps_telemetry", []) if s.get("safety_intervened"))
            results.append({
                "algorithm": label,
                "key": p_key,
                "status": res_json.get("status"),
                "path_length_m": res_json.get("total_path_length"),
                "execution_steps": res_json.get("steps_count"),
                "planning_time_ms": res_json.get("planning_time_ms"),
                "min_clearance_m": res_json.get("min_clearance"),
                "safety_interventions": interv_count,
                "path": res_json.get("path"),
                "steps_telemetry": res_json.get("steps_telemetry", []),
                "explored_nodes": res_json.get("explored_nodes", [])
            })

    return jsonify({"comparison": results})

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" 3D UAV PATH-PLANNING RESEARCH SIMULATOR BACKEND")
    print(" Serving WebGL Dashboard at: http://127.0.0.1:5000")
    print("=" * 80)
    app.run(host="127.0.0.1", port=5000, debug=False)
