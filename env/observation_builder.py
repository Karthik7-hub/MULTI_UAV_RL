"""
Unified Observation Builder and Architectural Schema for Multi-UAV Systems.
Single source of truth for observation vector construction, feature ordering,
normalization constants, and decoupled navigation/offloading stream indices.

Canonical 22-Dimensional Feature Schema per UAV:
------------------------------------------------------------------------------------------
Index      | Feature Description                     | Units / Range     | Normalization
------------------------------------------------------------------------------------------
obs[0:3]   | 3D Position Coordinates (x, y, z)       | meters [0, 200]   | / 200.0 m (Airspace bound)
obs[3:6]   | 3D Flight Velocity (vx, vy, vz)         | m/s [-15, 15]     | / 15.0 m/s (v_max)
obs[6:9]   | Relative Vector to Station (dx, dy, dz) | meters [-200, 200]| / 200.0 m
obs[9]     | Euclidean Distance to Station (d_s)     | meters [0, 283]   | / 200.0 m
obs[10]    | Computational Task CPU Cycles           | cycles [0, 1.5e9] | / 1.5e9 cycles
obs[11]    | Computational Task Data Size            | bits [0, 3e6]     | / 3.0e6 bits
obs[12]    | Computational Task Latency Deadline     | seconds [0, 3]    | / 3.0 s
obs[13]    | Wireless Channel Rate to Edge Server    | bps [0, 1e8]      | / 1.0e8 bps (100 Mbps)
obs[14:17] | Relative Distances to Peer UAVs (x3)    | meters [0, 200]   | / 200.0 m
obs[17:19] | 2D Direction Vector to Closest Obstacle | unit vector       | towards center [dx/d, dy/d]
obs[19]    | Distance to Closest Obstacle Surface    | meters [-50, 50]  | / 50.0 m (clipped [-1, 1])
obs[20:22] | 2D Obstacle Clearance Tangent Vector    | unit vector [0, 1]| lateral detour direction [tx, ty]
------------------------------------------------------------------------------------------
Total dimension: 3 + 3 + 3 + 1 + 3 + 1 + 3 + 3 + 2 = 22 floats.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# =============================================================================
# AUTHORITATIVE ARCHITECTURAL CONSTANTS & FEATURE INDICES
# =============================================================================
OBS_DIM: int = 22

# Feature Slices and Indices
OBS_POS_SLICE = slice(0, 3)
OBS_VEL_SLICE = slice(3, 6)
OBS_STATION_VEC_SLICE = slice(6, 9)
OBS_STATION_DIST_IDX = 9
OBS_TASK_CYCLES_IDX = 10
OBS_TASK_DATA_IDX = 11
OBS_TASK_DEADLINE_IDX = 12
OBS_TASK_SLICE = slice(10, 13)
OBS_CHANNEL_RATE_IDX = 13
OBS_PEER_DISTS_SLICE = slice(14, 17)
OBS_OBSTACLE_SLICE = slice(17, 20)
OBS_TANGENT_SLICE = slice(20, 22)

# Two-Stream Decoupled Architecture Slices
# Navigation Stream: Spatial kinematics, station vector/distance, peer distances, obstacle vector/distance, clearance tangent
NAV_FEATURE_INDICES: List[int] = list(range(0, 10)) + list(range(14, 22))
NAV_DIM: int = len(NAV_FEATURE_INDICES)  # Exactly 18 dimensions

# Offloading Stream: Spatial coordinates, task attributes, wireless channel rate, peer distances
OFFLOAD_FEATURE_INDICES: List[int] = list(range(0, 3)) + list(range(10, 17))
OFFLOAD_DIM: int = len(OFFLOAD_FEATURE_INDICES)  # Exactly 10 dimensions

# Centralized Global State Schema per UAV (Canonical 13 Dimensions):
# Includes explicit 3D relative goal vectors (dx, dy, dz) to allow centralized policies (e.g., CPPO)
# to distinguish target directions at identical Euclidean distances:
# - pos[0:3]          : 3D Position Coordinates (x, y, z) / 200.0 m
# - vel[3:6]          : 3D Flight Velocity (vx, vy, vz) / 15.0 m/s
# - rel_station[6:9]  : 3D Relative Vector to Target Station (dx, dy, dz) / 200.0 m
# - station_dist[9]   : Euclidean Distance to Target Station / 200.0 m
# - task[10:13]       : Computational Task Attributes (cycles, data, deadline)
STATE_DIM_PER_UAV: int = 13
LEGACY_STATE_DIM_PER_UAV: int = 10


def build_centralized_state(
    positions: Dict[int, np.ndarray],
    velocities: Dict[int, np.ndarray],
    stations: Dict[int, np.ndarray],
    current_tasks: Dict[int, Any],
    num_uavs: int = 4,
    airspace_bound: float = 200.0,
    v_max: float = 15.0
) -> np.ndarray:
    """
    Constructs the canonical centralized global state vector across all UAVs.
    Guarantees that states differing only in goal direction produce distinct vectors.
    """
    parts = []
    for j in range(num_uavs):
        p_j = np.array(positions[j], dtype=np.float32) / airspace_bound
        v_j = np.array(velocities[j], dtype=np.float32) / v_max
        st_j = np.array(stations[j], dtype=np.float32)
        rel_g = (st_j - np.array(positions[j], dtype=np.float32)) / airspace_bound
        d_s = np.array([np.linalg.norm(positions[j] - st_j) / airspace_bound], dtype=np.float32)
        task = current_tasks.get(j)
        if task is not None:
            t_f = np.array([
                task.cycles / 1.5e9,
                task.data_size / 3.0e6,
                task.deadline / 3.0
            ], dtype=np.float32)
        else:
            t_f = np.zeros(3, dtype=np.float32)
        parts.extend([p_j, v_j, rel_g, d_s, t_f])
    return np.concatenate(parts).astype(np.float32)


def compute_data_rate_to_edge(
    pos: np.ndarray,
    edge_pos: np.ndarray,
    bandwidth: float = 20e6,
    tx_power: float = 0.1,
    noise_power_dbm: float = -100.0,
    beta_0_db: float = -30.0,
    num_concurrent: int = 1
) -> float:
    """Computes free-space RF data rate to Ground Edge Server in bps using Shannon channel model."""
    dist = max(1.0, float(np.linalg.norm(pos - edge_pos)))
    beta_0 = 10.0 ** (beta_0_db / 10.0)
    channel_gain = beta_0 / (dist ** 2)
    noise_w = (10.0 ** (noise_power_dbm / 10.0)) * 1e-3
    snr = (tx_power * channel_gain) / max(1e-15, noise_w)
    effective_bw = bandwidth / max(1, num_concurrent)
    rate = effective_bw * np.log2(1.0 + snr)
    return float(rate)


def compute_obstacle_clearance_tangent(
    pos: np.ndarray,
    station_pos: np.ndarray,
    obstacles: List[Any],
    max_influence_dist: float = 25.0
) -> np.ndarray:
    """
    Computes an explicit 2D unit tangent clearance vector around the nearest obstacle
    in the direction of the shortest and safest lateral bypass toward the target station.

    Bidirectional Detour Geometry:
        - Let r_vec be the unit vector from UAV to obstacle center in XY plane.
        - Let u_goal be the unit vector from UAV toward the target station in XY plane.
        - The 2D cross product: cross = u_goal_x * r_vec_y - u_goal_y * r_vec_x
          * cross > 0: obstacle is to the LEFT of the flight path -> steer RIGHT (-tangent)
          * cross < 0: obstacle is to the RIGHT of the flight path -> steer LEFT (+tangent)
          * cross == 0: obstacle directly blocks flight path -> tie-break to steer right
        - Invariance conditions:
          * Obstacle behind flight path (u_goal . r_vec <= 0.0) -> return [0, 0]
          * UAV altitude cleanly above obstacle top (z > height + 5.0m) -> return [0, 0]
          * UAV surface distance > max_influence_dist (default 25.0m) -> return [0, 0]

    Returns:
        np.ndarray of shape (2,) with float32 values. Norm is strictly bounded in [0.0, 1.0].
    """
    pos_xy = np.asarray(pos[:2], dtype=np.float32)
    pos_z = float(pos[2]) if len(pos) > 2 else 50.0
    station_xy = np.asarray(station_pos[:2], dtype=np.float32)

    min_d_surf = float("inf")
    closest_obs_info = None

    for obs in obstacles:
        if isinstance(obs, dict):
            if obs.get("type") == "sphere":
                cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                h = cz + r
            else:
                cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
        else:
            cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]

        c_xy = np.array([cx, cy], dtype=np.float32)
        d_xy = float(np.linalg.norm(pos_xy - c_xy))
        d_surf = d_xy - r

        if d_surf < min_d_surf:
            min_d_surf = d_surf
            closest_obs_info = (cx, cy, r, h, d_xy, d_surf)

    if closest_obs_info is None or min_d_surf > max_influence_dist:
        return np.zeros(2, dtype=np.float32)

    cx, cy, r, h, d_xy, d_surf = closest_obs_info

    # If UAV flies cleanly above obstacle altitude (+ margin), no horizontal detour needed
    if pos_z > (h + 5.0):
        return np.zeros(2, dtype=np.float32)

    # Unit vector from UAV to obstacle center
    r_vec = (np.array([cx, cy], dtype=np.float32) - pos_xy) / max(1e-6, d_xy)

    # Unit vector from UAV toward station
    vec_goal = station_xy - pos_xy
    d_goal = float(np.linalg.norm(vec_goal))
    if d_goal < 1e-4:
        return np.zeros(2, dtype=np.float32)
    u_goal = vec_goal / d_goal

    # If obstacle is behind the UAV flight path (u_goal . r_vec <= 0.0), no detour needed
    if float(np.dot(u_goal, r_vec)) <= 0.0:
        return np.zeros(2, dtype=np.float32)

    # 2D cross product: u_goal_x * r_y - u_goal_y * r_x
    cross = float(u_goal[0] * r_vec[1] - u_goal[1] * r_vec[0])
    # cross >= 0 => obstacle to the left => steer right (-tangent)
    # cross < 0 => obstacle to the right => steer left (+tangent)
    sign = -1.0 if cross >= 0.0 else 1.0

    # Perpendicular tangent to r_vec: [-r_y, r_x]
    tangent = sign * np.array([-r_vec[1], r_vec[0]], dtype=np.float32)
    norm = float(np.linalg.norm(tangent))
    if norm > 1e-6:
        tangent = tangent / norm
    else:
        tangent = np.zeros(2, dtype=np.float32)

    return tangent


def build_single_uav_obs(
    pos: np.ndarray,
    vel: np.ndarray,
    station_pos: np.ndarray,
    edge_pos: np.ndarray,
    obstacles: List[Any],
    task_features: Optional[Tuple[float, float, float]] = None,
    peer_positions: Optional[List[np.ndarray]] = None,
    comm_model: Optional[Any] = None,
    v_max: float = 15.0,
    airspace_bound: float = 200.0,
    num_peers_expected: int = 3
) -> np.ndarray:
    """
    Constructs the canonical 22-dimensional observation vector for a single UAV.

    Parameters:
        pos: (3,) array of UAV coordinates [x, y, z] in meters.
        vel: (3,) array of UAV velocity [vx, vy, vz] in m/s.
        station_pos: (3,) array of destination station coordinates in meters.
        edge_pos: (3,) array of Ground Edge Server coordinates in meters.
        obstacles: List of cylindrical obstacle specs (cx, cy, radius, height) or dicts.
        task_features: Optional (cycles, data_size, deadline). If None, defaults to [0.0, 0.0, 0.0].
        peer_positions: Optional list of other UAV positions (each shape (3,)).
        comm_model: Optional wireless channel model instance. If None, computes analytical free-space rate.
        v_max: Maximum physical flight speed (m/s).
        airspace_bound: Maximum horizontal airspace dimension for normalization (meters).
        num_peers_expected: Number of peer distance features (default: 3 for N=4 fleet).

    Returns:
        np.ndarray of shape (22,) with dtype float32.
    """
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    station_pos = np.asarray(station_pos, dtype=np.float32)
    edge_pos = np.asarray(edge_pos, dtype=np.float32)

    # 1. Kinematics (6 floats)
    p_norm = pos / airspace_bound
    v_norm = vel / v_max

    # 2. Target Station Vector and Distance (4 floats)
    vec_s = station_pos - pos
    d_s = float(np.linalg.norm(vec_s))
    vec_s_norm = vec_s / airspace_bound
    d_s_norm = np.array([d_s / airspace_bound], dtype=np.float32)

    # 3. Computational Task Features (3 floats)
    if task_features is not None:
        cycles, data_size, deadline = task_features
        t_feat = np.array([
            cycles / 1.5e9,
            data_size / 3.0e6,
            deadline / 3.0
        ], dtype=np.float32)
    else:
        t_feat = np.zeros(3, dtype=np.float32)

    # 4. Wireless Channel to Ground Edge Server (1 float)
    d_edge = float(np.linalg.norm(pos - edge_pos))
    if comm_model is not None and hasattr(comm_model, "data_rate"):
        rate_bps = comm_model.data_rate(d_edge)
    else:
        rate_bps = compute_data_rate_to_edge(pos, edge_pos)
    ch_feat = np.array([rate_bps / 1.0e8], dtype=np.float32)

    # 5. Relative Swarm Peer Distances (num_peers_expected floats)
    peer_dists = []
    if peer_positions is not None and len(peer_positions) > 0:
        for p_k in peer_positions:
            d_p = float(np.linalg.norm(pos - np.asarray(p_k, dtype=np.float32)))
            peer_dists.append(d_p / airspace_bound)
    while len(peer_dists) < num_peers_expected:
        peer_dists.append(1.0)
    peer_feat = np.array(peer_dists[:num_peers_expected], dtype=np.float32)

    # 6. Closest Obstacle Direction and Surface Clearance (3 floats)
    min_obs_d = float("inf")
    closest_dir = np.zeros(2, dtype=np.float32)

    for obs in obstacles:
        if isinstance(obs, dict):
            if obs.get("type") == "sphere":
                cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                d_3d = float(np.linalg.norm(pos - np.array([cx, cy, cz])))
                d_surf = d_3d - r
                if d_surf < min_obs_d:
                    min_obs_d = d_surf
                    d_xy = float(np.linalg.norm(pos[:2] - np.array([cx, cy])))
                    if d_xy > 1e-4:
                        closest_dir = (np.array([cx, cy], dtype=np.float32) - pos[:2]) / d_xy
                continue
            else:
                cx, cy, r, h = obs["cx"], obs["cy"], obs["radius"], obs["height"]
        else:
            cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]

        d_xy = float(np.linalg.norm(pos[:2] - np.array([cx, cy])))
        d_surf = d_xy - r
        if d_surf < min_obs_d:
            min_obs_d = d_surf
            if d_xy > 1e-4:
                closest_dir = (np.array([cx, cy], dtype=np.float32) - pos[:2]) / d_xy

    surf_dist_norm = float(np.clip(min_obs_d / 50.0, -1.0, 1.0))
    obs_feat = np.array([closest_dir[0], closest_dir[1], surf_dist_norm], dtype=np.float32)

    # 7. Obstacle Clearance Tangent Vector (2 floats)
    tangent_feat = compute_obstacle_clearance_tangent(pos, station_pos, obstacles)

    obs_vec = np.concatenate([
        p_norm,
        v_norm,
        vec_s_norm,
        d_s_norm,
        t_feat,
        ch_feat,
        peer_feat,
        obs_feat,
        tangent_feat
    ]).astype(np.float32)

    assert obs_vec.shape == (OBS_DIM,), f"Observation vector shape mismatch: expected ({OBS_DIM},), got {obs_vec.shape}"
    return obs_vec
