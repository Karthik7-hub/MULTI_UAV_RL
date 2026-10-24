"""
Shared Observation Builder for Multi-UAV Navigation and Task Offloading.
Provides a unified contract for observation vector construction across both
gym environment training (MultiUAVEnv) and runtime inference (simulator_server.py).

Guarantees identical feature ordering, scaling, normalization, and bounds:
-------------------------------------------------------------------------
Index   | Feature Description                  | Units / Range | Normalization
-------------------------------------------------------------------------
obs[0:3]| 3D Position (x, y, z)                | meters [0,200]| / 200.0 m (Airspace bound)
obs[3:6]| 3D Velocity (vx, vy, vz)             | m/s [-15, 15] | / 15.0 m/s (v_max)
obs[6:9]| Relative Vector to Station (dx,dy,dz)| meters [-200,200]| / 200.0 m
obs[9]  | Euclidean Distance to Station (d_s)  | meters [0, 283]| / 200.0 m
obs[10] | Task Required CPU Cycles             | cycles [0, 1.5e9]| / 1.5e9 cycles
obs[11] | Task Data Size                       | bits [0, 3e6] | / 3.0e6 bits
obs[12] | Task Latency Deadline                | seconds [0, 3]| / 3.0 s
obs[13] | Wireless Channel Rate to Edge Server | bps [0, 1e8]  | / 1.0e8 bps (100 Mbps)
obs[14:17]| Relative Distances to Peer UAVs (x3)| meters [0, 200]| / 200.0 m
obs[17:19]| 2D Direction Vector to Closest Obs | unit vector   | (dx/d, dy/d) towards center
obs[19] | Distance to Closest Obstacle Surface | meters [-50,50]| / 50.0 m (clipped [-1, 1])
-------------------------------------------------------------------------
Total dimension: 3 + 3 + 3 + 1 + 3 + 1 + 3 + 3 = 20 floats.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def compute_data_rate_to_edge(
    pos: np.ndarray,
    edge_pos: np.ndarray,
    bandwidth: float = 20e6,
    tx_power: float = 0.1,
    noise_power_dbm: float = -100.0,
    beta_0_db: float = -30.0,
    num_concurrent: int = 1
) -> float:
    """Computes free-space RF data rate to edge server in bps."""
    dist = max(1.0, float(np.linalg.norm(pos - edge_pos)))
    beta_0 = 10.0 ** (beta_0_db / 10.0)
    channel_gain = beta_0 / (dist ** 2)
    noise_w = (10.0 ** (noise_power_dbm / 10.0)) * 1e-3
    snr = (tx_power * channel_gain) / max(1e-15, noise_w)
    effective_bw = bandwidth / max(1, num_concurrent)
    rate = effective_bw * np.log2(1.0 + snr)
    return float(rate)

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
    Constructs the canonical 20-dimensional observation vector for a single UAV.

    Parameters:
        pos: (3,) array of UAV coordinates [x, y, z] in meters.
        vel: (3,) array of UAV velocity [vx, vy, vz] in m/s.
        station_pos: (3,) array of destination station coordinates in meters.
        edge_pos: (3,) array of edge server coordinates in meters.
        obstacles: List of cylindrical obstacle specs (cx, cy, radius, height) or dicts.
        task_features: Optional (cycles, data_size, deadline). If None, defaults to [0.0, 0.0, 0.0].
        peer_positions: Optional list of other UAV positions (each shape (3,)).
        comm_model: Optional wireless channel model instance. If None, computes analytical free-space rate.
        v_max: Maximum physical flight speed (m/s).
        airspace_bound: Maximum horizontal airspace dimension for normalization (meters).
        num_peers_expected: Number of peer distance features (default 3 for N=4 UAV swarm).

    Returns:
        np.ndarray of shape (20,) with dtype float32.
    """
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    station_pos = np.asarray(station_pos, dtype=np.float32)
    edge_pos = np.asarray(edge_pos, dtype=np.float32)

    # 1. Kinematics (6)
    p_norm = pos / airspace_bound
    v_norm = vel / v_max

    # 2. Target Station Vector and Distance (4)
    vec_s = station_pos - pos
    d_s = float(np.linalg.norm(vec_s))
    vec_s_norm = vec_s / airspace_bound
    d_s_norm = np.array([d_s / airspace_bound], dtype=np.float32)

    # 3. Task Features (3)
    if task_features is not None:
        cycles, data_size, deadline = task_features
        t_feat = np.array([
            cycles / 1.5e9,
            data_size / 3.0e6,
            deadline / 3.0
        ], dtype=np.float32)
    else:
        t_feat = np.zeros(3, dtype=np.float32)

    # 4. Wireless Channel to Edge Server (1)
    d_edge = float(np.linalg.norm(pos - edge_pos))
    if comm_model is not None and hasattr(comm_model, "data_rate"):
        rate_bps = comm_model.data_rate(d_edge)
    else:
        rate_bps = compute_data_rate_to_edge(pos, edge_pos)
    ch_feat = np.array([rate_bps / 1.0e8], dtype=np.float32)

    # 5. Relative Peer Distances (num_peers_expected)
    peer_dists = []
    if peer_positions is not None and len(peer_positions) > 0:
        for p_k in peer_positions:
            d_p = float(np.linalg.norm(pos - np.asarray(p_k, dtype=np.float32)))
            peer_dists.append(d_p / airspace_bound)
    # Pad or truncate to exactly num_peers_expected
    while len(peer_dists) < num_peers_expected:
        # Documented default: 1.0 represents nominal uncrowded separation (full airspace scale)
        peer_dists.append(1.0)
    peer_feat = np.array(peer_dists[:num_peers_expected], dtype=np.float32)

    # 6. Closest Obstacle Direction and Surface Distance (3)
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

    obs_vec = np.concatenate([
        p_norm,
        v_norm,
        vec_s_norm,
        d_s_norm,
        t_feat,
        ch_feat,
        peer_feat,
        obs_feat
    ]).astype(np.float32)

    assert obs_vec.shape == (20,), f"Expected obs shape (20,), got {obs_vec.shape}"
    return obs_vec
