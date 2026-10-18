"""
Gymnasium Multi-Agent Environment for Joint Task Offloading and 3D Path Planning.
Integrates aerodynamics, RF wireless communications, MEC computation queues, and kinematic safety.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional

from config import MasterConfig
from env.aerodynamics import AerodynamicsModel
from env.comms_compute import WirelessChannelModel, ComputationManager, TaskGenerator
from env.safety_filter import SafetyFilter

class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[MasterConfig] = None):
        super().__init__()
        self.cfg = cfg or MasterConfig()
        self.num_uavs = self.cfg.physics.num_uavs
        self.episode_length = self.cfg.mappo.episode_length
        self.current_step = 0

        # Sub-modules
        self.aero = AerodynamicsModel(
            P_0=self.cfg.physics.P_0,
            P_i=self.cfg.physics.P_i,
            U_tip=self.cfg.physics.U_tip,
            v_0=self.cfg.physics.v_0,
            d_0=self.cfg.physics.d_0,
            rho=self.cfg.physics.rho,
            s=self.cfg.physics.s,
            A=self.cfg.physics.A,
            v_max=self.cfg.physics.v_max
        )
        self.comm = WirelessChannelModel(
            bandwidth=self.cfg.comm.bandwidth,
            tx_power=self.cfg.comm.tx_power,
            noise_power_dbm=self.cfg.comm.noise_power_dbm,
            beta_0_db=self.cfg.comm.beta_0_db
        )
        self.comp = ComputationManager(
            f_local=self.cfg.comp.f_local,
            f_peer=self.cfg.comp.f_peer,
            f_edge=self.cfg.comp.f_edge,
            kappa=self.cfg.comp.kappa
        )
        self.task_gen = TaskGenerator(
            poisson_lambda=self.cfg.task.poisson_lambda,
            cycles_range=(self.cfg.task.cycles_min, self.cfg.task.cycles_max),
            data_range=(self.cfg.task.data_size_min, self.cfg.task.data_size_max),
            deadline_range=(self.cfg.task.deadline_min, self.cfg.task.deadline_max)
        )
        self.safety = SafetyFilter(
            obstacles=self.cfg.airspace.obstacles,
            d_safe=self.cfg.physics.d_safe,
            obstacle_margin=self.cfg.physics.obstacle_margin,
            bounds=(
                self.cfg.airspace.x_min, self.cfg.airspace.x_max,
                self.cfg.airspace.y_min, self.cfg.airspace.y_max,
                self.cfg.airspace.z_min, self.cfg.airspace.z_max
            )
        )

        # Action Space per UAV:
        # 'move': continuous 3D velocity displacement [dx, dy, dz] bounded by v_max * dt
        # 'offload': discrete target [0=Local, 1..N-1=Peer UAV, N=Edge Server]
        max_disp = self.cfg.physics.v_max * self.cfg.physics.delta_t
        self.action_spaces = {
            j: spaces.Dict({
                "move": spaces.Box(low=-max_disp, high=max_disp, shape=(3,), dtype=np.float32),
                "offload": spaces.Discrete(self.num_uavs + 1)
            }) for j in range(self.num_uavs)
        }

        # Local Observation Space per UAV:
        # position (3) + velocity (3) + vector_to_station (3) + dist_to_station (1) +
        # task_features (3) + channel_to_edge (1) + relative_peer_distances (N-1) +
        # closest_obstacle_vector_and_dist (3)
        self.obs_dim = 3 + 3 + 3 + 1 + 3 + 1 + (self.num_uavs - 1) + 3
        self.observation_spaces = {
            j: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for j in range(self.num_uavs)
        }

        # Centralized Critic Global State Space:
        # All positions (N*3) + All velocities (N*3) + All distances (N*1) + All tasks (N*3)
        self.state_dim = self.num_uavs * (3 + 3 + 1 + 3)

        # Designated symmetric mission stations (station_radius around center at 50m altitude)
        cx, cy, cz = self.cfg.airspace.mission_center
        self.stations = {}
        for j in range(self.num_uavs):
            angle = j * (2.0 * np.pi / self.num_uavs)
            self.stations[j] = np.array([
                cx + self.cfg.physics.station_radius * np.cos(angle),
                cy + self.cfg.physics.station_radius * np.sin(angle),
                cz
            ], dtype=np.float32)

        # Analytical station safety verification
        assert self.verify_station_feasibility(), "Critical configuration error: station is inside an obstacle or outside airspace bounds!"
        self.reset(seed=42)

    def verify_station_feasibility(self) -> bool:
        """Verifies that all designated mission stations are outside all obstacle cylinders and within airspace bounds."""
        p_cfg = self.cfg.physics
        a_cfg = self.cfg.airspace
        for j, pos in self.stations.items():
            if not (a_cfg.x_min <= pos[0] <= a_cfg.x_max and
                    a_cfg.y_min <= pos[1] <= a_cfg.y_max and
                    a_cfg.z_min <= pos[2] <= a_cfg.z_max):
                return False
            for cx, cy, r, h in a_cfg.obstacles:
                d_xy = np.linalg.norm(pos[:2] - np.array([cx, cy]))
                if d_xy < (r + p_cfg.obstacle_margin) and pos[2] <= h:
                    return False
        return True

    def reset(self, seed: Optional[int] = None, scenario_mode: str = "default") -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(42)

        self.current_step = 0
        cx, cy, cz = self.cfg.airspace.mission_center
        init_radius = self.cfg.physics.start_radius  # Initial deployment ring

        self.positions = {}
        self.velocities = {}
        for j in range(self.num_uavs):
            angle = j * (2.0 * np.pi / self.num_uavs)
            if scenario_mode == "randomized":
                # Controlled perturbation for held-out evaluation scenarios
                angle_offset = float(self.rng.uniform(-0.1, 0.1))
                rad_offset = float(self.rng.uniform(-5.0, 5.0))
                self.positions[j] = np.array([
                    cx + (init_radius + rad_offset) * np.cos(angle + angle_offset),
                    cy + (init_radius + rad_offset) * np.sin(angle + angle_offset),
                    cz
                ], dtype=np.float32)
            else:
                self.positions[j] = np.array([
                    cx + init_radius * np.cos(angle),
                    cy + init_radius * np.sin(angle),
                    cz
                ], dtype=np.float32)
            self.velocities[j] = np.zeros(3, dtype=np.float32)

        self.current_tasks = {j: self.task_gen.sample_task(j, self.current_step, self.rng) for j in range(self.num_uavs)}
        self._prev_d_stations = {j: float(np.linalg.norm(self.positions[j] - self.stations[j])) for j in range(self.num_uavs)}
        return self._get_observations(), self._get_global_state()

    def step(self, actions: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        """
        # =========================================================================
        # PURPOSE:
        # Executes one discrete physical and computational environment step (delta_t = 1.0s).
        #
        # FULL SEQUENCE OF EVENTS DURING ONE STEP:
        # 1. Kinematic Velocity & Proposed Displacement:
        #    Extracts 3D continuous movement action for each UAV, clamps magnitude to v_max * delta_t.
        #    p_proposed = p_current + displacement
        #
        # 2. Hard Kinematic Safety Projection Filter:
        #    Checks p_proposed against all 4 cylinder obstacles, inter-UAV separation (d_safe=8m),
        #    and airspace bounds. Projects any violating coordinates back to the safe boundary.
        #
        # 3. Aerodynamics & Mechanical Propulsion Energy:
        #    Actual velocity v = (p_safe - p_current) / delta_t.
        #    Calculates propulsion power P(v) using Zeng et al. (2019) formula and converts to
        #    propulsion energy E_prop = P(v) * delta_t [Joules].
        #
        # 4. Task Offloading, Channel Contention, & Computation:
        #    Gathers offloading decisions across all 4 UAVs (Local, Peer, or Edge).
        #    - Edge: Computes RF data rate with OFDMA subchannel sharing (B / K_tx).
        #            Edge CPU is sliced among all concurrent offloaders (f_edge / K_edge).
        #    - Peer: Cooperatively offloads to an adjacent UAV; shares CPU if peer is busy.
        #    - Local: Evaluates on-board CPU processing at 1.5 GHz.
        #    Checks end-to-end delay (t_tx + t_comp) against task deadline.
        #
        # 5. Composite Reward Calculation:
        #    Evaluates multi-objective performance (task completion, deadlines, energy, delay,
        #    safety interventions, goal distance progress, and anti-loitering damping).
        #
        # 6. Task Regeneration & State Transition:
        #    Samples new stochastic Poisson tasks for the next step, increments step counter,
        #    and constructs new observation and global state vectors.
        # =========================================================================
        """
        self.current_step += 1
        p_cfg = self.cfg.physics

        # ---------------------------------------------------------------------
        # 1. Kinematic Velocity and Displacement Integration
        # ---------------------------------------------------------------------
        proposed_pos = {}
        for j in range(self.num_uavs):
            disp = np.array(actions[j]["move"], dtype=np.float32)
            disp_norm = np.linalg.norm(disp)
            # Enforce physical maximum flight speed: max displacement per second is v_max * delta_t
            if disp_norm > p_cfg.v_max * p_cfg.delta_t:
                disp = disp * (p_cfg.v_max * p_cfg.delta_t / max(1e-6, disp_norm))
            proposed_pos[j] = self.positions[j] + disp

        # ---------------------------------------------------------------------
        # 2. Safety Projection Filter (Guarantees Collision-Free Trajectories)
        # ---------------------------------------------------------------------
        safe_pos, interventions = self.safety.filter_trajectories(proposed_pos, self.positions)

        # ---------------------------------------------------------------------
        # 3. Update Speeds and Calculate Mechanical Propulsion Power
        # ---------------------------------------------------------------------
        step_propulsion_energy = 0.0
        speeds = {}
        for j in range(self.num_uavs):
            actual_disp = safe_pos[j] - self.positions[j]
            self.velocities[j] = actual_disp / p_cfg.delta_t
            speed = float(np.linalg.norm(self.velocities[j]))
            speeds[j] = speed
            self.positions[j] = safe_pos[j]
            # Accumulate aerodynamic propulsion energy E = P(v) * delta_t in Joules
            step_propulsion_energy += self.aero.energy(speed, p_cfg.delta_t)

        # ---------------------------------------------------------------------
        # 4. Process Computation Tasks and Offloading Decisions with Multi-User Contention
        # ---------------------------------------------------------------------
        tasks_completed = 0
        deadline_misses = 0
        total_delay = 0.0
        step_comm_energy = 0.0
        step_comp_energy = 0.0
        edge_pos = np.array(self.cfg.airspace.edge_server_pos)

        # Gather fleet-wide offload intents to calculate channel & CPU contention
        edge_offloaders = []
        peer_offloaders = []  # (src_j, target_peer_j)
        local_executors = []
        for j in range(self.num_uavs):
            if self.current_tasks[j] is None:
                continue
            target = int(actions[j]["offload"])
            if target == 0:
                local_executors.append(j)
            elif 1 <= target < self.num_uavs:
                peer_idx = (j + target) % self.num_uavs
                peer_offloaders.append((j, peer_idx))
            else:
                edge_offloaders.append(j)

        num_edge_tasks = max(1, len(edge_offloaders))
        total_tx_nodes = max(1, len(edge_offloaders) + len(peer_offloaders))

        for j in range(self.num_uavs):
            task = self.current_tasks[j]
            if task is None:
                continue

            target = int(actions[j]["offload"])
            if target == 0:
                # Local compute: UAV processes on its own CPU (1.5 GHz)
                res = self.comp.compute_local(task)
                step_comp_energy += res["energy"]
            elif 1 <= target < self.num_uavs:
                # Peer offload: cooperative execution with channel sharing
                peer_idx = (j + target) % self.num_uavs
                peer_busy = peer_idx in local_executors
                d_peer = np.linalg.norm(self.positions[j] - self.positions[peer_idx])
                res = self.comp.compute_peer(task, d_peer, self.comm, peer_busy=peer_busy, num_concurrent_tx=total_tx_nodes)
                step_comm_energy += res["energy"]
                if "peer_energy" in res:
                    step_comp_energy += res["peer_energy"]
            else:
                # Edge server offload: shared multi-access uplink & edge CPU division
                d_edge = np.linalg.norm(self.positions[j] - edge_pos)
                res = self.comp.compute_edge(task, d_edge, self.comm, num_edge_tasks=num_edge_tasks, num_concurrent_tx=total_tx_nodes)
                step_comm_energy += res["energy"]

            total_delay += res["total_delay"]
            if res["total_delay"] <= task.deadline:
                tasks_completed += 1
            else:
                deadline_misses += 1

        total_step_energy = step_propulsion_energy + step_comm_energy + step_comp_energy

        # ---------------------------------------------------------------------
        # 5. Composite Reward Function
        # ---------------------------------------------------------------------
        # EQUATION:
        # R = R_task + R_deadline + R_energy + R_delay + R_safety + R_progress + R_station + R_loiter
        #
        # DETAILED TERM BREAKDOWN:
        # - R_task     = +2.0 * tasks_completed
        #   Encourages successfully finishing computational tasks before their deadline.
        #
        # - R_deadline = -2.5 * deadline_misses
        #   Strict negative penalty for tasks whose delay exceeded task.deadline.
        #
        # - R_energy   = -0.001 * total_step_energy
        #   Penalizes total Joules spent (propulsion + RF comm + CPU computing).
        #   Scaled so ~500 J of energy produces an aligned reward penalty of -0.5.
        #
        # - R_delay    = -1.0 * total_delay
        #   Penalizes total latency across all tasks in seconds.
        #
        # - R_safety   = -5.0 * interventions
        #   Heavy penalty whenever the analytical safety filter had to intervene.
        #   Prevents the policy from "riding the collision boundary" or depending on the filter.
        #
        # - R_progress = +1.5 * sum(d_prev - d_curr) - 1.0 * sum(d_curr / 50.0)
        #   Potential-based navigation shaping: rewards closing the distance to the assigned station.
        #
        # - R_station  = +3.0 (if d <= 5.0m) + 0.5 (if speed <= 1.0 m/s)
        #   Rewards reaching and maintaining station-holding inside the reach tolerance.
        #
        # - R_loiter   = -1.5 * (speed / v_max) (when d <= 5.0m and speed > 1.0 m/s)
        #   Anti-loitering damping term: penalizes orbiting or circling at high speed around station.
        # ---------------------------------------------------------------------
        reward_task = 2.0 * tasks_completed
        reward_deadline = -2.5 * deadline_misses
        reward_energy = -0.001 * total_step_energy
        reward_delay = -1.0 * total_delay
        reward_safety = -5.0 * interventions

        reward_progress = 0.0
        reward_station = 0.0
        reward_loiter = 0.0

        uavs_on_station = 0
        for j in range(self.num_uavs):
            curr_d = float(np.linalg.norm(self.positions[j] - self.stations[j]))
            if hasattr(self, "_prev_d_stations") and j in self._prev_d_stations:
                progress = self._prev_d_stations[j] - curr_d
                reward_progress += 1.5 * progress

            if curr_d <= p_cfg.reach_tolerance:
                uavs_on_station += 1
                reward_station += 3.0  # Station keeping reward
                # Anti-loitering: penalize high residual velocity when at station
                if speeds[j] > 1.0:
                    reward_loiter -= 1.5 * (speeds[j] / p_cfg.v_max)
                else:
                    reward_station += 0.5  # Precision station holding bonus
            else:
                reward_progress -= 1.0 * (curr_d / 50.0)

        reward = reward_task + reward_deadline + reward_energy + reward_delay + reward_safety + reward_progress + reward_station + reward_loiter

        # Cache distances for next step progress calculation
        self._prev_d_stations = {j: float(np.linalg.norm(self.positions[j] - self.stations[j])) for j in range(self.num_uavs)}

        # Generate new stochastic Poisson tasks for the next step
        self.current_tasks = {j: self.task_gen.sample_task(j, self.current_step, self.rng) for j in range(self.num_uavs)}

        done = (self.current_step >= self.episode_length)
        info = {
            "step_energy": total_step_energy,
            "propulsion_energy": step_propulsion_energy,
            "comm_energy": step_comm_energy,
            "comp_energy": step_comp_energy,
            "tasks_completed": tasks_completed,
            "deadline_misses": deadline_misses,
            "total_delay": total_delay,
            "uavs_on_station": uavs_on_station,
            "safety_interventions": interventions,
            "offload_edge": len(edge_offloaders),
            "offload_peer": len(peer_offloaders),
            "offload_local": len(local_executors),
            "reward_components": {
                "task": reward_task,
                "deadline": reward_deadline,
                "energy": reward_energy,
                "delay": reward_delay,
                "safety": reward_safety,
                "progress": reward_progress,
                "station": reward_station,
                "loiter": reward_loiter
            }
        }
        return self._get_observations(), self._get_global_state(), reward, done, info

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        # =========================================================================
        # PURPOSE:
        # Constructs the decentralized local observation vector for each UAV j.
        #
        # EXACT FEATURE INDEX BREAKDOWN (Total dimension = 20 for N=4 UAVs):
        # -------------------------------------------------------------------------
        # Slice / Index | Feature Description                  | Normalization
        # -------------------------------------------------------------------------
        # obs[0:3]      | 3D Position (x, y, z)                | / 200.0 m (Airspace bound)
        # obs[3:6]      | 3D Velocity (vx, vy, vz)             | / 15.0 m/s (v_max)
        # obs[6:9]      | Relative Vector to Station (dx,dy,dz)| / 200.0 m
        # obs[9]        | Euclidean Distance to Station (d_s)  | / 200.0 m
        # obs[10]       | Task Required CPU Cycles             | / 1.5e9 cycles (max cycles)
        # obs[11]       | Task Data Size                       | / 3.0e6 bits (max data size)
        # obs[12]       | Task Latency Deadline                | / 3.0 s (max deadline)
        # obs[13]       | Wireless Channel Rate to Edge Server | / 1.0e8 bps (100 Mbps)
        # obs[14:17]    | Relative Distances to Peer UAVs (x3) | / 200.0 m
        # obs[-3:-1]    | 2D Direction Vector to Closest Obs   | Unit vector (dx/d, dy/d)
        # obs[-1]       | Distance to Closest Obstacle Surface | / 50.0 m (clipped [-1, 1])
        # -------------------------------------------------------------------------
        # Total Dimension: 3 + 3 + 3 + 1 + 3 + 1 + (4-1) + 3 = 20 floats
        # =========================================================================
        """
        obs = {}
        edge_pos = np.array(self.cfg.airspace.edge_server_pos)

        for j in range(self.num_uavs):
            p_j = self.positions[j]
            v_j = self.velocities[j]
            vec_s = self.stations[j] - p_j
            d_s = float(np.linalg.norm(vec_s))

            task = self.current_tasks[j]
            if task is not None:
                task_feat = [task.cycles / 1.5e9, task.data_size / 3e6, task.deadline / 3.0]
            else:
                task_feat = [0.0, 0.0, 0.0]

            d_edge = float(np.linalg.norm(p_j - edge_pos))
            ch_edge = self.comm.data_rate(d_edge) / 1e8

            peer_dists = []
            for k in range(self.num_uavs):
                if k != j:
                    peer_dists.append(float(np.linalg.norm(p_j - self.positions[k])) / 200.0)

            # Closest obstacle relative vector and surface distance
            min_obs_d = float("inf")
            obs_vec = np.zeros(2, dtype=np.float32)
            for cx, cy, r, h in self.cfg.airspace.obstacles:
                d_c = float(np.linalg.norm(p_j[:2] - np.array([cx, cy])))
                d_surf = d_c - r
                if d_surf < min_obs_d:
                    min_obs_d = d_surf
                    obs_vec = (np.array([cx, cy]) - p_j[:2]) / max(1e-4, d_c)
            obs_feat = [obs_vec[0], obs_vec[1], float(np.clip(min_obs_d / 50.0, -1.0, 1.0))]

            feat = np.concatenate([
                p_j / 200.0,
                v_j / self.cfg.physics.v_max,
                vec_s / 200.0,
                [d_s / 200.0],
                task_feat,
                [ch_edge],
                peer_dists,
                obs_feat
            ]).astype(np.float32)
            obs[j] = feat
        return obs

    def _get_global_state(self) -> np.ndarray:
        """
        # =========================================================================
        # PURPOSE:
        # Constructs the centralized global state vector s for the Centralized Critic
        # in MAPPO, MADDPG, and CPPO (CTDE paradigm).
        #
        # Concatenates true global state information across all N UAVs:
        # Per UAV:
        # - Position: 3 floats (normalized / 200.0)
        # - Velocity: 3 floats (normalized / 15.0)
        # - Station Distance: 1 float (normalized / 200.0)
        # - Task Attributes: 3 floats (cycles, data_size, deadline)
        # Total per UAV = 10 floats.
        # Global State Dimension for N=4 UAVs = 4 * 10 = 40 floats.
        # =========================================================================
        """
        parts = []
        for j in range(self.num_uavs):
            p_j = self.positions[j] / 200.0
            v_j = self.velocities[j] / self.cfg.physics.v_max
            d_s = np.array([np.linalg.norm(self.positions[j] - self.stations[j]) / 200.0])
            task = self.current_tasks[j]
            if task is not None:
                t_f = np.array([task.cycles / 1.5e9, task.data_size / 3e6, task.deadline / 3.0])
            else:
                t_f = np.zeros(3)
            parts.extend([p_j, v_j, d_s, t_f])
        return np.concatenate(parts).astype(np.float32)
