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
from env.observation_builder import (
    build_single_uav_obs,
    compute_obstacle_clearance_tangent,
    OBS_DIM,
    STATE_DIM_PER_UAV
)

class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[MasterConfig] = None):
        super().__init__()
        self.cfg = cfg or MasterConfig()
        self.num_uavs = self.cfg.physics.num_uavs
        if self.num_uavs != 4:
            raise ValueError(
                f"MultiUAVEnv canonical Two-Stream architecture strictly enforces fleet size N=4 "
                f"(3 peer distances, 5 offload actions, 40-dim critic state). Got num_uavs={self.num_uavs}."
            )
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
            kappa=self.cfg.comp.kappa,
            edge_background_load=self.cfg.comp.edge_background_load
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

        # Local Observation Space per UAV: canonical 22-dimensional observation vector
        self.obs_dim = OBS_DIM
        self.observation_spaces = {
            j: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for j in range(self.num_uavs)
        }

        # Centralized Critic Global State Space:
        # All positions (N*3) + All velocities (N*3) + All distances (N*1) + All tasks (N*3) = N * 10
        self.state_dim = self.num_uavs * STATE_DIM_PER_UAV

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

    def _is_pos_outside_obstacles(self, p: np.ndarray, margin: float = 4.0) -> bool:
        """Helper to verify coordinates are strictly outside all cylindrical and spherical obstacle volumes."""
        for obs in self.cfg.airspace.obstacles:
            if isinstance(obs, dict):
                if obs.get("type") == "sphere":
                    cx, cy, cz, r = obs["cx"], obs["cy"], obs["cz"], obs["radius"]
                    if float(np.linalg.norm(p - np.array([cx, cy, cz]))) < (r + margin):
                        return False
                else:
                    cx, cy, r, h = float(obs["cx"]), float(obs["cy"]), float(obs["radius"]), float(obs["height"])
                    z_min = float(obs.get("z_min", 0.0))
                    z_max = float(obs.get("z_max", h))
                    if float(np.linalg.norm(p[:2] - np.array([cx, cy]))) < (r + margin) and (z_min - margin) <= p[2] <= (z_max + margin):
                        return False
            elif len(obs) == 5:
                cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
                if float(np.linalg.norm(p - np.array([cx, cy, cz]))) < (r + margin):
                    return False
            else:
                cx, cy, r, h = obs[0], obs[1], obs[2], obs[3]
                if float(np.linalg.norm(p[:2] - np.array([cx, cy]))) < (r + margin) and p[2] <= (h + margin):
                    return False
        return True

    def reset(self, seed: Optional[int] = None, scenario_mode: str = "default", curriculum_ratio: float = 1.0) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(42)

        self.current_step = 0
        cx, cy, cz = self.cfg.airspace.mission_center
        nominal_start_radius = self.cfg.physics.start_radius  # Initial deployment ring (default 65m)
        station_radius = self.cfg.physics.station_radius      # Station ring (20m)
        
        # Smooth curriculum interpolation:
        curriculum_ratio = float(np.clip(curriculum_ratio, 0.0, 1.0))
        init_radius = station_radius + curriculum_ratio * (nominal_start_radius - station_radius)

        self.positions = {}
        self.velocities = {}
        
        # Determine whether to sample arbitrary long-range 3D navigation or ring deployment.
        # When curriculum is active (curriculum_ratio < 1.0) or during standard training, deploy on perimeter ring.
        use_long_range = (scenario_mode == "nav_general") or ((scenario_mode == "randomized") and (curriculum_ratio >= 1.0) and (float(self.rng.uniform(0.0, 1.0)) < 0.20))
        
        if use_long_range:
            for j in range(self.num_uavs):
                # Sample random start location across entire airspace
                for _ in range(50):
                    sp = np.array([
                        float(self.rng.uniform(25.0, 175.0)),
                        float(self.rng.uniform(25.0, 175.0)),
                        float(self.rng.uniform(15.0, 60.0))
                    ], dtype=np.float32)
                    if self._is_pos_outside_obstacles(sp):
                        self.positions[j] = sp
                        break
                else:
                    self.positions[j] = np.array([30.0 + j * 25.0, 30.0, 20.0], dtype=np.float32)

                # Sample random distant target location across airspace (distance >= 40m)
                for _ in range(50):
                    gp = np.array([
                        float(self.rng.uniform(25.0, 175.0)),
                        float(self.rng.uniform(25.0, 175.0)),
                        float(self.rng.uniform(15.0, 60.0))
                    ], dtype=np.float32)
                    if float(np.linalg.norm(gp - self.positions[j])) >= 40.0 and self._is_pos_outside_obstacles(gp):
                        self.stations[j] = gp
                        break
                else:
                    self.stations[j] = np.array([170.0 - j * 25.0, 170.0, 25.0], dtype=np.float32)
                self.velocities[j] = np.zeros(3, dtype=np.float32)
        else:
            rot = float(self.rng.uniform(0.0, 2.0 * np.pi)) if scenario_mode == "randomized" else 0.0
            for j in range(self.num_uavs):
                angle = j * (2.0 * np.pi / self.num_uavs) + rot
                if scenario_mode == "randomized":
                    angle_offset = float(self.rng.uniform(-0.15, 0.15))
                    rad_offset = float(self.rng.uniform(-10.0, 10.0))
                    z_offset = float(self.rng.uniform(-20.0, 20.0))
                    self.positions[j] = np.array([
                        cx + (init_radius + rad_offset) * np.cos(angle + angle_offset),
                        cy + (init_radius + rad_offset) * np.sin(angle + angle_offset),
                        float(np.clip(cz + z_offset, 20.0, 80.0))
                    ], dtype=np.float32)
                    self.stations[j] = np.array([
                        cx + self.cfg.physics.station_radius * np.cos(angle),
                        cy + self.cfg.physics.station_radius * np.sin(angle),
                        cz
                    ], dtype=np.float32)
                else:
                    self.positions[j] = np.array([
                        cx + init_radius * np.cos(angle),
                        cy + init_radius * np.sin(angle),
                        cz
                    ], dtype=np.float32)
                    self.stations[j] = np.array([
                        cx + self.cfg.physics.station_radius * np.cos(angle),
                        cy + self.cfg.physics.station_radius * np.sin(angle),
                        cz
                    ], dtype=np.float32)
                self.velocities[j] = np.zeros(3, dtype=np.float32)

        self.current_tasks = {j: self.task_gen.sample_task(j, self.current_step, self.rng) for j in range(self.num_uavs)}
        self._prev_d_stations = {j: float(np.linalg.norm(self.positions[j] - self.stations[j])) for j in range(self.num_uavs)}

        # Phase 2: Disentangled Mission Protocol & Energy Tracking
        self.arrival_steps = {j: None for j in range(self.num_uavs)}
        self.arrival_times = {j: None for j in range(self.num_uavs)}
        self.energy_to_goal = {j: 0.0 for j in range(self.num_uavs)}
        self.path_length_to_goal = {j: 0.0 for j in range(self.num_uavs)}
        self.service_energy = {j: 0.0 for j in range(self.num_uavs)}
        self.total_path_length = {j: 0.0 for j in range(self.num_uavs)}
        self.cumulative_prop_energy = {j: 0.0 for j in range(self.num_uavs)}
        self.episode_proposed_collisions = 0
        self.episode_interventions = 0
        self.episode_actual_collisions = 0

        # Reset persistent MEC computation queues
        self.comp.reset_queues()

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
        prev_positions = {j: np.copy(self.positions[j]) for j in range(self.num_uavs)}

        # Advance persistent computation queues
        self.comp.step_queues(p_cfg.delta_t)

        # ---------------------------------------------------------------------
        # 1. Kinematic Velocity and Displacement Integration
        # ---------------------------------------------------------------------
        proposed_pos = {}
        for j in range(self.num_uavs):
            disp = np.array(actions[j]["move"], dtype=np.float32)
            # Station-holding deadband: if UAV is within station deadband, hold stationary hover
            if float(np.linalg.norm(self.positions[j] - self.stations[j])) <= p_cfg.deadband_radius:
                disp = np.zeros(3, dtype=np.float32)
            else:
                disp_norm = np.linalg.norm(disp)
                # Enforce physical maximum flight speed: max displacement per second is v_max * delta_t
                if disp_norm > p_cfg.v_max * p_cfg.delta_t:
                    disp = disp * (p_cfg.v_max * p_cfg.delta_t / max(1e-6, disp_norm))
            proposed_pos[j] = self.positions[j] + disp

        # ---------------------------------------------------------------------
        # 2. Safety Projection Filter (Continuous Interception & Collision Tracking)
        # ---------------------------------------------------------------------
        enable_filter = getattr(p_cfg, "enable_safety_filter", True)
        safe_pos, interventions = self.safety.filter_trajectories(proposed_pos, self.positions, enable_filter=enable_filter)
        safety_stats = self.safety.last_stats
        self.episode_proposed_collisions += safety_stats["proposed_collisions"]
        self.episode_interventions += safety_stats["interventions"]
        self.episode_actual_collisions += safety_stats["actual_collisions"]

        # ---------------------------------------------------------------------
        # 3. Update Speeds, Arrival States, and Aerodynamic Propulsion Power
        # ---------------------------------------------------------------------
        step_propulsion_energy = 0.0
        speeds = {}
        for j in range(self.num_uavs):
            actual_disp = safe_pos[j] - self.positions[j]
            disp_dist = float(np.linalg.norm(actual_disp))
            self.total_path_length[j] += disp_dist
            self.velocities[j] = actual_disp / p_cfg.delta_t
            speed = float(np.linalg.norm(self.velocities[j]))
            speeds[j] = speed
            self.positions[j] = safe_pos[j]

            # Accumulate aerodynamic propulsion energy E = P(v) * delta_t in Joules
            uav_e = self.aero.energy(speed, p_cfg.delta_t)
            step_propulsion_energy += uav_e
            self.cumulative_prop_energy[j] += uav_e

            curr_d = float(np.linalg.norm(self.positions[j] - self.stations[j]))
            if self.arrival_steps[j] is None:
                self.energy_to_goal[j] += uav_e
                self.path_length_to_goal[j] += disp_dist
                if curr_d <= p_cfg.reach_tolerance:
                    self.arrival_steps[j] = self.current_step
                    self.arrival_times[j] = float(self.current_step * p_cfg.delta_t)
            else:
                self.service_energy[j] += uav_e

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
                res = self.comp.compute_peer(
                    task,
                    d_peer,
                    self.comm,
                    peer_busy=peer_busy,
                    num_concurrent_tx=total_tx_nodes,
                    peer_id=peer_idx,
                )
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
        # R = R_task + R_deadline + R_energy + R_delay + R_safety + R_progress + R_station + R_loiter + R_arrival + R_proximity
        #
        # DETAILED TERM BREAKDOWN:
        # - R_task      = +2.0 * tasks_completed
        #   Encourages successfully finishing computational tasks before their deadline.
        #
        # - R_deadline  = -2.5 * deadline_misses
        #   Strict negative penalty for tasks whose delay exceeded task.deadline.
        #
        # - R_energy    = -0.001 * total_step_energy
        #   Penalizes total Joules spent (propulsion + RF comm + CPU computing).
        #   Scaled so ~500 J of energy produces an aligned reward penalty of -0.5.
        #
        # - R_delay     = -1.0 * total_delay
        #   Penalizes total latency across all tasks in seconds.
        #
        # - R_safety    = -5.0 * interventions
        #   Heavy penalty whenever the analytical safety filter had to intervene.
        #   Prevents the policy from "riding the collision boundary" or depending on the filter.
        #
        # - R_progress  = +4.0 * sum(d_prev - d_curr) - 0.5 * sum(d_curr / 20.0)
        #   Potential-based navigation shaping: rewards closing the distance to the assigned station.
        #
        # - R_station   = +5.0 (if d <= 5.0m) + 2.0 (if speed <= 1.5 m/s)
        #   Rewards reaching and maintaining station-holding inside the reach tolerance.
        #
        # - R_loiter    = -2.0 * (speed / v_max) (when d <= 5.0m and speed > 1.5 m/s)
        #   Anti-loitering damping term: penalizes orbiting or circling at high speed around station.
        #
        # - R_arrival   = +80.0 milestone bonus on arrival + 30.0 fleet coordination bonus.
        # - R_proximity = Continuous penalty & tangential guidance within 12m of obstacle surface.
        # ---------------------------------------------------------------------
        # Multi-Objective Reward Design with Disentangled Progress & Offloading:
        # Scale task reward so offloading (+0.2/task) does not overwhelm spatial navigation progress (+4.0 * delta_d)
        reward_task = 0.2 * tasks_completed
        reward_deadline = -0.5 * deadline_misses
        reward_energy = -0.0002 * total_step_energy
        reward_delay = -0.2 * total_delay
        reward_safety = -2.0 * min(4, interventions)

        reward_progress = 0.0
        reward_station = 0.0
        reward_loiter = 0.0
        reward_arrival = 0.0
        reward_proximity = 0.0
        reward_distance = 0.0

        uavs_on_station = 0
        for j in range(self.num_uavs):
            curr_d = float(np.linalg.norm(self.positions[j] - self.stations[j]))
            if hasattr(self, "_prev_d_stations") and j in self._prev_d_stations:
                progress = self._prev_d_stations[j] - curr_d

                # Detour-Tolerant Progress Reward:
                # If Euclidean distance increases during a valid lateral obstacle detour,
                # neutralize the negative Euclidean penalty and reward tangential progress.
                disp = self.positions[j] - prev_positions[j]
                tan = compute_obstacle_clearance_tangent(prev_positions[j], self.stations[j], self.cfg.airspace.obstacles)
                tan_prog = float(np.dot(disp[:2], tan))
                tan_active = float(np.linalg.norm(tan)) > 0.5

                if progress < 0.0 and tan_prog > 0.05 and tan_active:
                    # Neutralize Euclidean penalty during valid obstacle detour and reward tangential progress
                    reward_progress += 2.0 * tan_prog
                else:
                    reward_progress += 4.0 * progress

            # Continuous distance penalty to discourage wandering or hovering far from goal
            reward_distance -= 0.02 * (curr_d / 20.0)

            # Goal Arrival Milestone Bonus: +50.0 awarded at the instant of arrival
            if self.arrival_steps[j] is not None and self.arrival_steps[j] == self.current_step:
                reward_arrival += 50.0

            if curr_d <= p_cfg.reach_tolerance:
                uavs_on_station += 1
                reward_station += 2.0  # Station keeping reward
                # Anti-loitering: penalize high residual velocity when at station
                if speeds[j] > 1.5:
                    reward_loiter -= 0.5 * (speeds[j] / p_cfg.v_max)
                else:
                    reward_station += 1.0  # Precision station holding bonus

            # Airspace boundary proximity penalty (< 15m)
            p_j = self.positions[j]
            d_wall = min(
                p_j[0] - self.cfg.airspace.x_min, self.cfg.airspace.x_max - p_j[0],
                p_j[1] - self.cfg.airspace.y_min, self.cfg.airspace.y_max - p_j[1]
            )
            if d_wall < 15.0:
                reward_safety -= 1.5 * (15.0 - max(0.0, d_wall)) / 15.0

            # Smooth obstacle proximity penalty (< 10m from obstacle surface)
            for cx, cy, r, h in self.cfg.airspace.obstacles:
                d_c = float(np.linalg.norm(self.positions[j][:2] - np.array([cx, cy])))
                d_surf = d_c - r
                if self.positions[j][2] <= h + 5.0 and d_surf < 10.0:
                    reward_proximity -= 1.5 * (10.0 - max(0.0, d_surf)) / 10.0

        # Coordinated Fleet Mission Bonus when all UAVs reach designated stations
        if all(self.arrival_steps[k] is not None for k in range(self.num_uavs)):
            if any(self.arrival_steps[k] == self.current_step for k in range(self.num_uavs)):
                reward_arrival += 30.0

        reward = (reward_task + reward_deadline + reward_energy + reward_delay +
                  reward_safety + reward_progress + reward_station + reward_loiter +
                  reward_arrival + reward_proximity + reward_distance)

        # Cache distances for next step progress calculation
        self._prev_d_stations = {j: float(np.linalg.norm(self.positions[j] - self.stations[j])) for j in range(self.num_uavs)}

        # Generate new stochastic Poisson tasks for the next step
        self.current_tasks = {j: self.task_gen.sample_task(j, self.current_step, self.rng) for j in range(self.num_uavs)}

        # Mission Termination Logic: Explicitly differentiate arrival, collision, and truncation
        all_arrived = all(self.arrival_steps[j] is not None for j in range(self.num_uavs))
        terminate_on_arrival = getattr(p_cfg, "terminate_on_arrival", False)
        terminate_on_collision = getattr(p_cfg, "terminate_on_collision", False)

        terminated = False
        if terminate_on_arrival and all_arrived:
            terminated = True
        if terminate_on_collision and safety_stats["actual_collisions"] > 0:
            terminated = True

        truncated = (self.current_step >= self.episode_length)
        done = terminated or truncated

        info = {
            "step_energy": total_step_energy,
            "propulsion_energy": step_propulsion_energy,
            "comm_energy": step_comm_energy,
            "comp_energy": step_comp_energy,
            "tasks_completed": tasks_completed,
            "deadline_misses": deadline_misses,
            "total_delay": total_delay,
            "uavs_on_station": uavs_on_station,
            "all_arrived": all_arrived,
            "arrival_rate": float(sum(1 for a in self.arrival_steps.values() if a is not None) / self.num_uavs * 100.0),
            "arrival_steps": dict(self.arrival_steps),
            "arrival_times": dict(self.arrival_times),
            "energy_to_goal": dict(self.energy_to_goal),
            "fleet_energy_to_goal": float(sum(self.energy_to_goal.values())),
            "path_length_to_goal": dict(self.path_length_to_goal),
            "fleet_path_length_to_goal": float(sum(self.path_length_to_goal.values())),
            "service_energy": dict(self.service_energy),
            "fleet_service_energy": float(sum(self.service_energy.values())),
            "safety_interventions": interventions,
            "proposed_collisions": safety_stats["proposed_collisions"],
            "actual_collisions": safety_stats["actual_collisions"],
            "goal_distances": {j: float(np.linalg.norm(self.positions[j] - self.stations[j])) for j in range(self.num_uavs)},
            "fleet_mean_goal_error": float(np.mean([np.linalg.norm(self.positions[j] - self.stations[j]) for j in range(self.num_uavs)])),
            "terminated": terminated,
            "truncated": truncated,
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
                "loiter": reward_loiter,
                "arrival": reward_arrival,
                "proximity": reward_proximity,
                "distance": reward_distance
            }
        }
        return self._get_observations(), self._get_global_state(), reward, done, info

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Constructs the decentralized local observation vector for each UAV j
        using the unified build_single_uav_obs builder.
        """
        obs = {}
        edge_pos = np.array(self.cfg.airspace.edge_server_pos)

        for j in range(self.num_uavs):
            task = self.current_tasks[j]
            task_f = (task.cycles, task.data_size, task.deadline) if task is not None else None
            peer_positions = [self.positions[k] for k in range(self.num_uavs) if k != j]

            obs[j] = build_single_uav_obs(
                pos=self.positions[j],
                vel=self.velocities[j],
                station_pos=self.stations[j],
                edge_pos=edge_pos,
                obstacles=self.cfg.airspace.obstacles,
                task_features=task_f,
                peer_positions=peer_positions,
                comm_model=self.comm,
                v_max=self.cfg.physics.v_max,
                airspace_bound=200.0,
                num_peers_expected=self.num_uavs - 1
            )
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
