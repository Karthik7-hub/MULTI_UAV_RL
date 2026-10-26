"""
Configuration parameters for Multi-UAV Edge Computing with Joint MAPPO.
Single source of truth for physical, communication, computation, and RL hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

ENV_PRESETS = {
    "benchmark": [
        (142.5, 100.0, 10.0, 80.0),  # East corridor
        (100.0, 142.5, 10.0, 85.0),  # North corridor
        (57.5, 100.0, 10.0, 80.0),   # West corridor
        (100.0, 57.5, 10.0, 75.0),   # South corridor
    ],
    "metropolis": [
        (140.0, 100.0, 10.0, 85.0),
        (100.0, 140.0, 10.0, 90.0),
        (60.0, 100.0, 10.0, 80.0),
        (100.0, 60.0, 10.0, 75.0),
        (135.0, 135.0, 11.0, 92.0),
        (65.0, 135.0, 12.0, 88.0),
        (65.0, 65.0, 11.0, 82.0),
        (135.0, 65.0, 12.0, 86.0),
        (165.0, 145.0, 10.0, 70.0),
        (35.0, 145.0, 9.0, 75.0),
        (35.0, 55.0, 10.0, 68.0),
        (165.0, 55.0, 9.0, 72.0),
        (145.0, 165.0, 8.0, 65.0),
        (55.0, 165.0, 8.0, 78.0),
        (55.0, 35.0, 9.0, 60.0),
        (145.0, 35.0, 8.0, 70.0),
        (175.0, 100.0, 8.0, 55.0),
        (100.0, 175.0, 8.0, 60.0),
        (25.0, 100.0, 8.0, 50.0),
        (100.0, 25.0, 8.0, 55.0),
    ],
    "chokepoint": [
        (142.5, 112.0, 7.0, 90.0),  # East bottleneck (North flank)
        (142.5, 88.0, 7.0, 90.0),   # East bottleneck (South flank)
        (57.5, 112.0, 7.0, 90.0),   # West bottleneck (North flank)
        (57.5, 88.0, 7.0, 90.0),    # West bottleneck (South flank)
        (112.0, 142.5, 7.0, 90.0),  # North bottleneck (East flank)
        (88.0, 142.5, 7.0, 90.0),   # North bottleneck (West flank)
        (112.0, 57.5, 7.0, 90.0),   # South bottleneck (East flank)
        (88.0, 57.5, 7.0, 90.0),    # South bottleneck (West flank)
    ],
    "canyon": [
        (100.0, 60.0, 24.0, 90.0),
        (100.0, 140.0, 24.0, 90.0),
        (80.0, 100.0, 12.0, 80.0),
    ],
}

@dataclass
class AirspaceConfig:
    """
    3D Airspace Boundaries and Obstacle Geometry.
    
    Coordinate System:
        Cartesian 3D Euclidean frame: Workspace W = [x_min, x_max] x [y_min, y_max] x [z_min, z_max]
        - Origin (0, 0, 0): Ground level at the south-west corner of the airspace.
        - X-axis: East-West horizontal coordinate [meters (m)].
        - Y-axis: North-South horizontal coordinate [meters (m)].
        - Z-axis: Vertical flight altitude [meters (m)].
        
    Obstacle Representation:
        Cylindrical buildings defined by 4-tuples: (cx, cy, radius, height)
        - cx, cy: Horizontal centerline coordinates in XY plane [m]
        - radius: Physical cylindrical radius [m]
        - height: Top elevation of the obstacle [m]
    """
    preset_name: str = "benchmark"
    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 200.0
    z_min: float = 0.0
    z_max: float = 100.0
    
    # Obstacle cylinders: (center_x, center_y, radius, height) in meters
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (142.5, 100.0, 10.0, 80.0),  # East corridor building
        (100.0, 142.5, 10.0, 85.0),  # North corridor building
        (57.5, 100.0, 10.0, 80.0),   # West corridor building
        (100.0, 57.5, 10.0, 75.0),   # South corridor building
    ])
    
    # Mission center & Ground Edge Server location (X, Y, Z in meters)
    mission_center: Tuple[float, float, float] = (100.0, 100.0, 50.0)
    edge_server_pos: Tuple[float, float, float] = (100.0, 100.0, 0.0)

@dataclass
class UAVPhysicsConfig:
    """
    Physical UAV Kinematics, Safety Thresholds, and Aerodynamic Power Constants.
    
    Kinematics:
        - delta_t: Discrete simulation step duration [seconds (s)].
        - v_max: Physical maximum forward flight speed [m/s].
        - mass: UAV total mass [kg].
        
    Safety and Navigation Thresholds:
        - d_safe: Minimum inter-UAV separation distance to prevent mid-air collisions [m].
        - obstacle_margin: Clearance buffer added to obstacle radius [m].
        - reach_tolerance: Distance threshold within which a UAV is considered to have arrived [m].
        - deadband_radius: Station-holding tolerance where velocity is zeroed to prevent jitter [m].
        
    Rotary-Wing Aerodynamics (Zeng et al., IEEE TWC 2019):
        P(v) = P_0 * (1 + 3*v^2 / U_tip^2) + P_i * sqrt(sqrt(1 + v^4/(4*v_0^4)) - v^2/(2*v_0^2)) + 0.5 * d_0 * rho * s * A * v^3
        - P_0: Blade profile power in hover [Watts (W)].
        - P_i: Induced power in hover [Watts (W)].
        - U_tip: Linear rotor blade tip speed [m/s].
        - v_0: Mean rotor induced velocity during hover [m/s].
        - d_0: Fuselage aerodynamic drag coefficient [dimensionless].
        - rho: Ambient air density at sea level [kg/m^3].
        - s: Rotor solidity ratio (blade area / disc area) [dimensionless].
        - A: Total rotor disc surface area [m^2].
    """
    num_uavs: int = 4
    delta_t: float = 1.0           # Time step (seconds)
    v_max: float = 15.0            # Maximum flight velocity (m/s)
    v_initial: float = 0.0         # Initial velocity (m/s)
    mass: float = 1.5              # Mass of UAV (kg)
    d_safe: float = 8.0            # Minimum inter-UAV separation (m)
    obstacle_margin: float = 2.0   # Safety margin around obstacles (m)
    reach_tolerance: float = 5.0   # Goal reach tolerance (m)
    deadband_radius: float = 0.5   # Station-holding deadband (m)
    start_radius: float = 65.0     # Initial perimeter ring radius (m)
    station_radius: float = 20.0   # Designated mission station radius (m)
    terminate_on_arrival: bool = False   # Terminate episode when all UAVs reach mission station
    terminate_on_collision: bool = False # Terminate episode on genuine obstacle/peer collision
    enable_safety_filter: bool = True    # Analytical safety filter enforcement toggle
    
    # Zeng et al. (2019) Rotary-Wing Aerodynamic Constants
    P_0: float = 79.86             # Blade profile power in hover (W)
    P_i: float = 88.63             # Induced power in hover (W)
    U_tip: float = 120.0           # Tip speed of rotor blade (m/s)
    v_0: float = 4.03              # Mean rotor induced velocity in hover (m/s)
    d_0: float = 0.6               # Fuselage drag ratio
    rho: float = 1.225             # Air density (kg/m^3)
    s: float = 0.05                # Rotor solidity
    A: float = 0.5                 # Rotor disc area (m^2)

@dataclass
class CommunicationConfig:
    """
    Wireless Radio Frequency (RF) Communication Channel Parameters.
    
    Channel Model:
        Free-space path loss channel gain: h(d) = beta_0 / d^2
        Shannon Channel Capacity with OFDMA subchannel sharing:
            R = (B / K_tx) * log2(1 + SNR)
            SNR = (P_tx * h(d)) / sigma^2
            
    Parameters:
        - bandwidth (B): Total RF channel bandwidth = 20 MHz (20e6 Hz).
        - carrier_freq: RF center frequency = 2.4 GHz (2.4e9 Hz).
        - tx_power (P_tx): UAV transmit power = 100 mW (0.1 Watts).
        - noise_power_dbm: Thermal noise power spectral density = -100 dBm (1e-13 Watts).
        - beta_0_db: Channel power gain at 1m reference distance = -50 dB (1e-5).
    """
    bandwidth: float = 20e6        # RF Bandwidth (20 MHz)
    carrier_freq: float = 2.4e9    # Carrier Frequency (2.4 GHz)
    tx_power: float = 0.1          # UAV RF Transmit power (100 mW)
    noise_power_dbm: float = -100.0# Noise power spectral density (-100 dBm)
    beta_0_db: float = -50.0       # Channel power gain at 1m reference (-50 dB)

@dataclass
class ComputationConfig:
    """
    Mobile Edge Computing (MEC) Server and On-Board CPU Parameters.
    
    Energy Model (Dynamic Voltage and Frequency Scaling - DVFS):
        E_comp = kappa * f^2 * C
        - kappa: Effective switched capacitance of the processor chip [J / (cycle * Hz^2)].
        - f: Processor clock frequency [Hz = cycles/second].
        - C: Required computation cycles [cycles].
        
    Parameters:
        - f_local: Local UAV on-board processor clock frequency = 1.5 GHz.
        - f_peer: Neighbor UAV processor clock frequency = 1.5 GHz.
        - f_edge: Ground Edge Server total processor clock frequency = 20.0 GHz.
        - kappa: Switched capacitance constant = 1e-28 J/(cycle*Hz^2).
        - edge_background_load: Number of background cellular tasks sharing Edge CPU.
    """
    f_local: float = 1.5e9         # Local UAV CPU frequency (1.5 GHz)
    f_peer: float = 1.5e9          # Peer UAV CPU frequency (1.5 GHz)
    f_edge: float = 20.0e9         # Ground Edge Server CPU frequency (20 GHz)
    kappa: float = 1e-28           # Effective switched capacitance coefficient (J/(cycle * Hz^2))
    edge_background_load: int = 2  # Background devices/tasks sharing Ground Edge Server capacity

@dataclass
class TaskConfig:
    """
    Stochastic Computational Task Generation Parameters.
    
    Generation Model:
        Task arrival follows a Poisson process with parameter lambda (tasks/second).
        When a task arrives, its attributes are uniformly sampled within ranges:
        - CPU Cycles: C ~ Uniform(cycles_min, cycles_max) [cycles]
        - Data Size: D ~ Uniform(data_size_min, data_size_max) [bits]
        - Latency Deadline: T_max ~ Uniform(deadline_min, deadline_max) [seconds]
    """
    poisson_lambda: float = 0.8    # Average task arrival rate per UAV (tasks/sec)
    cycles_min: float = 0.5e9      # Min CPU cycles (0.5 Gcycles)
    cycles_max: float = 1.5e9      # Max CPU cycles (1.5 Gcycles)
    data_size_min: float = 0.5e6   # Min data size (0.5 Mbits)
    data_size_max: float = 3.0e6   # Max data size (3.0 Mbits)
    deadline_min: float = 1.0      # Min deadline (seconds)
    deadline_max: float = 3.0      # Max deadline (seconds)

@dataclass
class MAPPOConfig:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) Hyperparameters.
    
    Algorithms:
        - gamma: Temporal reward discount factor (0.99).
        - gae_lambda: Generalized Advantage Estimation (GAE) exponential decay parameter (0.95).
        - clip_epsilon: PPO probability ratio clipping parameter (0.2).
        - entropy_coef: Entropy regularization coefficient to encourage exploration (0.01).
        - value_loss_coef: Critic value function MSE loss coefficient (0.5).
        - max_grad_norm: Maximum gradient norm for gradient clipping (0.5).
        - ppo_epochs: Number of gradient updates per rollout (5).
        - batch_size: Mini-batch size for SGD/Adam updates (64).
        - num_episodes: Total training episodes (150-250).
        - episode_length: Simulation steps per episode (100).
    """
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 5
    batch_size: int = 64
    num_episodes: int = 150
    episode_length: int = 100

@dataclass
class MasterConfig:
    """Consolidated master configuration containing all sub-system configurations."""
    airspace: AirspaceConfig = field(default_factory=AirspaceConfig)
    physics: UAVPhysicsConfig = field(default_factory=UAVPhysicsConfig)
    comm: CommunicationConfig = field(default_factory=CommunicationConfig)
    comp: ComputationConfig = field(default_factory=ComputationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)

def get_preset_config(preset: str = "benchmark", num_uavs: int = 4) -> MasterConfig:
    """
    Instantiates MasterConfig configured with a designated environment preset and fleet size.
    
    Args:
        preset: Name of preset scenario ('benchmark', 'metropolis', 'chokepoint', 'canyon').
        num_uavs: Number of autonomous UAVs in the fleet (default: 4).
        
    Returns:
        MasterConfig instance initialized with preset obstacles and fleet parameters.
    """
    cfg = MasterConfig()
    cfg.physics.num_uavs = num_uavs
    cfg.airspace.preset_name = preset
    if preset in ENV_PRESETS:
        cfg.airspace.obstacles = list(ENV_PRESETS[preset])
    if preset == "canyon":
        # The canyon's two central 24 m-radius cylinders overlap the default
        # 20 m north/south station ring.  Keep the obstacle geometry intact
        # and place both the initial ring and mission stations in valid airspace.
        cfg.physics.station_radius = 70.0
        cfg.physics.start_radius = 90.0
    return cfg


