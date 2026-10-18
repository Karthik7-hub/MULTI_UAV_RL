"""
Configuration parameters for Multi-UAV Edge Computing with Joint MAPPO.
Single source of truth for physical, communication, computation, and RL hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class AirspaceConfig:
    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 200.0
    z_min: float = 20.0
    z_max: float = 100.0
    
    # 4 Cylindrical obstacles situated directly IN BETWEEN initial UAV positions and mission stations
    # (center_x, center_y, radius, height)
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (142.5, 100.0, 10.0, 80.0),  # East corridor: directly in between (165, 100) and (120, 100)
        (100.0, 142.5, 10.0, 85.0),  # North corridor: directly in between (100, 165) and (100, 120)
        (57.5, 100.0, 10.0, 80.0),   # West corridor: directly in between (35, 100) and (80, 100)
        (100.0, 57.5, 10.0, 75.0),   # South corridor: directly in between (100, 35) and (100, 80)
    ])
    
    # Mission center & Ground Edge Server location
    mission_center: Tuple[float, float, float] = (100.0, 100.0, 50.0)
    edge_server_pos: Tuple[float, float, float] = (100.0, 100.0, 0.0)

@dataclass
class UAVPhysicsConfig:
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
    bandwidth: float = 20e6        # RF Bandwidth (20 MHz)
    carrier_freq: float = 2.4e9    # Carrier Frequency (2.4 GHz)
    tx_power: float = 0.1          # UAV RF Transmit power (100 mW)
    noise_power_dbm: float = -100.0# Noise power spectral density (-100 dBm)
    beta_0_db: float = -50.0       # Channel power gain at 1m reference (-50 dB)

@dataclass
class ComputationConfig:
    f_local: float = 1.5e9         # Local UAV CPU frequency (1.5 GHz)
    f_peer: float = 1.5e9          # Peer UAV CPU frequency (1.5 GHz)
    f_edge: float = 20.0e9         # Ground Edge Server CPU frequency (20 GHz)
    kappa: float = 1e-28           # Effective switched capacitance coefficient (J/(cycle * Hz^2))

@dataclass
class TaskConfig:
    poisson_lambda: float = 0.8    # Average task arrival rate per UAV (tasks/sec)
    cycles_min: float = 0.5e9      # Min CPU cycles (0.5 Gcycles)
    cycles_max: float = 1.5e9      # Max CPU cycles (1.5 Gcycles)
    data_size_min: float = 0.5e6   # Min data size (0.5 Mbits)
    data_size_max: float = 3.0e6   # Max data size (3.0 Mbits)
    deadline_min: float = 1.0      # Min deadline (seconds)
    deadline_max: float = 3.0      # Max deadline (seconds)

@dataclass
class MAPPOConfig:
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
    airspace: AirspaceConfig = field(default_factory=AirspaceConfig)
    physics: UAVPhysicsConfig = field(default_factory=UAVPhysicsConfig)
    comm: CommunicationConfig = field(default_factory=CommunicationConfig)
    comp: ComputationConfig = field(default_factory=ComputationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
