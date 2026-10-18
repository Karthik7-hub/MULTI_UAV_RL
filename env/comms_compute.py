"""
Wireless RF Communication and MEC Computation Dynamics.
Implements:
- Shannon Line-of-Sight (LoS) channel capacity over 3D path loss
- Dynamic execution queues for Local, Peer UAV, and Ground Edge computation
- CPU dynamic frequency scaling energy: E = kappa * f^2 * C
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

@dataclass
class Task:
    uav_id: int
    cycles: float       # Required CPU computation cycles
    data_size: float    # Raw input data size in bits
    deadline: float     # Latency deadline in seconds
    arrival_time: float # Episode timestep of arrival

class WirelessChannelModel:
    """
    # =========================================================================
    # PURPOSE:
    # Models the wireless air-to-ground (UAV-to-Edge) and air-to-air (UAV-to-UAV)
    # radio frequency (RF) communication channel using free-space path loss and
    # Shannon's channel capacity theorem.
    # =========================================================================
    """
    def __init__(self, bandwidth: float = 20e6, tx_power: float = 0.1, noise_power_dbm: float = -100.0, beta_0_db: float = -50.0):
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        # Convert noise power from dBm to linear Watts: P(W) = 10^((P(dBm) - 30) / 10)
        self.noise_power = 10.0 ** ((noise_power_dbm - 30.0) / 10.0)
        # Convert channel reference power gain from dB to linear ratio: beta_0 = 10^(beta_0_db / 10)
        self.beta_0 = 10.0 ** (beta_0_db / 10.0)

    def data_rate(self, distance: float, num_concurrent: int = 1) -> float:
        """
        # =========================================================================
        # PURPOSE:
        # Calculates achievable wireless transmission data rate (throughput) in bits/second.
        #
        # EQUATIONS:
        # 1. Free-Space Path Loss Channel Gain:
        #    h(d) = beta_0 / d^2
        #    - Represents signal attenuation over 3D physical Euclidean distance d.
        #
        # 2. Signal-to-Noise Ratio (SNR):
        #    SNR = (P_tx * h(d)) / sigma^2
        #
        # 3. Shannon Capacity with OFDMA Subchannel Slicing:
        #    R = B_eff * log2(1 + SNR)
        #    Where: B_eff = Bandwidth / num_concurrent_transmitters
        #
        # VARIABLES & UNITS:
        # - distance (d)        : 3D Euclidean distance between transmitter and receiver [m]
        # - num_concurrent (K)  : Number of UAVs simultaneously transmitting over the band (OFDMA)
        # - beta_0              : Channel power gain at reference distance d_0 = 1 m (-50 dB = 1e-5) [dimensionless]
        # - P_tx                : UAV RF transmit power = 0.1 W (100 mW) [Watts]
        # - sigma^2 (noise)     : Gaussian thermal noise power = 10^-13 W (-100 dBm) [Watts]
        # - Bandwidth (B)       : Total allocated RF spectrum = 20 MHz (20e6 Hz) [Hertz]
        #
        # RESULT:
        # - R : Achievable transmission rate [bits per second (bps)]
        #
        # WHY THIS MATTERS:
        # Flying closer to the Edge Server increases h(d), boosts SNR, and raises the data
        # rate, drastically cutting transmission delay!
        # =========================================================================
        """
        d = max(1.0, float(distance))
        channel_gain = self.beta_0 / (d ** 2)
        snr = (self.tx_power * channel_gain) / self.noise_power
        eff_bandwidth = self.bandwidth / max(1, num_concurrent)
        return float(eff_bandwidth * np.log2(1.0 + snr))

    def transmit_energy(self, data_size: float, distance: float, num_concurrent: int = 1) -> float:
        """
        # =========================================================================
        # PURPOSE:
        # Calculates RF electrical transmission energy consumed by the sending UAV's antenna.
        #
        # EQUATION:
        # E_tx = P_tx * t_tx = P_tx * (D / R)
        #
        # VARIABLES & UNITS:
        # - data_size (D) : Input task payload size [bits] (NOT bytes!)
        # - distance (d)  : Distance to destination [m]
        # - R             : Shannon transmission data rate [bits / second]
        # - P_tx          : RF transmission power [Watts = Joules / second]
        #
        # RESULT:
        # - E_tx : Radio frequency transmission energy [Joules (J)]
        # =========================================================================
        """
        r = self.data_rate(distance, num_concurrent=num_concurrent)
        t_tx = data_size / max(1e3, r)
        return float(self.tx_power * t_tx)

class ComputationManager:
    """
    # =========================================================================
    # PURPOSE:
    # Manages computational latency and energy across all three execution destinations:
    # 1. Local UAV CPU
    # 2. Peer UAV CPU (Mobile-to-Mobile Cooperative Offloading)
    # 3. Ground Edge Server (Base Station MEC Server)
    # =========================================================================
    """
    def __init__(self, f_local: float = 1.5e9, f_peer: float = 1.5e9, f_edge: float = 20.0e9, kappa: float = 1e-28):
        # CPU Frequencies in Hertz (Hz = cycles/second)
        self.f_local = f_local  # 1.5 GHz = 1.5e9 Hz
        self.f_peer = f_peer    # 1.5 GHz = 1.5e9 Hz
        self.f_edge = f_edge    # 20.0 GHz = 20.0e9 Hz
        # Kappa is the effective hardware capacitance coefficient of the processor chip
        self.kappa = kappa      # 1e-28 [J / (cycles * Hz^2)]

    def compute_local(self, task: Task, concurrent_tasks: int = 1) -> Dict[str, float]:
        """
        # =========================================================================
        # PURPOSE:
        # Evaluates local on-board UAV CPU task execution (Zero wireless transmission).
        #
        # EQUATIONS:
        # 1. Effective CPU Frequency:
        #    f_eff = f_local / concurrent_tasks
        #
        # 2. Computation Execution Delay:
        #    t_comp = C / f_eff
        #
        # 3. Dynamic CMOS CPU Energy Consumption:
        #    E_comp = kappa * (f_eff^2) * C
        #
        # VARIABLES & UNITS:
        # - task.cycles (C) : Required CPU clock cycles to execute the task [cycles]
        # - f_local         : UAV on-board processor clock speed = 1.5e9 [Hz]
        # - kappa           : Effective chip switched capacitance = 1e-28 [F]
        #
        # RESULT:
        # - t_tx   = 0.0 s (No wireless transmission required)
        # - t_comp : Processing duration [s]
        # - energy : On-board battery energy spent processing the task [J]
        # =========================================================================
        """
        eff_freq = self.f_local / max(1, concurrent_tasks)
        t_comp = task.cycles / eff_freq
        e_comp = self.kappa * (eff_freq ** 2) * task.cycles
        return {"t_tx": 0.0, "t_comp": t_comp, "total_delay": t_comp, "energy": e_comp}

    def compute_peer(self, task: Task, distance: float, comm: WirelessChannelModel, peer_busy: bool = False, num_concurrent_tx: int = 1) -> Dict[str, float]:
        """
        # =========================================================================
        # PURPOSE:
        # Evaluates cooperative peer-to-peer offloading to an adjacent UAV.
        #
        # EQUATIONS:
        # 1. Wireless Air-to-Air Transmission Delay:
        #    t_tx = D / R_peer
        # 2. Sending UAV RF Energy:
        #    E_tx = P_tx * t_tx
        # 3. Peer UAV CPU Computation Delay:
        #    t_comp = C / f_peer_eff (where f_peer_eff is halved if peer is busy with own task)
        # 4. Total End-to-End Latency:
        #    T_total = t_tx + t_comp
        #
        # TRADE-OFF:
        # Useful when the offloading UAV is close to a peer and the Edge server is far,
        # but subject to air-to-air channel conditions and peer availability.
        # =========================================================================
        """
        r = comm.data_rate(distance, num_concurrent=num_concurrent_tx)
        t_tx = task.data_size / max(1e3, r)
        e_tx = comm.tx_power * t_tx
        eff_f = self.f_peer / (2.0 if peer_busy else 1.0)
        t_comp = task.cycles / eff_f
        e_comp = self.kappa * (eff_f ** 2) * task.cycles
        return {"t_tx": t_tx, "t_comp": t_comp, "total_delay": t_tx + t_comp, "energy": e_tx, "peer_energy": e_comp}

    def compute_edge(self, task: Task, distance: float, comm: WirelessChannelModel, num_edge_tasks: int = 1, num_concurrent_tx: int = 1) -> Dict[str, float]:
        """
        # =========================================================================
        # PURPOSE:
        # Evaluates offloading to the Ground Edge Computing Server.
        #
        # EQUATIONS:
        # 1. Uplink Transmission Delay:
        #    t_tx = D / R_edge
        # 2. Sending UAV RF Energy:
        #    E_tx = P_tx * t_tx
        # 3. Edge Server Sliced Computation Delay:
        #    t_comp = C / (f_edge / num_edge_tasks)
        # 4. Total End-to-End Latency:
        #    T_total = t_tx + t_comp
        #
        # KEY INSIGHT:
        # The Edge Server possesses massive compute capacity (20 GHz), so t_comp is
        # tiny (< 0.05 s). The offloading UAV pays ZERO computation energy—it only
        # pays the RF transmit energy E_tx!
        # =========================================================================
        """
        r = comm.data_rate(distance, num_concurrent=num_concurrent_tx)
        t_tx = task.data_size / max(1e3, r)
        e_tx = comm.tx_power * t_tx
        eff_f_edge = self.f_edge / max(1, num_edge_tasks)
        t_comp = task.cycles / eff_f_edge
        return {"t_tx": t_tx, "t_comp": t_comp, "total_delay": t_tx + t_comp, "energy": e_tx}

class TaskGenerator:
    def __init__(self, poisson_lambda: float = 0.8, cycles_range=(0.5e9, 1.5e9), data_range=(0.5e6, 3.0e6), deadline_range=(1.0, 3.0)):
        self.poisson_lambda = poisson_lambda
        self.cycles_range = cycles_range
        self.data_range = data_range
        self.deadline_range = deadline_range

    def sample_task(self, uav_id: int, current_step: float, rng: np.random.RandomState) -> Optional[Task]:
        """Stochastically samples an arriving MEC task according to Poisson distribution."""
        if rng.rand() < self.poisson_lambda:
            return Task(
                uav_id=uav_id,
                cycles=float(rng.uniform(*self.cycles_range)),
                data_size=float(rng.uniform(*self.data_range)),
                deadline=float(rng.uniform(*self.deadline_range)),
                arrival_time=current_step
            )
        return None
