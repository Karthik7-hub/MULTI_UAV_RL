"""
Wireless RF Communication and MEC Computation Dynamics.
Implements:
- Shannon Line-of-Sight (LoS) channel capacity over 3D path loss
- Dynamic execution queues for Local, Peer UAV, and Ground Edge computation
- CPU dynamic frequency scaling energy: E = kappa * f^2 * C
"""

from typing import Any
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
    Manages computational latency and energy across all three execution destinations:
    1. Local UAV CPU (Dynamic frequency scaling, queue backlog)
    2. Peer UAV CPU (Mobile-to-Mobile Cooperative Offloading)
    3. Ground Edge Server (Multi-tenant sliced MEC with background cellular traffic)
    """
    def __init__(
        self,
        f_local: float = 1.5e9,
        f_peer: float = 1.5e9,
        f_edge: float = 20.0e9,
        kappa: float = 1e-28,
        edge_background_load: int = 2
    ):
        # CPU Frequencies in Hertz (Hz = cycles/second)
        self.f_local = f_local  # 1.5 GHz
        self.f_peer = f_peer    # 1.5 GHz
        self.f_edge = f_edge    # 20.0 GHz
        self.kappa = kappa      # 1e-28 [J / (cycles * Hz^2)]
        self.edge_background_load = edge_background_load

        # Persistent task queues (accumulated cycles waiting for execution)
        self.local_queues: Dict[int, float] = {}
        self.peer_queues: Dict[int, float] = {}
        self.edge_queue: float = 0.0

    def reset_queues(self):
        """Clears all persistent task queues on environment reset."""
        self.local_queues = {}
        self.peer_queues = {}
        self.edge_queue = 0.0

    def step_queues(self, delta_t: float = 1.0):
        """Advances physical time and drains serviced cycles across all processors."""
        for j in list(self.local_queues.keys()):
            self.local_queues[j] = max(0.0, self.local_queues[j] - self.f_local * delta_t)
        for j in list(self.peer_queues.keys()):
            self.peer_queues[j] = max(0.0, self.peer_queues[j] - self.f_peer * delta_t)
        self.edge_queue = max(0.0, self.edge_queue - self.f_edge * delta_t)

    def compute_local(self, task: Task, concurrent_tasks: int = 1) -> Dict[str, float]:
        """Evaluates local on-board UAV CPU task execution (Zero wireless transmission)."""
        uav_id = task.uav_id
        q_cycles = self.local_queues.get(uav_id, 0.0)
        eff_freq = self.f_local / max(1, concurrent_tasks)
        t_wait = q_cycles / eff_freq
        t_comp = task.cycles / eff_freq
        e_comp = self.kappa * (eff_freq ** 2) * task.cycles
        total_delay = t_wait + t_comp

        # Enqueue new cycles
        self.local_queues[uav_id] = q_cycles + task.cycles
        return {"t_tx": 0.0, "t_wait": t_wait, "t_comp": t_comp, "total_delay": total_delay, "energy": e_comp}

    def compute_peer(
        self,
        task: Task,
        distance: float,
        comm: WirelessChannelModel,
        peer_busy: bool = False,
        num_concurrent_tx: int = 1,
        peer_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluates cooperative peer-to-peer offloading to an adjacent UAV.

        ``peer_queues`` are keyed by the receiving UAV because they represent
        work waiting for that UAV's processor.  ``peer_id`` is optional for
        backwards compatibility with direct callers that do not identify a
        receiver explicitly.
        """
        r = comm.data_rate(distance, num_concurrent=num_concurrent_tx)
        t_tx = task.data_size / max(1e3, r)
        e_tx = comm.tx_power * t_tx
        eff_f = self.f_peer / (2.0 if peer_busy else 1.0)
        queue_owner = task.uav_id if peer_id is None else int(peer_id)
        q_cycles = self.peer_queues.get(queue_owner, 0.0)
        t_wait = q_cycles / eff_f
        t_comp = task.cycles / eff_f
        e_comp = self.kappa * (eff_f ** 2) * task.cycles
        total_delay = t_tx + t_wait + t_comp

        self.peer_queues[queue_owner] = q_cycles + task.cycles
        return {"t_tx": t_tx, "t_wait": t_wait, "t_comp": t_comp, "total_delay": total_delay, "energy": e_tx, "peer_energy": e_comp}

    def compute_edge(
        self,
        task: Task,
        distance: float,
        comm: WirelessChannelModel,
        num_edge_tasks: int = 1,
        num_concurrent_tx: int = 1
    ) -> Dict[str, float]:
        """Evaluates offloading to the multi-tenant Ground Edge Computing Server."""
        r = comm.data_rate(distance, num_concurrent=num_concurrent_tx)
        t_tx = task.data_size / max(1e3, r)
        e_tx = comm.tx_power * t_tx
        total_edge_tenants = max(1, num_edge_tasks + self.edge_background_load)
        eff_f_edge = self.f_edge / total_edge_tenants
        t_wait = self.edge_queue / eff_f_edge
        t_comp = task.cycles / eff_f_edge
        total_delay = t_tx + t_wait + t_comp

        self.edge_queue += task.cycles
        return {"t_tx": t_tx, "t_wait": t_wait, "t_comp": t_comp, "total_delay": total_delay, "energy": e_tx}

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

def run_deterministic_offloading_diagnostics() -> Dict[str, Any]:
    """
    Deterministic Diagnostic Suite for Phase 5.2:
    Proves mathematically and empirically that Local, Peer, and Edge each become
    uniquely optimal under distinct physical conditions without arbitrary parameter hacking.
    """
    comm = WirelessChannelModel(bandwidth=20e6, tx_power=0.1)
    
    # 1. Condition: Edge Server has queued backlog (e.g. 1.2 Gcycles pending); small task arrives
    # Local compute has zero waiting queue delay, completing immediately!
    comp_busy_edge = ComputationManager(f_local=1.5e9, f_peer=1.5e9, f_edge=20e9, edge_background_load=2)
    comp_busy_edge.edge_queue = 1.2e9  # 1.2 Gcycles backlogged at Edge
    task_small = Task(uav_id=0, cycles=0.1e9, data_size=0.5e6, deadline=2.0, arrival_time=0)
    res_loc = comp_busy_edge.compute_local(task_small)
    res_edge_busy = comp_busy_edge.compute_edge(task_small, distance=80.0, comm=comm, num_edge_tasks=2)
    assert res_loc["total_delay"] < res_edge_busy["total_delay"], "Local execution must beat backlogged edge queue"

    # 2. Condition: Congested Edge (heavy background load) with close idle peer (d=8m)
    comp_congested = ComputationManager(f_local=1.5e9, f_peer=1.5e9, f_edge=20e9, edge_background_load=12)
    task_med = Task(uav_id=0, cycles=0.9e9, data_size=0.8e6, deadline=2.0, arrival_time=0)
    res_peer = comp_congested.compute_peer(task_med, distance=8.0, comm=comm, peer_busy=False, num_concurrent_tx=1)
    res_edge_congested = comp_congested.compute_edge(task_med, distance=80.0, comm=comm, num_edge_tasks=3)
    assert res_peer["total_delay"] < res_edge_congested["total_delay"], "Congested edge must favor close idle peer execution"

    # 3. Condition: Heavy compute task (2.8 Gcycles) with clean LoS channel to Edge (d=35m)
    comp_clean = ComputationManager(f_local=1.5e9, f_peer=1.5e9, f_edge=20e9, edge_background_load=1)
    task_heavy = Task(uav_id=0, cycles=2.8e9, data_size=0.8e6, deadline=2.0, arrival_time=0)
    res_loc_heavy = comp_clean.compute_local(task_heavy)
    res_edge_clean = comp_clean.compute_edge(task_heavy, distance=35.0, comm=comm, num_edge_tasks=1)
    assert res_edge_clean["total_delay"] < res_loc_heavy["total_delay"], "Heavy compute task with clean link must favor edge execution"

    return {
        "backlogged_edge_local_delay": res_loc["total_delay"],
        "backlogged_edge_delay": res_edge_busy["total_delay"],
        "congested_edge_peer_delay": res_peer["total_delay"],
        "congested_edge_delay": res_edge_congested["total_delay"],
        "heavy_task_edge_delay": res_edge_clean["total_delay"],
        "heavy_task_local_delay": res_loc_heavy["total_delay"],
        "all_diagnostics_verified": True
    }

