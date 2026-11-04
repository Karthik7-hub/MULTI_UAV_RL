# Mathematical Notation and Parameter Definitions

**Document Version:** 1.0  
**Target Repository:** `research_v2`  
**Date:** November 2026  

---

## 1. Coordinate Frames, Kinematics & Airspace Geometry

| Symbol | Definition | Units | Physical Value / Domain in Simulation | Implementation Reference |
| :---: | :--- | :---: | :---: | :--- |
| $\mathcal{W}$ | 3D Cartesian Airspace Boundary | $\text{m}^3$ | $[0, 200] \times [0, 200] \times [0, 100]$ | `config.AirspaceConfig` |
| $\mathbf{p}_{j, t}$ | Position vector of UAV $j$ at step $t$ | $\text{m}$ | $[x, y, z]^T \in \mathcal{W}$ | `MultiUAVEnv.positions` |
| $\mathbf{v}_{j, t}$ | Executed physical velocity vector | $\text{m/s}$ | $\|\mathbf{v}\| \le v_{\max} = 15.0\,\text{m/s}$ | `MultiUAVEnv.velocities` |
| $\mathbf{g}_j$ | Designated 3D mission station of UAV $j$ | $\text{m}$ | Station radius $20.0\,\text{m}$ ring | `MultiUAVEnv.stations` |
| $\Delta t$ | Discrete simulation step duration | $\text{s}$ | $1.0\,\text{s}$ | `config.UAVPhysicsConfig.delta_t` |
| $N$ | Total number of autonomous UAVs | integer | $4$ | `config.UAVPhysicsConfig.num_uavs` |
| $d_{\text{safe}}$ | Inter-UAV collision threshold | $\text{m}$ | $8.0\,\text{m}$ | `config.UAVPhysicsConfig.d_safe` |
| $d_{\text{reach}}$ | Goal station reach tolerance | $\text{m}$ | $5.0\,\text{m}$ | `config.UAVPhysicsConfig.reach_tolerance` |
| $\mathcal{C}_k$ | 3D cylindrical obstacle cylinder $k$ | set | Center $(c_{x,k}, c_{y,k})$, radius $r_k$, height $h_k$ | `config.AirspaceConfig.obstacles` |
| $d_{\text{margin}}$ | Clearance buffer around obstacles | $\text{m}$ | $2.0\,\text{m}$ | `config.UAVPhysicsConfig.obstacle_margin` |
| $r_{\text{eff}}$ | Effective collision radius ($r_k + d_{\text{margin}}$)| $\text{m}$ | $12.0\,\text{m}$ ($r_k = 10.0\,\text{m}$) | `env/safety_filter.py` |

---

## 2. Rotary-Wing Aerodynamic Propulsion Model (Zeng et al. 2019)

| Symbol | Definition | Units | Implemented Value | Physical Meaning |
| :---: | :--- | :---: | :---: | :--- |
| $P(v)$ | Total rotary-wing mechanical power | $\text{W}$ | Eq. (12) in Zeng et al. (2019) | Total instantaneous flight power |
| $P_0$ | Blade profile power in pure hover | $\text{W}$ | $79.86\,\text{W}$ | Power to overcome profile blade drag |
| $P_i$ | Induced power in pure hover | $\text{W}$ | $88.63\,\text{W}$ | Power required to produce aerodynamic lift |
| $U_{\text{tip}}$ | Rotor blade tip linear velocity | $\text{m/s}$ | $120.0\,\text{m/s}$ | Speed of outer blade tip |
| $v_0$ | Mean rotor induced velocity in hover | $\text{m/s}$ | $4.03\,\text{m/s}$ | Mean downward slipstream velocity |
| $d_0$ | Fuselage parasitic drag ratio | dimensionless | $0.6$ | Aerodynamic shape form factor |
| $\rho$ | Atmospheric air density at sea level | $\text{kg/m}^3$ | $1.225\,\text{kg/m}^3$ | Ambient air density |
| $s$ | Rotor blade solidity ratio | dimensionless | $0.05$ | Total blade area / swept disc area |
| $A$ | Total rotor disc swept area | $\text{m}^2$ | $0.5\,\text{m}^2$ | $\pi \times R_{\text{rotor}}^2$ |
| $E_{\text{prop}}$ | Propulsion energy expended in step | $\text{J}$ | $P(v) \cdot \Delta t$ | Mechanical battery drain |

---

## 3. Wireless Communication & OFDMA Channel Model

| Symbol | Definition | Units | Implemented Value | Reference / Notes |
| :---: | :--- | :---: | :---: | :--- |
| $B$ | Total allocated RF spectrum | $\text{Hz}$ | $20\,\text{MHz} = 20 \times 10^6\,\text{Hz}$ | `config.CommunicationConfig.bandwidth` |
| $P_{\text{tx}}$ | UAV RF transmission power | $\text{W}$ | $0.1\,\text{W} = 100\,\text{mW}$ | `config.CommunicationConfig.tx_power` |
| $\beta_0$ | Channel gain at reference $d_0=1\,\text{m}$ | dimensionless | $-50\,\text{dB} = 10^{-5}$ | `config.CommunicationConfig.beta_0_db` |
| $\sigma^2$ | Gaussian thermal noise power | $\text{W}$ | $-100\,\text{dBm} = 10^{-13}\,\text{W}$ | `config.CommunicationConfig.noise_power_dbm` |
| $h(d)$ | Free-space path loss channel gain | dimensionless | $\beta_0 / d^2$ | Line-of-Sight (LoS) model |
| $\text{SNR}$ | Signal-to-Noise Ratio | dimensionless | $(P_{\text{tx}} \cdot h(d)) / \sigma^2$ | Wireless SNR |
| $K_{\text{tx}}$ | Concurrent OFDMA transmitting nodes | integer | $1 \le K_{\text{tx}} \le N$ | Subchannel bandwidth slicing: $B / K_{\text{tx}}$ |
| $R$ | Shannon transmission throughput | $\text{bps}$ | $(B / K_{\text{tx}}) \log_2(1 + \text{SNR})$ | Available transmission bit rate |
| $T_{\text{tx}}$ | Uplink transmission delay | $\text{s}$ | $D / R$ | Data transmission latency |
| $E_{\text{tx}}$ | RF communication energy | $\text{J}$ | $P_{\text{tx}} \cdot T_{\text{tx}}$ | Antenna power consumption |

---

## 4. Mobile Edge Computing & Task Queues

| Symbol | Definition | Units | Implemented Value | Physical Meaning |
| :---: | :--- | :---: | :---: | :--- |
| $C$ | Task computational cycles | $\text{cycles}$ | $[0.5, 1.5] \times 10^9\,\text{cycles}$ | Required CPU cycles per task |
| $D$ | Task input payload data size | $\text{bits}$ | $[0.5, 3.0] \times 10^6\,\text{bits}$ | Payload bits to transmit |
| $T_{\text{max}}$ | Latency deadline | $\text{s}$ | $[1.0, 3.0]\,\text{s}$ (Stress: $[0.4, 1.0]\,\text{s}$) | Maximum permissible execution delay |
| $\lambda$ | Poisson arrival rate | $\text{tasks/s}$ | $0.8$ (Stress: $1.6\text{--}2.5$) | Average arrival frequency |
| $f_{\text{local}}$ | Onboard UAV CPU frequency | $\text{Hz}$ | $1.5\,\text{GHz}$ (Stress: $0.8\,\text{GHz}$) | Onboard clock rate |
| $f_{\text{peer}}$ | Neighbor UAV CPU frequency | $\text{Hz}$ | $1.5\,\text{GHz}$ (Stress: $0.8\,\text{GHz}$) | Cooperative peer clock rate |
| $f_{\text{edge}}$ | Ground Edge Server CPU frequency | $\text{Hz}$ | $20.0\,\text{GHz}$ | Base station multi-core capacity |
| $\kappa$ | Switched capacitance constant | $\text{J/(cycle}\cdot\text{Hz}^2)$| $10^{-28}$ | DVFS dynamic energy coefficient |
| $E_{\text{comp}}$ | Processor dynamic computing energy | $\text{J}$ | $\kappa \cdot f^2 \cdot C$ | Processor dynamic energy consumption |
| $Q_{\text{local}}, Q_{\text{edge}}$ | Accumulated queue backlog | $\text{cycles}$ | Drained at $f \cdot \Delta t$ per step | Persistent execution queues |

---

## 5. Multi-Agent Reinforcement Learning (MAPPO / CTDE)

| Symbol | Definition | Units / Dimensionality | Description |
| :---: | :--- | :---: | :--- |
| $o_j \in \mathbb{R}^{22}$ | Decentralized local observation vector | 22-dimensional continuous | UAV $j$ local features (pos, vel, goal, obs, peers, task) |
| $s \in \mathbb{R}^{52}$ | Centralized global state vector | 52-dimensional continuous | Fleet state: $13 \times 4$ concatenated UAV vectors |
| $\pi_\theta(a_j \mid o_j)$ | Decentralized actor policy | Dual-head probability | Continuous velocity head + Discrete offload head |
| $V_\phi(s)$ | Centralized critic value function | Scalar $\mathbb{R}$ | Expected discounted team return from state $s$ |
| $\gamma$ | Reward discount factor | $0.99$ | Discounting parameter |
| $\lambda_{\text{gae}}$ | Generalized Advantage Estimation factor | $0.95$ | GAE bias-variance trade-off parameter |
| $\epsilon_{\text{clip}}$ | PPO probability ratio clip parameter | $0.20$ | Surrogate policy clipping threshold |
| $c_2$ | Entropy regularization coefficient | $0.01$ | Policy exploration regularization coefficient |

