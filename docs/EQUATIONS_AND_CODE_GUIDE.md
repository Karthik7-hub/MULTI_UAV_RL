# Complete Mathematical & Algorithmic Code Guide

**Project Title:** Deep Reinforcement Learning for Joint Task Offloading and Energy-Efficient Path Planning in Multi-UAV Edge Computing  
**Codebase:** `research_v2`  
**Target Audience:** Engineering student / Researcher defending this project in academic review.

---

## Table of Contents
1. [Coordinate System and UAV Motion](#1-coordinate-system-and-uav-motion)
2. [Three-Dimensional Distance and Obstacle Geometry](#2-three-dimensional-distance-and-obstacle-geometry)
3. [UAV Propulsion Power and Energy](#3-uav-propulsion-power-and-energy)
4. [Wireless Communication and Transmission Delay](#4-wireless-communication-and-transmission-delay)
5. [Task Computation and Offloading Dynamics](#5-task-computation-and-offloading-dynamics)
6. [Reward Function Architecture](#6-reward-function-architecture)
7. [Actor and Critic Neural Networks](#7-actor-and-critic-neural-networks)
8. [PPO, MAPPO, IPPO, and GAE](#8-ppo-mappo-ippo-and-gae)
9. [MADDPG and Replay-Based Learning](#9-maddpg-and-replay-based-learning)
10. [Classical Path Planning (A*, RRT*, SAPF)](#10-classical-path-planning-a-rrt-sapf)
11. [Analytical Safety Filtering](#11-analytical-safety-filtering)
12. [Evaluation Metrics and Statistical Comparisons](#12-evaluation-metrics-and-statistical-comparisons)

---

## 1. Coordinate System and UAV Motion

### 1.1 Airspace Coordinate Frame
The airspace is defined as a bounded 3D Euclidean workspace:
$$\mathcal{W} = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}] = [0, 200] \times [0, 200] \times [20, 100] \text{ meters}$$
- **Origin $(0,0,0)$:** Ground level at the south-west corner of the airspace.
- **Mission Center:** $(100.0, 100.0, 50.0)\text{ m}$.
- **Ground Edge Server:** $(100.0, 100.0, 0.0)\text{ m}$ (on the ground directly below the mission center).

### 1.2 Discrete-Time Kinematic Position Integration
- **Implementation:** `env/multi_uav_env.py` $\to$ `step()`

$$\mathbf{p}_{j, t+1}^{\text{proposed}} = \mathbf{p}_{j, t} + \Delta \mathbf{p}_{j, t}$$
$$\mathbf{v}_{j, t} = \frac{\mathbf{p}_{j, t+1}^{\text{safe}} - \mathbf{p}_{j, t}}{\Delta t}$$

- **Plain English:** At each timestep, the agent proposes a continuous 3D displacement vector $\Delta \mathbf{p}_{j, t}$. The displacement is physically constrained by the UAV's maximum airspeed $v_{\max} = 15\text{ m/s}$. The actual executed velocity $\mathbf{v}_{j, t}$ is the net displacement divided by the timestep $\Delta t = 1.0\text{ s}$.
- **Variables & Units:**
  - $\mathbf{p}_{j, t} = [x, y, z]^T$: Current 3D position of UAV $j$ $[\text{meters (m)}]$.
  - $\Delta \mathbf{p}_{j, t} = [\Delta x, \Delta y, \Delta z]^T$: Displacement command $[\text{meters (m)}]$.
  - $v_{\max} = 15.0\text{ m/s}$: Maximum physical airspeed $[\text{m/s}]$.
  - $\Delta t = 1.0\text{ s}$: Discrete simulation step interval $[\text{seconds (s)}]$.
  - $\mathbf{v}_{j, t}$: Resulting 3D flight velocity vector $[\text{m/s}]$.
- **Numerical Example:** If UAV 0 is at $(165, 100, 50)\text{ m}$ and outputs action $\Delta \mathbf{p} = [-10, 0, 0]\text{ m}$, the next position is $(155, 100, 50)\text{ m}$, giving a velocity of $10\text{ m/s}$ westwards.
- **Assumptions:** Zero wind gust disturbance; ideal low-level autopilot tracking.

---

## 2. Three-Dimensional Distance and Obstacle Geometry

### 2.1 3D Euclidean Distance
- **Implementation:** `env/multi_uav_env.py` $\to$ `_get_observations()`, `step()`

$$d(\mathbf{p}_a, \mathbf{p}_b) = \|\mathbf{p}_a - \mathbf{p}_b\|_2 = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2 + (z_a - z_b)^2}$$

- **Plain English:** Measures the straight-line physical separation between two objects (e.g., UAV-to-station, UAV-to-peer, or UAV-to-edge).
- **Units:** Meters $[\text{m}]$.

### 2.2 Cylindrical Obstacle Model
- **Implementation:** `config.py` $\to$ `AirspaceConfig.obstacles`, `env/safety_filter.py`
Obstacles are modelled as vertical cylinders extending from the ground up to height $h_k$:
$$\mathcal{C}_k = \{(x, y, z) \in \mathbb{R}^3 \mid (x - c_{x,k})^2 + (y - c_{y,k})^2 \le r_k^2 \quad \text{and} \quad 0 \le z \le h_k\}$$

The horizontal Euclidean distance from a UAV at $(x, y, z)$ to the cylinder centerline is:
$$d_{xy, k} = \sqrt{(x - c_{x,k})^2 + (y - c_{y,k})^2}$$

The physical clearance to the cylinder surface is:
$$d_{\text{clearance}, k} = d_{xy, k} - r_k$$

- **Variables & Units:**
  - $(c_{x,k}, c_{y,k})$: Horizontal coordinates of the cylinder axis $[\text{m}]$.
  - $r_k = 10.0\text{ m}$: Physical radius of obstacle cylinder $[\text{m}]$.
  - $h_k \in [75, 85]\text{ m}$: Top height of the cylinder $[\text{m}]$.
  - $d_{\text{safe}} = 8.0\text{ m}$: Minimum separation safety buffer $[\text{m}]$.
  - $d_{\text{margin}} = 2.0\text{ m}$: Additional obstacle margin $[\text{m}]$.
  - Effective safety cylinder radius: $r_{\text{eff}} = r_k + d_{\text{margin}} = 12.0\text{ m}$.

---

## 3. UAV Propulsion Power and Energy

### 3.1 Rotary-Wing Propulsion Power Equation
- **Implementation:** `env/aerodynamics.py` $\to$ `AerodynamicsModel.power()`
- **Reference:** Y. Zeng, J. Xu, and R. Zhang, *IEEE Transactions on Wireless Communications*, 2019.

$$P(v) = \underbrace{P_0 \left(1 + \frac{3 v^2}{U_{\text{tip}}^2}\right)}_{\text{Blade Profile Power}} + \underbrace{P_i \left(\sqrt{1 + \frac{v^4}{4 v_0^4}} - \frac{v^2}{2 v_0^2}\right)^{1/2}}_{\text{Induced Power (Translational Lift)}} + \underbrace{\frac{1}{2} d_0 \rho s A v^3}_{\text{Parasite Drag Power}}$$

### 3.2 Term-by-Term Physical Breakdown
1. **Blade Profile Power $P_{\text{profile}}(v)$:**
   - Power required to overcome aerodynamic profile drag of the spinning rotor blades.
   - At hover ($v=0$), $P_{\text{profile}} = P_0 = 79.86\text{ W}$. Increases quadratically with forward airspeed $v$.
2. **Induced Power $P_{\text{induced}}(v)$:**
   - Power required to generate lift and keep the UAV aloft against gravity.
   - At hover ($v=0$), $P_{\text{induced}} = P_i = 88.63\text{ W}$.
   - **Crucial Physical Effect (Translational Lift):** As forward airspeed $v$ increases from 0 to $\sim 10\text{ m/s}$, the rotor disc encounters fresh air faster, requiring *less* downward induced velocity to support the weight. Thus, induced power decreases!
3. **Parasite Drag Power $P_{\text{parasite}}(v)$:**
   - Power required to overcome the form drag of the fuselage body pushing through ambient air.
   - Increases cubically ($v^3$) with airspeed. Negligible at low speed, dominant at $v > 12\text{ m/s}$.

### 3.3 Physical Parameters & Constants
| Symbol | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| $P_0$ | 79.86 | $\text{W}$ | Blade profile power in hover |
| $P_i$ | 88.63 | $\text{W}$ | Induced power in hover |
| $U_{\text{tip}}$ | 120.0 | $\text{m/s}$ | Rotor blade tip speed ($\Omega R_{\text{rotor}}$) |
| $v_0$ | 4.03 | $\text{m/s}$ | Mean rotor induced velocity in hover |
| $d_0$ | 0.6 | dimensionless | Fuselage drag ratio |
| $\rho$ | 1.225 | $\text{kg/m}^3$ | Air density at sea level ($15^\circ\text{C}$) |
| $s$ | 0.05 | dimensionless | Rotor solidity (ratio of blade area to disc area) |
| $A$ | 0.5 | $\text{m}^2$ | Rotor disc area |
| $v$ | $[0, 15]$ | $\text{m/s}$ | Horizontal flight speed |

### 3.4 Mechanical Propulsion Energy
- **Implementation:** `env/aerodynamics.py` $\to$ `energy()`
$$E_{\text{prop}} = P(v) \cdot \Delta t \quad [\text{Joules (J)}]$$
- **Hover ($v=0\text{ m/s}$):** $P(0) = 79.86 + 88.63 = 168.49\text{ W} \implies 168.49\text{ J}$ per second.
- **Efficient Cruise ($v=10\text{ m/s}$):** $P(10) \approx 81.5 + 48.2 + 18.4 = 148.1\text{ W} \implies 148.1\text{ J}$ per second.
- **High Speed ($v=15\text{ m/s}$):** $P(15) \approx 83.6 + 32.5 + 62.0 = 178.1\text{ W} \implies 178.1\text{ J}$ per second.

---

## 4. Wireless Communication and Transmission Delay

### 4.1 Free-Space Path Loss & Channel Gain
- **Implementation:** `env/comms_compute.py` $\to$ `WirelessChannelModel.data_rate()`

$$h(d) = \frac{\beta_0}{d^2}$$

- **Variables & Units:**
  - $d$: 3D Euclidean distance between transmitter and receiver $[\text{m}]$.
  - $\beta_0 = 10^{\beta_{0,\text{dB}} / 10} = 10^{-50/10} = 10^{-5}$: Reference channel power gain at $d_0 = 1\text{ m}$ [dimensionless].

### 4.2 Signal-to-Noise Ratio (SNR)
$$\text{SNR} = \frac{P_{\text{tx}} \cdot h(d)}{\sigma^2} = \frac{P_{\text{tx}} \cdot \beta_0}{d^2 \cdot \sigma^2}$$

- **Variables & Units:**
  - $P_{\text{tx}} = 0.1\text{ W}$ ($100\text{ mW}$): UAV RF transmission power $[\text{Watts}]$.
  - $\sigma^2 = 10^{(N_0 - 30)/10} = 10^{(-100 - 30)/10} = 10^{-13}\text{ W}$: Thermal noise power $[\text{Watts}]$ over the channel.

### 4.3 Shannon Capacity with OFDMA Subchannel Slicing
$$R = B_{\text{eff}} \cdot \log_2(1 + \text{SNR}) = \left(\frac{B}{K_{\text{tx}}}\right) \cdot \log_2\left(1 + \frac{P_{\text{tx}} \beta_0}{d^2 \sigma^2}\right)$$

- **Variables & Units:**
  - $B = 20\text{ MHz} = 20 \times 10^6\text{ Hz}$: Total RF bandwidth $[\text{Hertz}]$.
  - $K_{\text{tx}}$: Number of UAVs concurrently transmitting over the uplink (OFDMA resource sharing).
  - $R$: Achievable data rate $[\text{bits per second (bps)}]$.
- **Numerical Example:** If $d = 82\text{ m}$ to the Edge server and $K_{\text{tx}} = 1$:
  $$\text{SNR} = \frac{0.1 \times 10^{-5}}{(82)^2 \times 10^{-13}} \approx 1487$$
  $$R = 20 \times 10^6 \times \log_2(1 + 1487) \approx 20 \times 10^6 \times 10.54 = 210.8\text{ Mbps}$$

### 4.4 Transmission Delay & Transmission Energy
$$t_{\text{tx}} = \frac{D}{R} \quad [\text{seconds (s)}], \qquad E_{\text{tx}} = P_{\text{tx}} \cdot t_{\text{tx}} \quad [\text{Joules (J)}]$$
- **Units Note:** $D$ is in **bits** ($0.5 \times 10^6$ to $3.0 \times 10^6\text{ bits}$), NOT bytes!
- For $D = 1.75\text{ Mbits}$ and $R = 210.8\text{ Mbps}$, $t_{\text{tx}} = 0.0083\text{ s}$ ($8.3\text{ ms}$), consuming $E_{\text{tx}} = 0.1 \times 0.0083 = 0.00083\text{ J}$.

---

## 5. Task Computation and Offloading Dynamics

### 5.1 Dynamic CMOS CPU Energy Scaling
- **Implementation:** `env/comms_compute.py` $\to$ `ComputationManager`

$$E_{\text{comp}} = \kappa \cdot f_{\text{eff}}^2 \cdot C \quad [\text{Joules (J)}]$$
$$t_{\text{comp}} = \frac{C}{f_{\text{eff}}} \quad [\text{seconds (s)}]$$

- **Variables & Units:**
  - $C$: Required CPU computation cycles $[\text{cycles}]$ ($0.5 \times 10^9$ to $1.5 \times 10^9\text{ cycles}$).
  - $f_{\text{eff}}$: Effective CPU clock frequency allocated to the task $[\text{Hz = cycles/s}]$.
  - $\kappa = 10^{-28}\text{ F}$: Effective chip switched capacitance coefficient $[\text{Farads}]$.

### 5.2 Three Execution Destinations
1. **Local UAV Execution:**
   - $t_{\text{tx}} = 0.0\text{ s}$.
   - $f_{\text{eff}} = f_{\text{local}} = 1.5\text{ GHz} = 1.5 \times 10^9\text{ Hz}$.
   - $t_{\text{local}} = \frac{C}{f_{\text{local}}} \approx 0.67\text{ s}$.
   - $E_{\text{local}} = 10^{-28} \times (1.5 \times 10^9)^2 \times 1.0 \times 10^9 = 0.225\text{ Joules}$.
2. **Peer UAV Cooperative Offloading:**
   - Air-to-air transmission: $t_{\text{tx}} = D / R_{\text{peer}}$.
   - Peer CPU: $f_{\text{peer}} = 1.5\text{ GHz}$. If peer is busy with its own task, $f_{\text{eff}} = f_{\text{peer}} / 2$.
   - $t_{\text{peer}} = t_{\text{tx}} + \frac{C}{f_{\text{eff}}}$.
   - Offloading UAV pays RF transmission energy $E_{\text{tx}}$; peer pays compute energy.
3. **Ground Edge Server Offloading:**
   - Air-to-ground transmission: $t_{\text{tx}} = D / R_{\text{edge}}$.
   - Edge CPU: $f_{\text{edge}} = 20.0\text{ GHz} = 20.0 \times 10^9\text{ Hz}$. Sliced evenly across $K_{\text{edge}}$ active offloaders: $f_{\text{eff}} = f_{\text{edge}} / K_{\text{edge}}$.
   - $t_{\text{edge}} = t_{\text{tx}} + \frac{C}{f_{\text{eff}}}$.
   - Offloading UAV pays **zero** computation energy (handled by ground power grid).

---

## 6. Reward Function Architecture

### 6.1 Mathematical Expression
- **Implementation:** `env/multi_uav_env.py` $\to$ `step()`

$$R_t = R_{\text{task}} + R_{\text{deadline}} + R_{\text{energy}} + R_{\text{delay}} + R_{\text{safety}} + R_{\text{progress}} + R_{\text{station}} + R_{\text{loiter}}$$

### 6.2 Detailed Term Breakdown
| Term | Formula | Weight | Sign | Units | Purpose / Behavioral Incentive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $R_{\text{task}}$ | $+2.0 \times N_{\text{completed}}$ | $+2.0$ | Positive | dimensionless | Rewards successfully processing tasks before their deadline. |
| $R_{\text{deadline}}$ | $-2.5 \times N_{\text{missed}}$ | $-2.5$ | Negative | dimensionless | Strong penalty for exceeding task latency deadline. |
| $R_{\text{energy}}$ | $-0.001 \times E_{\text{total}}$ | $-0.001$ | Negative | $\text{J}^{-1}$ | Penalizes energy consumption ($E_{\text{prop}} + E_{\text{tx}} + E_{\text{comp}}$). $\sim 500\text{ J} \implies -0.5$. |
| $R_{\text{delay}}$ | $-1.0 \times T_{\text{delay}}$ | $-1.0$ | Negative | $\text{s}^{-1}$ | Encourages minimizing end-to-end task completion latency. |
| $R_{\text{safety}}$ | $-5.0 \times N_{\text{interventions}}$ | $-5.0$ | Negative | dimensionless | Heavy penalty whenever the kinematic safety filter intervenes. |
| $R_{\text{progress}}$ | $+1.5 \sum (d_{prev} - d_{curr}) - 1.0 \sum \frac{d_{curr}}{50}$ | $+1.5, -1.0$ | Mixed | $\text{m}^{-1}$ | Potential-based distance shaping; guides UAV toward assigned station. |
| $R_{\text{station}}$ | $+3.0$ (if $d \le 5\text{m}$) $+ 0.5$ (if $v \le 1\text{m/s}$) | $+3.0, +0.5$ | Positive | dimensionless | Rewards reaching station and precision station-holding. |
| $R_{\text{loiter}}$ | $-1.5 \times \frac{v}{v_{\max}}$ (if $d \le 5\text{m}, v > 1$) | $-1.5$ | Negative | dimensionless | Anti-loitering damping; stops UAV from high-speed orbiting around station. |

---

## 7. Actor and Critic Neural Networks

### 7.1 Multi-Agent Actor Network (Decentralized Execution)
- **Implementation:** `models/mappo.py` $\to$ `ActorNetwork`
- **Input:** Local observation $o_j \in \mathbb{R}^{20}$.
- **Backbone:** Two fully-connected layers with LayerNorm and Tanh activation (hidden dimension $H = 128$).
- **Dual Output Heads:**
  1. **Discrete Task Offloading Head:** $\mathbf{z}_{\text{offload}} \in \mathbb{R}^{N+1}$. Modeled as a Categorical distribution:
     $$P(a_{\text{offload}} = k) = \frac{\exp(z_k)}{\sum_m \exp(z_m)}$$
  2. **Continuous 3D Movement Head:** $\boldsymbol{\mu}_{\text{move}} \in \mathbb{R}^3$, $\boldsymbol{\sigma}_{\text{move}} \in \mathbb{R}^3$. Modeled as a diagonal Gaussian distribution:
     $$a_{\text{move}} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{move}}, \boldsymbol{\sigma}_{\text{move}}^2), \qquad \boldsymbol{\mu}_{\text{move}} = v_{\max} \cdot \tanh(\mathbf{W} \mathbf{h} + \mathbf{b}_{\text{prior}})$$
- **Total Log Probability:**
  $$\log \pi(a_{j, t} \mid o_{j, t}) = \log P(a_{\text{offload}}) + \sum_{i=1}^3 \log \mathcal{N}(a_{\text{move}, i}; \mu_i, \sigma_i^2)$$

### 7.2 Centralized Critic Network (Centralized Training)
- **Implementation:** `models/mappo.py` $\to$ `CriticNetwork`
- **Input:** Global environmental state $s_t \in \mathbb{R}^{40}$ (concatenation of all UAV states).
- **Architecture:** MLP ($40 \to 128 \to 128 \to 1$) with LayerNorm and Tanh.
- **Output:** Estimated scalar state-value $V_\phi(s_t) \in \mathbb{R}$.

---

## 8. PPO, MAPPO, IPPO, and GAE

### 8.1 Generalized Advantage Estimation (GAE)
- **Implementation:** `models/mappo.py` $\to$ `RolloutBuffer.compute_gae()`

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$A_t^{\text{GAE}} = \delta_t + (\gamma \lambda) (1 - d_t) A_{t+1}^{\text{GAE}}$$
$$R_t^{\text{target}} = A_t^{\text{GAE}} + V(s_t)$$

- **Hyperparameters:** Discount factor $\gamma = 0.99$, GAE trace factor $\lambda = 0.95$.

### 8.2 PPO Clipped Surrogate Objective
- **Implementation:** `models/mappo.py` $\to$ `MAPPOAgent.update()`

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)} = \exp\left(\log \pi_\theta(a_t \mid o_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid o_t)\right)$$

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right) \right] - c_{\text{entropy}} \mathcal{H}(\pi_\theta)$$

- **Clipping Parameter:** $\epsilon = 0.2$. Prevents updates where $r_t(\theta) \notin [0.8, 1.2]$.
- **Advantage Normalization:** $\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$.

### 8.3 Centralized Critic Loss
$$L^{\text{VF}}(\phi) = \frac{1}{|B|} \sum_{i \in B} \left(V_\phi(s_i) - R_i^{\text{target}}\right)^2$$

### 8.4 MAPPO vs. IPPO Distinction
- **MAPPO (CTDE):** Critic $V_\phi(s)$ takes the true global state $s \in \mathbb{R}^{40}$. Mitigates non-stationarity caused by concurrent multi-agent exploration.
- **IPPO (DTDE):** Decentralized Critic $V_{\phi, j}(o_j)$ takes only local observation $o_j \in \mathbb{R}^{20}$. Treats all peer UAVs as stochastic environmental noise.

---

## 9. MADDPG and Replay-Based Learning

### 9.1 Continuous Centralized Q-Critic
- **Implementation:** `models/maddpg.py` $\to$ `MADDPGCritic`
- **Conditioning:** $Q_\phi(s, \mathbf{a}_1, \dots, \mathbf{a}_N)$ takes the global state $s$ and joint action vector of all UAVs.
- **Action Vector per UAV:** Continuous displacement (3) + Gumbel-Softmax discrete offload distribution ($N+1 = 5$) $\implies 8$ dimensions per UAV $\implies 32$ total action dimensions.

### 9.2 Temporal-Difference Target & Loss
$$y_i = r_i + \gamma \left. Q_{\phi'}(s', \mathbf{a}'_1, \dots, \mathbf{a}'_N) \right|_{\mathbf{a}'_k = \mu_{\theta'_k}(o'_k)}$$
$$L(\phi) = \frac{1}{|B|} \sum_{i \in B} \left(Q_\phi(s_i, \mathbf{a}_{1,i}, \dots, \mathbf{a}_{N,i}) - y_i\right)^2$$

### 9.3 Deterministic Policy Gradient
$$\nabla_{\theta_j} J = \frac{1}{|B|} \sum_{i \in B} \nabla_{\theta_j} \mu_j(o_{j,i}) \left. \nabla_{\mathbf{a}_j} Q_\phi(s_i, \mathbf{a}_{1,i}, \dots, \mathbf{a}_{N,i}) \right|_{\mathbf{a}_j = \mu_j(o_{j,i})}$$

### 9.4 Polyak Soft Target Updates
$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta', \qquad \tau = 0.01$$

---

## 10. Classical Path Planning (A*, RRT*, SAPF)

### 10.1 3D Graph Search A*
- **Implementation:** `models/astar.py` $\to$ `AStar3DPlanner`

$$f(n) = g(n) + h(n)$$
- **$g(n)$:** True cumulative path length from start node to node $n$ $[\text{m}]$.
- **$h(n) = \|n - \text{goal}\|_2$:** Euclidean distance admissible heuristic to destination $[\text{m}]$.
- **Graph Expansion:** 26-connectivity discrete spatial neighbors $(\Delta x, \Delta y, \Delta z) \in \{-res, 0, res\}^3 \setminus \{(0,0,0)\}$ with grid resolution $res = 3.0\text{ m}$.
- **Line-of-Sight Shortcut Smoothing:** Prunes redundant staircase grid waypoints if direct segment is collision-free.

### 10.2 3D Asymptotically Optimal RRT*
- **Implementation:** `models/rrt_star.py` $\to$ `RRTStar3DPlanner`
1. **Sampling:** $x_{\text{rand}} \sim \mathcal{U}(\mathcal{W})$ with $15\%$ goal biasing.
2. **Nearest Neighbor:** $x_{\text{nearest}} = \arg\min_{v \in V} \|v - x_{\text{rand}}\|$.
3. **Steering:** $x_{\text{new}} = x_{\text{nearest}} + \min(\text{step\_size}, \|x_{\text{rand}} - x_{\text{nearest}}\|) \frac{x_{\text{rand}} - x_{\text{nearest}}}{\|x_{\text{rand}} - x_{\text{nearest}}\|}$.
4. **Optimal Parent Selection:** $x_{\text{parent}} = \arg\min_{v \in V_{\text{near}}} (\text{cost}(v) + \|v - x_{\text{new}}\|)$.
5. **Rewiring:** For each $v \in V_{\text{near}}$, if $\text{cost}(x_{\text{new}}) + \|x_{\text{new}} - v\| < \text{cost}(v)$, update parent $v.\text{parent} \leftarrow x_{\text{new}}$.

### 10.3 Sequential Adaptive Potential Field (SAPF)
- **Implementation:** `models/apf.py` $\to$ `SAPFNavigator`

$$\mathbf{F}_{\text{net}} = \mathbf{F}_{\text{att}} + \mathbf{F}_{\text{rep}} + \mathbf{F}_{\text{tangent}} + \mathbf{F}_{\text{peer}}$$
1. **Attractive Force:** $\mathbf{F}_{\text{att}} = k_{\text{att}} \cdot (\mathbf{g} - \mathbf{p})$.
2. **Obstacle Repulsion:** $\mathbf{F}_{\text{rep}} = k_{\text{rep}} \left(\frac{1}{d_{\text{surf}}} - \frac{1}{d_0}\right) \frac{1}{d_{\text{surf}}^2} \mathbf{n}_{\text{out}}$ (active when $d_{\text{surf}} < d_0 = 25\text{ m}$).
3. **Tangential Circulation Field:** $\mathbf{F}_{\text{tangent}} = k_{\text{tangent}} \left(\frac{1}{d_{\text{surf}}}\right) \mathbf{n}_{\perp}$ (breaks symmetry, preventing local minima deadlocks).
4. **Peer Repulsion:** $\mathbf{F}_{\text{peer}} = k_{\text{peer}} \left(\frac{1}{d_p} - \frac{1}{d_{\text{sep}}}\right) \frac{1}{d_p^2} \mathbf{u}_p$.

---

## 11. Analytical Safety Filtering

- **Implementation:** `env/safety_filter.py` $\to$ `SafetyFilter.filter_trajectories()`

### 11.1 Cylinder Boundary Projection
If $d_{xy} < r + d_{\text{margin}}$ and $z \le h$:
$$\mathbf{n}_{\text{out}} = \frac{\mathbf{p}_{xy} - \mathbf{c}_{xy}}{\|\mathbf{p}_{xy} - \mathbf{c}_{xy}\|_2}, \qquad \mathbf{p}_{xy}^{\text{safe}} = \mathbf{c}_{xy} + \mathbf{n}_{\text{out}} \cdot (r + d_{\text{margin}})$$

### 11.2 Inter-UAV Separation Preservation
If $\|\mathbf{p}_j - \mathbf{p}_k\| < d_{\text{safe}}$:
$$\mathbf{p}_j^{\text{safe}} = \mathbf{p}_j + \frac{1}{2} (d_{\text{safe}} - d_{jk}) \frac{\mathbf{p}_j - \mathbf{p}_k}{d_{jk}}, \qquad \mathbf{p}_k^{\text{safe}} = \mathbf{p}_k - \frac{1}{2} (d_{\text{safe}} - d_{jk}) \frac{\mathbf{p}_j - \mathbf{p}_k}{d_{jk}}$$
- **Singularity Handling:** If $d_{jk} < 10^{-4}\text{ m}$, the filter uses previous relative displacement $\mathbf{p}_{j, t-1} - \mathbf{p}_{k, t-1}$ to prevent division by zero.

---

## 12. Evaluation Metrics and Statistical Comparisons

### 12.1 Metrics Definition
1. **Mission Success Rate (%):**
   $$\text{Success Rate} = \frac{1}{N_{\text{trials}}} \sum_{i=1}^{N_{\text{trials}}} \mathbb{I}(d_{\text{final}, i} \le 5.0\text{ m}) \times 100\%$$
2. **Mean Flight Path Length (m):**
   $$L_j = \sum_{t=0}^{T-1} \|\mathbf{p}_{j, t+1} - \mathbf{p}_{j, t}\|_2$$
3. **Task Completion Rate (%):**
   $$\text{Completion Rate} = \frac{N_{\text{completed}}}{N_{\text{completed}} + N_{\text{missed}}} \times 100\%$$
4. **Total End-to-End Delay (s):**
   $$T_{\text{total}} = \sum_{k} (t_{\text{tx}, k} + t_{\text{comp}, k})$$
5. **Energy per Task (J/task):**
   $$E_{\text{per\_task}} = \frac{E_{\text{prop}} + E_{\text{tx}} + E_{\text{comp}}}{N_{\text{completed}}}$$
6. **Multi-Seed Aggregation:**
   $$\bar{X} \pm s_X = \frac{1}{K} \sum_{k=1}^K X_k \pm \sqrt{\frac{1}{K-1} \sum_{k=1}^K (X_k - \bar{X})^2}$$
