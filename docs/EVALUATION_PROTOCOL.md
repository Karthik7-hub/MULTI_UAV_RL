# Standardized Scientific Evaluation Protocol for Multi-UAV 3D Path-Planning & MEC Offloading

**Document Version:** 2.0 (Phase 2 Post-Forensic Audit Standard)  
**Target Repository:** `research_v2` (Multi-UAV 3D RL Navigation & Offloading)  
**Active Git Branch:** `audit/phase-2-reverification`  
**Date:** October 2026  

---

## 1. Executive Summary & Objective

This document establishes the standardized, scientifically reproducible evaluation protocol for benchmarking Multi-Agent Deep Reinforcement Learning (MADRL) algorithms and classical path-planning baselines in 3D multi-UAV environments with Mobile Edge Computing (MEC) task offloading.

The protocol resolves historical evaluation artifacts identified during the Phase 1 forensic audit:
1. **Disentanglement of Navigation Energy from Service Energy:** Eliminates metric distortion caused by post-arrival hovering.
2. **Disentanglement of Navigation Path Length from Wandering/Loitering:** Separates true physical flight distance to goal from post-goal loitering.
3. **Rigorous Arrival Criteria:** Clear mathematical definitions for Per-UAV Success Rate, Fleet Mission Success Rate, and Goal Error.
4. **Observation Distribution Parity:** Guarantees that observation space construction across evaluation and deployment matches training normalization identically.
5. **Multi-Seed Statistical Aggregation:** Standardized evaluation across multiple deterministic random seeds ($N=5$) with sample mean $\pm$ sample standard deviation reporting.

---

## 2. Airspace & Simulation Environment Specification

### 2.1 Environmental Boundaries & Coordinate System
- **Coordinate Frame:** Right-handed Cartesian coordinates $(x, y, z)$ in meters ($\text{m}$), where $z \ge 0$ represents altitude Above Ground Level (AGL).
- **Airspace Bounds:**
  - $x \in [0, 400]\,\text{m}$
  - $y \in [0, 400]\,\text{m}$
  - $z \in [20, 120]\,\text{m}$
- **Simulation Time Step:** $\Delta t = 1.0\,\text{s}$
- **Episode Horizon:** $T = 100$ steps ($100.0\,\text{s}$ physical flight time).

### 2.2 UAV Fleet & Kinematics
- **Fleet Size:** $N = 4$ quadrotor UAVs.
- **State Vector per UAV $j$:** $\mathbf{p}_j(t) = [x_j(t), y_j(t), z_j(t)]^T \in \mathbb{R}^3$, velocity $\mathbf{v}_j(t) = [\dot{x}_j(t), \dot{y}_j(t), \dot{z}_j(t)]^T \in \mathbb{R}^3$.
- **Speed Bounds:**
  - Horizontal speed limit: $v_{xy,\max} = 20.0\,\text{m/s}$
  - Vertical climb/descent speed limit: $v_{z,\max} = 10.0\,\text{m/s}$
  - Overall kinematic cap: $\|\mathbf{v}_j(t)\|_2 \le 25.0\,\text{m/s}$
- **Acceleration Limits:**
  - Maximum linear acceleration: $a_{\max} = 5.0\,\text{m/s}^2$
- **Discrete Velocity Action Discretization:**
  - Acceleration actions $\mathbf{a}_j(t) \in \{-a_{\max}, 0, +a_{\max}\}^3$ mapped to target velocity updates:
    $$\mathbf{v}_j(t+1) = \text{clip}\left(\mathbf{v}_j(t) + \mathbf{a}_j(t)\Delta t, -v_{\max}, v_{\max}\right)$$
    $$\mathbf{p}_j(t+1) = \mathbf{p}_j(t) + \mathbf{v}_j(t+1)\Delta t$$

### 2.3 Cylindrical Obstacle Topography
Obstacles are modelled as vertical cylinders extending from ground $z=0$ up to height $H_k$:
$$\mathcal{O}_k = \left\{(x, y, z) \in \mathbb{R}^3 \mid (x - c_{x,k})^2 + (y - c_{y,k})^2 \le (R_k + r_{\text{safe}})^2 \land 0 \le z \le H_k\right\}$$
- **Physical Obstacle Radius:** $R_k$ (configured per preset, e.g. $18.0\text{--}30.0\,\text{m}$).
- **Safety Margin:** $r_{\text{safe}} = 2.0\,\text{m}$ envelope buffer.
- **UAV Collision Radius:** $r_{\text{uav}} = 1.5\,\text{m}$.
- **Inter-UAV Collision Distance:** $d_{\text{uav-col}} = 5.0\,\text{m}$.

### 2.4 Mission Target & Arrival Condition
- **Target Coordinates:** Ground Station / Mission Goal $\mathbf{g}_j = [x_{g,j}, y_{g,j}, z_{g,j}]^T$.
- **Reach Tolerance Threshold:** $\epsilon_{\text{reach}} = 10.0\,\text{m}$.
- **Arrival Condition:** UAV $j$ is deemed to have reached its destination at timestep $t$ if:
  $$\|\mathbf{p}_j(t) - \mathbf{g}_j\|_2 \le \epsilon_{\text{reach}}$$
- **Arrival Timestep ($t_{\text{arr}, j}$):**
  $$t_{\text{arr}, j} = \min \left\{ t \in \{1, \dots, T\} \mid \|\mathbf{p}_j(t) - \mathbf{g}_j\|_2 \le \epsilon_{\text{reach}} \right\}$$
  If UAV $j$ never satisfies the condition within $T$ steps, $t_{\text{arr}, j} = \infty$ (or penalty horizon $T = 100.0\,\text{s}$).

---

## 3. Disentangled Metric Definitions, Formulas & Units

### 3.1 Propulsion & Kinetic Energy Model
Power consumption for quadrotor UAV $j$ at velocity $\mathbf{v}_j(t)$ follows the verified physical helicopter aerodynamic model:
$$P_j(t) = P_0 \left( 1 + \frac{3 \|\mathbf{v}_j(t)\|^2}{U_{\text{tip}}^2} \right) + P_i \left( \sqrt{1 + \frac{\|\mathbf{v}_j(t)\|^4}{4 v_0^4}} - \frac{\|\mathbf{v}_j(t)\|^2}{2 v_0^2} \right)^{1/2} + \frac{1}{2} d_0 \rho s A \|\mathbf{v}_j(t)\|^3$$
where:
- $P_0 = 79.86\,\text{W}$ (Blade profile zero-velocity power)
- $P_i = 88.63\,\text{W}$ (Induced hovering power)
- $U_{\text{tip}} = 120.0\,\text{m/s}$ (Rotor tip blade speed)
- $v_0 = 4.03\,\text{m/s}$ (Mean rotor induced hover velocity)
- $d_0 = 0.6$ (Fuselage drag ratio)
- $s = 0.05$ (Rotor solidity)
- $\rho = 1.225\,\text{kg/m}^3$ (Air density at sea level)
- $A = 0.503\,\text{m}^2$ (Rotor disc area)
- Zero-speed Hover Power: $P_{\text{hover}} = P_0 + P_i \approx 168.49\,\text{W}$.

#### A. Navigation Energy ($E_{\text{nav}}$)
The physical energy expended by the UAV strictly while navigating toward the mission destination prior to arrival:
$$E_{\text{nav}, j} = \sum_{t=1}^{\min(t_{\text{arr}, j}, T)} P_j(t) \cdot \Delta t \quad [\text{Joules, J}]$$
$$\text{Fleet Navigation Energy: } E_{\text{nav, fleet}} = \frac{1}{1000} \sum_{j=1}^N E_{\text{nav}, j} \quad [\text{kiloJoules, kJ}]$$
*Note:* If a UAV fails to arrive ($t_{\text{arr}, j} = \infty$), $E_{\text{nav}, j}$ integrates over all $T$ timesteps.

#### B. Service / Post-Arrival Hovering Energy ($E_{\text{serv}}$)
The physical energy expended by the UAV remaining on station after reaching its destination (hovering at target or loitering):
$$E_{\text{serv}, j} = \begin{cases} \sum_{t=t_{\text{arr}, j} + 1}^{T} P_j(t) \cdot \Delta t & \text{if } t_{\text{arr}, j} < T \\ 0 & \text{if } t_{\text{arr}, j} \ge T \end{cases} \quad [\text{Joules, J}]$$
$$\text{Fleet Service Energy: } E_{\text{serv, fleet}} = \frac{1}{1000} \sum_{j=1}^N E_{\text{serv}, j} \quad [\text{kiloJoules, kJ}]$$

#### C. Total Episode Energy ($E_{\text{total}}$)
The total combined propulsion energy over the full episode duration:
$$E_{\text{total, fleet}} = E_{\text{nav, fleet}} + E_{\text{serv, fleet}} \quad [\text{kiloJoules, kJ}]$$

---

### 3.2 Trajectory Length Metrics

#### A. Navigation Path Length ($L_{\text{nav}}$)
The cumulative 3D Euclidean distance traversed by UAV $j$ up to its arrival timestep:
$$L_{\text{nav}, j} = \sum_{t=1}^{\min(t_{\text{arr}, j}, T)} \|\mathbf{p}_j(t) - \mathbf{p}_j(t-1)\|_2 \quad [\text{meters, m}]$$
$$\text{Mean Fleet Navigation Path Length: } \bar{L}_{\text{nav}} = \frac{1}{N} \sum_{j=1}^N L_{\text{nav}, j} \quad [\text{m}]$$

#### B. Total Episode Path Length ($L_{\text{total}}$)
The total distance traversed over all $T$ timesteps including any post-goal wandering:
$$L_{\text{total}, j} = \sum_{t=1}^{T} \|\mathbf{p}_j(t) - \mathbf{p}_j(t-1)\|_2 \quad [\text{meters, m}]$$
$$\text{Mean Fleet Total Path Length: } \bar{L}_{\text{total}} = \frac{1}{N} \sum_{j=1}^N L_{\text{total}, j} \quad [\text{m}]$$
*Scientific Note:* For stationary hovering baselines like A*, $L_{\text{nav}} \equiv L_{\text{total}}$. For learned policies that exhibit limit cycles or wandering around the waypoint, $L_{\text{total}} \gg L_{\text{nav}}$.

---

### 3.3 Goal Distance Error & Success Rates

#### A. Final Goal Distance Error ($e_{\text{goal}}$)
The 3D Euclidean distance between the UAV's position at the end of the episode and its designated station:
$$e_{\text{goal}, j} = \|\mathbf{p}_j(T) - \mathbf{g}_j\|_2 \quad [\text{meters, m}]$$
$$\text{Mean Fleet Goal Error: } \bar{e}_{\text{goal}} = \frac{1}{N} \sum_{j=1}^N e_{\text{goal}, j} \quad [\text{m}]$$

#### B. Per-UAV Success Rate ($S_{\text{uav}}$)
The percentage of UAVs in the fleet that satisfy the arrival condition at least once during the episode:
$$S_{\text{uav}} = \frac{100\%}{N} \sum_{j=1}^N \mathbb{I}\left( t_{\text{arr}, j} \le T \right)$$

#### C. Fleet Mission Success Rate ($S_{\text{fleet}}$)
Binary indicator of complete fleet mission fulfillment (all $N$ UAVs reach their designated stations within $T$ timesteps):
$$S_{\text{fleet}} = 100\% \times \prod_{j=1}^N \mathbb{I}\left( t_{\text{arr}, j} \le T \right)$$

---

### 3.4 Safety, Clearance & Collisions

#### A. Obstacle Clearance ($c_{\min}$)
Minimum distance from any UAV center to the surface of any obstacle cylinder:
$$c_{j,k}(t) = \sqrt{(x_j(t) - c_{x,k})^2 + (y_j(t) - c_{y,k})^2} - (R_k + r_{\text{safe}})$$
$$\text{Fleet Minimum Clearance: } c_{\min} = \min_{j, k, t} c_{j,k}(t) \quad [\text{meters, m}]$$

#### B. Safety Interventions & Physical Collisions
- **Intervention Count:** The number of times the physical safety layer clamped or deflected an action vector that would have violated obstacle clearance.
- **Physical Collisions:** The number of unmitigated penetrations where $c_{j,k}(t) < 0$.

---

### 3.5 Computational MEC Task Offloading Metrics

Each UAV receives dynamic IoT computational tasks characterised by data size $D_m$ [bits], computation cycles $C_m$ [cycles], and strict latency deadline $\tau_m$ [seconds].

#### A. Task Completion Rate ($R_{\text{task}}$)
$$\text{Tasks Done (\%)} = \frac{\text{Completed Tasks}}{\text{Total Arrived Tasks}} \times 100\%$$

#### B. Mean Processing Delay ($\bar{D}$)
$$\bar{D} = \frac{1}{K_{\text{comp}}} \sum_{m=1}^{K_{\text{comp}}} D_{\text{proc}, m} \quad [\text{seconds, s}]$$
where $D_{\text{proc}, m} = t_{\text{tx}, m} + t_{\text{exe}, m}$.

#### C. Offloading Distribution
Percentage of tasks executed:
- **Local:** on UAV onboard CPU ($f_{\text{local}} = 1.5\,\text{GHz}$)
- **Edge:** offloaded via wireless uplink to Ground Station MEC ($f_{\text{edge}} = 10.0\,\text{GHz}$)
- **Peer-to-Peer:** offloaded via UAV-to-UAV sidelink to neighboring UAV ($f_{\text{peer}} = 1.5\,\text{GHz}$)

---

## 4. Multi-Seed Statistical Aggregation Protocol

### 4.1 Evaluation Seeds
All benchmark experiments must evaluate over the fixed canonical evaluation seed suite:
$$\mathcal{S}_{\text{eval}} = \{101, 102, 103, 104, 105\}$$
- Seeds govern pseudo-random IoT task arrivals, channel fading realizations, and initial velocity perturbations.
- Initial UAV starting coordinates and obstacle layouts remain deterministic across runs within the same scenario preset.

### 4.2 Aggregation Rules
For any reported scalar metric $X$, report the sample mean $\bar{X}$ and sample standard deviation $s_X$:
$$\bar{X} = \frac{1}{|\mathcal{S}_{\text{eval}}|} \sum_{s \in \mathcal{S}_{\text{eval}}} X_s$$
$$s_X = \sqrt{\frac{1}{|\mathcal{S}_{\text{eval}}| - 1} \sum_{s \in \mathcal{S}_{\text{eval}}} (X_s - \bar{X})^2}$$
In summary tables and text, format as:
$$\bar{X} \pm s_X$$

---

## 5. Software & Execution Environment

- **Operating System:** Windows 11 Pro (x86_64)
- **Python Runtime:** Python 3.14.8 (`C:\Python314\python.exe`)
- **Core Dependencies:**
  - `torch`: 2.14.1+cpu
  - `numpy`: 2.4.2
  - `pandas`: 2.3.1
  - `scipy`: 1.15.1
  - `matplotlib`: 3.10.1
  - `pytest`: 9.1.1
- **Hardware Profile:** 14-thread CPU execution, deterministic seed control via PyTorch and NumPy random generators.
- **Output Artifacts Directory:** All Phase 2 evaluation artifacts are strictly output to:
  `results/audit_phase_2/`
