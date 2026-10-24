# Mathematical Model: Implemented Equations and Formulations

This document provides a rigorous, comprehensive reference of every mathematical equation actually implemented in the `research_v2` codebase. Every formula is accompanied by variable definitions, physical units, algorithmic context, and its downstream role in the simulation.

---

## 1. 3D Kinematics and Position Integration

### 1.1 Discrete-Time Kinematic Equations
* **Implementation**: `env/multi_uav_env.py` $\to$ `step()`

$$\mathbf{p}_{j, t+1}^{\text{proposed}} = \mathbf{p}_{j, t} + \Delta \mathbf{p}_{j, t}$$

$$\mathbf{v}_{j, t} = \frac{\mathbf{p}_{j, t+1}^{\text{safe}} - \mathbf{p}_{j, t}}{\Delta t}$$

* **Plain English**: At each simulation step, the policy proposes a continuous 3D displacement vector $\Delta \mathbf{p}_{j, t}$. After passing through the analytical safety filter, the safe position $\mathbf{p}_{j, t+1}^{\text{safe}}$ determines the actual physical velocity vector $\mathbf{v}_{j, t}$.
* **Variables & Units**:
  * $\mathbf{p}_{j, t} = [x, y, z]^T$: Current 3D position of UAV $j$ $[\text{meters (m)}]$.
  * $\Delta \mathbf{p}_{j, t} = [\Delta x, \Delta y, \Delta z]^T$: Commanded displacement $[\text{meters (m)}]$.
  * $\mathbf{v}_{j, t}$: Executed 3D flight velocity vector $[\text{meters per second (m/s)}]$.
  * $\Delta t = 1.0\text{ s}$: Discrete simulation step interval $[\text{seconds (s)}]$.
* **Physical Velocity Clamping Constraint**:
  $$\|\Delta \mathbf{p}_{j, t}\| \le v_{\max} \Delta t \quad \text{where} \quad v_{\max} = 15.0\text{ m/s}$$
  If $\|\Delta \mathbf{p}\| > 15.0$, it is scaled proportionally: $\Delta \mathbf{p} \leftarrow \Delta \mathbf{p} \cdot \frac{v_{\max} \Delta t}{\|\Delta \mathbf{p}\|}$.

---

## 2. 3D Cylindrical Geometry & Continuous Collision Detection

### 2.1 Cylindrical Obstacle Model
* **Implementation**: `config.py` $\to$ `AirspaceConfig.obstacles`, `env/safety_filter.py`

A vertical cylindrical building $k$ is parameterized by centerline $(c_{x,k}, c_{y,k})$, radius $r_k$, and top height $h_k$:
$$\mathcal{C}_k = \left\{ (x, y, z) \in \mathbb{R}^3 \;\middle|\; (x - c_{x,k})^2 + (y - c_{y,k})^2 \le (r_k + d_{\text{margin}})^2 \quad \text{and} \quad 0 \le z \le h_k + d_{\text{margin}} \right\}$$

* **Parameters**:
  * $r_k = 10.0\text{ m}$: Physical cylinder radius $[\text{m}]$.
  * $d_{\text{margin}} = 2.0\text{ m}$: Safety clearance margin $[\text{m}]$.
  * $r_{\text{eff}} = r_k + d_{\text{margin}} = 12.0\text{ m}$: Effective collision boundary radius $[\text{m}]$.
  * $h_k \in [75.0, 85.0]\text{ m}$: Building elevation $[\text{m}]$.

### 2.2 Exact Continuous Line-Segment / Cylinder Intersection
* **Implementation**: `env/safety_filter.py` $\to$ `check_segment_cylinder_intersection()`

To prevent discrete tunneling (where a fast-moving UAV jumps through an obstacle between two timesteps), the path segment between current position $\mathbf{p}_0$ and proposed position $\mathbf{p}_1$ is parameterized by $t \in [0, 1]$:
$$\mathbf{p}(t) = \mathbf{p}_0 + t (\mathbf{p}_1 - \mathbf{p}_0) = \mathbf{p}_0 + t \Delta \mathbf{p}$$

In the horizontal $XY$ plane, the intersection occurs when $\|\mathbf{p}_{xy}(t) - \mathbf{c}_{xy}\|^2 = r_{\text{eff}}^2$. Substituting $\mathbf{p}(t)$ yields the quadratic equation:
$$a \cdot t^2 + b \cdot t + c = 0$$

Where:
$$a = \|\Delta \mathbf{p}_{xy}\|^2 = \Delta x^2 + \Delta y^2$$
$$b = 2 \left( \Delta x (p_{0,x} - c_x) + \Delta y (p_{0,y} - c_y) \right)$$
$$c = (p_{0,x} - c_x)^2 + (p_{0,y} - c_y)^2 - r_{\text{eff}}^2$$

The discriminant is $\Delta = b^2 - 4ac$:
- If $\Delta < 0$: No intersection in the horizontal plane.
- If $\Delta \ge 0$: Entry and exit parameters in $XY$ are $t_{xy, 1} = \frac{-b - \sqrt{\Delta}}{2a}$, $t_{xy, 2} = \frac{-b + \sqrt{\Delta}}{2a}$.

Simultaneously, the vertical interval $[t_{z,1}, t_{z,2}]$ is evaluated along $Z$:
$$t_a = \frac{z_{\min} - p_{0,z}}{\Delta z}, \quad t_b = \frac{z_{\max} - p_{0,z}}{\Delta z}$$

The continuous 3D collision interval is the intersection of the two intervals:
$$t_{\text{enter}} = \max(0.0, t_{z,1}, t_{xy,1}), \quad t_{\text{exit}} = \min(1.0, t_{z,2}, t_{xy,2})$$
If $t_{\text{enter}} \le t_{\text{exit}}$, an intersection occurs at fractional distance $t_{\text{enter}}$ along the flight path.

---

## 3. Rotary-Wing Aerodynamic Propulsion Power

### 3.1 Total Mechanical Propulsion Power
* **Implementation**: `env/aerodynamics.py` $\to$ `AerodynamicsModel.power()`
* **Reference**: Y. Zeng, J. Xu, and R. Zhang, *IEEE Transactions on Wireless Communications*, 2019.

$$P(v) = \underbrace{P_0 \left(1 + \frac{3 v^2}{U_{\text{tip}}^2}\right)}_{\text{Blade Profile Power } P_{\text{profile}}} + \underbrace{P_i \left(\sqrt{1 + \frac{v^4}{4 v_0^4}} - \frac{v^2}{2 v_0^2}\right)^{1/2}}_{\text{Induced Power } P_{\text{induced}}} + \underbrace{\frac{1}{2} d_0 \rho s A v^3}_{\text{Parasite Drag Power } P_{\text{parasite}}}$$

### 3.2 Physical Definitions and Constants

| Symbol | Parameter | Implemented Value | Units | Physical Interpretation |
|---|---|---|---|---|
| $v$ | Airspeed | $\in [0, 15]$ | $\text{m/s}$ | Horizontal flight speed |
| $P_0$ | Blade profile power | $79.86$ | $\text{W}$ | Power to spin blades in pure hover |
| $P_i$ | Induced power in hover | $88.63$ | $\text{W}$ | Power to produce lift in pure hover |
| $U_{\text{tip}}$ | Rotor tip speed | $120.0$ | $\text{m/s}$ | Linear blade tip velocity |
| $v_0$ | Mean induced velocity | $4.03$ | $\text{m/s}$ | Downward airflow velocity in hover |
| $d_0$ | Fuselage drag ratio | $0.6$ | dimensionless | Aerodynamic form drag coefficient |
| $\rho$ | Air density | $1.225$ | $\text{kg/m}^3$ | Sea-level ambient atmospheric density |
| $s$ | Rotor solidity | $0.05$ | dimensionless | Ratio of blade area to disc area |
| $A$ | Rotor disc area | $0.5$ | $\text{m}^2$ | Area swept by spinning rotors |

### 3.3 Aerodynamic Power Behavior
- **Hover ($v=0$)**: $P(0) = P_0 + P_i = 79.86 + 88.63 = 168.49\text{ W}$.
- **Translational Lift Sweet Spot ($v \approx 8 - 11\text{ m/s}$)**: As forward airspeed increases, fresh incoming airflow provides aerodynamic lift, decreasing $P_{\text{induced}}$ faster than parasite drag grows. Minimum power occurs around $10\text{ m/s}$ ($P \approx 145\text{ W}$).
- **High-Speed Regime ($v > 12\text{ m/s}$)**: Parasite fuselage drag grows cubically ($v^3$), dominating total power consumption ($P(15) \approx 230\text{ W}$).

---

## 4. Wireless Communication and Channel Capacity

### 4.1 Free-Space Path Loss and Channel Gain
* **Implementation**: `env/comms_compute.py` $\to$ `WirelessChannelModel.data_rate()`

$$h(d) = \frac{\beta_0}{d^2}$$

* **Variables & Units**:
  * $d = \|\mathbf{p}_{\text{uav}} - \mathbf{p}_{\text{rx}}\|_2$: 3D Euclidean distance $[\text{m}]$.
  * $\beta_0 = 10^{\beta_{0,\text{dB}} / 10} = 10^{-50 / 10} = 10^{-5}$: Reference channel power gain at $d_0 = 1\text{ m}$ $[\text{dimensionless}]$.
  * $h(d)$: Effective channel power gain at distance $d$ $[\text{dimensionless}]$.

### 4.2 Signal-to-Noise Ratio (SNR) and Shannon Capacity with OFDMA
$$\text{SNR} = \frac{P_{\text{tx}} \cdot h(d)}{\sigma^2}$$

$$R(d, K_{\text{tx}}) = \left(\frac{B}{K_{\text{tx}}}\right) \log_2(1 + \text{SNR})$$

* **Variables & Units**:
  * $P_{\text{tx}} = 0.1\text{ W}$: UAV RF transmit power ($100\text{ mW}$) $[\text{Watts}]$.
  * $\sigma^2 = 10^{(\sigma_{\text{dBm}}^2 - 30)/10} = 10^{(-100 - 30)/10} = 10^{-13}\text{ W}$: Thermal noise power $[\text{Watts}]$.
  * $B = 20\text{ MHz} = 20 \times 10^6\text{ Hz}$: Total RF bandwidth $[\text{Hertz}]$.
  * $K_{\text{tx}}$: Number of UAVs simultaneously transmitting over the band (OFDMA subchannel slicing) $[\text{integer}]$.
  * $R$: Achievable data transmission rate $[\text{bits per second (bps)}]$.

### 4.3 Wireless Transmission Energy
* **Implementation**: `env/comms_compute.py` $\to$ `WirelessChannelModel.transmit_energy()`

$$T_{\text{tx}} = \frac{D}{R}$$

$$E_{\text{tx}} = P_{\text{tx}} \cdot T_{\text{tx}} = P_{\text{tx}} \cdot \left(\frac{D}{R}\right)$$

* **Variables & Units**:
  * $D$: Task input data payload size $[\text{bits}]$.
  * $T_{\text{tx}}$: Wireless transmission delay $[\text{seconds (s)}]$.
  * $E_{\text{tx}}$: Electrical energy consumed by the RF transceiver $[\text{Joules (J)}]$.

---

## 5. Computation Offloading and MEC Processing

### 5.1 Processor Dynamic Energy Consumption (DVFS)
* **Implementation**: `env/comms_compute.py` $\to$ `ComputationManager`

$$E_{\text{comp}} = \kappa \cdot f^2 \cdot C$$

* **Variables & Units**:
  * $\kappa = 10^{-28}\text{ J/(cycle}\cdot\text{Hz}^2)$: Effective switched capacitance of processor chip.
  * $f$: Processor operating frequency $[\text{Hz = cycles/second}]$.
  * $C$: Required CPU computation cycles $[\text{cycles}]$.
  * $E_{\text{comp}}$: Computational energy consumed by processor $[\text{Joules (J)}]$.

### 5.2 Offloading Execution Destinations

1. **Local UAV Execution**:
   $$f_{\text{eff, local}} = f_{\text{local}} = 1.5 \times 10^9\text{ Hz}$$
   $$T_{\text{wait}} = \frac{Q_{\text{local}}}{f_{\text{eff, local}}}, \quad T_{\text{comp}} = \frac{C}{f_{\text{eff, local}}}$$
   $$T_{\text{total, local}} = T_{\text{wait}} + T_{\text{comp}}, \quad E_{\text{local}} = \kappa f_{\text{local}}^2 C$$

2. **Peer UAV Cooperative Execution**:
   $$f_{\text{eff, peer}} = \frac{f_{\text{peer}}}{2.0 \text{ if peer is busy else } 1.0}$$
   $$T_{\text{total, peer}} = T_{\text{tx, peer}} + \frac{Q_{\text{peer}}}{f_{\text{eff, peer}}} + \frac{C}{f_{\text{eff, peer}}}$$
   $$E_{\text{peer}} = E_{\text{tx, peer}} + \kappa f_{\text{eff, peer}}^2 C$$

3. **Ground Edge Server Execution**:
   $$f_{\text{eff, edge}} = \frac{f_{\text{edge}}}{K_{\text{edge}} + K_{\text{background}}} = \frac{20 \times 10^9}{K_{\text{edge}} + 2}$$
   $$T_{\text{total, edge}} = T_{\text{tx, edge}} + \frac{Q_{\text{edge}}}{f_{\text{eff, edge}}} + \frac{C}{f_{\text{eff, edge}}}$$
   $$E_{\text{edge}} = E_{\text{tx, edge}} \quad (\text{Edge server grid power is not billed to the UAV battery})$$

---

## 6. Multi-Objective Composite Reward Function

* **Implementation**: `env/multi_uav_env.py` $\to$ `step()`

$$R_t = R_{\text{task}} + R_{\text{deadline}} + R_{\text{energy}} + R_{\text{delay}} + R_{\text{safety}} + R_{\text{progress}} + R_{\text{station}} + R_{\text{loiter}} + R_{\text{arrival}}$$

### Detailed Term Breakdown

$$\begin{aligned}
R_{\text{task}} &= +2.0 \cdot N_{\text{completed}} \\
R_{\text{deadline}} &= -2.5 \cdot N_{\text{missed}} \\
R_{\text{energy}} &= -0.001 \cdot E_{\text{total}} \quad (E_{\text{total}} = E_{\text{prop}} + E_{\text{tx}} + E_{\text{comp}} \text{ in Joules}) \\
R_{\text{delay}} &= -1.0 \cdot \sum T_{\text{delay}} \\
R_{\text{safety}} &= -5.0 \cdot \min(6, N_{\text{interventions}}) \\
R_{\text{progress}} &= +4.0 \sum_j (d_{j, t-1} - d_{j, t}) - 0.5 \sum_j \left(\frac{d_{j, t}}{20.0}\right) \\
R_{\text{station}} &= +5.0 \quad (\text{if } d \le 5.0\text{ m}) + 2.0 \quad (\text{if } \|\mathbf{v}\| \le 1.5\text{ m/s}) \\
R_{\text{loiter}} &= -2.0 \cdot \left(\frac{\|\mathbf{v}\|}{v_{\max}}\right) \quad (\text{if } d \le 5.0\text{ m and } \|\mathbf{v}\| > 1.5\text{ m/s}) \\
R_{\text{arrival}} &= +80.0 \quad (\text{milestone bonus awarded at arrival step})
\end{aligned}$$

---

## 7. MAPPO Optimization Formulations

### 7.1 Temporal Difference Error and Generalized Advantage Estimation (GAE)
* **Implementation**: `models/mappo.py` $\to$ `RolloutBuffer.compute_gae()`

$$\delta_t^V = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$

$$\hat{A}_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \gamma \lambda (1 - d_t) \hat{A}_{t+1}$$

$$\hat{R}_t = \hat{A}_t + V(s_t)$$

* **Parameters**:
  * $\gamma = 0.99$: Discount factor.
  * $\lambda = 0.95$: GAE exponential weight parameter.
  * $d_t$: Termination indicator ($d_t = 1$ on terminal episode state).

### 7.2 PPO Clipped Surrogate Objective
* **Implementation**: `models/mappo.py` $\to$ `MAPPOAgent.update()`

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)} = \exp\left( \log \pi_\theta(a_t \mid o_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid o_t) \right)$$

$$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right] - c_2 \mathcal{H}(\pi_\theta(\cdot \mid o_t))$$

$$L^{\text{VF}}(\phi) = \frac{1}{|B|} \sum_{s \in B} \left( V_\phi(s) - \hat{R} \right)^2$$

* **Parameters**:
  * $\epsilon = 0.2$: PPO clip boundary.
  * $c_2 = 0.01$: Entropy regularization coefficient.
  * $\mathcal{H}$: Policy entropy bonus to prevent premature policy collapse.
