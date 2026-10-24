# Code Walkthrough: File-by-File Technical Guide

This guide provides a beginner-friendly, comprehensive file-by-file walkthrough of the entire UAV simulation codebase. It explains which file executes first, how data flows through the system, and which functions call other functions.

---

## 1. Where Does the Code Start? (Entry Points)

There are three primary entry points depending on what you want to achieve:

```
                  ┌────────────────────────────────────────┐
                  │           WHICH SCRIPT TO RUN?         │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
   [train.py]                   [evaluate.py]             [simulator_server.py]
  Train all 4 RL models       Benchmark models across    Launch interactive 3D Web
  (MAPPO, IPPO, MADDPG, CPPO) 5 held-out test seeds      Dashboard (Three.js/Flask)
```

1. **`train.py`**: The training pipeline. Trains the RL algorithms using curriculum learning, evaluates them on periodic validation seeds, and saves checkpoint weights to `results/checkpoints/`.
2. **`evaluate.py`**: The scientific benchmarking script. Loads trained checkpoints alongside classical baselines (3D A*, 3D SAPF, 3D RRT*), runs 5 held-out test seeds (101 to 105), records physical 10 Hz flight telemetry to `results/trajectory_records.csv`, and generates evaluation CSV summaries.
3. **`simulator_server.py`**: The interactive visualizer backend. Launches a Flask REST + WebSocket web server on `http://127.0.0.1:5000` to serve the 3D AeroPlan dashboard (rendered with Three.js in your web browser).

---

## 2. File-by-File Breakdown

### `config.py` — The Central Configuration File
* **Role in Project**: Single source of truth. Every other module imports its physical, aerodynamic, communication, and RL parameters from this file.
* **Key Classes**:
  * `AirspaceConfig`: Defines the 3D workspace boundaries ($[0, 200] \times [0, 200] \times [0, 100]\text{ m}$), obstacle cylinder tuples `(cx, cy, radius, height)`, and the Ground Edge Server position $(100, 100, 0)$.
  * `UAVPhysicsConfig`: Defines UAV physical parameters (mass = $1.5\text{ kg}$, $v_{\max} = 15.0\text{ m/s}$, $\Delta t = 1.0\text{ s}$, inter-UAV minimum separation $d_{\text{safe}} = 8.0\text{ m}$) and Zeng aerodynamic constants.
  * `CommunicationConfig`: Defines RF spectrum parameters (bandwidth $B = 20\text{ MHz}$, transmit power $P_{\text{tx}} = 0.1\text{ W}$, carrier frequency $2.4\text{ GHz}$).
  * `ComputationConfig`: Defines processor clock frequencies ($f_{\text{local}} = 1.5\text{ GHz}$, $f_{\text{peer}} = 1.5\text{ GHz}$, $f_{\text{edge}} = 20.0\text{ GHz}$) and switched capacitance $\kappa = 10^{-28}\text{ J/(cycle}\cdot\text{Hz}^2)$.
  * `TaskConfig`: Defines task Poisson arrival rates and uniform cycle/data size ranges.
  * `MAPPOConfig`: Defines RL hyperparameters ($\gamma = 0.99, \lambda = 0.95, \epsilon_{\text{clip}} = 0.2$).
  * `MasterConfig`: Bundles all the above dataclasses into a single object.
* **Helper Function**: `get_preset_config(preset, num_uavs)` returns a `MasterConfig` populated with a named scenario preset (`benchmark`, `metropolis`, `chokepoint`, `canyon`).

---

### `env/multi_uav_env.py` — The 3D Multi-Agent Environment
* **Role in Project**: The core physics and game loop, implementing the standard Gymnasium interface.
* **Key Lifecycle Functions**:
  1. `__init__(cfg)`:
     - Initializes airspace bounds, loads obstacle cylinders, and sets up sub-managers: `SafetyFilter`, `AerodynamicsModel`, `WirelessChannelModel`, and `ComputationManager`.
     - Establishes symmetric start positions on an outer perimeter ring ($R_{\text{start}} = 65\text{ m}$) and assigns target stations on an inner ring ($R_{\text{station}} = 20\text{ m}$) at $Z = 50\text{ m}$.
     - Calls `verify_station_feasibility()` to mathematically assert that all assigned stations are outside obstacle cylinders.
  2. `reset(seed, scenario_mode, curriculum_ratio)`:
     - Resets the episode timestep to $t = 0$.
     - Re-spawns UAVs at their initial deployment coordinates.
     - Samples initial Poisson computational tasks for all UAVs via `task_gen.generate_tasks()`.
     - Returns initial local observations $o_j$ for each UAV and the centralized global state $s_t$.
  3. `step(actions)`:
     - **Input**: Dictionary mapping each UAV index $j$ to its action `{"move": np.ndarray(3), "offload": int}`.
     - **Displacement Check**: Clamps displacement magnitude to $v_{\max} \Delta t$. If the UAV is within the station deadband ($0.5\text{ m}$), velocity is zeroed to prevent hovering jitter.
     - **Safety Intervention**: Passes proposed positions to `self.safety.filter_trajectories()`, which returns safe coordinates `safe_pos`.
     - **Aerodynamic Energy**: Calculates actual airspeed $v = \|\mathbf{p}_{\text{safe}} - \mathbf{p}_{\text{old}}\| / \Delta t$ and queries `self.aero.energy(v, delta_t)` to accumulate mechanical propulsion energy in Joules.
     - **MEC Computation & Wireless Slicing**: Evaluates offloading choices (Local, Peer, Edge) through `self.comp`, taking into account OFDMA spectrum division and Edge CPU multi-tenancy.
     - **Composite Reward**: Computes multi-objective reward combining task success, deadline penalties, total energy consumption, latency, safety interventions, and distance progress toward the station.
     - **Returns**: `(obs_dict, global_state, reward, done, info)`.

---

### `env/safety_filter.py` — The Analytical Kinematic Projection Safety Filter
* **Role in Project**: Provides deterministic safety guarantees to prevent the UAV policy from colliding with obstacles or peers during training and evaluation.
* **Key Functions**:
  1. `check_segment_cylinder_intersection(p0, p1, cx, cy, r, h)`:
     - Performs continuous ray-cylinder intersection using the parametric segment equation $\mathbf{p}(t) = \mathbf{p}_0 + t(\mathbf{p}_1 - \mathbf{p}_0)$ for $t \in [0, 1]$.
     - Solves the quadratic equation $a t^2 + b t + c = 0$ in the horizontal plane and intersects it with vertical height bounds $[z_{\min}, z_{\max}]$.
     - Returns `(intersects: bool, t_entry: float)`, completely eliminating discrete tunneling.
  2. `filter_trajectories(proposed_pos, current_pos, enable_filter)`:
     - For any UAV proposing a coordinate inside an obstacle cylinder, clamps the movement along the ray to $t_{\text{entry}}$ with a safety margin.
     - Checks all pairwise inter-UAV distances. If $\|\mathbf{p}_j - \mathbf{p}_k\| < d_{\text{safe}} = 8.0\text{ m}$, projects both UAVs symmetrically outward along their relative vector.
     - Clamps coordinates within workspace geofence bounds.
     - Accurately increments `proposed_collisions`, `interventions`, and `actual_collisions`.

---

### `env/aerodynamics.py` — Rotary-Wing Propulsion Model
* **Role in Project**: Converts flight velocity into true mechanical power and energy using the aerodynamics model of Zeng et al. (*IEEE Transactions on Wireless Communications*, 2019).
* **Key Functions**:
  1. `power(speed)`:
     - Evaluates the three aerodynamic power components:
       - **Blade Profile Power**: $P_{\text{profile}}(v) = P_0 (1 + 3 v^2 / U_{\text{tip}}^2)$ (quadratic in $v$).
       - **Induced Power**: $P_{\text{induced}}(v) = P_i \sqrt{\sqrt{1 + v^4 / (4 v_0^4)} - v^2 / (2 v_0^2)}$ (decreases with forward speed due to translational lift).
       - **Parasite Drag Power**: $P_{\text{parasite}}(v) = \frac{1}{2} d_0 \rho s A v^3$ (cubic in $v$).
     - Sums all three to return instantaneous power in Watts.
  2. `energy(speed, delta_t)`:
     - Returns energy consumed: $E = P(v) \cdot \Delta t$ in Joules.

---

### `env/comms_compute.py` — RF Channel & MEC Processor Queues
* **Role in Project**: Models the physics of wireless data transmission and CPU task processing queues.
* **Key Classes**:
  1. `WirelessChannelModel`:
     - `data_rate(distance, num_concurrent)`: Evaluates free-space path loss $h(d) = \beta_0 / d^2$, computes Signal-to-Noise Ratio (SNR), and evaluates Shannon capacity with OFDMA subchannel sharing: $R = \frac{B}{K_{\text{tx}}} \log_2(1 + \text{SNR})$.
     - `transmit_energy(data_size, distance)`: Computes transmission energy $E_{\text{tx}} = P_{\text{tx}} \cdot (D / R)$ in Joules.
  2. `ComputationManager`:
     - Tracks persistent task queues for Local, Peer, and Edge processors.
     - `compute_local(task)`: Models on-board execution at $1.5\text{ GHz}$. Dynamic voltage/frequency scaling energy: $E_{\text{comp}} = \kappa f^2 C$.
     - `compute_peer(task, distance, comm)`: Models wireless transfer to a peer UAV followed by cooperative execution.
     - `compute_edge(task, distance, comm)`: Models wireless uplink transmission to the Ground Edge Server followed by execution on the $20\text{ GHz}$ multi-tenant MEC server.
     - `step_queues(delta_t)`: Advances physical time and drains processed cycles across all CPUs.

---

### `models/mappo.py` — Proposed Joint MAPPO Algorithm
* **Role in Project**: Implements Multi-Agent PPO with Centralized Training and Decentralized Execution (CTDE).
* **Key Classes**:
  1. `ActorNetwork`:
     - Multi-layer perceptron with LayerNorm and Tanh activations.
     - **Dual Heads**:
       - `offload_head`: Outputs logits for Categorical distribution over discrete offloading targets.
       - `move_mean_head`: Outputs mean 3D continuous velocity displacement.
     - Supports both `pure` policy mode (100% learned neural policy) and `residual` policy mode (learned correction added to nominal APF guidance).
  2. `CriticNetwork`:
     - Centralized state-value network $V(s)$ taking the full global state $s \in \mathbb{R}^{40}$ and predicting expected cumulative future discounted return.
  3. `RolloutBuffer`:
     - Stores observations, global states, actions, log probabilities, rewards, values, and termination flags.
     - `compute_gae(last_val, last_terminated, gamma, gae_lambda)`: Computes Generalized Advantage Estimations $\hat{A}_t$ and returns backwards through time.
  4. `MAPPOAgent`:
     - `get_actions(obs_dict, deterministic)`: Generates actions for all UAVs in the fleet.
     - `update(buffer, next_state, ...)`: Performs mini-batch gradient updates using the PPO clipped surrogate objective:
       $$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] - c_2 \mathcal{H}(\pi_\theta)$$

---

### `models/astar.py` — 3D Grid Graph Search A* Planner
* **Role in Project**: Serves as the primary classical path-planning benchmark.
* **Key Class**: `AStar3DPlanner`:
  - Discretizes the 3D airspace into a regular grid with resolution $\Delta x = 3.0\text{ m}$.
  - Explores spatial neighbors using **26-connected spatial adjacency** (horizontal, vertical, and diagonal spatial moves).
  - Evaluates priority queue cost $f(n) = g(n) + h(n)$, where $g(n)$ is path cost and $h(n) = \|\mathbf{p}_n - \mathbf{p}_{\text{goal}}\|$ is the Euclidean distance heuristic.
  - Implements fast line-of-sight shortcutting to eliminate discrete grid stairstep artifacts.
  - Paired with greedy lowest-latency/energy task offloading in `AStarGreedyAgent`.

---

### `models/apf.py` — Sequential Adaptive Potential Field (SAPF)
* **Role in Project**: Physics-inspired deterministic navigation baseline.
* **Key Class**: `SAPFNavigator`:
  - Computes net guidance force: $\mathbf{F}_{\text{net}} = \mathbf{F}_{\text{att}} + \mathbf{F}_{\text{rep}} + \mathbf{F}_{\text{tangent}} + \mathbf{F}_{\text{peer}}$.
  - Introduces a **tangential circulation field** ($\mathbf{F}_{\text{tangent}}$) orthogonal to the obstacle normal. This creates a virtual fluid vortex around cylinders, allowing UAVs to slide around obstacles rather than getting trapped in head-on local minima.

---

### `plots/plot_curves.py` — 2D Quantitative Performance Figures
* **Role in Project**: Generates high-resolution scientific line plots, convergence curves, and Pareto trade-off diagrams.
* **Key Functions**:
  - `plot_all_marl_training_convergence()`: Plots Episode Return, Task Completion Rate, Fleet Energy, and Held-Out Validation Return over training episodes for all 4 models.
  - `plot_pareto_latency_energy()`: Generates the Multi-Objective Pareto Frontier plot (Latency vs. Energy) with error bars and IEEE Transactions serif styling.
  - `plot_offload_distribution_stacked()`: Generates the percentage breakdown of Local, Peer, and Edge task executions across all algorithms.

---

### `plots/render_publication_figures.py` — 3D/2D Spatial Trajectory Figures
* **Role in Project**: Renders peer-reviewed journal quality 3D flight trajectories and 2D planar projection figures.
* **Key Functions**:
  - `generate_figure_b_multimodel_trajectories()`: Generates the dual-panel comparison figure:
    - **Panel (a)**: 3D Isometric Canyon corridor perspective showing trajectories of Proposed Hierarchical A*+MAPPO, 3D A*, 3D SAPF, and 3D RRT* with slender, semi-transparent cylindrical towers.
    - **Panel (b)**: 2D Planar XY projection showing obstacle cross-sections and dashed safety clearance rings.
  - `generate_figure_a_coordinated_multi_uav()`: Renders Figure 5 showing the 4-UAV fleet traversing the 20-building Metropolis skyline.
  - `generate_figure_c_potential_field_surface()`: Renders the 3D potential energy surface and 2D negative gradient streamline flow field ($-\nabla U$).
  - `plot_publication_3d_trajectory_benchmark()`: Backward-compatible 4-panel trajectory benchmark plotter for navigation experiment scripts.
