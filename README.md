# Research_V2: Deep Reinforcement Learning for Joint Task Offloading and Energy-Efficient Path Planning in Multi-UAV Edge Computing

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-darkgreen.svg)](https://gymnasium.farama.org/)
[![Tests](https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen.svg)](tests/test_framework.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, modular, and scientifically rigorous research framework for Multi-Agent Reinforcement Learning (**MAPPO**, **CPPO**, **IPPO**, **MADDPG**), 3D Geometric/Sampling/Potential Navigation (**3D A\***, **3D RRT\***, **3D SAPF**), and Classical Multi-Objective Mobile Edge Computing (MEC) baselines for joint continuous 3D UAV path planning and discrete computational task offloading.

---

## 1. Directory Structure

```
research_v2/
├── requirements.txt          # Framework dependencies (PyTorch, Gymnasium, NumPy, Pandas, Matplotlib, PyTest)
├── config.py                 # Central system configuration, physical constants & obstacle geometry
├── env/                      # Modular Environment, Physics & Safety Layer
│   ├── __init__.py
│   ├── aerodynamics.py       # Zeng et al. (2019) rotary-wing aerodynamic power model
│   ├── comms_compute.py      # Shannon RF rate, OFDMA bandwidth sharing, computation queues & MEC contention
│   ├── safety_filter.py      # Analytical kinematic projection filter for obstacle avoidance & peer separation
│   └── multi_uav_env.py      # Gymnasium multi-agent environment with station verification & decomposed rewards
├── models/                   # DRL Policies, Planners & Baselines
│   ├── __init__.py
│   ├── mappo.py              # Joint MAPPO (CTDE with dual Categorical/Gaussian heads & Centralized Critic)
│   ├── cppo.py               # Centralized PPO (CTCE upper-bound centralized coordination baseline)
│   ├── ippo.py               # Independent PPO (DTDE decentralized local critic baseline)
│   ├── maddpg.py             # Multi-Agent DDPG (Off-policy continuous actor + Centralized Q-Critic)
│   ├── astar.py              # 3D A* Path Planner (Discretized grid with obstacle clearance margin)
│   ├── rrt_star.py           # 3D RRT* Path Planner (Sampling-based asymptotic optimal tree with dynamic rewiring)
│   ├── apf.py                # 3D SAPF Navigator (Sequential adaptive potential field with tangential circulation)
│   └── baselines.py          # Classical MEC baselines (Energy-Deadline Greedy, Lowest-Latency, Nearest, Local, Edge, Random)
├── utils/                    # Utilities & Device Management
│   ├── __init__.py
│   └── device.py             # Unified DeviceManager (auto, cuda, cpu) with diagnostics & seed verification
├── plots/                    # Publication Plotting Suite
│   ├── __init__.py
│   ├── plot_curves.py        # Publication curves, Pareto frontiers, 3D trajectories, and bar breakdowns
│   └── plot_trajectory_comparison.py # Multi-navigator 3D trajectory visualizer (saves 300 DPI PNG & vector PDF)
├── docs/                     # Comprehensive Documentation & Defense Guides
│   ├── EQUATIONS_AND_CODE_GUIDE.md # Exhaustive mathematical derivations, code mappings & viva questions
│   └── PROJECT_WALKTHROUGH.md      # Architecture diagrams, step-by-step pipeline lifecycle & sequence flows
├── train.py                  # Training pipeline with validation scenarios, throughput metrics & ablation modes
├── evaluate.py               # Multi-seed held-out evaluation & scientific benchmarking across 9 methods
├── run_navigation_experiments.py # Experiment A: Dedicated 3D path planning & trajectory benchmark suite
├── plot_results.py           # Standalone publication figure regenerator (no retraining required)
├── main.py                   # Master CLI entry point (train / evaluate / plot / ablation / all)
├── tests/
│   └── test_framework.py     # 7-part automated verification & contract test suite
├── results/                  # Generated checkpoints, CSV logs, JSON configs & publication figures
│   ├── checkpoints/          # Best validation & final model weights (*.pt)
│   ├── figures/              # Publication figures (.png at 300 DPI, .pdf vector)
│   ├── benchmark_summary.csv # Multi-seed aggregate benchmark table (Mean ± Std)
│   ├── navigation_benchmark.csv # 3D navigation benchmark table (Path length, clearance, planning time, success)
│   ├── trajectory_records.csv# Granular step-by-step 3D flight coordinates across all navigators
│   ├── offloading_distribution.csv # % Local vs % Peer vs % Edge breakdown
│   ├── ablation_summary.csv  # Component ablation comparison table
│   └── benchmark_detailed.csv# Per-seed granular episode records
└── README.md
```

---

## 2. Core Physics & Theoretical Formulation

### 2.1 Rotary-Wing Aerodynamics (Zeng et al., 2019)
The propulsion power consumption for a rotary-wing UAV flying at instantaneous horizontal airspeed $v$ is modeled following the closed-form aerodynamic formulation:

$$P(v) = \underbrace{P_0 \left(1 + \frac{3v^2}{U_{\text{tip}}^2}\right)}_{\text{Blade Profile Power}} + \underbrace{P_i \left(\sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2}\right)^{1/2}}_{\text{Induced Power}} + \underbrace{\frac{1}{2} d_0 \rho s A v^3}_{\text{Parasite Drag Power}}$$

* **Hovering Power:** $P(0) = P_0 + P_i = 79.86\text{ W} + 88.63\text{ W} = 168.49\text{ W}$.
* **Optimal Cruise Speed:** Translational lift reduces induced power at moderate speeds, yielding an optimal cruising speed $v^* \approx 10.3\text{ m/s}$ where power drops to $\approx 126.5\text{ W}$ (a $\approx 25\%$ energy saving over hovering).
* **Anti-Loitering Damping:** Enforces zero velocity once within station deadband ($r \le 0.5\text{ m}$) to prevent high-speed orbiting or loitering waste.

### 2.2 Wireless Communication & Multi-User Contention
UAVs communicate with peer UAVs and the Ground Edge Server using Orthogonal Frequency-Division Multiple Access (OFDMA):
* **Shannon Channel Capacity with OFDMA Slicing:**
  $$R_j(d_j) = \frac{B}{K_{tx}} \log_2\left(1 + \frac{P_{tx} \beta_0 d_j^{-2}}{\sigma^2}\right)$$
  where $B = 20\text{ MHz}$, $K_{tx}$ is the instantaneous count of concurrent transmitting nodes sharing the band, $P_{tx} = 0.1\text{ W}$ ($20\text{ dBm}$), $\beta_0 = -50\text{ dB}$ is reference channel power gain at $1\text{ m}$, and $\sigma^2 = -100\text{ dBm}$ is noise power spectral density.
* **Transmission Delay & Energy:**
  $$T_{tx} = \frac{D_k}{R_j(d_j)}, \quad E_{tx} = P_{tx} \cdot T_{tx}$$

### 2.3 Mobile Edge Computing & DVFS Dynamics
Tasks are generated via a Poisson arrival process with data size $D_k \in [1, 5]\text{ Mbits}$ and workload $C_k = D_k \times X_k$ CPU cycles (where $X_k = 1000\text{ cycles/bit}$):
* **Ground Edge Server:** Total CPU capacity $f_{edge} = 20\text{ GHz}$ dynamically sliced across all concurrent offloaders:
  $$f_{edge\_eff} = \frac{f_{edge}}{K_{edge}}$$
* **Peer UAV Computing:** Offload target UAVs allocate $f_{peer} / 2$ to peer tasks while executing local tasks ($f_{uav} = 2.0\text{ GHz}$).
* **Local DVFS Energy:**
  $$E_{comp} = \kappa \, (f_{local})^2 \, C_k, \quad \kappa = 10^{-28}$$

### 2.4 Airspace Obstacles & Analytical Kinematic Safety Filter
* **Geometry:** 4 Cylindrical obstacles ($r = 10\text{ m}$, heights $75\text{--}85\text{ m}$) placed directly in between the initial deployment perimeter ring ($R_{start} = 65\text{ m}$) and mission stations ($R_{station} = 20\text{ m}$).
* **Analytical Projection Filter:** Intercepts proposed displacement vectors $\Delta \mathbf{p}_j$, analytically projecting trajectories onto obstacle tangent surfaces with a safety margin $d_{margin} = 2.0\text{ m}$ and enforcing pairwise separation $d_{safe} = 8.0\text{ m}$ between peer UAVs with non-zero fallback separation vectors.

### 2.5 Multi-Agent Reinforcement Learning Hierarchy
1. **Proposed Joint MAPPO (CTDE):** Decentralized dual-head actor policies (Gaussian continuous 3D velocity $\Delta \mathbf{p}_j$ + Categorical discrete offloading target $a_{off} \in \{0, \dots, N\}$) paired with a Centralized Value Function $V(S)$ conditioned on the full state. Optimized via Generalized Advantage Estimation (GAE, $\gamma=0.99, \lambda=0.95$).
2. **Centralized PPO (CPPO, CTCE):** Centralized Actor and Centralized Critic commanding joint actions for the entire fleet simultaneously (upper bound on coordination).
3. **Independent PPO (IPPO, DTDE):** Fully decentralized actors and decentralized local critics $V_j(o_j)$ operating under partial observability.
4. **Multi-Agent DDPG (MADDPG, CTDE):** Off-policy continuous actor with Gumbel-Softmax discrete offloading relaxation and Centralized Continuous Action-Value Critic $Q(S, A)$.

### 2.6 Dedicated 3D Path Planners & Classical MEC Baselines
* **3D A\* Path Planner:** Discretized 3D grid with heuristic distance cost and obstacle bypass waypoints.
* **3D RRT\* Path Planner:** Continuous sampling-based tree with dynamic rewiring radius for asymptotic path optimality around cylindrical obstacles.
* **3D SAPF Navigator:** Sequential Adaptive Potential Field with attractive goal force, nonlinear obstacle repulsive fields, dynamic inter-UAV repulsion, and tangential circulation to prevent local minima.
* **Classical MEC Baselines:**
  * **Energy-Deadline Aware Greedy MEC:** Optimizes $J = \text{Delay} + \beta E + \text{Penalty}_{deadline}$.
  * **Lowest-Latency MEC:** Greedily selects the route minimizing transmission plus computation delay.
  * **Nearest Feasible MEC:** Offloads to the geometrically nearest operational node.
  * **Local-Only Execution:** Forces 100% on-board CPU processing (lower bound for offloading benefits).
  * **Edge-Only Execution:** Forces 100% offloading to the Ground Edge Server (upper bound for edge reliance).
  * **Random Feasible MEC:** Stochastic random feasible node allocation (statistical lower bound).

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
* Python 3.9, 3.10, 3.11, 3.12, or 3.14
* Recommended: Virtual environment (`venv` or `conda`)
* OS: Windows, Linux, or macOS

### 3.2 Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Karthik7-hub/MULTI_UAV_RL.git
   cd MULTI_UAV_RL
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS (Bash):**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) PyTorch with GPU / CUDA Support:**
   If you have an NVIDIA GPU and want hardware acceleration, install PyTorch matching your CUDA version:
   ```bash
   # Example for CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 4. Quick Start & Execution Workflow

### A. Run Verification Tests First
Ensure all physical contracts, environment spaces, safety filters, and model dimensions pass:
```powershell
python -m pytest tests/
# or
python tests/test_framework.py
```

### B. Hardware Device Options
All entry points accept `--device auto` (default: auto-detects CUDA, falls back cleanly to CPU), `--device cuda`, or `--device cpu`:
```powershell
# Auto-detect CUDA if available, fallback to CPU
python main.py --mode all --device auto

# Explicitly target CUDA (raises informative error if CUDA is missing)
python main.py --mode train --model mappo --device cuda

# Force CPU execution
python main.py --mode train --model mappo --device cpu
```

### C. Training Models
Train individual MARL policies or train all policies sequentially:
```powershell
# Train Proposed Joint MAPPO
python train.py --model mappo --episodes 80 --seed 42 --device auto

# Train Centralized PPO (CPPO)
python train.py --model cppo --episodes 80 --seed 42 --device auto

# Train Independent PPO (IPPO) Baseline
python train.py --model ippo --episodes 80 --seed 42 --device auto

# Train Multi-Agent DDPG (MADDPG) Baseline
python train.py --model maddpg --episodes 80 --seed 42 --device auto

# Train all models sequentially
python train.py --model all --episodes 80 --seed 42 --device auto
```

### D. Component Ablation Experiments
Validate individual reward components and structural mechanisms:
```powershell
# Ablation 1: MAPPO without energy reward shaping
python train.py --model ablation --ablation no_energy --episodes 80 --device auto

# Ablation 2: MAPPO without safety penalty
python train.py --model ablation --ablation no_safety --episodes 80 --device auto

# Ablation 3: MAPPO navigation with fixed greedy offloading heuristic
python train.py --model ablation --ablation fixed_offload --episodes 80 --device auto
```

### E. Multi-Seed Benchmark Evaluation
Benchmarking across 9 distinct learning and classical methods over held-out random seeds:
```powershell
python evaluate.py --device auto --num_seeds 5
```

### F. Experiment A: 3D Path Planning & Trajectory Comparison Benchmark
Simulate and benchmark all 7 navigation algorithms (3D A*, 3D RRT*, 3D SAPF, MAPPO, IPPO, MADDPG, CPPO) under identical obstacle geometries:
```powershell
python run_navigation_experiments.py --device auto --seed 101
```
* Generates `results/trajectory_records.csv` (step-by-step 3D flight coordinates)
* Generates `results/navigation_benchmark.csv` (path length, clearance margin, planning latency, success rate)
* Generates `results/figures/3d_uav_trajectory_comparison.png` (300 DPI) & `3d_uav_trajectory_comparison.pdf` (vector PDF)

### G. Regenerate Figures from Saved Data (Without Retraining)
Instantly re-create all publication figures from stored CSV logs without re-running training:
```powershell
python plot_results.py
```

### H. End-to-End Execution Pipeline
Execute the full train $\to$ evaluate $\to$ plot pipeline in one command:
```powershell
python main.py --mode all --device auto
```

---

## 5. Generated Publication-Grade Figures

All figures are automatically formatted for academic publication and saved at 300 DPI to `results/figures/`:

| Figure Name | Description | Key Insight / Result |
| :--- | :--- | :--- |
| `3d_uav_trajectory_comparison.png` / `.pdf` | 4-panel visualizer: 3D spatial flight paths, top-down 2D bypass corridors, distance-to-station convergence, and metric comparison table. | Demonstrates MAPPO learning continuous aerodynamic curves around obstacles while RRT*/A* produce piecewise paths. |
| `trajectory_diagnostic_comparison.png` | 4-panel diagnostic comparison of 2D trajectories, station distances, flight velocities, and fleet cumulative energy over 100 s. | Proves velocity stabilization at optimal cruise speed ($10.3\text{ m/s}$) followed by smooth loitering deceleration. |
| `multimodel_performance_curves.png` | 4-panel comparative evaluation comparing MAPPO, IPPO, MADDPG, and 3D A*. | Shows superior return convergence and energy efficiency of MAPPO over decentralized baselines. |
| `mappo_vs_astar_diagnostic.png` | Pairwise comparison of Proposed MAPPO vs 3D A* Baseline. | Confirms smooth kinematic curvatures outperform rigid grid transitions in propulsion energy. |
| `mappo_vs_ippo_ablation.png` | CTDE vs DTDE ablation evaluating multi-agent coordination. | Demonstrates that centralized critic mitigates non-stationarity and eliminates inter-agent collisions. |
| `mappo_vs_maddpg_comparison.png` | Policy architecture comparison (On-Policy MAPPO vs Off-Policy MADDPG). | Highlights MAPPO's training stability and higher deadline satisfaction rate. |
| `offloading_distribution_breakdown.png` | Stacked horizontal bar chart showing % Local, % Peer, and % Edge decisions. | Demonstrates balanced multi-tier offloading adapting to wireless channel quality and queue backlogs. |
| `pareto_latency_energy_tradeoff.png` | Multi-objective Pareto frontier showing Latency (s) vs Fleet Energy (kJ). | Establishes MAPPO at the optimal non-dominated frontier tradeoff point. |
| `ablation_study_comparison.png` | Grouped comparative analysis of reward and modular ablations. | Quantifies performance drops when removing energy shaping or safety projection. |
| `3d_flight_trajectories.png` | 3D airspace visualization with cylindrical obstacle geometries and flight paths. | Visualizes fleet clearance from cylindrical obstacle boundaries. |
| `training_convergence_curves.png` | Episodic return progression with moving averages and confidence bounds. | Demonstrates stable asymptotic convergence within 80 episodes. |
| `aerodynamic_power_curve.png` | Zeng et al. mechanical power curve breakdown (induced, profile, parasite). | Theoretical curve validating $168.5\text{ W}$ hover power and $126.5\text{ W}$ cruise minimum. |

---

## 6. Mathematical Foundations & Defense Documentation

For detailed mathematical proofs, parameter tables, and viva defense preparation, refer to the guides in `docs/`:

* **[Mathematical & Algorithmic Code Guide](docs/EQUATIONS_AND_CODE_GUIDE.md):** Complete line-by-line mapping between mathematical equations and code implementations, physical constant derivation, reward formulation, and expected academic review defense questions with detailed answers.
* **[Project Architecture Walkthrough](docs/PROJECT_WALKTHROUGH.md):** High-level Mermaid sequence flows, simulation step lifecycles, and policy execution diagrams.

---

## 7. Citation & References

```bibtex
@article{zeng2019energy,
  title={Energy-efficient UAV communication with trajectory optimization},
  author={Zeng, Yong and Zhang, Rui and Lim, Teng Joon},
  journal={IEEE Transactions on Wireless Communications},
  volume={16},
  number={6},
  pages={3747--3760},
  year={2017}
}

@article{yu2022surprising,
  title={The surprising effectiveness of PPO in cooperative multi-agent games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24611--24624},
  year={2022}
}
```
