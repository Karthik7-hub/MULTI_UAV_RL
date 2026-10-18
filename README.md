# Research_V2: Deep Reinforcement Learning for Joint Task Offloading and Energy-Efficient Path Planning in Multi-UAV Edge Computing

A clean, modular, and scientifically rigorous research codebase for Multi-Agent Reinforcement Learning (MAPPO, CPPO, IPPO, MADDPG) and Classical Baselines (3D A* + Multi-Objective MEC) for joint continuous 3D UAV navigation and discrete computational task offloading.

---

## 1. Directory Structure

```
research_v2/
├── config.py                 # Central system configuration & physical constants
├── env/                      # Modular Environment & Physics
│   ├── __init__.py
│   ├── aerodynamics.py       # Zeng et al. (2019) rotary-wing aerodynamic power model
│   ├── comms_compute.py      # Shannon RF rate, OFDMA bandwidth sharing, computation queues & MEC contention
│   ├── safety_filter.py      # Analytical kinematic projection filter for obstacle avoidance & peer separation
│   └── multi_uav_env.py      # Gymnasium environment with station safety verification & decomposed rewards
├── models/                   # Algorithms, Policies & Baselines
│   ├── __init__.py
│   ├── mappo.py              # Joint MAPPO (CTDE with dual Categorical/Gaussian heads & Centralized Critic)
│   ├── cppo.py               # Centralized PPO (CTCE upper-bound centralized coordination baseline)
│   ├── ippo.py               # Independent PPO (DTDE decentralized local critic baseline)
│   ├── maddpg.py             # Multi-Agent DDPG (Off-policy deterministic actor + Centralized Q-Critic)
│   ├── astar.py              # 3D A* Path Planner (Waypoint obstacle bypass)
│   └── baselines.py          # Classical MEC baselines (Energy-Deadline Greedy, Lowest-Latency, Nearest, Local, Edge, Random)
├── utils/                    # Utilities & Device Management
│   ├── __init__.py
│   └── device.py             # Unified DeviceManager (auto, cuda, cpu) with diagnostics and verification
├── plots/                    # Publication Plotting Suite
│   ├── __init__.py
│   └── plot_curves.py        # Generates publication curves, Pareto frontiers, 3D trajectories, and bar breakdowns
├── train.py                  # Training pipeline with validation scenarios, throughput metrics, and ablation modes
├── evaluate.py               # Multi-seed held-out evaluation & scientific benchmarking across 9 methods
├── plot_results.py           # Standalone publication figure regenerator (no retraining required)
├── main.py                   # Master CLI entry point (train / evaluate / plot / ablation / all)
├── tests/
│   └── test_framework.py     # 7-part automated verification test suite
├── results/                  # Generated checkpoints, CSV records, JSON configs, and publication figures
│   ├── checkpoints/          # Best validation & final model weights (*.pt)
│   ├── figures/              # Publication figures (.png at 300 DPI)
│   ├── benchmark_summary.csv # Multi-seed aggregate benchmark table (Mean ± Std)
│   ├── offloading_distribution.csv # % Local vs % Peer vs % Edge breakdown
│   ├── ablation_summary.csv  # Component ablation comparison table
│   └── benchmark_detailed.csv# Per-seed granular episode records
└── README.md
```

---

## 2. Core Physics & Theoretical Formulation

1. **Rotary-Wing Aerodynamics (Zeng et al., 2019)**:
   $$P(v) = P_0 \left(1 + \frac{3v^2}{U_{tip}^2}\right) + P_i \left(\sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2}\right)^{1/2} + \frac{1}{2} d_0 \rho s A v^3$$
   Hover power is $168.49\text{ W}$, and forward cruising power drops to $\approx 126.5\text{ W}$ at optimal cruise speed $v \approx 10.3\text{ m/s}$ due to translational lift. Anti-loitering damping enforces position holding and prevents high-speed orbiting on station.

2. **Wireless Communication & Multi-User Contention**:
   * **Shannon Channel Rate with OFDMA Bandwidth Sharing**:
     $$R_j(d_j) = \frac{B}{K_{tx}} \log_2\left(1 + \frac{P_{tx} \beta_0 d_j^{-2}}{\sigma^2}\right)$$
     where $K_{tx}$ is the number of concurrent transmitting nodes.
   * **MEC Computation Contention**:
     * Ground Edge Server CPU capacity ($20\text{ GHz}$) is dynamically sliced across concurrent offloaders: $f_{edge\_eff} = f_{edge} / K_{edge}$.
     * Peer UAVs share CPU capacity ($f_{peer} / 2$) if simultaneously computing local tasks.
     * Dynamic frequency scaling energy: $E_{comp} = \kappa f^2 C$.

3. **In-Between Obstacles & Kinematic Safety Filter**:
   * 4 Cylindrical obstacles situated in-between UAV deployment ring and mission stations.
   * Analytical projection filter projects proposed continuous displacements to obstacle tangent surfaces and guarantees minimum separation $d_{safe} = 8.0\text{ m}$ between peer UAVs with non-zero fallback separation vectors.

4. **Multi-Agent Reinforcement Learning Hierarchy**:
   * **Centralized PPO (CPPO, CTCE)**: Single centralized actor takes joint state and commands fleet actions.
   * **Proposed Joint MAPPO (CTDE)**: Decentralized dual-head actors (Categorical offloading + Gaussian 3D navigation) with Centralized Critic $V(S)$ trained via GAE.
   * **Independent PPO (IPPO, DTDE)**: Fully decentralized actor and local critic $V_j(o_j)$.
   * **Multi-Agent DDPG (MADDPG, CTDE)**: Off-policy continuous actor with Gumbel-Softmax discrete relaxation and Centralized Q-Critic.

5. **Classical & Heuristic Baselines**:
   * **3D A\* + Energy- and Deadline-Aware Greedy MEC**: Strong classical benchmark optimizing $J = \text{Delay} + \beta E + \text{Penalty}_{deadline}$.
   * **3D A\* + Lowest-Latency MEC**: Greedily minimizes transmission plus execution delay.
   * **3D A\* + Local-Only Execution**: Lower bound for offloading benefits.
   * **3D A\* + Edge-Only Execution**: Upper bound for edge reliance.
   * **3D A\* + Random Feasible MEC**: Statistical lower bound baseline.

---

## 3. Quick Start & Execution Commands

### A. Run Verification Tests First
```powershell
python tests/test_framework.py
```

### B. Hardware Device Options
All scripts support `--device auto` (default: auto-detects CUDA, falls back cleanly to CPU), `--device cuda`, or `--device cpu`.

```powershell
# Auto-detect CUDA if available, fallback to CPU
python main.py --mode all --device auto

# Explicitly target CUDA (raises informative error if CUDA is missing)
python main.py --mode train --model mappo --device cuda

# Force CPU execution
python main.py --mode train --model mappo --device cpu
```

### C. Training Models
```powershell
# Train Proposed Joint MAPPO
python train.py --model mappo --episodes 80 --seed 42 --device auto

# Train Centralized PPO (CPPO)
python train.py --model cppo --episodes 80 --seed 42 --device auto

# Train IPPO Baseline
python train.py --model ippo --episodes 80 --seed 42 --device auto

# Train MADDPG Baseline
python train.py --model maddpg --episodes 80 --seed 42 --device auto

# Train all models sequentially
python train.py --model all --episodes 80 --seed 42 --device auto
```

### D. Component Ablation Experiments
```powershell
# Train MAPPO without energy reward shaping
python train.py --model ablation --ablation no_energy --episodes 80 --device auto

# Train MAPPO without safety penalty
python train.py --model ablation --ablation no_safety --episodes 80 --device auto

# Train MAPPO navigation with fixed greedy offloading heuristic
python train.py --model ablation --ablation fixed_offload --episodes 80 --device auto
```

### E. Multi-Seed Benchmark Evaluation
```powershell
# Evaluates all 9 learning & classical algorithms over 5 held-out test seeds
python evaluate.py --device auto --num_seeds 5
```

### F. Regenerate Figures from Saved Data (Without Retraining)
```powershell
python plot_results.py
```

---

## 4. Generated Publication-Grade Figures

All figures are automatically exported at 300 DPI to `results/figures/`:
1. `trajectory_diagnostic_comparison.png`: 4-panel diagnostic comparison of 2D trajectories, station distances, flight velocities, and fleet cumulative energy over 100 seconds.
2. `multimodel_performance_curves.png`: 4-panel comparative evaluation comparing MAPPO, IPPO, MADDPG, and 3D A*.
3. `mappo_vs_astar_diagnostic.png`: Deep-dive pairwise comparison of Proposed MAPPO vs 3D A* Baseline.
4. `mappo_vs_ippo_ablation.png`: CTDE vs DTDE ablation evaluating coordination and inter-UAV collision avoidance.
5. `mappo_vs_maddpg_comparison.png`: Policy architecture comparison (On-Policy MAPPO vs Off-Policy MADDPG).
6. `offloading_distribution_breakdown.png`: Stacked horizontal bar chart showing % Local, % Peer, and % Edge decisions.
7. `pareto_latency_energy_tradeoff.png`: Multi-objective Pareto frontier showing Latency (s) vs Fleet Energy (kJ).
8. `ablation_study_comparison.png`: Grouped comparative analysis of reward and modular ablations.
9. `3d_flight_trajectories.png`: 3D airspace visualization with cylindrical obstacle geometries and flight paths.
10. `training_convergence_curves.png`: Training returns, moving averages, and confidence bounds.
11. `aerodynamic_power_curve.png`: Zeng et al. mechanical power curve breakdown (induced, profile, parasite).
