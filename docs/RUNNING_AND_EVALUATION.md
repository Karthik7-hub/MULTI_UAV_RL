# Running and Evaluation Guide

This guide provides the verified commands for training, benchmarking, visualizing, and testing the Multi-UAV 3D simulation framework.

---

## 1. Environment and Prerequisites

Ensure you are executing within the project virtual environment or system Python where PyTorch is installed:
```powershell
cd c:\Users\vkart\Music\WORK\SEM\SEM_7\Minor_project\research_project\research_v2
python -c "import torch; print('PyTorch Version:', torch.__version__)"
```

> [!NOTE]
> When executing `pytest`, always invoke it via `python -m pytest tests/test_framework.py` to ensure that pytest uses the active Python interpreter containing PyTorch.

---

## 2. Automated Test Suite

To verify that all 12 core simulation units (aerodynamics, path loss, safety filter, A*, MAPPO actor/critic, and environment step) are working:

```powershell
python -m pytest tests/test_framework.py
```

Expected output:
```text
tests\test_framework.py ............                                     [100%]
============================= 12 passed in 12.25s =============================
```

---

## 3. Training RL Models (`train.py`)

The training pipeline trains the multi-agent reinforcement learning models with curriculum learning, held-out validation checks, and automatic checkpoint saving.

### Train All 4 Models Sequentially (Recommended)
```powershell
python -u train.py --model all --episodes 250 --curriculum
```
This trains:
1. `Proposed Pure MAPPO` (`models/mappo.py`, pure variant)
2. `Independent PPO (IPPO)` (`models/ippo.py`)
3. `Multi-Agent DDPG (MADDPG)` (`models/maddpg.py`)
4. `Centralized PPO (CPPO)` (`models/cppo.py`)

### Train an Individual Model
```powershell
# Train Proposed Pure MAPPO
python -u train.py --model mappo --variant pure --episodes 250 --curriculum

# Train Residual MAPPO (APF-guided)
python -u train.py --model mappo --variant residual --episodes 250 --curriculum

# Train Independent PPO (IPPO)
python -u train.py --model ippo --episodes 250 --curriculum

# Train Multi-Agent DDPG (MADDPG)
python -u train.py --model maddpg --episodes 250 --curriculum

# Train Centralized PPO (CPPO)
python -u train.py --model cppo --episodes 250 --curriculum
```

### Outputs Generated:
- Checkpoints saved in `results/checkpoints/`:
  - `best_mappo_pure_agent.pt`, `final_mappo_pure_agent.pt`
  - `best_ippo_pure_agent.pt`, `final_ippo_pure_agent.pt`
  - `best_maddpg_pure_agent.pt`, `final_maddpg_pure_agent.pt`
  - `best_cppo_agent.pt`, `final_cppo_agent.pt`
- Training CSV logs saved in `results/`:
  - `training_logs_mappo_pure.csv`
  - `training_logs_ippo_pure.csv`
  - `training_logs_maddpg_pure.csv`
  - `training_logs_cppo.csv`

---

## 4. Multi-Seed Scientific Benchmarking (`evaluate.py`)

Evaluation evaluates the trained models against classical path planners across **5 held-out test seeds** (101 to 105).

```powershell
python evaluate.py
```

### What Happens During Evaluation:
1. Evaluates all 9 algorithms:
   - APF-Assisted Residual MAPPO (Proposed)
   - Proposed Pure MAPPO (No APF)
   - Multi-Agent DDPG (MADDPG)
   - Centralized PPO (CPPO)
   - Independent PPO (IPPO)
   - 3D A* Baseline (Manhattan Grid Search)
   - 3D SAPF (Sequential Adaptive Potential Field)
   - 3D RRT* (Optimal Random Tree)
   - Heuristic Baselines (Local-Only, Edge-Only, Random)
2. Records 10 Hz physical flight trajectories to `results/trajectory_records.csv`.
3. Computes mean and standard deviation across seeds for:
   - `Success Rate (%)`: Fraction of seeds where all UAVs reached their designated station within tolerance.
   - `Arrival Time (s)`: Mean time taken to arrive at the station.
   - `Nav Energy (kJ)`: Mechanical propulsion energy consumed before reaching station.
   - `Service Energy (kJ)`: Mechanical energy consumed while hovering at the station.
   - `Total Energy (kJ)`: Total energy consumed across propulsion, RF communications, and CPU computing.
   - `Tasks Done (%)`: Percentage of generated Poisson computational tasks completed before deadline.
   - `Delay (s)`: Cumulative end-to-end task latency.
   - `Path Length (m)`: Total physical 3D distance flown.
   - `Interventions`: Number of safety filter interventions triggered.
   - `Collisions`: Genuine physical obstacle or inter-UAV collisions.
4. Outputs saved:
   - `results/benchmark_summary.csv`
   - `results/benchmark_detailed.csv`
   - `results/offloading_distribution.csv`

---

## 5. Generating Publication Figures (`plots/`)

The framework generates peer-reviewed journal quality figures adhering to IEEE Transactions standards:

```powershell
# 1. Generate 3D & 2D Spatial Trajectory Figures
python plots/render_publication_figures.py

# 2. Generate 2D Performance & Convergence Curves
python plot_results.py
```

### Generated Figures in `results/figures/`:
1. `figure_b_multimodel_trajectory_comparison.png` & `.pdf`:
   - Dual-panel multi-model trajectory comparison (Proposed Hierarchical A*+MAPPO vs 3D A* vs 3D SAPF vs 3D RRT*).
   - Panel (a): 3D isometric canyon view with slender semi-transparent buildings and unobstructed corridor perspective.
   - Panel (b): 2D planar projection showing obstacle footprints and dashed safety clearance rings.
2. `figure5_3d_metropolis_flight_trajectories.png` & `.pdf`:
   - 3D isometric perspective of the 4-UAV fleet traversing the 20-skyscraper Metropolis canyon.
3. `pareto_latency_energy_tradeoff.png` & `.pdf`:
   - Multi-Objective Pareto Frontier plot (Task Latency vs. Fleet Energy) with empirical Pareto frontier $\mathcal{P}^*$.
4. `marl_training_convergence_comparison.png` & `.pdf`:
   - 4-panel convergence comparison: Episode Return, Task Completion Rate, Total Energy, and Held-Out Validation Return over 250 training episodes.

---

## 6. Real-Time 3D Web Dashboard (`simulator_server.py`)

To inspect UAV flight trajectories, obstacle geometries, and offloading in real time through an interactive 3D browser dashboard:

```powershell
python simulator_server.py
```

Open your browser to:
```text
http://127.0.0.1:5000/
```

Features:
- Three.js 3D rendering with orbit, zoom, and pan controls.
- Dynamic obstacle presets: `benchmark`, `metropolis`, `chokepoint`, `canyon`.
- Live playback of trajectory waypoints and offloading decisions.
- Real-time telemetry panels showing airspeed, energy consumption, and task completion.
