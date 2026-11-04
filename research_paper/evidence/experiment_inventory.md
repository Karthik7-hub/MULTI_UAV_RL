# Experiment Inventory & Evaluation Suite Catalog

**Document Version:** 1.0  
**Target Repository:** `research_v2`  
**Date:** November 2026  

---

## 1. Benchmarked Algorithms & Policy Modes

The evaluation suite encompasses 12 distinct navigation and offloading algorithms evaluated under identical 3D physical constraints:

| ID | Formal Academic Designation | Movement Controller | Offloading Policy | RL Architecture | Primary Checkpoint / Source |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **M1** | Classical 3D A\* + Edge MEC | 3D A\* Grid Planner + Pure Pursuit | Edge-Only (100% BS) | Zero RL | Baseline heuristic in `models/astar.py` |
| **M2** | Classical 3D A\* + Pure Pursuit Baseline | 3D A\* Grid Planner + Pure Pursuit | Local-Only (100% Local CPU) | Zero RL | Baseline heuristic in `models/astar.py` |
| **M3** | Hierarchical 3D A\* + MAPPO ($k_{\text{rl}}=0.0$) | 3D A\* Waypoint Tracking | Dual-Head MAPPO Actor | CTDE (22-dim obs, 52-dim critic) | `results/checkpoints/best_mappo_pure_agent.pt` |
| **M4** | Hierarchical 3D A\* + MAPPO Residual ($k_{\text{rl}}=0.2$) | 3D A\* + 20% RL Residual Velocity | Dual-Head MAPPO Actor | CTDE Residual Architecture | `results/checkpoints/best_mappo_residual_agent.pt` |
| **M5** | Hierarchical 3D A\* + MAPPO Residual ($k_{\text{rl}}=0.5$) | 3D A\* + 50% RL Residual Velocity | Dual-Head MAPPO Actor | CTDE Residual Architecture | `results/checkpoints/best_mappo_residual_agent.pt` |
| **M6** | APF-Only Controller (Gain 2.0) | Artificial Potential Field | Uniform Random Feasible | Zero RL | Baseline heuristic in `models/apf.py` |
| **M7** | APF-Assisted Residual MAPPO (Terminal Damped) | APF Force + RL Residual Action | Dual-Head MAPPO Actor | CTDE Residual with Damping | `results/checkpoints/best_mappo_residual_agent.pt` |
| **M8** | APF-Assisted Residual MAPPO (Undamped) | APF Force + RL Residual Action | Dual-Head MAPPO Actor | CTDE Residual Undamped | Historical baseline in `models/apf.py` |
| **M9** | Proposed Pure MAPPO (8.4k updates, Zero Prior) | End-to-End Neural Velocity Head | Dual-Head MAPPO Actor | CTDE Pure MARL (Zero geometric prior) | `results/checkpoints/best_mappo_pure_agent.pt` |
| **M10** | Independent PPO (IPPO) | End-to-End Neural Velocity Head | Dual-Head IPPO Actor | DTDE (Decentralized Critic, 22-dim obs) | `results/checkpoints/best_ippo_pure_agent.pt` |
| **M11** | Multi-Agent DDPG (MADDPG) | Continuous Deterministic Actor | Gumbel-Softmax Offload Head | Centralized Continuous Q-Critic | `results/checkpoints/best_maddpg_pure_agent.pt` |
| **M12** | Centralized PPO (CPPO) | Centralized Fleet Velocity Head | Centralized Multi-Task Head | CTCE (Fully Centralized 52-dim State) | `results/checkpoints/best_cppo_agent.pt` |

---

## 2. Airspace Environments & Presets

All environments represent right-handed 3D Cartesian coordinates $\mathcal{W} = [0, 200] \times [0, 200] \times [0, 100]\,\text{m}$:

| Preset Name | Cylindrical Obstacles Count | Cylinder Dimensions $(c_x, c_y, r, h)$ | Physical Operational Context | Tested In Benchmark? |
| :--- | :---: | :--- | :--- | :---: |
| **`benchmark`** | 4 | $(142.5, 100.0, 10.0, 80.0)$, $(100.0, 142.5, 10.0, 85.0)$, $(57.5, 100.0, 10.0, 80.0)$, $(100.0, 57.5, 10.0, 75.0)$ | Symmetric 4-corridor canyon with central mission ring | **YES (Primary Benchmark)** |
| **`metropolis`** | 20 | Dense urban core with 20 high-rise towers ($h \in [50, 92]\,\text{m}, r \in [8, 12]\,\text{m}$) | Dense urban high-density airspace | **YES (Generalization Test)** |
| **`chokepoint`** | 8 | Paired bottleneck cylinders with narrow $14\,\text{m}$ flight corridors | Narrow passage avoidance & queue contention | **YES (Ablation Test)** |
| **`canyon`** | 3 | Massive central cylinders ($r=24.0\,\text{m}, h=90.0\,\text{m}$) | Deep urban canyon obstacle blocking direct LoS | **YES (Stress Test)** |

---

## 3. Seed Protocols

1. **Training Runs:**
   - Evaluated across seeds: `42`, `101`, `202`.
   - Rollouts per update: 100 steps $\times$ 4 UAVs = 400 transitions.
   - PPO updates: 5 epochs per rollout, mini-batch size 64.
2. **Validation Scenarios:**
   - Evaluated periodically during training on held-out seeds: `1001`, `1002`, `1003`, `1004`.
   - Scenarios alternate between canonical `benchmark` and randomized station configurations.
3. **Multi-Seed Benchmark Protocol:**
   - 5 strictly held-out evaluation seeds: `101`, `102`, `103`, `104`, `105`.
   - Metrics reported as: $\mu \pm \sigma$ across all 5 seeds.

---

## 4. Hardware and Computational Profile

- **Operating System:** Microsoft Windows 11 Enterprise (x86_64)
- **Python Environment:** Python 3.14 (Virtual Environment `.\.venv`)
- **PyTorch Device:** CPU Device Manager (`torch.device("cpu")`), fully vectorized NumPy kinematics.
- **Evaluation Throughput:** 12 algorithms $\times$ 5 seeds $\times$ 100 steps = 6,000 environment steps evaluated in $45.2\,\text{seconds}$ ($\approx 132.7\,\text{steps/s}$).

