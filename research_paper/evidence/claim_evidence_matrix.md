# Scientific Claim-to-Evidence Traceability Matrix

**Document Version:** 1.0 (Phase 2 Post-Forensic Audit Standard)  
**Target Repository:** `research_v2`  
**Auditor:** Senior Academic Research Specialist & Code Integrity Auditor  
**Date:** November 2026  

---

## 1. Traceability Matrix

The table below catalogs every scientific claim made in the manuscript, mapping each to its exact implementation code line, checkpoint weights, raw evaluation logs, and verification status:

| Claim ID | Research Claim / Hypothesis | Implementation Source | Checkpoint / Artifact Source | Evaluation Log Source | Verification Status | Reportable in Paper? |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **C1** | Quadrotor propulsion power follows the non-linear aerodynamics model of Zeng et al. (2019). | [`env/aerodynamics.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/aerodynamics.py#L45-L75) | Validated via `tests/test_framework.py` | `results/publication_figures_v3/fig10_aerodynamic_power.png` | **Verified** | **YES** |
| **C2** | Continuous line-segment/cylinder intersection filter prevents obstacle penetration between discrete steps. | [`env/safety_filter.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/safety_filter.py#L120-L180) | Parameterized in `config.py` ($r_{\text{eff}}=12.0\,\text{m}$) | `results/benchmark_summary.csv` (`Collisions = 0.0 ± 0.0`) | **Verified** | **YES** |
| **C3** | Proposed Pure MAPPO (8.4k updates, zero prior) fails to achieve complete fleet navigation in 3D urban canyons. | [`models/mappo.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/mappo.py) | `results/checkpoints/best_mappo_pure_agent.pt` | `results/benchmark_summary.csv` (UAV: 25.0%, Fleet: 0.0%, Goal Error: 28.76m) | **Verified (Negative Finding)** | **YES (Mandatory Disclosure)** |
| **C4** | Adding causal residual RL actions ($k_{\text{rl}}=0.5$) to 3D A* navigation destabilizes terminal goal holding. | [`models/astar.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/astar.py#L280-L340) | Evaluated across seeds 101–105 | `results/benchmark_summary.csv` (UAV: 50.0%, Fleet: 0.0%, Interventions: 198.0) | **Verified (Negative Finding)** | **YES (Mandatory Disclosure)** |
| **C5** | Hierarchical 3D A* with Pure Pursuit ($k_{\text{rl}}=0.0$) + MAPPO Offload achieves 100% mission arrival with zero collisions. | [`models/astar.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/astar.py), [`evaluate.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/evaluate.py) | Checkpoints loaded in `evaluate.py` | `results/benchmark_summary.csv` (UAV: 100.0%, Fleet: 100.0%, Interv: 0.0, Error: 0.41m) | **Verified** | **YES** |
| **C6** | In default baseline configuration, task completion rate is saturated at ~100% before training. | [`config.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/config.py#L187-L212), [`env/comms_compute.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/comms_compute.py) | Mathematical proof: $t_{\text{comp}} \le 1.0\,\text{s} \le T_{\text{max}}$ | Empirical validation script run on seeds 42, 101 | **Verified (Ceiling Effect)** | **YES (Documented as Baseline Pitfall)** |
| **C7** | Under sub-second delay-critical stress ($f_{\text{local}}=0.8\,\text{GHz}, T_{\text{max}} \le 1.0\,\text{s}$), local execution collapses to 0.0%. | Tested via `env/comms_compute.py` | Configured parameters | `results/algorithm_workload_stress_comparison.csv` (Local: 0.0% completion) | **Verified** | **YES** |
| **C8** | Proposed MARL coordination achieves ~89% completion under sub-second stress vs 18% for random offloading. | Tested via `evaluate.py` stress harness | Validated model checkpoints | `results/algorithm_workload_stress_comparison.csv` | **Verified** | **YES** |
| **C9** | 1000-episode training runs improve MAPPO return and offload efficiency over 250 episodes. | `archive/` 1000ep CSV files | `training_logs_mappo_pure_1000ep.csv` | Forensic audit proved files have 6 dummy cols, return dropped to -75 | **Invalid / Fabricated** | **NO (Explicitly Rejected)** |
| **C10** | Centralized Critic utilizes a 52-dimensional global state vector ($13 \times 4$). | [`env/observation_builder.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/observation_builder.py#L58), [`env/multi_uav_env.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/multi_uav_env.py#L646) | Critic network parameter shape `[128, 52]` | Verified via PyTorch `state_dict` inspection in `evaluate.py` | **Verified** | **YES** |
| **C11** | APF-Only Controller achieves 100% fleet station arrival in benchmark corridor. | [`models/apf.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/apf.py) | Evaluated across seeds 101–105 | `results/benchmark_summary.csv` (UAV: 0.0%, Fleet: 0.0%, Error: 34.52m, Interv: 397.0) | **Invalid (APF Fails)** | **NO (Corrected: APF Fails with 0% Arrival)** |
| **C12** | Centralized PPO (CPPO, CTCE) solves multi-UAV navigation without decentralized actors. | [`models/cppo.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/cppo.py) | `results/checkpoints/best_cppo_agent.pt` | `results/benchmark_summary.csv` (UAV: 0.0%, Goal Error: 70.52m, Interv: 51.0) | **Verified (Negative Finding)** | **YES (Report CPPO Failure Honestly)** |

---

## 2. Integrity and Discrepancy Audits

### 2.1 The Reward Exploitation & Continuous Cruising Issue
- **Audit Findings:** In early versions, policies circled around goal stations at maximum speed ($15\,\text{m/s}$) to continuously harvest proximity rewards.
- **Resolution in Code:** MultiUAVEnv implemented an anti-loitering damping reward ($R_{\text{loiter}} = -0.5 \cdot (\|\mathbf{v}\| / v_{\max})$ for $\|\mathbf{v}\| > 1.5\,\text{m/s}$ when $d \le 5.0\,\text{m}$) and station-holding bonus ($R_{\text{station}} = +1.0$ for $\|\mathbf{v}\| \le 1.5\,\text{m/s}$). Disentangled energy metrics separate Navigation Energy ($E_{\text{nav}}$) from Post-Arrival Service Energy ($E_{\text{serv}}$).

### 2.2 The APF Dependence in Residual MAPPO
- **Audit Findings:** Trajectory plots previously labeled as "Hierarchical MAPPO" were running with residual weight $k_{\text{rl}} = 0.0$, meaning spatial flight was 100% classical A* pure pursuit.
- **Resolution in Code:** The paper explicitly describes this controller as **Hierarchical 3D A* + Pure Pursuit Tracking with MAPPO Task Offloading**. When $k_{\text{rl}} > 0.0$ was tested, the RL actions degraded navigation (50% success), which is reported as a core scientific finding.

### 2.3 Task Accounting Parity
- **Audit Findings:** In default configuration, task queues drain completely every second ($1.5\,\text{Gcycles/s}$ drain vs $0.8\,\text{tasks/s}$ arrival), causing zero backlog.
- **Resolution in Code:** Dual-regime benchmarking reports both the nominal low-stress regime ($98\text{--}100\%$) and the literature-backed stress regime ($0\text{--}89\%$), preventing false claims of algorithmic superiority.

### 2.4 The IPPO vs. MAPPO Multi-Objective Tradeoff
- **Audit Findings:** In Table II, IPPO exhibits higher individual UAV arrival ($50.0\%$) than Pure MAPPO ($25.0\%$), leading to potential misinterpretation that IPPO is superior.
- **Resolution in Code:** A full-spectrum evaluation reveals that IPPO operates via selfish, uncoordinated DTDE navigation, causing $170.0$ safety filter interventions ($3.3\times$ higher than MAPPO's $51.0$), $+57.8\%$ higher task latency ($279.7\,\text{s}$ vs $177.3\,\text{s}$), and higher task drop rates ($93.8\%$ vs $99.4\%$). Furthermore, both IPPO and Pure MAPPO achieve $0.0\%$ fleet mission success, whereas the proposed Hierarchical 3D A* + MAPPO achieves $100.0\%$ fleet arrival and $100.0\%$ task completion.

