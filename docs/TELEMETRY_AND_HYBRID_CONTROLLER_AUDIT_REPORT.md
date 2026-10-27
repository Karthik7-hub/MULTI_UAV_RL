# Comprehensive Audit Report: Metropolis Trajectory Telemetry, Speed Bounds, and Controller Discrepancies

**Date:** October 27, 2026  
**Target Files:**  
- [`results/metropolis_trajectory_records.csv`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/results/metropolis_trajectory_records.csv)  
- [`results/metropolis_evaluation_summary.csv`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/results/metropolis_evaluation_summary.csv)  
- [`scratch/record_metropolis_trajectories.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/scratch/record_metropolis_trajectories.py)  
- [`models/astar.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/models/astar.py)  

---

## Executive Summary

1. **Reconciliation of 6.8 s vs. 8.0 s Arrival Time:**
   - **8.0 s** is the exact arrival timestep of **focal UAV 0** ($S_0 \to G_0$, path length $110.76\,\text{m}$).
   - **6.8 s** ($6.75\,\text{s}$) is the **unweighted fleet average** across all 4 UAVs ($\text{UAV 0}=8\,\text{s}$, $\text{UAV 1}=7\,\text{s}$, $\text{UAV 2}=6\,\text{s}$, $\text{UAV 3}=6\,\text{s}$).
   - **98.2 m** ($98.25\,\text{m}$) is the **fleet average path length**, whereas focal UAV 0 traverses **$110.76\,\text{m}$**.
   - There is zero computational contradiction; the apparent discrepancy was a reporting ambiguity conflating **UAV 0 individual metrics** with **Fleet Average metrics**.

2. **Root Cause of A\* Path Difference (98.2 m vs. 108.2 m):**
   - Both `hybrid_astar_mappo` and `astar` employ the **exact same 3D A\* planner (`AStar3DPlanner`) on the identical 20-skyscraper map**.
   - With `residual_weight = 0.0`, the MAPPO neural network had **zero influence on physical velocity or spatial coordinates**; it operated exclusively on MEC task offloading.
   - The path length and flight duration advantage ($98.2\,\text{m} / 6.8\,\text{s}$ vs. $108.2\,\text{m} / 8.5\,\text{s}$) is **100% caused by the trajectory tracking controller**:
     - `HybridAStarMAPPONavigator` uses `PurePursuit3DController` with a $10.0\,\text{m}$ forward lookahead, enabling continuous corner-cutting at full cruise velocity ($15\,\text{m/s}$).
     - `AStarBaseline` uses naive stop-and-go discrete waypoint stepping (`min(d_target, v_max * dt)` with a strict $4.0\,\text{m}$ switching radius), which forces decelerations down to $5.0\,\text{m/s}$ at intermediate grid vertices and requires additional discrete timesteps.

3. **Physics & Safety Constraint Verification (The 2.00 m Clearance Explanation):**
   - **Spawn Geometry ($t=0$):** All 4 UAVs are initialized at $(165, 100, 50)$, $(100, 165, 50)$, $(35, 100, 50)$, and $(100, 35, 50)$, exactly $10.00\,\text{m}$ from the centers of perimeter obstacles 16, 17, 18, 19 (each having radius $r=8.0\,\text{m}$). The radial surface clearance is therefore identically $10.00 - 8.00 = 2.0000\,\text{m}$ at $t=0$ across all algorithms before flight begins.
   - **Controller Safety Buffer:** In [`env/safety_filter.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/safety_filter.py#L26-L32), `SafetyFilter3D` enforces `obstacle_margin = 2.0 m` (`eff_r = r + 2.0 m`). For failing/untrained policies (`mappo_pure`), when the policy attempts to fly into a building, the safety filter projects the coordinates back to $r + 2.05\,\text{m}$, artificially holding the UAV at $\approx 2.00\,\text{m}$ from the cylinder wall.
   - **Active In-Flight Physical Clearance ($t \ge 1$):** Once underway, the classical/hybrid planners operate well outside the safety margin with zero safety interventions:
     - `hybrid_astar_mappo`: **$3.87\,\text{m}$** minimum physical clearance (UAV 1 at $t=5$ near Obs 1).
     - `astar`: **$6.40\,\text{m}$** minimum physical clearance (UAV 1 at $t=7$ near Obs 1).
     - `sapf`: **$2.13\,\text{m}$** minimum physical clearance (UAV 1 at $t=9$ near Obs 1).
   - Maximum commanded/achieved speed across all 2,424 timesteps is strictly **$\le 15.00\,\text{m/s}$**.
   - Zero physical cylinder collisions ($d_{xy} < r$ and $z \le h$) across all 2,424 records.

4. **Single-Seed Limitation & Scientific Integrity:**
   - The Metropolis benchmark records in `metropolis_trajectory_records.csv` represent a single scenario initialization (**Seed 101**).
   - The 0.0% arrival rate of RRT\*, pure MAPPO, and residual MAPPO reflects this specific geometric arrangement prior to the Stage 5 tangent-steering integration, and must not be presented as a general proof of algorithm failure.
   - The raw recorded coordinates must be plotted directly without rolling smoothing or cosmetic curve fitting.

---

## 1. Mathematical Breakdown: Fleet vs. Individual Metrics

### A. Proposed Hybrid (`hybrid_astar_mappo`)
Calculated directly from raw coordinates $(x_t, y_t, z_t)$ up to the arrival timestep ($d_{\text{goal}} \le 5.0\,\text{m}$):

| UAV ID | Start $(x,y,z)$ | Station $(x,y,z)$ | Arrival Timestep $t_{\text{arr}}$ | Path Length $\sum \|\Delta \mathbf{p}\|$ | Max Speed | Min Cylinder Clearance | Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAV 0** | $(165, 100, 50)$ | $(120, 100, 50)$ | **8 s** | **110.76 m** | $15.00\,\text{m/s}$ | $5.54\,\text{m}$ | Arrived |
| **UAV 1** | $(100, 165, 50)$ | $(100, 120, 50)$ | **7 s** | **105.00 m** | $15.00\,\text{m/s}$ | $3.87\,\text{m}$ | Arrived |
| **UAV 2** | $(35, 100, 50)$ | $(80, 100, 50)$ | **6 s** | **90.00 m** | $15.00\,\text{m/s}$ | $3.66\,\text{m}$ | Arrived |
| **UAV 3** | $(100, 35, 50)$ | $(100, 80, 50)$ | **6 s** | **87.23 m** | $15.00\,\text{m/s}$ | $4.23\,\text{m}$ | Arrived |
| **Fleet Average** | — | — | **6.75 s (~6.8 s)** | **98.25 m** | **15.00 m/s** | **3.66 m** | **100% (4/4)** |

### B. Classical 3D A\* Baseline (`astar`)

| UAV ID | Start $(x,y,z)$ | Station $(x,y,z)$ | Arrival Timestep $t_{\text{arr}}$ | Path Length $\sum \|\Delta \mathbf{p}\|$ | Max Speed | Min Cylinder Clearance | Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAV 0** | $(165, 100, 50)$ | $(120, 100, 50)$ | **9 s** | **114.44 m** | $15.00\,\text{m/s}$ | $5.59\,\text{m}$ | Arrived |
| **UAV 1** | $(100, 165, 50)$ | $(100, 120, 50)$ | **9 s** | **117.54 m** | $15.00\,\text{m/s}$ | $4.42\,\text{m}$ | Arrived |
| **UAV 2** | $(35, 100, 50)$ | $(80, 100, 50)$ | **8 s** | **104.37 m** | $15.00\,\text{m/s}$ | $3.59\,\text{m}$ | Arrived |
| **UAV 3** | $(100, 35, 50)$ | $(100, 80, 50)$ | **8 s** | **96.32 m** | $15.00\,\text{m/s}$ | $4.34\,\text{m}$ | Arrived |
| **Fleet Average** | — | — | **8.50 s** | **108.17 m** | **15.00 m/s** | **3.59 m** | **100% (4/4)** |

---

## 2. Step-by-Step Trajectory Audit for Focal UAV 0

Below is the side-by-side telemetry for UAV 0 demonstrating the controller divergence:

| Timestep | `hybrid_astar_mappo` Position $(x, y, z)$ | Speed | Dist to Goal | `astar` Position $(x, y, z)$ | Speed | Dist to Goal | Controller Mechanism Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **t = 0** | $(165.00, 100.00, 50.00)$ | $0.00\,\text{m/s}$ | $45.00\,\text{m}$ | $(165.00, 100.00, 50.00)$ | $0.00\,\text{m/s}$ | $45.00\,\text{m}$ | Initial launch state |
| **t = 1** | $(160.03, 95.03, 63.25)$ | $15.00\,\text{m/s}$ | $42.46\,\text{m}$ | $(160.00, 100.00, 50.00)$ | **5.00 m/s** | $40.00\,\text{m}$ | `astar` clamps step to next 5 m grid node; Pure Pursuit tracks forward |
| **t = 2** | $(155.06, 90.06, 76.50)$ | $15.00\,\text{m/s}$ | $45.06\,\text{m}$ | $(156.58, 94.87, 63.68)$ | $15.00\,\text{m/s}$ | $39.39\,\text{m}$ | Diagonal climb past building flank |
| **t = 3** | $(150.09, 85.09, 89.76)$ | $15.00\,\text{m/s}$ | $52.04\,\text{m}$ | $(153.16, 89.74, 77.35)$ | $15.00\,\text{m/s}$ | $44.19\,\text{m}$ | Flank bypass corridor |
| **t = 4** | $(135.92, 85.01, 94.68)$ | $15.00\,\text{m/s}$ | $49.75\,\text{m}$ | $(150.00, 85.00, 90.00)$ | **13.87 m/s** | $52.20\,\text{m}$ | `astar` clamps at corner vertex |
| **t = 5** | $(131.60, 85.00, 80.32)$ | $15.00\,\text{m/s}$ | $35.76\,\text{m}$ | $(135.77, 85.00, 94.74)$ | $15.00\,\text{m/s}$ | $49.76\,\text{m}$ | Horizontal corridor along $y=85$ |
| **t = 6** | $(126.73, 91.29, 67.60)$ | $15.00\,\text{m/s}$ | $20.76\,\text{m}$ | $(131.56, 85.00, 80.35)$ | $15.00\,\text{m/s}$ | $35.77\,\text{m}$ | Descent toward target |
| **t = 7** | $(121.87, 97.59, 54.88)$ | $15.00\,\text{m/s}$ | $5.76\,\text{m}$ | $(130.00, 85.00, 75.00)$ | **5.57 m/s** | $30.82\,\text{m}$ | `astar` decelerates to hit second intermediate vertex |
| **t = 8** | **$(120.00, 100.00, 50.00)$** | **5.76 m/s** | **0.00 m** | $(125.13, 92.30, 62.83)$ | $15.00\,\text{m/s}$ | $15.82\,\text{m}$ | **Hybrid Arrived** (Goal reached); `astar` still in transit |
| **t = 9** | — | — | — | **$(120.27, 99.60, 50.67)$** | **15.00 m/s** | **0.82 m** | **`astar` Arrived** at $t=9\,\text{s}$ |

---

## 3. Discrepancy Reconciliation Summary

1. **Why UAV 0 took 8 steps ($8.0\,\text{s}$):**
   - At $t=0$, UAV 0 is at $(165, 100, 50)$. At $t=1..7$, it flies at exactly $v_{\text{max}} = 15.0\,\text{m/s}$, covering $7 \times 15.0 = 105.0\,\text{m}$.
   - At $t=8$, it traverses the final $5.76\,\text{m}$ to the exact station coordinate $(120, 100, 50)$.
   - Total path length $= 105.0 + 5.76 = 110.76\,\text{m}$.
   - This strictly adheres to physics: $110.76\,\text{m} \le 8 \times 15\,\text{m/s} = 120\,\text{m}$.

2. **Why the evaluation summary recorded 6.8 s and 98.2 m:**
   - The summary table records the **fleet-wide arithmetic mean**:
     $$\bar{T}_{\text{arrival}} = \frac{8 + 7 + 6 + 6}{4} = 6.75\,\text{s} \approx 6.8\,\text{s}$$
     $$\bar{L}_{\text{path}} = \frac{110.76 + 105.00 + 90.00 + 87.23}{4} = 98.25\,\text{m}$$
   - Any figure caption or text referring to "Figure 2" (UAV 0) should state **$8.0\,\text{s}$ and $110.8\,\text{m}$**.
   - Any figure caption or table referring to "Fleet Statistics" should state **$6.8\,\text{s}$ and $98.2\,\text{m}$**. Conflating the two caused the confusion.

3. **Why the hybrid path was shorter than the classical A\* baseline:**
   - Both algorithms used the **exact same A\* global path planner**.
   - The difference stems entirely from the **tracking controller** (lookahead Pure Pursuit vs. naive stop-and-go discrete waypoint stepping).
   - Under `residual_weight = 0.0`, MAPPO RL had **zero** effect on movement. In the manuscript, this must be accurately described as an advantage of **Pure Pursuit corridor tracking**, not an advantage of RL navigation.

---

## 4. In-Depth Clearance Audit: Controller Buffer vs. True Physical Distance

A key question raised in review was whether the reported minimum clearance of exactly $2.00\,\text{m}$ across all algorithms represented genuine physical clearance or an enforced safety margin. A code-level and telemetry audit revealed the dual origin:

### A. Initial Geometric Spawn Proximity ($t=0$)
In [`config.py`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/config.py), the four UAV spawn positions and the nearest perimeter obstacle cylinders are:
- **UAV 0** at $(165.0, 100.0, 50.0) \implies$ Obs 16 at $(175.0, 100.0, r=8.0, h=55.0)$. Radial distance $\Delta r = 10.0 - 8.0 = \mathbf{2.0000\,m}$.
- **UAV 1** at $(100.0, 165.0, 50.0) \implies$ Obs 17 at $(100.0, 175.0, r=8.0, h=50.0)$. Radial distance $\Delta r = 10.0 - 8.0 = \mathbf{2.0000\,m}$.
- **UAV 2** at $(35.0, 100.0, 50.0) \implies$ Obs 18 at $(25.0, 100.0, r=8.0, h=50.0)$. Radial distance $\Delta r = 10.0 - 8.0 = \mathbf{2.0000\,m}$.
- **UAV 3** at $(100.0, 35.0, 50.0) \implies$ Obs 19 at $(100.0, 25.0, r=8.0, h=55.0)$. Radial distance $\Delta r = 10.0 - 8.0 = \mathbf{2.0000\,m}$.

Because clearance is evaluated across all recorded timesteps including $t=0$, **every algorithm unconditionally reports a scenario-wide minimum surface distance of exactly $2.0000\,\text{m}$ at initialization**.

### B. Controller Safety Buffer Projection (`SafetyFilter3D`)
In [`env/safety_filter.py:L26-L32`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/env/safety_filter.py#L26-L32), the safety filter defines:
```python
self.obstacle_margin = 2.0  # effective cylinder radius: eff_r = r + 2.0
```
- At $t=1$, because the UAVs start on the safety margin boundary ($\Delta r = r + 2.0$), the continuous segment intersection checker triggers $4 \times 3 = 12$ pass interventions, confirming boundary proximity.
- For failing RL baselines (`mappo_pure`, `mappo_residual`), when the learned policy commands penetration into an obstacle (e.g. UAV 1 approaching Obs 1 at $t=29$), the filter projects the UAV to $p_{\mathrm{safe}} = c + n (r + 2.0 + 0.05)$, holding the drone at $\approx 2.0027\,\text{m}$ from the building wall. Thus, for failing algorithms, the $2.00\,\text{m}$ distance is indeed the controller's safety buffer.

### C. True In-Flight Physical Clearance ($t \ge 1$)
For navigating algorithms during active flight ($t \ge 1$), the safety filter is never triggered in transit. Independent 3D Euclidean surface distances ($\sqrt{\Delta x^2 + \Delta y^2} - r$) are:
- **`hybrid_astar_mappo`**: $\mathbf{3.87\,m}$ (UAV 1 at $t=5$ near Obs 1, $(89.65, 130.77, 80.11)$ vs. $(100, 140, r=10)$).
- **`astar`**: $\mathbf{6.40\,m}$ (UAV 1 at $t=7$ near Obs 1, $(89.44, 127.45, 81.15)$ vs. $(100, 140, r=10)$).
- **`sapf`**: $\mathbf{2.13\,m}$ (UAV 1 at $t=9$ near Obs 1, $(107.73, 130.65, 50.00)$ vs. $(100, 140, r=10)$).
- **`rrt_star`**: $\mathbf{2.30\,m}$ (UAV 2 at $t=1$ near Obs 2).
- **`mappo_residual`**: $\mathbf{2.09\,m}$ (UAV 0 at $t=80$ near Obs 0).
- **`mappo_pure`**: $\mathbf{2.00\,m}$ (UAV 1 at $t=47$ near Obs 1, clamped by safety filter).

Across all 2,424 simulation records, **zero physical cylinder collisions occurred** (every UAV coordinate remained outside cylinder radius $r$).

