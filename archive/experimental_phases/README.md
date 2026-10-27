# Archive: Historical Research Phases & Experimental Artifacts

This archive preserves the exact configurations, scripts, and results from Phases 1, 2, and 3 prior to the final production cleanup.

---

## Phase 1: Actor Movement Initialization Controlled Ablation
- **Objective:** Determine whether MAPPO navigation failure was driven by movement head initialization and directional exploration bias.
- **Key Findings:**
  - Standard PyTorch initialization (gain=0.4, log_std=-0.2) produced initial velocity means strongly misaligned with target stations, with $P(v_\parallel > 0) < 0.01$ in 25% of states.
  - Proposed validated initialization (gain=0.01, log_std=0.5) mathematically guaranteed isotropic exploration $P(v_\parallel > 0) = 50.4\% \pm 4.0\%$.
  - Obstacle-free station arrival improved from $0.0\%$ to $33.3\% \pm 14.4\%$.
- **Historical Artifacts & Scripts:**
  - `results/ablation_initialization_study/statistical_summary.csv`
  - `results/ablation_initialization_study/raw_eval_records.csv`
  - `scratch/run_controlled_initialization_study.py`
  - `scratch/verify_action_probabilities.py`

---

## Phase 2: Production Initialization Integration
- **Objective:** Integrate validated movement initialization into production `models/mappo.py`.
- **Implementation:**
  - Configurable `gain=0.01` and `log_std=0.5` on `move_mean_head`.
  - Regression test suite `tests/test_movement_init.py`.

---

## Phase 3: Obstacle Trapping Diagnosis & Circumnavigation Variants
- **Objective:** Overcome obstacle trapping deadlock in standard 4-cylinder environment.
- **Key Findings:**
  - Diagnostic trajectory trace (`results/phase3_diagnosis/trapped_trajectory_trace.csv`) confirmed Hypothesis 4: interaction between safety-filter zeroing forward speed ($v_{\text{exec}} = 0$ for steps 11–99), missing directional tangent features in canonical 20-dim obs, and Euclidean progress reward penalizing valid lateral bypasses.
  - 4-way controlled multi-seed evaluation ($N_{\text{train}}=3$ seeds $\times$ 5 held-out eval seeds):
    * Baseline: $0.0\% \pm 0.0\%$ arrival, $35.4 \pm 2.8\,\text{m}$ final distance, $19.4\,\text{m}$ path length, $294.7$ interventions.
    * Obs Variant (22-dim): $1.7\% \pm 2.9\%$ arrival, $115.9\,\text{m}$ path length (deadlock broken!), $119.7$ interventions ($-59\%$).
    * Reward Variant: $0.0\% \pm 0.0\%$ arrival, $42.6\,\text{m}$ path length.
    * Combined Variant: $11.7\% \pm 10.4\%$ arrival (up to $20.0\%$ in Seed 44), $29.4 \pm 9.2\,\text{m}$ final distance (Seeds 42/44: $23.3\text{--}24.8\,\text{m}$), $+15.6\,\text{m}$ net progress.
- **Historical Artifacts & Scripts:**
  - `results/phase3_variants/statistical_variant_summary.csv`
  - `results/phase3_variants/raw_variant_eval_records.csv`
  - `results/phase3_diagnosis/trapped_trajectory_trace.csv`
  - `scratch/run_phase3_obstacle_variants.py`
  - `scratch/trace_obstacle_trapping.py`

---

## Production Cleanup Refactoring
In this final cleanup step, experimental modes (`movement_init_mode`, `enable_tangent_obs`, `enable_detour_reward`) are removed from production code. The validated initialization (`gain=0.01`, `log_std=0.5`), the 22-dimensional observation vector, and the detour-aware progress reward are integrated directly into the canonical architecture.
