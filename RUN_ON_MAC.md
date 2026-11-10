# 🚀 Complete Multi-UAV Experiment & Execution Guide (macOS · Windows · Linux)

This guide provides tested, copy-paste commands and automated scripts to train **ALL MARL models and baselines**, run the 5-seed benchmark evaluation, generate publication-grade per-episode figures, and sync all tables and figures directly to the LaTeX manuscript.

---

## ⚡ Option 1: Run ALL in One Shot

### 🍏 macOS (Terminal / Apple Silicon & Intel)
The `caffeinate` wrapper keeps your Mac awake and running at full performance even overnight:

```bash
# Navigate to workspace
cd ~/research_v2

# Run full pipeline with sleep prevention (works under both sh and bash)
caffeinate -dimsu sh run_experiments.sh
```

*(Alternatively, paste the entire inline block:)*
```bash
cd ~/research_v2
caffeinate -dimsu sh -c '
set -e
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    if [ ! -f ".venv/bin/activate" ]; then rm -rf .venv && python3 -m venv .venv; fi
    . ./.venv/bin/activate
fi
python3 -c "import torch, numpy, pandas, gymnasium, matplotlib, flask, pytest" 2>/dev/null || pip install --upgrade-strategy only-if-needed -r requirements.txt

python train.py --model mappo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model mappo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model ippo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model ippo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model maddpg --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model maddpg --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model cppo --episodes 1000 --eval_freq 1 --seed 42 --device auto

python evaluate.py --num_seeds 5 --strict_ckpts --device auto
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --per_episode --dpi 300

mkdir -p research_paper/tables research_paper/figures
cp results/*.csv research_paper/tables/ 2>/dev/null || true
cp results/publication_figures_v3/*.png research_paper/figures/ 2>/dev/null || true
echo "[DONE] Complete suite executed and synced successfully!"
'
```

---

### 🐧 Linux (Ubuntu / Debian / CentOS / Arch / Quadro P2000 & Modern GPUs)

> [!TIP]
> **NVIDIA Quadro P2000 / Pascal (`sm_61`) Compatibility:**
> - `run_experiments.sh` automatically preserves your active `conda` / `.venv` Python environment (e.g. `torch==2.0.1` + CUDA 11.8) without upgrading PyTorch or NumPy.
> - If you run on a newer PyTorch build (`>=2.7`) that dropped `sm_61`, `--device auto` automatically probes GPU kernel execution and falls back cleanly to CPU without crashing.
> - To use CUDA directly on a Quadro P2000 (`sm_61`), install PyTorch CUDA 11.8 before running:
>   `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 "numpy<2.0"`

Run with `sh run_experiments.sh` or directly with `./run_experiments.sh`:

```bash
# Navigate to workspace
cd ~/research_v2

# Run directly with sh or bash (both work cleanly!)
sh run_experiments.sh
```

*(For remote servers over SSH without disconnection risk:)*
```bash
chmod +x run_experiments.sh
nohup sh run_experiments.sh > training_suite.log 2>&1 &
tail -f training_suite.log
```

---

### 🪟 Windows (PowerShell)
Open **PowerShell** (or Windows Terminal) in your project directory:

```powershell
# Navigate to workspace
cd "c:\Users\vkart\Music\WORK\SEM\SEM_7\Minor_project\research_v2"

# Run full pipeline script
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
```

*(Alternatively, paste this inline block into PowerShell:)*
```powershell
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt

python train.py --model mappo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model mappo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model ippo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model ippo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model maddpg --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model maddpg --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
python train.py --model cppo --episodes 1000 --eval_freq 1 --seed 42 --device auto

python evaluate.py --num_seeds 5 --strict_ckpts --device auto
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --per_episode --dpi 300

New-Item -ItemType Directory -Force -Path "research_paper/tables", "research_paper/figures" | Out-Null
Copy-Item results/*.csv -Destination research_paper/tables/ -Force
Copy-Item results/publication_figures_v3/*.png -Destination research_paper/figures/ -Force
Write-Host "[DONE] Complete suite executed and synced successfully!" -ForegroundColor Green
```

---

## 🛠 Option 2: Step-by-Step Execution (Individual Models)

If you prefer to train, evaluate, and inspect models individually:

### Step 0: Activate Environment
* **macOS / Linux (`sh` / `bash`):**
  ```bash
  . ./.venv/bin/activate
  ```
* **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

---

### Step 1: Proposed Pure MAPPO (1000 Episodes)
*Centralized Training with Decentralized Execution (CTDE) — Pure neural policy.*
```bash
python train.py --model mappo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_mappo_pure_agent.pt`
* **Log:** `results/training_logs_mappo_pure.csv`

---

### Step 2: Proposed Residual MAPPO (1000 Episodes)
*Hybrid architecture: Classical Artificial Potential Field (APF) nominal guidance + learned residual DRL corrections ($k_{\mathrm{rl}} = 0.5$).*
```bash
python train.py --model mappo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_mappo_residual_agent.pt`
* **Log:** `results/training_logs_mappo_residual.csv`

---

### Step 3: Independent PPO (IPPO Pure, 1000 Episodes)
*Decentralized Training with Decentralized Execution (DTDE) baseline — Local critic per UAV.*
```bash
python train.py --model ippo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_ippo_pure_agent.pt`
* **Log:** `results/training_logs_ippo_pure.csv`

---

### Step 4: Residual IPPO (1000 Episodes)
*Hybrid nominal APF guidance + learned decentralized residual IPPO policy.*
```bash
python train.py --model ippo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_ippo_residual_agent.pt`
* **Log:** `results/training_logs_ippo_residual.csv`

---

### Step 5: Multi-Agent DDPG (MADDPG Pure, 1000 Episodes)
*Continuous off-policy actor-critic baseline with centralized Q-critic and decaying exploration.*
```bash
python train.py --model maddpg --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_maddpg_pure_agent.pt`
* **Log:** `results/training_logs_maddpg_pure.csv`

---

### Step 6: Residual MADDPG (1000 Episodes)
*Hybrid nominal APF guidance + learned off-policy MADDPG policy.*
```bash
python train.py --model maddpg --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_maddpg_residual_agent.pt`
* **Log:** `results/training_logs_maddpg_residual.csv`

---

### Step 7: Centralized PPO (CPPO, 1000 Episodes)
*Centralized Training with Centralized Execution (CTCE) — Theoretical benchmark.*
```bash
python train.py --model cppo --episodes 1000 --eval_freq 1 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_cppo_agent.pt`
* **Log:** `results/training_logs_cppo.csv`

---

### Step 8: Multi-Seed Benchmark Evaluation Across 5 Seeds
*Evaluates all algorithms deterministically across held-out seeds (101, 102, 103, 104, 105).*
```bash
python evaluate.py --num_seeds 5 --strict_ckpts --device auto
```
* **Output CSVs Generated in `results/`:**
  * `benchmark_summary.csv` (Means ± standard deviations)
  * `benchmark_detailed.csv` (Per-seed metrics)
  * `offloading_distribution.csv` (Local vs. Peer vs. Edge breakdown)
  * `raw_evaluation_trajectories.csv` (Full waypoint traces)

---

### Step 9: Generate Publication Figures (Per-Episode Resolution, 300 DPI)
*Renders high-resolution figures displaying **ALL trained models** at per-episode resolution without blur or smoothing:*
```bash
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --per_episode --dpi 300
```

* **Generated in `results/publication_figures_v3/`:**
  * `fig01_training_return.png` (Training return vs. episode)
  * `fig02_training_rates.png` (Task completion & arrival rate vs. episode)
  * `fig03_training_energy_safety.png` (Fleet mechanical energy & safety interventions)
  * `fig04_validation_curves.png` (Held-out validation return, task completion, and arrival rate)
  * `fig06_benchmark_table.png` (Publication comparison table with all MARL and classical baselines)
  * `fig07_navigation_benchmark.png` (3D navigation speed, path length, and docking accuracy)
  * `fig08_offloading_distribution.png` (Stacked offloading distribution breakdown)
  * `fig09_pareto_front.png` (Latency vs. Energy Pareto trade-off curve)
  * `fig10_aerodynamic_power.png` (Rotary-wing aerodynamic power curve)
  * `fig12_per_seed_variability.png` (Multi-seed boxplot distributions)
  * `fig13_multiseed_ci.png` (Confidence interval bands)
  * Wide-format publication curves: `fig_task_completion_rate_wide.png`, `fig_episode_return_wide.png`, `fig_fleet_energy_wide.png`, `fig_arrival_rate_wide.png`, `fig_safety_interventions_wide.png`

---

### Step 10: Sync Results to LaTeX Manuscript
* **macOS / Linux:**
  ```bash
  mkdir -p research_paper/tables research_paper/figures
  cp results/*.csv research_paper/tables/
  cp results/publication_figures_v3/*.png research_paper/figures/
  ```
* **Windows PowerShell:**
  ```powershell
  New-Item -ItemType Directory -Force -Path "research_paper/tables", "research_paper/figures" | Out-Null
  Copy-Item results/*.csv -Destination research_paper/tables/ -Force
  Copy-Item results/publication_figures_v3/*.png -Destination research_paper/figures/ -Force
  ```

---

## 🔍 Verification & Sanity Check

Run this cross-platform Python snippet to verify that all checkpoints, 1000-episode logs, and publication figures exist and are non-empty:

```bash
python -c "
import os, glob

print('=== ALL MODEL CHECKPOINTS ===')
for f in sorted(glob.glob('results/checkpoints/*.pt')):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.1f} KB)')

print('\n=== ALL TRAINING LOGS ===')
for f in sorted(glob.glob('results/training_logs_*.csv')):
    with open(f) as fp:
        lines = len(fp.readlines()) - 1
    print(f'  {os.path.basename(f)}: {lines} episodes')

print('\n=== ALL PUBLICATION FIGURES ===')
for f in sorted(glob.glob('results/publication_figures_v3/*.png')):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.1f} KB)')
"
```
