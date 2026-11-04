# 🍏 Complete Multi-UAV Experiment Guide for macOS (Apple Silicon M-Series)

This guide provides clean, copy-paste terminal commands to train **ALL MARL models and baselines**, run the multi-seed evaluation, generate the publication curves (displaying all models), and sync tables and figures on macOS (MacBook Air / Pro with M1, M2, M3, M4, or M5 chips).

---

## ⚡ Option 1: Run ALL Models in One Shot (Copy & Paste)

Open **Terminal** on your Mac, navigate to the project directory, and paste this single block.

> [!TIP]
> The `caffeinate` wrapper prevents your Mac from sleeping or throttling while training unattended overnight.

```bash
# 1. Navigate to the project directory
cd ~/research_v2

# 2. Run the complete pipeline under caffeinate (prevents Mac sleep)
caffeinate -dimsu bash -c '
set -e

echo "=== [0/8] Setting up Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [1/8] Training Proposed Pure MAPPO (CTDE, 1000 Episodes, Seed 42) ==="
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [2/8] Training Proposed Residual MAPPO (Hybrid APF+RL, 1000 Episodes, Seed 42) ==="
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [3/8] Training Independent PPO (IPPO Pure, DTDE, 1000 Episodes, Seed 42) ==="
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [4/8] Training Residual IPPO (Hybrid APF+DTDE, 1000 Episodes, Seed 42) ==="
python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [5/8] Training Multi-Agent DDPG (MADDPG Pure, 1000 Episodes, Seed 42) ==="
python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [6/8] Training Residual MADDPG (Hybrid APF+DDPG, 1000 Episodes, Seed 42) ==="
python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [7/8] Training Centralized PPO (CPPO Theoretical Bound, 1000 Episodes, Seed 42) ==="
python train.py --model cppo --episodes 1000 --seed 42 --device auto

echo "=== [8/8] Evaluating ALL 12 Algorithms Across 5 Held-Out Seeds (101-105) ==="
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

echo "=== Generating All-Model Publication Plots & Wide Curves (300 DPI) ==="
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

echo "=== Syncing Results to research_paper/ Directory ==="
mkdir -p research_paper/tables research_paper/figures
cp results/*.csv research_paper/tables/
cp results/publication_figures_v3/*.png research_paper/figures/

# Enable \includegraphics in LaTeX manuscript
python -c "
import re
tex_path = \"research_paper/manuscript/research_paper.tex\"
with open(tex_path, \"r\") as f:
    c = f.read()
c = c.replace(\"% \\\\includegraphics\", \"\\\\includegraphics\")
c = re.sub(r\"\\\\fbox\\{\\\\parbox\\{.*?\\}\\}\", \"\", c, flags=re.DOTALL)
with open(tex_path, \"w\") as f:
    f.write(c)
print(\"  [DONE] Enabled figures in research_paper.tex\")
"

echo "=========================================================="
echo " [SUCCESS] All Models Trained, Evaluated, and Synced!"
echo " Checkpoints: results/checkpoints/"
echo " CSV Tables:  research_paper/tables/"
echo " Figures:     research_paper/figures/"
echo "=========================================================="
'
```

---

## 🛠 Option 2: Step-by-Step Copy & Paste (Individual Models)

If you prefer to train, evaluate, and plot specific models individually:

### Step 0: Set Up Environment & Dependencies
```bash
cd ~/research_v2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 1: Proposed Pure MAPPO (1000 Episodes)
*Centralized Training with Decentralized Execution (CTDE) — Pure neural policy.*
```bash
source .venv/bin/activate
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_mappo_pure_agent.pt`
* **Log:** `results/training_logs_mappo_pure.csv`

---

### Step 2: Proposed Residual MAPPO (1000 Episodes)
*Hybrid architecture: Classical Artificial Potential Field (APF) nominal guidance + learned residual DRL corrections ($k_{\mathrm{rl}} = 0.5$).*
```bash
source .venv/bin/activate
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_mappo_residual_agent.pt`
* **Log:** `results/training_logs_mappo_residual.csv`

---

### Step 3: Independent PPO (IPPO Pure, 1000 Episodes)
*Decentralized Training with Decentralized Execution (DTDE) baseline — Selfish policy without centralized critic.*
```bash
source .venv/bin/activate
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_ippo_pure_agent.pt`
* **Log:** `results/training_logs_ippo_pure.csv`

---

### Step 4: Residual IPPO (1000 Episodes)
*Hybrid nominal APF guidance + learned residual IPPO policy.*
```bash
source .venv/bin/activate
python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_ippo_residual_agent.pt`
* **Log:** `results/training_logs_ippo_residual.csv`

---

### Step 5: Multi-Agent DDPG (MADDPG Pure, 1000 Episodes)
*Continuous off-policy actor-critic baseline with centralized Q-critic.*
```bash
source .venv/bin/activate
python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_maddpg_pure_agent.pt`
* **Log:** `results/training_logs_maddpg_pure.csv`

---

### Step 6: Residual MADDPG (1000 Episodes)
*Hybrid nominal APF guidance + learned off-policy MADDPG policy.*
```bash
source .venv/bin/activate
python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_maddpg_residual_agent.pt`
* **Log:** `results/training_logs_maddpg_residual.csv`

---

### Step 7: Centralized PPO (CPPO, 1000 Episodes)
*Centralized Training with Centralized Execution (CTCE) — Single centralized policy controlling all UAVs (theoretical benchmark).*
```bash
source .venv/bin/activate
python train.py --model cppo --episodes 1000 --seed 42 --device auto
```
* **Checkpoint:** `results/checkpoints/best_cppo_agent.pt`
* **Log:** `results/training_logs_cppo.csv`

---

### Step 8: MAPPO Ablation Studies (Optional, 1000 Episodes each)
```bash
source .venv/bin/activate

# Ablation 1: No energy awareness in reward function
python train.py --model ablation --ablation no_energy --episodes 1000 --seed 42 --device auto

# Ablation 2: No continuous analytical safety shield
python train.py --model ablation --ablation no_safety --episodes 1000 --seed 42 --device auto

# Ablation 3: Fixed heuristic computation offloading (no learned offloading)
python train.py --model ablation --ablation fixed_offload --episodes 1000 --seed 42 --device auto
```

---

### Step 9: Multi-Seed Benchmark Evaluation Across 5 Seeds
*Evaluates all 12 algorithms deterministically across 5 held-out evaluation seeds (101, 102, 103, 104, 105).*
```bash
source .venv/bin/activate
python evaluate.py --num_seeds 5 --strict_ckpts --device auto
```
* **Output CSVs Generated in `results/`:**
  * `benchmark_summary.csv` (Means ± standard deviations)
  * `benchmark_detailed.csv` (Per-seed metrics)
  * `offloading_distribution.csv` (Local vs. Edge vs. Miss breakdown)
  * `raw_evaluation_trajectories.csv` (Full waypoint traces)

---

### Step 10: Generate All-Model Publication Plots & Wide Curves (300 DPI)
*Renders high-resolution figures displaying **ALL trained models** with $w=30$ rolling smoothing, four-border frame, and dashed grid.*
```bash
source .venv/bin/activate
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300
```

* **Plots Generated in `results/publication_figures_v3/`:**
  * `fig_task_completion_rate_wide.png` (**ALL MODELS**: Pure MAPPO, Residual MAPPO, IPPO, MADDPG, CPPO)
  * `fig_task_completion_rate_pure_vs_ippo.png` (Pairwise: Pure MAPPO vs. IPPO)
  * `fig_episode_return_wide.png` (**ALL MODELS**: Cumulative returns, $w=30$)
  * `fig_episode_return_pure_vs_ippo.png` (Pairwise: Pure MAPPO vs. IPPO return)
  * `fig_fleet_energy_wide.png` (**ALL MODELS**: Total fleet mechanical & comm energy in kJ)
  * `fig_safety_interventions_wide.png` (**ALL MODELS**: Safety filter interventions)
  * `fig_arrival_rate_wide.png` (**ALL MODELS**: UAV arrival rate %)
  * `fig06_benchmark_table.png` (Publication comparison table)
  * `fig08_offloading_distribution.png` (Stacked offloading distribution)
  * `fig09_pareto_front.png` (Latency vs. Energy Pareto frontier)
  * `fig10_aerodynamic_power.png` (Rotary-wing power curve)

---

### Step 11: Sync Results to `research_paper/`
```bash
source .venv/bin/activate
mkdir -p research_paper/tables research_paper/figures
cp results/*.csv research_paper/tables/
cp results/publication_figures_v3/*.png research_paper/figures/

# Enable \includegraphics in LaTeX manuscript
python -c '
import re
tex_path = "research_paper/manuscript/research_paper.tex"
with open(tex_path, "r") as f:
    c = f.read()
c = c.replace("% \\includegraphics", "\\includegraphics")
c = re.sub(r"\\fbox\{\\parbox\{.*?\}\}", "", c, flags=re.DOTALL)
with open(tex_path, "w") as f:
    f.write(c)
print("[DONE] Enabled figures in research_paper/manuscript/research_paper.tex")
'
```

---

## 🔍 Verification & Sanity Check

Run this verification snippet to check that all models, 1000-episode logs, and publication plots exist and have non-zero file sizes:

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
for f in sorted(glob.glob('research_paper/figures/*.png')):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.1f} KB)')
"
```
