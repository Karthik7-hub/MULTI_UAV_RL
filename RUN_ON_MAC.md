# 🍏 Complete 1000-Episode Experiment Guide for macOS (Apple Silicon M-Series)

This guide provides clean, copy-paste terminal commands to run the entire **1000-Episode Multi-UAV Reinforcement Learning experiment** on macOS (MacBook Air / Pro with M1, M2, M3, M4, or M5 chips).

---

## ⚡ Option 1: Run Everything in One Shot (Copy & Paste)

Open **Terminal** on your Mac, navigate to the project directory, and paste this single block. 

> [!TIP]
> The `caffeinate` wrapper prevents your Mac from sleeping or throttling while training unattended.

```bash
# 1. Navigate to the project directory (update path if needed)
cd ~/research_v2

# 2. Prevent Mac from sleeping during execution
caffeinate -dimsu bash -c '
set -e

echo "=== [0/6] Setting up Virtual Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [1/6] Training Pure MAPPO (1000 episodes, Seed 42) ==="
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [2/6] Training Residual MAPPO (1000 episodes, Seed 42) ==="
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [3/6] Training IPPO Baseline (1000 episodes, Seed 42) ==="
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [4/6] Evaluating 12 Algorithms Across 5 Held-Out Seeds (101-105) ==="
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

echo "=== [5/6] Generating Publication Plots & Wide-Format Curves (300 DPI) ==="
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

echo "=== [6/6] Syncing Results to research_paper/ ==="
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
echo " [SUCCESS] 1000-Episode Experiment Completed Successfully!"
echo " Checkpoints: results/checkpoints/"
echo " CSV Tables:  research_paper/tables/"
echo " Figures:     research_paper/figures/"
echo "=========================================================="
'
```

---

## 🛠 Option 2: Step-by-Step Copy & Paste

If you prefer to run and inspect each phase individually, follow these steps:

### Step 0: Set Up Environment & Dependencies

```bash
cd ~/research_v2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 1: Train Pure MAPPO (1000 Episodes)
*Trains the proposed Centralized Training with Decentralized Execution (CTDE) cooperative policy without APF nominal guidance.*

```bash
source .venv/bin/activate
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto
```

* **Output Checkpoint:** `results/checkpoints/best_mappo_pure_agent.pt`
* **Output Log:** `results/training_logs_mappo_pure.csv` (1000 rows)

---

### Step 2: Train Residual MAPPO (1000 Episodes)
*Trains the hybrid policy combining classical Artificial Potential Field (APF) nominal guidance with learned reinforcement learning residual corrections ($k_{\mathrm{rl}} = 0.5$).*

```bash
source .venv/bin/activate
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto
```

* **Output Checkpoint:** `results/checkpoints/best_mappo_residual_agent.pt`
* **Output Log:** `results/training_logs_mappo_residual.csv` (1000 rows)

---

### Step 3: Train IPPO Baseline (1000 Episodes)
*Trains the Independent PPO (DTDE) baseline to produce the fresh comparison curves.*

```bash
source .venv/bin/activate
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto
```

* **Output Checkpoint:** `results/checkpoints/best_ippo_pure_agent.pt`
* **Output Log:** `results/training_logs_ippo_pure.csv` (1000 rows)

---

### Step 4: Multi-Seed Benchmark Evaluation Across 5 Seeds
*Evaluates all 12 navigation and task-offloading algorithms deterministically across 5 held-out evaluation seeds (101, 102, 103, 104, 105).*

```bash
source .venv/bin/activate
python evaluate.py --num_seeds 5 --strict_ckpts --device auto
```

* **Output CSVs Generated in `results/`:**
  * `benchmark_summary.csv` (Means ± standard deviations)
  * `benchmark_detailed.csv` (All individual run metrics)
  * `offloading_distribution.csv` (Local vs. Edge vs. Miss distribution)
  * `raw_evaluation_trajectories.csv` (Waypoint traces for all UAVs)

---

### Step 5: Generate Publication Figures & Wide-Format Curves (300 DPI)
*Renders the complete suite of publication plots, including the wide-format curves with rolling mean smoothing ($w=30$), dashed grid, and clean four-border frame.*

```bash
source .venv/bin/activate
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300
```

* **Plots Generated in `results/publication_figures_v3/`:**
  * `fig_task_completion_rate_wide.png` (Task Completion Rate: Pure MAPPO vs. IPPO, $w=30$)
  * `fig_episode_return_wide.png` (Episode Return: Pure MAPPO vs. IPPO, $w=30$)
  * `fig_fleet_energy_wide.png` (Fleet Mechanical & Comm Energy)
  * `fig_safety_interventions_wide.png` (Analytical Safety Shield Activations)
  * `fig_arrival_rate_wide.png` (UAV Navigation Arrival Rate)
  * `fig06_benchmark_table.png` (High-resolution benchmark comparison table)
  * `fig08_offloading_distribution.png` (Offloading breakdown)
  * `fig09_pareto_front.png` (Latency vs. Energy Pareto frontier)
  * `fig10_aerodynamic_power.png` (Rotary-wing power curve)

---

### Step 6: Sync Fresh Tables and Figures to `research_paper/`

```bash
source .venv/bin/activate

# Create directories
mkdir -p research_paper/tables research_paper/figures

# Copy CSV tables
cp results/*.csv research_paper/tables/

# Copy 300 DPI publication plots
cp results/publication_figures_v3/*.png research_paper/figures/

# Enable \includegraphics in research_paper.tex
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

## 🔍 Step 7: Verify Results (Sanity Check)

To verify that all models, tables, and figures were generated cleanly, run:

```bash
python -c "
import os, glob

print('=== CHECKPOINTS ===')
for f in sorted(glob.glob('results/checkpoints/*.pt')):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.1f} KB)')

print('\n=== TRAINING LOGS ===')
for f in sorted(glob.glob('results/training_logs_*.csv')):
    with open(f) as fp:
        lines = len(fp.readlines()) - 1
    print(f'  {os.path.basename(f)}: {lines} episodes')

print('\n=== PUBLICATION FIGURES ===')
for f in sorted(glob.glob('research_paper/figures/*.png')):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.1f} KB)')
"
```

Expected output:
* Checkpoints: `best_mappo_pure_agent.pt`, `best_mappo_residual_agent.pt`, `best_ippo_pure_agent.pt`.
* Training logs: 1000 episodes in `training_logs_mappo_pure.csv`, `training_logs_mappo_residual.csv`, `training_logs_ippo_pure.csv`.
* Figures: All `.png` figures present in `research_paper/figures/`.

---

## 💡 Mac M-Series Tips & Troubleshooting

1. **Keep Mac Awake:** Always run training under `caffeinate` if you plan to leave the Mac unattended or close the lid:
   ```bash
   caffeinate -dimsu python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto
   ```
2. **Apple Silicon Hardware Acceleration:** `--device auto` will automatically detect Apple Silicon's GPU via Metal Performance Shaders (MPS). If you want to explicitly force CPU or MPS:
   * `--device mps` (uses Apple Silicon GPU / Neural Engine)
   * `--device cpu` (uses Apple Silicon high-efficiency and performance CPU cores)
3. **Python Version:** Python 3.10, 3.11, or 3.12 works best on macOS. Check with `python3 --version`. If needed, install via Homebrew:
   ```bash
   brew install python@3.11
   ```

