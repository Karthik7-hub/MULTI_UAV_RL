# 🚀 Complete 1000-Episode Multi-UAV Research Pipeline Guide
### Cross-Platform Execution for macOS, Windows, and Linux

This guide provides clean, ready-to-run copy-paste commands to execute the complete **1000-episode Multi-UAV Reinforcement Learning experiment**, multi-seed evaluation, publication-quality figure generation (with $w=30$ rolling smoothing), and paper syncing across **macOS**, **Windows**, and **Linux**.

---

## Table of Contents
1. [🍏 macOS (Apple Silicon M1–M5 & Intel)](#-option-a-macos-apple-silicon-m1m5--intel)
2. [🪟 Windows (PowerShell)](#-option-b-windows-powershell)
3. [🐧 Linux (Ubuntu / Debian / Remote Cloud / Headless Server)](#-option-c-linux-ubuntu--debian--headless-server)
4. [🔍 Step-by-Step Phase Reference (All Platforms)](#-step-by-step-phase-reference-all-platforms)
5. [✅ Sanity Check & Verification](#-verification--sanity-check)

---

## 🍏 Option A: macOS (Apple Silicon M1–M5 & Intel)

### One-Shot Copy & Paste (Terminal)
Open **Terminal**, navigate to the project directory, and paste this block. It uses `caffeinate` to keep your Mac awake during training:

```bash
cd ~/research_v2

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
echo " [SUCCESS] Pipeline complete on macOS!"
echo " Checkpoints: results/checkpoints/"
echo " Tables:      research_paper/tables/"
echo " Figures:     research_paper/figures/"
echo "=========================================================="
'
```

---

## 🪟 Option B: Windows (PowerShell)

### One-Shot Copy & Paste (PowerShell)
Open **PowerShell** as regular user or administrator, navigate to the project directory, and paste this block:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
Set-Location "C:\Users\vkart\Music\WORK\SEM\SEM_7\Minor_project\research_project\research_v2"

Write-Host "=== [0/6] Setting up Virtual Environment ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "=== [1/6] Training Pure MAPPO (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

Write-Host "=== [2/6] Training Residual MAPPO (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

Write-Host "=== [3/6] Training IPPO Baseline (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

Write-Host "=== [4/6] Evaluating 12 Algorithms Across 5 Held-Out Seeds (101-105) ===" -ForegroundColor Cyan
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

Write-Host "=== [5/6] Generating Publication Plots & Wide-Format Curves (300 DPI) ===" -ForegroundColor Cyan
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

Write-Host "=== [6/6] Syncing Results to research_paper/ ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "research_paper\tables", "research_paper\figures" | Out-Null
Copy-Item "results\*.csv" -Destination "research_paper\tables\" -Force
Copy-Item "results\publication_figures_v3\*.png" -Destination "research_paper\figures\" -Force

python -c "
import re
tex_path = 'research_paper/manuscript/research_paper.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('% \\includegraphics', '\\includegraphics')
c = re.sub(r'\\fbox\{\\parbox\{.*?\}\}', '', c, flags=re.DOTALL)
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(c)
print('  [DONE] Enabled figures in research_paper.tex')
"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Pipeline complete on Windows!" -ForegroundColor Green
Write-Host " Checkpoints: results\checkpoints\" -ForegroundColor Green
Write-Host " Tables:      research_paper\tables\" -ForegroundColor Green
Write-Host " Figures:     research_paper\figures\" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
```

---

## 🐧 Option C: Linux (Ubuntu / Debian / Headless Server)

### One-Shot Copy & Paste (Bash / `tmux`)
For remote servers or headless clusters, it is strongly recommended to run inside `tmux` or `screen` so the job survives SSH disconnects:

```bash
# Optional: start a tmux session
tmux new -s multi_uav

# Navigate to project directory
cd ~/research_v2

# Run full pipeline
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

python -c "
import re
tex_path = \"research_paper/manuscript/research_paper.tex\"
with open(tex_path, \"r\", encoding=\"utf-8\") as f:
    c = f.read()
c = c.replace(\"% \\\\includegraphics\", \"\\\\includegraphics\")
c = re.sub(r\"\\\\fbox\\{\\\\parbox\\{.*?\\}\\}\", \"\", c, flags=re.DOTALL)
with open(tex_path, \"w\", encoding=\"utf-8\") as f:
    f.write(c)
print(\"  [DONE] Enabled figures in research_paper.tex\")
"

echo "=========================================================="
echo " [SUCCESS] Pipeline complete on Linux!"
echo " Checkpoints: results/checkpoints/"
echo " Tables:      research_paper/tables/"
echo " Figures:     research_paper/figures/"
echo "=========================================================="
```

---

## 🛠 Step-by-Step Phase Reference (All Platforms)

If you prefer to execute commands manually one by one, use the following standardized CLI commands:

| Step | Command (Run in activated environment) | Primary Artifact Generated |
|---|---|---|
| **1. Pure MAPPO** | `python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto` | `results/checkpoints/best_mappo_pure_agent.pt`<br>`results/training_logs_mappo_pure.csv` |
| **2. Residual MAPPO** | `python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto` | `results/checkpoints/best_mappo_residual_agent.pt`<br>`results/training_logs_mappo_residual.csv` |
| **3. IPPO Baseline** | `python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto` | `results/checkpoints/best_ippo_pure_agent.pt`<br>`results/training_logs_ippo_pure.csv` |
| **4. Multi-Seed Eval** | `python evaluate.py --num_seeds 5 --strict_ckpts --device auto` | `results/benchmark_summary.csv`<br>`results/benchmark_detailed.csv` |
| **5. Plot Curves** | `python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300` | `fig_task_completion_rate_wide.png`<br>`fig_episode_return_wide.png` ($w=30$) |
| **6. Sync Paper** | *See Step 6 script block above* | `research_paper/tables/`<br>`research_paper/figures/` |

---

## 🔍 Verification / Sanity Check

Run this cross-platform verification script to confirm that all weights, 1000-episode logs, and publication plots exist and have valid file sizes:

```bash
# macOS / Linux
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

```powershell
# Windows PowerShell
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

---

## ⚡ Hardware Acceleration Summary Across Platforms

* **macOS (Apple Silicon M1–M5)**: Automatically routes tensors through Apple Metal Performance Shaders (`mps`).
* **Windows & Linux (NVIDIA GPU)**: Automatically routes tensors through NVIDIA CUDA (`cuda:0`).
* **Fallback**: Runs gracefully on multi-threaded CPU across all operating systems.

