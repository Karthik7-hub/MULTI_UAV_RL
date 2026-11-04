# 🚀 Complete Multi-UAV Research Pipeline Guide
### Cross-Platform Execution for macOS, Windows, and Linux (All MARL Models)

This guide provides clean, copy-paste terminal commands to train **ALL MARL models and baselines**, evaluate across 5 held-out test seeds, generate wide-format publication curves (displaying all models), and sync tables and figures across **macOS**, **Windows**, and **Linux**.

---

## Table of Contents
1. [🍏 macOS (Apple Silicon M1–M5 & Intel)](#-option-a-macos-apple-silicon-m1m5--intel)
2. [🪟 Windows (PowerShell)](#-option-b-windows-powershell)
3. [🐧 Linux (Ubuntu / Debian / Remote Cloud / Headless Server)](#-option-c-linux-ubuntu--debian--headless-server)
4. [🔍 Step-by-Step Reference for All Models](#-step-by-step-reference-for-all-models)
5. [✅ Verification & Sanity Check](#-verification--sanity-check)

---

## 🍏 Option A: macOS (Apple Silicon M1–M5 & Intel)

### One-Shot Copy & Paste (Terminal)
Open **Terminal**, navigate to the project directory, and paste this block. It uses `caffeinate` to keep your Mac awake during training:

```bash
cd ~/research_v2

caffeinate -dimsu bash -c '
set -e

echo "=== [0/8] Setting up Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [1/8] Training Pure MAPPO (1000 Episodes, Seed 42) ==="
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [2/8] Training Residual MAPPO (1000 Episodes, Seed 42) ==="
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [3/8] Training IPPO Pure (1000 Episodes, Seed 42) ==="
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [4/8] Training Residual IPPO (1000 Episodes, Seed 42) ==="
python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [5/8] Training MADDPG Pure (1000 Episodes, Seed 42) ==="
python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [6/8] Training Residual MADDPG (1000 Episodes, Seed 42) ==="
python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [7/8] Training CPPO Centralized (1000 Episodes, Seed 42) ==="
python train.py --model cppo --episodes 1000 --seed 42 --device auto

echo "=== [8/8] Evaluating ALL 12 Algorithms Across 5 Held-Out Seeds (101-105) ==="
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

echo "=== Generating All-Model Publication Plots & Wide Curves (300 DPI) ==="
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

echo "=== Syncing Results to research_paper/ Directory ==="
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
Open **PowerShell**, navigate to the project directory, and paste this block:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
Set-Location "C:\Users\vkart\Music\WORK\SEM\SEM_7\Minor_project\research_project\research_v2"

Write-Host "=== [0/8] Setting up Virtual Environment ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "=== [1/8] Training Pure MAPPO (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

Write-Host "=== [2/8] Training Residual MAPPO (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

Write-Host "=== [3/8] Training IPPO Pure (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

Write-Host "=== [4/8] Training Residual IPPO (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto

Write-Host "=== [5/8] Training MADDPG Pure (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto

Write-Host "=== [6/8] Training Residual MADDPG (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto

Write-Host "=== [7/8] Training CPPO Centralized (1000 episodes, Seed 42) ===" -ForegroundColor Cyan
python train.py --model cppo --episodes 1000 --seed 42 --device auto

Write-Host "=== [8/8] Evaluating ALL 12 Algorithms Across 5 Held-Out Seeds (101-105) ===" -ForegroundColor Cyan
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

Write-Host "=== Generating All-Model Publication Plots & Wide Curves (300 DPI) ===" -ForegroundColor Cyan
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

Write-Host "=== Syncing Results to research_paper/ ===" -ForegroundColor Cyan
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
```bash
tmux new -s multi_uav

cd ~/research_v2
set -e

echo "=== [0/8] Setting up Virtual Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [1/8] Training Pure MAPPO (1000 episodes, Seed 42) ==="
python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [2/8] Training Residual MAPPO (1000 episodes, Seed 42) ==="
python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [3/8] Training IPPO Pure (1000 episodes, Seed 42) ==="
python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [4/8] Training Residual IPPO (1000 episodes, Seed 42) ==="
python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [5/8] Training MADDPG Pure (1000 episodes, Seed 42) ==="
python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto

echo "=== [6/8] Training Residual MADDPG (1000 episodes, Seed 42) ==="
python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto

echo "=== [7/8] Training CPPO Centralized (1000 episodes, Seed 42) ==="
python train.py --model cppo --episodes 1000 --seed 42 --device auto

echo "=== [8/8] Evaluating ALL 12 Algorithms Across 5 Held-Out Seeds (101-105) ==="
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

echo "=== Generating All-Model Publication Plots & Wide Curves (300 DPI) ==="
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --dpi 300

echo "=== Syncing Results to research_paper/ ==="
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
echo "=========================================================="
```

---

## 🔍 Step-by-Step Reference for All Models

| Model | Variant | Command | Generated Checkpoint & Log |
|---|---|---|---|
| **Proposed Pure MAPPO** | Pure | `python train.py --model mappo --variant pure --episodes 1000 --seed 42 --device auto` | `best_mappo_pure_agent.pt`<br>`training_logs_mappo_pure.csv` |
| **Proposed Residual MAPPO**| Hybrid | `python train.py --model mappo --variant residual --episodes 1000 --seed 42 --device auto` | `best_mappo_residual_agent.pt`<br>`training_logs_mappo_residual.csv` |
| **Independent PPO (IPPO)** | Pure | `python train.py --model ippo --variant pure --episodes 1000 --seed 42 --device auto` | `best_ippo_pure_agent.pt`<br>`training_logs_ippo_pure.csv` |
| **Residual IPPO** | Hybrid | `python train.py --model ippo --variant residual --episodes 1000 --seed 42 --device auto` | `best_ippo_residual_agent.pt`<br>`training_logs_ippo_residual.csv` |
| **Multi-Agent DDPG** | Pure | `python train.py --model maddpg --variant pure --episodes 1000 --seed 42 --device auto` | `best_maddpg_pure_agent.pt`<br>`training_logs_maddpg_pure.csv` |
| **Residual MADDPG** | Hybrid | `python train.py --model maddpg --variant residual --episodes 1000 --seed 42 --device auto` | `best_maddpg_residual_agent.pt`<br>`training_logs_maddpg_residual.csv` |
| **Centralized PPO (CPPO)** | CTCE | `python train.py --model cppo --episodes 1000 --seed 42 --device auto` | `best_cppo_agent.pt`<br>`training_logs_cppo.csv` |
| **Ablation: No Energy** | MAPPO | `python train.py --model ablation --ablation no_energy --episodes 1000 --seed 42 --device auto` | Checkpoint in `results/checkpoints/` |
| **Ablation: No Safety** | MAPPO | `python train.py --model ablation --ablation no_safety --episodes 1000 --seed 42 --device auto` | Checkpoint in `results/checkpoints/` |
| **Ablation: Fixed Offload**| MAPPO | `python train.py --model ablation --ablation fixed_offload --episodes 1000 --seed 42 --device auto` | Checkpoint in `results/checkpoints/` |
| **All-Algorithm Suite** | All | `python train.py --model all --variant pure --episodes 1000 --seed 42 --device auto` | Trains MAPPO, IPPO, MADDPG, CPPO sequentially |

---

## 📊 Publication Figures Generated (All Models Displayed)

Running `python generate_publication_plots.py` renders every single model on each wide comparison curve:

1. **`fig_task_completion_rate_wide.png`**: **ALL MODELS** (Pure MAPPO, Residual MAPPO, IPPO, Residual IPPO, MADDPG, Residual MADDPG, CPPO) task completion % vs. episode.
2. **`fig_episode_return_wide.png`**: **ALL MODELS** cumulative episode return curve with rolling $w=30$ smoothing.
3. **`fig_fleet_energy_wide.png`**: **ALL MODELS** fleet mechanical & communication energy (kJ).
4. **`fig_safety_interventions_wide.png`**: **ALL MODELS** continuous safety shield interventions per episode.
5. **`fig_arrival_rate_wide.png`**: **ALL MODELS** UAV navigation arrival rate %.
6. **`fig_task_completion_rate_pure_vs_ippo.png`**: Dedicated isolated Pure MAPPO vs. IPPO curve.
7. **`fig_episode_return_pure_vs_ippo.png`**: Dedicated isolated Pure MAPPO vs. IPPO return curve.

---

## ✅ Verification & Sanity Check

```bash
# macOS / Linux
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
