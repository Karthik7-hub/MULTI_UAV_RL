# Multi-UAV MARL Reproducibility Suite (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " MULTI-UAV MARL REPRODUCIBILITY SUITE (WINDOWS)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Setup virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[Init] Creating Python virtual environment .venv..." -ForegroundColor Yellow
    python -m venv .venv
}
& ".\.venv\Scripts\Activate.ps1"

Write-Host "[Init] Upgrading pip and verifying dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Sequential training of all algorithms (1000 episodes, per-episode validation logging)
Write-Host "`n=== [1/8] Training Proposed Pure MAPPO (CTDE, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model mappo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [2/8] Training Proposed Residual MAPPO (Hybrid APF+RL, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model mappo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [3/8] Training Independent PPO (IPPO Pure, DTDE, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model ippo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [4/8] Training Residual IPPO (Hybrid APF+DTDE, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model ippo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [5/8] Training Multi-Agent DDPG (MADDPG Pure, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model maddpg --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [6/8] Training Residual MADDPG (Hybrid APF+DDPG, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model maddpg --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [7/8] Training Centralized PPO (CPPO Theoretical Bound, 1000 Episodes) ===" -ForegroundColor Green
python train.py --model cppo --episodes 1000 --eval_freq 1 --seed 42 --device auto

Write-Host "`n=== [8/8] Evaluating ALL Algorithms Across 5 Held-Out Seeds (101-105) ===" -ForegroundColor Green
python evaluate.py --num_seeds 5 --strict_ckpts --device auto

Write-Host "`n=== Generating High-Resolution Publication Plots (Per-Episode Resolution, 300 DPI) ===" -ForegroundColor Green
python generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --per_episode --dpi 300

Write-Host "`n=== Syncing Tables & Figures to research_paper/ Directory ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "research_paper/tables", "research_paper/figures" | Out-Null
Copy-Item results/*.csv -Destination research_paper/tables/ -Force -ErrorAction SilentlyContinue
Copy-Item results/publication_figures_v3/*.png -Destination research_paper/figures/ -Force -ErrorAction SilentlyContinue

# Enable \includegraphics in LaTeX manuscript
python -c '
import re, os
tex_path = "research_paper/manuscript/research_paper.tex"
if os.path.exists(tex_path):
    with open(tex_path, "r") as f:
        c = f.read()
    c = c.replace("% \\includegraphics", "\\includegraphics")
    c = re.sub(r"\\fbox\{\\parbox\{.*?\}\}", "", c, flags=re.DOTALL)
    with open(tex_path, "w") as f:
        f.write(c)
    print("  [DONE] Enabled figures in research_paper.tex")
'

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " [SUCCESS] Complete Suite Finished!" -ForegroundColor Cyan
Write-Host " Checkpoints: results/checkpoints/" -ForegroundColor Cyan
Write-Host " CSV Tables:  research_paper/tables/ and results/" -ForegroundColor Cyan
Write-Host " Figures:     research_paper/figures/ and results/publication_figures_v3/" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

