#!/bin/sh
set -e

echo "=========================================================================="
echo " MULTI-UAV MARL REPRODUCIBILITY SUITE (MAC / LINUX)"
echo "=========================================================================="

# Ensure execution from repository root even if invoked from another directory
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# 1. Select Python environment:
#    - Respect active venv or conda env (e.g. torch 2.0.1 + cuda 11.8 for Quadro P2000)
#    - Else activate local .venv if valid
#    - Else use existing python3/python if torch is already installed
#    - Else create a fresh .venv
if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[Init] Using active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
elif [ -f ".venv/bin/activate" ]; then
    echo "[Init] Activating existing .venv..."
    . ./.venv/bin/activate
elif command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
    echo "[Init] Using system python3 with PyTorch already installed."
elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
    echo "[Init] Using system python with PyTorch already installed."
else
    echo "[Init] Creating Python virtual environment .venv..."
    rm -rf .venv
    python3 -m venv .venv
    . ./.venv/bin/activate
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

echo "[Init] Python: $(command -v "$PYTHON_BIN") ($("$PYTHON_BIN" --version))"
if "$PYTHON_BIN" -c "import torch, numpy, pandas, gymnasium, matplotlib, flask, pytest" >/dev/null 2>&1; then
    echo "[Init] All dependencies already installed — keeping current PyTorch/CUDA & NumPy versions intact."
else
    echo "[Init] Installing missing dependencies..."
    "$PYTHON_BIN" -m pip install --upgrade-strategy only-if-needed -r requirements.txt
fi

# 2. Sequential training of all algorithms (1000 episodes, per-episode validation logging)
echo ""
echo "=== [1/8] Training Proposed Pure MAPPO (CTDE, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model mappo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [2/8] Training Proposed Residual MAPPO (Hybrid APF+RL, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model mappo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [3/8] Training Independent PPO (IPPO Pure, DTDE, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model ippo --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [4/8] Training Residual IPPO (Hybrid APF+DTDE, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model ippo --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [5/8] Training Multi-Agent DDPG (MADDPG Pure, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model maddpg --variant pure --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [6/8] Training Residual MADDPG (Hybrid APF+DDPG, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model maddpg --variant residual --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [7/8] Training Centralized PPO (CPPO Theoretical Bound, 1000 Episodes) ==="
"$PYTHON_BIN" train.py --model cppo --episodes 1000 --eval_freq 1 --seed 42 --device auto

echo ""
echo "=== [8/8] Evaluating ALL Algorithms Across 5 Held-Out Seeds (101-105) ==="
"$PYTHON_BIN" evaluate.py --num_seeds 5 --strict_ckpts --device auto

echo ""
echo "=== Generating High-Resolution Publication Plots (Per-Episode Resolution, 300 DPI) ==="
"$PYTHON_BIN" generate_publication_plots.py --res_dir results --fig_dir results/publication_figures_v3 --per_episode --dpi 300

echo ""
echo "=== Syncing Tables & Figures to research_paper/ Directory ==="
mkdir -p research_paper/tables research_paper/figures
cp results/*.csv research_paper/tables/ 2>/dev/null || true
cp results/publication_figures_v3/*.png research_paper/figures/ 2>/dev/null || true

# Enable \includegraphics in LaTeX manuscript
"$PYTHON_BIN" -c '
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

echo "=========================================================================="
echo " [SUCCESS] Complete Suite Finished!"
echo " Checkpoints: results/checkpoints/"
echo " CSV Tables:  research_paper/tables/ and results/"
echo " Figures:     research_paper/figures/ and results/publication_figures_v3/"
echo "=========================================================================="
