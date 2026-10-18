"""
Master CLI Entry Point for Multi-UAV Deep Reinforcement Learning Research Pipeline.
Usage:
    python main.py --mode train --model mappo --device auto
    python main.py --mode evaluate --device auto
    python main.py --mode ablation --ablation no_energy
    python main.py --mode plot
    python main.py --mode all --device auto
"""

import argparse
import os
import sys

# Add directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_all, train_mappo, train_ippo, train_maddpg, train_cppo
from evaluate import run_evaluation
from plot_results import regenerate_all_figures

def main():
    parser = argparse.ArgumentParser(description="Multi-UAV MEC Autonomous Research Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "plot", "ablation", "all"], default="all",
                        help="Execution mode: train, evaluate, plot, ablation, or all")
    parser.add_argument("--model", type=str, choices=["all", "mappo", "ippo", "maddpg", "cppo"], default="all",
                        help="Model to train (all, mappo, ippo, maddpg, cppo)")
    parser.add_argument("--ablation", type=str, choices=["no_energy", "no_safety", "fixed_offload"], default="no_energy",
                        help="Ablation type when mode=ablation")
    parser.add_argument("--episodes", type=int, default=80, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_test_seeds", type=int, default=5, help="Number of held-out test seeds for evaluation")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                        help="Compute device: auto (CUDA if available, else CPU), cuda, or cpu")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if args.mode in ["train", "all"]:
        print("\n>>> STEP 1: TRAINING MULTI-AGENT RL POLICIES <<<")
        if args.model == "all":
            train_all(num_episodes=args.episodes, seed=args.seed, device=args.device)
        elif args.model == "mappo":
            train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device)
        elif args.model == "ippo":
            train_ippo(num_episodes=args.episodes, seed=args.seed, device=args.device)
        elif args.model == "maddpg":
            train_maddpg(num_episodes=args.episodes, seed=args.seed, device=args.device)
        elif args.model == "cppo":
            train_cppo(num_episodes=args.episodes, seed=args.seed, device=args.device)

    if args.mode == "ablation":
        print(f"\n>>> RUNNING ABLATION STUDY: {args.ablation} <<<")
        train_mappo(num_episodes=args.episodes, seed=args.seed, device=args.device, ablation=args.ablation)

    if args.mode in ["evaluate", "all"]:
        print("\n>>> STEP 2: MULTI-MODEL BENCHMARK EVALUATION (LEARNED & CLASSICAL BASELINES) <<<")
        test_seeds = [101 + i for i in range(args.num_test_seeds)]
        run_evaluation(test_seeds=test_seeds, device=args.device)

    if args.mode in ["plot", "all"]:
        print("\n>>> STEP 3: REGENERATING ALL PUBLICATION FIGURES <<<")
        regenerate_all_figures()
        print(f"\n[Complete] Pipeline finished successfully. Figures saved in {fig_dir}.")

if __name__ == "__main__":
    main()
