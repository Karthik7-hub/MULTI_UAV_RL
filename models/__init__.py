"""
# =============================================================================
# MULTI-UAV REINFORCEMENT LEARNING & GEOMETRIC PATH PLANNING MODELS
# =============================================================================
# This package implements the complete algorithmic suite evaluated in the research:
#
# 1. Multi-Agent Reinforcement Learning (MARL):
#    - MAPPO: Centralized Training & Decentralized Execution (CTDE) (Proposed)
#    - IPPO:  Independent PPO (Decentralized Training & Execution / DTDE)
#    - CPPO:  Centralized PPO (Centralized Training & Execution / CTCE Upper Bound)
#    - MADDPG: Off-policy Multi-Agent DDPG with continuous actor and Gumbel-Softmax
#
# 2. Geometric & Sampling Path Planners:
#    - AStar3DPlanner: Deterministic 26-connected 3D grid graph search
#    - RRTStar3DPlanner: Asymptotically optimal sampling-based tree planner
#    - SAPFNavigator: Sequential Adaptive Potential Field with circulation vortex
#
# 3. Hierarchical Hybrid Architecture:
#    - HybridAStarMAPPONavigator: High-level 3D A* global corridor planning +
#      low-level 3D Pure Pursuit tracking and decentralized MAPPO MEC offloading
#
# 4. Classical Rule-Based MEC Offloading Heuristics:
#    - EnergyDeadlineAwareGreedyPolicy, LowestLatencyPolicy, NearestFeasiblePolicy,
#      LocalOnlyPolicy, EdgeOnlyPolicy, RandomFeasiblePolicy
# =============================================================================
"""

from models.mappo import MAPPOAgent, ActorNetwork, CriticNetwork, RolloutBuffer
from models.astar import AStarBaseline, AStar3DPlanner, HybridAStarMAPPONavigator
from models.ippo import IPPOAgent, LocalCriticNetwork
from models.maddpg import MADDPGAgent, MADDPGActor, MADDPGCritic, ReplayBuffer
from models.cppo import CPPOAgent
from models.rrt_star import RRTStarBaseline, RRTStar3DPlanner
from models.apf import SAPFBaseline, SAPFNavigator
from models.baselines import (
    AStarJointBaseline,
    EnergyDeadlineAwareGreedyPolicy,
    LowestLatencyPolicy,
    NearestFeasiblePolicy,
    LocalOnlyPolicy,
    EdgeOnlyPolicy,
    RandomFeasiblePolicy
)

__all__ = [
    "MAPPOAgent",
    "ActorNetwork",
    "CriticNetwork",
    "RolloutBuffer",
    "AStarBaseline",
    "AStar3DPlanner",
    "HybridAStarMAPPONavigator",
    "IPPOAgent",
    "LocalCriticNetwork",
    "MADDPGAgent",
    "MADDPGActor",
    "MADDPGCritic",
    "ReplayBuffer",
    "CPPOAgent",
    "RRTStarBaseline",
    "RRTStar3DPlanner",
    "SAPFBaseline",
    "SAPFNavigator",
    "AStarJointBaseline",
    "EnergyDeadlineAwareGreedyPolicy",
    "LowestLatencyPolicy",
    "NearestFeasiblePolicy",
    "LocalOnlyPolicy",
    "EdgeOnlyPolicy",
    "RandomFeasiblePolicy"
]

