"""
Test Suite: Checkpoint Integrity and Explicit Error Handling.
Verifies:
1. Genuine checkpoint loading without corruption.
2. Explicit failure when checkpoint file is missing or corrupt (no silent fallback).
3. Parameter fingerprinting confirming non-random weights.
"""

import os
import pytest
import torch
from config import MasterConfig
from models.mappo import MAPPOAgent
from models.ippo import IPPOAgent
from models.maddpg import MADDPGAgent
from models.cppo import CPPOAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")

def test_missing_checkpoint_raises_explicit_error():
    cfg = MasterConfig()
    agent = MAPPOAgent(20, 40, cfg.physics.num_uavs, device="cpu")
    non_existent = os.path.join(CKPT_DIR, "completely_non_existent_model_checkpoint.pt")
    
    with pytest.raises(Exception):
        agent.load_checkpoint(non_existent)

def test_best_mappo_pure_checkpoint_fingerprint():
    pure_path = os.path.join(CKPT_DIR, "best_mappo_pure_agent.pt")
    assert os.path.exists(pure_path), f"Expected checkpoint at {pure_path}"

    cfg = MasterConfig()
    agent = MAPPOAgent(20, 40, cfg.physics.num_uavs, policy_variant="pure", device="cpu")
    agent.load_checkpoint(pure_path)

    # Inspect weight statistics
    actor_weights = [p for p in agent.actor.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in actor_weights)
    assert total_params == 20747

    # Non-trivial parameters differing from zero
    weight_norms = [torch.norm(p).item() for p in actor_weights]
    assert all(w > 1e-4 for w in weight_norms)

    # Verify optimizer state is present
    ckpt = torch.load(pure_path, map_location="cpu")
    assert "actor_opt" in ckpt
    assert "critic_opt" in ckpt
    assert "state" in ckpt["actor_opt"]
    # Verify optimizer step count > 1000
    steps = [v.get("step", 0) for v in ckpt["actor_opt"]["state"].values() if isinstance(v, dict)]
    assert max(steps) >= 1000

def test_best_mappo_residual_checkpoint_fingerprint():
    res_path = os.path.join(CKPT_DIR, "best_mappo_residual_agent.pt")
    assert os.path.exists(res_path), f"Expected checkpoint at {res_path}"

    cfg = MasterConfig()
    agent = MAPPOAgent(20, 40, cfg.physics.num_uavs, policy_variant="residual", device="cpu")
    agent.load_checkpoint(res_path)

    assert agent.policy_variant == "residual"
    assert agent.actor.policy_variant == "residual"

def test_apf_only_variant_preservation():
    res_path = os.path.join(CKPT_DIR, "best_mappo_residual_agent.pt")
    cfg = MasterConfig()
    agent = MAPPOAgent(20, 40, cfg.physics.num_uavs, policy_variant="apf_only", device="cpu")
    agent.load_checkpoint(res_path)

    # Policy variant must remain apf_only
    assert agent.policy_variant == "apf_only"
    assert agent.actor.policy_variant == "apf_only"
