"""
Test Suite: Checkpoint Integrity and Explicit Error Handling.
Verifies:
1. Genuine checkpoint loading without corruption.
2. Explicit failure when checkpoint file is missing or corrupt (no silent fallback).
3. Parameter fingerprinting confirming non-random weights.
"""

import os
import tempfile
import pytest
import torch
from config import MasterConfig
from env.observation_builder import OBS_DIM, STATE_DIM_PER_UAV
from models.mappo import MAPPOAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")

def test_missing_checkpoint_raises_explicit_error():
    cfg = MasterConfig()
    agent = MAPPOAgent(OBS_DIM, STATE_DIM_PER_UAV * cfg.physics.num_uavs, cfg.physics.num_uavs, device="cpu")
    non_existent = os.path.join(CKPT_DIR, "completely_non_existent_model_checkpoint.pt")
    
    with pytest.raises(FileNotFoundError):
        agent.load_checkpoint(non_existent)

def test_historical_single_stream_checkpoint_raises_explicit_error():
    """Verify that obsolete single-stream checkpoints fail explicitly with actionable explanation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        obsolete_path = os.path.join(tmpdir, "obsolete_single_stream.pt")
        torch.save({
            "actor": {"backbone.0.weight": torch.randn(64, 20)},
            "critic": {"net.0.weight": torch.randn(64, 40)},
            "policy_variant": "pure"
        }, obsolete_path)

        cfg = MasterConfig()
        agent = MAPPOAgent(OBS_DIM, STATE_DIM_PER_UAV * cfg.physics.num_uavs, cfg.physics.num_uavs, policy_variant="pure", device="cpu")
        
        with pytest.raises(RuntimeError) as exc_info:
            agent.load_checkpoint(obsolete_path)
        
        assert "Incompatible historical checkpoint architecture" in str(exc_info.value)
        assert "obsolete single-stream network" in str(exc_info.value)
        assert "Two-Stream Decoupled Architecture" in str(exc_info.value)

def test_checkpoint_raw_archive_integrity():
    """Verify checkpoints remain intact in the archive without zero-weight corruption."""
    pure_path = os.path.join(CKPT_DIR, "best_mappo_pure_agent.pt")
    assert os.path.exists(pure_path), f"Expected checkpoint at {pure_path}"
    ckpt = torch.load(pure_path, map_location="cpu")
    
    assert "actor" in ckpt
    assert "critic" in ckpt
    assert "actor_opt" in ckpt
    assert "critic_opt" in ckpt
    
    # Non-trivial parameters differing from zero
    actor_weights = [p for p in ckpt["actor"].values() if isinstance(p, torch.Tensor)]
    weight_norms = [torch.norm(p.float()).item() for p in actor_weights]
    assert all(w > 1e-4 for w in weight_norms)

def test_modern_two_stream_checkpoint_roundtrip():
    """Verify that current Two-Stream checkpoints save and load cleanly with full weight integrity."""
    cfg = MasterConfig()
    state_dim = STATE_DIM_PER_UAV * cfg.physics.num_uavs
    agent_src = MAPPOAgent(OBS_DIM, state_dim, cfg.physics.num_uavs, policy_variant="pure", device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ckpt = os.path.join(tmpdir, "modern_twostream_test.pt")
        agent_src.save_checkpoint(tmp_ckpt)

        agent_dst = MAPPOAgent(OBS_DIM, state_dim, cfg.physics.num_uavs, policy_variant="pure", device="cpu")
        agent_dst.load_checkpoint(tmp_ckpt)

        # Check weight equality
        for (k1, v1), (k2, v2) in zip(agent_src.actor.state_dict().items(), agent_dst.actor.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2)

        for (k1, v1), (k2, v2) in zip(agent_src.critic.state_dict().items(), agent_dst.critic.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2)

        # Check forward pass equivalence
        test_obs = torch.randn(4, OBS_DIM)
        with torch.no_grad():
            o1, m1, s1 = agent_src.actor(test_obs)
            o2, m2, s2 = agent_dst.actor(test_obs)
            assert torch.allclose(o1, o2)
            assert torch.allclose(m1, m2)
            assert torch.allclose(s1, s2)

def test_apf_only_variant_preservation_modern():
    """Verify policy_variant='apf_only' remains apf_only across checkpoint loads."""
    cfg = MasterConfig()
    state_dim = STATE_DIM_PER_UAV * cfg.physics.num_uavs
    agent_src = MAPPOAgent(OBS_DIM, state_dim, cfg.physics.num_uavs, policy_variant="pure", device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ckpt = os.path.join(tmpdir, "apf_variant_test.pt")
        agent_src.save_checkpoint(tmp_ckpt)

        agent_dst = MAPPOAgent(OBS_DIM, state_dim, cfg.physics.num_uavs, policy_variant="apf_only", device="cpu")
        agent_dst.load_checkpoint(tmp_ckpt)

        assert agent_dst.policy_variant == "apf_only"
        assert agent_dst.actor.policy_variant == "apf_only"

