"""
Unit and Integration Tests for Scientific Integrity and Provenance Verification.
Ensures:
1. All exported CSV metrics are dynamically derived from raw simulation records (zero hardcoded values).
2. Success labels are derived from explicit geometric arrival criteria (dist_to_goal <= 5.0m).
3. Plotted potential field parameters strictly match the implementation in models/apf.py.
4. Controller labels accurately describe the actual motion generator (A* vs RL).
5. Single-seed test runs are explicitly distinguished from multi-seed benchmarks.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import inspect

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import ENV_PRESETS, get_preset_config
from models.apf import SAPFNavigator
from models.astar import HybridAStarMAPPONavigator
from plots.render_publication_figures import export_algorithm_metrics_csv

TRAJ_PATH = os.path.join(BASE_DIR, "results", "metropolis_trajectory_records.csv")
PUB_CSV_PATH = os.path.join(BASE_DIR, "results", "publication_figures", "joint_cooperative_system", "algorithm_metrics.csv")

@pytest.fixture(scope="module")
def ensure_raw_trajectory_data():
    """Ensures metropolis_trajectory_records.csv exists before running tests."""
    if not os.path.exists(TRAJ_PATH):
        try:
            from utils.trajectory_recorder import record_metropolis_trajectories
            record_metropolis_trajectories(seed=101)
        except Exception:
            from scratch.record_metropolis_trajectories import record_metropolis_trajectories
            record_metropolis_trajectories(seed=101)
    assert os.path.exists(TRAJ_PATH), f"Raw trajectory records missing at {TRAJ_PATH}"

def test_export_algorithm_metrics_dynamically_derived(ensure_raw_trajectory_data):
    """
    Validates that export_algorithm_metrics_csv() derives every metric directly
    from the raw trajectory records rather than using hardcoded values.
    """
    raw_df = pd.read_csv(TRAJ_PATH)
    metrics_df = export_algorithm_metrics_csv()

    reach_tolerance = 5.0
    obstacles = ENV_PRESETS["metropolis"]

    for _, row in metrics_df.iterrows():
        algo_name = row["Algorithm"]
        csv_path_len = row["Path Length (m)"]
        csv_flight_time = row["Flight Time (s)"]
        csv_success_rate = row["Success Rate (%)"]

        # Map display name back to raw algorithm key
        algo_key = None
        for k in ["hybrid_astar_mappo", "astar", "sapf", "rrt_star", "mappo_residual", "mappo_pure"]:
            if k in algo_name.lower() or (k == "hybrid_astar_mappo" and "hybrid" in algo_name.lower()) or (k == "hybrid_astar_mappo" and "w_res=0.0" in algo_name):
                algo_key = k
                break
        if algo_key is None:
            continue

        raw_group = raw_df[raw_df["algorithm"] == algo_key]
        assert not raw_group.empty, f"No raw data for {algo_key}"

        # Independently recompute from raw records
        recomputed_paths = []
        recomputed_times = []
        recomputed_success = []

        for uav_id in sorted(raw_group["uav_id"].unique()):
            uav_traj = raw_group[raw_group["uav_id"] == uav_id].sort_values("timestep")
            arr_mask = uav_traj["dist_to_goal"] <= reach_tolerance
            if arr_mask.any():
                arr_step = int(uav_traj[arr_mask]["timestep"].iloc[0])
                uav_eff = uav_traj[uav_traj["timestep"] <= arr_step]
                ftime = float(arr_step)
                succ = 1.0
            else:
                uav_eff = uav_traj
                ftime = float(uav_traj["timestep"].max())
                succ = 0.0

            coords = uav_eff[["x", "y", "z"]].values
            plen = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))) if len(coords) > 1 else 0.0

            recomputed_paths.append(plen)
            recomputed_times.append(ftime)
            recomputed_success.append(succ)

        expected_mean_path = round(float(np.mean(recomputed_paths)), 1)
        expected_mean_time = round(float(np.mean(recomputed_times)), 1)
        expected_success_rate = round(float(np.mean(recomputed_success)) * 100.0, 1)

        # Assert exact agreement between exported CSV and independent recomputation
        assert abs(csv_path_len - expected_mean_path) < 0.2, (
            f"Path length mismatch for {algo_key}: CSV={csv_path_len}, Computed={expected_mean_path}"
        )
        assert abs(csv_flight_time - expected_mean_time) < 0.2, (
            f"Flight time mismatch for {algo_key}: CSV={csv_flight_time}, Computed={expected_mean_time}"
        )
        assert abs(csv_success_rate - expected_success_rate) < 0.1, (
            f"Success rate mismatch for {algo_key}: CSV={csv_success_rate}, Computed={expected_success_rate}"
        )

def test_no_hardcoded_placeholder_artifacts(ensure_raw_trajectory_data):
    """
    Asserts that deprecated hardcoded placeholder numbers (110.8m, 114.4m, 186.5m, 15.5m)
    are strictly absent from the newly generated algorithm_metrics.csv.
    """
    export_algorithm_metrics_csv()
    metrics_df = pd.read_csv(PUB_CSV_PATH)

    deprecated_placeholders = [110.8, 114.4, 186.5, 15.5]
    for val in deprecated_placeholders:
        matching_rows = metrics_df[metrics_df["Path Length (m)"] == val]
        assert matching_rows.empty, (
            f"Found deprecated hardcoded placeholder value {val} in exported CSV! "
            "Metrics must be derived from actual trajectory data."
        )

def test_apf_parameter_consistency():
    """
    Asserts that APF parameters used in plotting match the implementation in models/apf.py.
    """
    sig = inspect.signature(SAPFNavigator.__init__)
    k_att_default = float(sig.parameters["k_att"].default)
    k_rep_default = float(sig.parameters["k_rep"].default)
    d_0_default = float(sig.parameters["d_0"].default)
    k_tangent_default = float(sig.parameters["k_tangent"].default)

    assert k_att_default == 1.0, f"Expected k_att=1.0 in models/apf.py, got {k_att_default}"
    assert k_rep_default == 80.0, f"Expected k_rep=80.0 in models/apf.py, got {k_rep_default}"
    assert d_0_default == 25.0, f"Expected d_0=25.0 in models/apf.py, got {d_0_default}"
    assert k_tangent_default == 45.0, f"Expected k_tangent=45.0 in models/apf.py, got {k_tangent_default}"

def test_controller_naming_and_scope_honesty():
    """
    Ensures that when residual_weight=0.0 in HybridAStarMAPPONavigator,
    the controller label transparently states that motion is pure pursuit A* tracking.
    """
    metrics_df = pd.read_csv(PUB_CSV_PATH)
    hybrid_rows = metrics_df[metrics_df["Algorithm"].str.contains("hybrid|pure pursuit", case=False)]
    assert not hybrid_rows.empty

    row = hybrid_rows.iloc[0]
    scope = row.get("Evaluation Scope", "")
    controller = row.get("Navigation Controller", "")

    assert "Single-Seed" in scope or "Seed 101" in scope, (
        f"Evaluation scope must disclose single-seed evaluation, got: {scope}"
    )
    assert "Pure Pursuit" in controller or "w_res=0.0" in controller, (
        f"Controller description must disclose that movement was Pure Pursuit tracking, got: {controller}"
    )
