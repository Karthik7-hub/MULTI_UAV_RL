"""
3D UAV Path-Planning Research Simulator - REST API Backend Server
Serving WebGL Three.js client dashboard and exposing endpoints:
- GET  /api/world             -> Airspace bounds, default endpoints, obstacles
- GET  /api/presets           -> Scenario presets (Benchmark, Metropolis, Canyon, etc.)
- POST /api/random_endpoints  -> Generate collision-free random start & target
- POST /api/plan              -> Simulate trajectory & telemetry for any planner
- POST /api/compare           -> Run matched multi-algorithm benchmark comparison
"""

import os
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import MasterConfig, ENV_PRESETS
from simulator.flight_engine import FlightEngine

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "simulator", "static"))

# Initialize global configuration and modular flight simulation engine
cfg = MasterConfig()
engine = FlightEngine(cfg=cfg, device_name="cpu")

# Backward-compatible function aliases for test suites
compute_min_clearance = engine.compute_min_clearance
make_obs_vector = engine.make_obs_vector

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/world", methods=["GET"])
def get_world():
    """Returns default airspace geometry, default obstacles, and mission bounds."""
    return jsonify({
        "bounds": {
            "x_min": cfg.airspace.x_min, "x_max": cfg.airspace.x_max,
            "y_min": cfg.airspace.y_min, "y_max": cfg.airspace.y_max,
            "z_min": cfg.airspace.z_min, "z_max": cfg.airspace.z_max
        },
        "obstacles": [
            {"id": i, "type": "cylinder", "cx": obs[0], "cy": obs[1], "radius": obs[2], "height": obs[3]}
            for i, obs in enumerate(cfg.airspace.obstacles)
        ],
        "default_source": [20.0, 20.0, 10.0],
        "default_target": [180.0, 180.0, 10.0],
        "safety_margin": 2.0,
        "uav_radius": cfg.physics.d_safe / 2.0,
        "v_max": cfg.physics.v_max
    })

@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns available airspace scenario presets."""
    preset_data = {}
    for name, obs_list in ENV_PRESETS.items():
        preset_data[name] = [
            {"id": i, "type": "cylinder", "cx": o[0], "cy": o[1], "radius": o[2], "height": o[3], "enabled": True}
            for i, o in enumerate(obs_list)
        ]
    return jsonify(preset_data)

@app.route("/api/random_endpoints", methods=["POST", "GET"])
def random_endpoints():
    """Generates random, valid, non-colliding origin and target coordinates."""
    data = request.json or {}
    raw_obs = data.get("obstacles", cfg.airspace.obstacles)
    margin = float(data.get("margin", 2.0))
    result = engine.generate_random_endpoints(raw_obs, margin)
    return jsonify(result)

@app.route("/api/plan", methods=["POST"])
def plan_path():
    """
    Executes specified path planning or RL algorithm from start to target.
    Delegates to modular FlightEngine.
    """
    data = request.json or {}
    planner_key = data.get("planner", "astar")
    start = np.array(data.get("start", [20.0, 20.0, 10.0]), dtype=np.float32)
    target = np.array(data.get("target", [180.0, 180.0, 10.0]), dtype=np.float32)
    raw_obs = data.get("obstacles", [])
    margin = float(data.get("margin", 2.0))
    allow_untrained = bool(data.get("allow_untrained", False))

    try:
        result = engine.simulate_flight(
            planner_key=planner_key,
            start=start,
            target=target,
            obstacles=raw_obs,
            margin=margin,
            seed=seed,
            allow_untrained=allow_untrained
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "ERROR", "error": f"Simulation failure: {e}"}), 500

@app.route("/api/model_status", methods=["GET"])
def model_status():
    """Returns status and integrity diagnostics for all registered RL models."""
    return jsonify({
        "status": "OK",
        "agent_status": engine.agent_status,
        "mode": "single_uav_illustrative_inference",
        "scope_disclosure": (
            "Browser flight visualizer demonstrates illustrative single-UAV inference. "
            "It does not evaluate multi-agent collision avoidance, cooperative critic behavior, "
            "or joint MEC queue dynamics. For full multi-agent benchmarks, execute evaluate.py."
        )
    })

@app.route("/api/compare", methods=["POST"])
def compare_planners():
    """
    Executes side-by-side benchmark comparison on identical world layout.
    Delegates to modular FlightEngine.
    """
    data = request.json or {}
    start = np.array(data.get("start", [20.0, 20.0, 30.0]), dtype=np.float32)
    target = np.array(data.get("target", [180.0, 180.0, 30.0]), dtype=np.float32)
    raw_obs = data.get("obstacles", [])
    margin = float(data.get("margin", 2.0))
    selected = data.get("selected_planners", None)

    try:
        comparison_results = engine.compare_flights(
            start=start,
            target=target,
            obstacles=raw_obs,
            margin=margin,
            selected=selected
        )
        return jsonify({"comparison": comparison_results})
    except Exception as e:
        return jsonify({"status": "ERROR", "error": f"Comparison failure: {e}"}), 500

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" 3D UAV PATH-PLANNING RESEARCH SIMULATOR BACKEND")
    print(" Serving WebGL Dashboard at: http://127.0.0.1:5000")
    print("=" * 80)
    app.run(host="127.0.0.1", port=5000, debug=False)
