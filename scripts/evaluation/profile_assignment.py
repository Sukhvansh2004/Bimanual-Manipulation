#!/usr/bin/env python3
"""Profile the assignment solver with cProfile to find bottlenecks.

Delegates to ``segment_assignment.py --profile`` so that all assignment
logic lives in one place (DRY).  This script resolves the ablation config
→ CLI args mapping and adds a few profiling-specific options.

Usage::

    # Profile the Proposed config on ycb_plate_table_to_shelf (default):
    python scripts/evaluation/profile_assignment.py

    # Profile a different experiment:
    python scripts/evaluation/profile_assignment.py --experiment mug_inversion

    # Profile a specific config (by label):
    python scripts/evaluation/profile_assignment.py --config Greedy

    # Custom output path for the .prof binary:
    python scripts/evaluation/profile_assignment.py --output results/my_profile.prof
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import (
    PROPOSED, GREEDY, SINGLE_ROBOT, NO_SWITCH_PENALTY, NO_MANIPULABILITY,
    NO_DIVERSITY, config_id,
)

_ASSIGNMENT_SCRIPT = str(_PROJECT_ROOT / "scripts" / "pipeline" / "segment_assignment.py")
EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
DATA_DIR = _PROJECT_ROOT / "data"

CONFIG_MAP = {
    "Proposed": PROPOSED,
    "Greedy": GREEDY,
    "Single Robot": SINGLE_ROBOT,
    "No Diversity": NO_DIVERSITY,
    "No Switch Penalty": NO_SWITCH_PENALTY,
    "No Manipulability": NO_MANIPULABILITY,
}


def main():
    parser = argparse.ArgumentParser(description="Profile assignment solver with cProfile")
    parser.add_argument("--experiment", type=str, default="ycb_plate_table_to_shelf",
                        help="Experiment name (default: ycb_plate_table_to_shelf)")
    parser.add_argument("--config", type=str, default="Proposed",
                        help="Config label: Proposed, Greedy, 'Single Robot', 'No Diversity', "
                             "'No Switch Penalty', 'No Manipulability'")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .prof file path (default: results/profile_<exp>_<config>.prof)")
    args = parser.parse_args()

    # Resolve experiment
    exp_json = str(EXPERIMENTS_DIR / f"{args.experiment}.json")
    if not os.path.isfile(exp_json):
        print(f"ERROR: Experiment config not found: {exp_json}")
        sys.exit(1)

    # Resolve config
    cfg = CONFIG_MAP.get(args.config)
    if cfg is None:
        print(f"ERROR: Unknown config '{args.config}'. Available: {list(CONFIG_MAP.keys())}")
        sys.exit(1)

    cid = config_id(cfg)

    # Resolve data directory
    base_data_dir = str(DATA_DIR / args.experiment)
    if not cfg.get("use_dpp", True):
        data_dir = base_data_dir + "_no_dpp"
    else:
        data_dir = base_data_dir

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Output paths
    prof_path = args.output or str(
        _PROJECT_ROOT / "results" / f"profile_{args.experiment}_{cid}.prof"
    )
    os.makedirs(os.path.dirname(prof_path), exist_ok=True)

    output_dir = str(Path(data_dir) / "configs" / cid)

    # --- Build command ---
    cmd = [
        sys.executable, _ASSIGNMENT_SCRIPT,
        "--experiment", exp_json,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--no-plot",
        "--profile",
        "--profile-output", prof_path,
    ]

    # Config parameter overrides
    if cfg["alpha"] is not None:
        cmd.extend(["--alpha", str(cfg["alpha"])])
    if cfg["beta"] is not None:
        cmd.extend(["--beta", str(cfg["beta"])])
    if cfg["beam_width"] is not None:
        cmd.extend(["--beam-width", str(cfg["beam_width"])])
    if cfg["handover_stride"] != 1:
        cmd.extend(["--handover-stride", str(cfg["handover_stride"])])
    if cfg["clearance_radius"] is not None:
        cmd.extend(["--clearance-radius", str(cfg["clearance_radius"])])
    if cfg["disable_feasibility_filter"]:
        cmd.append("--disable-feasibility-filter")
    if cfg["switch_cost_override"] is not None:
        cmd.extend(["--switch-cost-override", str(cfg["switch_cost_override"])])
    if not cfg["use_ik"]:
        cmd.append("--no-env")
    if args.debug:
        cmd.append("--debug")

    # --- Run ---
    print(f"\n{'='*70}")
    print(f"  PROFILING: {args.experiment} × {args.config}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Profile output: {prof_path}")
    print()

    proc = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    if proc.returncode != 0:
        print(f"\nERROR: segment_assignment.py exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    if os.path.isfile(prof_path):
        print(f"\nBinary profile saved to: {prof_path}")
        print(f"  → View with: python -m snakeviz {prof_path}")
        print(f"  → Or:        python -m pstats {prof_path}")


if __name__ == "__main__":
    main()
