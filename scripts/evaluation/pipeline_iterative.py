#!/usr/bin/env python3
"""
Full Pipeline Iterative Runner.

Orchestrates multiple iterations of:
  1. Trajectory Generation (with different seeds)
  2. Segment Assignment (ALL CONFIGS)
  3. Execution

Collects metrics across all iterations to evaluate robustness.

Usage:
    python scripts/evaluation/pipeline_iterative.py --experiments mug_inversion --iterations 5 --workers 4
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import json
import numpy as np

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import PAPER_EXPERIMENTS, ALL_ABLATIONS, config_id, ALL_CONFIGS
from scripts.evaluation.results_store import load_result, RESULTS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

TRAJ_GEN_SCRIPT = "scripts/pipeline/trajectory_generation.py"
RUN_ALL_SCRIPT = "scripts/evaluation/run_all.py"

def run_command(cmd, description):
    logger.info(f"Running {description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"{description} FAILED with exit code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Iterative pipeline runner")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiments to run (default: all in experiments/)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations per experiment")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for parallel processing")
    parser.add_argument("--paper", action="store_true",
                        help="Run paper configurations only")
    args = parser.parse_args()

    if args.experiments:
        experiments = args.experiments
    else:
        # All .json files in experiments/
        exp_dir = Path(_PROJECT_ROOT) / "experiments"
        experiments = [f.stem for f in exp_dir.glob("*.json")]
    
    iters = args.iterations
    
    for i in range(iters):
        logger.info(f"\n{'='*80}\n  ITERATION {i+1}/{iters}\n{'='*80}")
        
        # 1. Trajectory Generation (one by one for each experiment)
        for exp in experiments:
            logger.info(f"--- Generating trajectories for {exp} (Iter {i+1}) ---")
            # For each iteration, we MUST clear the old data because run_all.py expects it in data/<exp>
            # Actually, to be truly iterative, we should generate new data each time.
            gen_cmd = [
                sys.executable, TRAJ_GEN_SCRIPT,
                "--experiment", f"experiments/{exp}.json",
                "--save-traj",
                "--no-meshcat",
                "--workers", str(args.workers),
                "--seed", str(42 + i)
            ]
            if not run_command(gen_cmd, f"Trajectory Generation {exp} Iter {i+1}"):
                continue

        # 2 & 3. Assignment and Execution (ALL CONFIGS for the given experiments)
        logger.info(f"--- Running Assignment and Execution (Iter {i+1}) ---")
        run_all_cmd = [
            sys.executable, RUN_ALL_SCRIPT,
            "--only"
        ] + experiments + [
            "--execute",
            "--workers", str(args.workers),
            "--results-dir", f"results/iter_{i+1}"
        ]
        if args.paper:
            run_all_cmd.append("--paper")
        # Else it runs ALL_CONFIGS by default in run_all.py if no ablations specified
            
        if not run_command(run_all_cmd, f"Batch Runner Iter {i+1}"):
            continue

    # 4. Aggregation
    logger.info(f"\n{'='*80}\n  FINAL AGGREGATION AND ANALYSIS\n{'='*80}")
    
    # We'll use a nested dict: summary[experiment][config_label] = stats
    summary = {}
    
    for exp in experiments:
        summary[exp] = {}
        for i in range(iters):
            iter_dir = Path(_PROJECT_ROOT) / "results" / f"iter_{i+1}" / exp
            if not iter_dir.exists(): continue
            
            for json_file in iter_dir.glob("*.json"):
                if json_file.name == "summary.json": continue
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    cfg_label = data["config_label"]
                    if cfg_label not in summary[exp]:
                        summary[exp][cfg_label] = {
                            "successes": 0,
                            "total_iters": 0,
                            "costs": [],
                            "handovers": [],
                            "mean_tracking_errors": [],
                            "max_tracking_errors": [],
                            "manipulabilities": [],
                            "collisions": []
                        }
                    
                    stats = summary[exp][cfg_label]
                    stats["total_iters"] += 1
                    
                    p = data.get("planning")
                    e = data.get("execution")
                    
                    if p and p.get("total_cost") < 1e10:
                        stats["costs"].append(p["total_cost"])
                        stats["handovers"].append(p["n_handovers"])
                    
                    if e:
                        if e.get("task_success"):
                            stats["successes"] += 1
                        stats["mean_tracking_errors"].append(e.get("object_mean_tracking_error", 0))
                        stats["max_tracking_errors"].append(e.get("object_max_tracking_error", 0))
                        stats["manipulabilities"].append(e.get("mean_manipulability", 0))
                        stats["collisions"].append(e.get("n_collision_incidents", 0))
                        
                except Exception as e:
                    logger.warning(f"Failed to load result from {json_file}: {e}")

    # Print Summary Table
    print(f"\n{'Experiment':<30} {'Config':<25} {'Succ %':>8} {'Cost':>10} {'HOs':>6} {'ErrAvg':>8} {'ErrMax':>8} {'Coll':>5}")
    print("-" * 115)
    
    for exp in sorted(summary.keys()):
        for cfg in sorted(summary[exp].keys()):
            stats = summary[exp][cfg]
            if stats["total_iters"] == 0: continue
            
            sr = (stats["successes"] / stats["total_iters"]) * 100
            m_cost = np.mean(stats["costs"]) if stats["costs"] else float('nan')
            m_hos = np.mean(stats["handovers"]) if stats["handovers"] else float('nan')
            m_err = np.mean(stats["mean_tracking_errors"]) if stats["mean_tracking_errors"] else 0.0
            max_err = np.max(stats["max_tracking_errors"]) if stats["max_tracking_errors"] else 0.0
            m_coll = np.mean(stats["collisions"]) if stats["collisions"] else 0.0
            
            print(f"{exp:<30} {cfg:<25} {sr:>7.1f}% {m_cost:>10.1f} {m_hos:>6.1f} {m_err:>8.4f} {max_err:>8.4f} {m_coll:>5.1f}")

    # Optional: Save aggregate summary to JSON
    with open(Path(_PROJECT_ROOT) / "results" / "aggregate_summary.json", 'w') as f:
        # Need to convert numpy types for JSON
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            return str(obj)
        json.dump(summary, f, indent=2, default=default_serializer)

if __name__ == "__main__":
    main()
