#!/usr/bin/env python3
"""
Full Pipeline Iterative Runner.

Orchestrates multiple iterations of:
  1. Trajectory Generation (with different seeds)
  2. Segment Assignment
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

from scripts.evaluation.configs import PAPER_EXPERIMENTS, ALL_ABLATIONS, config_id
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
    parser.add_argument("--experiments", nargs="+", default=PAPER_EXPERIMENTS,
                        help="Experiments to run")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations per experiment")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for parallel processing")
    parser.add_argument("--paper", action="store_true",
                        help="Run paper configurations")
    parser.add_argument("--ablations", nargs="*", default=["baseline"],
                        help="Ablation groups to run (if not --paper)")
    args = parser.parse_args()

    experiments = args.experiments
    iters = args.iterations
    
    overall_results = {}

    for i in range(iters):
        logger.info(f"\n{'='*80}\n  ITERATION {i+1}/{iters}\n{'='*80}")
        
        # 1. Trajectory Generation
        for exp in experiments:
            # We use iteration-specific data directory to avoid overwriting and allow parallel assignment later if needed
            # However, run_all.py expects data in data/<exp>. 
            # We can either patch experiments JSON on the fly or just run sequentially and collect.
            # To keep it simple and compatible with existing run_all.py, we'll run iterations one by one.
            
            logger.info(f"--- Generating trajectories for {exp} (Iter {i+1}) ---")
            gen_cmd = [
                sys.executable, TRAJ_GEN_SCRIPT,
                "--experiment", f"experiments/{exp}.json",
                "--save-traj",
                "--no-meshcat",
                "--workers", str(args.workers),
                "--seed", str(42 + i) # Use deterministic but different seeds per iteration
            ]
            if not run_command(gen_cmd, f"Trajectory Generation {exp} Iter {i+1}"):
                continue

        # 2 & 3. Assignment and Execution
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
        else:
            run_all_cmd.extend(["--ablations"] + args.ablations)
            
        if not run_command(run_all_cmd, f"Batch Runner Iter {i+1}"):
            continue

    # 4. Aggregation
    logger.info(f"\n{'='*80}\n  FINAL AGGREGATION\n{'='*80}")
    
    summary = {}
    for exp in experiments:
        summary[exp] = {
            "successes": 0,
            "total_iters": 0,
            "metrics": []
        }
        for i in range(iters):
            # Find the config results. If --paper, we probably care about 'Proposed'.
            # If multiple ablations, it's more complex. For now, let's assume we want the first ablation's result.
            iter_results_dir = Path(_PROJECT_ROOT) / "results" / f"iter_{i+1}" / exp
            if not iter_results_dir.exists():
                continue
            
            for json_file in iter_results_dir.glob("*.json"):
                if json_file.name == "summary.json": continue
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    summary[exp]["total_iters"] += 1
                    exec_data = data.get("execution")
                    if exec_data and exec_data.get("task_success"):
                        summary[exp]["successes"] += 1
                    
                    summary[exp]["metrics"].append(data)
                except Exception as e:
                    logger.warning(f"Failed to load result from {json_file}: {e}")

    # Print Summary
    print(f"\n{'Experiment':<40} {'Success Rate':>12} {'Mean Cost':>12} {'Mean HOs':>10}")
    print("-" * 80)
    for exp, data in summary.items():
        if data["total_iters"] == 0:
            print(f"{exp:<40} {'N/A':>12}")
            continue
            
        sr = (data["successes"] / data["total_iters"]) * 100
        costs = [m["planning"]["total_cost"] for m in data["metrics"] if m.get("planning") and m["planning"]["total_cost"] < float('inf')]
        hos = [m["planning"]["n_handovers"] for m in data["metrics"] if m.get("planning") and m["planning"]["total_cost"] < float('inf')]
        
        mean_cost = np.mean(costs) if costs else float('nan')
        mean_hos = np.mean(hos) if hos else float('nan')
        
        print(f"{exp:<40} {sr:>11.1f}% {mean_cost:>12.1f} {mean_hos:>10.1f}")

if __name__ == "__main__":
    main()
