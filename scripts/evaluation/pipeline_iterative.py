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
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import PAPER_EXPERIMENTS, ALL_ABLATIONS, config_id, ALL_CONFIGS
from scripts.evaluation.results_store import load_result, RESULTS_ROOT

TRAJ_GEN_SCRIPT = "scripts/pipeline/trajectory_generation.py"
RUN_ALL_SCRIPT = "scripts/evaluation/run_all.py"

def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [PIPELINE] %(levelname)s  %(message)s",
        handlers=handlers,
        force=True
    )

def run_command(cmd, description):
    logger = logging.getLogger(__name__)
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
    parser.add_argument("--ablations", nargs="*", default=["baseline"],
                        help="Ablation groups to run (if not --paper)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to save the execution log")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    if args.experiments:
        experiments = args.experiments
    else:
        # All .json files in experiments/
        exp_dir = Path(_PROJECT_ROOT) / "experiments"
        experiments = [f.stem for f in exp_dir.glob("*.json")]
    
    iters = args.iterations
    
    # Total jobs = iterations * experiments
    pbar = tqdm(total=iters * len(experiments), desc="Evaluation Progress")
    
    for i in range(iters):
        logger.info(f"\n{'='*80}\n  ITERATION {i+1}/{iters}\n{'='*80}")
        
        for exp in experiments:
            pbar.set_description(f"Iter {i+1}/{iters} | {exp}")
            
            # 1a. Standard Trajectory Generation (DPP)
            logger.info(f"--- [Iter {i+1}] Generating standard trajectories for {exp} ---")
            gen_cmd = [
                sys.executable, TRAJ_GEN_SCRIPT,
                "--experiment", f"experiments/{exp}.json",
                "--save-traj",
                "--no-meshcat",
                "--workers", str(args.workers),
                "--seed", str(42 + i)
            ]
            run_command(gen_cmd, f"Standard Gen {exp} Iter {i+1}")

            # 1b. No-DPP Trajectory Generation
            logger.info(f"--- [Iter {i+1}] Generating No-DPP trajectories for {exp} ---")
            no_dpp_cmd = gen_cmd + ["--no-dpp"]
            run_command(no_dpp_cmd, f"No-DPP Gen {exp} Iter {i+1}")

            # 2 & 3. Assignment and Execution
            logger.info(f"--- [Iter {i+1}] Running Assignment and Execution for {exp} ---")
            run_all_cmd = [
                sys.executable, RUN_ALL_SCRIPT,
                "--only", exp,
                "--execute",
                "--workers", str(args.workers),
                "--results-dir", f"results/iter_{i+1}"
            ]
            if args.paper:
                run_all_cmd.append("--paper")
                
            run_command(run_all_cmd, f"Batch Runner {exp} Iter {i+1}")
            pbar.update(1)

    pbar.close()

    logger.info(f"\n{'='*80}\n  ITERATIVE RUN COMPLETE\n{'='*80}")
    logger.info("Run 'python scripts/evaluation/analyze.py' to see aggregated results.")

if __name__ == "__main__":
    main()
