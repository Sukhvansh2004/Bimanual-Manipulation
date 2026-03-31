#!/usr/bin/env python3
"""
Full Pipeline Iterative Runner.

Orchestrates multiple iterations of:
  1. Trajectory Generation (with different seeds)
  2. Segment Assignment (ALL CONFIGS)
  3. Execution

Collects metrics across all iterations to evaluate robustness.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import ALL_ABLATIONS, config_id, ALL_CONFIGS

TRAJ_GEN_SCRIPT = "scripts/pipeline/trajectory_generation.py"
RUN_ALL_SCRIPT = "scripts/evaluation/run_all.py"

def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
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

def trajectories_exist(data_dir: str) -> bool:
    path = Path(_PROJECT_ROOT) / data_dir
    if not path.exists(): return False
    # Check if we have at least some robot0 trajectories
    return len(list(path.glob("robot0_trajectory_*.npz"))) > 0

def is_experiment_finished(results_dir: Path, exp_name: str, configs: List[dict]) -> bool:
    exp_results = results_dir / exp_name
    if not exp_results.exists(): return False
    for cfg in configs:
        cid = config_id(cfg)
        if not (exp_results / f"{cid}.json").is_file():
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Iterative pipeline runner")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiments to run")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations per experiment")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--paper", action="store_true",
                        help="Run paper configurations only")
    parser.add_argument("--ablations", nargs="*", default=None,
                        help="Specific ablation groups to run")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to save the execution log")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed iterations/experiments")
    parser.add_argument("--reuse-object-traj", action="store_true",
                        help="Reuse cached object trajectories across iterations")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    if args.experiments:
        experiments = args.experiments
    else:
        exp_dir = Path(_PROJECT_ROOT) / "experiments"
        experiments = [f.stem for f in exp_dir.glob("*.json")]
    
    iters = args.iterations
    
    if args.paper:
        from scripts.evaluation.configs import ALL_ABLATIONS
        check_configs = ALL_ABLATIONS["paper"]
    elif args.ablations:
        from scripts.evaluation.run_all import resolve_configs
        check_configs = resolve_configs(args.ablations)
    else:
        check_configs = ALL_CONFIGS

    pbar = tqdm(total=iters, desc="Iterations", unit="iter")
    
    for i in range(iters):
        iter_results_dir = Path(_PROJECT_ROOT) / "results" / f"iter_{i+1}"
        logger.info(f"\n{'='*80}\n  ITERATION {i+1}/{iters}\n{'='*80}")
        pbar.set_description(f"Iter {i+1}/{iters}")

        # ── Phase 1: Generate trajectories for ALL experiments ──────────
        # Trajectory generation already uses --workers internally for
        # parallel environments, so we run experiments sequentially here.
        ready_experiments = []
        for exp in experiments:
            if args.resume and is_experiment_finished(iter_results_dir, exp, check_configs):
                logger.info(f"--- [Iter {i+1}] {exp} already finished. Skipping entirely. ---")
                continue

            # 1a. Standard Trajectory Generation
            data_dir = f"data/{exp}"
            if args.resume and trajectories_exist(data_dir):
                logger.info(f"--- [Iter {i+1}] Standard trajectories for {exp} exist. Skipping Gen. ---")
            else:
                logger.info(f"--- [Iter {i+1}] Generating standard trajectories for {exp} ---")
                gen_cmd = [
                    sys.executable, TRAJ_GEN_SCRIPT,
                    "--experiment", f"experiments/{exp}.json",
                    "--save-traj", "--no-meshcat",
                    "--workers", str(args.workers), "--seed", str(42 + i),
                ]
                if args.reuse_object_traj:
                    gen_cmd.append("--reuse-object-traj")
                if not run_command(gen_cmd, f"Standard Gen {exp} Iter {i+1}"):
                    logger.error(f"Skipping {exp} iter {i+1} due to trajectory generation failure.")
                    continue

            # 1b. No-DPP Trajectory Generation
            no_dpp_data_dir = f"data/{exp}_no_dpp"
            if args.resume and trajectories_exist(no_dpp_data_dir):
                logger.info(f"--- [Iter {i+1}] No-DPP trajectories for {exp} exist. Skipping Gen. ---")
            else:
                logger.info(f"--- [Iter {i+1}] Generating No-DPP trajectories for {exp} ---")
                no_dpp_cmd = [
                    sys.executable, TRAJ_GEN_SCRIPT,
                    "--experiment", f"experiments/{exp}.json",
                    "--save-traj", "--no-meshcat",
                    "--workers", str(args.workers), "--seed", str(42 + i), "--no-dpp",
                ]
                if args.reuse_object_traj:
                    no_dpp_cmd.append("--reuse-object-traj")
                if not run_command(no_dpp_cmd, f"No-DPP Gen {exp} Iter {i+1}"):
                    logger.warning(f"No-DPP gen failed for {exp} iter {i+1}; No Diversity config will be skipped.")

            ready_experiments.append(exp)

        # ── Phase 2: Assignment + Execution for ALL experiments at once ─
        # This distributes ALL (experiment × config) jobs across workers,
        # so no worker sits idle while a slow config blocks others.
        if ready_experiments:
            logger.info(f"\n--- [Iter {i+1}] Running Assignment and Execution for {len(ready_experiments)} experiments ---")
            run_all_cmd = [
                sys.executable, RUN_ALL_SCRIPT,
                "--only", *ready_experiments,
                "--execute",
                "--workers", str(args.workers),
                "--results-dir", str(iter_results_dir),
            ]
            if args.resume: run_all_cmd.append("--resume")
            if args.paper: run_all_cmd.append("--paper")
            elif args.ablations: run_all_cmd.extend(["--ablations"] + args.ablations)

            run_command(run_all_cmd, f"Batch Runner All Experiments Iter {i+1}")
        else:
            logger.warning(f"--- [Iter {i+1}] No experiments ready for assignment. ---")

        pbar.update(1)

    pbar.close()
    logger.info(f"\n{'='*80}\n  ITERATIVE RUN COMPLETE\n{'='*80}")

if __name__ == "__main__":
    main()
