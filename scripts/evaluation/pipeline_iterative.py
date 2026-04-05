#!/usr/bin/env python3
"""
Full Pipeline Iterative Runner.

Orchestrates multiple iterations of:
  1. Trajectory Generation (with different seeds)
  2. Segment Assignment (ALL CONFIGS)
  3. Execution

Collects metrics across all iterations to evaluate robustness.

Supports ``--resume`` for crash recovery: completed iterations are
auto-detected and skipped, partially-completed iterations pick up
where they left off (both trajectory generation and assignment/execution).
"""

import argparse
import json
import logging
import os
import shutil
import signal
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
_CHECKPOINT_FILE = Path(_PROJECT_ROOT) / "results" / ".pipeline_checkpoint.json"

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    logger = logging.getLogger(__name__)
    logger.warning("Signal %s received — finishing current job then exiting.", signum)
    _shutdown_requested = True


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


# ---------------------------------------------------------------------------
# Trajectory completeness checks
# ---------------------------------------------------------------------------

def trajectories_complete(data_dir: str) -> bool:
    """Check that trajectory generation finished for *data_dir*.

    Verifies that the directory contains:
      * ``object_trajectory.npz``
      * At least one ``robot0_trajectory_*.npz``
      * At least one ``robot1_trajectory_*.npz``
    """
    path = Path(_PROJECT_ROOT) / data_dir
    if not path.exists():
        return False
    has_obj = (path / "object_trajectory.npz").is_file()
    has_r0 = len(list(path.glob("robot0_trajectory_*.npz"))) > 0
    has_r1 = len(list(path.glob("robot1_trajectory_*.npz"))) > 0
    return has_obj and has_r0 and has_r1


def is_experiment_finished(results_dir: Path, exp_name: str, configs: List[dict]) -> bool:
    exp_results = results_dir / exp_name
    if not exp_results.exists():
        return False
    for cfg in configs:
        cid = config_id(cfg)
        if not (exp_results / f"{cid}.json").is_file():
            return False
    return True


def find_resume_iteration(results_base: Path, total_iters: int,
                          experiments: List[str], configs: List[dict]) -> int:
    """Return the 0-based index of the first incomplete iteration."""
    for i in range(total_iters):
        iter_dir = results_base / f"iter_{i+1}"
        if not iter_dir.exists():
            return i
        if not all(is_experiment_finished(iter_dir, exp, configs) for exp in experiments):
            return i
    return total_iters  # everything done


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def save_checkpoint(iteration: int, phase: str, experiment: str = ""):
    """Write a small JSON checkpoint so progress survives crashes."""
    _CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"iteration": iteration, "phase": phase, "experiment": experiment}
    _CHECKPOINT_FILE.write_text(json.dumps(data))


def load_checkpoint() -> Optional[dict]:
    if _CHECKPOINT_FILE.is_file():
        try:
            return json.loads(_CHECKPOINT_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def clear_checkpoint():
    if _CHECKPOINT_FILE.is_file():
        _CHECKPOINT_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Iterative pipeline runner with crash-resilient resume",
    )
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
                        help="Resume from last checkpoint — skips completed "
                             "iterations and already-finished experiments")
    parser.add_argument("--reuse-object-traj", action="store_true",
                        help="Reuse cached object trajectories across iterations")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base random seed (iteration i uses base_seed + i)")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    # Install signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if args.experiments:
        experiments = sorted(args.experiments)
    else: # Run all the experiments if none are specified; discover them by looking for JSON files in the experiments directory
        exp_dir = Path(_PROJECT_ROOT) / "experiments"
        experiments = sorted(f.stem for f in exp_dir.glob("*.json"))
    
    iters = args.iterations
    
    if args.paper:
        check_configs = ALL_ABLATIONS["paper"]
    elif args.ablations:
        from scripts.evaluation.run_all import resolve_configs
        check_configs = resolve_configs(args.ablations)
    else:
        check_configs = ALL_CONFIGS

    results_base = Path(_PROJECT_ROOT) / "results"

    # ── Resume: fast-forward to first incomplete iteration ──────────────
    start_iter = 0
    if args.resume:
        start_iter = find_resume_iteration(results_base, iters, experiments, check_configs)
        if start_iter > 0:
            logger.info("Resume: skipping %d fully completed iteration(s), "
                        "starting at iteration %d.", start_iter, start_iter + 1)
        if start_iter >= iters:
            logger.info("All %d iterations already complete — nothing to do.", iters)
            clear_checkpoint()
            return

    remaining = iters - start_iter
    pbar = tqdm(total=remaining, desc="Iterations", unit="iter")

    for i in range(start_iter, iters):
        if _shutdown_requested:
            logger.warning("Shutdown requested — stopping before iteration %d.", i + 1)
            break

        iter_results_dir = results_base / f"iter_{i+1}"
        logger.info(f"\n{'='*80}\n  ITERATION {i+1}/{iters}\n{'='*80}")
        pbar.set_description(f"Iter {i+1}/{iters}")
        save_checkpoint(i + 1, "start")

        # ── Phase 1: Generate trajectories for ALL experiments ──────────
        ready_experiments = []
        for exp in experiments:
            if _shutdown_requested:
                break

            if args.resume and is_experiment_finished(iter_results_dir, exp, check_configs):
                logger.info(f"--- [Iter {i+1}] {exp} already finished. Skipping entirely. ---")
                continue

            # 1a. Standard Trajectory Generation
            data_dir = f"data/{exp}"
            save_checkpoint(i + 1, "traj_gen", exp)
            if args.resume and trajectories_complete(data_dir):
                logger.info(f"--- [Iter {i+1}] Standard trajectories for {exp} complete. Skipping. ---")
            else:
                logger.info(f"--- [Iter {i+1}] Generating standard trajectories for {exp} ---")
                gen_cmd = [
                    sys.executable, TRAJ_GEN_SCRIPT,
                    "--experiment", f"experiments/{exp}.json",
                    "--save-traj", "--no-meshcat",
                    "--workers", str(args.workers), "--seed", str(args.base_seed + i),
                ]
                if args.reuse_object_traj:
                    gen_cmd.append("--reuse-object-traj")
                if not run_command(gen_cmd, f"Standard Gen {exp} Iter {i+1}"):
                    logger.error(f"Skipping {exp} iter {i+1} due to trajectory generation failure.")
                    continue

            # 1b. No-DPP Trajectory Generation
            no_dpp_data_dir = f"data/{exp}_no_dpp"
            save_checkpoint(i + 1, "traj_gen_no_dpp", exp)
            if args.resume and trajectories_complete(no_dpp_data_dir):
                logger.info(f"--- [Iter {i+1}] No-DPP trajectories for {exp} complete. Skipping. ---")
            else:
                logger.info(f"--- [Iter {i+1}] Generating No-DPP trajectories for {exp} ---")
                no_dpp_cmd = [
                    sys.executable, TRAJ_GEN_SCRIPT,
                    "--experiment", f"experiments/{exp}.json",
                    "--save-traj", "--no-meshcat",
                    "--workers", str(args.workers), "--seed", str(args.base_seed + i), "--no-dpp",
                ]
                if args.reuse_object_traj:
                    no_dpp_cmd.append("--reuse-object-traj")
                if not run_command(no_dpp_cmd, f"No-DPP Gen {exp} Iter {i+1}"):
                    logger.warning(f"No-DPP gen failed for {exp} iter {i+1}; No Diversity config will be skipped.")

            ready_experiments.append(exp)

        # ── Phase 2: Assignment + Execution for ALL experiments at once ─
        if _shutdown_requested:
            logger.warning("Shutdown requested — skipping Phase 2 for iteration %d.", i + 1)
            break

        if ready_experiments:
            save_checkpoint(i + 1, "assignment_execution")
            logger.info(f"\n--- [Iter {i+1}] Running Assignment and Execution for {len(ready_experiments)} experiments ---")
            run_all_cmd = [
                sys.executable, RUN_ALL_SCRIPT,
                "--only", *ready_experiments,
                "--execute",
                "--workers", str(args.workers),
                "--results-dir", str(iter_results_dir),
            ]
            # Always pass --resume to run_all so it skips already-completed
            # (experiment, config) pairs — safe even on a fresh run.
            run_all_cmd.append("--resume")
            if args.paper:
                run_all_cmd.append("--paper")
            elif args.ablations:
                run_all_cmd.extend(["--ablations"] + args.ablations)

            run_command(run_all_cmd, f"Batch Runner All Experiments Iter {i+1}")
        else:
            logger.warning(f"--- [Iter {i+1}] No experiments ready for assignment. ---")

        # ── Phase 3: Clean data directories for next iteration ──────────
        if i < iters - 1:  # Skip cleanup after the final iteration
            logger.info(f"--- [Iter {i+1}] Cleaning data directories for next iteration ---")
            for exp in ready_experiments:
                for suffix in ("", "_no_dpp"):
                    data_dir = Path(_PROJECT_ROOT) / "data" / f"{exp}{suffix}"
                    if not data_dir.exists():
                        continue
                    # Remove generated robot trajectories and configs, keep object trajectories
                    for pattern in ("robot*_trajectory_*.npz", "configs"):
                        for p in data_dir.glob(pattern):
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                shutil.rmtree(p)
                    logger.debug(f"  Cleaned {data_dir}")

        pbar.update(1)

    pbar.close()

    if _shutdown_requested:
        logger.info("Pipeline stopped gracefully. Re-run with --resume to continue.")
    else:
        clear_checkpoint()
        logger.info(f"\n{'='*80}\n  ITERATIVE RUN COMPLETE\n{'='*80}")

if __name__ == "__main__":
    main()
