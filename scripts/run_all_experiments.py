#!/usr/bin/env python3
"""
Batch runner for all trajectory generation experiments.

Runs every experiment JSON in experiments/ sequentially with:
  - All meshcat / mesh visualizations disabled
  - Automatic trajectory saving enabled (--save-traj)

Usage:
    python scratch/run_all_experiments.py                  # run all experiments
    python scratch/run_all_experiments.py --only flatPlate_shelf_to_box flatPlate_table_to_shelf
    python scratch/run_all_experiments.py --workers 4      # override worker count
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BATCH] %(levelname)s  %(message)s",
)
logger = logging.getLogger("batch_runner")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
TRAJ_GEN_SCRIPT = os.path.join(ROOT_DIR, "scripts", "planning", "trajectory_generation.py")


def patch_config(cfg: dict, worker_override: int | None = None) -> dict:
    """Return a modified copy of the experiment config with viz off and save on."""
    cfg = json.loads(json.dumps(cfg))  # deep copy

    # Disable all visualizations (including meshcat)
    vis = cfg.setdefault("visualization", {})
    vis["visualize_mesh"] = False
    vis["enable_meshcat"] = False

    # Force-enable trajectory saving
    out = cfg.setdefault("output", {})
    out["save_trajectories"] = True

    # Optionally override worker count
    if worker_override is not None:
        rt = cfg.setdefault("runtime", {})
        rt["num_workers"] = worker_override

    return cfg


def discover_experiments(only: list[str] | None = None) -> list[str]:
    """Return sorted list of experiment JSON paths."""
    all_jsons = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*.json")))
    if not only:
        return all_jsons
    filtered = []
    for path in all_jsons:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in only:
            filtered.append(path)
    return filtered


def run_experiment(json_path: str, worker_override: int | None = None) -> bool:
    """Run a single experiment. Returns True on success."""
    name = os.path.splitext(os.path.basename(json_path))[0]
    logger.info("=" * 60)
    logger.info("STARTING experiment: %s", name)
    logger.info("=" * 60)

    # Load and patch the config
    with open(json_path, "r") as f:
        cfg = json.load(f)

    patched = patch_config(cfg, worker_override=worker_override)

    # Write patched config to a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"{name}_patched_")
    try:
        with os.fdopen(tmp_fd, "w") as tmp_f:
            json.dump(patched, tmp_f, indent=2)

        cmd = [
            sys.executable,
            TRAJ_GEN_SCRIPT,
            "--experiment", tmp_path,
            "--save-traj",
            "--no-meshcat",
        ]

        logger.info("Running: %s", " ".join(cmd))
        t0 = time.time()

        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,  # no interactive prompts
        )

        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info("COMPLETED %s in %.1fs", name, elapsed)
            return True
        else:
            logger.error("FAILED %s (exit code %d) after %.1fs", name, result.returncode, elapsed)
            return False
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Batch-run trajectory generation experiments")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only these experiment names (without .json extension). "
             "E.g. --only flatPlate_shelf_to_box flatPlate_table_to_shelf",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override num_workers for all experiments",
    )
    args = parser.parse_args()

    experiments = discover_experiments(args.only)
    if not experiments:
        logger.error("No experiment configs found in %s", EXPERIMENTS_DIR)
        sys.exit(1)

    logger.info("Found %d experiment(s): %s",
                len(experiments),
                [os.path.splitext(os.path.basename(e))[0] for e in experiments])

    results: dict[str, bool] = {}
    total_t0 = time.time()

    for exp_path in experiments:
        name = os.path.splitext(os.path.basename(exp_path))[0]
        results[name] = run_experiment(exp_path, worker_override=args.workers)

    total_elapsed = time.time() - total_t0

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH SUMMARY  (%.1fs total)", total_elapsed)
    logger.info("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        logger.info("  %-30s %s", name, status)

    failed = sum(1 for ok in results.values() if not ok)
    if failed:
        logger.error("%d experiment(s) failed.", failed)
        sys.exit(1)
    else:
        logger.info("All experiments succeeded.")


if __name__ == "__main__":
    main()
