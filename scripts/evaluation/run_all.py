#!/usr/bin/env python3
"""
Batch orchestrator for all planning experiments and ablation studies.

Runs assignment planning for every (experiment, ablation_config) combination.
Optionally also runs headless execution for each config to collect execution
metrics (task success, handover rate, manipulability, collisions).

For each experiment the simulation environment is created once and reused
across all configs, saving significant setup time.

Usage::

    # Run all 13 experiments × all ablation configs (planning only):
    python scripts/evaluation/run_all.py

    # Run a subset of experiments:
    python scripts/evaluation/run_all.py --only flatPlate_table_to_shelf ycb_mug_table_to_rack

    # Run only specific ablation groups:
    python scripts/evaluation/run_all.py --ablations baseline switch_penalty beam_width

    # Skip environment creation (faster but no IK/Jacobian):
    python scripts/evaluation/run_all.py --no-env

    # Run with 4 parallel workers (each gets its own MuJoCo env):
    python scripts/evaluation/run_all.py --workers 4

    # Planning + headless execution (collects execution metrics):
    python scripts/evaluation/run_all.py --execute

    # Generate analysis report after running:
    python scripts/evaluation/run_all.py --analyze
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from bimanual_manipulation.experiment_config import load_experiment

from scripts.evaluation.configs import (
    ALL_ABLATIONS,
    ALL_CONFIGS,
    AblationConfig,
    config_id,
    BASELINE,
    PAPER_EXPERIMENTS,
)
from scripts.evaluation.metrics import (
    ExperimentResult,
    ExecutionMetrics,
    PlanningMetrics,
)
from scripts.evaluation.results_store import save_result, load_result, build_summary, RESULTS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BATCH] %(levelname)s  %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
DATA_DIR = _PROJECT_ROOT / "data"

_ASSIGNMENT_SCRIPT = str(_PROJECT_ROOT / "scripts" / "pipeline" / "segment_assignment.py")

_ROBOT_GRIPPER = {0: "finger", 1: "suction"}


def _enrich_planning_from_npz(exp_result, asgn_npz_path: str) -> None:
    """Read assignment sequence from NPZ to populate grasp_gripper and handover_directions."""
    if not os.path.isfile(asgn_npz_path) or exp_result.planning is None:
        return
    if exp_result.planning.total_cost >= float("inf"):
        return
    try:
        asgn = np.load(asgn_npz_path, allow_pickle=True)
        seq = asgn["sequence"].tolist()
        hp = asgn["handover_points"].tolist() if "handover_points" in asgn.files else []
        exp_result.planning.grasp_gripper = _ROBOT_GRIPPER.get(seq[0], f"robot{seq[0]}")
        dirs = []
        for pt in hp:
            if pt > 0:
                src = _ROBOT_GRIPPER.get(seq[pt - 1], f"robot{seq[pt - 1]}")
                dst = _ROBOT_GRIPPER.get(seq[pt], f"robot{seq[pt]}")
                dirs.append(f"{src}→{dst}")
        exp_result.planning.handover_directions = dirs
    except Exception as exc:
        logger.debug("Could not enrich planning from NPZ: %s", exc)


def _child_preexec() -> None:
    """Ask the kernel to send SIGTERM to this child when its parent dies.

    Uses ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` (Linux-only).  This ensures
    that subprocesses are automatically terminated when the orchestrating
    Python process is killed.
    """
    _PR_SET_PDEATHSIG = 1
    libc_name = ctypes.util.find_library("c")
    if libc_name:
        libc = ctypes.CDLL(libc_name, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


# ═══════════════════════════════════════════════════════════════════════════
# Assignment runner (subprocess — delegates to segment_assignment.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_assignment_subprocess(
    exp_json_path: str,
    data_dir: str,
    cfg: AblationConfig,
    output_dir: str,
    metrics_file: str,
    no_env: bool = False,
    timeout: int = 14400,
    results_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run assignment planning via subprocess (calls segment_assignment.py).

    Builds the command line from the ablation config and delegates to the
    canonical pipeline script.  Metrics are written as JSON by the subprocess
    and read back here.

    Args:
        exp_json_path: Path to the experiment JSON file.
        data_dir: Directory containing trajectory NPZ files.
        cfg: Ablation config dict.
        output_dir: Directory where assignment_results.npz will be saved.
        metrics_file: Path to write the planning metrics JSON.
        no_env: Skip environment creation (--no-env).
        timeout: Subprocess timeout in seconds (default: 4 hours).
        results_dir: Directory for log files (default: RESULTS_ROOT).

    Returns:
        ExperimentResult with planning metrics populated.
    """
    cid = config_id(cfg)
    label = cfg["label"]
    exp_name = Path(exp_json_path).stem

    logger.info("  Config: %s", label)

    # --- Build command ---
    cmd = [
        sys.executable, _ASSIGNMENT_SCRIPT,
        "--experiment", exp_json_path,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--output-metrics", metrics_file,
        "--no-plot",
    ]

    # Parameter overrides from ablation config
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
    if not cfg["use_ik"] or no_env:
        cmd.append("--no-env")

    logger.info("    Command: %s", " ".join(cmd))

    # --- Run subprocess, redirecting to a log file ---
    _rdir = results_dir or RESULTS_ROOT
    log_file = str(_rdir / exp_name / f"{cid}_assignment.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=lf,
                text=True,
                timeout=timeout,
                preexec_fn=_child_preexec,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        planning_time = time.time() - t0

        if proc.returncode != 0:
            logger.error(
                "    Assignment FAILED (exit %d) — see %s",
                proc.returncode, log_file,
            )
            return ExperimentResult(
                experiment=exp_name,
                config_label=label,
                config_id=cid,
                planning=PlanningMetrics(
                    experiment=exp_name,
                    config_label=label,
                    planning_time_s=planning_time,
                ),
            )

        # --- Read metrics JSON written by segment_assignment.py ---
        if os.path.isfile(metrics_file):
            with open(metrics_file) as f:
                d = json.load(f)

            # Construct PlanningMetrics from the JSON
            d["config_label"] = label
            planning_metrics = PlanningMetrics(**{
                k: v for k, v in d.items()
                if k in PlanningMetrics.__dataclass_fields__
            })

            logger.info(
                "    ✓ cost=%.2f, handovers=%d, manip=%.4f, time=%.1fs",
                planning_metrics.total_cost, planning_metrics.n_handovers,
                planning_metrics.mean_manipulability, planning_metrics.planning_time_s,
            )
        else:
            logger.warning("    Metrics file not created: %s", metrics_file)
            planning_metrics = PlanningMetrics(
                experiment=exp_name,
                config_label=label,
                planning_time_s=planning_time,
            )

        return ExperimentResult(
            experiment=exp_name,
            config_label=label,
            config_id=cid,
            planning=planning_metrics,
        )

    except subprocess.TimeoutExpired:
        planning_time = time.time() - t0
        logger.error("    Assignment TIMED OUT (%ds)", timeout)
        return ExperimentResult(
            experiment=exp_name,
            config_label=label,
            config_id=cid,
            planning=PlanningMetrics(
                experiment=exp_name,
                config_label=label,
                planning_time_s=planning_time,
            ),
        )
    except Exception as exc:
        planning_time = time.time() - t0
        logger.error("    Assignment error after %.1fs: %s", planning_time, exc)
        return ExperimentResult(
            experiment=exp_name,
            config_label=label,
            config_id=cid,
            planning=PlanningMetrics(
                experiment=exp_name,
                config_label=label,
                planning_time_s=planning_time,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Execution runner (headless simulation)
# ═══════════════════════════════════════════════════════════════════════════

_EXEC_SCRIPT = str(_PROJECT_ROOT / "scripts" / "pipeline" / "dual_arm_handover_traj.py")


def run_execution(
    exp_json_path: str,
    assignment_file: str,
    experiment_name: str,
    config_label: str,
    results_dir: Path,
    config_id_str: str,
    timeout: int = 600,
    data_dir: Optional[str] = None,
) -> Optional[ExecutionMetrics]:
    """Run the dual-arm execution script headlessly for a single config.

    Args:
        data_dir: Optional override for the trajectory data directory.
            When ``None`` the execution script resolves the directory from
            the experiment JSON.  Pass explicitly for configs that use
            alternative trajectory data (e.g. No-Diversity / no-DPP).

    Returns ExecutionMetrics loaded from the JSON output, or None on failure.
    """
    metrics_file = str(results_dir / experiment_name / f"{config_id_str}_exec.json")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    cmd = [
        sys.executable, _EXEC_SCRIPT,
        "--experiment", exp_json_path,
        "--assignment", assignment_file,
        "--output-metrics", metrics_file,
        "--headless",
        "--no-meshcat",
        "--smoothen",
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", data_dir])

    logger.info("    Running execution: %s × %s", experiment_name, config_label)
    logger.info("    Command: %s", " ".join(cmd))

    log_file = str(results_dir / experiment_name / f"{config_id_str}_exec.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=lf,
                text=True,
                timeout=timeout,
                preexec_fn=_child_preexec,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

        if proc.returncode != 0:
            logger.error(
                "    Execution FAILED (exit %d) — see %s",
                proc.returncode, log_file,
            )

        # Load metrics from the JSON file (script writes it even on partial failure)
        if os.path.isfile(metrics_file):
            with open(metrics_file) as f:
                d = json.load(f)
            d["experiment"] = experiment_name
            d["config_label"] = config_label
            metrics = ExecutionMetrics(**{
                k: v for k, v in d.items()
                if k in ExecutionMetrics.__dataclass_fields__
            })
            logger.info(
                "    Execution result: task=%s, handover_rate=%.0f%%, collisions=%d, time=%.1fs",
                metrics.task_success,
                metrics.handover_success_rate * 100,
                metrics.n_collision_incidents,
                metrics.execution_time_s,
            )
            return metrics
        else:
            logger.error("    Metrics file not created: %s", metrics_file)
            return None

    except subprocess.TimeoutExpired:
        logger.error("    Execution TIMED OUT (%ds)", timeout)
        return ExecutionMetrics(
            experiment=experiment_name,
            config_label=config_label,
            error_message=f"Execution timed out after {timeout}s",
        )
    except Exception as exc:
        logger.error("    Execution error: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Experiment-level runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    exp_json_path: str,
    configs: List[AblationConfig],
    no_env: bool = False,
    results_dir: Optional[Path] = None,
    execute: bool = False,
) -> List[ExperimentResult]:
    """Run all ablation configs for a single experiment.

    Each config is run as a subprocess call to ``segment_assignment.py``,
    which creates its own simulation environment internally.
    If ``execute=True``, also runs headless execution for each config.
    """
    exp_cfg = load_experiment(exp_json_path)
    exp_name = exp_cfg.name

    # Resolve data directory
    data_dir = exp_cfg.output.output_dir or str(DATA_DIR / exp_name)

    # Check that trajectory data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Data directory not found: %s — skipping %s", data_dir, exp_name)
        return []

    n_r0 = len(list(data_path.glob("robot0_trajectory_*.npz")))
    n_r1 = len(list(data_path.glob("robot1_trajectory_*.npz")))
    if n_r0 == 0 or n_r1 == 0:
        logger.error(
            "No trajectory data found in %s (robot0=%d, robot1=%d) — skipping %s",
            data_dir, n_r0, n_r1, exp_name,
        )
        return []

    logger.info("=" * 70)
    logger.info("EXPERIMENT: %s  (%d configs, %d+%d trajectories)", exp_name, len(configs), n_r0, n_r1)
    logger.info("=" * 70)

    # Run each config via subprocess
    results: List[ExperimentResult] = []
    for i, cfg in enumerate(configs, 1):
        logger.info("[%d/%d] %s × %s", i, len(configs), exp_name, cfg["label"])
        cid = config_id(cfg)

        # Configs with use_dpp=False need trajectory data from the _no_dpp
        # directory (generated without DPP entropy maximization).
        if not cfg.get("use_dpp", True):
            cfg_data_dir = data_dir + "_no_dpp"
            cfg_data_path = Path(cfg_data_dir)
            if not cfg_data_path.exists() or not list(cfg_data_path.glob("robot0_trajectory_*.npz")):
                logger.warning(
                    "    No-DPP data dir not found or empty: %s — skipping %s",
                    cfg_data_dir, cfg["label"],
                )
                continue
            logger.info("    Using no-DPP data: %s", cfg_data_dir)
        else:
            cfg_data_dir = data_dir

        # Output paths for this config
        config_data_dir = Path(cfg_data_dir) / "configs" / cid
        metrics_file = str(config_data_dir / "planning_metrics.json")

        exp_result = run_assignment_subprocess(
            exp_json_path=exp_json_path,
            data_dir=cfg_data_dir,
            cfg=cfg,
            output_dir=str(config_data_dir),
            metrics_file=metrics_file,
            no_env=no_env,
            results_dir=results_dir,
        )

        # Enrich planning with gripper / handover direction from NPZ
        _enrich_planning_from_npz(
            exp_result, str(config_data_dir / "assignment_results.npz")
        )

        # Optionally run headless execution
        if execute and exp_result.planning and exp_result.planning.total_cost < float("inf"):
            asgn_file = str(config_data_dir / "assignment_results.npz")
            if os.path.isfile(asgn_file):
                exec_metrics = run_execution(
                    exp_json_path=exp_json_path,
                    assignment_file=asgn_file,
                    experiment_name=exp_name,
                    config_label=cfg["label"],
                    results_dir=results_dir or RESULTS_ROOT,
                    config_id_str=cid,
                    data_dir=cfg_data_dir if cfg_data_dir != data_dir else None,
                )
                if exec_metrics is not None:
                    exp_result.execution = exec_metrics
            else:
                logger.warning("    Assignment NPZ not found at %s — skipping execution", asgn_file)

        # Save immediately (so we don't lose progress on crash)
        save_result(exp_result, results_dir)
        results.append(exp_result)

    return results


def run_single_config(
    exp_json_path: str,
    cfg: AblationConfig,
    no_env: bool = False,
    results_dir: Optional[Path] = None,
    execute: bool = False,
) -> Optional[ExperimentResult]:
    """Run one (experiment, config) pair.

    Per-job equivalent of :func:`run_experiment`; designed for fine-grained
    parallel distribution where each worker handles a single config rather
    than an entire experiment.
    """
    exp_cfg = load_experiment(exp_json_path)
    exp_name = exp_cfg.name
    data_dir = exp_cfg.output.output_dir or str(DATA_DIR / exp_name)
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Data directory not found: %s — skipping", data_dir)
        return None

    cid = config_id(cfg)
    label = cfg["label"]
    logger.info("  %s × %s", exp_name, label)

    # Handle no-DPP configs
    if not cfg.get("use_dpp", True):
        cfg_data_dir = data_dir + "_no_dpp"
        cfg_data_path = Path(cfg_data_dir)
        if not cfg_data_path.exists() or not list(cfg_data_path.glob("robot0_trajectory_*.npz")):
            logger.warning(
                "    No-DPP data dir not found or empty: %s — skipping %s",
                cfg_data_dir, label,
            )
            return None
        logger.info("    Using no-DPP data: %s", cfg_data_dir)
    else:
        cfg_data_dir = data_dir

    config_data_dir = Path(cfg_data_dir) / "configs" / cid
    metrics_file = str(config_data_dir / "planning_metrics.json")

    exp_result = run_assignment_subprocess(
        exp_json_path=exp_json_path,
        data_dir=cfg_data_dir,
        cfg=cfg,
        output_dir=str(config_data_dir),
        metrics_file=metrics_file,
        no_env=no_env,
        results_dir=results_dir,
    )

    _enrich_planning_from_npz(exp_result, str(config_data_dir / "assignment_results.npz"))

    if execute and exp_result.planning and exp_result.planning.total_cost < float("inf"):
        asgn_file = str(config_data_dir / "assignment_results.npz")
        if os.path.isfile(asgn_file):
            exec_metrics = run_execution(
                exp_json_path=exp_json_path,
                assignment_file=asgn_file,
                experiment_name=exp_name,
                config_label=label,
                results_dir=results_dir or RESULTS_ROOT,
                config_id_str=cid,
                data_dir=cfg_data_dir if cfg_data_dir != data_dir else None,
            )
            if exec_metrics is not None:
                exp_result.execution = exec_metrics
        else:
            logger.warning("    Assignment NPZ not found — skipping execution")

    save_result(exp_result, results_dir)
    return exp_result


def single_config_worker(
    exp_json_path: str,
    cfg: AblationConfig,
    no_env: bool,
    results_dir_str: str,
    execute: bool = False,
) -> Optional[dict]:
    """Run a single (experiment, config) job in a worker process.

    Returns a dict (safe for cross-process serialisation), or ``None``.
    """
    exp_name = Path(exp_json_path).stem
    label = cfg["label"]
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W-{os.getpid()} {exp_name}×{label}] %(levelname)s  %(message)s",
        force=True,
    )
    results_dir = Path(results_dir_str)
    result = run_single_config(exp_json_path, cfg, no_env, results_dir, execute)
    return result.to_dict() if result else None


def is_completed(results_dir: Path, exp_name: str, cfg: AblationConfig) -> bool:
    """Check whether an (experiment, config) already has a result JSON."""
    cid = config_id(cfg)
    return (results_dir / exp_name / f"{cid}.json").is_file()


# ═══════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_experiments(
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return sorted list of experiment JSON paths."""
    all_jsons = sorted(glob.glob(str(EXPERIMENTS_DIR / "*.json")))
    if only:
        all_jsons = [
            p for p in all_jsons
            if Path(p).stem in only
        ]
    if exclude:
        all_jsons = [
            p for p in all_jsons
            if Path(p).stem not in exclude
        ]
    return all_jsons


def resolve_configs(ablation_names: Optional[List[str]] = None) -> List[AblationConfig]:
    """Resolve ablation names → unique list of configs."""
    if ablation_names is None:
        # All configs
        return ALL_CONFIGS

    configs = []
    seen = set()
    for name in ablation_names:
        if name not in ALL_ABLATIONS:
            logger.warning("Unknown ablation group: %s — skipping", name)
            continue
        for c in ALL_ABLATIONS[name]:
            if c["label"] not in seen:
                seen.add(c["label"])
                configs.append(c)

    if not configs:
        logger.error("No valid ablation configs resolved. Available: %s", list(ALL_ABLATIONS.keys()))
        sys.exit(1)

    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Multiprocessing worker
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_results(dicts: List[dict]) -> List[ExperimentResult]:
    """Rebuild ExperimentResult objects from worker-returned dicts."""
    results = []
    for d in dicts:
        planning = None
        if "planning" in d and d["planning"] is not None:
            planning = PlanningMetrics(**d["planning"])
        execution = None
        if "execution" in d and d["execution"] is not None:
            execution = ExecutionMetrics(**d["execution"])
        results.append(ExperimentResult(
            experiment=d["experiment"],
            config_label=d["config_label"],
            config_id=d["config_id"],
            planning=planning,
            execution=execution,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_batch_summary(all_results: List[ExperimentResult], elapsed: float):
    """Print a final summary of the batch run."""
    print(f"\n{'═' * 80}")
    print(f"  BATCH RUN COMPLETE  ({elapsed:.1f}s total, {len(all_results)} results)")
    print(f"{'═' * 80}")

    # Group by experiment
    by_exp: Dict[str, List[ExperimentResult]] = {}
    for r in all_results:
        by_exp.setdefault(r.experiment, []).append(r)

    print(f"\n{'Experiment':<40} {'Configs':>8} {'BestCost':>12} {'BestHandovers':>14}")
    print("─" * 80)

    for exp_name in sorted(by_exp):
        results = by_exp[exp_name]
        valid = [r for r in results if r.planning and r.planning.total_cost < float("inf")]
        if valid:
            best = min(valid, key=lambda r: r.planning.total_cost)
            print(
                f"{exp_name:<40} {len(results):>8} "
                f"{best.planning.total_cost:>12.2f} {best.planning.n_handovers:>14}"
            )
        else:
            print(f"{exp_name:<40} {len(results):>8} {'FAILED':>12} {'—':>14}")

    print()

    # Stats
    n_success = sum(1 for r in all_results if r.planning and r.planning.total_cost < float("inf"))
    n_fail = len(all_results) - n_success
    print(f"  Successful: {n_success}/{len(all_results)}")
    if n_fail:
        print(f"  Failed:     {n_fail}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner for planning ablations")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only these experiment names (without .json). "
             "E.g. --only flatPlate_table_to_shelf ycb_mug_table_to_rack",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=None,
        help="Exclude these experiment names.",
    )
    parser.add_argument(
        "--ablations", nargs="*", default=None,
        help="Run only these ablation groups. "
             f"Available: {list(ALL_ABLATIONS.keys())}",
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run only paper experiments and configurations.",
    )
    parser.add_argument(
        "--no-env", action="store_true",
        help="Skip environment creation (faster but no IK/Jacobian).",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Directory for results (default: {RESULTS_ROOT})",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis and generate plots after experiments finish.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="After planning each config, run headless execution to collect "
             "execution metrics (task success, handover rate, collisions, etc.).",
    )
    parser.add_argument(
        "--execute-only", action="store_true",
        help="Skip planning; run execution only for configs that already have "
             "assignment NPZs from a previous planning run.",
    )
    parser.add_argument(
        "--exec-timeout", type=int, default=600,
        help="Timeout in seconds for each execution run (default: 600).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes. Jobs (experiment × config "
             "pairs) are distributed across workers. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (experiment, config) pairs that already have a valid "
             "result (finite cost) in the results directory.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT

    # Resolve configs and experiments based on --paper flag
    if args.paper:
        only_exps = args.only if args.only else PAPER_EXPERIMENTS
        ablations = ["paper"]
    else:
        only_exps = args.only
        ablations = args.ablations

    # Discover experiments
    exp_paths = discover_experiments(only=only_exps, exclude=args.exclude)
    if not exp_paths:
        logger.error("No experiment configs found in %s", EXPERIMENTS_DIR)
        sys.exit(1)

    # Resolve configs
    configs = resolve_configs(ablations)
    exp_names = [Path(p).stem for p in exp_paths]

    logger.info("Batch plan: %d experiments × %d configs = %d runs",
                len(exp_paths), len(configs), len(exp_paths) * len(configs))
    logger.info("Experiments: %s", exp_names)
    logger.info("Configs: %s", [c["label"] for c in configs])
    logger.info("Results dir: %s", results_dir)

    if args.dry_run:
        mode = "execute-only" if args.execute_only else ("plan+execute" if args.execute else "plan-only")
        print(f"\nDRY RUN ({mode}) — would execute:")
        n_skip = 0
        for ep in exp_paths:
            for cfg in configs:
                ename = Path(ep).stem
                if args.resume and is_completed(results_dir, ename, cfg):
                    print(f"  {ename} × {cfg['label']}  [SKIP — already done]")
                    n_skip += 1
                else:
                    print(f"  {ename} × {cfg['label']}")
        total = len(exp_paths) * len(configs)
        print(f"\nTotal: {total} runs ({total - n_skip} to run, {n_skip} skipped)")
        return

    # --execute-only: skip planning, run execution on existing assignment NPZs
    if args.execute_only:
        logger.info("EXECUTE-ONLY mode — skipping planning, running execution on existing assignments")
        total_t0 = time.time()
        all_results: List[ExperimentResult] = []

        for ep in exp_paths:
            exp_name = Path(ep).stem
            exp_cfg = load_experiment(ep)
            data_dir = exp_cfg.output.output_dir or str(DATA_DIR / exp_name)

            for cfg in configs:
                cid = config_id(cfg)

                # Configs with use_dpp=False store data in the _no_dpp directory
                if not cfg.get("use_dpp", True):
                    cfg_data_dir = data_dir + "_no_dpp"
                else:
                    cfg_data_dir = data_dir

                asgn_file = str(Path(cfg_data_dir) / "configs" / cid / "assignment_results.npz")

                if not os.path.isfile(asgn_file):
                    logger.warning("  %s × %s — no assignment NPZ found, skipping", exp_name, cfg["label"])
                    continue

                exec_metrics = run_execution(
                    exp_json_path=ep,
                    assignment_file=asgn_file,
                    experiment_name=exp_name,
                    config_label=cfg["label"],
                    results_dir=results_dir,
                    config_id_str=cid,
                    timeout=args.exec_timeout,
                    data_dir=cfg_data_dir if cfg_data_dir != data_dir else None,
                )

                # Load existing planning result if available
                planning = None
                result_json = results_dir / exp_name / f"{cid}.json"
                if result_json.is_file():
                    existing = load_result(result_json)
                    planning = existing.planning

                exp_result = ExperimentResult(
                    experiment=exp_name,
                    config_label=cfg["label"],
                    config_id=cid,
                    planning=planning,
                    execution=exec_metrics,
                )
                save_result(exp_result, results_dir)
                all_results.append(exp_result)

        total_elapsed = time.time() - total_t0
        print_batch_summary(all_results, total_elapsed)

        build_summary(results_dir)

        if args.analyze:
            logger.info("Running analysis ...")
            from scripts.evaluation.analyze import main as analyze_main
            sys.argv = ["analyze.py", "--results-dir", str(results_dir), "--no-show"]
            analyze_main()
        return

    # Build job list: all (experiment, config) pairs, filtering completed if --resume
    num_workers = max(1, args.workers)
    total_t0 = time.time()
    all_results: List[ExperimentResult] = []

    jobs: list[tuple[str, AblationConfig]] = []
    n_skipped = 0
    for ep in exp_paths:
        exp_name = Path(ep).stem
        for cfg in configs:
            if args.resume and is_completed(results_dir, exp_name, cfg):
                logger.info("  SKIP (resume): %s × %s", exp_name, cfg["label"])
                n_skipped += 1
                continue
            jobs.append((ep, cfg))

    if n_skipped:
        logger.info("Resume: %d already completed, %d remaining", n_skipped, len(jobs))

    if not jobs:
        logger.info("All jobs already completed — nothing to run.")
    elif num_workers == 1:
        # Sequential — simpler output, easier to debug
        total_jobs = len(jobs)
        desc = f"Planning [0/{total_jobs} done, {total_jobs} remaining]"
        pbar = tqdm(total=total_jobs, desc=desc, unit="job", ncols=120)
        n_done = 0
        for exp_path, cfg in jobs:
            ename = Path(exp_path).stem
            pbar.set_postfix_str(f"{ename}×{cfg['label']}")
            result = run_single_config(
                exp_path, cfg,
                no_env=args.no_env,
                results_dir=results_dir,
                execute=args.execute,
            )
            if result:
                all_results.append(result)
            n_done += 1
            pbar.set_description(
                f"Planning [{n_done}/{total_jobs} done, {total_jobs - n_done} remaining]"
            )
            pbar.update(1)
        pbar.close()
    else:
        logger.info(
            "Launching %d parallel workers for %d jobs …",
            min(num_workers, len(jobs)), len(jobs),
        )
        total_jobs = len(jobs)
        desc = f"Planning [0/{total_jobs} done, {total_jobs} remaining]"
        pbar = tqdm(total=total_jobs, desc=desc, unit="job", ncols=120)
        n_done = 0
        # Use 'spawn' start method — safest with MuJoCo C libraries
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(jobs)),
            mp_context=ctx,
        ) as pool:
            future_to_job = {
                pool.submit(
                    single_config_worker,
                    ep,
                    cfg,
                    args.no_env,
                    str(results_dir),
                    args.execute,
                ): (Path(ep).stem, cfg["label"])
                for ep, cfg in jobs
            }
            for future in as_completed(future_to_job):
                exp_name, label = future_to_job[future]
                try:
                    result_dict = future.result()
                    if result_dict:
                        results = reconstruct_results([result_dict])
                        all_results.extend(results)
                    logger.info("✓ %s × %s finished", exp_name, label)
                except Exception as exc:
                    logger.error("✗ %s × %s FAILED: %s", exp_name, label, exc)

                n_done += 1
                pbar.set_description(
                    f"Planning [{n_done}/{total_jobs} done, {total_jobs - n_done} remaining]"
                )
                pbar.set_postfix_str(f"{exp_name}×{label}")
                pbar.update(1)
        pbar.close()

    total_elapsed = time.time() - total_t0
    print_batch_summary(all_results, total_elapsed)

    # Build summary
    build_summary(results_dir)

    # Optional analysis
    if args.analyze:
        logger.info("Running analysis ...")
        from scripts.evaluation.analyze import main as analyze_main
        sys.argv = [
            "analyze.py",
            "--results-dir", str(results_dir),
            "--no-show",
        ]
        analyze_main()


if __name__ == "__main__":
    main()
