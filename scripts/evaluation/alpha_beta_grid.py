#!/usr/bin/env python3
"""Grid search for assignment solver alpha/beta weights.

Runs the existing evaluation pipeline for a Cartesian product of alpha and beta
values, then writes aggregate summaries focused on execution success and error
metrics.

By default the script evaluates the selected experiments once per (alpha, beta)
pair and computes aggregates across experiments:
  - success_rate = mean(task_success)
  - mean_position_tracking_error_m = mean(object_mean_tracking_error)
  - mean_orientation_error_deg = mean(placement_orientation_error)

Notes:
  - Position tracking error is the existing trajectory-wide object tracking
    metric already emitted by the execution pipeline.
  - Orientation error is currently the final placement orientation error in
    degrees; the execution stack does not yet expose a trajectory-wide mean
    orientation tracking error.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
import logging
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List

from tqdm import tqdm

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import BASELINE, AblationConfig, config_id, PAPER_EXPERIMENTS
from scripts.evaluation.results_store import RESULTS_ROOT, load_result
from scripts.evaluation.run_all import (
    discover_experiments,
    run_single_config,
    single_config_worker,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ALPHA-BETA] %(levelname)s  %(message)s",
        force=True,
    )


def _make_cfg(alpha: float, beta: float) -> AblationConfig:
    cfg = dict(BASELINE)
    cfg["label"] = f"Alpha={alpha:g}, Beta={beta:g}"
    cfg["alpha"] = alpha
    cfg["beta"] = beta
    return cfg


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _aggregate_pair(alpha: float, beta: float, rows: List[Dict]) -> Dict:
    success_values = [1.0 if row["task_success"] else 0.0 for row in rows if row["task_success"] is not None]
    pos_values = [row["object_mean_tracking_error"] for row in rows]
    ori_values = [row["placement_orientation_error"] for row in rows]
    valid_assignments = [1.0 if row["assignment_succeeded"] else 0.0 for row in rows]

    return {
        "alpha": alpha,
        "beta": beta,
        "n_runs": len(rows),
        "assignment_success_rate": _safe_mean(valid_assignments),
        "task_success_rate": _safe_mean(success_values),
        "mean_position_tracking_error_m": _safe_mean(pos_values),
        "mean_orientation_error_deg": _safe_mean(ori_values),
    }


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _result_path(results_dir: Path, exp_name: str, cfg: AblationConfig) -> Path:
    return results_dir / exp_name / f"{config_id(cfg)}.json"


def _is_completed_job(result_path: Path, require_execution: bool) -> bool:
    if not result_path.is_file():
        return False
    try:
        res = load_result(result_path)
    except Exception:
        return False

    if res.planning is None:
        return False

    # If execution metrics are required, ensure successful planning runs
    # have execution populated. Failed planning runs are considered complete.
    if require_execution and res.planning.total_cost < float("inf") and res.execution is None:
        return False

    return True


def _to_detail_row(
    exp_name: str,
    alpha: float,
    beta: float,
    cfg: AblationConfig,
    result,
) -> Dict:
    return {
        "experiment": exp_name,
        "alpha": alpha,
        "beta": beta,
        "config_id": config_id(cfg),
        "assignment_succeeded": bool(
            result and result.planning and result.planning.total_cost < float("inf")
        ),
        "task_success": (
            bool(result.execution.task_success)
            if result and result.execution is not None else None
        ),
        "object_mean_tracking_error": (
            float(result.execution.object_mean_tracking_error)
            if result and result.execution is not None else float("nan")
        ),
        "placement_orientation_error": (
            float(result.execution.placement_orientation_error)
            if result and result.execution is not None and result.execution.placement_orientation_error is not None
            else float("nan")
        ),
    }


def _to_detail_row_from_dict(
    exp_name: str,
    alpha: float,
    beta: float,
    cfg: AblationConfig,
    result_dict: Dict | None,
) -> Dict:
    if not result_dict:
        return {
            "experiment": exp_name,
            "alpha": alpha,
            "beta": beta,
            "config_id": config_id(cfg),
            "assignment_succeeded": False,
            "task_success": None,
            "object_mean_tracking_error": float("nan"),
            "placement_orientation_error": float("nan"),
        }

    planning = result_dict.get("planning")
    execution = result_dict.get("execution")
    return {
        "experiment": exp_name,
        "alpha": alpha,
        "beta": beta,
        "config_id": config_id(cfg),
        "assignment_succeeded": bool(
            planning is not None and planning.get("total_cost", float("inf")) < float("inf")
        ),
        "task_success": (
            bool(execution.get("task_success")) if execution is not None else None
        ),
        "object_mean_tracking_error": (
            float(execution.get("object_mean_tracking_error", float("nan")))
            if execution is not None else float("nan")
        ),
        "placement_orientation_error": (
            float(execution.get("placement_orientation_error"))
            if execution is not None and execution.get("placement_orientation_error") is not None
            else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search alpha/beta for assignment solver")
    parser.add_argument("--alpha-values", nargs="+", type=float,
                        default=[2.5, 5.0, 10.0, 20.0, 40.0, 50.0],
                        help="List of alpha values to sweep (default: 2.5 5 10 20 40)")
    parser.add_argument("--beta-values", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                        help="List of beta values to sweep (default: 0 0.25 0.5 1 2 4)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these experiments (names without .json)")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Exclude these experiments")
    parser.add_argument("--paper", action="store_true",
                        help="Run only the paper experiment subset")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for per-run JSONs and grid summaries")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for (experiment, alpha, beta) jobs")
    parser.add_argument("--no-env", action="store_true",
                        help="Skip environment creation for assignment (faster, no IK/Jacobian)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted sweep by skipping completed jobs with existing result JSONs")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip headless execution. Success/error metrics will then be empty.")
    args = parser.parse_args()

    _setup_logging()

    if args.paper:
        selected = PAPER_EXPERIMENTS
        if args.only is not None:
            selected = [exp for exp in selected if exp in set(args.only)]
        if args.exclude is not None:
            selected = [exp for exp in selected if exp not in set(args.exclude)]
        experiments = discover_experiments(only=selected)
    else:
        experiments = discover_experiments(only=args.only, exclude=args.exclude)

    if not experiments:
        raise SystemExit("No experiments selected.")

    pair_cfgs = [
        (alpha, beta, _make_cfg(alpha, beta))
        for alpha, beta in itertools.product(args.alpha_values, args.beta_values)
    ]
    total_jobs = len(experiments) * len(pair_cfgs)
    results_dir = Path(args.results_dir) if args.results_dir else (RESULTS_ROOT / "alpha_beta_grid")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Experiments: %s", ", ".join(Path(p).stem for p in experiments))
    logger.info("Grid size: %d alpha x %d beta = %d pairs", len(args.alpha_values), len(args.beta_values), len(pair_cfgs))
    logger.info("Workers: %d", args.workers)
    logger.info("Results dir: %s", results_dir)
    logger.info(
        "Trajectory fairness: this sweep does not regenerate trajectories; "
        "all alpha/beta pairs use the same object trajectory files from disk."
    )
    if args.resume:
        logger.info("Resume mode enabled: completed jobs will be skipped.")

    detail_rows: List[Dict] = []
    pending_jobs: List[tuple[str, str, float, float, AblationConfig]] = []
    progress = tqdm(total=total_jobs, desc="alpha/beta sweep", unit="job")

    for exp_json in experiments:
        exp_name = Path(exp_json).stem
        for alpha, beta, cfg in pair_cfgs:
            result_path = _result_path(results_dir, exp_name, cfg)
            if args.resume and _is_completed_job(result_path, require_execution=not args.no_execute):
                result = load_result(result_path)
                detail_rows.append(_to_detail_row(exp_name, alpha, beta, cfg, result))
                progress.update(1)
            else:
                pending_jobs.append((exp_json, exp_name, alpha, beta, cfg))

    if args.workers <= 1:
        for exp_json, exp_name, alpha, beta, cfg in pending_jobs:
            result = run_single_config(
                exp_json_path=exp_json,
                cfg=cfg,
                no_env=args.no_env,
                results_dir=results_dir,
                execute=not args.no_execute,
            )
            detail_rows.append(_to_detail_row(exp_name, alpha, beta, cfg, result))
            progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(
                    single_config_worker,
                    exp_json,
                    cfg,
                    args.no_env,
                    str(results_dir),
                    (not args.no_execute),
                ): (exp_name, alpha, beta, cfg)
                for exp_json, exp_name, alpha, beta, cfg in pending_jobs
            }

            for fut in as_completed(future_map):
                exp_name, alpha, beta, cfg = future_map[fut]
                try:
                    result_dict = fut.result()
                except Exception as exc:
                    logger.error(
                        "Worker failed for %s (alpha=%s, beta=%s): %s",
                        exp_name, alpha, beta, exc,
                    )
                    result_dict = None

                detail_rows.append(
                    _to_detail_row_from_dict(exp_name, alpha, beta, cfg, result_dict)
                )
                progress.update(1)

    progress.close()
    detail_rows.sort(key=lambda r: (r["experiment"], float(r["alpha"]), float(r["beta"])))

    summary_rows: List[Dict] = []
    for alpha, beta, _cfg in pair_cfgs:
        rows = [row for row in detail_rows if row["alpha"] == alpha and row["beta"] == beta]
        summary_rows.append(_aggregate_pair(alpha, beta, rows))

    summary_rows.sort(key=lambda row: (-row["task_success_rate"], row["mean_position_tracking_error_m"], row["mean_orientation_error_deg"]))

    detail_csv = results_dir / "alpha_beta_grid_details.csv"
    summary_csv = results_dir / "alpha_beta_grid_summary.csv"
    summary_json = results_dir / "alpha_beta_grid_summary.json"
    _write_csv(detail_csv, detail_rows)
    _write_csv(summary_csv, summary_rows)
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info("Wrote detail CSV: %s", detail_csv)
    logger.info("Wrote summary CSV: %s", summary_csv)
    logger.info("Wrote summary JSON: %s", summary_json)

    if summary_rows:
        best = summary_rows[0]
        logger.info(
            "Best pair by task success then error: alpha=%s beta=%s success_rate=%.3f pos_err=%.4f ori_err=%.3f",
            best["alpha"], best["beta"], best["task_success_rate"],
            best["mean_position_tracking_error_m"], best["mean_orientation_error_deg"],
        )


if __name__ == "__main__":
    main()