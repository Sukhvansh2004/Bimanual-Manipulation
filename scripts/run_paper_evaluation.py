#!/usr/bin/env python3
"""
Paper Evaluation Runner — runs all 4 paper experiments with the paper configs.

This script orchestrates:
  1. Planning (assignment solver) for each experiment × config
  2. Execution (headless simulation) for configs with finite planning cost
  3. Metric collection and reporting

The 4 paper experiments are:
  A) ycb_plate_table_to_shelf   — Table-to-Shelf
  B) flatPlate_shelf_to_box     — Shelf-to-Box
  C) mug_inversion             — Mug Inversion (orientation change)
  D) flatPlate_box_to_box       — Box-to-Box (dual handover)

Paper configs correspond to Table 1 (method comparison) + Table 2 (ablation):
  - Proposed (our full method)
  - Greedy (beam=1)
  - Single Robot (switch cost = inf)
  - No Switching Penalty (C_switch=0)
  - No Manipulability (β=0)

Usage::

    # Full evaluation (planning + execution):
    python scripts/run_paper_evaluation.py

    # Planning only:
    python scripts/run_paper_evaluation.py --plan-only

    # Execution only (uses existing assignment NPZs):
    python scripts/run_paper_evaluation.py --exec-only

    # Single experiment:
    python scripts/run_paper_evaluation.py --only flatPlate_shelf_to_box

    # Debug mode (verbose logging):
    python scripts/run_paper_evaluation.py --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.experiments.configs import (
    PROPOSED, GREEDY, SINGLE_ROBOT, NO_SWITCH_PENALTY, NO_MANIPULABILITY,
    config_id, AblationConfig,
)
from scripts.experiments.run_all import (
    run_experiment, run_execution, print_batch_summary,
    _PROJECT_ROOT as PROJECT_ROOT, EXPERIMENTS_DIR, DATA_DIR,
)
from scripts.experiments.metrics import ExperimentResult, PlanningMetrics
from scripts.experiments.results_store import save_result, RESULTS_ROOT, build_summary

logger = logging.getLogger(__name__)

# Paper experiments
PAPER_EXPERIMENTS = [
    "ycb_plate_table_to_shelf",
    "flatPlate_shelf_to_box",
    "mug_inversion",
    "flatPlate_box_to_box",
]

# Paper configs (Table 1 + Table 2)
PAPER_CONFIGS = [PROPOSED, GREEDY, SINGLE_ROBOT, NO_SWITCH_PENALTY, NO_MANIPULABILITY]


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        force=True,
    )
    # Also log to a file
    log_file = _PROJECT_ROOT / "results" / "paper_evaluation.log"
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s  %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s (level=%s)", log_file, logging.getLevelName(level))


def run_planning_phase(
    experiments: list[str],
    configs: list[AblationConfig],
    results_dir: Path,
) -> list[ExperimentResult]:
    """Run planning for all experiment × config combinations."""
    all_results = []

    for exp_name in experiments:
        exp_json = str(EXPERIMENTS_DIR / f"{exp_name}.json")
        if not os.path.isfile(exp_json):
            logger.error("Experiment config not found: %s", exp_json)
            continue

        logger.info("=" * 70)
        logger.info("PLANNING: %s", exp_name)
        logger.info("=" * 70)

        results = run_experiment(
            exp_json, configs,
            no_env=False,  # Always use IK
            results_dir=results_dir,
            execute=False,
        )
        all_results.extend(results)

        # Summary for this experiment
        valid = [r for r in results if r.planning and r.planning.total_cost < float("inf")]
        logger.info(
            "  %s: %d/%d configs have finite cost",
            exp_name, len(valid), len(results),
        )
        for r in results:
            cost = r.planning.total_cost if r.planning else float("inf")
            status = "OK" if cost < float("inf") else "FAILED (inf)"
            logger.info("    %-40s  cost=%.2f  [%s]", r.config_label, cost, status)

    return all_results


def run_execution_phase(
    experiments: list[str],
    configs: list[AblationConfig],
    results_dir: Path,
    timeout: int = 600,
) -> list[ExperimentResult]:
    """Run execution for all configs that have valid planning results."""
    from robosuite_franka_manipulation.experiment_config import load_experiment
    all_results = []

    for exp_name in experiments:
        exp_json = str(EXPERIMENTS_DIR / f"{exp_name}.json")
        if not os.path.isfile(exp_json):
            continue

        exp_cfg = load_experiment(exp_json)
        data_dir = exp_cfg.output.output_dir or str(DATA_DIR / exp_name)

        for cfg in configs:
            cid = config_id(cfg)
            asgn_file = str(Path(data_dir) / "configs" / cid / "assignment_results.npz")

            if not os.path.isfile(asgn_file):
                logger.info("  %s × %s — no assignment NPZ, skip execution", exp_name, cfg["label"])
                continue

            logger.info("=" * 70)
            logger.info("EXECUTION: %s × %s", exp_name, cfg["label"])
            logger.info("=" * 70)

            exec_metrics = run_execution(
                exp_json_path=exp_json,
                assignment_file=asgn_file,
                experiment_name=exp_name,
                config_label=cfg["label"],
                results_dir=results_dir,
                config_id_str=cid,
                timeout=timeout,
            )

            # Load existing planning result
            planning = None
            result_json = results_dir / exp_name / f"{cid}.json"
            if result_json.is_file():
                try:
                    from scripts.experiments.results_store import load_result
                    existing = load_result(result_json)
                    planning = existing.planning
                except Exception:
                    pass

            exp_result = ExperimentResult(
                experiment=exp_name,
                config_label=cfg["label"],
                config_id=cid,
                planning=planning,
                execution=exec_metrics,
            )
            save_result(exp_result, results_dir)
            all_results.append(exp_result)

    return all_results


def print_paper_metrics(results_dir: Path, experiments: list[str] | None = None):
    """Print paper-ready metrics tables."""
    from scripts.experiments.paper_metrics import (
        load_all_results as load_all_dicts, group_by_config, group_by_experiment,
        is_valid_planning, compute_method_metrics,
    )

    filter_exps = experiments or PAPER_EXPERIMENTS
    results = load_all_dicts(results_dir)
    # Filter to paper experiments only
    results = [r for r in results if r["experiment"] in filter_exps]
    print(f"\nLoaded {len(results)} paper-experiment results\n")

    by_config = group_by_config(results)
    by_exp = group_by_experiment(results)

    # === Table 1: Method Comparison ===
    print("=" * 100)
    print("  TABLE 1: METHOD COMPARISON (4 paper experiments)")
    print("=" * 100)
    print(f"{'Method':<35} {'Plan%':>6} {'#HO':>6} {'Cost':>10} {'Manip':>8} {'Feas%':>6} {'Time(s)':>8}")
    print("-" * 100)

    for config_label in ["Proposed", "Greedy (beam=1)", "Single Robot", "No Switching Penalty (Cswitch=0)", "No Manipulability (β=0)"]:
        if config_label not in by_config:
            print(f"  {config_label:<35}  NO RESULTS")
            continue
        m = compute_method_metrics(by_config[config_label])
        plan_rate = f"{m['planning_success_rate']:.0f}%"
        ho = f"{m.get('avg_handovers', 0):.1f}" if m['n_planning_valid'] > 0 else "—"
        cost = f"{m.get('avg_total_cost', 0):.1f}" if m['n_planning_valid'] > 0 else "—"
        manip = f"{m.get('avg_manipulability', 0):.4f}" if m['n_planning_valid'] > 0 else "—"
        feas = f"{m.get('avg_feasibility', 0):.0f}%" if m['n_planning_valid'] > 0 else "—"
        time_s = f"{m.get('avg_planning_time', 0):.0f}" if m['n_planning_valid'] > 0 else "—"
        print(f"  {config_label:<35} {plan_rate:>6} {ho:>6} {cost:>10} {manip:>8} {feas:>6} {time_s:>8}")

        if m.get('n_exec_total', 0) > 0:
            print(f"    EXEC: task_sr={m.get('task_success_rate',0):.0f}%, "
                  f"ho_sr={m.get('avg_ho_success_rate',0):.0f}%, "
                  f"collisions={m.get('avg_collisions',0):.1f}, "
                  f"time={m.get('avg_exec_time',0):.0f}s")

    # === Per-experiment breakdown ===
    print("\n" + "=" * 100)
    print("  PER-EXPERIMENT BREAKDOWN")
    print("=" * 100)

    for exp_name in PAPER_EXPERIMENTS:
        if exp_name not in by_exp:
            print(f"\n--- {exp_name}: NO RESULTS ---")
            continue
        print(f"\n--- {exp_name} ---")
        exp_results = by_exp[exp_name]
        for r in exp_results:
            p = r.get("planning", {})
            cost = p.get("total_cost", float("inf"))
            if cost > 1e30:
                cost_str = "INF"
            else:
                cost_str = f"{cost:.2f}"
            ho = p.get("n_handovers", 0)
            manip = p.get("mean_manipulability", 0)
            time_s = p.get("planning_time_s", 0)
            label = r.get("config_label", "?")

            line = f"  {label:<40} cost={cost_str:>10}  ho={ho}  manip={manip:.4f}  time={time_s:.0f}s"
            e = r.get("execution")
            if e:
                line += f"  | EXEC: ok={e.get('task_success')}, ho_rate={e.get('handover_success_rate',0):.2f}, coll={e.get('n_collision_incidents',0)}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Paper evaluation runner")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--plan-only", action="store_true", help="Run planning only (skip execution)")
    parser.add_argument("--exec-only", action="store_true", help="Run execution only (use existing assignments)")
    parser.add_argument("--metrics-only", action="store_true", help="Only print metrics from existing results")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only specific experiments (e.g. --only flatPlate_shelf_to_box)")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--exec-timeout", type=int, default=600, help="Execution timeout (seconds)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.debug)

    experiments = args.only if args.only else PAPER_EXPERIMENTS
    # Validate experiment names
    experiments = [e for e in experiments if e in PAPER_EXPERIMENTS]
    if not experiments:
        logger.error("No valid experiments specified. Paper experiments: %s", PAPER_EXPERIMENTS)
        sys.exit(1)

    logger.info("Paper evaluation: %d experiments × %d configs", len(experiments), len(PAPER_CONFIGS))
    logger.info("Experiments: %s", experiments)
    logger.info("Configs: %s", [c["label"] for c in PAPER_CONFIGS])

    if args.metrics_only:
        print_paper_metrics(results_dir, experiments)
        return

    total_t0 = time.time()

    if not args.exec_only:
        logger.info("\n" + "=" * 80)
        logger.info("  PHASE 1: PLANNING")
        logger.info("=" * 80)
        planning_results = run_planning_phase(experiments, PAPER_CONFIGS, results_dir)
    else:
        planning_results = []

    if not args.plan_only:
        logger.info("\n" + "=" * 80)
        logger.info("  PHASE 2: EXECUTION")
        logger.info("=" * 80)
        exec_results = run_execution_phase(
            experiments, PAPER_CONFIGS, results_dir,
            timeout=args.exec_timeout,
        )
    else:
        exec_results = []

    total_elapsed = time.time() - total_t0
    logger.info("\nTotal evaluation time: %.1f seconds", total_elapsed)

    # Build summary
    build_summary(results_dir)

    # Print final metrics
    print_paper_metrics(results_dir, experiments)


if __name__ == "__main__":
    main()
