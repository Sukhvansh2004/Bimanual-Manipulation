#!/usr/bin/env python3
"""Analysis and visualization of experiment/ablation results.

Usage::

    python scripts/evaluation/analyze.py                   # full report
    python scripts/evaluation/analyze.py --results-dir results/
    python scripts/evaluation/analyze.py --no-show          # save only, no GUI
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running as a script from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.configs import ALL_ABLATIONS, config_id
from scripts.evaluation.metrics import ExperimentResult, PlanningMetrics, extract_trajectory_generation_metrics
from scripts.evaluation.results_store import load_all_results, RESULTS_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ANALYZE] %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _fmt(val, fmt=".2f"):
    if val is None: return "—"
    if isinstance(val, bool): return "✓" if val else "✗"
    if isinstance(val, float):
        if np.isinf(val): return "∞"
        return f"{val:{fmt}}"
    return str(val)


def print_summary_table(results: List[ExperimentResult]):
    """Print a concise summary table of all results."""
    print(f"\n{'═' * 120}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'═' * 120}")

    header = (
        f"{'Experiment':<40} {'Config':<30} {'Cost':>10} {'HOs':>5} "
        f"{'Manip':>8} {'Plan(s)':>8} {'Succ':>5} {'Coll':>5}"
    )
    print(header)
    print("─" * 120)

    for r in sorted(results, key=lambda x: (x.experiment, x.config_label)):
        p = r.planning
        e = r.execution
        
        cost = _fmt(p.total_cost, '.1f') if p else "—"
        hos = str(p.n_handovers) if p else "—"
        manip = _fmt(p.mean_manipulability, '.4f') if p else "—"
        ptime = _fmt(p.planning_time_s, '.1f') if p else "—"
        
        succ = _fmt(e.task_success) if e else "—"
        coll = str(e.n_collision_incidents) if e else "—"
        
        row = (
            f"{r.experiment:<40} {r.config_label:<30} "
            f"{cost:>10} {hos:>5} {manip:>8} {ptime:>8} {succ:>5} {coll:>5}"
        )
        print(row)
    print()


def export_csv(results: List[ExperimentResult], out_path: Path):
    """Export all metrics to CSV."""
    import csv
    fields = [
        "experiment", "config_label", "total_cost", "n_handovers", "mean_manipulability",
        "planning_time_s", "task_success", "n_collision_incidents", "execution_time_s"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {"experiment": r.experiment, "config_label": r.config_label}
            if r.planning:
                for k in ["total_cost", "n_handovers", "mean_manipulability", "planning_time_s"]:
                    row[k] = getattr(r.planning, k, "")
            if r.execution:
                for k in ["task_success", "n_collision_incidents", "execution_time_s"]:
                    row[k] = getattr(r.execution, k, "")
            writer.writerow(row)
    logger.info("CSV exported: %s", out_path)


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_results(results: List[ExperimentResult], out_dir: Path, show: bool = False):
    """Generate basic bar charts for key metrics."""
    plt = _ensure_matplotlib()
    
    # Filter to cases with planning results
    valid = [r for r in results if r.planning]
    if not valid: return

    # Cost plot
    fig, ax = plt.subplots(figsize=(12, 6))
    exps = sorted(list(set(r.experiment for r in valid)))
    configs = sorted(list(set(r.config_label for r in valid)))
    
    x = np.arange(len(exps))
    width = 0.8 / len(configs)
    
    for i, cfg in enumerate(configs):
        costs = []
        for exp in exps:
            r = next((r for r in valid if r.experiment == exp and r.config_label == cfg), None)
            costs.append(r.planning.total_cost if r and r.planning.total_cost < 1e6 else 0)
        ax.bar(x + i*width, costs, width, label=cfg)
    
    ax.set_ylabel('Total Cost')
    ax.set_title('Costs by Experiment and Configuration')
    ax.set_xticks(x + width * (len(configs)-1) / 2)
    ax.set_xticklabels(exps, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_dir / "costs_summary.png")
    if show: plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        logger.warning("No results found.")
        return

    print_summary_table(results)
    export_csv(results, out_dir / "results.csv")
    plot_results(results, out_dir, not args.no_show)
    logger.info("Analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
