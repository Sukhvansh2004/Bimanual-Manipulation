#!/usr/bin/env python3
"""Analysis and visualization of experiment/ablation results.

Aggregates multiple iterations (if found) into Mean ± Std Dev.

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
from collections import defaultdict

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
    if isinstance(val, (float, np.floating)):
        if np.isinf(val): return "∞"
        if np.isnan(val): return "—"
        return f"{val:{fmt}}"
    return str(val)


def group_results(results: List[ExperimentResult]):
    """Group flat list of results by (experiment, config_label)."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.experiment, r.config_label)].append(r)
    return groups


def print_summary_table(groups: Dict[Tuple[str, str], List[ExperimentResult]]):
    """Print an aggregate summary table."""
    print(f"\n{'═' * 155}")
    print(f"  EXPERIMENT AGGREGATE SUMMARY")
    print(f"{'═' * 155}")

    header = (
        f"{'Experiment':<35} {'Config':<30} {'Iters':>5} {'Succ%':>8} "
        f"{'Cost (Avg±Std)':>20} {'HOs':>5} {'ErrAvg':>8} {'ErrMax':>8} {'Coll':>5}"
    )
    print(header)
    print("─" * 155)

    for (exp, cfg), rs in sorted(groups.items()):
        n_iters = len(rs)
        
        # Success rate
        successes = sum(1 for r in rs if r.execution and r.execution.task_success)
        succ_rate = (successes / n_iters) * 100
        
        # Costs
        costs = [r.planning.total_cost for r in rs if r.planning and r.planning.total_cost < 1e10]
        m_cost = np.mean(costs) if costs else float('nan')
        s_cost = np.std(costs) if len(costs) > 1 else 0.0
        cost_str = f"{_fmt(m_cost, '.1f')}±{_fmt(s_cost, '.1f')}"
        
        # HOs
        hos = [r.planning.n_handovers for r in rs if r.planning and r.planning.total_cost < 1e10]
        m_hos = np.mean(hos) if hos else float('nan')
        
        # Tracking errors
        err_avgs = [r.execution.object_mean_tracking_error for r in rs if r.execution]
        m_err = np.mean(err_avgs) if err_avgs else 0.0
        
        err_maxs = [r.execution.object_max_tracking_error for r in rs if r.execution]
        max_err = np.max(err_maxs) if err_maxs else 0.0
        
        # Collisions
        colls = [r.execution.n_collision_incidents for r in rs if r.execution]
        m_coll = np.mean(colls) if colls else 0.0
        
        row = (
            f"{exp:<35} {cfg:<30} {n_iters:>5} {succ_rate:>7.1f}% "
            f"{cost_str:>20} {m_hos:>5.1f} {_fmt(m_err, '.4f'):>8} {_fmt(max_err, '.4f'):>8} {m_coll:>5.1f}"
        )
        print(row)
    print()


def plot_results(groups: Dict[Tuple[str, str], List[ExperimentResult]], out_dir: Path, show: bool = False):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exps = sorted(list(set(k[0] for k in groups.keys())))
    configs = sorted(list(set(k[1] for k in groups.keys())))
    
    if not exps or not configs: return

    # 1. Success Rate
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(exps))
    width = 0.8 / len(configs)
    
    for i, cfg in enumerate(configs):
        rates = []
        for exp in exps:
            rs = groups.get((exp, cfg), [])
            if not rs:
                rates.append(0)
                continue
            successes = sum(1 for r in rs if r.execution and r.execution.task_success)
            rates.append((successes / len(rs)) * 100)
        ax.bar(x + i*width, rates, width, label=cfg)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Task Success Rate by Experiment and Configuration')
    ax.set_xticks(x + width * (len(configs)-1) / 2)
    ax.set_xticklabels(exps, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    fig.savefig(out_dir / "aggregate_success_rates.png")
    plt.close(fig)

    # 2. Tracking Error Boxplot
    fig, ax = plt.subplots(figsize=(14, 7))
    error_data = []
    labels = []
    for exp in exps:
        all_errs = []
        for cfg in configs:
            rs = groups.get((exp, cfg), [])
            errs = [r.execution.object_mean_tracking_error for r in rs if r.execution]
            all_errs.extend(errs)
        if all_errs:
            error_data.append(all_errs)
            labels.append(exp)
    
    if error_data:
        ax.boxplot(error_data, tick_labels=labels)
        ax.set_ylabel('Mean Tracking Error (m)')
        ax.set_title('Object Tracking Error Distribution across Iterations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(out_dir / "aggregate_tracking_errors.png")
        plt.close(fig)

    logger.info(f"Plots saved to {out_dir}")


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

    groups = group_results(results)
    print_summary_table(groups)
    plot_results(groups, out_dir, not args.no_show)
    logger.info("Analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
