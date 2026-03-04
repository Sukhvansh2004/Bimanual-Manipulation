#!/usr/bin/env python3
"""Analysis and visualization of experiment/ablation results.

Generates:
  1. Summary tables (console + LaTeX + CSV)
  2. Bar charts per metric across experiments
  3. Ablation comparison plots
  4. Trajectory generation overview heatmaps

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


# ═══════════════════════════════════════════════════════════════════════════
# Table formatting
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt=".2f"):
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "✓" if val else "✗"
    if isinstance(val, float):
        if np.isinf(val):
            return "∞"
        return f"{val:{fmt}}"
    return str(val)


def print_planning_table(results: List[ExperimentResult], title: str = "Planning Metrics"):
    """Print a formatted planning metrics table to console."""
    print(f"\n{'═' * 120}")
    print(f"  {title}")
    print(f"{'═' * 120}")

    header = (
        f"{'Experiment':<40} {'Config':<30} {'Cost':>10} {'Handovers':>10} "
        f"{'MeanTask':>10} {'SwitchCost':>12} {'Manip':>8} {'FeasR0':>7} "
        f"{'FeasR1':>7} {'Time(s)':>8}"
    )
    print(header)
    print("─" * 120)

    for r in sorted(results, key=lambda x: (x.experiment, x.config_label)):
        p = r.planning
        if p is None:
            continue
        row = (
            f"{r.experiment:<40} {r.config_label:<30} "
            f"{_fmt(p.total_cost, '.1f'):>10} {p.n_handovers:>10} "
            f"{_fmt(p.mean_task_cost, '.2f'):>10} {_fmt(p.total_switching_cost, '.1f'):>12} "
            f"{_fmt(p.mean_manipulability, '.4f'):>8} "
            f"{_fmt(p.feasibility_rate_robot0 * 100, '.0f'):>6}% "
            f"{_fmt(p.feasibility_rate_robot1 * 100, '.0f'):>6}% "
            f"{_fmt(p.planning_time_s, '.1f'):>8}"
        )
        print(row)
    print()


def print_execution_table(results: List[ExperimentResult], title: str = "Execution Metrics"):
    """Print a formatted execution metrics table to console."""
    has_exec = [r for r in results if r.execution is not None]
    if not has_exec:
        print(f"\n  {title}: No execution results available.\n")
        return

    print(f"\n{'═' * 110}")
    print(f"  {title}")
    print(f"{'═' * 110}")

    header = (
        f"{'Experiment':<40} {'Config':<30} {'TaskOK':>7} {'HORate':>8} "
        f"{'Collisions':>11} {'Time(s)':>8}"
    )
    print(header)
    print("─" * 110)

    for r in sorted(has_exec, key=lambda x: (x.experiment, x.config_label)):
        e = r.execution
        row = (
            f"{r.experiment:<40} {r.config_label:<30} "
            f"{_fmt(e.task_success):>7} "
            f"{_fmt(e.handover_success_rate * 100, '.0f'):>7}% "
            f"{e.n_collision_incidents:>11} "
            f"{_fmt(e.execution_time_s, '.1f'):>8}"
        )
        print(row)
    print()


def export_csv(results: List[ExperimentResult], out_path: Path):
    """Export planning + execution metrics to CSV."""
    import csv

    fields = [
        "experiment", "config_label",
        "total_cost", "n_handovers", "mean_task_cost", "total_switching_cost",
        "mean_manipulability", "min_manipulability",
        "feasibility_rate_robot0", "feasibility_rate_robot1",
        "assigned_feasibility_rate", "planning_time_s",
        "task_success", "handover_success_rate", "n_collision_incidents",
        "execution_time_s",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.experiment, x.config_label)):
            row = {"experiment": r.experiment, "config_label": r.config_label}
            if r.planning:
                for k in fields[2:12]:
                    row[k] = getattr(r.planning, k, "")
            if r.execution:
                for k in fields[12:]:
                    row[k] = getattr(r.execution, k, "")
            writer.writerow(row)

    logger.info("CSV exported: %s", out_path)


def export_latex(results: List[ExperimentResult], out_path: Path):
    """Export planning metrics as a LaTeX table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Planning Metrics Across Experiments and Ablations}",
        r"\label{tab:planning_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Experiment & Configuration & Cost & Handovers & Mean Task & Switch Cost & Manipulability & Time (s) \\",
        r"\midrule",
    ]

    for r in sorted(results, key=lambda x: (x.experiment, x.config_label)):
        p = r.planning
        if p is None:
            continue
        exp = r.experiment.replace("_", r"\_")
        cfg = r.config_label.replace("_", r"\_")
        lines.append(
            f"  {exp} & {cfg} & {p.total_cost:.1f} & {p.n_handovers} & "
            f"{p.mean_task_cost:.2f} & {p.total_switching_cost:.1f} & "
            f"{p.mean_manipulability:.4f} & {p.planning_time_s:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    logger.info("LaTeX table exported: %s", out_path)


def export_paper_table1(results: List[ExperimentResult], out_path: Path):
    """Export Table 1: Method Comparison (Greedy, Single Robot, No-Diversity, Proposed).

    Averages metrics across all experiments for each method.
    """
    methods = [
        ("Greedy (beam=1)", "Greedy"),
        ("Single Robot", "Single Robot"),
        ("No Diversity", "No Diversity"),
        ("Proposed", "Proposed"),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Method Comparison on Dual-Arm Manipulation Tasks}",
        r"\label{tab:method_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Method & Task Succ. (\%) & HO Succ. (\%) & Avg Switches & Exec Time (s) & Plan Time (s) & Manipulability & Collisions \\",
        r"\midrule",
    ]

    for config_label, display_name in methods:
        matching = [r for r in results if r.config_label == config_label]
        if not matching:
            lines.append(f"  {display_name} & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue

        # Planning metrics (always available)
        plan_results = [r for r in matching if r.planning is not None]
        avg_switches = np.mean([r.planning.n_handovers for r in plan_results]) if plan_results else 0
        avg_plan_time = np.mean([r.planning.planning_time_s for r in plan_results]) if plan_results else 0
        avg_manip = np.mean([r.planning.mean_manipulability for r in plan_results]) if plan_results else 0

        # Execution metrics (may not be available)
        exec_results = [r for r in matching if r.execution is not None]
        if exec_results:
            task_succ = np.mean([r.execution.task_success for r in exec_results]) * 100
            ho_succ = np.mean([r.execution.handover_success_rate for r in exec_results]) * 100
            avg_exec_time = np.mean([r.execution.execution_time_s for r in exec_results])
            avg_collisions = np.mean([r.execution.n_collision_incidents for r in exec_results])
        else:
            task_succ = ho_succ = avg_exec_time = avg_collisions = float("nan")

        lines.append(
            f"  {display_name} & {task_succ:.0f} & {ho_succ:.0f} & {avg_switches:.1f} "
            f"& {avg_exec_time:.1f} & {avg_plan_time:.1f} & {avg_manip:.4f} & {avg_collisions:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    logger.info("Paper Table 1 (Method Comparison) exported: %s", out_path)


def export_paper_table2(results: List[ExperimentResult], out_path: Path):
    """Export Table 2: Ablation Studies.

    Shows effect of removing C_switch, β, and feasibility filter.
    """
    ablations = [
        ("Baseline (α=10, β=1)", "Full Method (Proposed)"),
        ("No Switching Penalty (Cswitch=0)", r"$C_\text{switch}=0$"),
        ("No Manipulability (β=0)", r"$\beta=0$"),
        ("No Feasibility Filter", "No Feasibility Filter"),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation Study: Effect of Individual Components}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Configuration & Total Cost & Handovers & Manipulability & Switch Cost & Feasibility (\%) & Plan Time (s) \\",
        r"\midrule",
    ]

    for config_label, display_name in ablations:
        matching = [r for r in results if r.config_label == config_label and r.planning is not None]
        if not matching:
            lines.append(f"  {display_name} & -- & -- & -- & -- & -- & -- \\\\")
            continue

        avg_cost = np.mean([r.planning.total_cost for r in matching])
        avg_ho = np.mean([r.planning.n_handovers for r in matching])
        avg_manip = np.mean([r.planning.mean_manipulability for r in matching])
        avg_switch_cost = np.mean([r.planning.total_switching_cost for r in matching])
        avg_feas = np.mean([r.planning.assigned_feasibility_rate for r in matching]) * 100
        avg_plan_time = np.mean([r.planning.planning_time_s for r in matching])

        lines.append(
            f"  {display_name} & {avg_cost:.1f} & {avg_ho:.1f} & {avg_manip:.4f} "
            f"& {avg_switch_cost:.1f} & {avg_feas:.0f} & {avg_plan_time:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    logger.info("Paper Table 2 (Ablation Study) exported: %s", out_path)


def export_paper_per_experiment(results: List[ExperimentResult], out_path: Path):
    """Export a per-experiment breakdown table for the Proposed method.

    Shows planning + execution metrics for each experiment individually.
    """
    # Use "Proposed" config; fall back to "Baseline (α=10, β=1)"
    proposed = [r for r in results if r.config_label in ("Proposed", "Baseline (α=10, β=1)")]
    # Deduplicate by experiment, preferring "Proposed" label
    by_exp: Dict[str, ExperimentResult] = {}
    for r in proposed:
        if r.experiment not in by_exp or r.config_label == "Proposed":
            by_exp[r.experiment] = r

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-Experiment Results (Proposed Method)}",
        r"\label{tab:per_experiment}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Experiment & Cost & Handovers & Manip. & Plan (s) & Task OK & HO Rate (\%) & Collisions \\",
        r"\midrule",
    ]

    for exp_name in sorted(by_exp):
        r = by_exp[exp_name]
        exp_disp = exp_name.replace("_", r"\_")
        p = r.planning
        e = r.execution

        cost = f"{p.total_cost:.1f}" if p else "--"
        ho = str(p.n_handovers) if p else "--"
        manip = f"{p.mean_manipulability:.4f}" if p else "--"
        plan_t = f"{p.planning_time_s:.1f}" if p else "--"
        task_ok = r"\checkmark" if (e and e.task_success) else (r"\texttimes" if e else "--")
        ho_rate = f"{e.handover_success_rate * 100:.0f}" if e else "--"
        collisions = str(e.n_collision_incidents) if e else "--"

        lines.append(
            f"  {exp_disp} & {cost} & {ho} & {manip} & {plan_t} "
            f"& {task_ok} & {ho_rate} & {collisions} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    logger.info("Per-experiment table exported: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_baseline_overview(results: List[ExperimentResult], out_dir: Path, show: bool = False):
    """Bar charts of key metrics across all experiments (baseline config only)."""
    plt = _ensure_matplotlib()

    baseline = [r for r in results if "Baseline" in r.config_label and r.planning is not None]
    if not baseline:
        logger.warning("No baseline results to plot.")
        return

    baseline.sort(key=lambda r: r.experiment)
    experiments = [r.experiment for r in baseline]
    short_names = [e.replace("_", "\n", 1) for e in experiments]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Baseline Planning Metrics Across Experiments", fontsize=14, fontweight="bold")

    metrics_info = [
        ("total_cost", "Total Cost", "tab:blue"),
        ("n_handovers", "Number of Handovers", "tab:orange"),
        ("mean_task_cost", "Mean Task Cost", "tab:green"),
        ("mean_manipulability", "Mean Manipulability", "tab:red"),
        ("planning_time_s", "Planning Time (s)", "tab:purple"),
        ("assigned_feasibility_rate", "Assigned Feasibility Rate", "tab:brown"),
    ]

    for ax, (attr, label, color) in zip(axes.flat, metrics_info):
        vals = [getattr(r.planning, attr, 0) for r in baseline]
        bars = ax.bar(range(len(experiments)), vals, color=color, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_title(label)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                _fmt(v, ".1f"), ha="center", va="bottom", fontsize=6,
            )

    plt.tight_layout()
    fig.savefig(out_dir / "baseline_overview.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved baseline_overview.png")


def plot_ablation_comparison(
    results: List[ExperimentResult],
    ablation_name: str,
    config_labels: List[str],
    metric_attr: str,
    metric_label: str,
    out_dir: Path,
    show: bool = False,
):
    """Grouped bar chart comparing an ablation across experiments."""
    plt = _ensure_matplotlib()

    # Filter to matching configs
    matching = [r for r in results if r.config_label in config_labels and r.planning is not None]
    if not matching:
        logger.warning("No results for ablation '%s'", ablation_name)
        return

    experiments = sorted({r.experiment for r in matching})
    configs = config_labels  # preserve order

    n_exp = len(experiments)
    n_cfg = len(configs)
    x = np.arange(n_exp)
    width = 0.8 / max(n_cfg, 1)

    fig, ax = plt.subplots(figsize=(max(14, n_exp * 1.5), 6))

    colors = plt.cm.Set2(np.linspace(0, 1, n_cfg))

    for ci, cfg_label in enumerate(configs):
        vals = []
        for exp in experiments:
            r = next(
                (r for r in matching if r.experiment == exp and r.config_label == cfg_label),
                None,
            )
            if r and r.planning:
                vals.append(getattr(r.planning, metric_attr, 0))
            else:
                vals.append(0)
        ax.bar(x + ci * width, vals, width, label=cfg_label, color=colors[ci])

    ax.set_ylabel(metric_label)
    ax.set_title(f"Ablation: {ablation_name} — {metric_label}")
    ax.set_xticks(x + width * (n_cfg - 1) / 2)
    short_names = [e.replace("_", "\n", 1) for e in experiments]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = f"ablation_{ablation_name}_{metric_attr}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_all_ablations(results: List[ExperimentResult], out_dir: Path, show: bool = False):
    """Generate comparison plots for every ablation group × key metric."""

    key_metrics = [
        ("total_cost", "Total Cost"),
        ("n_handovers", "Number of Handovers"),
        ("mean_manipulability", "Mean Manipulability"),
        ("planning_time_s", "Planning Time (s)"),
    ]

    for abl_name, cfgs in ALL_ABLATIONS.items():
        if abl_name == "baseline":
            continue
        labels = [c["label"] for c in cfgs]
        for attr, label in key_metrics:
            plot_ablation_comparison(results, abl_name, labels, attr, label, out_dir, show)


def plot_trajectory_generation_overview(data_root: str, out_dir: Path, show: bool = False):
    """Heatmap showing feasibility rates across all experiments."""
    plt = _ensure_matplotlib()

    data_path = Path(data_root)
    experiments = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if not experiments:
        return

    stats_list = []
    for exp in experiments:
        stats = extract_trajectory_generation_metrics(str(data_path / exp), exp)
        stats_list.append(stats)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(experiments) * 0.4)))
    fig.suptitle("Trajectory Generation Feasibility Overview", fontsize=14, fontweight="bold")

    for ax_idx, (prefix, title) in enumerate([("robot0", "Robot 0"), ("robot1", "Robot 1")]):
        ax = axes[ax_idx]

        for i, stats in enumerate(stats_list):
            rates = stats[f"{prefix}_feasibility_rates"]
            if rates:
                sorted_rates = sorted(rates, reverse=True)
                colors = ["#2ecc71" if r == 1.0 else "#e74c3c" if r < 0.5 else "#f39c12"
                          for r in sorted_rates]
                ax.barh(
                    [i] * len(sorted_rates),
                    sorted_rates,
                    left=[sum(sorted_rates[:j]) / len(sorted_rates) * 0 + j / len(sorted_rates)
                          for j in range(len(sorted_rates))],
                    height=0.6,
                    color=colors,
                    alpha=0.7,
                )
                # Summary text
                n_full = stats[f"{prefix}_fully_feasible"]
                n_total = stats[f"n_{prefix}_trajs"]
                ax.text(
                    1.02, i,
                    f"{n_full}/{n_total}",
                    va="center", ha="left", fontsize=8,
                )

        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(experiments, fontsize=7)
        ax.set_xlabel("Trajectory Index")
        ax.set_title(f"{title} Trajectory Feasibility")
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_dir / "trajectory_generation_overview.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved trajectory_generation_overview.png")


def plot_handover_vs_cost(results: List[ExperimentResult], out_dir: Path, show: bool = False):
    """Scatter plot: number of handovers vs total cost, colored by config."""
    plt = _ensure_matplotlib()

    planning_results = [r for r in results if r.planning is not None]
    if not planning_results:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    configs = sorted({r.config_label for r in planning_results})
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    config_color = dict(zip(configs, colors))

    for r in planning_results:
        p = r.planning
        ax.scatter(
            p.n_handovers, p.total_cost,
            color=config_color[r.config_label],
            label=r.config_label,
            alpha=0.7, s=60,
        )

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")

    ax.set_xlabel("Number of Handovers")
    ax.set_ylabel("Total Cost")
    ax.set_title("Handover Count vs Total Assignment Cost")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "handover_vs_cost.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved handover_vs_cost.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment and ablation results")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots (save only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (default: results/figures/)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    # Load results
    results = load_all_results(results_dir)
    logger.info("Loaded %d results", len(results))

    if not results:
        logger.warning("No results found in %s. Run experiments first.", results_dir)
        logger.info("Generating trajectory generation overview only...")
        plot_trajectory_generation_overview(args.data_dir, out_dir, show)
        return

    # Console tables
    print_planning_table(results)
    print_execution_table(results)

    # Exports
    export_csv(results, out_dir / "results.csv")
    export_latex(results, out_dir / "planning_table.tex")
    export_paper_table1(results, out_dir / "paper_table1_methods.tex")
    export_paper_table2(results, out_dir / "paper_table2_ablation.tex")
    export_paper_per_experiment(results, out_dir / "paper_per_experiment.tex")

    # Plots
    plot_baseline_overview(results, out_dir, show)
    plot_all_ablations(results, out_dir, show)
    plot_trajectory_generation_overview(args.data_dir, out_dir, show)
    plot_handover_vs_cost(results, out_dir, show)

    logger.info("Analysis complete. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
