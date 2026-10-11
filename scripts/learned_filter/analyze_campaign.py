#!/usr/bin/env python3
"""Failure taxonomy and learned-filter comparison for campaign results.

Reads ``results/campaign/<task>/s<seed>/<config>.json`` records written by
``campaign.py`` and produces:

* a per-config outcome table where every instance gets exactly one label:
  ``success``, ``execution:<mode>``, ``planning:reachability_gap``,
  ``planning:no_handover_window``, ``planning:grasp_continuity``,
  ``planning:joint_continuity``,
  ``planning:no_feasible_handover``,
  ``planning:search_pruning``,
  ``planning:timeout`` or ``planning:other``;
* a learned-filter table (planning time, speedup, success, cost gap,
  skipped handover evaluations) for every ``*_learned*`` config against
  its exact counterpart on the same instances;
* a stacked-bar taxonomy figure.

Usage::

    python scripts/learned_filter/analyze_campaign.py --results results/campaign
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from bimanual_manipulation import constants  # noqa: E402
from bimanual_manipulation.planning.assignment.utils import (  # noqa: E402
    normalize_trajectory_configs,
)

OUTCOME_ORDER = [
    "success",
    "execution:handover_drop",
    "execution:grasp",
    "execution:transport",
    "execution:place",
    "execution:other",
    "planning:search_pruning",
    "planning:no_feasible_handover",
    "planning:no_handover_window",
    "planning:grasp_continuity",
    "planning:joint_continuity",
    "planning:reachability_gap",
    "planning:timeout",
    "planning:other",
]

TASK_LABELS = {
    "flatPlate_table_to_shelf": "A",
    "ycb_plate_shelf_to_table": "B",
    "mug_inversion": "C",
    "flatPlate_box_to_box": "D",
}


def arm_coverage(data_dir: Path) -> Optional[np.ndarray]:
    """Per-arm waypoint reachability under the sampled grasps.

    Args:
        data_dir: Directory with ``robot*_trajectory_*.npz`` files.

    Returns:
        ``(2, N)`` bool array; entry ``[r, i]`` is ``True`` when at least one
        candidate trajectory of robot *r* is feasible at waypoint *i*.
        ``None`` if either arm has no candidates.
    """
    per_arm = []
    for r in (0, 1):
        feas = [np.load(f)["is_feasible"].astype(bool)
                for f in sorted(data_dir.glob(f"robot{r}_trajectory_*.npz"))]
        if not feas:
            return None
        n = min(len(f) for f in feas)
        per_arm.append(np.any(np.stack([f[:n] for f in feas]), axis=0))
    n = min(len(a) for a in per_arm)
    return np.stack([a[:n] for a in per_arm])


def waypoint_coverage(data_dir: Path) -> Tuple[float, int]:
    """Measure how much of the trajectory the sampled grasps can reach.

    Args:
        data_dir: Directory with ``robot*_trajectory_*.npz`` files.

    Returns:
        ``(coverage, first_uncovered)``: fraction of waypoints feasible for
        at least one candidate of either arm, and the first uncovered
        waypoint (``-1`` if all are covered).
    """
    cov = arm_coverage(data_dir)
    if cov is None:
        return 0.0, 0
    covered = cov.any(axis=0)
    uncovered = np.flatnonzero(~covered)
    return float(covered.mean()), int(uncovered[0]) if len(uncovered) else -1


def needs_handover_without_window(data_dir: Path) -> bool:
    """Whether the path admits no switching sequence even under a relaxation.

    Runs a reachability DP over waypoints that ignores grasp identity and
    continuity: arm *r* can hold the object at waypoint *i* if it reaches
    *i* and held it at ``i - 1``, or if the other arm held it and both arms
    reach *i* (a handover window). If neither arm can hold the object at the
    final waypoint, no plan can exist regardless of search.

    Args:
        data_dir: Directory with ``robot*_trajectory_*.npz`` files.

    Returns:
        ``True`` when the relaxed problem is infeasible although every
        waypoint is reachable by some arm.
    """
    cov = arm_coverage(data_dir)
    if cov is None or not cov.any(axis=0).all():
        return False
    both = cov[0] & cov[1]
    hold = cov[:, 0].copy()
    for i in range(1, cov.shape[1]):
        stay = hold & cov[:, i]
        switch = both[i] and bool(hold.any())
        hold = stay | (cov[:, i] & switch)
        if not hold.any():
            return True
    return False


def _load_trajectory_masks(data_dir: Path, use_joint_steps: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-arm waypoint feasibility and same-trajectory continuation masks.

    Args:
        data_dir: Directory with ``robot*_trajectory_*.npz`` files.
        use_joint_steps: Apply the planner's joint-step limit to continuations
            (after the same angle normalisation the solver uses).

    Returns:
        For each arm, ``(feasible, can_continue)`` arrays of shape
        ``(n_trajectories, N)``; ``can_continue[k, i]`` says trajectory *k*
        may continue from waypoint ``i - 1`` to ``i``.
    """
    masks = []
    for r in (0, 1):
        feas, cont = [], []
        for f in sorted(data_dir.glob(f"robot{r}_trajectory_*.npz")):
            d = np.load(f)
            is_feasible = d["is_feasible"].astype(bool)
            step_ok = np.ones_like(is_feasible)
            if use_joint_steps:
                _, step_ok = normalize_trajectory_configs(
                    d["joint_configs"], d["is_feasible"],
                    constants.FRANKA_JOINT_LIMITS_LO, constants.FRANKA_JOINT_LIMITS_HI,
                    constants.ASSIGNMENT_MAX_JOINT_STEP,
                )
            feas.append(is_feasible)
            cont.append(is_feasible & np.asarray(step_ok, dtype=bool))
        masks.append((np.stack(feas), np.stack(cont)))
    return masks


def trajectory_relaxation_infeasible(data_dir: Path, use_joint_steps: bool = False) -> bool:
    """Whether no plan exists even when handovers are assumed always feasible.

    Runs the assignment graph at trajectory level -- a state is one sampled
    grasp trajectory, which can continue while it stays feasible or hand over
    to any feasible trajectory of the other arm at the same waypoint -- but
    skips handover IK/collision checks.

    Args:
        data_dir: Directory with ``robot*_trajectory_*.npz`` files.
        use_joint_steps: Also enforce the planner's joint-step limit on
            same-trajectory continuations.

    Returns:
        ``True`` when no trajectory sequence reaches the final waypoint.
    """
    masks = _load_trajectory_masks(data_dir, use_joint_steps)
    n = min(f.shape[1] for f, _ in masks)
    alive = [f[:, 0] for f, _ in masks]
    for i in range(1, n):
        cont = [a & c[:, i] for a, (_, c) in zip(alive, masks)]
        # A handover into arm r at waypoint i needs the other arm holding the object at i.
        recv = [masks[r][0][:, i] & bool(cont[1 - r].any()) for r in (0, 1)]
        alive = [c | h for c, h in zip(cont, recv)]
        if not (alive[0].any() or alive[1].any()):
            return True
    return False


def execution_mode(ex: Optional[dict]) -> str:
    """Label the outcome of a simulated execution.

    Args:
        ex: Execution metrics dict (``None`` if execution produced none).

    Returns:
        ``"success"`` or an ``"execution:<mode>"`` label from
        :data:`OUTCOME_ORDER`.
    """
    if not ex:
        return "execution:other"
    if ex.get("task_success"):
        return "success"
    if not ex.get("grasp_success", True):
        return "execution:grasp"
    if ex.get("n_handovers_lost_object", 0) > 0 or ex.get("handover_success_rate", 1.0) < 1.0:
        return "execution:handover_drop"
    if not ex.get("transport_success", True):
        return "execution:transport"
    if not ex.get("place_success", True):
        return "execution:place"
    return "execution:other"


def classify(rec: dict, exact_rec: Optional[dict], data_dir: Path) -> str:
    """Assign exactly one outcome label to an instance record.

    Args:
        rec: Campaign record for one (task, seed, config).
        exact_rec: Exact-DP record for the same instance, used to detect
            search-pruning failures (``None`` if unavailable).
        data_dir: Trajectory directory of the instance.

    Returns:
        An outcome label from :data:`OUTCOME_ORDER`.
    """
    if rec.get("planning_success"):
        return execution_mode(rec.get("execution"))
    reason = rec.get("failure_reason", "")
    if reason == "timeout":
        return "planning:timeout"
    # A pruned/filtered search that fails where exact DP succeeds is a search failure.
    if exact_rec is not None and exact_rec is not rec and exact_rec.get("planning_success"):
        return "planning:search_pruning"
    coverage, _ = waypoint_coverage(data_dir)
    if coverage < 1.0:
        return "planning:reachability_gap"
    if needs_handover_without_window(data_dir):
        return "planning:no_handover_window"
    if trajectory_relaxation_infeasible(data_dir):
        return "planning:grasp_continuity"
    if trajectory_relaxation_infeasible(data_dir, use_joint_steps=True):
        return "planning:joint_continuity"
    if reason.startswith(("dp_dead_end", "no_terminal_state", "no_feasible_start")):
        return "planning:no_feasible_handover"
    return "planning:other"


def load_records(root: Path) -> Dict[Tuple[str, int], Dict[str, dict]]:
    """Load all campaign records under *root*.

    Args:
        root: Campaign results directory.

    Returns:
        ``{(task, seed): {config_name: record}}``.
    """
    out: Dict[Tuple[str, int], Dict[str, dict]] = defaultdict(dict)
    for f in root.glob("*/s*/*.json"):
        if f.name.endswith(("_planning.json", "_exec.json")):
            continue
        rec = json.loads(f.read_text())
        out[(rec["task"], int(rec["seed"]))][rec["config"]] = rec
    return out


def exact_counterpart(config: str) -> str:
    """Return the unfiltered config a ``*_learned*`` config is compared to.

    Args:
        config: Config name, e.g. ``"beam_learned_s08"``.

    Returns:
        ``"beam"`` for beam variants, otherwise ``"exact"``.
    """
    return "beam" if config.startswith("beam") else "exact"


def taxonomy(
    records: Dict[Tuple[str, int], Dict[str, dict]], data_root: Path,
) -> Dict[str, Counter]:
    """Count outcome labels per config.

    Args:
        records: Output of :func:`load_records`.
        data_root: Root of the per-instance trajectory directories.

    Returns:
        ``{config_name: Counter(outcome_label -> n_instances)}``.
    """
    tax: Dict[str, Counter] = defaultdict(Counter)
    for (task, seed), cfgs in sorted(records.items()):
        data_dir = data_root / task / f"s{seed}"
        for name, rec in cfgs.items():
            # Beam and learned variants are compared against exact DP on the same instance.
            ref = cfgs.get("exact")
            tax[name][classify(rec, ref, data_dir)] += 1
    return tax


def print_taxonomy(tax: Dict[str, Counter]) -> None:
    """Print the outcome taxonomy as a Markdown table.

    Args:
        tax: Output of :func:`taxonomy`.
    """
    print("\n## Outcome taxonomy (instances)\n")
    cols = [o for o in OUTCOME_ORDER if any(c[o] for c in tax.values())]
    print("| config | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name in sorted(tax):
        c = tax[name]
        n = sum(c.values())
        cells = [f"{c[o]} ({c[o] / n:.0%})" if c[o] else "–" for o in cols]
        print(f"| {name} | {n} | " + " | ".join(cells) + " |")


def exact_eval_count(root: Path, task: str, seed: int, config: str) -> Optional[int]:
    """Number of exact handover evaluations reported in a planning log.

    Args:
        root: Campaign results directory.
        task: Task name.
        seed: Instance seed.
        config: Config name.

    Returns:
        The solver's "total IK evals" count, or ``None`` if unavailable.
    """
    log = root / task / f"s{seed}" / f"{config}_assignment.log"
    if not log.exists():
        return None
    match = re.search(r"total IK evals=(\d+)", log.read_text(errors="ignore"))
    return int(match.group(1)) if match else None


def filter_comparison(records: Dict[Tuple[str, int], Dict[str, dict]], root: Path) -> None:
    """Print learned-filter configs against their unfiltered counterparts.

    Speed and cost are compared only on *solvable* instances (the unfiltered
    planner finds a plan); instances that no planner can solve are excluded
    because every variant fails on them quickly. Reports how often the filter
    loses a plan the unfiltered planner finds, task success, median speedup,
    median reduction in exact handover evaluations, and the cost gap.

    Args:
        records: Output of :func:`load_records`.
        root: Campaign results directory (for planning logs).
    """
    rows: Dict[str, List[dict]] = defaultdict(list)
    for (task, seed), cfgs in records.items():
        for name, rec in cfgs.items():
            if "_learned" not in name:
                continue
            ref_name = exact_counterpart(name)
            ref = cfgs.get(ref_name)
            if ref is None or not ref.get("planning_success"):
                continue
            p, pr = rec.get("planning") or {}, ref.get("planning") or {}
            ok = bool(rec.get("planning_success"))
            n_eval = exact_eval_count(root, task, seed, name)
            n_eval_ref = exact_eval_count(root, task, seed, ref_name)
            rows[name].append({
                "ok": ok,
                "exec_ok": bool((rec.get("execution") or {}).get("task_success")),
                "exec_ok_ref": bool((ref.get("execution") or {}).get("task_success")),
                "speedup": (pr["planning_time_s"] / p["planning_time_s"]) if ok else np.nan,
                "eval_cut": (1 - n_eval / n_eval_ref) if ok and n_eval is not None and n_eval_ref else np.nan,
                "cost_gap": (p["total_cost"] / pr["total_cost"] - 1) if ok else np.nan,
            })
    if not rows:
        return
    print("\n## Learned filter vs. unfiltered planner (instances the unfiltered planner solves)\n")
    print("| config | n | plan kept | task succ | ref task succ | speedup (median, min–max) "
          "| exact evals cut (median) | cost gap (mean / max) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, rs in sorted(rows.items()):
        sp = np.array([r["speedup"] for r in rs], dtype=float)
        cut = np.array([r["eval_cut"] for r in rs], dtype=float)
        gap = np.array([r["cost_gap"] for r in rs], dtype=float)
        sp_txt = (f"{np.nanmedian(sp):.2f}× ({np.nanmin(sp):.2f}–{np.nanmax(sp):.2f})"
                  if np.isfinite(sp).any() else "–")
        cut_txt = f"{np.nanmedian(cut):.0%}" if np.isfinite(cut).any() else "–"
        gap_txt = f"{np.nanmean(gap):+.2%} / {np.nanmax(gap):+.2%}" if np.isfinite(gap).any() else "–"
        print(f"| {name} | {len(rs)} | {sum(r['ok'] for r in rs)}/{len(rs)} "
              f"| {np.mean([r['exec_ok'] for r in rs]):.0%} | {np.mean([r['exec_ok_ref'] for r in rs]):.0%} "
              f"| {sp_txt} | {cut_txt} | {gap_txt} |")


KAPPA_CONFIGS = {
    0.0: "exact_learned_k0",
    0.65: "exact_learned_k065",
    0.8: "exact_learned_k080",
    0.9: "exact_learned_k090",
    1.0: "exact_learned_k100",
}
"""Cost-slack value -> config name of the transformer-filtered exact DP runs."""

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # validated categorical slots 1-3
SERIES_MARKERS = ("o", "s", "^")
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


OUTCOME_COLORS = {
    "success": "#2a78d6",
    "execution:handover_drop": "#eb6834",
    "execution:place": "#1baf7a",
    "planning:search_pruning": "#eda100",
    "planning:no_handover_window": "#e87ba4",
    "planning:joint_continuity": "#6250d6",
    "planning:reachability_gap": "#e34948",
    "planning:grasp_continuity": "#008300",
}
"""Fixed outcome -> color mapping (validated categorical slots); other outcomes are gray."""

OTHER_COLOR = "#9a9890"
SURFACE = "#fcfcfb"
TAXONOMY_CONFIGS = {"beam": "Beam (10, stride 3)", "exact": "Exact DP"}


def plot_taxonomy(tax: Dict[str, Counter], path: Path) -> None:
    """Save the outcome taxonomy of the unfiltered planners as stacked bars.

    Only the configs in :data:`TAXONOMY_CONFIGS` are plotted, since they run
    on every instance; filtered and baseline configs run on subsets.

    Args:
        tax: Output of :func:`taxonomy`.
        path: Output figure path.
    """
    names = [c for c in TAXONOMY_CONFIGS if c in tax]
    if not names:
        return
    cols = [o for o in OUTCOME_ORDER if any(tax[n][o] for n in names)]
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    left = np.zeros(len(names))
    y = np.arange(len(names))[::-1]
    for o in cols:
        counts = np.array([tax[n][o] for n in names], dtype=float)
        color = OUTCOME_COLORS.get(o, OTHER_COLOR)
        ax.barh(y, counts, left=left, height=0.6, color=color, edgecolor=SURFACE,
                linewidth=2, label=o.split(":")[-1].replace("_", " "))
        for yi, li, ci in zip(y, left, counts):
            if ci:
                ax.text(li + ci / 2, yi, f"{int(ci)}", ha="center", va="center",
                        fontsize=7, color="white" if o == "success" or o.endswith("continuity") else TEXT_PRIMARY)
        left += counts
    ax.set_yticks(y, [TAXONOMY_CONFIGS[n] for n in names], fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("instances", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, left.max())
    ax.set_xticks(range(0, int(left.max()) + 1, 5))
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.45, -0.32), labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved taxonomy figure to {path}")


def kappa_sweep(
    records: Dict[Tuple[str, int], Dict[str, dict]], root: Path, path: Path,
) -> None:
    """Print and plot speedup / exact-evaluation cut against the cost slack.

    Uses the instances that unfiltered exact DP solves and that have a run
    for every slack in :data:`KAPPA_CONFIGS`.

    Args:
        records: Output of :func:`load_records`.
        root: Campaign results directory (for planning logs).
        path: Output figure path.
    """
    series = []
    for (task, seed), cfgs in sorted(records.items()):
        ref = cfgs.get("exact")
        if not (ref and ref.get("planning_success")) or not all(c in cfgs for c in KAPPA_CONFIGS.values()):
            continue
        n_ref = exact_eval_count(root, task, seed, "exact")
        points = []
        for kappa, name in KAPPA_CONFIGS.items():
            rec = cfgs[name]
            if not rec.get("planning_success"):
                points.append((kappa, np.nan, np.nan, np.nan, False))
                continue
            n = exact_eval_count(root, task, seed, name)
            points.append((
                kappa,
                ref["planning"]["planning_time_s"] / rec["planning"]["planning_time_s"],
                1 - n / n_ref if n is not None and n_ref else np.nan,
                rec["planning"]["total_cost"] / ref["planning"]["total_cost"] - 1,
                True,
            ))
        series.append((f"{TASK_LABELS.get(task, task)} (s{seed})", points))
    if not series:
        return

    print(f"\n## Cost-slack sweep ({len(series)} solvable held-out instances)\n")
    print("| kappa | plans kept | speedup (median, min–max) | exact evals cut (median) | max cost gap |")
    print("|---|---|---|---|---|")
    for j, kappa in enumerate(KAPPA_CONFIGS):
        col = [pts[j] for _, pts in series]
        sp = np.array([c[1] for c in col])
        cut = np.array([c[2] for c in col])
        gap = np.array([c[3] for c in col])
        kept = sum(c[4] for c in col)
        print(f"| {kappa:.2f} | {kept}/{len(col)} | {np.nanmedian(sp):.2f}× ({np.nanmin(sp):.2f}–{np.nanmax(sp):.2f}) "
              f"| {np.nanmedian(cut):.0%} | {np.nanmax(gap):+.2%} |")

    kappas = np.array(list(KAPPA_CONFIGS))
    xpos = np.arange(len(kappas))  # evenly spaced ordinal positions
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panels = ((axes[0], 1, "Speedup over exact DP (×)"), (axes[1], 2, "Exact handover checks cut (%)"))
    for ax, k, ylabel in panels:
        scale = 100.0 if k == 2 else 1.0
        for (label, pts), color, marker in zip(series, SERIES_COLORS, SERIES_MARKERS):
            y = np.array([p[k] for p in pts]) * scale
            ax.plot(xpos, y, color=color, lw=2, marker=marker, ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
            if k == 1:
                ax.annotate(label, (xpos[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7, color=TEXT_SECONDARY)
        median = np.nanmedian([[p[k] for p in pts] for _, pts in series], axis=0) * scale
        ax.plot(xpos, median, color=TEXT_PRIMARY, lw=2, ls="--", label="median", zorder=4)
        ax.set_xlabel(r"cost slack $\kappa$", color=TEXT_SECONDARY, fontsize=8)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
        ax.set_xticks(xpos, [f"{k:g}" for k in kappas])
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlim(-0.3, xpos[-1] + (1.1 if k == 1 else 0.3))
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right", labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nSaved cost-slack figure to {path}")


def baseline_comparison(records: Dict[Tuple[str, int], Dict[str, dict]]) -> None:
    """Compare planners on the instances where all of them were run.

    Plan costs are not comparable across configs with different objectives
    (the regrasp-graph baseline minimises handover count), so quality is
    reported through shared metrics: number of switches, mean manipulability
    along the plan, and simulated execution outcome.

    Args:
        records: Output of :func:`load_records`.
    """
    configs = ["exact", "beam", "regrasp_graph", "exact_learned_k065", "beam_learned_k065"]
    instances = [cfgs for cfgs in records.values() if all(c in cfgs for c in configs)]
    if not instances:
        return
    print(f"\n## Planner comparison on {len(instances)} test instances\n")
    print("| config | plan succ | task succ | handover succ | switches | mean manip. | plan time (s) |")
    print("|---|---|---|---|---|---|---|")
    for c in configs:
        recs = [cfgs[c] for cfgs in instances]
        planned = [r for r in recs if r.get("planning_success")]
        ex = [r.get("execution") or {} for r in planned]
        ho = [e["handover_success_rate"] for e in ex if e.get("n_handovers_attempted")]
        n_task = sum(bool(e.get("task_success")) for e in ex)
        cells = [
            c,
            f"{len(planned)}/{len(recs)}",
            f"{n_task}/{len(recs)}",
            f"{np.mean(ho):.0%}" if ho else "–",
        ]
        if planned:
            cells += [
                f"{np.mean([r['planning']['n_handovers'] for r in planned]):.2f}",
                f"{np.mean([r['planning']['mean_manipulability'] for r in planned]):.3f}",
                f"{np.mean([r['planning']['planning_time_s'] for r in planned]):.0f}",
            ]
        else:
            cells += ["–", "–", "–"]
        print("| " + " | ".join(cells) + " |")


def handover_stats(root: Path) -> None:
    """Print exact-DP handover evaluations broken down by outcome category.

    Args:
        root: Campaign results directory.
    """
    cats: Counter = Counter()
    for f in root.glob("*/s*/handover_log_exact.npz"):
        cats.update(np.load(f)["category"].tolist())
    n = sum(cats.values())
    if not n:
        return
    print(f"\n## Exact handover evaluations (n={n})\n")
    for k, v in cats.most_common():
        print(f"- {k}: {v} ({v / n:.1%})")


def per_task_success(records: Dict[Tuple[str, int], Dict[str, dict]]) -> None:
    """Print planning / end-to-end success per config and task.

    Args:
        records: Output of :func:`load_records`.
    """
    print("\n## Per-task success (planning / end-to-end)\n")
    table: Dict[str, Dict[str, List[Tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))
    for (task, _), cfgs in records.items():
        for name, rec in cfgs.items():
            table[name][task].append((
                bool(rec.get("planning_success")),
                bool((rec.get("execution") or {}).get("task_success")),
            ))
    tasks = sorted({t for v in table.values() for t in v}, key=lambda t: TASK_LABELS.get(t, t))
    print("| config | " + " | ".join(f"{TASK_LABELS.get(t, t)}" for t in tasks) + " | all |")
    print("|---|" + "---|" * (len(tasks) + 1))
    for name in sorted(table):
        cells, allv = [], []
        for t in tasks:
            v = table[name].get(t, [])
            allv += v
            cells.append(f"{np.mean([a for a, _ in v]):.0%} / {np.mean([b for _, b in v]):.0%} (n={len(v)})"
                         if v else "–")
        cells.append(f"{np.mean([a for a, _ in allv]):.0%} / {np.mean([b for _, b in allv]):.0%}")
        print(f"| {name} | " + " | ".join(cells) + " |")


def main() -> None:
    """Load campaign results and print all tables and the taxonomy figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results/campaign")
    ap.add_argument("--data", default="data/campaign")
    ap.add_argument("--figure", default="results/campaign/taxonomy.pdf")
    args = ap.parse_args()

    root = Path(args.results)
    records = load_records(root)
    print(f"Loaded {sum(len(v) for v in records.values())} records from {len(records)} instances")
    per_task_success(records)
    tax = taxonomy(records, Path(args.data))
    print_taxonomy(tax)
    handover_stats(root)
    filter_comparison(records, root)
    baseline_comparison(records)
    kappa_sweep(records, root, Path(args.figure).with_name("kappa_sweep.pdf"))
    if tax:
        plot_taxonomy(tax, Path(args.figure))


if __name__ == "__main__":
    main()
