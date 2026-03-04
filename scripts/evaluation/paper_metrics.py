#!/usr/bin/env python3
"""Extract paper-ready metrics from all experiment results.

This script loads all result JSONs, computes aggregate metrics across experiments,
and prints the values needed for the paper's quantitative results and ablation tables.

Usage:
    python scripts/evaluation/paper_metrics.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"

# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

def load_result(path: Path) -> Optional[Dict[str, Any]]:
    """Load a result JSON, handling Infinity values."""
    with open(path) as f:
        text = f.read()
    # JSON doesn't support Infinity, but Python's json module may choke
    text = text.replace("Infinity", "1e9999")
    try:
        return json.loads(text)
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
        return None


def load_all_results(results_dir: Path = RESULTS_ROOT) -> List[Dict[str, Any]]:
    """Load all result JSONs (excluding *_exec.json standalone files)."""
    results = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for json_file in sorted(exp_dir.glob("*.json")):
            if json_file.name == "summary.json":
                continue
            if json_file.name.endswith("_exec.json"):
                continue  # standalone exec files are already merged into main results
            data = load_result(json_file)
            if data:
                results.append(data)
    return results


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def is_valid_planning(p: Optional[dict]) -> bool:
    """Check if planning result is valid (not inf cost / zero segments)."""
    if p is None:
        return False
    cost = p.get("total_cost", float("inf"))
    if math.isinf(cost) or cost > 1e30:
        return False
    if p.get("n_segments", 0) == 0:
        return False
    return True


def group_by_config(results: List[dict]) -> Dict[str, List[dict]]:
    """Group results by config_label."""
    groups = defaultdict(list)
    for r in results:
        groups[r["config_label"]].append(r)
    return dict(groups)


def group_by_experiment(results: List[dict]) -> Dict[str, List[dict]]:
    """Group results by experiment name."""
    groups = defaultdict(list)
    for r in results:
        groups[r["experiment"]].append(r)
    return dict(groups)


# ─────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────

def compute_method_metrics(results_for_config: List[dict]) -> dict:
    """Compute aggregated metrics for a set of results (one config, all experiments)."""
    n_total = len(results_for_config)

    # Planning metrics (only valid ones)
    valid_planning = [r for r in results_for_config if is_valid_planning(r.get("planning"))]
    n_planning_valid = len(valid_planning)

    # Execution metrics
    has_exec = [r for r in results_for_config if r.get("execution") is not None]

    metrics = {
        "n_experiments": n_total,
        "n_planning_valid": n_planning_valid,
        "planning_success_rate": n_planning_valid / max(n_total, 1) * 100,
    }

    if valid_planning:
        planning_data = [r["planning"] for r in valid_planning]
        metrics["avg_handovers"] = np.mean([p["n_handovers"] for p in planning_data])
        metrics["std_handovers"] = np.std([p["n_handovers"] for p in planning_data])
        metrics["avg_total_cost"] = np.mean([p["total_cost"] for p in planning_data])
        metrics["avg_mean_task_cost"] = np.mean([p["mean_task_cost"] for p in planning_data])
        metrics["avg_switching_cost"] = np.mean([p["total_switching_cost"] for p in planning_data])
        metrics["avg_manipulability"] = np.mean([p["mean_manipulability"] for p in planning_data])
        metrics["std_manipulability"] = np.std([p["mean_manipulability"] for p in planning_data])
        metrics["avg_min_manipulability"] = np.mean([p["min_manipulability"] for p in planning_data])
        metrics["avg_feasibility"] = np.mean([p["assigned_feasibility_rate"] for p in planning_data]) * 100
        metrics["avg_planning_time"] = np.mean([p["planning_time_s"] for p in planning_data])
    else:
        metrics["avg_handovers"] = 0
        metrics["avg_manipulability"] = 0
        metrics["avg_planning_time"] = 0

    if has_exec:
        exec_data = [r["execution"] for r in has_exec]
        n_task_success = sum(1 for e in exec_data if e.get("task_success", False))
        metrics["n_exec_total"] = len(exec_data)
        metrics["task_success_rate"] = n_task_success / len(exec_data) * 100
        metrics["avg_ho_success_rate"] = np.mean([e.get("handover_success_rate", 0) for e in exec_data]) * 100
        metrics["avg_exec_time"] = np.mean([e.get("execution_time_s", 0) for e in exec_data])
        metrics["avg_collisions"] = np.mean([e.get("n_collision_incidents", 0) for e in exec_data])
        metrics["total_collisions"] = sum(e.get("n_collision_incidents", 0) for e in exec_data)
        metrics["avg_exec_switches"] = np.mean([e.get("n_switches", 0) for e in exec_data])
        metrics["avg_exec_manipulability"] = np.mean([e.get("mean_manipulability", 0) for e in exec_data])
        n_collision_free = sum(1 for e in exec_data if e.get("n_collision_incidents", 0) == 0)
        metrics["collision_free_rate"] = n_collision_free / len(exec_data) * 100
    else:
        metrics["n_exec_total"] = 0
        metrics["task_success_rate"] = None
        metrics["avg_ho_success_rate"] = None
        metrics["avg_exec_time"] = None
        metrics["avg_collisions"] = None

    return metrics


# ─────────────────────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────────────────────

def main():
    results = load_all_results()
    print(f"Loaded {len(results)} result files from {RESULTS_ROOT}\n")

    by_config = group_by_config(results)

    print("=" * 100)
    print("  CONFIGS AVAILABLE AND EXPERIMENT COUNT")
    print("=" * 100)
    for cfg, rs in sorted(by_config.items()):
        valid = sum(1 for r in rs if is_valid_planning(r.get("planning")))
        has_exec = sum(1 for r in rs if r.get("execution") is not None)
        print(f"  {cfg:<45} total={len(rs):>2}  valid_planning={valid:>2}  has_exec={has_exec:>2}")

    # ─────────────────────────────────────────────────────────────────
    # TABLE 1: Method Comparison
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  TABLE 1: METHOD COMPARISON (aggregated across all experiments)")
    print("=" * 100)

    method_configs = [
        ("Proposed", "Proposed (Ours)"),
        ("Greedy (beam=1)", "Greedy Reachability"),
        ("Single Robot", "Single Arm"),
        ("No Feasibility Filter", "No Grasp Diversity"),
    ]

    for config_label, display_name in method_configs:
        if config_label not in by_config:
            print(f"\n--- {display_name}: NO RESULTS ---")
            continue

        m = compute_method_metrics(by_config[config_label])
        print(f"\n--- {display_name} ({config_label}) ---")
        print(f"  Experiments: {m['n_experiments']}")
        print(f"  Planning Valid: {m['n_planning_valid']} / {m['n_experiments']}  ({m['planning_success_rate']:.1f}%)")

        if m['n_planning_valid'] > 0:
            print(f"  Avg Handovers:    {m['avg_handovers']:.2f} ± {m.get('std_handovers', 0):.2f}")
            print(f"  Avg Total Cost:   {m.get('avg_total_cost', 0):.2f}")
            print(f"  Avg Manip:        {m['avg_manipulability']:.4f} ± {m.get('std_manipulability', 0):.4f}")
            print(f"  Avg Feasibility:  {m.get('avg_feasibility', 0):.1f}%")
            print(f"  Avg Plan Time:    {m.get('avg_planning_time', 0):.1f}s")

        if m['n_exec_total'] > 0:
            print(f"  Execution Trials: {m['n_exec_total']}")
            print(f"  Task Success:     {m['task_success_rate']:.1f}%")
            print(f"  HO Success Rate:  {m['avg_ho_success_rate']:.1f}%")
            print(f"  Avg Exec Time:    {m['avg_exec_time']:.1f}s")
            print(f"  Avg Collisions:   {m['avg_collisions']:.2f}")
            print(f"  Exec Manip:       {m.get('avg_exec_manipulability', 0):.4f}")
        else:
            print(f"  Execution: No execution results")

    # ─────────────────────────────────────────────────────────────────
    # ABLATION: Performance vs Number of Switches
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  ABLATION: PERFORMANCE vs NUMBER OF SWITCHES")
    print("=" * 100)

    # Use Proposed + No_Switching_Penalty results to categorize by number of switches
    proposed_results = by_config.get("Proposed", [])
    all_exec_results = []
    for cfg in by_config.values():
        for r in cfg:
            if r.get("execution") is not None and is_valid_planning(r.get("planning")):
                all_exec_results.append(r)

    # Categorize Proposed runs by number of switches
    proposed_with_exec = [r for r in proposed_results if r.get("execution") is not None]
    if proposed_with_exec:
        switch_groups = defaultdict(list)
        for r in proposed_with_exec:
            n_sw = r["execution"].get("n_switches", r["planning"]["n_handovers"])
            switch_groups[n_sw].append(r)

        for n_sw in sorted(switch_groups.keys()):
            grp = switch_groups[n_sw]
            n_success = sum(1 for r in grp if r["execution"]["task_success"])
            succ_rate = n_success / len(grp) * 100
            avg_time = np.mean([r["execution"]["execution_time_s"] for r in grp])
            print(f"  {n_sw}-Switch: {len(grp)} experiments, Task Success = {succ_rate:.1f}%, Avg Time = {avg_time:.1f}s")

    # Also look at ALL configs to get a broader view
    print("\n  Across ALL configs with execution:")
    switch_groups_all = defaultdict(list)
    for r in all_exec_results:
        n_sw = r["execution"].get("n_switches", r["planning"]["n_handovers"])
        switch_groups_all[n_sw].append(r)

    for n_sw in sorted(switch_groups_all.keys()):
        grp = switch_groups_all[n_sw]
        n_success = sum(1 for r in grp if r["execution"]["task_success"])
        succ_rate = n_success / len(grp) * 100
        avg_time = np.mean([r["execution"]["execution_time_s"] for r in grp])
        avg_ho_rate = np.mean([r["execution"]["handover_success_rate"] for r in grp]) * 100
        print(f"  {n_sw}-Switch: n={len(grp)}, Task Success={succ_rate:.1f}%, HO Rate={avg_ho_rate:.1f}%, Avg Time={avg_time:.1f}s")

    # ─────────────────────────────────────────────────────────────────
    # ABLATION 1: Without Switching Penalty (C_switch = 0)
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  ABLATION 1: WITHOUT SWITCHING PENALTY (C_switch = 0)")
    print("=" * 100)

    no_switch_label = "No Switching Penalty (Cswitch=0)"
    if no_switch_label in by_config:
        m_ns = compute_method_metrics(by_config[no_switch_label])
        m_prop = compute_method_metrics(by_config["Proposed"])

        print(f"\n  Proposed method:")
        print(f"    Avg Handovers: {m_prop['avg_handovers']:.2f}")
        print(f"    Avg Exec Time: {m_prop.get('avg_exec_time', 'N/A')}")

        print(f"\n  C_switch = 0:")
        print(f"    Avg Handovers: {m_ns['avg_handovers']:.2f}")
        print(f"    Avg Exec Time: {m_ns.get('avg_exec_time', 'N/A')}")

        # Per-experiment comparison where both have valid planning
        print("\n  Per-experiment comparison (Proposed vs C_switch=0):")
        by_exp_prop = {r["experiment"]: r for r in by_config["Proposed"]}
        by_exp_ns = {r["experiment"]: r for r in by_config[no_switch_label]}

        common_exps = set(by_exp_prop.keys()) & set(by_exp_ns.keys())
        switch_diffs = []
        time_diffs = []
        for exp in sorted(common_exps):
            rp = by_exp_prop[exp]
            rn = by_exp_ns[exp]
            if is_valid_planning(rp.get("planning")) and is_valid_planning(rn.get("planning")):
                ho_p = rp["planning"]["n_handovers"]
                ho_n = rn["planning"]["n_handovers"]
                switch_diffs.append(ho_n - ho_p)
                print(f"    {exp:<40} Proposed HO={ho_p}  NoSwitch HO={ho_n}  Δ={ho_n - ho_p:+d}")

                # Execution time comparison
                if rp.get("execution") and rn.get("execution"):
                    tp = rp["execution"]["execution_time_s"]
                    tn = rn["execution"]["execution_time_s"]
                    pct = (tn - tp) / tp * 100 if tp > 0 else 0
                    time_diffs.append(pct)
                    print(f"      Exec time: Proposed={tp:.1f}s  NoSwitch={tn:.1f}s  Δ={pct:+.1f}%")

        if switch_diffs:
            print(f"\n    Summary: Avg additional handovers with C_switch=0: {np.mean(switch_diffs):.2f}")
            print(f"    Avg handovers (C_switch=0): {m_ns['avg_handovers']:.1f}")
        if time_diffs:
            print(f"    Avg execution time increase: {np.mean(time_diffs):+.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # ABLATION 2: Without Manipulability Cost (β = 0)
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  ABLATION 2: WITHOUT MANIPULABILITY COST (β = 0)")
    print("=" * 100)

    no_manip_label = "No Manipulability (β=0)"
    if no_manip_label in by_config:
        m_nm = compute_method_metrics(by_config[no_manip_label])
        m_prop = compute_method_metrics(by_config["Proposed"])

        print(f"\n  Proposed method:")
        print(f"    Avg Manipulability: {m_prop['avg_manipulability']:.4f}")
        if m_prop.get('task_success_rate') is not None:
            print(f"    Task Success Rate:  {m_prop['task_success_rate']:.1f}%")
        if m_prop.get('avg_exec_manipulability'):
            print(f"    Exec Manipulability: {m_prop['avg_exec_manipulability']:.4f}")

        print(f"\n  β = 0:")
        print(f"    Avg Manipulability: {m_nm['avg_manipulability']:.4f}")
        if m_nm.get('task_success_rate') is not None:
            print(f"    Task Success Rate:  {m_nm['task_success_rate']:.1f}%")
        if m_nm.get('avg_exec_manipulability'):
            print(f"    Exec Manipulability: {m_nm['avg_exec_manipulability']:.4f}")

        # Per-experiment comparison
        print("\n  Per-experiment comparison (Proposed vs β=0):")
        by_exp_prop = {r["experiment"]: r for r in by_config["Proposed"]}
        by_exp_nm = {r["experiment"]: r for r in by_config[no_manip_label]}

        common_exps = set(by_exp_prop.keys()) & set(by_exp_nm.keys())
        exec_failures = {"proposed": 0, "no_manip": 0, "total": 0}
        manip_diffs = []

        for exp in sorted(common_exps):
            rp = by_exp_prop[exp]
            rn = by_exp_nm[exp]
            if is_valid_planning(rp.get("planning")) and is_valid_planning(rn.get("planning")):
                mp = rp["planning"]["mean_manipulability"]
                mn = rn["planning"]["mean_manipulability"]
                print(f"    {exp:<40} Proposed manip={mp:.4f}  β=0 manip={mn:.4f}")

                if rp.get("execution") and rn.get("execution"):
                    sp = rp["execution"]["task_success"]
                    sn = rn["execution"]["task_success"]
                    ep = rp["execution"].get("mean_manipulability", 0)
                    en = rn["execution"].get("mean_manipulability", 0)
                    cp = rp["execution"].get("n_collision_incidents", 0)
                    cn = rn["execution"].get("n_collision_incidents", 0)
                    print(f"      TaskOK: Proposed={sp}  β=0={sn}  |  ExecManip: Proposed={ep:.4f}  β=0={en:.4f}  |  Collisions: Proposed={cp}  β=0={cn}")
                    exec_failures["total"] += 1
                    if not sp: exec_failures["proposed"] += 1
                    if not sn: exec_failures["no_manip"] += 1
                    manip_diffs.append(en - ep)

        if exec_failures["total"] > 0:
            fail_prop = exec_failures["proposed"] / exec_failures["total"] * 100
            fail_nm = exec_failures["no_manip"] / exec_failures["total"] * 100
            print(f"\n    Execution failure rate: Proposed={fail_prop:.1f}%  β=0={fail_nm:.1f}%  Δ={fail_nm - fail_prop:+.1f}%")
        if manip_diffs:
            print(f"    Avg exec manipulability change (β=0 - Proposed): {np.mean(manip_diffs):+.4f}")

    # ─────────────────────────────────────────────────────────────────
    # ABLATION 3: Without Grasp Pair Feasibility Filtering
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  ABLATION 3: WITHOUT GRASP PAIR FEASIBILITY FILTERING")
    print("=" * 100)

    no_feas_label = "No Feasibility Filter"
    if no_feas_label in by_config:
        m_nf = compute_method_metrics(by_config[no_feas_label])

        n_total_nf = len(by_config[no_feas_label])
        n_valid_nf = m_nf["n_planning_valid"]
        n_infeasible = n_total_nf - n_valid_nf
        infeasible_pct = n_infeasible / max(n_total_nf, 1) * 100

        print(f"\n  No Feasibility Filter:")
        print(f"    Total experiments: {n_total_nf}")
        print(f"    Valid planning:    {n_valid_nf}")
        print(f"    Infeasible (cost=inf): {n_infeasible} ({infeasible_pct:.0f}%)")
        print(f"    → Infeasible transitions selected by DP: {infeasible_pct:.0f}%")
        print(f"    → Task success (considering infeasible as failures): {n_valid_nf / max(n_total_nf, 1) * 100:.0f}%")

        # Also check No Diversity config
        no_div_label = "No Diversity"
        if no_div_label in by_config:
            m_nd = compute_method_metrics(by_config[no_div_label])
            print(f"\n  No Diversity (alternative label):")
            print(f"    Total experiments: {m_nd['n_experiments']}")
            print(f"    Valid planning:    {m_nd['n_planning_valid']}")
    else:
        print("  No Feasibility Filter results not found.")

    # ─────────────────────────────────────────────────────────────────
    # DETAILED PER-EXPERIMENT RESULTS (Proposed)
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  PER-EXPERIMENT RESULTS (Proposed Method)")
    print("=" * 100)

    proposed = by_config.get("Proposed", [])
    print(f"\n  {'Experiment':<40} {'Cost':>10} {'HO':>4} {'Manip':>8} {'Plan(s)':>10} {'TaskOK':>7} {'HORate':>8} {'ExecTime':>10} {'Coll':>5}")
    print("  " + "-" * 105)

    for r in sorted(proposed, key=lambda x: x["experiment"]):
        exp = r["experiment"]
        p = r.get("planning", {})
        e = r.get("execution")

        if is_valid_planning(r.get("planning")):
            cost = f"{p['total_cost']:.1f}"
            ho = str(p["n_handovers"])
            manip = f"{p['mean_manipulability']:.4f}"
            plan_t = f"{p['planning_time_s']:.1f}"
        else:
            cost = "INF"
            ho = "-"
            manip = "-"
            plan_t = f"{p.get('planning_time_s', 0):.1f}"

        if e:
            task_ok = "✓" if e["task_success"] else "✗"
            ho_rate = f"{e['handover_success_rate'] * 100:.0f}%"
            exec_t = f"{e['execution_time_s']:.1f}"
            coll = str(e["n_collision_incidents"])
        else:
            task_ok = "-"
            ho_rate = "-"
            exec_t = "-"
            coll = "-"

        print(f"  {exp:<40} {cost:>10} {ho:>4} {manip:>8} {plan_t:>10} {task_ok:>7} {ho_rate:>8} {exec_t:>10} {coll:>5}")

    # ─────────────────────────────────────────────────────────────────
    # OVERALL PROPOSED METHOD SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  SUMMARY STATISTICS FOR PROPOSED METHOD")
    print("=" * 100)

    m_prop = compute_method_metrics(proposed)
    print(f"  Total Experiments:         {m_prop['n_experiments']}")
    print(f"  Planning Valid:            {m_prop['n_planning_valid']} ({m_prop['planning_success_rate']:.1f}%)")
    print(f"  Avg Handovers:             {m_prop['avg_handovers']:.2f} ± {m_prop.get('std_handovers', 0):.2f}")
    print(f"  Avg Total Cost:            {m_prop.get('avg_total_cost', 0):.2f}")
    print(f"  Avg Mean Task Cost:        {m_prop.get('avg_mean_task_cost', 0):.4f}")
    print(f"  Avg Switching Cost:        {m_prop.get('avg_switching_cost', 0):.2f}")
    print(f"  Avg Manipulability:        {m_prop['avg_manipulability']:.4f} ± {m_prop.get('std_manipulability', 0):.4f}")
    print(f"  Avg Min Manipulability:    {m_prop.get('avg_min_manipulability', 0):.4f}")
    print(f"  Avg Feasibility Rate:      {m_prop.get('avg_feasibility', 0):.1f}%")
    print(f"  Avg Planning Time:         {m_prop.get('avg_planning_time', 0):.1f}s")

    if m_prop['n_exec_total'] > 0:
        print(f"\n  Execution Trials:          {m_prop['n_exec_total']}")
        print(f"  Task Success Rate:         {m_prop['task_success_rate']:.1f}%")
        print(f"  HO Success Rate:           {m_prop['avg_ho_success_rate']:.1f}%")
        print(f"  Avg Execution Time:        {m_prop['avg_exec_time']:.1f}s")
        print(f"  Avg Collisions:            {m_prop['avg_collisions']:.2f}")
        print(f"  Total Collisions:          {m_prop.get('total_collisions', 0)}")
        print(f"  Exec Manipulability:       {m_prop.get('avg_exec_manipulability', 0):.4f}")
        print(f"  Collision Free Rate:       {m_prop.get('collision_free_rate', 0):.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # GREEDY DETAILED ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  GREEDY BASELINE DETAILED")
    print("=" * 100)

    greedy = by_config.get("Greedy (beam=1)", [])
    m_greedy = compute_method_metrics(greedy)
    print(f"  Total Experiments:     {m_greedy['n_experiments']}")
    print(f"  Planning Valid:        {m_greedy['n_planning_valid']} ({m_greedy['planning_success_rate']:.1f}%)")
    if m_greedy['n_planning_valid'] > 0:
        print(f"  Avg Handovers:         {m_greedy['avg_handovers']:.2f}")
        print(f"  Avg Manipulability:    {m_greedy['avg_manipulability']:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # SINGLE ROBOT DETAILED
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  SINGLE ROBOT BASELINE DETAILED")
    print("=" * 100)

    single = by_config.get("Single Robot", [])
    m_single = compute_method_metrics(single)
    print(f"  Total Experiments:     {m_single['n_experiments']}")
    print(f"  Planning Valid:        {m_single['n_planning_valid']} ({m_single['planning_success_rate']:.1f}%)")

    # List which experiments Single Robot succeeded/failed
    for r in sorted(single, key=lambda x: x["experiment"]):
        valid = is_valid_planning(r.get("planning"))
        print(f"    {r['experiment']:<40} {'VALID' if valid else 'INFEASIBLE'}")

    # ─────────────────────────────────────────────────────────────────
    # COLLISION ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  COLLISION ANALYSIS ACROSS ALL CONFIGS WITH EXECUTION")
    print("=" * 100)

    for cfg_label in sorted(by_config.keys()):
        rs = by_config[cfg_label]
        exec_results = [r for r in rs if r.get("execution") is not None]
        if not exec_results:
            continue
        total_coll = sum(r["execution"].get("n_collision_incidents", 0) for r in exec_results)
        avg_coll = total_coll / len(exec_results)
        print(f"  {cfg_label:<45} n={len(exec_results):>2}  total_coll={total_coll:>3}  avg_coll={avg_coll:.2f}")

    # ─────────────────────────────────────────────────────────────────
    # LATEX-READY VALUES
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  LATEX-READY VALUES FOR PAPER")
    print("=" * 100)

    m_prop = compute_method_metrics(by_config.get("Proposed", []))
    m_ns = compute_method_metrics(by_config.get(no_switch_label, [])) if no_switch_label in by_config else {}
    m_nm = compute_method_metrics(by_config.get(no_manip_label, [])) if no_manip_label in by_config else {}
    m_nf = compute_method_metrics(by_config.get(no_feas_label, [])) if no_feas_label in by_config else {}

    print("\n  --- Performance vs Number of Switches ---")
    # Calculate from Proposed execution results
    if proposed_with_exec:
        for n_sw in sorted(switch_groups.keys()):
            grp = switch_groups[n_sw]
            n_success = sum(1 for r in grp if r["execution"]["task_success"])
            succ_rate = n_success / len(grp) * 100
            print(f"  {n_sw}-Switch & Task Success (\\%) & {succ_rate:.1f} \\\\")

    print("\n  --- Ablations ---")

    # C_switch = 0
    if m_ns:
        print(f"  $C_{{\\text{{switch}}}} = 0$ & Avg. Switches & {m_ns.get('avg_handovers', 0):.1f} \\\\")
        if m_ns.get('avg_exec_time') is not None and m_prop.get('avg_exec_time') is not None:
            time_increase = (m_ns['avg_exec_time'] - m_prop['avg_exec_time']) / m_prop['avg_exec_time'] * 100
            print(f"  $C_{{\\text{{switch}}}} = 0$ & Execution Time & +{time_increase:.0f}\\% \\\\")
        else:
            # Compare per-experiment where both have exec
            by_exp_prop = {r["experiment"]: r for r in by_config.get("Proposed", [])}
            by_exp_ns = {r["experiment"]: r for r in by_config.get(no_switch_label, [])}
            time_pcts = []
            for exp in set(by_exp_prop.keys()) & set(by_exp_ns.keys()):
                rp = by_exp_prop[exp]
                rn = by_exp_ns[exp]
                if rp.get("execution") and rn.get("execution"):
                    tp = rp["execution"]["execution_time_s"]
                    tn = rn["execution"]["execution_time_s"]
                    if tp > 0:
                        time_pcts.append((tn - tp) / tp * 100)
            if time_pcts:
                avg_pct = np.mean(time_pcts)
                print(f"  $C_{{\\text{{switch}}}} = 0$ & Execution Time & +{avg_pct:.0f}\\% \\\\")

    # β = 0
    if m_nm:
        # Calculate execution failure increase
        if m_nm.get('task_success_rate') is not None and m_prop.get('task_success_rate') is not None:
            prop_fail = 100 - m_prop['task_success_rate']
            nm_fail = 100 - m_nm['task_success_rate']
            fail_increase = nm_fail - prop_fail
            print(f"  $\\beta = 0$ & Execution Failures & +{fail_increase:.0f}\\% \\\\")
        else:
            # Compare collision/failure rates where both have exec
            by_exp_prop = {r["experiment"]: r for r in by_config.get("Proposed", [])}
            by_exp_nm = {r["experiment"]: r for r in by_config.get(no_manip_label, [])}
            prop_failures = 0
            nm_failures = 0
            total_compared = 0
            for exp in set(by_exp_prop.keys()) & set(by_exp_nm.keys()):
                rp = by_exp_prop[exp]
                rn = by_exp_nm[exp]
                if rp.get("execution") and rn.get("execution"):
                    total_compared += 1
                    if not rp["execution"]["task_success"]:
                        prop_failures += 1
                    if not rn["execution"]["task_success"]:
                        nm_failures += 1
            if total_compared > 0:
                fail_diff = (nm_failures - prop_failures) / total_compared * 100
                print(f"  $\\beta = 0$ & Execution Failures & +{fail_diff:.0f}\\% (from {total_compared} compared experiments) \\\\")

    # No Feasibility Filter
    if m_nf:
        n_total_nf = len(by_config[no_feas_label])
        n_valid_nf = m_nf["n_planning_valid"]
        infeasible_pct = (n_total_nf - n_valid_nf) / max(n_total_nf, 1) * 100
        task_succ_nf = n_valid_nf / max(n_total_nf, 1) * 100
        print(f"  No Grasp Pair Filtering & Infeasible Transitions & {infeasible_pct:.0f}\\% \\\\")
        print(f"  No Grasp Pair Filtering & Task Success (\\%) & {task_succ_nf:.0f}\\% \\\\")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
