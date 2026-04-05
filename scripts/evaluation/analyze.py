#!/usr/bin/env python3
"""Comprehensive analysis and visualization of experiment results.

Generates a detailed Markdown report and a suite of plots aggregating
results across multiple iterations.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Project root setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.evaluation.metrics import ExperimentResult
from scripts.evaluation.results_store import load_all_results, RESULTS_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ANALYZE] %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def _fmt(val, fmt=".3f"):
    if val is None: return "N/A"
    if isinstance(val, bool): return "Yes" if val else "No"
    if isinstance(val, (float, np.floating)):
        if np.isnan(val): return "N/A"
        if np.isinf(val): return "Inf"
        return f"{val:{fmt}}"
    return str(val)

class ResultAggregator:
    def __init__(self, results: List[ExperimentResult]):
        self.raw_results = results
        self.groups = defaultdict(list)
        for r in results:
            self.groups[(r.experiment, r.config_label)].append(r)
        
        self.exps = sorted(list(set(r.experiment for r in results)))
        self.configs = sorted(list(set(r.config_label for r in results)))

    def get_stats(self, exp: str, config: str):
        rs = self.groups.get((exp, config), [])
        if not rs: return None
        
        n = len(rs)
        successes = sum(1 for r in rs if r.execution and r.execution.task_success)
        
        n_assign_ok = sum(1 for r in rs if r.planning and r.planning.total_cost < 1e10)
        costs = [r.planning.total_cost for r in rs if r.planning and r.planning.total_cost < 1e10]
        hos = [r.planning.n_handovers for r in rs if r.planning and r.planning.total_cost < 1e10]
        
        mean_errs = [r.execution.object_mean_tracking_error for r in rs if r.execution]
        max_errs = [r.execution.object_max_tracking_error for r in rs if r.execution]
        colls = [r.execution.n_collision_incidents for r in rs if r.execution]
        ptime = [r.planning.planning_time_s for r in rs if r.planning]
        pos_errs = [r.execution.placement_position_error for r in rs if r.execution and r.execution.placement_position_error is not None]
        ori_errs = [r.execution.placement_orientation_error for r in rs if r.execution and r.execution.placement_orientation_error is not None]
        ho_rates = [r.execution.handover_success_rate for r in rs if r.execution]
        grasp_ok = sum(1 for r in rs if r.execution and r.execution.grasp_success)
        n_exec = sum(1 for r in rs if r.execution)

        # Grasp breakdown by gripper type
        grasp_by_gripper = defaultdict(lambda: {"total": 0, "ok": 0})
        for r in rs:
            g = getattr(r.planning, 'grasp_gripper', '') if r.planning else ''
            if g:
                grasp_by_gripper[g]["total"] += 1
                if r.execution and r.execution.grasp_success:
                    grasp_by_gripper[g]["ok"] += 1

        # Handover direction counts
        ho_dir_counts = defaultdict(lambda: {"planned": 0, "in_success": 0})
        for r in rs:
            dirs = getattr(r.planning, 'handover_directions', []) if r.planning else []
            task_ok = r.execution and r.execution.task_success
            for d in dirs:
                ho_dir_counts[d]["planned"] += 1
                if task_ok:
                    ho_dir_counts[d]["in_success"] += 1

        return {
            "n": n,
            "n_assign_ok": n_assign_ok,
            "n_exec": n_exec,
            "assign_rate": (n_assign_ok / n) * 100 if n > 0 else 0.0,
            "exec_rate": (n_exec / n_assign_ok) * 100 if n_assign_ok > 0 else 0.0,
            "success_rate": (successes / n) * 100,
            "grasp_rate": (grasp_ok / n_exec) * 100 if n_exec > 0 else 0.0,
            "ho_rate_mean": np.mean(ho_rates) * 100 if ho_rates else np.nan,
            "cost_mean": np.mean(costs) if costs else np.nan,
            "cost_std": np.std(costs) if len(costs) > 1 else 0.0,
            "ho_mean": np.mean(hos) if hos else np.nan,
            "err_mean": np.mean(mean_errs) if mean_errs else 0.0,
            "err_max": np.max(max_errs) if max_errs else 0.0,
            "coll_mean": np.mean(colls) if colls else 0.0,
            "ptime_mean": np.mean(ptime) if ptime else 0.0,
            "place_pos_mean": np.mean(pos_errs) if pos_errs else np.nan,
            "place_pos_std": np.std(pos_errs) if len(pos_errs) > 1 else 0.0,
            "place_ori_mean": np.mean(ori_errs) if ori_errs else np.nan,
            "place_ori_std": np.std(ori_errs) if len(ori_errs) > 1 else 0.0,
            "raw_costs": costs,
            "raw_errs": mean_errs,
            "raw_place_pos": pos_errs,
            "raw_place_ori": ori_errs,
            "raw_ho_rates": ho_rates,
            "raw_colls": colls,
            "grasp_by_gripper": dict(grasp_by_gripper),
            "ho_dir_counts": dict(ho_dir_counts),
        }

    def get_global_stats(self, config: str):
        """Aggregate stats for a configuration across ALL experiments."""
        rs_all = []
        for exp in self.exps:
            rs_all.extend(self.groups.get((exp, config), []))
        
        if not rs_all: return None
        
        n_total = len(rs_all)
        successes = sum(1 for r in rs_all if r.execution and r.execution.task_success)
        n_assign_ok = sum(1 for r in rs_all if r.planning and r.planning.total_cost < 1e10)
        
        costs = [r.planning.total_cost for r in rs_all if r.planning and r.planning.total_cost < 1e10]
        hos = [r.planning.n_handovers for r in rs_all if r.planning and r.planning.total_cost < 1e10]
        
        mean_errs = [r.execution.object_mean_tracking_error for r in rs_all if r.execution]
        max_errs = [r.execution.object_max_tracking_error for r in rs_all if r.execution]
        colls = [r.execution.n_collision_incidents for r in rs_all if r.execution]
        ptime = [r.planning.planning_time_s for r in rs_all if r.planning]
        pos_errs = [r.execution.placement_position_error for r in rs_all if r.execution and r.execution.placement_position_error is not None]
        ori_errs = [r.execution.placement_orientation_error for r in rs_all if r.execution and r.execution.placement_orientation_error is not None]
        ho_rates = [r.execution.handover_success_rate for r in rs_all if r.execution]
        grasp_ok = sum(1 for r in rs_all if r.execution and r.execution.grasp_success)
        n_exec = sum(1 for r in rs_all if r.execution)

        # Grasp breakdown by gripper type
        grasp_by_gripper = defaultdict(lambda: {"total": 0, "ok": 0})
        for r in rs_all:
            g = getattr(r.planning, 'grasp_gripper', '') if r.planning else ''
            if g:
                grasp_by_gripper[g]["total"] += 1
                if r.execution and r.execution.grasp_success:
                    grasp_by_gripper[g]["ok"] += 1

        # Handover direction counts
        ho_dir_counts = defaultdict(lambda: {"planned": 0, "in_success": 0})
        for r in rs_all:
            dirs = getattr(r.planning, 'handover_directions', []) if r.planning else []
            task_ok = r.execution and r.execution.task_success
            for d in dirs:
                ho_dir_counts[d]["planned"] += 1
                if task_ok:
                    ho_dir_counts[d]["in_success"] += 1

        return {
            "n": n_total,
            "n_assign_ok": n_assign_ok,
            "n_exec": n_exec,
            "assign_rate": (n_assign_ok / n_total) * 100 if n_total > 0 else 0.0,
            "exec_rate": (n_exec / n_assign_ok) * 100 if n_assign_ok > 0 else 0.0,
            "success_rate": (successes / n_total) * 100,
            "grasp_rate": (grasp_ok / n_exec) * 100 if n_exec > 0 else 0.0,
            "ho_rate_mean": np.mean(ho_rates) * 100 if ho_rates else np.nan,
            "cost_mean": np.mean(costs) if costs else np.nan,
            "ho_mean": np.mean(hos) if hos else np.nan,
            "err_mean": np.mean(mean_errs) if mean_errs else 0.0,
            "err_max": np.max(max_errs) if max_errs else 0.0,
            "coll_mean": np.mean(colls) if colls else 0.0,
            "ptime_mean": np.mean(ptime) if ptime else 0.0,
            "place_pos_mean": np.mean(pos_errs) if pos_errs else np.nan,
            "place_ori_mean": np.mean(ori_errs) if ori_errs else np.nan,
            "grasp_by_gripper": dict(grasp_by_gripper),
            "ho_dir_counts": dict(ho_dir_counts),
        }

def generate_visualizations(agg: ResultAggregator, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Global Success Rate Heatmap (Manual Impl)
    n_exp, n_cfg = len(agg.exps), len(agg.configs)
    hm_w = max(10, n_cfg * 0.7 + 3)
    hm_h = max(4, n_exp * 0.55 + 2)
    plt.figure(figsize=(hm_w, hm_h))
    data = np.zeros((n_exp, n_cfg))
    for i, exp in enumerate(agg.exps):
        for j, cfg in enumerate(agg.configs):
            stats = agg.get_stats(exp, cfg)
            data[i, j] = stats["success_rate"] if stats else 0
            
    im = plt.imshow(data, cmap="RdYlGn", aspect='auto')
    plt.colorbar(im, label="Success Rate (%)", shrink=0.8)
    plt.xticks(range(n_cfg), agg.configs, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(n_exp), agg.exps, fontsize=7)
    
    # Annotate values
    for i in range(n_exp):
        for j in range(n_cfg):
            plt.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center", color="black", fontsize=6)

    plt.title("Success Rate (%) per Experiment and Configuration")
    plt.tight_layout()
    plt.savefig(out_dir / "global_success_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Global Cost Distribution (Boxplot per Config)
    plt.figure(figsize=(14, 8))
    cost_data = []
    labels = []
    for cfg in agg.configs:
        cfg_costs = []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            if stats: cfg_costs.extend(stats["raw_costs"])
        if cfg_costs:
            cost_data.append(cfg_costs)
            labels.append(cfg)
    
    if cost_data:
        plt.boxplot(cost_data)
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel("Planning Cost")
        plt.title("Distribution of Planning Costs by Configuration")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / "global_cost_distribution.png")
        plt.close()

    # 3. Tracking Error (Average across iterations)
    plt.figure(figsize=(14, 8))
    for i, cfg in enumerate(agg.configs):
        errs = []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            errs.append(stats["err_mean"] if stats else 0)
        plt.plot(agg.exps, errs, marker='o', label=cfg, linewidth=1, markersize=4)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel("Mean Tracking Error (m)")
    plt.title("Mean Object Tracking Error per Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_tracking_errors.png")
    plt.close()

    # 4. Handover Analysis
    plt.figure(figsize=(14, 8))
    ho_means = []
    for cfg in agg.configs:
        cfg_hos = []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            if stats: cfg_hos.append(stats["ho_mean"])
        ho_means.append(np.nanmean(cfg_hos) if cfg_hos else 0)
    
    plt.bar(agg.configs, ho_means, color='teal', alpha=0.8)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel("Average Number of Handovers")
    plt.title("Average Handovers across all Experiments by Configuration")
    plt.tight_layout()
    plt.savefig(out_dir / "global_handover_summary.bar.png")
    plt.close()

    # 5. Placement Error per Experiment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for cfg in agg.configs:
        pos_vals, ori_vals, exp_labels = [], [], []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            if stats and stats["raw_place_pos"]:
                pos_vals.append(np.mean(stats["raw_place_pos"]))
                ori_vals.append(np.mean(stats["raw_place_ori"]) if stats["raw_place_ori"] else 0)
                exp_labels.append(exp)
        if pos_vals:
            ax1.plot(exp_labels, pos_vals, marker='o', label=cfg, linewidth=1, markersize=4)
            ax2.plot(exp_labels, ori_vals, marker='s', label=cfg, linewidth=1, markersize=4)

    ax1.set_ylabel("Position Error (m)")
    ax1.set_title("Placement Position Error per Experiment")
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Orientation Error (°)")
    ax2.set_title("Placement Orientation Error per Experiment")
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "global_placement_errors.png")
    plt.close()

    # 6. Assignment Success Rate Heatmap
    plt.figure(figsize=(hm_w, hm_h))
    data_assign = np.zeros((n_exp, n_cfg))
    for i, exp in enumerate(agg.exps):
        for j, cfg in enumerate(agg.configs):
            stats = agg.get_stats(exp, cfg)
            data_assign[i, j] = stats["assign_rate"] if stats else 0
    im = plt.imshow(data_assign, cmap="Blues", vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, label="Assignment Success Rate (%)", shrink=0.8)
    plt.xticks(range(n_cfg), agg.configs, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(n_exp), agg.exps, fontsize=7)
    for i in range(n_exp):
        for j in range(n_cfg):
            plt.text(j, i, f"{data_assign[i,j]:.0f}", ha="center", va="center", color="black", fontsize=6)
    plt.title("Assignment Success Rate (%) — Non-Infinity Cost")
    plt.tight_layout()
    plt.savefig(out_dir / "global_assignment_success_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6b. Execution Success Rate Heatmap (of valid assignments)
    plt.figure(figsize=(hm_w, hm_h))
    data_exec = np.zeros((n_exp, n_cfg))
    for i, exp in enumerate(agg.exps):
        for j, cfg in enumerate(agg.configs):
            stats = agg.get_stats(exp, cfg)
            data_exec[i, j] = stats["exec_rate"] if stats else 0
    im = plt.imshow(data_exec, cmap="Purples", vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, label="Execution Rate (%)", shrink=0.8)
    plt.xticks(range(n_cfg), agg.configs, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(n_exp), agg.exps, fontsize=7)
    for i in range(n_exp):
        for j in range(n_cfg):
            plt.text(j, i, f"{data_exec[i,j]:.0f}", ha="center", va="center", color="black", fontsize=6)
    plt.title("Execution Rate (%) — Valid Assignments That Were Executed")
    plt.tight_layout()
    plt.savefig(out_dir / "global_execution_success_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Grasp Success Rate Heatmap
    plt.figure(figsize=(hm_w, hm_h))
    data_grasp = np.zeros((n_exp, n_cfg))
    for i, exp in enumerate(agg.exps):
        for j, cfg in enumerate(agg.configs):
            stats = agg.get_stats(exp, cfg)
            data_grasp[i, j] = stats["grasp_rate"] if stats else 0
    im = plt.imshow(data_grasp, cmap="YlOrRd", vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, label="Grasp Success Rate (%)", shrink=0.8)
    plt.xticks(range(n_cfg), agg.configs, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(n_exp), agg.exps, fontsize=7)
    for i in range(n_exp):
        for j in range(n_cfg):
            plt.text(j, i, f"{data_grasp[i,j]:.0f}", ha="center", va="center", color="black", fontsize=6)
    plt.title("Grasp Success Rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "global_grasp_success_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Handover Success Rate Heatmap
    plt.figure(figsize=(hm_w, hm_h))
    data_ho = np.full((n_exp, n_cfg), np.nan)
    for i, exp in enumerate(agg.exps):
        for j, cfg in enumerate(agg.configs):
            stats = agg.get_stats(exp, cfg)
            if stats and not np.isnan(stats["ho_rate_mean"]):
                data_ho[i, j] = stats["ho_rate_mean"]
    masked = np.ma.array(data_ho, mask=np.isnan(data_ho))
    im = plt.imshow(masked, cmap="PuBuGn", vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, label="Handover Success Rate (%)", shrink=0.8)
    plt.xticks(range(n_cfg), agg.configs, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(n_exp), agg.exps, fontsize=7)
    for i in range(n_exp):
        for j in range(n_cfg):
            v = data_ho[i, j]
            plt.text(j, i, f"{v:.0f}" if not np.isnan(v) else "—", ha="center", va="center", color="black", fontsize=6)
    plt.title("Handover Success Rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "global_handover_success_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 9. Collision Incidents per Experiment (grouped bar)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(agg.exps))
    w = 0.8 / max(len(agg.configs), 1)
    for j, cfg in enumerate(agg.configs):
        vals = []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            vals.append(stats["coll_mean"] if stats else 0)
        ax.bar(x + j * w, vals, w, label=cfg, alpha=0.85)
    ax.set_xticks(x + w * len(agg.configs) / 2)
    ax.set_xticklabels(agg.exps, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Mean Collision Incidents")
    ax.set_title("Collision Incidents per Experiment")
    ax.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_collision_incidents.png")
    plt.close()

    # 10. Planning Time Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    for j, cfg in enumerate(agg.configs):
        vals = []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            vals.append(stats["ptime_mean"] if stats else 0)
        ax.bar(x + j * w, vals, w, label=cfg, alpha=0.85)
    ax.set_xticks(x + w * len(agg.configs) / 2)
    ax.set_xticklabels(agg.exps, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Planning Time (s)")
    ax.set_title("Average Planning Time per Experiment")
    ax.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_planning_time.png")
    plt.close()

    # 11. Stacked Pipeline Funnel — Assign → Grasp → Handover → Task
    fig, ax = plt.subplots(figsize=(14, 8))
    for j, cfg in enumerate(agg.configs):
        gs = agg.get_global_stats(cfg)
        if not gs:
            continue
        stages = [gs["assign_rate"], gs["exec_rate"], gs["grasp_rate"],
                  gs["ho_rate_mean"] if not np.isnan(gs["ho_rate_mean"]) else 0,
                  gs["success_rate"]]
        stage_labels = ["Assignment", "Execution", "Grasp", "Handover", "Task"]
        ax.plot(stage_labels, stages, marker='o', label=cfg, linewidth=2, markersize=6)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Pipeline Funnel: Success Rate at Each Stage")
    ax.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_pipeline_funnel.png")
    plt.close()

    # 12. Placement Error Boxplots (all configs side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    pos_data, ori_data, bp_labels = [], [], []
    for cfg in agg.configs:
        all_pos, all_ori = [], []
        for exp in agg.exps:
            stats = agg.get_stats(exp, cfg)
            if stats:
                all_pos.extend(stats["raw_place_pos"])
                all_ori.extend(stats["raw_place_ori"])
        if all_pos:
            pos_data.append(all_pos)
            ori_data.append(all_ori)
            bp_labels.append(cfg)
    if pos_data:
        ax1.boxplot(pos_data)
        ax1.set_xticklabels(bp_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel("Position Error (m)")
        ax1.set_title("Placement Position Error Distribution")
        ax1.grid(axis='y', alpha=0.3)
        ax2.boxplot(ori_data)
        ax2.set_xticklabels(bp_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel("Orientation Error (°)")
        ax2.set_title("Placement Orientation Error Distribution")
        ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_placement_error_boxplots.png")
    plt.close()

    # 13. Handovers vs Success Rate (valid assignments only)
    ho_buckets = defaultdict(lambda: {"total": 0, "success": 0})
    for r in agg.raw_results:
        if r.planning and r.planning.total_cost < 1e10:
            nh = r.planning.n_handovers
            ho_buckets[nh]["total"] += 1
            if r.execution and r.execution.task_success:
                ho_buckets[nh]["success"] += 1
    if ho_buckets:
        sorted_nh = sorted(ho_buckets.keys())
        rates = [ho_buckets[nh]["success"] / ho_buckets[nh]["total"] * 100 for nh in sorted_nh]
        counts = [ho_buckets[nh]["total"] for nh in sorted_nh]

        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sorted_nh))
        bars = ax1.bar(x, rates, color='steelblue', alpha=0.85, zorder=2)
        ax1.set_xlabel("Number of Handovers (planned)")
        ax1.set_ylabel("Task Success Rate (%)")
        ax1.set_ylim(0, 105)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(nh) for nh in sorted_nh])
        ax1.grid(axis='y', alpha=0.3, zorder=0)
        # Annotate bar values
        for xi, r_val, cnt in zip(x, rates, counts):
            ax1.text(xi, r_val + 2, f"{r_val:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax1.text(xi, r_val / 2, f"n={cnt}", ha='center', va='center', fontsize=8, color='white')
        ax1.set_title("Task Success Rate vs Number of Planned Handovers (Valid Assignments Only)")
        plt.tight_layout()
        plt.savefig(out_dir / "global_handovers_vs_success.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Also store for the report table
        agg._ho_vs_success = [(nh, ho_buckets[nh]["total"], ho_buckets[nh]["success"],
                               ho_buckets[nh]["success"] / ho_buckets[nh]["total"] * 100)
                              for nh in sorted_nh]

def write_markdown_report(agg: ResultAggregator, out_path: Path):
    lines = [
        "# Comprehensive Dual-Arm Franka Control Evaluation Report",
        f"\n**Scope:** {len(agg.exps)} Experiments, {len(agg.configs)} Configurations, {len(agg.raw_results)} Total Runs.",
        "\n## 1. Global Aggregate by Configuration",
        "\nThis section averages performance metrics across all experiments for each configuration.",
        "\n| Configuration | Assign % | Exec % | Grasp % | Success % | Handover % | Plan Time (s) | Avg HOs | Mean Err (m) | Max Err (m) | Place Pos Err (m) | Place Ori Err (°) | Collisions |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cfg in agg.configs:
        s = agg.get_global_stats(cfg)
        if not s: continue
        pp = _fmt(s.get('place_pos_mean'), '.4f')
        po = _fmt(s.get('place_ori_mean'), '.1f')
        hr = _fmt(s.get('ho_rate_mean'), '.1f')
        ar = _fmt(s.get('assign_rate'), '.1f')
        er = _fmt(s.get('exec_rate'), '.1f')
        gr = _fmt(s.get('grasp_rate'), '.1f')
        lines.append(f"| {cfg} | {ar}% | {er}% | {gr}% | {s['success_rate']:.1f}% | {hr}% | {s['ptime_mean']:.2f} | {s['ho_mean']:.1f} | {s['err_mean']:.4f} | {s['err_max']:.4f} | {pp} | {po} | {s['coll_mean']:.1f} |")

    lines.append("\n## 2. Detailed Summary Table")
    lines.append("\n| Experiment | Configuration | Runs | Assign % | Exec % | Grasp % | Success % | Handover % | Cost (Avg ± Std) | Mean Err (m) | Max Err (m) | Place Pos Err (m) | Place Ori Err (°) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |")
    
    for exp in agg.exps:
        for cfg in agg.configs:
            s = agg.get_stats(exp, cfg)
            if not s: continue
            cost_str = f"{s['cost_mean']:.1f} ± {s['cost_std']:.1f}"
            hr = _fmt(s.get('ho_rate_mean'), '.1f')
            ar = _fmt(s.get('assign_rate'), '.1f')
            er = _fmt(s.get('exec_rate'), '.1f')
            gr = _fmt(s.get('grasp_rate'), '.1f')
            pp = f"{s['place_pos_mean']:.4f} ± {s['place_pos_std']:.4f}" if not np.isnan(s['place_pos_mean']) else 'N/A'
            po = f"{s['place_ori_mean']:.1f} ± {s['place_ori_std']:.1f}" if not np.isnan(s['place_ori_mean']) else 'N/A'
            lines.append(f"| {exp} | {cfg} | {s['n']} | {ar}% | {er}% | {gr}% | {s['success_rate']:.1f}% | {hr}% | {cost_str} | {s['err_mean']:.4f} | {s['err_max']:.4f} | {pp} | {po} |")

    lines.append("\n## 3. Key Findings and Visualizations")
    lines.append("\n### 3.1 Success Rates")
    lines.append("![Success Heatmap](figures/global_success_heatmap.png)")
    lines.append("\n### 3.2 Planning Costs")
    lines.append("![Cost Distribution](figures/global_cost_distribution.png)")
    lines.append("\n### 3.3 Tracking Accuracy")
    lines.append("![Tracking Error](figures/global_tracking_errors.png)")
    lines.append("\n### 3.4 Handover Optimization")
    lines.append("![Handover Summary](figures/global_handover_summary.bar.png)")
    lines.append("\n### 3.5 Placement Accuracy")
    lines.append("![Placement Errors](figures/global_placement_errors.png)")
    lines.append("![Placement Error Boxplots](figures/global_placement_error_boxplots.png)")
    lines.append("\n### 3.6 Assignment Success")
    lines.append("![Assignment Success](figures/global_assignment_success_heatmap.png)")
    lines.append("\n### 3.6b Execution Success (of valid assignments)")
    lines.append("![Execution Success](figures/global_execution_success_heatmap.png)")
    lines.append("\n### 3.7 Grasp Success")
    lines.append("![Grasp Success](figures/global_grasp_success_heatmap.png)")
    lines.append("\n### 3.8 Handover Success")
    lines.append("![Handover Success](figures/global_handover_success_heatmap.png)")
    lines.append("\n### 3.9 Collision Incidents")
    lines.append("![Collisions](figures/global_collision_incidents.png)")
    lines.append("\n### 3.10 Planning Time")
    lines.append("![Planning Time](figures/global_planning_time.png)")
    lines.append("\n### 3.11 Pipeline Funnel")
    lines.append("![Pipeline Funnel](figures/global_pipeline_funnel.png)")
    lines.append("\n### 3.12 Handovers vs Success Rate")
    lines.append("![Handovers vs Success](figures/global_handovers_vs_success.png)")
    lines.append("\nSuccess rate for valid-assignment runs grouped by planned number of handovers.")
    if hasattr(agg, '_ho_vs_success') and agg._ho_vs_success:
        lines.append("\n| # Handovers | Valid Runs | Successes | Success Rate |")
        lines.append("| :---: | :---: | :---: | :---: |")
        total_runs = sum(row[1] for row in agg._ho_vs_success)
        total_succ = sum(row[2] for row in agg._ho_vs_success)
        for nh, n_total, n_succ, rate in agg._ho_vs_success:
            lines.append(f"| {nh} | {n_total} | {n_succ} | {rate:.1f}% |")
        lines.append(f"| **Total** | **{total_runs}** | **{total_succ}** | **{total_succ/total_runs*100:.1f}%** |")

    # -- 3.13 Grasp Success by Gripper Type --
    lines.append("\n### 3.13 Grasp Success by Gripper Type")
    lines.append("\n| Experiment | Config | Gripper | Attempts | Grasp OK | Rate |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for exp in agg.exps:
        for cfg in agg.configs:
            s = agg.get_stats(exp, cfg)
            if not s:
                continue
            gbg = s.get("grasp_by_gripper", {})
            for gripper in sorted(gbg.keys()):
                t = gbg[gripper]["total"]
                ok = gbg[gripper]["ok"]
                rate = (ok / t * 100) if t > 0 else 0.0
                lines.append(f"| {exp} | {cfg} | {gripper} | {t} | {ok} | {rate:.1f}% |")

    # -- 3.14 Handover Direction Breakdown --
    lines.append("\n### 3.14 Handover Direction Breakdown")
    lines.append("\n| Experiment | Config | Direction | Planned | In Successful Runs | Rate |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for exp in agg.exps:
        for cfg in agg.configs:
            s = agg.get_stats(exp, cfg)
            if not s:
                continue
            hdc = s.get("ho_dir_counts", {})
            for direction in sorted(hdc.keys()):
                p = hdc[direction]["planned"]
                ok = hdc[direction]["in_success"]
                rate = (ok / p * 100) if p > 0 else 0.0
                lines.append(f"| {exp} | {cfg} | {direction} | {p} | {ok} | {rate:.1f}% |")

    lines.append("\n## 4. Per-Experiment Details")
    for exp in agg.exps:
        lines.append(f"\n### {exp.replace('_', ' ').title()}")
        lines.append("| Config | Assign % | Exec % | Grasp % | Success % | Handover % | Avg HOs | Avg Collisions | Place Pos (m) | Place Ori (°) | Plan Time (s) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for cfg in agg.configs:
            s = agg.get_stats(exp, cfg)
            if not s: continue
            hr = _fmt(s.get('ho_rate_mean'), '.1f')
            ar = _fmt(s.get('assign_rate'), '.1f')
            er = _fmt(s.get('exec_rate'), '.1f')
            gr = _fmt(s.get('grasp_rate'), '.1f')
            pp = _fmt(s.get('place_pos_mean'), '.4f')
            po = _fmt(s.get('place_ori_mean'), '.1f')
            lines.append(f"| {cfg} | {ar}% | {er}% | {gr}% | {s['success_rate']:.1f}% | {hr}% | {s['ho_mean']:.1f} | {s['coll_mean']:.1f} | {pp} | {po} | {s['ptime_mean']:.2f} |")

    out_path.write_text("\n".join(lines))
    logger.info(f"Markdown report generated: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Detailed result analysis")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    fig_dir = out_dir / "figures"
    
    logger.info(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir)
    if not results:
        logger.error("No result files found. Ensure the iterative pipeline has completed.")
        return

    agg = ResultAggregator(results)
    
    logger.info("Generating visualizations...")
    generate_visualizations(agg, fig_dir)
    
    logger.info("Writing comprehensive report...")
    write_markdown_report(agg, out_dir / "comprehensive_report.md")
    
    # Simple console summary
    print(f"\n{'='*115}")
    print(f"{'Experiment':<35} {'Config':<30} {'Assign %':>9} {'Exec %':>8} {'Succ %':>8} {'Cost Avg':>10}")
    print(f"{'='*115}")
    for exp in agg.exps:
        for cfg in agg.configs:
            s = agg.get_stats(exp, cfg)
            if not s: continue
            print(f"{exp:<35} {cfg:<30} {s['assign_rate']:>8.1f}% {s['exec_rate']:>7.1f}% {s['success_rate']:>7.1f}% {s['cost_mean']:>10.1f}")

    logger.info(f"Analysis complete. Full report at {out_dir}/comprehensive_report.md")

if __name__ == "__main__":
    main()
