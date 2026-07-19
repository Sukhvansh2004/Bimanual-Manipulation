#!/usr/bin/env python3
"""Visualize an assignment result in MeshCat and generate diagnostic plots.

Usage:
    python scripts/utils/visualize_assignment.py \\
        data/flatPlate_box_to_box/assignment_results.npz \\
        --experiment experiments/flatPlate_box_to_box.json

Features:
    1. MeshCat 3D visualisation of the assigned EEF path, object trajectory,
       and handover waypoints — colour-coded by robot.
    2. Terminal log: per-waypoint joint configs, angular deltas, handover
       transfer configs, feasibility, and cost breakdown.
    3. Matplotlib plots saved to ``--output-dir``:
       - Per-joint angles across waypoints (with handover markers).
       - Per-joint angular step (|Δq|) with the max-step threshold.
       - Task cost and switching cost per segment.
       - Cost-matrix heatmap (per-robot).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — always safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import meshcat
import meshcat.geometry as g
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.planning.assignment.utils import (
    load_assignment_result,
)
from bimanual_manipulation.planning.assignment.result import AssignmentResult
from bimanual_manipulation.transforms import (
    StampedPose,
    TransformManager,
    stream_transforms_to_meshcat,
    MeshcatTFVisualizerConfig,
)
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.object_factory import register_mesh_extraction_env
from bimanual_manipulation.experiment_config import load_experiment
from bimanual_manipulation.planning.visualization import RobotMeshVisualizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Robot colours used throughout
ROBOT_COLORS = {0: 0x1F77B4, 1: 0xFF7F0E}  # blue / orange
ROBOT_LABELS = {0: "Robot 0 (left)", 1: "Robot 1 (right)"}
HANDOVER_COLOR = 0xE74C3C  # red
OBJECT_COLOR = 0xAA00FF  # purple

JOINT_NAMES = [f"j{i}" for i in range(7)]


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize an assignment NPZ in MeshCat with diagnostic plots.",
    )
    p.add_argument("assignment", help="Path to assignment_results.npz")
    p.add_argument(
        "--experiment", required=True,
        help="Path to experiment JSON (needed for env / TF manager).",
    )
    p.add_argument("--output-dir", default=None,
                   help="Directory for saved plots (default: alongside NPZ).")
    p.add_argument("--no-meshcat", action="store_true",
                   help="Skip MeshCat 3D visualisation.")
    p.add_argument("--zmq-url", default=None, help="MeshCat ZMQ URL.")
    p.add_argument("--clear", action="store_true",
                   help="Clear MeshCat scene before drawing.")
    p.add_argument("--step", type=int, default=1,
                   help="Show robot mesh every N-th waypoint (default: 1 = all).")
    p.add_argument("--opacity", type=float, default=0.4,
                   help="Robot mesh opacity (default: 0.4).")
    return p.parse_args()


# ── Terminal log ───────────────────────────────────────────────────────────

def print_assignment_summary(a: AssignmentResult) -> None:
    """Print a compact human-readable summary to the terminal."""
    n_seg = len(a.sequence)
    n_wp = n_seg + 1
    dof = a.normalized_configs.shape[1]

    print("\n" + "=" * 80)
    print("  ASSIGNMENT SUMMARY")
    print("=" * 80)
    print(f"  Segments:       {n_seg}")
    print(f"  Waypoints:      {n_wp}")
    print(f"  DOF:            {dof}")
    print(f"  Total cost:     {a.total_cost:.4f}")
    print(f"  Handover count: {len(a.handover_points)}")
    print(f"  Handover segs:  {a.handover_points}")
    print(f"  Sequence:       {a.sequence}")
    print()

    # Per-segment table
    hdr = f"{'Seg':>4}  {'Robot':>5}  {'Traj':>28}  {'Task':>8}  {'Switch':>8}  {'Feasible':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i in range(n_seg):
        feas_start = "Y" if a.is_feasible[i] else "N"
        feas_end = "Y" if a.is_feasible[i + 1] else "N"
        feas_str = f"{feas_start},{feas_end}"
        print(
            f"{i:4d}  {a.sequence[i]:5d}  {a.trajectory_paths[i]:>28s}  "
            f"{a.task_costs[i]:8.4f}  {a.switching_costs[i]:8.4f}  {feas_str:>8s}"
        )
    print()


def print_joint_configs(a: AssignmentResult) -> None:
    """Print per-waypoint joint configs and per-step deltas."""
    n_wp, dof = a.normalized_configs.shape
    configs = a.normalized_configs

    print("=" * 80)
    print("  PER-WAYPOINT JOINT CONFIGS  (rad)")
    print("=" * 80)
    hdr_parts = [f"{'WP':>4}", f"{'Robot':>5}"]
    hdr_parts += [f"{JOINT_NAMES[j]:>8}" for j in range(dof)]
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))
    for i in range(n_wp):
        # Which robot owns this waypoint?
        seg = min(i, len(a.sequence) - 1)
        rid = a.sequence[seg]
        vals = "  ".join(f"{configs[i, j]:8.4f}" for j in range(dof))
        marker = " *HO*" if i in a.handover_points else ""
        print(f"{i:4d}  {rid:5d}  {vals}{marker}")
    print()

    # Angular deltas
    print("=" * 80)
    print("  PER-STEP ANGULAR DELTA  |Δq|  (rad)")
    print("=" * 80)
    hdr_parts = [f"{'Step':>6}"]
    hdr_parts += [f"{JOINT_NAMES[j]:>8}" for j in range(dof)]
    hdr_parts += [f"{'||Δq||':>8}", f"{'max|Δq|':>8}"]
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))
    max_step = constants.ASSIGNMENT_MAX_JOINT_STEP
    for i in range(1, n_wp):
        dq = configs[i] - configs[i - 1]
        vals = "  ".join(f"{abs(dq[j]):8.4f}" for j in range(dof))
        norm = float(np.linalg.norm(dq))
        maxd = float(np.max(np.abs(dq)))
        flag = " !" if maxd > max_step else ""
        marker = " *HO*" if i in a.handover_points else ""
        print(f"{i - 1:3d}→{i:<3d}{vals}  {norm:8.4f}  {maxd:8.4f}{flag}{marker}")
    print(f"\n  (Max-step threshold = {max_step:.2f} rad.  '!' = exceeds threshold.)")
    print()


def print_handover_details(a: AssignmentResult) -> None:
    """Print handover transfer configs and deltas."""
    if not a.handover_points:
        print("  No handovers.\n")
        return

    dof = a.normalized_configs.shape[1]
    print("=" * 80)
    print("  HANDOVER TRANSFER CONFIGS")
    print("=" * 80)

    for hi, hp in enumerate(a.handover_points):
        transfer_cfg = a.handover_transfer_configs[hi]
        recv_cfg = a.normalized_configs[hp]
        prev_cfg = a.normalized_configs[hp - 1]
        dq_approach = transfer_cfg - prev_cfg

        print(f"\n  Handover #{hi} at segment {hp}  "
              f"(robot {a.sequence[hp - 1]} → robot {a.sequence[hp]})")
        print(f"    Transferor config @ wp {hp}:   "
              + "  ".join(f"{v:8.4f}" for v in transfer_cfg))
        print(f"    Transferor config @ wp {hp-1}: "
              + "  ".join(f"{v:8.4f}" for v in prev_cfg))
        print(f"    Δq (approach to hold):        "
              + "  ".join(f"{abs(v):8.4f}" for v in dq_approach)
              + f"  ||Δq||={np.linalg.norm(dq_approach):.4f}")
        print(f"    Receiver config   @ wp {hp}:   "
              + "  ".join(f"{v:8.4f}" for v in recv_cfg))
    print()


def print_cost_breakdown(a: AssignmentResult) -> None:
    """Print cost matrix summary."""
    n_seg = len(a.sequence)
    print("=" * 80)
    print("  COST MATRIX  (best-per-robot)")
    print("=" * 80)
    print(f"{'Seg':>4}  {'Robot0':>10}  {'Robot1':>10}  {'Chosen':>8}  {'Margin':>10}")
    print("-" * 52)
    for i in range(n_seg):
        c0 = a.cost_matrix[i, 0]
        c1 = a.cost_matrix[i, 1]
        chosen = a.sequence[i]
        chosen_cost = c0 if chosen == 0 else c1
        other_cost = c1 if chosen == 0 else c0
        margin = other_cost - chosen_cost
        print(f"{i:4d}  {c0:10.4f}  {c1:10.4f}  {chosen:8d}  {margin:10.4f}")
    print()


# ── Matplotlib plots ──────────────────────────────────────────────────────

def _segment_colors(a: AssignmentResult) -> List[str]:
    """Return a per-waypoint colour list for matplotlib."""
    n_wp = a.normalized_configs.shape[0]
    colors = []
    for i in range(n_wp):
        seg = min(i, len(a.sequence) - 1)
        colors.append("#1f77b4" if a.sequence[seg] == 0 else "#ff7f0e")
    return colors


def plot_joint_angles(a: AssignmentResult, out_dir: str) -> None:
    """Per-joint angles vs waypoint, colour-coded by robot."""
    configs = a.normalized_configs
    n_wp, dof = configs.shape
    wp_idx = np.arange(n_wp)

    fig, axes = plt.subplots(dof, 1, figsize=(12, 2.2 * dof), sharex=True)
    if dof == 1:
        axes = [axes]

    colors = _segment_colors(a)

    for j, ax in enumerate(axes):
        # Draw segments as coloured spans
        for i in range(n_wp - 1):
            ax.plot(
                [wp_idx[i], wp_idx[i + 1]],
                [configs[i, j], configs[i + 1, j]],
                color=colors[i], linewidth=1.5,
            )
        ax.scatter(wp_idx, configs[:, j], c=colors, s=12, zorder=3)

        # Mark handovers
        for hp in a.handover_points:
            ax.axvline(hp, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        # Joint limits
        ax.axhline(constants.FRANKA_JOINT_LIMITS_LO[j], color="grey",
                    linestyle=":", alpha=0.4)
        ax.axhline(constants.FRANKA_JOINT_LIMITS_HI[j], color="grey",
                    linestyle=":", alpha=0.4)

        ax.set_ylabel(f"j{j} (rad)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Waypoint index")
    axes[0].set_title("Joint Angles per Waypoint  (red dashed = handover)")

    # Legend
    legend_elems = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="Robot 0"),
        Line2D([0], [0], color="#ff7f0e", lw=2, label="Robot 1"),
        Line2D([0], [0], color="red", ls="--", lw=1, label="Handover"),
    ]
    axes[0].legend(handles=legend_elems, fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(out_dir, "joint_angles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_joint_deltas(a: AssignmentResult, out_dir: str) -> None:
    """Per-joint |Δq| between consecutive waypoints."""
    configs = a.normalized_configs
    n_wp, dof = configs.shape
    dq = np.abs(np.diff(configs, axis=0))  # (n_wp-1, dof)
    step_idx = np.arange(n_wp - 1)

    fig, axes = plt.subplots(dof, 1, figsize=(12, 2.2 * dof), sharex=True)
    if dof == 1:
        axes = [axes]

    colors = _segment_colors(a)[:-1]  # one fewer than waypoints

    for j, ax in enumerate(axes):
        ax.bar(step_idx, dq[:, j], color=colors, width=0.8, alpha=0.8)

        # Threshold line
        ax.axhline(constants.ASSIGNMENT_MAX_JOINT_STEP, color="red",
                    linestyle="-", alpha=0.6, linewidth=1, label="max step")

        for hp in a.handover_points:
            # The step entering the handover is hp-1 → hp
            if 0 <= hp - 1 < len(step_idx):
                ax.axvline(hp - 0.5, color="red", linestyle="--", alpha=0.5,
                           linewidth=0.8)

        ax.set_ylabel(f"|Δj{j}| (rad)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step  (wp i → i+1)")
    axes[0].set_title("Per-Joint Angular Step  (red line = max-step threshold)")

    fig.tight_layout()
    path = os.path.join(out_dir, "joint_deltas.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_costs(a: AssignmentResult, out_dir: str) -> None:
    """Stacked bar: task cost + switching cost per segment."""
    n_seg = len(a.sequence)
    seg_idx = np.arange(n_seg)
    colors = ["#1f77b4" if r == 0 else "#ff7f0e" for r in a.sequence]

    fig, ax = plt.subplots(figsize=(max(8, n_seg * 0.35), 4))
    ax.bar(seg_idx, a.task_costs, color=colors, alpha=0.8, label="task cost")
    ax.bar(seg_idx, a.switching_costs, bottom=a.task_costs, color="red",
           alpha=0.6, label="switching cost")

    for hp in a.handover_points:
        ax.axvline(hp - 0.5, color="red", linestyle="--", alpha=0.4)

    ax.set_xlabel("Segment")
    ax.set_ylabel("Cost")
    ax.set_title(f"Per-Segment Cost Breakdown  (total = {a.total_cost:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    path = os.path.join(out_dir, "costs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_cost_matrix(a: AssignmentResult, out_dir: str) -> None:
    """Heatmap of the cost matrix (best cost per robot per segment)."""
    cm = a.cost_matrix.copy()
    # Clamp inf for display
    finite_max = np.nanmax(cm[np.isfinite(cm)]) if np.any(np.isfinite(cm)) else 1.0
    cm[~np.isfinite(cm)] = finite_max * 1.5

    fig, ax = plt.subplots(figsize=(4, max(4, len(a.sequence) * 0.25)))
    im = ax.imshow(cm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Robot")
    ax.set_ylabel("Segment")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Robot 0", "Robot 1"])
    ax.set_title("Cost Matrix  (best per-robot)")
    fig.colorbar(im, ax=ax, shrink=0.6)

    # Mark chosen robot per segment
    for i, rid in enumerate(a.sequence):
        ax.plot(rid, i, marker="o", color="blue", markersize=6)

    fig.tight_layout()
    path = os.path.join(out_dir, "cost_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_eef_3d(a: AssignmentResult, out_dir: str) -> None:
    """3D scatter of EEF positions, colour-coded by robot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    pos = a.eef_positions
    n_wp = pos.shape[0]
    colors = _segment_colors(a)

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=18, zorder=3)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="grey", alpha=0.3, linewidth=0.8)

    # Mark handovers
    for hp in a.handover_points:
        ax.scatter(
            pos[hp, 0], pos[hp, 1], pos[hp, 2],
            marker="x", color="red", s=80, zorder=5,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EEF Positions  (x = handover)")
    fig.tight_layout()

    path = os.path.join(out_dir, "eef_3d.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── MeshCat visualisation ─────────────────────────────────────────────────

def _hex_to_rgb_float(h: int):
    return ((h >> 16) & 0xFF) / 255, ((h >> 8) & 0xFF) / 255, (h & 0xFF) / 255


def visualize_in_meshcat(
    a: AssignmentResult,
    vis: meshcat.Visualizer,
    tf_manager: TransformManager,
    step: int = 1,
    opacity: float = 0.4,
) -> None:
    """Draw the assignment in MeshCat."""
    root = f"{constants.VIS_PLANNING_ROOT}/assignment"
    vis[root].delete()

    n_wp = a.eef_positions.shape[0]
    n_seg = len(a.sequence)

    # ── 1.  EEF path per contiguous robot run ──────────────────────────
    runs: list = []
    run_start = 0
    for i in range(1, n_seg):
        if a.sequence[i] != a.sequence[run_start]:
            runs.append((a.sequence[run_start], run_start, i))
            run_start = i
    runs.append((a.sequence[run_start], run_start, n_seg))

    for rid, seg_s, seg_e in runs:
        # Waypoint range is [seg_s, seg_e] (inclusive endpoints)
        wp_s = seg_s
        wp_e = seg_e  # one past the last segment → last waypoint
        eef = a.eef_positions[wp_s:wp_e + 1]

        color = ROBOT_COLORS[rid]
        path = f"{root}/eef/run_{seg_s}_{seg_e}"

        # Line
        pts = eef.T  # (3, N)
        vis[f"{path}/line"].set_object(
            g.Line(g.PointsGeometry(pts),
                   g.LineBasicMaterial(linewidth=3, color=color)),
        )

        # Waypoint spheres (colour-coded feasibility)
        for wi in range(wp_s, wp_e + 1):
            local_i = wi - wp_s
            wp_path = f"{path}/wp_{wi}"
            sphere_color = (constants.VIS_COLOR_FEASIBLE
                            if a.is_feasible[wi] else constants.VIS_COLOR_INFEASIBLE)
            vis[f"{wp_path}/sphere"].set_object(
                g.Sphere(0.008),
                g.MeshBasicMaterial(color=sphere_color, opacity=0.85),
            )
            T = np.eye(4)
            T[:3, 3] = a.eef_positions[wi]
            vis[wp_path].set_transform(T)

    # ── 2.  Object trajectory ──────────────────────────────────────────
    obj_pts = a.object_positions.T  # (3, N)
    vis[f"{root}/object/line"].set_object(
        g.Line(g.PointsGeometry(obj_pts),
               g.LineBasicMaterial(linewidth=2, color=OBJECT_COLOR)),
    )

    # ── 3.  Handover markers ──────────────────────────────────────────
    for hi, hp in enumerate(a.handover_points):
        hp_path = f"{root}/handovers/ho_{hi}"
        # Large red sphere at EEF
        vis[f"{hp_path}/eef"].set_object(
            g.Sphere(0.015),
            g.MeshBasicMaterial(color=HANDOVER_COLOR, opacity=0.9),
        )
        T = np.eye(4)
        T[:3, 3] = a.eef_positions[hp]
        vis[f"{hp_path}/eef"].set_transform(T)

        # Object position at handover
        vis[f"{hp_path}/obj"].set_object(
            g.Sphere(0.012),
            g.MeshBasicMaterial(color=0xFF00FF, opacity=0.7),
        )
        T_obj = np.eye(4)
        T_obj[:3, 3] = a.object_positions[hp]
        vis[f"{hp_path}/obj"].set_transform(T_obj)

    # ── 4.  Robot meshes at waypoint configs ───────────────────────
    robot_vis = RobotMeshVisualizer(
        vis=vis,
        tf_manager=tf_manager,
        base_path=f"{root}/robot_configs",
    )

    n_wp = a.normalized_configs.shape[0]
    for wi in range(n_wp):
        if wi % step != 0 and wi != n_wp - 1:
            continue
        seg = min(wi, n_seg - 1)
        rid = a.sequence[seg]
        # Use feasibility-based colour: green if feasible, red if not
        color = (constants.VIS_COLOR_FEASIBLE
                 if a.is_feasible[wi] else constants.VIS_COLOR_INFEASIBLE)
        wp_path = f"{root}/robot_configs/wp_{wi:04d}"
        robot_vis.visualize_robot_at_config(
            robot_id=rid,
            joint_config=a.normalized_configs[wi],
            path=wp_path,
            color=color,
            opacity=opacity,
        )

    logger.info(
        "MeshCat: drew %d runs, %d handovers, robot meshes (step=%d) under '%s'",
        len(runs), len(a.handover_points), step, root,
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load assignment ────────────────────────────────────────────────
    if not os.path.isfile(args.assignment):
        logger.error("Assignment file not found: %s", args.assignment)
        sys.exit(1)

    logger.info("Loading assignment from %s …", args.assignment)
    assignment = load_assignment_result(args.assignment)
    logger.info(
        "Loaded: %d segments, %d waypoints, %d handovers, total_cost=%.4f",
        len(assignment.sequence),
        assignment.normalized_configs.shape[0],
        len(assignment.handover_points),
        assignment.total_cost,
    )

    # ── Terminal logs ──────────────────────────────────────────────────
    print_assignment_summary(assignment)
    print_joint_configs(assignment)
    print_handover_details(assignment)
    print_cost_breakdown(assignment)

    # ── Plots ──────────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.assignment))
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving plots to %s", out_dir)

    plot_joint_angles(assignment, out_dir)
    plot_joint_deltas(assignment, out_dir)
    plot_costs(assignment, out_dir)
    plot_cost_matrix(assignment, out_dir)
    plot_eef_3d(assignment, out_dir)

    # ── MeshCat ────────────────────────────────────────────────────────
    if args.no_meshcat:
        logger.info("MeshCat visualisation skipped (--no-meshcat).")
        return

    exp_cfg = load_experiment(args.experiment)
    zmq_url = (
        args.zmq_url
        or getattr(exp_cfg.visualization, "zmq_url", None)
        or str(constants.MESHCAT_ZMQ_URL)
    )

    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if args.clear:
        vis.delete()

    # Spin up lightweight env for TF manager
    env = make_dual_arm_env(
        ycb_objects=exp_cfg.ycb_objects,
        custom_objects=exp_cfg.custom_objects,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        table_full_size=exp_cfg.table_full_size,
        table_offset=exp_cfg.table_offset,
        robot_base_configs=exp_cfg.robot_base_configs,
    )
    for _ in range(10):
        env_step(env)

    register_mesh_extraction_env(env)
    tf_manager = TransformManager(env=env)

    # Stream scene + TFs once
    env_lock = threading.Lock()
    stream_cfg = MeshcatTFVisualizerConfig(clear=False, visualize_scene=True)
    threading.Thread(
        target=stream_transforms_to_meshcat,
        kwargs={
            "env": env, "env_lock": env_lock,
            "tf_manager": tf_manager, "config": stream_cfg,
        },
        daemon=True,
    ).start()

    time.sleep(1)  # let the TF thread publish once

    visualize_in_meshcat(
        assignment, vis, tf_manager,
        step=args.step, opacity=args.opacity,
    )

    logger.info(
        "MeshCat running at %s — press Ctrl-C to quit.", zmq_url,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
