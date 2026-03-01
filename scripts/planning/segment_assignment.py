#!/usr/bin/env python3
"""
Optimal segment assignment planner for dual-arm manipulation.

Loads pre-generated trajectory data produced by ``trajectory_generation.py``,
computes the globally optimal robot-to-segment assignment using dynamic
programming, and saves the result for execution by the handover script.

By default, creates a simulation environment so that IK and Jacobian functions
can be used for accurate handover-switching and manipulability costs.  Use
``--no-env`` for a quick dry-run that skips environment creation (falls back to
simpler cost heuristics).

Workflow::

    trajectory_generation.py  -->  data/<experiment>/
        └── segment_assignment.py  -->  data/<experiment>/assignment_results.npz
            └── dual_arm_handover_traj.py  (reads assignment)

Usage::

    python scripts/planning/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json

    # Override cost weights from the command line:
    python scripts/planning/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json \\
        --alpha 15.0 --beta 2.0

    # Quick dry-run without environment (no IK / Jacobian):
    python scripts/planning/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json --no-env

    # Specify a custom data directory:
    python scripts/planning/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json \\
        --data-dir data/flatPlate_table_to_shelf
"""

import argparse
import logging
import os
import sys
import threading

import numpy as np

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.experiment_config import (
    load_experiment,
    get_collision_filter_fn,
)
from robosuite_franka_manipulation.planning.assignment_planner import (
    compute_optimal_robot_assignment,
    load_all_trajectories,
    make_ik_solver_fn,
    make_jacobian_fns,
    save_assignment_result,
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import (
    visualize_trajectory_feasibility,
    visualize_assignment_result,
)
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation.utils.env_utils import env_step


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


logger = logging.getLogger(__name__)


def _parse_transfer_feasibility(raw):
    """Convert the JSON-friendly transfer_feasibility dict to tuple-keyed dict.

    JSON keys are strings like ``"0,3,1,5"`` mapping to bool values.
    Non-dict values (e.g. ``true`` / ``false``) are treated as *no constraint*
    and return ``None`` with a warning.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "transfer_feasibility should be a dict or null, got %s — ignoring.",
            type(raw).__name__,
        )
        return None
    out = {}
    for key_str, val in raw.items():
        parts = tuple(int(x) for x in key_str.split(","))
        if len(parts) != 4:
            logger.warning("Ignoring malformed transfer_feasibility key: %s", key_str)
            continue
        out[parts] = bool(val)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute optimal segment assignment for dual-arm manipulation",
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        help="Path to experiment JSON config.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing trajectory NPZ files.  Default: data/<experiment_name>",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Smoothness weight (overrides JSON)")
    parser.add_argument("--beta", type=float, default=None, help="Manipulability weight (overrides JSON)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualizations")
    parser.add_argument(
        "--no-env", action="store_true",
        help="Skip environment creation (disables IK / Jacobian — faster but less accurate)",
    )

    args = parser.parse_args()

    # --- Load experiment config ---
    exp_cfg = load_experiment(args.experiment)
    asgn_cfg = exp_cfg.assignment

    debug = args.debug or exp_cfg.runtime.debug
    setup_logging(logging.DEBUG if debug else logging.INFO)

    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    # Resolve data directory
    data_dir = args.data_dir
    if data_dir is None:
        if exp_cfg.output.output_dir is not None:
            data_dir = exp_cfg.output.output_dir
        else:
            data_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)

    # --- Load trajectories ---
    logger.info("Loading trajectories from %s ...", data_dir)
    trajectories = load_all_trajectories(data_dir)

    # --- Optional feasibility heatmap ---
    if not args.no_plot:
        heatmap_path = os.path.join(data_dir, "trajectory_feasibility.png")
        logger.info("Generating feasibility heatmap ...")
        visualize_trajectory_feasibility(trajectories, save_path=heatmap_path)

    # --- Resolve assignment parameters (CLI > JSON > defaults) ---
    alpha = args.alpha if args.alpha is not None else asgn_cfg.alpha
    beta = args.beta if args.beta is not None else asgn_cfg.beta
    epsilon = asgn_cfg.epsilon
    initial_robot = asgn_cfg.initial_robot
    transfer_feasibility = _parse_transfer_feasibility(asgn_cfg.transfer_feasibility)

    logger.info("Assignment parameters: α=%.2f, β=%.2f, ε=%.1e", alpha, beta, epsilon)

    # --- Build IK / Jacobian closures from the simulation environment ---
    ik_solver_fn = None
    jacobian_fns = None

    if not args.no_env:
        logger.info("Creating simulation environment for IK and Jacobian evaluation ...")
        env = make_dual_arm_env(
            ycb_objects=exp_cfg.ycb_objects,
            custom_objects=exp_cfg.custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            table_full_size=exp_cfg.table_full_size,
        )
        env.reset()
        for _ in range(10):
            env_step(env)

        # Disable irrelevant collisions for speed
        _is_relevant = get_collision_filter_fn(exp_cfg.collision_filter)
        model = env.sim.model
        for i in range(model.ngeom):
            name = model.geom_id2name(i) or ""
            if not _is_relevant(name):
                model.geom_contype[i] = 0

        env_lock = threading.RLock()
        tf_manager = TransformManager(env, env_lock=env_lock)

        ik_solver_fn = make_ik_solver_fn(env, tf_manager, env_lock)
        jacobian_fns = make_jacobian_fns(env, env_lock)
        logger.info("IK solver and Jacobian functions ready.")
    else:
        logger.warning(
            "Environment creation skipped (--no-env).  Handover costs will use "
            "fallback heuristic; manipulability will use simple joint-distance proxy."
        )

    # --- Run assignment planner ---
    result = compute_optimal_robot_assignment(
        trajectories,
        transfer_feasibility=transfer_feasibility,
        initial_robot=initial_robot,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        ik_solver_fn=ik_solver_fn,
        jacobian_fns=jacobian_fns,
    )

    # --- Print summary ---
    _print_summary(result)

    # --- Save result ---
    out_path = os.path.join(data_dir, "assignment_results.npz")
    save_assignment_result(result, out_path)

    # --- Optional assignment bar chart ---
    if not args.no_plot:
        chart_path = os.path.join(data_dir, "assignment_result.png")
        visualize_assignment_result(result, save_path=chart_path)

    logger.info("Done.")


def _print_summary(result) -> None:
    """Log a human-readable summary of the assignment result."""
    logger.info("=" * 60)
    logger.info("Optimal Assignment Results")
    logger.info("=" * 60)
    logger.info("  Total cost     : %.4f", result.total_cost)
    logger.info("  Segments       : %d", len(result.sequence))
    logger.info("  Handovers      : %d  %s", len(result.handover_points), result.handover_points)

    robot_names = {0: "Robot0", 1: "Robot1"}
    for i, robot_id in enumerate(result.sequence):
        traj_idx = result.trajectory_indices[i]
        tc = result.task_costs[i]
        sc = result.switching_costs[i]
        traj_str = f"{traj_idx:3d}" if traj_idx >= 0 else "N/A"
        logger.info(
            "  Seg %3d: %s  traj=%s  task=%.4f  switch=%.4f  total=%.4f",
            i, robot_names[robot_id], traj_str, tc, sc, tc + sc,
        )

    # Per-robot feasibility summary
    cm = result.cost_matrix
    for rid in range(2):
        finite = np.isfinite(cm[:, rid])
        logger.info(
            "  %s — feasible segments: %d/%d  cost range: [%.4f, %.4f]",
            robot_names[rid], finite.sum(), len(cm),
            np.nanmin(cm[:, rid]) if finite.any() else float("inf"),
            np.nanmax(cm[finite, rid]) if finite.any() else float("inf"),
        )


if __name__ == "__main__":
    main()
