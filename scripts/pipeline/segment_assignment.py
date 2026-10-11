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

    python scripts/pipeline/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json

    # Override cost weights from the command line:
    python scripts/pipeline/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json \\
        --alpha 15.0 --beta 2.0

    # Quick dry-run without environment (no IK / Jacobian):
    python scripts/pipeline/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json --no-env

    # Specify a custom data directory:
    python scripts/pipeline/segment_assignment.py \\
        --experiment experiments/flatPlate_table_to_shelf.json \\
        --data-dir data/flatPlate_table_to_shelf
"""

import argparse
import cProfile
import io
import json
import logging
import os
import random
import pstats
import sys
import threading
import time
from pathlib import Path

import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.experiment_config import (
    load_experiment,
    get_collision_filter_fn,
)
import bimanual_manipulation.planning.assignment.solver as solver_mod
from bimanual_manipulation.planning.assignment import (
    compute_optimal_robot_assignment,
    load_all_trajectories,
    make_collision_check_fn,
    make_handover_clearance_fn,
    make_ik_solver_fn,
    make_jacobian_fns,
)
from bimanual_manipulation.planning.assignment.learned_filter import (
    HandoverFeasibilityNet,
    HandoverLogger,
    LearnedHandoverFilter,
)
from bimanual_manipulation.planning.assignment.utils import (
    save_assignment_result,
)
from bimanual_manipulation.planning.visualization.trajectory import (
    visualize_trajectory_feasibility,
    visualize_assignment_result,
)
from bimanual_manipulation.transforms import TransformManager
from bimanual_manipulation.transforms.meshcat_tf_visualizer import (
    stream_transforms_to_meshcat,
    MeshcatTFVisualizerConfig,
)
from bimanual_manipulation.utils.env_utils import env_step
import meshcat
from bimanual_manipulation.planning.visualization import (
    HandoverVisualizer,
)
from bimanual_manipulation.planning.ik.manipulability import (
    compute_trajectory_manipulability,
)

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
    parser.add_argument(
        "--beam-width", type=int, default=None,
        help="Beam width for DP solver (limits surviving states per segment). "
             "Recommended 8–12 when using IK to avoid hour-long solves.",
    )
    parser.add_argument(
        "--handover-stride", type=int, default=None,
        help="Only attempt handovers every K segments (default: 1 = every segment).",
    )
    parser.add_argument(
        "--clearance-radius", type=float, default=None,
        help="Minimum distance (metres) from object to environment geoms for a "
             "handover to be attempted.  Disabled when omitted.",
    )
    parser.add_argument(
        "--num-traj0", type=int, default=None,
        help="Number of robot-0 trajectories to evaluate (default: all). "
             "E.g. --num-traj0 10 uses robot0_trajectory_0 … robot0_trajectory_9.",
    )
    parser.add_argument(
        "--num-traj1", type=int, default=None,
        help="Number of robot-1 trajectories to evaluate (default: all).",
    )
    parser.add_argument(
        "--output-assignment", type=str, default=None,
        help="Save assignment_results.npz directly to this file path "
             "(e.g. in the results dir).  When omitted, saves to <data-dir>/assignment_results.npz.",
    )
    parser.add_argument(
        "--output-metrics", type=str, default=None,
        help="Path to write a JSON file with planning metrics (cost, handovers, time, etc.).",
    )
    parser.add_argument(
        "--switch-cost-override", type=float, default=None,
        help="Override the switching cost returned by compute_switching_cost. "
             "Set to 0 for free switches, inf to prohibit switches.",
    )
    parser.add_argument(
        "--disable-feasibility-filter", action="store_true",
        help="Disable the transfer feasibility filter (allow all handovers).",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable cProfile profiling and print top functions at the end.",
    )
    parser.add_argument(
        "--profile-output", type=str, default=None,
        help="Path to save the cProfile .prof binary (for snakeviz). "
             "Only used when --profile is set.",
    )
    parser.add_argument(
        "--visualize-handovers", action="store_true",
        help="Visualize each evaluated handover attempt in Meshcat "
             "(requires --no-env to be off).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--handover-log", type=str, default=None,
        help="Save features and outcomes of every exact handover evaluation "
             "to this NPZ (training data for the learned filter).",
    )
    parser.add_argument(
        "--learned-filter", type=str, default=None,
        help="Path to a trained handover model; candidates predicted infeasible "
             "or not to improve the best transition skip exact evaluation.",
    )
    parser.add_argument(
        "--filter-threshold", type=float, default=0.1,
        help="Minimum predicted P(feasible) for exact handover evaluation.",
    )
    parser.add_argument(
        "--cost-slack", type=float, default=0.0,
        help="Prune candidates with acc_cost + slack * predicted_cost >= best_cost "
             "(0 disables cost-based pruning).",
    )
    parser.add_argument(
        "--filter-order", action="store_true",
        help="Evaluate each state's handover candidates best-predicted-first "
             "(requires --learned-filter).",
    )

    args = parser.parse_args()

    # --- Load experiment config ---
    exp_cfg = load_experiment(args.experiment)
    asgn_cfg = exp_cfg.assignment

    debug = args.debug or exp_cfg.runtime.debug
    setup_logging(logging.DEBUG if debug else logging.INFO)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info("Set random seeds to %d", args.seed)

    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    # Resolve data directory
    data_dir = args.data_dir
    if data_dir is None:
        if exp_cfg.output.output_dir is not None:
            data_dir = exp_cfg.output.output_dir
        else:
            data_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)

    # --- Load trajectories ---
    max_r0 = args.num_traj0 if args.num_traj0 is not None else asgn_cfg.num_trajectories_robot0
    max_r1 = args.num_traj1 if args.num_traj1 is not None else asgn_cfg.num_trajectories_robot1
    logger.info("Loading trajectories from %s ...", data_dir)
    if max_r0 is not None or max_r1 is not None:
        logger.info(
            "  Trajectory count limits: robot0=%s, robot1=%s",
            max_r0 if max_r0 is not None else "all",
            max_r1 if max_r1 is not None else "all",
        )
    trajectories = load_all_trajectories(data_dir, max_robot0=max_r0, max_robot1=max_r1)

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
    if args.disable_feasibility_filter:
        transfer_feasibility = None
    else:
        transfer_feasibility = _parse_transfer_feasibility(asgn_cfg.transfer_feasibility)
    beam_width = args.beam_width if args.beam_width is not None else asgn_cfg.beam_width
    handover_stride = (
        args.handover_stride if args.handover_stride is not None else asgn_cfg.handover_stride
    )
    clearance_radius = (
        args.clearance_radius if args.clearance_radius is not None
        else asgn_cfg.handover_clearance_radius
    )

    logger.info(
        "Assignment parameters: α=%.2f, β=%.2f, ε=%.1e, beam=%s, stride=%d, clearance=%s",
        alpha, beta, epsilon,
        beam_width if beam_width is not None else "unlimited",
        handover_stride,
        f"{clearance_radius:.3f}m" if clearance_radius is not None else "off",
    )

    # --- Build IK / Jacobian / collision closures from the simulation environment ---
    ik_solver_fn = None
    jacobian_fns = None
    collision_check_fn = None
    handover_clearance_fn = None
    handover_callback = None
    tf_stop_event = None

    if not args.no_env:
        logger.info("Creating simulation environment for IK and Jacobian evaluation ...")
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
        collision_check_fn = make_collision_check_fn(
            env, env_lock,
            manipulated_object=exp_cfg.task.manipulated_object,
        )
        logger.info("IK solver, Jacobian functions, and collision checker ready.")

        if clearance_radius is not None:
            handover_clearance_fn = make_handover_clearance_fn(
                env,
                clearance_radius=clearance_radius,
                manipulated_object=exp_cfg.task.manipulated_object,
                env_lock=env_lock,
            )
            logger.info(
                "Handover clearance filter active (radius=%.3fm).",
                clearance_radius,
            )

        if args.visualize_handovers:
            tf_stop_event = threading.Event()
            tf_stream_thread = threading.Thread(
                target=stream_transforms_to_meshcat,
                kwargs=dict(
                    env=env,
                    env_lock=env_lock,
                    tf_manager=tf_manager,
                    config=MeshcatTFVisualizerConfig(clear=True),
                    stop_event=tf_stop_event,
                ),
                daemon=True,
            )
            tf_stream_thread.start()
            logger.info("TF stream thread started.")

            vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
            logger.info("Handover visualizer connected to %s", constants.MESHCAT_ZMQ_URL)
            hv = HandoverVisualizer(
                vis, tf_manager,
                object_name=exp_cfg.task.manipulated_object,
            )
            handover_callback = hv
            logger.info("Handover visualizer enabled.")
    else:
        logger.warning(
            "Environment creation skipped (--no-env).  Handover costs will use "
            "fallback heuristic; manipulability will use simple joint-distance proxy. "
            "Inter-arm collision checks are disabled."
        )

    # --- Apply switch cost override if specified ---
    _restore_fn = None
    if args.switch_cost_override is not None:
        _original_sw = solver_mod.compute_switching_cost
        _override = args.switch_cost_override

        def _patched_sw(*a, **kw):
            cost = _original_sw(*a, **kw)
            if np.isinf(cost):
                return cost
            return _override

        solver_mod.compute_switching_cost = _patched_sw
        _restore_fn = lambda: setattr(solver_mod, "compute_switching_cost", _original_sw)
        logger.info("Switch cost override active: %.2f", _override)

    # --- Learned handover filter / handover logging ---
    learned_filter = None
    if args.learned_filter is not None:
        learned_filter = LearnedHandoverFilter(
            HandoverFeasibilityNet.load(args.learned_filter),
            threshold=args.filter_threshold,
            cost_slack=args.cost_slack,
        )
        logger.info(
            "Learned handover filter active: %s (threshold=%.2f, cost_slack=%.2f, order=%s)",
            args.learned_filter, args.filter_threshold, args.cost_slack, args.filter_order,
        )

    handover_logger = None
    if args.handover_log is not None:
        handover_logger = HandoverLogger(trajectories, chain=handover_callback)
        handover_callback = handover_logger

    # --- Run assignment planner (optionally profiled) ---
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    t0 = time.time()
    try:
        result = compute_optimal_robot_assignment(
            trajectories,
            transfer_feasibility=transfer_feasibility,
            initial_robot=initial_robot,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            ik_solver_fn=ik_solver_fn,
            jacobian_fns=jacobian_fns,
            collision_check_fn=collision_check_fn,
            handover_clearance_fn=handover_clearance_fn,
            handover_callback=handover_callback,
            beam_width=beam_width,
            handover_stride=handover_stride,
            handover_prefilter=learned_filter,
            handover_order=(
                learned_filter.order
                if learned_filter is not None and args.filter_order else None
            ),
        )
    finally:
        if handover_logger is not None:
            os.makedirs(os.path.dirname(args.handover_log) or ".", exist_ok=True)
            handover_logger.save(args.handover_log)
        if learned_filter is not None:
            logger.info("Learned filter stats: %s", learned_filter.stats())

    planning_time = time.time() - t0

    if profiler is not None:
        profiler.disable()

    if _restore_fn is not None:
        _restore_fn()

    # Stop the TF stream background thread if it was started
    if tf_stop_event is not None:
        tf_stop_event.set()

    # --- Print summary ---
    _print_summary(result)

    # --- Save result ---
    if args.output_assignment is not None:
        out_path = args.output_assignment
    else:
        out_path = os.path.join(data_dir, "assignment_results.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_assignment_result(result, out_path)

    # --- Write JSON metrics if requested ---
    if args.output_metrics is not None:
        os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
        metrics = _build_planning_metrics(
            result, exp_cfg, data_dir, planning_time, jacobian_fns,
        )
        if learned_filter is not None:
            metrics.update(learned_filter.stats())
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics written to %s", args.output_metrics)

    # --- Optional assignment bar chart ---
    if not args.no_plot:
        chart_path = os.path.join(os.path.dirname(out_path), "assignment_result.png")
        visualize_assignment_result(result, save_path=chart_path)

    # --- cProfile output ---
    if profiler is not None:
        if args.profile_output:
            os.makedirs(os.path.dirname(args.profile_output) or ".", exist_ok=True)
            profiler.dump_stats(args.profile_output)
            logger.info("Profile saved to %s", args.profile_output)

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        print("\n" + "=" * 70)
        print("  cProfile: Top 40 by CUMULATIVE time")
        print("=" * 70)
        print(stream.getvalue())

        stream2 = io.StringIO()
        stats2 = pstats.Stats(profiler, stream=stream2)
        stats2.strip_dirs()
        stats2.sort_stats("tottime")
        stats2.print_stats(40)
        print("\n" + "=" * 70)
        print("  cProfile: Top 40 by TOTAL (self) time")
        print("=" * 70)
        print(stream2.getvalue())

    logger.info("Done. (%.1fs)", planning_time)


def _build_planning_metrics(result, exp_cfg, data_dir, planning_time, jacobian_fns):
    """Build a flat dict with all PlanningMetrics-compatible fields.

    Computes manipulability along the assigned path using Yoshikawa measure
    when *jacobian_fns* is available, otherwise falls back to the simple
    joint-distance proxy.

    The dict is JSON-serialisable and designed to be read back by
    ``scripts.evaluation.metrics.PlanningMetrics(**d)``.
    """
    data_path = Path(data_dir)
    sequence = result.sequence
    traj_paths = result.trajectory_paths

    # --- Manipulability along the assigned path ---
    manip_values = []
    feasible_count = 0

    for seg_idx in range(len(sequence)):
        tpath = str(traj_paths[seg_idx])
        fpath = data_path / tpath
        if not fpath.exists():
            continue
        traj_data = np.load(fpath)
        joint_configs = traj_data["joint_configs"]
        is_feasible = traj_data["is_feasible"]

        if seg_idx < len(joint_configs) and is_feasible[seg_idx]:
            q = joint_configs[seg_idx].reshape(1, -1)
            robot_id = int(sequence[seg_idx])
            jac_fn = jacobian_fns.get(robot_id) if jacobian_fns else None
            m = compute_trajectory_manipulability(
                q,
                use_jacobian=(jac_fn is not None),
                jacobian_fn=jac_fn,
            )
            manip_values.append(float(m[0]))
            feasible_count += 1

    mean_manip = float(np.mean(manip_values)) if manip_values else 0.0
    min_manip = float(np.min(manip_values)) if manip_values else 0.0

    # --- Per-robot feasibility rates ---
    feas_rates = {}
    for robot_id, prefix in [(0, "robot0"), (1, "robot1")]:
        traj_files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"))
        n_total = len(traj_files)
        n_fully_feasible = sum(
            1 for tf in traj_files if np.all(np.load(tf)["is_feasible"])
        )
        feas_rates[robot_id] = n_fully_feasible / max(n_total, 1)

    _ROBOT_GRIPPER = {0: "finger", 1: "suction"}
    seq_list = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
    hp_list = (result.handover_points.tolist()
               if hasattr(result.handover_points, "tolist")
               else list(result.handover_points))

    grasp_gripper = _ROBOT_GRIPPER.get(int(sequence[0]), "") if len(sequence) > 0 else ""
    handover_dirs = []
    for hp in hp_list:
        if 0 < hp < len(sequence):
            src = _ROBOT_GRIPPER.get(int(sequence[hp - 1]), "?")
            dst = _ROBOT_GRIPPER.get(int(sequence[hp]), "?")
            handover_dirs.append(f"{src}→{dst}")

    return {
        "experiment": exp_cfg.name,
        "config_label": "",          # filled by orchestrator if needed
        "total_cost": float(result.total_cost),
        "n_segments": len(sequence),
        "n_handovers": len(hp_list),
        "handover_points": hp_list,
        "mean_task_cost": float(np.mean(result.task_costs)),
        "total_switching_cost": float(np.sum(result.switching_costs)),
        "mean_manipulability": mean_manip,
        "min_manipulability": min_manip,
        "feasibility_rate_robot0": feas_rates[0],
        "feasibility_rate_robot1": feas_rates[1],
        "assigned_feasibility_rate": feasible_count / max(len(sequence), 1),
        "planning_time_s": planning_time,
        "grasp_gripper": grasp_gripper,
        "handover_directions": handover_dirs,
        "sequence": seq_list,
    }


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
        traj_path = result.trajectory_paths[i]
        tc = result.task_costs[i]
        sc = result.switching_costs[i]
        logger.info(
            "  Seg %3d: %s  traj=%s  task=%.4f  switch=%.4f  total=%.4f",
            i, robot_names[robot_id], traj_path, tc, sc, tc + sc,
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
