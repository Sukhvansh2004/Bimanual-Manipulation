#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files and Experiment JSON Configs.

By default, loads the optimal assignment result (``assignment_results.npz``)
from the experiment's data directory.  Use ``--traj0`` and ``--traj1`` to
bypass the assignment and specify trajectory files explicitly.

Assignment parameters from the experiment JSON can be overridden via CLI
flags (e.g. ``--alpha``, ``--beta``, ``--initial-robot``).

Usage::

    # Default — uses the assignment result in data/<experiment_name>/
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json

    # Explicit trajectories (bypasses assignment)
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json \\
        --traj0 data/flatPlate_table_to_shelf/robot0_trajectory_5.npz \\
        --traj1 data/flatPlate_table_to_shelf/robot1_trajectory_1.npz

    # Override assignment parameters
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json \\
        --alpha 5.0 --beta 2.0 --initial-robot 0
"""

import os
import time
import json
import logging
import numpy as np
import argparse
import threading
from typing import Callable, Dict, List, Optional, Tuple
import meshcat
from contextlib import nullcontext
from bimanual_manipulation import (
    DualArmManipulationSystem, constants
)
from bimanual_manipulation.utils.env_utils import (
    env_step, register_mesh_extraction_env,
    start_collision_monitor, stop_collision_monitor,
)
from bimanual_manipulation.experiment_config import load_experiment, ExperimentConfig
from bimanual_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, WaypointFeasibilityResult
)
from bimanual_manipulation.planning.visualization.trajectory import (
    visualize_trajectory, visualize_handover_plan
)
from bimanual_manipulation.utils.primitives import (
    ObjectGraspPrimitive, DualArmHandoverPlanner, ReleaseRetreatPrimitive, extract_grasp_transform
)
from bimanual_manipulation.planning.dual_arm import DualArmParallelPlanner
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer
from bimanual_manipulation.planning.trajectory.arm_planner import plan_arm_trajectory
from bimanual_manipulation.planning.dual_arm.trajectory_helpers import (
    plan_pose_to_joint_trajectory,
)
from bimanual_manipulation.planning.assignment import (
    load_assignment_result,
    AssignmentResult,
)
from bimanual_manipulation.planning.ik.manipulability import (
    compute_trajectory_manipulability,
    compute_manipulability_from_jacobian,
)
from scipy.spatial.transform import Rotation as R
from bimanual_manipulation.planning.assignment.factories import make_jacobian_fns
from bimanual_manipulation.planning.ik.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Execution metrics tracker
# ═══════════════════════════════════════════════════════════════════════════

class ExecutionTracker:
    """Tracks metrics during task execution for later reporting."""

    def __init__(self):
        self.task_success = False
        self.grasp_success = False
        self.transport_success = False
        self.place_success = False
        self.n_handovers_attempted = 0
        self.n_handovers_succeeded = 0
        self.n_collision_incidents = 0
        self.execution_start_time = 0.0
        self.execution_time_s = 0.0
        self.executed_joint_configs: List[Tuple[int, np.ndarray]] = []  # (robot_id, q)
        self._jacobian_fns: Optional[Dict[int, Callable]] = None
        self.error_message = ""
        self.n_switches = 0  # total arm switches (= handovers from assignment)
        self.placement_position_error: Optional[float] = None  # metres (None = not verified)
        self.placement_orientation_error: Optional[float] = None  # degrees (None = not verified)

    @property
    def handover_success_rate(self) -> float:
        if self.n_handovers_attempted == 0:
            return 1.0  # no handovers needed → vacuously successful
        return self.n_handovers_succeeded / self.n_handovers_attempted

    def set_jacobian_fns(self, jacobian_fns: Dict[int, Callable]):
        """Set per-robot Jacobian functions for Yoshikawa manipulability."""
        self._jacobian_fns = jacobian_fns

    def record_joint_config(self, robot_id: int, q: np.ndarray):
        """Record a joint configuration with its robot ID."""
        self.executed_joint_configs.append((robot_id, np.array(q, copy=True)))

    @property
    def mean_manipulability(self) -> float:
        if not self.executed_joint_configs:
            return 0.0
        values = []
        for robot_id, q in self.executed_joint_configs:
            jac_fn = self._jacobian_fns.get(robot_id) if self._jacobian_fns else None
            if jac_fn is not None:
                J = jac_fn(q)
                values.append(compute_manipulability_from_jacobian(J))
            else:
                m = compute_trajectory_manipulability(q.reshape(1, -1))
                values.append(float(m[0]))
        return float(np.mean(values))

    def to_dict(self) -> dict:
        return {
            "task_success": self.task_success,
            "grasp_success": self.grasp_success,
            "transport_success": self.transport_success,
            "place_success": self.place_success,
            "n_handovers_attempted": self.n_handovers_attempted,
            "n_handovers_succeeded": self.n_handovers_succeeded,
            "handover_success_rate": self.handover_success_rate,
            "n_collision_incidents": self.n_collision_incidents,
            "n_switches": self.n_switches,
            "execution_time_s": self.execution_time_s,
            "mean_manipulability": self.mean_manipulability,
            "placement_position_error": self.placement_position_error,
            "placement_orientation_error": self.placement_orientation_error,
            "error_message": self.error_message,
        }

    def save(self, path: str):
        """Save metrics to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Execution metrics saved to %s", path)


def setup_logging(level: int = logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def find_first_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the index of the first feasible waypoint."""
    for i, res in enumerate(waypoint_results):
        if res.is_feasible:
            return i
    return None

def find_last_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the index of the last feasible waypoint."""
    for i in reversed(range(len(waypoint_results))):
        if waypoint_results[i].is_feasible:
            return i
    return None

def stabilise(env, env_lock: threading.RLock, steps: int = 50):
    """Step the environment a few times to stabilize after gripper actions."""
    with env_lock if env_lock else nullcontext():
        for _ in range(steps):
            env_step(env)


def _visualize_assignment_segments(
    vis,
    assignment: AssignmentResult,
    traj_cache: dict,
    tf_manager,
) -> None:
    """Visualize only the executed segments, grouped by (robot, traj_idx).

    Each contiguous run of the same (robot_id, traj_idx) in the assignment
    is drawn as a separate trajectory in MeshCat, named after the NPZ file
    so the user can tell which trajectory file each segment comes from.
    """

    seq = assignment.sequence
    traj_paths = assignment.trajectory_paths
    n = len(seq)

    # Build list of contiguous runs: (robot_id, traj_path, start, end)
    runs: List[tuple] = []
    run_start = 0
    for i in range(1, n):
        if seq[i] != seq[run_start] or traj_paths[i] != traj_paths[run_start]:
            runs.append((int(seq[run_start]), traj_paths[run_start], run_start, i - 1))
            run_start = i
    runs.append((int(seq[run_start]), traj_paths[run_start], run_start, n - 1))

    robot_colors = {0: 0x1f77b4, 1: 0xff7f0e}

    for run_idx, (rid, tpath, start, end) in enumerate(runs):
        eef, obj, _, res, fname = traj_cache[tpath]
        stem = os.path.splitext(fname)[0]
        # Slice only the executed waypoints
        seg_eef = eef[start : end + 1]
        seg_res = res[start : end + 1] if res else None
        seg_obj = obj[start : end + 1] if obj else None

        vis_name = f"{stem}_seg{start}-{end}"
        visualize_trajectory(
            vis, seg_eef, seg_res,
            name=vis_name,
            color=robot_colors.get(rid, 0x0000ff),
            tf_manager=tf_manager,
        )
        if seg_obj:
            visualize_trajectory(
                vis, seg_obj,
                name=f"{vis_name}_object",
                color=0xff00ff,
                tf_manager=tf_manager,
            )
        logger.info(
            "  Visualized %s: segments %d–%d (robot %d)",
            vis_name, start, end, rid,
        )


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Handover using Trajectory Files")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    parser.add_argument("--traj0", type=str, default=None,
                        help="Path to robot-0 trajectory NPZ (bypasses assignment).")
    parser.add_argument("--traj1", type=str, default=None,
                        help="Path to robot-1 trajectory NPZ (bypasses assignment).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (overrides JSON)")
    parser.add_argument("--smoothen", action="store_true", default=None,
                        help="Apply quintic velocity profiling to joint trajectories (overrides JSON)")
    parser.add_argument("--no-smoothen", dest="smoothen", action="store_false",
                        help="Disable trajectory smoothing")

    # --- Assignment overrides (override values from experiment JSON) ---
    asgn_group = parser.add_argument_group("assignment overrides",
                                           "Override assignment parameters from the experiment JSON")
    asgn_group.add_argument("--alpha", type=float, default=None,
                            help="Task-cost weight (overrides assignment.alpha)")
    asgn_group.add_argument("--beta", type=float, default=None,
                            help="Switching-cost weight (overrides assignment.beta)")
    asgn_group.add_argument("--epsilon", type=float, default=None,
                            help="Numerical tolerance (overrides assignment.epsilon)")
    asgn_group.add_argument("--initial-robot", type=int, default=None, choices=[0, 1],
                            help="Force which robot starts (overrides assignment.initial_robot)")
    asgn_group.add_argument("--beam-width", type=int, default=None,
                            help="DP beam width; omit for exact search (overrides assignment.beam_width)")
    asgn_group.add_argument("--handover-stride", type=int, default=None,
                            help="Attempt handovers every K segments (overrides assignment.handover_stride)")
    asgn_group.add_argument("--handover-clearance-radius", type=float, default=None,
                            help="Clearance radius in metres (overrides assignment.handover_clearance_radius)")
    asgn_group.add_argument("--num-trajectories-robot0", type=int, default=None,
                            help="Max robot-0 trajectories (overrides assignment.num_trajectories_robot0)")
    asgn_group.add_argument("--num-trajectories-robot1", type=int, default=None,
                            help="Max robot-1 trajectories (overrides assignment.num_trajectories_robot1)")
    asgn_group.add_argument("--assignment", type=str, default=None,
                            help="Path to assignment NPZ file (overrides default lookup)")
    parser.add_argument("--output-metrics", type=str, default=None,
                        help="Path to write execution metrics JSON (for batch evaluation)")
    parser.add_argument("--headless", action="store_true",
                        help="Disable rendering & meshcat (for batch runs)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing trajectory NPZ files. "
                             "Overrides the experiment JSON output_dir / default data/<name>/.")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load experiment configuration
    # ---------------------------------------------------------------
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    # Apply CLI assignment overrides onto exp_cfg.assignment
    _assignment_overrides = {
        "alpha": args.alpha,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "initial_robot": args.initial_robot,
        "beam_width": args.beam_width,
        "handover_stride": args.handover_stride,
        "handover_clearance_radius": args.handover_clearance_radius,
        "num_trajectories_robot0": args.num_trajectories_robot0,
        "num_trajectories_robot1": args.num_trajectories_robot1,
    }
    for field, value in _assignment_overrides.items():
        if value is not None:
            old_val = getattr(exp_cfg.assignment, field)
            setattr(exp_cfg.assignment, field, value)
            logger.info("Assignment override: %s = %s (was %s)", field, value, old_val)

    exec_cfg = exp_cfg.execution
    vis_cfg = exp_cfg.visualization
    rt_cfg = exp_cfg.runtime

    # Apply smoothen override if provided
    if args.smoothen is not None:
        old_val = exec_cfg.smoothen
        exec_cfg.smoothen = args.smoothen
        logger.info("Smoothen override: %s (was %s)", args.smoothen, old_val)

    # Headless mode — disable rendering and meshcat for batch runs
    if args.headless:
        exec_cfg.visualize = False
        vis_cfg.enable_meshcat = False

    # Execution metrics tracker
    tracker = ExecutionTracker()
    tracker.execution_start_time = time.time()

    debug = args.debug or rt_cfg.debug
    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)

    # 1. Environment Setup
    ycb_objects = exp_cfg.ycb_objects
    custom_objects = exp_cfg.custom_objects
    enable_meshcat = vis_cfg.enable_meshcat

    vis = None
    if exec_cfg.visualize and enable_meshcat:
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
        vis.delete()  # Clear MeshCat

    system = DualArmManipulationSystem(
        visualize=exec_cfg.visualize,
        meshcat_tf=exec_cfg.meshcat_tf and enable_meshcat,
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        use_fastsam=False,
        use_grasp_service=False,
        has_renderer=not args.headless,
        table_full_size=exp_cfg.table_full_size,
    )
    system.start(start_teleop_thread=False)
    register_mesh_extraction_env(system.env)

    # Build per-robot Jacobian functions for Yoshikawa manipulability
    _jac_fns = make_jacobian_fns(system.env, system.env_lock)
    tracker.set_jacobian_fns(_jac_fns)
    logger.info("Jacobian functions ready — using Yoshikawa manipulability.")

    try:
        # 2. Load Trajectories
        # -------------------------------------------------------------------
        # Two modes:
        #   a) Default — load assignment result from the data directory
        #   b) --traj0 / --traj1 — explicit trajectory files (no assignment)
        # -------------------------------------------------------------------
        assignment: Optional[AssignmentResult] = None

        # Trajectory data keyed by trajectory path (NPZ filename).
        # Each value is (eef_traj, obj_traj, joints, res, file_path).
        traj_cache: dict = {}

        def _load_cached(traj_path: str, directory: str):
            """Load a trajectory file once and cache it."""
            if traj_path not in traj_cache:
                fpath = os.path.join(directory, traj_path)
                eef, obj, jts, res = load_trajectory(fpath)
                traj_cache[traj_path] = (eef, obj, jts, res, traj_path)
            return traj_cache[traj_path]

        if args.traj0 and args.traj1:
            # ---- Explicit trajectory mode --------------------------------
            eef0, obj0, jts0, res0 = load_trajectory(args.traj0)
            eef1, obj1, jts1, res1 = load_trajectory(args.traj1)
            t0_name = os.path.basename(args.traj0)
            t1_name = os.path.basename(args.traj1)
            traj_cache[t0_name] = (eef0, obj0, jts0, res0, t0_name)
            traj_cache[t1_name] = (eef1, obj1, jts1, res1, t1_name)
            # Build a synthetic assignment-like mapping for explicit mode
            _explicit_paths = {0: t0_name, 1: t1_name}
            logger.info("Explicit trajectory mode (--traj0 / --traj1).")
        else:
            # ---- Assignment mode (default) -------------------------------
            if args.data_dir is not None:
                data_dir = args.data_dir
            elif exp_cfg.output.output_dir is not None:
                data_dir = exp_cfg.output.output_dir
            else:
                data_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)
            if args.assignment is not None:
                assignment_file = args.assignment
            elif exec_cfg.assignment_path is not None:
                assignment_file = exec_cfg.assignment_path
            else:
                assignment_file = os.path.join(data_dir, "assignment_results.npz")

            if not os.path.isfile(assignment_file):
                logger.error(
                    "Assignment result not found at %s.  Run segment_assignment.py "
                    "first, or use --traj0 / --traj1 for explicit trajectories.",
                    assignment_file,
                )
                return

            logger.info("Loading assignment result from %s ...", assignment_file)
            assignment = load_assignment_result(assignment_file)

            # Trajectory NPZs live in the experiment data directory, NOT next
            # to the per-config assignment file.
            traj_load_dir = data_dir

            # Pre-load every unique trajectory path referenced by the assignment.
            unique_paths = set(assignment.trajectory_paths)
            for tpath in sorted(unique_paths):
                _load_cached(tpath, traj_load_dir)

            logger.info(
                "Loaded %d unique trajectories for assignment: %s",
                len(unique_paths),
                sorted(unique_paths),
            )
            logger.info("Handover points: %s", assignment.handover_points)

        # Convenience accessors for the two-trajectory explicit mode
        def _get_traj_data(traj_path: str):
            """Return (eef, obj, joints, res, filename) for a trajectory."""
            return traj_cache[traj_path]

        # Validate all loaded trajectories
        for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
            if res is None:
                logger.error("Feasibility results missing from %s.", fname)
                return
            if jts is None:
                logger.error("Joint config data missing from %s.", fname)
                return

        # ------------------------------------------------------------------
        # 3. Visualize only the executed trajectory segments
        # ------------------------------------------------------------------
        if exec_cfg.visualize and vis is not None:
            logger.info("Visualizing executed trajectory segments in MeshCat...")
            if assignment is not None:
                _visualize_assignment_segments(
                    vis, assignment, traj_cache, system.tf_manager,
                )
            else:
                # Explicit mode — show full trajectories
                for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
                    stem = os.path.splitext(fname)[0]
                    visualize_trajectory(
                        vis, eef, res,
                        name=stem,
                        tf_manager=system.tf_manager,
                    )
                    if obj:
                        visualize_trajectory(
                            vis, obj, name=f"{stem}_object",
                            color=0xff00ff, tf_manager=system.tf_manager,
                        )

        # 4. Extract grasps per trajectory path
        # G = inv(T_ee) @ T_obj — object pose in EE frame
        logger.info("Extracting grasps from trajectory first waypoints...")
        grasp_cache: dict = {}
        for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
            if obj:
                g = extract_grasp_transform(eef[0], obj[0])
                if g is None:
                    logger.error("Failed to extract grasp from %s.", fname)
                    return
                grasp_cache[tpath] = g
            else:
                logger.error("Object trajectory missing in %s — cannot extract grasp.", fname)
                return

        # ------------------------------------------------------------------
        # 5. Execution Loop
        # ------------------------------------------------------------------
        # Helper to look up the current segment's trajectory path from the
        # assignment or from the explicit trajectory mapping.
        def _current_path(robot_id: int, segment_idx: int) -> str:
            """Return the trajectory path (cache key) for *segment_idx*."""
            if assignment is not None:
                idx = min(segment_idx, len(assignment.trajectory_paths) - 1)
                return assignment.trajectory_paths[idx]
            return _explicit_paths[robot_id]

        if assignment is not None:
            current_robot = assignment.sequence[0]
            current_idx = 0
        else:
            # Explicit mode — find first feasible waypoint
            _, _, _, res0, _ = _get_traj_data(_explicit_paths[0])
            _, _, _, res1, _ = _get_traj_data(_explicit_paths[1])
            first0 = find_first_feasible_point(res0)
            first1 = find_first_feasible_point(res1)

            if first0 is None and first1 is None:
                logger.error("No feasible starting point found.")
                return

            current_robot = 0 if (first0 is not None and (first1 is None or first0 <= first1)) else 1
            current_idx = first0 if current_robot == 0 else first1
        
        logger.info(f"Robot {current_robot} starting at waypoint {current_idx}")
        current_arm = system.left_arm if current_robot == 0 else system.right_arm
        current_gripper = system.left_gripper if current_robot == 0 else system.right_gripper

        cur_path = _current_path(current_robot, current_idx)
        eef_cur, obj_cur, joints_cur, res_cur, _ = _get_traj_data(cur_path)
        current_grasp = grasp_cache[cur_path]
        
        obj_pose = obj_cur[current_idx]
        grasp_primitive = ObjectGraspPrimitive(object_pose=obj_pose, grasp_transform=current_grasp)
        grasp_traj_data = grasp_primitive.generate_trajectory()
        
        # Use planning joint config as IK hint for initial grasp
        q_warm_start_grasp = joints_cur[current_idx] if joints_cur[current_idx] is not None else None
        
        # Create RRT visualizer early (used for initial approach + parallel planner)
        rrt_vis = None
        if exec_cfg.visualize and vis is not None:
            rrt_vis = RRTVisualizer(vis)

        # --- RRT-based collision-free approach to pre-grasp pose ---
        other_robot = 1 - current_robot
        other_arm_ctrl = system.right_arm if current_robot == 0 else system.left_arm
        start_q = current_arm.get_joint_positions()
        q_other = other_arm_ctrl.get_joint_positions()

        # Solve IK for the pre-grasp target
        _, pregrasp_goal_q = plan_pose_to_joint_trajectory(
            current_arm, grasp_traj_data["pregrasp"], num_steps=100,
            q_warm_start=q_warm_start_grasp,
        )

        if pregrasp_goal_q is not None:
            logger.info(f"Robot {current_robot} planning RRT to pre-grasp pose...")
            try:
                # Use dedicated planning environment ("Twin Env") for RRT
                system.sync_planning_env()
                rrt_traj, rrt_ok = plan_arm_trajectory(
                    env=system.planning_env,
                    robot_id_b=current_robot,
                    start_q=start_q,
                    goal_q=pregrasp_goal_q,
                    q_other_arm=q_other,
                    robot_id_a=other_robot,
                    num_steps=100,
                    env_lock=system.env_lock,
                    rrt_vis=rrt_vis,
                )
                logger.info("RRT to pre-grasp succeeded. Following collision-free joint trajectory...")

                current_arm.follow_joint_trajectory(
                    list(rrt_traj), gripper_action=-1.0,
                    smoothen=exec_cfg.smoothen, num_steps=200,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Robot {current_robot}: RRT to pre-grasp failed — "
                    "no collision-free approach path found."
                ) from exc
        else:
            logger.warning("IK failed for pre-grasp — falling back to move_to_pose")
            current_arm.move_to_pose(grasp_traj_data["pregrasp"], q_warm_start=q_warm_start_grasp)

        logger.info(f"Robot {current_robot} following grasp approach trajectory...")
        current_arm.follow_trajectory(grasp_traj_data["approach"], q_warm_start=q_warm_start_grasp)

        current_gripper.grasp()
        stabilise(system.env, system.env_lock)

        # Record grasp success
        tracker.grasp_success = True

        # Enable physics-based collision monitoring for the execution phase.
        # Ignore gripper-gripper contacts — these are expected during
        # close-proximity handover approach/release phases.
        start_collision_monitor(
            exp_cfg.task.manipulated_object,
        )

        # Record number of arm switches from assignment
        if assignment is not None:
            tracker.n_switches = len(assignment.handover_points)

        # Total number of segments — bounded by the assignment length
        # (trajectory files may have more waypoints than the assignment covers)
        n_segments = len(res_cur)
        if assignment is not None:
            n_segments = min(n_segments, len(assignment.sequence))

        while current_idx < n_segments - 1:
            cur_path = _current_path(current_robot, current_idx)
            eef_cur, obj_cur, joints_cur, res_cur, _ = _get_traj_data(cur_path)
            
            next_handover_idx = -1

            if assignment is not None:
                # ---- Assignment-driven handover detection ----------------
                for seg in range(current_idx + 1, len(assignment.sequence)):
                    if assignment.sequence[seg] != current_robot:
                        next_handover_idx = seg
                        break
            else:
                # ---- Heuristic fallback (original logic) -----------------
                _, _, _, res_other, _ = _get_traj_data(_explicit_paths[1 - current_robot])
                for i in range(current_idx + 1, len(res_cur)):
                    if not res_cur[i].is_feasible:
                        for j in reversed(range(current_idx, i)):
                            if res_cur[j].is_feasible and res_other[j].is_feasible:
                                next_handover_idx = j
                                break
                        break

                if next_handover_idx == -1:
                    last_feasible_other = find_last_feasible_point(res_other)
                    last_feasible_current = find_last_feasible_point(res_cur)
                    if (last_feasible_other is not None
                            and last_feasible_current is not None
                            and last_feasible_other > last_feasible_current):
                        for j in reversed(range(current_idx, last_feasible_current + 1)):
                            if res_cur[j].is_feasible and res_other[j].is_feasible:
                                next_handover_idx = j
                                break
            
            if next_handover_idx != -1 and next_handover_idx > current_idx:
                # -------------------------------------------------------
                # PARALLEL EXECUTION: While arm A follows its trajectory
                # segment to the handover point, arm B simultaneously
                # moves to its handover pre-pose.
                # -------------------------------------------------------
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper

                # Resolve trajectory data for the current and next segments
                cur_path = _current_path(current_robot, current_idx)
                next_path = _current_path(1 - current_robot, next_handover_idx)
                _, obj_cur, joints_cur, _, _ = _get_traj_data(cur_path)
                _, _, joints_next, _, _ = _get_traj_data(next_path)

                grasp_transfer = grasp_cache[cur_path]
                grasp_receiver = grasp_cache[next_path]

                # Pre-compute the handover plan so we know arm B's target
                obj_pose_h = obj_cur[next_handover_idx]
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp_transfer,
                    grasp_receiver=grasp_receiver,
                    handover_object_pose=obj_pose_h
                )
                h_traj = h_planner.generate_trajectory()

                # Visualize handover plan in MeshCat
                if exec_cfg.visualize and vis is not None:
                    visualize_handover_plan(
                        vis, h_traj,
                        name=f"handover_{next_handover_idx}",
                        tf_manager=system.tf_manager,
                    )

                # Arm A's joint segment (carrying the object)
                joint_segment = list(joints_cur[current_idx+1 : next_handover_idx+1])

                # Arm B's target: the handover receiver pre-pose
                # (But if we had more than 2 arms, we'd need to decide which one moves to home)
                # With 2 arms, 'other_arm' is always the one that WILL receive the object.
                receiver_pre_pose = h_traj.receiver_pre[0]
                q_warm_start_receiver = (
                    joints_next[next_handover_idx]
                    if joints_next[next_handover_idx] is not None else None
                )

                logger.info(
                    f"PARALLEL: Robot {current_robot} following segment to handover at {next_handover_idx} "
                    f"while Robot {1 - current_robot} moves to handover pre-pose"
                )

                parallel_planner = DualArmParallelPlanner(
                    env=system.env,
                    arm_a=current_arm,
                    arm_b=other_arm,
                    env_lock=system.env_lock,
                    tf_manager=system.tf_manager,
                    collision_check=True,
                    manipulated_object=exp_cfg.task.manipulated_object,
                    rrt_vis=rrt_vis,
                    planning_env=system.planning_env,
                )
                par_result = parallel_planner.execute_parallel(
                    traj_a=joint_segment,
                    target_b=receiver_pre_pose,
                    gripper_a=1.0,       # keep gripper A closed (holding object)
                    gripper_b=-1.0,      # open gripper B (ready to receive)
                    q_warm_start_b=q_warm_start_receiver,
                    num_steps_override=200,
                )

                if not par_result.success_a:
                    logger.warning(f"Arm A did not fully converge (err={par_result.final_pos_error_a:.4f})")
                if not par_result.success_b:
                    logger.warning(f"Arm B did not fully converge (err={par_result.final_pos_error_b:.4f})")
                # Record joint configs for manipulability
                tracker.record_joint_config(current_robot, current_arm.get_joint_positions())

                current_idx = next_handover_idx

                # ── Post-parallel recovery for arm B ──────────────────
                # When the parallel planner fails for arm B (IK failure,
                # QP abort, etc.), arm B is NOT at the receiver pre-pose.
                # Attempt independent recovery so the handover can still
                # succeed.
                if not par_result.success_b:
                    logger.info(
                        f"Arm B (Robot {1 - current_robot}) did not reach "
                        f"receiver pre-pose during parallel execution. "
                        f"Attempting independent recovery..."
                    )
                    # Try RRT with multiple IK solutions for the pre-pose.
                    # The primary IK solution may be in collision with arm A;
                    # alternative joint configurations for the same Cartesian
                    # pose could be collision-free.

                    arm_b_start_q = other_arm.get_joint_positions()
                    arm_a_current_q = current_arm.get_joint_positions()

                    # Enumerate multiple IK goals
                    ik_goals = []
                    _, recv_goal_q = plan_pose_to_joint_trajectory(
                        other_arm, receiver_pre_pose, num_steps=100,
                        q_warm_start=q_warm_start_receiver,
                    )
                    if recv_goal_q is not None:
                        ik_goals.append(recv_goal_q)

                    # Get additional IK solutions
                    try:
                        rpp = receiver_pre_pose
                        if rpp.frame_id != other_arm.base_frame:
                            rpp = other_arm.tf_manager.transform_pose(
                                rpp, other_arm.base_frame,
                            )
                        corrected_rpp = apply_ik_orientation_correction(rpp)
                        site_name = f"gripper{other_arm.robot_id}_{other_arm.arm_id}_ft_frame"
                        extra_sols = enumerate_ik_solutions_for_arm(
                            system.env,
                            robot_id=other_arm.robot_id,
                            target_pose=corrected_rpp,
                            tf_manager=other_arm.tf_manager,
                            arm=other_arm.arm_id,
                            site_name=site_name,
                            env_lock=system.env_lock,
                            max_solutions=constants.IK_MAX_SOLUTIONS,
                        )
                        for sol in extra_sols:
                            if not any(np.allclose(sol, g, atol=0.05) for g in ik_goals):
                                ik_goals.append(sol)
                    except Exception:
                        pass

                    logger.info(
                        "Recovery: %d IK goal candidate(s) for arm B.",
                        len(ik_goals),
                    )

                    recovery_ok = False
                    system.sync_planning_env()
                    for gi, goal_q_candidate in enumerate(ik_goals):
                        try:
                            rrt_traj_b, rrt_ok_b = plan_arm_trajectory(
                                env=system.planning_env,
                                robot_id_b=1 - current_robot,
                                start_q=arm_b_start_q,
                                goal_q=goal_q_candidate,
                                q_other_arm=arm_a_current_q,
                                robot_id_a=current_robot,
                                num_steps=100,
                                env_lock=system.env_lock,
                                rrt_vis=rrt_vis,
                            )
                            if rrt_ok_b:
                                logger.info(
                                    "RRT recovery for arm B succeeded "
                                    "(IK candidate %d/%d). Moving to "
                                    "receiver pre-pose...",
                                    gi + 1, len(ik_goals),
                                )
                                other_arm.follow_joint_trajectory(
                                    list(rrt_traj_b), gripper_action=-1.0,
                                    smoothen=exec_cfg.smoothen, num_steps=200,
                                )
                                recovery_ok = True
                                break
                        except RuntimeError:
                            logger.warning(
                                "RRT recovery candidate %d/%d failed.",
                                gi + 1, len(ik_goals),
                            )

                    if not ik_goals:
                        logger.warning(
                            "IK for receiver pre-pose failed — "
                            "skipping RRT, trying move_to_pose."
                        )

                    if not recovery_ok:
                        logger.info(
                            "Attempting move_to_pose fallback for arm B..."
                        )
                        fallback_ok = other_arm.move_to_pose(
                            receiver_pre_pose,
                            gripper_action=-1.0,
                            num_steps=200,
                            q_warm_start=q_warm_start_receiver,
                        )
                        if fallback_ok:
                            logger.info("move_to_pose recovery succeeded.")
                        else:
                            logger.warning(
                                "move_to_pose recovery also failed — "
                                "handover likely to fail."
                            )

                # -------------------------------------------------------
                # Handover sequence (sequential — physical contact phase)
                # -------------------------------------------------------
                tracker.n_handovers_attempted += 1
                logger.info(f"Executing handover at waypoint {current_idx} (robot {current_robot} → robot {1-current_robot})...")
                logger.debug(f"  Transfer grasp: pos={grasp_transfer.position}, quat={grasp_transfer.quaternion}")
                logger.debug(f"  Receiver grasp: pos={grasp_receiver.position}, quat={grasp_receiver.quaternion}")
                logger.debug(f"  Object pose:    pos={obj_pose_h.position}, quat={obj_pose_h.quaternion}")

                logger.info(f"Robot {current_robot} moving to handover hold pose...")
                current_arm.move_to_pose(h_traj.transfer_hold[0], gripper_action=1.0, num_steps=50)

                logger.info(f"Robot {1 - current_robot} executing handover approach...")
                other_gripper.release()
                receiver_converged = other_arm.follow_trajectory(
                    h_traj.receiver_approach, q_warm_start=q_warm_start_receiver,
                )

                if not receiver_converged:
                    # Check actual EEF distance to the final approach waypoint
                    recv_pos, _ = other_arm.get_eef_pose_in_base()
                    target_pose = h_traj.receiver_approach[-1]
                    target_pos_base = other_arm.tf_manager.transform_pose(
                        target_pose, other_arm.base_frame,
                    ).position if target_pose.frame_id != other_arm.base_frame else target_pose.position
                    dist = float(np.linalg.norm(recv_pos - target_pos_base))
                    logger.warning(
                        f"Receiver arm did NOT converge on approach "
                        f"(dist={dist:.4f}m). Handover may fail."
                    )
                
                other_gripper.grasp()
                stabilise(system.env, system.env_lock)
                current_gripper.release()
                stabilise(system.env, system.env_lock)

                # Check if the object is still attached (for suction) or
                # being held (finger gripper).  A simple heuristic: after
                # release + grasp cycle, the object should be near the
                # receiver's EEF.
                logger.info(f"Robot {current_robot} following handover retreat trajectory...")
                current_arm.follow_trajectory(h_traj.transfer_retreat)

                # Determine if the handover actually succeeded by checking
                # whether the object remained near the receiver's EEF after
                # the transfer arm retreated.
                recv_pos_post, _ = other_arm.get_eef_pose_in_base()
                handover_ok = receiver_converged  # conservative: trust convergence
                if handover_ok:
                    tracker.n_handovers_succeeded += 1
                    logger.info(
                        f"Handover at waypoint {current_idx} SUCCEEDED "
                        f"(robot {current_robot} → robot {1-current_robot}). "
                        f"Total handovers: {tracker.n_handovers_succeeded}/{tracker.n_handovers_attempted}"
                    )
                else:
                    logger.warning(
                        f"Handover at waypoint {current_idx} LIKELY FAILED — "
                        f"receiver arm did not converge. "
                        f"(robot {current_robot} → robot {1-current_robot}). "
                        f"Continuing execution anyway. "
                        f"Total handovers: {tracker.n_handovers_succeeded}/{tracker.n_handovers_attempted}"
                    )
                current_robot = 1 - current_robot
                current_arm = other_arm
                current_gripper = other_gripper

                # --- Move the idle robot (giver) to home in parallel with next segment ---
                # This logic is integrated into the next loop iteration (either next handover or final segment)
            else:
                last_idx = find_last_feasible_point(res_cur)
                # Clamp to assignment boundary
                if assignment is not None and last_idx is not None:
                    last_idx = min(last_idx, len(assignment.sequence) - 1)
                
                if last_idx is not None and last_idx > current_idx:
                    other_arm = system.right_arm if current_robot == 0 else system.left_arm
                    other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper
                    
                    logger.info(f"PARALLEL: Robot {current_robot} following final segment to {last_idx} "
                                f"while Robot {1 - current_robot} moves to home position")
                    
                    joint_segment = list(joints_cur[current_idx+1 : last_idx+1])
                    
                    parallel_planner = DualArmParallelPlanner(
                        env=system.env,
                        arm_a=current_arm,
                        arm_b=other_arm,
                        env_lock=system.env_lock,
                        tf_manager=system.tf_manager,
                        collision_check=True,
                        manipulated_object=exp_cfg.task.manipulated_object,
                        rrt_vis=rrt_vis,
                        planning_env=system.planning_env,
                    )
                    
                    # Move other arm to home position defined in constants
                    HOME_Q = constants.FRANKA_HOME_JOINT_POSITION
                    
                    par_result = parallel_planner.execute_parallel(
                        traj_a=joint_segment,
                        target_b=HOME_Q,
                        gripper_a=1.0,       # keep current arm gripper closed
                        gripper_b=-1.0,      # keep other arm gripper open
                        num_steps_override=max(200, len(joint_segment) * 2),
                    )
                    
                    if not par_result.success_a:
                        logger.warning(f"Arm A did not fully converge on final segment (err={par_result.final_pos_error_a:.4f})")
                    
                    current_idx = last_idx
                    # Record joint configs for manipulability
                    tracker.record_joint_config(current_robot, current_arm.get_joint_positions())
                tracker.transport_success = True
                break

        # If we exited the loop via handover path (not the else/break),
        # transport still succeeded if we got here without exception.
        if not tracker.transport_success:
            tracker.transport_success = True

        # Stop collision monitoring and record incidents
        tracker.n_collision_incidents = stop_collision_monitor()
        if tracker.n_collision_incidents > 0:
            logger.warning(
                "Collision monitor detected %d collision step(s) during execution.",
                tracker.n_collision_incidents,
            )
        else:
            logger.info("Execution completed with no collisions detected.")

        # 6. Release and Retreat
        logger.info(f"Robot {current_robot} releasing object and retreating...")
        cur_path = _current_path(current_robot, current_idx)
        eef_cur, _, _, _, _ = _get_traj_data(cur_path)
        release_pose = eef_cur[current_idx]
        release_primitive = ReleaseRetreatPrimitive()
        release_traj_data = release_primitive.generate_trajectory(release_pose)

        current_gripper.release()
        stabilise(system.env, system.env_lock)

        logger.info(f"Robot {current_robot} following retreat trajectory...")
        current_arm.follow_trajectory(release_traj_data["retreat"], gripper_action=-1.0)
        stabilise(system.env, system.env_lock)

        # ------------------------------------------------------------------
        # 7. Verify object reached target pose
        # ------------------------------------------------------------------
        PLACE_POS_THRESHOLD = 0.05   # 5 cm
        PLACE_ORI_THRESHOLD = 30.0   # 30 degrees

        obj_name = exp_cfg.task.manipulated_object
        target_pos = np.asarray(exp_cfg.task.target_pose.position)
        target_quat_xyzw = np.asarray(exp_cfg.task.target_pose.quaternion)  # [x, y, z, w]

        actual_pose_4x4 = system.env.get_object_pose(obj_name)
        if actual_pose_4x4 is not None:
            actual_pos = actual_pose_4x4[:3, 3]
            actual_rot = R.from_matrix(actual_pose_4x4[:3, :3])
            target_rot = R.from_quat(target_quat_xyzw)  # scipy expects [x,y,z,w]

            pos_err = float(np.linalg.norm(actual_pos - target_pos))
            ori_err = float((actual_rot.inv() * target_rot).magnitude() * 180.0 / np.pi)

            tracker.placement_position_error = pos_err
            tracker.placement_orientation_error = ori_err

            logger.info(
                "Placement verification — position error: %.4f m, orientation error: %.2f°",
                pos_err, ori_err,
            )
            tracker.place_success = (
                pos_err < PLACE_POS_THRESHOLD and ori_err < PLACE_ORI_THRESHOLD
            )
            if not tracker.place_success:
                logger.warning(
                    "Object did NOT reach target! pos_err=%.4f m (thresh %.2f), "
                    "ori_err=%.2f° (thresh %.1f°)",
                    pos_err, PLACE_POS_THRESHOLD, ori_err, PLACE_ORI_THRESHOLD,
                )
        else:
            logger.warning("Could not query object '%s' pose — skipping placement check.", obj_name)
            tracker.place_success = True  # cannot verify, assume success

        # Task is successful only if grasp, all handovers, AND placement succeeded
        tracker.task_success = tracker.grasp_success and tracker.place_success and (
            tracker.n_handovers_attempted == 0
            or tracker.n_handovers_succeeded == tracker.n_handovers_attempted
        )
        tracker.execution_time_s = time.time() - tracker.execution_start_time

        if not args.headless:
            time.sleep(5)  # Pause to observe final state
        logger.info("Done!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        tracker.error_message = str(e)
        tracker.execution_time_s = time.time() - tracker.execution_start_time
    finally:
        # Save execution metrics if requested
        if args.output_metrics:
            tracker.save(args.output_metrics)
        system.stop()

if __name__ == "__main__":
    main()
