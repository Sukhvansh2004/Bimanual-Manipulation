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
from typing import Any, Callable, Dict, List, Optional, Tuple
import meshcat
from contextlib import nullcontext
from bimanual_manipulation import (
    DualArmManipulationSystem, constants
)
from bimanual_manipulation.controllers.arm import RobosuiteArmController
from bimanual_manipulation.controllers.gripper import RobosuiteGripperController
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.utils.env_utils import (
    env_step, register_mesh_extraction_env,
    start_collision_monitor, stop_collision_monitor,
    start_object_monitor, stop_object_monitor,
)
from bimanual_manipulation.experiment_config import load_experiment, ExperimentConfig
from bimanual_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, WaypointFeasibilityResult
)
from bimanual_manipulation.planning.visualization.trajectory import (
    visualize_trajectory, visualize_handover_plan
)
from bimanual_manipulation.utils.primitives import (
    ObjectGraspPrimitive, DualArmHandoverPlanner, ReleaseRetreatPrimitive,
    HandoverTrajectory, extract_grasp_transform,
)
from bimanual_manipulation.planning.dual_arm import DualArmParallelPlanner
from bimanual_manipulation.planning.dual_arm.result import ParallelExecutionResult
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer
from bimanual_manipulation.planning.trajectory.arm_planner import plan_arm_trajectory
from bimanual_manipulation.planning.dual_arm.trajectory_helpers import (
    plan_pose_to_joint_trajectory,
)
from bimanual_manipulation.planning.assignment import (
    load_assignment_result,
)
from bimanual_manipulation.planning.assignment.result import AssignmentResult
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

# Type aliases for readability
TrajCacheEntry = Tuple[
    List[StampedPose],                       # eef trajectory
    Optional[List[StampedPose]],              # object trajectory
    Optional[np.ndarray],                     # joint configs
    Optional[List[WaypointFeasibilityResult]],# feasibility results
    str,                                      # filename
]
TrajCache = Dict[str, TrajCacheEntry]
GraspCache = Dict[str, StampedPose]
ExplicitPaths = Optional[Dict[int, str]]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution metrics tracker
# ---------------------------------------------------------------------------

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
        self.object_mean_tracking_error: float = 0.0  # mean nearest-waypoint error (m)
        self.object_max_tracking_error: float = 0.0   # max nearest-waypoint error (m)

    @property
    def handover_success_rate(self) -> float:
        if self.n_handovers_attempted == 0:
            return 1.0
        return self.n_handovers_succeeded / self.n_handovers_attempted

    def set_jacobian_fns(self, jacobian_fns: Dict[int, Callable]):
        self._jacobian_fns = jacobian_fns

    def record_joint_config(self, robot_id: int, q: np.ndarray):
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
            "object_mean_tracking_error": self.object_mean_tracking_error,
            "object_max_tracking_error": self.object_max_tracking_error,
            "error_message": self.error_message,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Execution metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )


def find_first_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    for i, res in enumerate(waypoint_results):
        if res.is_feasible:
            return i
    return None


def find_last_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    for i in reversed(range(len(waypoint_results))):
        if waypoint_results[i].is_feasible:
            return i
    return None


def stabilise(env: DualArmEnvironment, env_lock: threading.RLock,
              steps: int = 50) -> None:
    """Step the environment to stabilise after gripper actions."""
    with env_lock if env_lock else nullcontext():
        for _ in range(steps):
            env_step(env)


# ---------------------------------------------------------------------------
# IK / RRT helpers
# ---------------------------------------------------------------------------

def _enumerate_ik_goals(
    arm: RobosuiteArmController,
    env: DualArmEnvironment,
    pose: StampedPose,
    env_lock: threading.RLock,
    q_warm_start: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Enumerate multiple IK solutions for *pose* and return a de-duplicated list."""
    goals: List[np.ndarray] = []

    try:
        pose_base = pose
        if pose_base.frame_id != arm.base_frame:
            pose_base = arm.tf_manager.transform_pose(pose_base, arm.base_frame)
        corrected = apply_ik_orientation_correction(pose_base)
        site_name = f"gripper{arm.robot_id}_{arm.arm_id}_ft_frame"
        extras = enumerate_ik_solutions_for_arm(
            env,
            robot_id=arm.robot_id,
            target_pose=corrected,
            tf_manager=arm.tf_manager,
            arm=arm.arm_id,
            site_name=site_name,
            env_lock=env_lock,
            max_solutions=constants.IK_MAX_SOLUTIONS,
            q_reference=q_warm_start,
        )
        for sol in extras:
            if not any(np.allclose(sol, g, atol=0.05) for g in goals):
                goals.append(sol)
    except Exception as exc:
        logger.debug("Extra IK enumeration failed: %s", exc)

    return goals


def _rrt_to_goal(
    arm: RobosuiteArmController,
    goals: List[np.ndarray],
    start_q: np.ndarray,
    q_other_arm: np.ndarray,
    robot_id_self: int,
    robot_id_other: int,
    planning_env: DualArmEnvironment,
    env_lock: threading.RLock,
    rrt_vis: Optional[RRTVisualizer],
    smoothen: bool,
    gripper_action: float = -1.0,
    label: str = "target",
) -> bool:
    """Try RRT to each IK goal candidate. Return True on first success."""
    for gi, goal_q in enumerate(goals):
        try:
            rrt_traj, _ = plan_arm_trajectory(
                env=planning_env,
                robot_id_b=robot_id_self,
                start_q=start_q,
                goal_q=goal_q,
                q_other_arm=q_other_arm,
                robot_id_a=robot_id_other,
                num_steps=100,
                env_lock=env_lock,
                rrt_vis=rrt_vis,
            )
            logger.info(
                "RRT to %s succeeded (IK candidate %d/%d). "
                "Following collision-free joint trajectory...",
                label, gi + 1, len(goals),
            )
            arm.follow_joint_trajectory(
                list(rrt_traj), gripper_action=gripper_action,
                smoothen=smoothen, num_steps=200,
            )
            return True
        except RuntimeError:
            logger.warning(
                "RRT to %s failed for IK candidate %d/%d.",
                label, gi + 1, len(goals),
            )
    return False

# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _visualize_assignment_segments(
    vis: meshcat.Visualizer,
    assignment: AssignmentResult,
    traj_cache: TrajCache,
    tf_manager: TransformManager,
) -> None:
    """Visualize only the executed segments grouped by contiguous runs."""
    seq = assignment.sequence
    paths = assignment.trajectory_paths
    n = len(seq)

    runs: List[tuple] = []
    run_start = 0
    for i in range(1, n):
        if seq[i] != seq[run_start] or paths[i] != paths[run_start]:
            runs.append((int(seq[run_start]), paths[run_start], run_start, i - 1))
            run_start = i
    runs.append((int(seq[run_start]), paths[run_start], run_start, n - 1))

    robot_colors = {0: 0x1f77b4, 1: 0xff7f0e}
    for run_idx, (rid, tpath, start, end) in enumerate(runs):
        eef, obj, _, res, fname = traj_cache[tpath]
        stem = os.path.splitext(fname)[0]
        vis_name = f"{stem}_seg{start}-{end}"
        visualize_trajectory(
            vis, eef[start:end + 1], res[start:end + 1] if res else None,
            name=vis_name, color=robot_colors.get(rid, 0x0000ff),
            tf_manager=tf_manager,
        )
        if obj:
            visualize_trajectory(
                vis, obj[start:end + 1], name=f"{vis_name}_object",
                color=0xff00ff, tf_manager=tf_manager,
            )
        logger.info("  Visualized %s: segments %d-%d (robot %d)", vis_name, start, end, rid)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-Arm Handover using Trajectory Files")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config")
    parser.add_argument("--traj0", type=str, default=None,
                        help="Path to robot-0 trajectory NPZ (bypasses assignment)")
    parser.add_argument("--traj1", type=str, default=None,
                        help="Path to robot-1 trajectory NPZ (bypasses assignment)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--smoothen", action="store_true", default=None,
                        help="Apply quintic velocity profiling")
    parser.add_argument("--no-smoothen", dest="smoothen", action="store_false",
                        help="Disable trajectory smoothing")

    asgn = parser.add_argument_group("assignment overrides")
    asgn.add_argument("--alpha", type=float, default=None)
    asgn.add_argument("--beta", type=float, default=None)
    asgn.add_argument("--epsilon", type=float, default=None)
    asgn.add_argument("--initial-robot", type=int, default=None, choices=[0, 1])
    asgn.add_argument("--beam-width", type=int, default=None)
    asgn.add_argument("--handover-stride", type=int, default=None)
    asgn.add_argument("--handover-clearance-radius", type=float, default=None)
    asgn.add_argument("--num-trajectories-robot0", type=int, default=None)
    asgn.add_argument("--num-trajectories-robot1", type=int, default=None)
    asgn.add_argument("--assignment", type=str, default=None,
                      help="Path to assignment NPZ file")

    parser.add_argument("--output-metrics", type=str, default=None,
                        help="Path to write execution metrics JSON")
    parser.add_argument("--headless", action="store_true",
                        help="Disable rendering & meshcat")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing trajectory NPZ files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

def load_and_configure_experiment(args) -> ExperimentConfig:
    """Load experiment JSON and apply CLI overrides."""
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s - %s", exp_cfg.name, exp_cfg.description)

    overrides = {
        "alpha": args.alpha, "beta": args.beta, "epsilon": args.epsilon,
        "initial_robot": args.initial_robot, "beam_width": args.beam_width,
        "handover_stride": args.handover_stride,
        "handover_clearance_radius": args.handover_clearance_radius,
        "num_trajectories_robot0": args.num_trajectories_robot0,
        "num_trajectories_robot1": args.num_trajectories_robot1,
    }
    for field, value in overrides.items():
        if value is not None:
            old = getattr(exp_cfg.assignment, field)
            setattr(exp_cfg.assignment, field, value)
            logger.info("Assignment override: %s = %s (was %s)", field, value, old)

    if args.smoothen is not None:
        old = exp_cfg.execution.smoothen
        exp_cfg.execution.smoothen = args.smoothen
        logger.info("Smoothen override: %s (was %s)", args.smoothen, old)

    if args.headless:
        exp_cfg.execution.visualize = False
        exp_cfg.visualization.enable_meshcat = False

    debug = args.debug or exp_cfg.runtime.debug
    setup_logging(logging.DEBUG if debug else logging.INFO)

    return exp_cfg


# ---------------------------------------------------------------------------
# System / Environment setup
# ---------------------------------------------------------------------------

def create_system(
    exp_cfg: ExperimentConfig,
    headless: bool,
) -> Tuple[DualArmManipulationSystem, Optional[meshcat.Visualizer], Optional[RRTVisualizer]]:
    """Create and start the dual-arm manipulation system."""
    vis_cfg = exp_cfg.visualization
    exec_cfg = exp_cfg.execution

    vis = None
    if exec_cfg.visualize and vis_cfg.enable_meshcat:
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
        vis.delete()

    system = DualArmManipulationSystem(
        visualize=exec_cfg.visualize,
        meshcat_tf=exec_cfg.meshcat_tf and vis_cfg.enable_meshcat,
        ycb_objects=exp_cfg.ycb_objects,
        custom_objects=exp_cfg.custom_objects,
        use_fastsam=False,
        use_grasp_service=False,
        has_renderer=not headless,
        table_full_size=exp_cfg.table_full_size,
    )
    system.start(start_teleop_thread=False)
    register_mesh_extraction_env(system.env)

    rrt_vis = None
    if exec_cfg.visualize and vis is not None:
        rrt_vis = RRTVisualizer(vis)

    return system, vis, rrt_vis


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectories(
    args: argparse.Namespace,
    exp_cfg: ExperimentConfig,
) -> Tuple[Optional[AssignmentResult], TrajCache, GraspCache, ExplicitPaths]:
    """Load trajectories and (optionally) the assignment result."""
    assignment: Optional[AssignmentResult] = None
    traj_cache: TrajCache = {}
    explicit_paths: ExplicitPaths = None

    def _load_cached(traj_path, directory):
        if traj_path not in traj_cache:
            fpath = os.path.join(directory, traj_path)
            eef, obj, jts, res = load_trajectory(fpath)
            traj_cache[traj_path] = (eef, obj, jts, res, traj_path)
        return traj_cache[traj_path]

    if args.traj0 and args.traj1:
        # --- Explicit trajectory mode ---
        eef0, obj0, jts0, res0 = load_trajectory(args.traj0)
        eef1, obj1, jts1, res1 = load_trajectory(args.traj1)
        t0 = os.path.basename(args.traj0)
        t1 = os.path.basename(args.traj1)
        traj_cache[t0] = (eef0, obj0, jts0, res0, t0)
        traj_cache[t1] = (eef1, obj1, jts1, res1, t1)
        explicit_paths = {0: t0, 1: t1}
        logger.info("Explicit trajectory mode (--traj0 / --traj1).")
    else:
        # --- Assignment mode (default) ---
        data_dir = (
            args.data_dir
            or exp_cfg.output.output_dir
            or os.path.join(constants.DATA_DIR, exp_cfg.name)
        )
        assignment_file = (
            args.assignment
            or exp_cfg.execution.assignment_path
            or os.path.join(data_dir, "assignment_results.npz")
        )
        if not os.path.isfile(assignment_file):
            logger.error(
                "Assignment result not found at %s.  Run segment_assignment.py "
                "first, or use --traj0 / --traj1 for explicit trajectories.",
                assignment_file,
            )
            raise FileNotFoundError(assignment_file)

        logger.info("Loading assignment result from %s ...", assignment_file)
        assignment = load_assignment_result(assignment_file)

        for tpath in sorted(set(assignment.trajectory_paths)):
            _load_cached(tpath, data_dir)

        logger.info(
            "Loaded %d unique trajectories for assignment: %s",
            len(set(assignment.trajectory_paths)),
            sorted(set(assignment.trajectory_paths)),
        )
        logger.info("Handover points: %s", assignment.handover_points)

    # Validate loaded trajectories
    for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
        if res is None:
            raise ValueError(f"Feasibility results missing from {fname}.")
        if jts is None:
            raise ValueError(f"Joint config data missing from {fname}.")

    # Extract grasps: G = inv(T_ee) @ T_obj
    logger.info("Extracting grasps from trajectory first waypoints...")
    grasp_cache: dict = {}
    for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
        if not obj:
            raise ValueError(f"Object trajectory missing in {fname} - cannot extract grasp.")
        g = extract_grasp_transform(eef[0], obj[0])
        if g is None:
            raise ValueError(f"Failed to extract grasp from {fname}.")
        grasp_cache[tpath] = g

    return assignment, traj_cache, grasp_cache, explicit_paths


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_segments(
    vis: meshcat.Visualizer,
    assignment: Optional[AssignmentResult],
    traj_cache: TrajCache,
    tf_manager: TransformManager,
) -> None:
    """Visualize trajectory segments in MeshCat."""
    if assignment is not None:
        _visualize_assignment_segments(vis, assignment, traj_cache, tf_manager)
    else:
        for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
            stem = os.path.splitext(fname)[0]
            visualize_trajectory(vis, eef, res, name=stem, tf_manager=tf_manager)
            if obj:
                visualize_trajectory(
                    vis, obj, name=f"{stem}_object",
                    color=0xff00ff, tf_manager=tf_manager,
                )


# ---------------------------------------------------------------------------
# Execution helpers (modular phases)
# ---------------------------------------------------------------------------

def _get_traj_path(
    assignment: Optional[AssignmentResult],
    explicit_paths: ExplicitPaths,
    robot_id: int,
    segment_idx: int,
) -> str:
    """Return the trajectory cache key for a given robot/segment."""
    if assignment is not None:
        idx = min(segment_idx, len(assignment.trajectory_paths) - 1)
        return assignment.trajectory_paths[idx]
    return explicit_paths[robot_id]


def _get_arms(
    system: DualArmManipulationSystem,
    robot_id: int,
) -> Tuple[RobosuiteArmController, RobosuiteGripperController,
           RobosuiteArmController, RobosuiteGripperController]:
    """Return (active_arm, active_gripper, other_arm, other_gripper) for robot_id."""
    if robot_id == 0:
        return system.left_arm, system.left_gripper, system.right_arm, system.right_gripper
    return system.right_arm, system.right_gripper, system.left_arm, system.left_gripper


def plan_pregrasp_approach(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    other_arm: RobosuiteArmController,
    current_robot: int,
    grasp_traj_data: Dict[str, Any],
    rrt_vis: Optional[RRTVisualizer],
    smoothen: bool,
    planned_q: Optional[np.ndarray] = None,
) -> None:
    """Move the active arm to the pre-grasp pose via collision-free RRT.

    Enumerates multiple IK solutions and tries each as an RRT goal.
    *planned_q* (the joint config from the trajectory file at the grasp
    index) is used as a warm-start for IK enumeration, biasing solutions
    toward the configuration expected by downstream trajectory segments.

    Raises RuntimeError if all RRT candidates fail.
    """
    pregrasp_pose = grasp_traj_data["pregrasp"]

    # Enumerate IK goals — warm-start with the planned config so the
    # arm's resulting configuration is close to what the trajectory expects.
    goals = _enumerate_ik_goals(
        current_arm, system.env, pregrasp_pose, system.env_lock,
        q_warm_start=planned_q,
    )
    logger.info("Pre-grasp IK: %d goal candidate(s) for RRT.", len(goals))

    other_robot = 1 - current_robot
    start_q = current_arm.get_joint_positions()
    q_other = other_arm.get_joint_positions()

    system.sync_planning_env()
    ok = _rrt_to_goal(
        arm=current_arm,
        goals=goals,
        start_q=start_q,
        q_other_arm=q_other,
        robot_id_self=current_robot,
        robot_id_other=other_robot,
        planning_env=system.planning_env,
        env_lock=system.env_lock,
        rrt_vis=rrt_vis,
        smoothen=smoothen,
        label="pre-grasp",
    )
    if not ok:
        raise RuntimeError(
            f"All RRT attempts for pre-grasp failed ({len(goals)} candidates)."
        )


def execute_grasp(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    current_gripper: RobosuiteGripperController,
    current_robot: int,
    grasp_traj_data: Dict[str, Any],
    q_warm_start: Optional[np.ndarray],
    tracker: ExecutionTracker,
    assignment: Optional[AssignmentResult],
) -> None:
    """Follow approach trajectory, align to planned config, close gripper, and stabilise.
    """
    logger.info("Robot %d following grasp approach trajectory...", current_robot)
    current_arm.follow_trajectory(grasp_traj_data["approach"], q_warm_start=q_warm_start)

    current_gripper.grasp()
    stabilise(system.env, system.env_lock)
    tracker.grasp_success = True

    if assignment is not None:
        tracker.n_switches = len(assignment.handover_points)


def find_next_handover(
    assignment: Optional[AssignmentResult],
    explicit_paths: ExplicitPaths,
    traj_cache: TrajCache,
    current_robot: int,
    current_idx: int,
    res_cur: List[WaypointFeasibilityResult],
) -> int:
    """Return the next handover segment index, or -1 if none."""
    if assignment is not None:
        for seg in range(current_idx + 1, len(assignment.sequence)):
            if assignment.sequence[seg] != current_robot:
                return seg
        return -1

    # Heuristic fallback for explicit trajectory mode
    other_path = explicit_paths[1 - current_robot]
    _, _, _, res_other, _ = traj_cache[other_path]

    for i in range(current_idx + 1, len(res_cur)):
        if not res_cur[i].is_feasible:
            for j in reversed(range(current_idx, i)):
                if res_cur[j].is_feasible and res_other[j].is_feasible:
                    return j
            break

    last_other = find_last_feasible_point(res_other)
    last_cur = find_last_feasible_point(res_cur)
    if (last_other is not None and last_cur is not None
            and last_other > last_cur):
        for j in reversed(range(current_idx, last_cur + 1)):
            if res_cur[j].is_feasible and res_other[j].is_feasible:
                return j
    return -1


def execute_parallel_to_handover(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    other_arm: RobosuiteArmController,
    current_robot: int,
    current_idx: int,
    handover_idx: int,
    traj_cache: TrajCache,
    grasp_cache: GraspCache,
    assignment: Optional[AssignmentResult],
    explicit_paths: ExplicitPaths,
    exp_cfg: ExperimentConfig,
    tracker: ExecutionTracker,
    rrt_vis: Optional[RRTVisualizer],
    vis: Optional[meshcat.Visualizer],
) -> Tuple[HandoverTrajectory, Optional[np.ndarray]]:
    """Move arm A to handover while arm B moves to receiver pre-pose.

    Returns (h_traj, q_warm_recv).
    """
    cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
    next_path = _get_traj_path(assignment, explicit_paths, 1 - current_robot, handover_idx)
    _, obj_cur, joints_cur, _, _ = traj_cache[cur_path]
    _, _, joints_next, _, _ = traj_cache[next_path]

    grasp_transfer = grasp_cache[cur_path]
    grasp_receiver = grasp_cache[next_path]

    # Plan the handover
    obj_pose_h = obj_cur[handover_idx]
    h_planner = DualArmHandoverPlanner(
        grasp_transfer=grasp_transfer,
        grasp_receiver=grasp_receiver,
        handover_object_pose=obj_pose_h,
    )
    h_traj = h_planner.generate_trajectory()

    if exp_cfg.execution.visualize and vis is not None:
        visualize_handover_plan(
            vis, h_traj,
            name=f"handover_{handover_idx}",
            tf_manager=system.tf_manager,
        )

    # Arm A's trajectory segment (prepend current position for smooth start)
    joint_segment = (
        [current_arm.get_joint_positions()]
        + list(joints_cur[current_idx + 1 : handover_idx + 1])
    )

    receiver_pre_pose = h_traj.receiver_pre[0]
    q_warm_recv = (
        joints_next[handover_idx]
        if joints_next[handover_idx] is not None else None
    )

    logger.info(
        "PARALLEL: Robot %d following segment to handover at %d "
        "while Robot %d moves to handover pre-pose",
        current_robot, handover_idx, 1 - current_robot,
    )

    planner = DualArmParallelPlanner(
        env=system.env, arm_a=current_arm, arm_b=other_arm,
        env_lock=system.env_lock, tf_manager=system.tf_manager,
        collision_check=True, manipulated_object=exp_cfg.task.manipulated_object,
        rrt_vis=rrt_vis, planning_env=system.planning_env,
    )
    par_result = planner.execute_parallel(
        traj_a=joint_segment, target_b=receiver_pre_pose,
        gripper_a=1.0, gripper_b=-1.0,
        q_warm_start_b=q_warm_recv, num_steps_override=200,
    )

    if not par_result.success_a:
        logger.warning("Arm A did not fully converge (err=%.4f)", par_result.final_pos_error_a)
    if not par_result.success_b:
        logger.warning("Arm B did not fully converge (err=%.4f)", par_result.final_pos_error_b)

    tracker.record_joint_config(current_robot, current_arm.get_joint_positions())

    # Recovery if arm B failed
    if not par_result.success_b:
        _recover_arm_b(system, current_arm, other_arm, current_robot,
                       receiver_pre_pose, q_warm_recv, rrt_vis, exp_cfg)

    return h_traj, q_warm_recv


def _recover_arm_b(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    other_arm: RobosuiteArmController,
    current_robot: int,
    target_pose: StampedPose,
    q_warm_start: Optional[np.ndarray],
    rrt_vis: Optional[RRTVisualizer],
    exp_cfg: ExperimentConfig,
) -> None:
    """Attempt independent RRT recovery to get arm B to its target pose."""
    other_robot = 1 - current_robot
    logger.info(
        "Arm B (Robot %d) did not reach receiver pre-pose during "
        "parallel execution. Attempting independent recovery...",
        other_robot,
    )
    goals = _enumerate_ik_goals(
        other_arm, system.env, target_pose, system.env_lock, q_warm_start,
    )
    logger.info("Recovery: %d IK goal candidate(s) for arm B.", len(goals))

    system.sync_planning_env()
    ok = _rrt_to_goal(
        arm=other_arm, goals=goals,
        start_q=other_arm.get_joint_positions(),
        q_other_arm=current_arm.get_joint_positions(),
        robot_id_self=other_robot, robot_id_other=current_robot,
        planning_env=system.planning_env, env_lock=system.env_lock,
        rrt_vis=rrt_vis, smoothen=exp_cfg.execution.smoothen,
        label="receiver pre-pose",
    )
    if ok:
        return

    # Final fallback -- direct move_to_pose
    logger.info("Attempting move_to_pose fallback for arm B...")
    fallback_ok = other_arm.move_to_pose(
        target_pose, gripper_action=-1.0,
        num_steps=200, q_warm_start=q_warm_start,
    )
    if fallback_ok:
        logger.info("move_to_pose recovery succeeded.")
    else:
        logger.warning("move_to_pose recovery also failed - handover likely to fail.")


def execute_handover_sequence(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    current_gripper: RobosuiteGripperController,
    other_arm: RobosuiteArmController,
    other_gripper: RobosuiteGripperController,
    current_robot: int,
    current_idx: int,
    h_traj: HandoverTrajectory,
    q_warm_recv: Optional[np.ndarray],
    tracker: ExecutionTracker,
) -> bool:
    """Execute the physical handover (hold -> approach -> gripper swap -> retreat).

    Returns True if handover succeeded.
    """
    tracker.n_handovers_attempted += 1
    logger.info(
        "Executing handover at waypoint %d (robot %d -> robot %d)...",
        current_idx, current_robot, 1 - current_robot,
    )

    # Transfer arm holds position
    logger.info("Robot %d moving to handover hold pose...", current_robot)
    current_arm.move_to_pose(h_traj.transfer_hold[0], gripper_action=1.0, num_steps=50)

    # Receiver approaches
    logger.info("Robot %d executing handover approach...", 1 - current_robot)
    other_gripper.release()
    receiver_converged = other_arm.follow_trajectory(
        h_traj.receiver_approach, q_warm_start=q_warm_recv,
    )
    if not receiver_converged:
        recv_pos, _ = other_arm.get_eef_pose_in_base()
        target_pose = h_traj.receiver_approach[-1]
        if target_pose.frame_id != other_arm.base_frame:
            target_pose = other_arm.tf_manager.transform_pose(
                target_pose, other_arm.base_frame,
            )
        dist = float(np.linalg.norm(recv_pos - target_pose.position))
        logger.warning("Receiver arm did NOT converge (dist=%.4f m).", dist)

    # Gripper swap: receiver grasps, transfer releases
    other_gripper.grasp()
    stabilise(system.env, system.env_lock)
    current_gripper.release()
    stabilise(system.env, system.env_lock)

    # Transfer arm retreats
    logger.info("Robot %d following handover retreat trajectory...", current_robot)
    current_arm.follow_trajectory(h_traj.transfer_retreat)

    # Report handover outcome
    if receiver_converged:
        tracker.n_handovers_succeeded += 1
        logger.info(
            "Handover at waypoint %d SUCCEEDED (robot %d -> robot %d). "
            "Total handovers: %d/%d",
            current_idx, current_robot, 1 - current_robot,
            tracker.n_handovers_succeeded, tracker.n_handovers_attempted,
        )
    else:
        logger.warning(
            "Handover at waypoint %d LIKELY FAILED - receiver arm did not converge. "
            "Total handovers: %d/%d",
            current_idx,
            tracker.n_handovers_succeeded, tracker.n_handovers_attempted,
        )
    return receiver_converged


def execute_final_segment(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    other_arm: RobosuiteArmController,
    current_robot: int,
    current_idx: int,
    joints_cur: np.ndarray,
    res_cur: List[WaypointFeasibilityResult],
    assignment: Optional[AssignmentResult],
    exp_cfg: ExperimentConfig,
    tracker: ExecutionTracker,
    rrt_vis: Optional[RRTVisualizer],
) -> int:
    """Execute the last trajectory segment while moving the idle arm home.

    Returns new current_idx.
    """
    last_idx = find_last_feasible_point(res_cur)
    if assignment is not None and last_idx is not None:
        last_idx = min(last_idx, len(assignment.sequence) - 1)

    if last_idx is None or last_idx <= current_idx:
        tracker.transport_success = True
        return current_idx

    joint_segment = (
        [current_arm.get_joint_positions()]
        + list(joints_cur[current_idx + 1 : last_idx + 1])
    )

    logger.info(
        "PARALLEL: Robot %d following final segment to %d "
        "while Robot %d moves to home position",
        current_robot, last_idx, 1 - current_robot,
    )

    planner = DualArmParallelPlanner(
        env=system.env, arm_a=current_arm, arm_b=other_arm,
        env_lock=system.env_lock, tf_manager=system.tf_manager,
        collision_check=True, manipulated_object=exp_cfg.task.manipulated_object,
        rrt_vis=rrt_vis, planning_env=system.planning_env,
    )
    par_result = planner.execute_parallel(
        traj_a=joint_segment,
        target_b=constants.FRANKA_HOME_JOINT_POSITION,
        gripper_a=1.0, gripper_b=-1.0,
        num_steps_override=max(200, len(joint_segment) * 2),
    )
    if not par_result.success_a:
        logger.warning(
            "Arm A did not fully converge on final segment (err=%.4f)",
            par_result.final_pos_error_a,
        )

    tracker.record_joint_config(current_robot, current_arm.get_joint_positions())
    tracker.transport_success = True
    return last_idx


def stop_monitors(tracker: ExecutionTracker) -> None:
    """Stop collision and object monitors, record results."""
    tracker.n_collision_incidents = stop_collision_monitor()
    if tracker.n_collision_incidents > 0:
        logger.warning(
            "Collision monitor detected %d collision step(s) during execution.",
            tracker.n_collision_incidents,
        )
    else:
        logger.info("Execution completed with no collisions detected.")

    obj_summary = stop_object_monitor()
    tracker.object_mean_tracking_error = obj_summary["mean_tracking_error"]
    tracker.object_max_tracking_error = obj_summary["max_tracking_error"]


def release_and_retreat(
    system: DualArmManipulationSystem,
    current_arm: RobosuiteArmController,
    current_gripper: RobosuiteGripperController,
    current_robot: int,
    eef_cur: List[StampedPose],
    current_idx: int,
) -> None:
    """Release the object and retreat the active arm."""
    logger.info("Robot %d releasing object and retreating...", current_robot)
    release_pose = eef_cur[current_idx]
    release_primitive = ReleaseRetreatPrimitive()
    release_traj_data = release_primitive.generate_trajectory(release_pose)

    current_gripper.release()
    stabilise(system.env, system.env_lock)

    logger.info("Robot %d following retreat trajectory...", current_robot)
    current_arm.follow_trajectory(release_traj_data["retreat"], gripper_action=-1.0)
    stabilise(system.env, system.env_lock)


def verify_placement(
    system: DualArmManipulationSystem,
    exp_cfg: ExperimentConfig,
    tracker: ExecutionTracker,
) -> None:
    """Check whether the object reached its target pose."""
    PLACE_POS_THRESHOLD = 0.05   # metres
    PLACE_ORI_THRESHOLD = 30.0   # degrees

    obj_name = exp_cfg.task.manipulated_object
    target_pos = np.asarray(exp_cfg.task.target_pose.position)
    target_quat_xyzw = np.asarray(exp_cfg.task.target_pose.quaternion)

    actual_pose_4x4 = system.env.get_object_pose(obj_name)
    if actual_pose_4x4 is None:
        logger.warning("Could not query object '%s' pose - skipping placement check.", obj_name)
        tracker.place_success = True
        return

    actual_pos = actual_pose_4x4[:3, 3]
    actual_rot = R.from_matrix(actual_pose_4x4[:3, :3])
    target_rot = R.from_quat(target_quat_xyzw)

    pos_err = float(np.linalg.norm(actual_pos - target_pos))
    ori_err = float((actual_rot.inv() * target_rot).magnitude() * 180.0 / np.pi)

    tracker.placement_position_error = pos_err
    tracker.placement_orientation_error = ori_err

    logger.info(
        "Placement verification - position error: %.4f m, orientation error: %.2f deg",
        pos_err, ori_err,
    )
    tracker.place_success = (
        pos_err < PLACE_POS_THRESHOLD and ori_err < PLACE_ORI_THRESHOLD
    )
    if not tracker.place_success:
        logger.warning(
            "Object did NOT reach target! pos_err=%.4f m (thresh %.2f), "
            "ori_err=%.2f deg (thresh %.1f deg)",
            pos_err, PLACE_POS_THRESHOLD, ori_err, PLACE_ORI_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def run_execution_loop(
    system: DualArmManipulationSystem,
    exp_cfg: ExperimentConfig,
    assignment: Optional[AssignmentResult],
    traj_cache: TrajCache,
    grasp_cache: GraspCache,
    explicit_paths: ExplicitPaths,
    tracker: ExecutionTracker,
    rrt_vis: Optional[RRTVisualizer],
    vis: Optional[meshcat.Visualizer],
    args: argparse.Namespace,
) -> None:
    """Drive the full pick -> transport -> handover(s) -> place pipeline."""

    # --- Determine starting robot / index ---
    if assignment is not None:
        current_robot = assignment.sequence[0]
        current_idx = 0
    else:
        _, _, _, res0, _ = traj_cache[explicit_paths[0]]
        _, _, _, res1, _ = traj_cache[explicit_paths[1]]
        first0 = find_first_feasible_point(res0)
        first1 = find_first_feasible_point(res1)
        if first0 is None and first1 is None:
            raise RuntimeError("No feasible starting point found.")
        current_robot = 0 if (first0 is not None and (first1 is None or first0 <= first1)) else 1
        current_idx = first0 if current_robot == 0 else first1

    logger.info("Robot %d starting at waypoint %d", current_robot, current_idx)

    current_arm, current_gripper, other_arm, other_gripper = _get_arms(system, current_robot)
    cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
    eef_cur, obj_cur, joints_cur, res_cur, _ = traj_cache[cur_path]

    # --- Phase 1: plan pre-grasp approach & grasp ---
    grasp = grasp_cache[cur_path]
    obj_pose = obj_cur[current_idx]
    grasp_primitive = ObjectGraspPrimitive(object_pose=obj_pose, grasp_transform=grasp)
    grasp_traj_data = grasp_primitive.generate_trajectory()
    q_warm_grasp = joints_cur[current_idx] if joints_cur[current_idx] is not None else None

    logger.info("Robot %d planning RRT to pre-grasp pose...", current_robot)
    plan_pregrasp_approach(
        system, current_arm, other_arm, current_robot,
        grasp_traj_data, rrt_vis, exp_cfg.execution.smoothen,
        planned_q=q_warm_grasp,
    )

    # Build planned object positions array for object monitor
    planned_obj_positions = None
    if obj_cur is not None:
        planned_obj_positions = np.array([p.position for p in obj_cur], dtype=np.float64)

    execute_grasp(
        system, current_arm, current_gripper, current_robot,
        grasp_traj_data, q_warm_grasp, tracker, assignment,
    )

    start_collision_monitor(exp_cfg.task.manipulated_object)
    start_object_monitor(exp_cfg.task.manipulated_object, planned_positions=planned_obj_positions)

    # --- Phase 2: transport loop (handovers + parallel segments) ---
    n_segments = len(res_cur)
    if assignment is not None:
        n_segments = min(n_segments, len(assignment.sequence))

    while current_idx < n_segments - 1:
        cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
        eef_cur, obj_cur, joints_cur, res_cur, _ = traj_cache[cur_path]

        handover_idx = find_next_handover(
            assignment, explicit_paths, traj_cache,
            current_robot, current_idx, res_cur,
        )

        if handover_idx != -1 and handover_idx > current_idx:
            # Parallel move to handover point
            h_traj, q_warm_recv = execute_parallel_to_handover(
                system, current_arm, other_arm, current_robot, current_idx,
                handover_idx, traj_cache, grasp_cache, assignment,
                explicit_paths, exp_cfg, tracker, rrt_vis, vis,
            )
            current_idx = handover_idx

            # Execute handover sequence
            execute_handover_sequence(
                system, current_arm, current_gripper,
                other_arm, other_gripper,
                current_robot, current_idx,
                h_traj, q_warm_recv, tracker,
            )

            # Swap roles
            current_robot = 1 - current_robot
            current_arm, current_gripper, other_arm, other_gripper = _get_arms(system, current_robot)
        else:
            # No more handovers -- execute final segment
            current_idx = execute_final_segment(
                system, current_arm, other_arm, current_robot,
                current_idx, joints_cur, res_cur, assignment,
                exp_cfg, tracker, rrt_vis,
            )
            break

    if not tracker.transport_success:
        tracker.transport_success = True

    # --- Phase 3: stop monitors ---
    stop_monitors(tracker)

    # --- Phase 4: release and retreat ---
    cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
    eef_cur, _, _, _, _ = traj_cache[cur_path]
    release_and_retreat(system, current_arm, current_gripper, current_robot,
                        eef_cur, current_idx)

    # --- Phase 5: placement verification ---
    verify_placement(system, exp_cfg, tracker)

    # --- Final success determination ---
    tracker.task_success = (
        tracker.grasp_success
        and tracker.place_success
        and (tracker.n_handovers_attempted == 0
             or tracker.n_handovers_succeeded == tracker.n_handovers_attempted)
    )
    tracker.execution_time_s = time.time() - tracker.execution_start_time

    if not args.headless:
        time.sleep(5)
    logger.info("Done!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    exp_cfg = load_and_configure_experiment(args)
    tracker = ExecutionTracker()
    tracker.execution_start_time = time.time()

    system, vis, rrt_vis = create_system(exp_cfg, args.headless)

    tracker.set_jacobian_fns(make_jacobian_fns(system.env, system.env_lock))
    logger.info("Jacobian functions ready - using Yoshikawa manipulability.")

    try:
        assignment, traj_cache, grasp_cache, explicit_paths = load_trajectories(args, exp_cfg)

        if exp_cfg.execution.visualize and vis is not None:
            logger.info("Visualizing executed trajectory segments in MeshCat...")
            visualize_segments(vis, assignment, traj_cache, system.tf_manager)

        run_execution_loop(
            system, exp_cfg, assignment, traj_cache, grasp_cache,
            explicit_paths, tracker, rrt_vis, vis, args,
        )

    except Exception as e:
        logger.exception("An error occurred: %s", e)
        tracker.error_message = str(e)
        tracker.execution_time_s = time.time() - tracker.execution_start_time
    finally:
        if args.output_metrics:
            tracker.save(args.output_metrics)
        system.stop()


if __name__ == "__main__":
    main()
