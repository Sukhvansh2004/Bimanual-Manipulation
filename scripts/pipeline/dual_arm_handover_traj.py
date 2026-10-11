#!/usr/bin/env python3
"""Dual-arm handover execution from trajectory files and experiment JSON configs.

By default, loads the assignment result (``assignment_results.npz``) from the
experiment's data directory (or ``--data-dir`` / ``--assignment``) and executes
it: pre-grasp RRT, grasp, parallel transport, handover(s), final segment,
release and placement verification. Use ``--traj0`` and ``--traj1`` together
to bypass the assignment and specify trajectory files explicitly. Execution
metrics can be written to JSON with ``--output-metrics``.

Assignment flags (e.g. ``--alpha``, ``--beta``, ``--initial-robot``) override
the corresponding fields of the experiment's ``assignment`` config. Note that
this script only loads a precomputed assignment, so these overrides are logged
but do not change what is executed.

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
import random
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
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.monitors import (
    start_collision_monitor, stop_collision_monitor,
    start_object_monitor, stop_object_monitor,
)
from bimanual_manipulation.utils.object_factory import register_mesh_extraction_env
from bimanual_manipulation.utils.suction_weld import is_object_grasped
from bimanual_manipulation.experiment_config import load_experiment, ExperimentConfig
from bimanual_manipulation.planning.trajectory.utils import (
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
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer
from bimanual_manipulation.planning.trajectory.arm_planner import plan_arm_trajectory
from bimanual_manipulation.planning.assignment.utils import (
    load_assignment_result,
    normalize_joint_angle,
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
from bimanual_manipulation.utils.trajectory_math import (
    velocity_profile_joint_trajectory,
)

def _realign_joint_segment(segment: list) -> list:
    """Re-normalise each config in ``segment`` relative to its predecessor.

    The first element (the actual robot state) is kept unchanged. Every
    subsequent config is shifted by multiples of 2π per joint (within Franka
    joint limits) to minimise the step from the preceding config, eliminating
    reflex-angle jumps that can arise when IK returns a wrapped solution.

    Args:
        segment: Sequence of joint configurations, each of shape ``(7,)``.

    Returns:
        New list of configs; ``segment`` itself if it has at most one element.
    """
    if len(segment) <= 1:
        return segment
    result = [segment[0]]
    prev = np.asarray(segment[0])
    for cfg in segment[1:]:
        aligned = np.array(cfg, dtype=float)
        for j in range(len(aligned)):
            aligned[j] = normalize_joint_angle(
                aligned[j], float(prev[j]),
                float(constants.FRANKA_JOINT_LIMITS_LO[j]),
                float(constants.FRANKA_JOINT_LIMITS_HI[j]),
            )
        result.append(aligned)
        prev = aligned
    return result


def _normalize_config_to_reference(
    actual: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Shift ``actual`` per joint by multiples of 2π to be closest to ``reference``.

    Fixes the case where IK / mink returns a physically-equivalent config
    that is 2π away on one or more joints. Results stay within Franka joint
    limits.

    Args:
        actual: Joint configuration to normalise, shape ``(7,)``.
        reference: Reference joint configuration, shape ``(7,)``.

    Returns:
        A normalised float copy of ``actual``.
    """
    out = np.array(actual, dtype=float)
    for j in range(len(out)):
        out[j] = normalize_joint_angle(
            out[j], float(reference[j]),
            float(constants.FRANKA_JOINT_LIMITS_LO[j]),
            float(constants.FRANKA_JOINT_LIMITS_HI[j]),
        )
    return out


def _log_joint_delta(
    label: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    """Log the per-joint delta between two configs at DEBUG level.

    Args:
        label: Tag included in the log line.
        actual: Measured joint configuration.
        expected: Planned joint configuration.

    Returns:
        Maximum absolute per-joint difference ``max|actual - expected|``.
    """
    delta = np.asarray(actual) - np.asarray(expected)
    max_d = float(np.max(np.abs(delta)))
    logger.debug(
        "JOINT-MONITOR [%s]  max|dq|=%.4f  "
        "dq=[%s]",
        label, max_d,
        ", ".join(f"{d:+.4f}" for d in delta),
    )
    return max_d


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
# Gripper-type mapping (Robot 0 = finger, Robot 1 = suction)
# ---------------------------------------------------------------------------
_ROBOT_GRIPPER = {0: "finger", 1: "suction"}

# ---------------------------------------------------------------------------
# Execution metrics tracker
# ---------------------------------------------------------------------------

class ExecutionTracker:
    """Tracks metrics during task execution for later reporting."""

    def __init__(self):
        """Initialize all metrics to their default (unsuccessful/zero) values."""
        self.task_success = False
        self.grasp_success = False
        self.transport_success = False
        self.place_success = False
        self.n_handovers_attempted = 0
        self.n_handovers_succeeded = 0
        self.n_handovers_lost_object = 0
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
        self.handover_results: List[Dict] = []  # per-handover outcome records
        self.grasp_result: Optional[Dict] = None  # initial grasp outcome record

    @property
    def handover_success_rate(self) -> float:
        """Fraction of attempted handovers that succeeded (1.0 if none attempted)."""
        if self.n_handovers_attempted == 0:
            return 1.0
        return self.n_handovers_succeeded / self.n_handovers_attempted

    def set_jacobian_fns(self, jacobian_fns: Dict[int, Callable]):
        """Set the per-robot Jacobian functions used for manipulability.

        Args:
            jacobian_fns: Mapping from robot id to a function ``q -> J``.
        """
        self._jacobian_fns = jacobian_fns

    def record_joint_config(self, robot_id: int, q: np.ndarray):
        """Record a copy of an executed joint configuration.

        Args:
            robot_id: Robot index (0 or 1).
            q: Joint configuration.
        """
        self.executed_joint_configs.append((robot_id, np.array(q, copy=True)))

    @property
    def mean_manipulability(self) -> float:
        """Mean Yoshikawa manipulability over recorded joint configs.

        Uses the robot's Jacobian function when set, otherwise falls back to
        ``compute_trajectory_manipulability``. Returns 0.0 if no configs
        were recorded.
        """
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
        """Return all metrics as a JSON-serialisable dict."""
        return {
            "task_success": self.task_success,
            "grasp_success": self.grasp_success,
            "transport_success": self.transport_success,
            "place_success": self.place_success,
            "n_handovers_attempted": self.n_handovers_attempted,
            "n_handovers_succeeded": self.n_handovers_succeeded,
            "n_handovers_lost_object": self.n_handovers_lost_object,
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
            "handover_results": self.handover_results,
            "grasp_result": self.grasp_result,
        }

    def save(self, path: str):
        """Write ``to_dict()`` as indented JSON.

        Args:
            path: Output JSON file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Execution metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a timestamped format.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )


def find_first_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Return the index of the first feasible waypoint.

    Args:
        waypoint_results: Per-waypoint feasibility results.

    Returns:
        Index of the first result with ``is_feasible`` set, or ``None``.
    """
    for i, res in enumerate(waypoint_results):
        if res.is_feasible:
            return i
    return None


def find_last_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Return the index of the last feasible waypoint.

    Args:
        waypoint_results: Per-waypoint feasibility results.

    Returns:
        Index of the last result with ``is_feasible`` set, or ``None``.
    """
    for i in reversed(range(len(waypoint_results))):
        if waypoint_results[i].is_feasible:
            return i
    return None


def stabilise(env: DualArmEnvironment, env_lock: threading.RLock,
              steps: int = 50) -> None:
    """Step the environment to stabilise after gripper actions.

    Args:
        env: Simulation environment.
        env_lock: Lock held while stepping; skipped if falsy.
        steps: Number of simulation steps.
    """
    with env_lock if env_lock else nullcontext():
        for _ in range(steps):
            env_step(env)


# ---------------------------------------------------------------------------
# Object-aware checks
# ---------------------------------------------------------------------------

# Contact-distance threshold for declaring gripper-object contact.
_CONTACT_DIST_THRESHOLD = 0.005  # 5 mm


def _object_held_by(
    system: DualArmManipulationSystem,
    robot_id: int,
    arm: RobosuiteArmController,
    obj_name: str,
    gripper: Optional[RobosuiteGripperController] = None,
) -> bool:
    """Return ``True`` if ``robot_id`` is physically holding the object.

    For suction grippers the suction weld equality constraint is checked;
    if the weld is active the object is held.

    For parallel-jaw grippers (or when ``gripper`` is ``None``), the MuJoCo
    contact buffer is scanned for contacts with distance at most
    ``_CONTACT_DIST_THRESHOLD`` between non-visual ``gripper<robot_id>_*``
    geoms and the geoms of the first body whose name contains ``obj_name``.

    Args:
        system: Running dual-arm system.
        robot_id: Robot index (0 or 1).
        arm: Arm controller of that robot (used for the suction weld check).
        obj_name: Name (substring) of the object body.
        gripper: Gripper controller; selects the suction check if suction.

    Returns:
        ``True`` if the object is held; ``False`` otherwise, including when
        there are no contacts or the gripper/object geoms cannot be found.
    """
    env = system.env
    sim = env.sim

    # --- Suction shortcut: weld active ⇒ held ---
    if gripper is not None and gripper.is_suction:
        return is_object_grasped(env, robot_id, arm.arm_id)

    # --- Contact-based check (parallel-jaw or unknown gripper) ---
    model = sim.model
    ncon = int(sim.data.ncon)
    if ncon <= 0:
        logger.warning(
            "No MuJoCo contacts — robot %d likely not holding '%s'.",
            robot_id, obj_name,
        )
        return False

    # Build gripper geom mask
    gripper_prefix = f"gripper{robot_id}_"
    gripper_geom_ids = set()
    for gid in range(model.ngeom):
        name = model.geom(gid).name if hasattr(model, 'geom') else model.geom_id2name(gid)
        if name and name.startswith(gripper_prefix) and "vis" not in name:
            gripper_geom_ids.add(gid)

    if not gripper_geom_ids:
        logger.warning("No gripper geoms found for robot %d.", robot_id)
        return False

    # Build object geom mask
    obj_body_id = None
    for name, bid in env.object_body_ids.items():
        if obj_name in name:
            obj_body_id = int(bid)
            break
    if obj_body_id is None:
        logger.warning("Object '%s' body not found.", obj_name)
        return False

    obj_geom_ids = set()
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == obj_body_id:
            obj_geom_ids.add(gid)

    # Scan contacts for gripper ↔ object pairs
    g1 = np.asarray(sim.data.contact.geom1[:ncon], dtype=np.int32)
    g2 = np.asarray(sim.data.contact.geom2[:ncon], dtype=np.int32)
    dist = np.asarray(sim.data.contact.dist[:ncon], dtype=np.float64)

    for i in range(ncon):
        if dist[i] > _CONTACT_DIST_THRESHOLD:
            continue
        g1i, g2i = int(g1[i]), int(g2[i])
        if (g1i in gripper_geom_ids and g2i in obj_geom_ids) or \
           (g2i in gripper_geom_ids and g1i in obj_geom_ids):
            return True

    logger.warning(
        "No contact between robot %d gripper and object '%s' — "
        "grip likely lost.",
        robot_id, obj_name,
    )
    return False


def _object_not_fallen(
    system: DualArmManipulationSystem,
    obj_name: str,
    min_z: float = 0.0,
) -> bool:
    """Return ``True`` if the object has not fallen below ``min_z``.

    A fallen object is a clear sign that a grasp or handover failed.

    Args:
        system: Running dual-arm system.
        obj_name: Name of the object.
        min_z: World z threshold in metres (0 is floor level in most
            robosuite environments).

    Returns:
        ``False`` if the object's z is below ``min_z``; ``True`` otherwise,
        including when the object pose cannot be queried.
    """
    obj_pose = system.env.get_object_pose(obj_name)
    if obj_pose is None:
        return True  # can't verify — assume ok
    z = float(obj_pose[2, 3])
    if z < min_z:
        logger.warning(
            "Object '%s' has fallen (z=%.3f m < %.2f m).", obj_name, z, min_z,
        )
        return False
    return True


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
    """Enumerate IK solutions for ``pose`` and return a de-duplicated list.

    The pose is transformed into the arm's base frame and orientation-corrected
    before calling ``enumerate_ik_solutions_for_arm``. Solutions within 0.05
    rad (``np.allclose``) of an earlier one are dropped. Any exception is
    logged at DEBUG level and yields the solutions collected so far.

    Args:
        arm: Arm controller to solve for.
        env: Simulation environment.
        pose: Target EEF pose (any frame).
        env_lock: Simulation lock.
        q_warm_start: Optional reference config used to bias solutions.

    Returns:
        List of joint configurations (possibly empty).
    """
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
    """Plan an RRT path to each IK goal in turn and follow the first that succeeds.

    Args:
        arm: Arm controller that executes the path.
        goals: Candidate goal joint configurations.
        start_q: Start joint configuration of ``arm``.
        q_other_arm: Fixed joint configuration of the other arm.
        robot_id_self: Robot index of ``arm``.
        robot_id_other: Robot index of the other arm.
        planning_env: Environment used for collision checking.
        env_lock: Simulation lock.
        rrt_vis: Optional RRT visualizer.
        smoothen: Whether to smooth the followed joint trajectory.
        gripper_action: Gripper command held while following the path.
        label: Name of the target used in log messages.

    Returns:
        ``True`` if a path was planned and followed, ``False`` if every
        candidate raised ``RuntimeError`` (or ``goals`` is empty).
    """
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
    """Visualize the executed segments, grouped into contiguous runs.

    A run is a maximal range of waypoints with the same robot and
    trajectory path in the assignment.

    Args:
        vis: Meshcat visualizer.
        assignment: Assignment result defining robot and path per waypoint.
        traj_cache: Trajectory cache keyed by trajectory path.
        tf_manager: Transform manager for frame conversion.
    """
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
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
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
                        help="Disable the MuJoCo viewer window (keeps meshcat)")
    parser.add_argument("--no-meshcat", action="store_true",
                        help="Disable meshcat visualization")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing trajectory NPZ files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

def load_and_configure_experiment(args) -> ExperimentConfig:
    """Load the experiment JSON, apply CLI overrides, set up logging and seeds.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The experiment config with assignment, smoothing and meshcat
        overrides applied in place.
    """
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

    if getattr(args, 'no_meshcat', False):
        exp_cfg.execution.visualize = False
        exp_cfg.visualization.enable_meshcat = False

    debug = args.debug or exp_cfg.runtime.debug
    setup_logging(logging.DEBUG if debug else logging.INFO)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
        except Exception:
            logger.debug("Torch unavailable for seeding; continuing with numpy/random only.")
        logger.info("Set random seeds to %d", args.seed)

    return exp_cfg


# ---------------------------------------------------------------------------
# System / Environment setup
# ---------------------------------------------------------------------------

def create_system(
    exp_cfg: ExperimentConfig,
    headless: bool,
) -> Tuple[DualArmManipulationSystem, Optional[meshcat.Visualizer], Optional[RRTVisualizer]]:
    """Create and start the dual-arm manipulation system.

    Args:
        exp_cfg: Experiment configuration.
        headless: If ``True``, disable the MuJoCo viewer.

    Returns:
        Tuple ``(system, vis, rrt_vis)``; ``vis`` and ``rrt_vis`` are ``None``
        when execution visualization or Meshcat is disabled.
    """
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
        table_offset=exp_cfg.table_offset,
        robot_base_configs=exp_cfg.robot_base_configs,
        start_services=False,
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

def _reconstruct_from_assignment(
    assignment: AssignmentResult,
) -> Tuple[TrajCache, GraspCache]:
    """Rebuild ``traj_cache`` and ``grasp_cache`` from embedded assignment data.

    This allows execution without loading individual trajectory NPZ files.
    Every unique trajectory path in the assignment maps to the same synthetic
    entry built from the assigned per-waypoint EEF/object poses and
    ``normalized_configs``.

    Args:
        assignment: Loaded assignment result.

    Returns:
        Tuple ``(traj_cache, grasp_cache)``.
    """
    n_wp = len(assignment.is_feasible)
    eef_poses = [
        StampedPose(
            position=assignment.eef_positions[i],
            quaternion=assignment.eef_quaternions[i],
            frame_id=assignment.eef_frame_ids[i],
        )
        for i in range(n_wp)
    ]
    obj_poses = [
        StampedPose(
            position=assignment.object_positions[i],
            quaternion=assignment.object_quaternions[i],
            frame_id=assignment.object_frame_ids[i],
        )
        for i in range(n_wp)
    ]

    jts = assignment.normalized_configs  # (n_wp, DOF)

    # Build WaypointFeasibilityResult list
    res: List[WaypointFeasibilityResult] = []
    for i in range(n_wp):
        res.append(WaypointFeasibilityResult(
            waypoint_index=i,
            eef_pose=eef_poses[i],
            joint_config=jts[i],
            is_feasible=bool(assignment.is_feasible[i]),
            ik_success=bool(assignment.is_feasible[i]),
            collision_free=bool(assignment.is_feasible[i]),
        ))

    # Create one traj_cache entry per unique trajectory path.
    # All entries share the same assigned-path data; the execution code
    # indexes by waypoint, and the waypoint data comes from the assigned
    # trajectory at each segment.
    traj_cache: TrajCache = {}
    for tpath in sorted(set(assignment.trajectory_paths)):
        traj_cache[tpath] = (eef_poses, obj_poses, jts, res, tpath)

    # Reconstruct grasp_cache from embedded grasp transforms
    grasp_cache: GraspCache = {}
    for gi, gpath in enumerate(assignment.grasp_paths):
        grasp_cache[gpath] = StampedPose(
            position=assignment.grasp_positions[gi],
            quaternion=assignment.grasp_quaternions[gi],
            frame_id=assignment.grasp_frame_ids[gi],
        )

    logger.info(
        "Reconstructed execution data from assignment: "
        "%d waypoints, %d traj_cache entries, %d grasp transforms",
        n_wp, len(traj_cache), len(grasp_cache),
    )
    return traj_cache, grasp_cache


def load_trajectories(
    args: argparse.Namespace,
    exp_cfg: ExperimentConfig,
) -> Tuple[Optional[AssignmentResult], TrajCache, GraspCache, ExplicitPaths]:
    """Load trajectories and, in assignment mode, the assignment result.

    If both ``--traj0`` and ``--traj1`` are given, those files are loaded and
    grasps are extracted from their first waypoints. Otherwise the assignment
    NPZ is resolved from ``--assignment``, ``execution.assignment_path`` or
    ``<data_dir>/assignment_results.npz`` and execution data is
    reconstructed from it.

    Args:
        args: Parsed CLI arguments.
        exp_cfg: Experiment configuration.

    Returns:
        Tuple ``(assignment, traj_cache, grasp_cache, explicit_paths)``;
        ``assignment`` is ``None`` in explicit mode and ``explicit_paths`` is
        ``None`` in assignment mode.

    Raises:
        FileNotFoundError: If the assignment file does not exist.
        ValueError: In explicit mode, if a trajectory lacks feasibility
            results, joint configs or object poses, or grasp extraction fails.
    """
    assignment: Optional[AssignmentResult] = None
    traj_cache: TrajCache = {}
    explicit_paths: ExplicitPaths = None

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

        # All execution data is embedded in the assignment NPZ —
        # no need to load individual trajectory files.
        logger.info(
            "Reconstructing execution data from assignment "
            "(no trajectory files needed)."
        )
        traj_cache, grasp_cache = _reconstruct_from_assignment(assignment)
        logger.info("Handover points: %s", assignment.handover_points)
        return assignment, traj_cache, grasp_cache, explicit_paths

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
    """Visualize trajectory segments in Meshcat.

    Args:
        vis: Meshcat visualizer.
        assignment: Assignment result; if ``None``, every cached trajectory
            is drawn in full.
        traj_cache: Trajectory cache keyed by trajectory path.
        tf_manager: Transform manager for frame conversion.
    """
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
    """Return the trajectory cache key for a given robot/segment.

    Args:
        assignment: Assignment result, or ``None`` in explicit mode.
        explicit_paths: Robot id to path mapping used when ``assignment``
            is ``None``.
        robot_id: Robot index (only used in explicit mode).
        segment_idx: Waypoint index (clamped to the assignment length).

    Returns:
        Key into ``traj_cache``.
    """
    if assignment is not None:
        idx = min(segment_idx, len(assignment.trajectory_paths) - 1)
        return assignment.trajectory_paths[idx]
    return explicit_paths[robot_id]


def _get_arms(
    system: DualArmManipulationSystem,
    robot_id: int,
) -> Tuple[RobosuiteArmController, RobosuiteGripperController,
           RobosuiteArmController, RobosuiteGripperController]:
    """Return the arm/gripper controllers for the active and other robot.

    Robot 0 maps to the left arm and robot 1 to the right arm.

    Args:
        system: Running dual-arm system.
        robot_id: Index of the active robot.

    Returns:
        Tuple ``(active_arm, active_gripper, other_arm, other_gripper)``.
    """
    if robot_id == 0:
        return system.left_arm, system.left_gripper, system.right_arm, system.right_gripper
    return system.right_arm, system.right_gripper, system.left_arm, system.left_gripper


def _get_joint_configs(
    assignment: Optional[AssignmentResult],
    traj_cache: TrajCache,
    traj_path: str,
) -> Optional[np.ndarray]:
    """Return the joint configs to use during execution.

    When an assignment is available its ``normalized_configs`` array is
    returned directly. Otherwise (explicit-trajectory mode) the raw configs
    of ``traj_path`` are returned.

    Args:
        assignment: Assignment result, or ``None``.
        traj_cache: Trajectory cache keyed by trajectory path.
        traj_path: Cache key used in explicit mode.

    Returns:
        Joint configs of shape ``(N, 7)``, or ``None`` if unavailable.
    """
    if assignment is not None:
        return assignment.normalized_configs
    
    logger.debug("Using raw trajectory joint configs for execution.")
    return traj_cache[traj_path][2]


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
    ``planned_q`` (the planned joint config at the grasp index) is used as a
    warm-start for IK enumeration, biasing solutions toward the configuration
    expected by downstream trajectory segments.

    Args:
        system: Running dual-arm system.
        current_arm: Arm that will grasp.
        other_arm: Idle arm (held fixed for collision checking).
        current_robot: Index of the grasping robot.
        grasp_traj_data: Output of ``ObjectGraspPrimitive.generate_trajectory``;
            must contain ``"pregrasp"``.
        rrt_vis: Optional RRT visualizer.
        smoothen: Whether to smooth the followed joint trajectory.
        planned_q: Optional IK warm-start configuration.

    Raises:
        RuntimeError: If all RRT candidates fail (or no IK goal is found).
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
    obj_name: str = "",
) -> None:
    """Follow the grasp approach, close the gripper, and record the grasp outcome.

    Sets ``tracker.grasp_success`` (via ``_object_held_by`` when ``obj_name``
    is given, otherwise always ``True``), ``tracker.grasp_result`` and, in
    assignment mode, ``tracker.n_switches``.

    Args:
        system: Running dual-arm system.
        current_arm: Arm that grasps.
        current_gripper: Gripper of that arm.
        current_robot: Index of the grasping robot.
        grasp_traj_data: Grasp primitive output; must contain ``"approach"``.
        q_warm_start: Optional IK warm-start for the approach.
        tracker: Metrics tracker to update.
        assignment: Assignment result, or ``None``.
        obj_name: Object name for grasp verification; empty skips the check.
    """
    logger.info("Robot %d following grasp approach trajectory...", current_robot)
    current_arm.follow_trajectory(grasp_traj_data["approach"], q_warm_start=q_warm_start,
                                  smoothen=True)

    current_gripper.grasp()
    stabilise(system.env, system.env_lock)

    # Verify the object is actually grasped
    if obj_name and not _object_held_by(system, current_robot, current_arm, obj_name, gripper=current_gripper):
        logger.error(
            "Robot %d FAILED to grasp object '%s' — grip not detected "
            "(object too far from EEF or weld inactive).",
            current_robot, obj_name,
        )
        tracker.grasp_success = False
    else:
        tracker.grasp_success = True

    tracker.grasp_result = {
        "robot": current_robot,
        "gripper": _ROBOT_GRIPPER.get(current_robot, "unknown"),
        "success": tracker.grasp_success,
    }

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
    """Return the next handover waypoint index, or -1 if none.

    In assignment mode this is the first waypoint after ``current_idx`` whose
    assigned robot differs from ``current_robot``. In explicit mode a
    heuristic picks the latest waypoint feasible for both robots before the
    current robot's first infeasible waypoint, or before its last feasible
    waypoint if the other robot stays feasible longer.

    Args:
        assignment: Assignment result, or ``None``.
        explicit_paths: Robot id to path mapping for explicit mode.
        traj_cache: Trajectory cache keyed by trajectory path.
        current_robot: Index of the robot holding the object.
        current_idx: Current waypoint index.
        res_cur: Feasibility results of the current robot's trajectory.

    Returns:
        Handover waypoint index, or -1.
    """
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
) -> Tuple[HandoverTrajectory, Optional[np.ndarray], Optional[np.ndarray]]:
    """Move arm A to the handover waypoint while arm B moves to its pre-pose.

    Plans the handover trajectory from the two grasps and the object pose at
    ``handover_idx``, builds arm A's joint segment (with an alignment waypoint
    if it is more than 0.05 rad from plan, 2π-realignment and quintic
    velocity profiling) and runs both arms with ``DualArmParallelPlanner``.
    Falls back to ``_recover_arm_b`` if arm B does not converge.

    Args:
        system: Running dual-arm system.
        current_arm: Transfer arm (A).
        other_arm: Receiver arm (B).
        current_robot: Index of the transfer robot.
        current_idx: Current waypoint index.
        handover_idx: Handover waypoint index.
        traj_cache: Trajectory cache keyed by trajectory path.
        grasp_cache: Grasp transforms keyed by trajectory path.
        assignment: Assignment result, or ``None``.
        explicit_paths: Robot id to path mapping for explicit mode.
        exp_cfg: Experiment configuration.
        tracker: Metrics tracker to update.
        rrt_vis: Optional RRT visualizer.
        vis: Optional Meshcat visualizer for the handover plan.

    Returns:
        Tuple ``(h_traj, q_warm_recv, q_warm_transfer)``: the handover
        trajectory and the planned receiver / transfer configs at the
        handover waypoint (either may be ``None``).

    Raises:
        ValueError: In assignment mode, if ``handover_idx`` is not in
            ``assignment.handover_points``.
    """
    cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
    next_path = _get_traj_path(assignment, explicit_paths, 1 - current_robot, handover_idx)
    _, obj_cur, _, _, _ = traj_cache[cur_path]
    joints_cur = _get_joint_configs(assignment, traj_cache, cur_path)
    # In assignment mode joints_next is the same array as joints_cur
    # (normalized_configs); in explicit mode it is the *receiver* robot's
    # raw trajectory and is needed for q_warm_recv.
    joints_next = _get_joint_configs(assignment, traj_cache, next_path)

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

    # Arm A's trajectory segment (prepend current position + planned config
    # at current_idx for smooth alignment before following the rest)
    actual_q = current_arm.get_joint_positions()
    planned_q = joints_cur[current_idx]
    joint_segment = [actual_q]
    if planned_q is not None:
        q_err = float(np.linalg.norm(np.asarray(actual_q) - np.asarray(planned_q)))
        if q_err > 0.05:
            logger.info(
                "Inserting alignment waypoint (q_err=%.4f rad) before "
                "trajectory segment %d -> %d.",
                q_err, current_idx, handover_idx,
            )
            joint_segment.append(planned_q)
    # Build the transfer arm's joint segment and compute q_warm_transfer.
    q_warm_transfer: Optional[np.ndarray] = None
    if assignment is not None:
        # normalized_configs at handover_idx belongs to the *receiving*
        # robot — exclude it.  Instead, append the transfer arm's own
        # planned config at the handover waypoint so the arm physically
        # arrives at the handover position during parallel execution.
        joint_segment.extend(joints_cur[current_idx + 1 : handover_idx])
        ho_list_idx = assignment.handover_points.index(handover_idx)
        ho_cfg = assignment.handover_transfer_configs[ho_list_idx]
        if not np.any(np.isnan(ho_cfg)):
            joint_segment.append(ho_cfg)
            q_warm_transfer = ho_cfg
    else:
        # Explicit-trajectory mode: the transfer arm's own trajectory
        # includes a valid config at handover_idx.
        joint_segment.extend(joints_cur[current_idx + 1 : handover_idx + 1])
        q_warm_transfer = (
            joints_cur[handover_idx]
            if joints_cur is not None else None
        )

    # Eliminate 2π reflex-angle jumps between actual position and
    # planned configs (can occur after IK-based handover motions).
    joint_segment = _realign_joint_segment(joint_segment)

    # Apply quintic velocity profiling so the arm accelerates and
    # decelerates smoothly instead of moving at uniform speed.
    n_profiled = max(200, len(joint_segment) * 8)
    joint_segment = velocity_profile_joint_trajectory(
        np.array(joint_segment), num_steps=n_profiled,
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
        q_warm_start_b=q_warm_recv,
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

    # --- Diagnostic: log receiver arm's actual vs planned config ---
    recv_actual = other_arm.get_joint_positions()
    if q_warm_recv is not None:
        _log_joint_delta(
            f"recv-post-parallel robot{1-current_robot} wp{handover_idx}",
            recv_actual, q_warm_recv,
        )
    # --- Diagnostic: log transfer arm's actual vs planned config ---
    xfer_actual = current_arm.get_joint_positions()
    if q_warm_transfer is not None:
        _log_joint_delta(
            f"xfer-post-parallel robot{current_robot} wp{handover_idx}",
            xfer_actual, q_warm_transfer,
        )

    return h_traj, q_warm_recv, q_warm_transfer


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
    """Attempt independent RRT recovery to get arm B to its target pose.

    Failure is only logged; no exception is raised.

    Args:
        system: Running dual-arm system.
        current_arm: Arm A (held fixed for collision checking).
        other_arm: Arm B to recover.
        current_robot: Index of arm A's robot.
        target_pose: Receiver pre-pose for arm B.
        q_warm_start: Optional IK warm-start for arm B.
        rrt_vis: Optional RRT visualizer.
        exp_cfg: Experiment configuration (for ``execution.smoothen``).
    """
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

    logger.warning("Recovery failed: arm B could not reach receiver pre-pose.")

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
    q_warm_transfer: Optional[np.ndarray],
    tracker: ExecutionTracker,
    obj_name: str = "",
) -> bool:
    """Execute the physical handover (hold -> approach -> gripper swap -> retreat).

    Increments ``tracker.n_handovers_attempted`` and appends a record to
    ``tracker.handover_results``. With ``obj_name``, success means the
    receiver holds the object and it has not fallen; the handover is aborted
    (and ``tracker.transport_success`` set ``False``) if the transfer arm is
    not holding the object beforehand. Without ``obj_name``, success is the
    receiver approach converging.

    Args:
        system: Running dual-arm system.
        current_arm: Transfer arm.
        current_gripper: Transfer gripper.
        other_arm: Receiver arm.
        other_gripper: Receiver gripper.
        current_robot: Index of the transfer robot.
        current_idx: Handover waypoint index.
        h_traj: Planned handover trajectory.
        q_warm_recv: Optional IK warm-start for the receiver.
        q_warm_transfer: Optional IK warm-start for the transfer arm.
        tracker: Metrics tracker to update.
        obj_name: Object name for verification; empty disables it.

    Returns:
        ``True`` if the handover succeeded.
    """
    tracker.n_handovers_attempted += 1
    other_robot = 1 - current_robot
    giver_gripper = _ROBOT_GRIPPER.get(current_robot, "unknown")
    receiver_gripper = _ROBOT_GRIPPER.get(other_robot, "unknown")
    direction = f"{giver_gripper}\u2192{receiver_gripper}"
    logger.info(
        "Executing handover at waypoint %d (robot %d -> robot %d)...",
        current_idx, current_robot, other_robot,
    )

    # Check object is still held by the transfer arm before starting
    if obj_name:
        if not _object_held_by(system, current_robot, current_arm, obj_name, gripper=current_gripper):
            logger.error(
                "Handover ABORTED at waypoint %d — robot %d is NOT holding "
                "object '%s' (lost during transport).",
                current_idx, current_robot, obj_name,
            )
            tracker.transport_success = False
            tracker.handover_results.append({
                "waypoint": current_idx,
                "giver": current_robot,
                "receiver": other_robot,
                "giver_gripper": giver_gripper,
                "receiver_gripper": receiver_gripper,
                "direction": direction,
                "success": False,
                "reason": "object_lost_before_handover",
            })
            return False

    # Transfer arm holds position
    logger.info("Robot %d moving to handover hold pose...", current_robot)
    current_arm.move_to_pose(h_traj.transfer_hold[0], gripper_action=1.0, num_steps=100,
                             q_warm_start=q_warm_transfer)

    # Receiver approaches
    logger.info("Robot %d executing handover approach...", other_robot)
    other_gripper.release()
    receiver_converged = other_arm.follow_trajectory(
        h_traj.receiver_approach, q_warm_start=q_warm_recv,
        smoothen=True,
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
    current_arm.follow_trajectory(h_traj.transfer_retreat, q_warm_start=q_warm_transfer,
                                  smoothen=True)

    # --- Object-aware handover verification ---
    handover_ok = False
    if obj_name:
        # Primary check: receiver holds the object (suction weld or gripper contact)
        receiver_holds = _object_held_by(system, other_robot, other_arm, obj_name, gripper=other_gripper)
        obj_safe = _object_not_fallen(system, obj_name)

        if receiver_holds and obj_safe:
            handover_ok = True
        elif not obj_safe:
            logger.error(
                "Handover at waypoint %d FAILED — object '%s' has fallen.",
                current_idx, obj_name,
            )
        else:
            logger.error(
                "Handover at waypoint %d FAILED — receiver robot %d does NOT "
                "hold object '%s' after gripper swap.",
                current_idx, other_robot, obj_name,
            )
    else:
        # Fallback: use arm convergence (original behaviour)
        handover_ok = receiver_converged

    # Build per-handover record
    ho_record: Dict[str, Any] = {
        "waypoint": current_idx,
        "giver": current_robot,
        "receiver": other_robot,
        "giver_gripper": giver_gripper,
        "receiver_gripper": receiver_gripper,
        "direction": direction,
        "success": handover_ok,
    }
    if obj_name:
        ho_record["receiver_holds_post"] = receiver_holds
        ho_record["object_fallen"] = not obj_safe
        if not handover_ok:
            ho_record["reason"] = "object_fallen" if not obj_safe else "receiver_no_hold"
    tracker.handover_results.append(ho_record)

    if handover_ok:
        tracker.n_handovers_succeeded += 1
        logger.info(
            "Handover at waypoint %d SUCCEEDED (robot %d -> robot %d). "
            "Total handovers: %d/%d",
            current_idx, current_robot, other_robot,
            tracker.n_handovers_succeeded, tracker.n_handovers_attempted,
        )
    else:
        logger.warning(
            "Handover at waypoint %d FAILED (robot %d -> robot %d). "
            "Total handovers: %d/%d",
            current_idx, current_robot, other_robot,
            tracker.n_handovers_succeeded, tracker.n_handovers_attempted,
        )
    return handover_ok


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

    Follows the current arm's configs up to the last feasible waypoint
    (clamped to the assignment length) and sets
    ``tracker.transport_success = True`` regardless of arm convergence.

    Args:
        system: Running dual-arm system.
        current_arm: Arm holding the object.
        other_arm: Idle arm sent to ``FRANKA_HOME_JOINT_POSITION``.
        current_robot: Index of the holding robot.
        current_idx: Current waypoint index.
        joints_cur: Planned joint configs, shape ``(N, 7)``.
        res_cur: Feasibility results of the current trajectory.
        assignment: Assignment result, or ``None``.
        exp_cfg: Experiment configuration.
        tracker: Metrics tracker to update.
        rrt_vis: Optional RRT visualizer.

    Returns:
        The final waypoint index (``current_idx`` unchanged if there is no
        later feasible waypoint).
    """
    last_idx = find_last_feasible_point(res_cur)
    if assignment is not None and last_idx is not None:
        last_idx = min(last_idx, len(assignment.sequence) - 1)

    if last_idx is None or last_idx <= current_idx:
        tracker.transport_success = True
        return current_idx

    actual_q = current_arm.get_joint_positions()
    planned_q = joints_cur[current_idx]
    joint_segment = [actual_q]
    if planned_q is not None:
        q_err = float(np.linalg.norm(np.asarray(actual_q) - np.asarray(planned_q)))
        if q_err > 0.05:
            logger.info(
                "Inserting alignment waypoint (q_err=%.4f rad) before "
                "final segment %d -> %d.",
                q_err, current_idx, last_idx,
            )
            joint_segment.append(planned_q)
    joint_segment.extend(joints_cur[current_idx + 1 : last_idx + 1])
    # Eliminate 2π reflex-angle jumps (same rationale as handover path).
    joint_segment = _realign_joint_segment(joint_segment)

    # Apply quintic velocity profiling for smooth acceleration/deceleration.
    n_profiled = max(200, len(joint_segment) * 8)
    joint_segment = velocity_profile_joint_trajectory(
        np.array(joint_segment), num_steps=n_profiled,
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
    """Stop the collision and object monitors and record their results.

    Args:
        tracker: Metrics tracker; collision count and object tracking errors
            are written to it.
    """
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
    """Release the object and retreat the active arm.

    Args:
        system: Running dual-arm system.
        current_arm: Arm holding the object.
        current_gripper: Gripper of that arm.
        current_robot: Index of the holding robot.
        eef_cur: EEF trajectory of the current robot.
        current_idx: Waypoint whose EEF pose is the release pose.
    """
    logger.info("Robot %d releasing object and retreating...", current_robot)
    release_pose = eef_cur[current_idx]
    release_primitive = ReleaseRetreatPrimitive()
    release_traj_data = release_primitive.generate_trajectory(release_pose)

    current_gripper.release()
    stabilise(system.env, system.env_lock)

    logger.info("Robot %d following retreat trajectory...", current_robot)
    current_arm.follow_trajectory(release_traj_data["retreat"], gripper_action=-1.0,
                                  smoothen=True)
    stabilise(system.env, system.env_lock)


def verify_placement(
    system: DualArmManipulationSystem,
    exp_cfg: ExperimentConfig,
    tracker: ExecutionTracker,
) -> None:
    """Check whether the object reached its target pose.

    Success requires position error below 0.05 m and orientation error below
    30 deg. If the object pose cannot be queried, ``place_success`` is set
    ``True`` without checking.

    Args:
        system: Running dual-arm system.
        exp_cfg: Experiment configuration with the target pose.
        tracker: Metrics tracker; placement errors and ``place_success`` are
            written to it.
    """
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
    """Drive the full pick -> transport -> handover(s) -> place pipeline.

    Sets ``tracker.task_success`` from the grasp, transport, placement and
    handover outcomes and records the execution time.

    Args:
        system: Running dual-arm system.
        exp_cfg: Experiment configuration.
        assignment: Assignment result, or ``None`` in explicit mode.
        traj_cache: Trajectory cache keyed by trajectory path.
        grasp_cache: Grasp transforms keyed by trajectory path.
        explicit_paths: Robot id to path mapping for explicit mode.
        tracker: Metrics tracker to update.
        rrt_vis: Optional RRT visualizer.
        vis: Optional Meshcat visualizer.
        args: Parsed CLI arguments (``headless`` skips the final 5 s pause).

    Raises:
        RuntimeError: In explicit mode if neither trajectory has a feasible
            waypoint, or if pre-grasp planning fails.
    """

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
    eef_cur, obj_cur, _, res_cur, _ = traj_cache[cur_path]
    joints_cur = _get_joint_configs(assignment, traj_cache, cur_path)

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
        obj_name=exp_cfg.task.manipulated_object,
    )

    start_collision_monitor(exp_cfg.task.manipulated_object)
    start_object_monitor(exp_cfg.task.manipulated_object, planned_positions=planned_obj_positions)

    # --- Phase 2: transport loop (handovers + parallel segments) ---
    n_segments = len(res_cur)
    if assignment is not None:
        n_segments = min(n_segments, len(assignment.sequence))

    while current_idx < n_segments - 1:
        cur_path = _get_traj_path(assignment, explicit_paths, current_robot, current_idx)
        eef_cur, obj_cur, _, res_cur, _ = traj_cache[cur_path]
        joints_cur = _get_joint_configs(assignment, traj_cache, cur_path)

        handover_idx = find_next_handover(
            assignment, explicit_paths, traj_cache,
            current_robot, current_idx, res_cur,
        )

        if handover_idx != -1 and handover_idx > current_idx:
            # Parallel move to handover point (also returns the
            # transfer arm's warm-start config at the handover waypoint).
            h_traj, q_warm_recv, q_warm_transfer = execute_parallel_to_handover(
                system, current_arm, other_arm, current_robot, current_idx,
                handover_idx, traj_cache, grasp_cache, assignment,
                explicit_paths, exp_cfg, tracker, rrt_vis, vis,
            )
            current_idx = handover_idx

            execute_handover_sequence(
                system, current_arm, current_gripper,
                other_arm, other_gripper,
                current_robot, current_idx,
                h_traj, q_warm_recv, q_warm_transfer, tracker,
                obj_name=exp_cfg.task.manipulated_object,
            )

            # --- Diagnostic: log both arms after handover ---
            recv_actual_post = other_arm.get_joint_positions()
            xfer_actual_post = current_arm.get_joint_positions()
            if q_warm_recv is not None:
                _log_joint_delta(
                    f"recv-post-handover robot{1-current_robot} wp{current_idx}",
                    recv_actual_post, q_warm_recv,
                )
            if q_warm_transfer is not None:
                _log_joint_delta(
                    f"xfer-post-handover robot{current_robot} wp{current_idx}",
                    xfer_actual_post, q_warm_transfer,
                )

            # Swap roles
            current_robot = 1 - current_robot
            current_arm, current_gripper, other_arm, other_gripper = _get_arms(system, current_robot)

            # --- Fix: correct 2π joint offsets on new current arm ---
            # After handover the receiver (now current_arm) may have
            # ended up at a 2π-offset solution from IK/mink.  If so,
            # normalize and physically drive to the expected config.
            if q_warm_recv is not None:
                actual_q = current_arm.get_joint_positions()
                corrected = _normalize_config_to_reference(actual_q, q_warm_recv)
                corr_err = float(np.max(np.abs(corrected - actual_q)))
                if corr_err > 0.1:
                    logger.info(
                        "Correcting 2π joint offset on robot %d after "
                        "handover (max correction=%.4f rad).",
                        current_robot, corr_err,
                    )
                    current_arm.follow_joint_trajectory(
                        [corrected], gripper_action=1.0,
                        num_steps=100, smoothen=True,
                    )
                    _log_joint_delta(
                        f"post-correction robot{current_robot} wp{current_idx}",
                        current_arm.get_joint_positions(), q_warm_recv,
                    )
        else:
            # No more handovers -- execute final segment
            current_idx = execute_final_segment(
                system, current_arm, other_arm, current_robot,
                current_idx, joints_cur, res_cur, assignment,
                exp_cfg, tracker, rrt_vis,
            )
            break

    # --- Phase 3: stop monitors ---
    stop_monitors(tracker)

    # --- Phase 3b: verify object still held before release ---
    obj_name = exp_cfg.task.manipulated_object
    if obj_name:
        still_held = _object_held_by(system, current_robot, current_arm, obj_name, gripper=current_gripper)
        obj_safe = _object_not_fallen(system, obj_name)
        if not still_held or not obj_safe:
            logger.error(
                "Transport FAILED — robot %d lost object '%s' before "
                "reaching placement pose.",
                current_robot, obj_name,
            )
            tracker.transport_success = False

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
        and tracker.transport_success
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
    """Load the experiment and trajectories, execute them and save metrics.

    Exceptions raised during loading or execution are logged and stored in
    the tracker's ``error_message``; metrics are written to
    ``--output-metrics`` (if given) and the system is stopped in all cases.
    """
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
