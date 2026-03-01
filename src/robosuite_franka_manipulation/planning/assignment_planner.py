"""
Optimal trajectory segment assignment planner using dynamic programming.

Finds the minimum-cost assignment of robot arms to object-trajectory segments,
where each segment ``[p_i, p_{i+1}]`` can be carried by either robot with
an associated task cost and handover switching cost.

Cost model
----------
.. math::
    C(A_{i,j}) = \\alpha \\, C_1(A_{i,j}) + \\beta \\, C_2(A_{i,j})

where:

* :math:`C_1` — **smoothness**: :math:`\\|q_{i+1} - q_i\\|^2`
* :math:`C_2` — **manipulability**: :math:`\\sum_k 1 / (w(q_k) + \\varepsilon)`
* :math:`\\alpha, \\beta` — user-specified weights
* infeasible segments receive :math:`\\infty`

The solver uses a Viterbi-style DP to find the globally optimal sequence
``(robot_id, trajectory_idx)`` for every segment, subject to the constraint
that the same arm must keep using the same grasp trajectory (no intra-arm
trajectory switching).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.utils.manipulability import (
    compute_trajectory_manipulability,
)
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation.utils.primitives import DualArmHandoverPlanner

from robosuite_franka_manipulation.environments import DualArmEnvironment
from robosuite_franka_manipulation.transforms import TransformManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AssignmentResult:
    """Outcome of :func:`compute_optimal_robot_assignment`."""

    sequence: List[int]               # robot id (0 or 1) per segment
    trajectory_indices: List[int]     # grasp-trajectory index per segment
    total_cost: float
    task_costs: np.ndarray            # per-segment task cost
    switching_costs: np.ndarray       # per-segment switching cost
    handover_points: List[int]        # segment indices where handovers occur
    cost_matrix: np.ndarray           # (n_segments, 2) – best-per-robot summary
    trajectory_matrix: np.ndarray     # (n_segments, 2) – best trajectory idx per robot


# ---------------------------------------------------------------------------
# Segment cost
# ---------------------------------------------------------------------------

def compute_segment_cost(
    trajectory: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    jacobian_fn: Optional[Callable] = None,
) -> Tuple[float, bool]:
    """Compute the task cost for one trajectory segment ``[start_idx, end_idx]``.

    Args:
        trajectory: Dict with ``'is_feasible'`` (bool array) and
            ``'joint_configs'`` (2-D float array).
        start_idx: First waypoint index.
        end_idx: Last waypoint index (inclusive).
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        jacobian_fn: Optional ``(q) -> J`` for Yoshikawa manipulability.

    Returns:
        ``(cost, is_feasible)`` — cost is ``inf`` when infeasible.
    """
    n_waypoints = len(trajectory["is_feasible"])
    if start_idx < 0 or end_idx >= n_waypoints or start_idx >= end_idx:
        return np.inf, False

    segment_feasible = trajectory["is_feasible"][start_idx : end_idx + 1]
    segment_joints = trajectory["joint_configs"][start_idx : end_idx + 1]

    if len(segment_feasible) == 0 or not np.all(segment_feasible):
        return np.inf, False
    if len(segment_joints) == 0:
        return np.inf, False

    if segment_joints.ndim == 1:
        segment_joints = segment_joints.reshape(1, -1)

    # C1: smoothness
    smoothness = 0.0
    if segment_joints.shape[0] >= 2:
        smoothness = float(np.linalg.norm(segment_joints[-1] - segment_joints[0]) ** 2)

    # C2: manipulability
    manip_vals = compute_trajectory_manipulability(
        segment_joints,
        use_jacobian=(jacobian_fn is not None),
        jacobian_fn=jacobian_fn,
    )
    manip_cost = float(np.sum(1.0 / (manip_vals + epsilon)))

    return alpha * smoothness + beta * manip_cost, True


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------

def compute_cost_matrix(
    robot0_trajectories: List[Dict[str, Any]],
    robot1_trajectories: List[Dict[str, Any]],
    n_segments: int,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """Build the per-segment cost for every ``(robot, trajectory)`` state.

    Args:
        jacobian_fns: Optional mapping ``{robot_id: (q) -> J}`` providing
            per-robot Jacobian functions for Yoshikawa manipulability.

    Returns:
        ``(cost_matrix, trajectory_info)``

        * ``cost_matrix[seg, state_idx]`` — task cost.
        * ``trajectory_info[state_idx]`` → ``(robot_id, traj_idx)``.
    """
    trajectory_info: Dict[int, Tuple[int, int]] = {}
    state_idx = 0
    for traj_idx in range(len(robot0_trajectories)):
        trajectory_info[state_idx] = (0, traj_idx)
        state_idx += 1
    for traj_idx in range(len(robot1_trajectories)):
        trajectory_info[state_idx] = (1, traj_idx)
        state_idx += 1

    n_states = state_idx
    cost_matrix = np.full((n_segments, n_states), np.inf)

    for si, (robot_id, traj_idx) in trajectory_info.items():
        traj = robot0_trajectories[traj_idx] if robot_id == 0 else robot1_trajectories[traj_idx]
        jac_fn = jacobian_fns.get(robot_id) if jacobian_fns else None
        n_wp = len(traj["is_feasible"])
        for seg in range(min(n_segments, n_wp - 1)):
            cost, ok = compute_segment_cost(
                traj, seg, seg + 1,
                alpha=alpha, beta=beta, epsilon=epsilon, jacobian_fn=jac_fn,
            )
            if ok:
                cost_matrix[seg, si] = cost

    return cost_matrix, trajectory_info


# ---------------------------------------------------------------------------
# Handover primitive cost
# ---------------------------------------------------------------------------

def _compute_handover_primitive_cost(
    handover_trajectory: Dict[str, List[StampedPose]],
    alpha: float,
    beta: float,
    epsilon: float,
    ik_solver_fn: Callable,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> Tuple[float, bool]:
    """Evaluate cost of a handover primitive using the same C1 + C2 model.

    Args:
        jacobian_fns: Optional ``{robot_id: (q) -> J}`` for per-arm
            Yoshikawa manipulability.
    """
    transfer_poses: List[StampedPose] = []
    receiver_poses: List[StampedPose] = []

    for key in ("transfer_hold", "transfer_retreat"):
        transfer_poses.extend(handover_trajectory.get(key, []))
    for key in ("receiver_pre", "receiver_approach"):
        receiver_poses.extend(handover_trajectory.get(key, []))

    if not transfer_poses and not receiver_poses:
        return np.inf, False

    # Solve IK for each phase
    def _solve_all(poses: List[StampedPose], robot_id: int):
        configs = []
        for p in poses:
            try:
                q = ik_solver_fn(p, robot_id)
                if q is None or len(q) == 0:
                    return None
                configs.append(q)
            except Exception:
                return None
        return configs

    transfer_configs = _solve_all(transfer_poses, 0) if transfer_poses else []
    receiver_configs = _solve_all(receiver_poses, 1) if receiver_poses else []

    if transfer_configs is None or receiver_configs is None:
        return np.inf, False

    # Smoothness per arm
    smoothness = 0.0
    for cfgs in (transfer_configs, receiver_configs):
        if cfgs and len(cfgs) >= 2:
            arr = np.array(cfgs)
            for i in range(arr.shape[0] - 1):
                smoothness += float(np.linalg.norm(arr[i + 1] - arr[i]) ** 2)

    # Manipulability — compute per-arm with the correct Jacobian function
    manip_cost = 0.0
    for robot_id, cfgs in ((0, transfer_configs), (1, receiver_configs)):
        if not cfgs:
            continue
        arr = np.array(cfgs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        jac_fn = jacobian_fns.get(robot_id) if jacobian_fns else None
        manip_vals = compute_trajectory_manipulability(
            arr,
            use_jacobian=(jac_fn is not None),
            jacobian_fn=jac_fn,
        )
        manip_cost += float(np.sum(1.0 / (manip_vals + epsilon)))

    return alpha * smoothness + beta * manip_cost, True


# ---------------------------------------------------------------------------
# Switching cost
# ---------------------------------------------------------------------------

def compute_switching_cost(
    prev_robot: int,
    curr_robot: int,
    segment_idx: int,
    trajectories: Dict[str, Any],
    prev_state_idx: int,
    curr_state_idx: int,
    trajectory_info: Dict[int, Tuple[int, int]],
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> float:
    """Compute the cost of switching from one arm to another at *segment_idx*.

    Returns 0 when `prev_robot == curr_robot`, ``inf`` when the transfer is
    blocked by *transfer_feasibility*, or the handover primitive cost otherwise.
    """
    if prev_robot == curr_robot:
        return 0.0

    prev_robot_id, prev_traj_idx = trajectory_info[prev_state_idx]
    curr_robot_id, curr_traj_idx = trajectory_info[curr_state_idx]

    # Check transfer feasibility constraints (bidirectional)
    if transfer_feasibility is not None:
        for key in (
            (prev_robot_id, prev_traj_idx, curr_robot_id, curr_traj_idx),
            (curr_robot_id, curr_traj_idx, prev_robot_id, prev_traj_idx),
        ):
            if key in transfer_feasibility and not transfer_feasibility[key]:
                logger.debug("Transfer blocked: %s", key)
                return np.inf

    if ik_solver_fn is None:
        return constants.ASSIGNMENT_FALLBACK_SWITCH_COST

    # Build the handover primitive and evaluate its cost
    obj_traj = trajectories["object_trajectory"]
    if segment_idx >= len(obj_traj["positions"]):
        return np.inf

    obj_pose = StampedPose(
        position=obj_traj["positions"][segment_idx],
        quaternion=obj_traj["quaternions"][segment_idx],
        frame_id=_frame_id_at(obj_traj, segment_idx),
    )

    prev_traj = _get_trajectory(trajectories, prev_robot_id, prev_traj_idx)
    curr_traj = _get_trajectory(trajectories, curr_robot_id, curr_traj_idx)

    if segment_idx >= len(prev_traj.get("eef_positions", [])):
        return np.inf
    if segment_idx >= len(curr_traj.get("eef_positions", [])):
        return np.inf

    # Compute grasp transforms: G = T_obj⁻¹ @ T_ee
    T_obj = obj_pose.to_matrix()
    T_obj_inv = np.linalg.inv(T_obj)

    grasp_transfer = StampedPose.from_matrix(
        T_obj_inv @ _eef_matrix(prev_traj, segment_idx), "object"
    )
    grasp_receiver = StampedPose.from_matrix(
        T_obj_inv @ _eef_matrix(curr_traj, segment_idx), "object"
    )

    h_planner = DualArmHandoverPlanner(
        grasp_transfer=grasp_transfer,
        grasp_receiver=grasp_receiver,
        handover_object_pose=obj_pose,
    )
    h_traj = h_planner.generate_trajectory()

    cost, ok = _compute_handover_primitive_cost(
        h_traj, alpha, beta, epsilon, ik_solver_fn, jacobian_fns,
    )
    return cost if ok else np.inf


# ---------------------------------------------------------------------------
# Viterbi DP solver
# ---------------------------------------------------------------------------

def _solve_optimal_assignment(
    cost_matrix: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    trajectories: Dict[str, Any],
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
    initial_robot: Optional[int] = None,
) -> AssignmentResult:
    """Solve the optimal assignment with a Viterbi-style DP.

    Enforces that within the same arm the trajectory index must stay constant
    (only arm-switches may change trajectory).
    """
    n_segments, n_states = cost_matrix.shape

    J_star = np.full((n_segments, n_states), np.inf)
    prev_state = np.full((n_segments, n_states), -1, dtype=int)

    # Initialise first segment
    for si in range(n_states):
        robot_id, _ = trajectory_info[si]
        if initial_robot is not None and robot_id != initial_robot:
            continue
        J_star[0, si] = cost_matrix[0, si]

    # Forward pass
    for i in range(1, n_segments):
        for cs in range(n_states):
            cr, ct = trajectory_info[cs]
            best_cost = np.inf
            best_prev = -1

            for ps in range(n_states):
                if np.isinf(J_star[i - 1, ps]):
                    continue
                pr, pt = trajectory_info[ps]

                # Same arm must keep same trajectory
                if pr == cr and pt != ct:
                    continue

                sw = compute_switching_cost(
                    pr, cr, i, trajectories, ps, cs, trajectory_info,
                    transfer_feasibility=transfer_feasibility,
                    alpha=alpha, beta=beta, epsilon=epsilon,
                    ik_solver_fn=ik_solver_fn, jacobian_fns=jacobian_fns,
                )
                if np.isinf(sw):
                    continue

                total = J_star[i - 1, ps] + sw
                if total < best_cost:
                    best_cost = total
                    best_prev = ps

            if best_prev != -1:
                J_star[i, cs] = best_cost + cost_matrix[i, cs]
                prev_state[i, cs] = best_prev

    # Find best terminal state
    best_final_cost = np.inf
    best_final_state = -1
    for si in range(n_states):
        if J_star[n_segments - 1, si] < best_final_cost:
            best_final_cost = J_star[n_segments - 1, si]
            best_final_state = si

    if best_final_state == -1:
        raise ValueError("No feasible assignment found — check trajectory feasibility.")

    # Backtrack
    sequence: List[int] = []
    traj_indices: List[int] = []
    task_costs_list: List[float] = []
    switch_costs_list: List[float] = []

    cs = best_final_state
    for i in range(n_segments - 1, -1, -1):
        robot_id, traj_idx = trajectory_info[cs]
        sequence.insert(0, robot_id)
        traj_indices.insert(0, traj_idx)
        task_costs_list.insert(0, float(cost_matrix[i, cs]))

        if i > 0:
            ps = prev_state[i, cs]
            pr, _ = trajectory_info[ps]
            sw = compute_switching_cost(
                pr, robot_id, i, trajectories, ps, cs, trajectory_info,
                transfer_feasibility=transfer_feasibility,
                alpha=alpha, beta=beta, epsilon=epsilon,
                ik_solver_fn=ik_solver_fn, jacobian_fns=jacobian_fns,
            )
            switch_costs_list.insert(0, sw)
            cs = ps
        else:
            switch_costs_list.insert(0, 0.0)

    # Handover points
    handover_points = [i for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]]

    # Summary matrices (best cost per robot per segment)
    cost_summary = np.full((n_segments, 2), np.inf)
    traj_summary = np.full((n_segments, 2), -1, dtype=int)
    for seg in range(n_segments):
        for si, (rid, tidx) in trajectory_info.items():
            if cost_matrix[seg, si] < cost_summary[seg, rid]:
                cost_summary[seg, rid] = cost_matrix[seg, si]
                traj_summary[seg, rid] = tidx

    return AssignmentResult(
        sequence=sequence,
        trajectory_indices=traj_indices,
        total_cost=best_final_cost,
        task_costs=np.array(task_costs_list),
        switching_costs=np.array(switch_costs_list),
        handover_points=handover_points,
        cost_matrix=cost_summary,
        trajectory_matrix=traj_summary,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_optimal_robot_assignment(
    trajectories: Dict[str, Any],
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    initial_robot: Optional[int] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> AssignmentResult:
    """Compute the globally optimal robot-to-segment assignment.

    Args:
        trajectories: Dict with keys ``'robot0_trajectories'``,
            ``'robot1_trajectories'``, and ``'object_trajectory'``.
        transfer_feasibility: Optional ``{(from_robot, from_traj, to_robot,
            to_traj): bool}`` — *False* blocks that handover.
        initial_robot: Force the first segment to a specific robot.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        ik_solver_fn: ``(pose, robot_id) -> q`` for handover cost evaluation.
        jacobian_fns: ``{robot_id: (q) -> J}`` per-robot Jacobian functions
            for Yoshikawa manipulability.

    Returns:
        :class:`AssignmentResult` with the optimal sequence and cost breakdown.

    Raises:
        ValueError: If no feasible assignment exists.
    """
    r0 = trajectories["robot0_trajectories"]
    r1 = trajectories["robot1_trajectories"]
    n_wp = len(trajectories["object_trajectory"]["positions"])
    n_segments = n_wp - 1

    logger.info(
        "Computing assignment for %d segments (%d R0 + %d R1 trajectories, "
        "α=%.2f, β=%.2f)",
        n_segments, len(r0), len(r1), alpha, beta,
    )

    cost_matrix, traj_info = compute_cost_matrix(
        r0, r1, n_segments,
        alpha=alpha, beta=beta, epsilon=epsilon, jacobian_fns=jacobian_fns,
    )
    logger.info("Cost matrix: %d segments × %d states", *cost_matrix.shape)

    result = _solve_optimal_assignment(
        cost_matrix, traj_info, trajectories,
        transfer_feasibility=transfer_feasibility,
        alpha=alpha, beta=beta, epsilon=epsilon,
        ik_solver_fn=ik_solver_fn, jacobian_fns=jacobian_fns,
        initial_robot=initial_robot,
    )

    logger.info(
        "Assignment solved: total_cost=%.4f, handovers=%d",
        result.total_cost, len(result.handover_points),
    )
    return result


# ---------------------------------------------------------------------------
# Factory helpers — create closures around the sim environment
# ---------------------------------------------------------------------------

def make_ik_solver_fn(
    env: DualArmEnvironment,
    tf_manager: TransformManager,
    env_lock: Optional[threading.RLock] = None,
) -> Callable[[StampedPose, int], Optional[np.ndarray]]:
    """Create an IK-solver closure compatible with :func:`compute_optimal_robot_assignment`.

    The returned function has signature ``(pose: StampedPose, robot_id: int) -> Optional[np.ndarray]``
    and internally delegates to :func:`~planning.utils.ik_utils.enumerate_ik_solutions_for_arm`,
    applying the orientation correction beforehand.

    Args:
        env: The dual-arm simulation environment.
        tf_manager: Transform manager for frame conversions.
        env_lock: Optional reentrant lock guarding the environment.
    """
    from robosuite_franka_manipulation.planning.utils.ik_utils import (
        enumerate_ik_solutions_for_arm,
        apply_ik_orientation_correction,
    )

    def _ik_solver(pose: StampedPose, robot_id: int) -> Optional[np.ndarray]:
        corrected = apply_ik_orientation_correction(pose)
        solutions = enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=corrected,
            tf_manager=tf_manager,
            env_lock=env_lock,
            max_solutions=1,
        )
        return solutions[0] if solutions else None

    return _ik_solver


def make_jacobian_fns(
    env: DualArmEnvironment,
    env_lock: Optional[threading.RLock] = None,
) -> Dict[int, Callable[[np.ndarray], np.ndarray]]:
    """Create per-robot Jacobian closures for :func:`compute_optimal_robot_assignment`.

    Returns a dict ``{0: jac_fn_robot0, 1: jac_fn_robot1}`` where each function
    has signature ``(q: np.ndarray) -> np.ndarray`` (6×7 Jacobian) and internally
    delegates to :func:`~planning.utils.ik_utils.compute_arm_jacobian`.

    Args:
        env: The dual-arm simulation environment.
        env_lock: Optional reentrant lock guarding the environment.
    """
    from robosuite_franka_manipulation.planning.utils.ik_utils import compute_arm_jacobian

    def _make_for_robot(robot_id: int) -> Callable[[np.ndarray], np.ndarray]:
        def _jacobian_fn(q: np.ndarray) -> np.ndarray:
            return compute_arm_jacobian(
                env, robot_id=robot_id, q_config=q, env_lock=env_lock,
            )
        return _jacobian_fn

    return {0: _make_for_robot(0), 1: _make_for_robot(1)}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_assignment_result(result: AssignmentResult, filepath: str) -> None:
    """Persist an :class:`AssignmentResult` to an ``.npz`` file."""
    np.savez(
        filepath,
        sequence=np.array(result.sequence),
        trajectory_indices=np.array(result.trajectory_indices),
        total_cost=result.total_cost,
        task_costs=result.task_costs,
        switching_costs=result.switching_costs,
        handover_points=np.array(result.handover_points),
        cost_matrix=result.cost_matrix,
        trajectory_matrix=result.trajectory_matrix,
    )
    logger.info("Assignment result saved to %s", filepath)


def load_assignment_result(filepath: str) -> AssignmentResult:
    """Load an :class:`AssignmentResult` from an ``.npz`` file."""
    data = np.load(filepath, allow_pickle=True)
    return AssignmentResult(
        sequence=data["sequence"].tolist(),
        trajectory_indices=data["trajectory_indices"].tolist(),
        total_cost=float(data["total_cost"]),
        task_costs=data["task_costs"],
        switching_costs=data["switching_costs"],
        handover_points=data["handover_points"].tolist(),
        cost_matrix=data["cost_matrix"],
        trajectory_matrix=data["trajectory_matrix"],
    )


# ---------------------------------------------------------------------------
# Trajectory loading helper  (matches save_trajectory / load_trajectory format)
# ---------------------------------------------------------------------------

def load_all_trajectories(data_dir: str) -> Dict[str, Any]:
    """Load all trajectory NPZ files produced by trajectory-generation scripts.

    Expects the directory to contain:
    * ``object_trajectory.npz``
    * ``robot0_trajectory_*.npz``
    * ``robot1_trajectory_*.npz``

    Returns a dict compatible with :func:`compute_optimal_robot_assignment`.
    """
    import os
    from pathlib import Path

    data_path = Path(data_dir)

    # Object trajectory
    obj_file = data_path / "object_trajectory.npz"
    if not obj_file.exists():
        raise FileNotFoundError(f"Object trajectory not found: {obj_file}")
    obj_data = np.load(obj_file)
    object_trajectory = {
        "positions": obj_data["object_positions"],
        "quaternions": obj_data["object_quaternions"],
        "frame_ids": obj_data["object_frame_ids"],
    }
    logger.info("Loaded object trajectory: %d waypoints", len(object_trajectory["positions"]))

    def _load_robot_trajectories(prefix: str) -> List[Dict[str, Any]]:
        files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"))
        trajs: List[Dict[str, Any]] = []
        for f in files:
            d = np.load(f)
            trajs.append({
                "eef_positions": d["eef_positions"],
                "eef_quaternions": d["eef_quaternions"],
                "eef_frame_ids": d["eef_frame_ids"],
                "object_positions": d["object_positions"],
                "object_quaternions": d["object_quaternions"],
                "object_frame_ids": d["object_frame_ids"],
                "is_feasible": d["is_feasible"],
                "ik_success": d["ik_success"],
                "collision_free": d["collision_free"],
                "joint_configs": d["joint_configs"],
            })
        logger.info("Loaded %d %s trajectories", len(trajs), prefix)
        return trajs

    return {
        "object_trajectory": object_trajectory,
        "robot0_trajectories": _load_robot_trajectories("robot0"),
        "robot1_trajectories": _load_robot_trajectories("robot1"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_id_at(obj_traj: Dict[str, Any], idx: int) -> str:
    """Extract frame_id from object trajectory at *idx*."""
    fids = obj_traj.get("frame_ids")
    if fids is None:
        return constants.WORLD_FRAME
    if isinstance(fids, np.ndarray):
        return str(fids[idx]) if idx < len(fids) else constants.WORLD_FRAME
    return str(fids)


def _get_trajectory(
    trajectories: Dict[str, Any], robot_id: int, traj_idx: int
) -> Dict[str, Any]:
    key = "robot0_trajectories" if robot_id == 0 else "robot1_trajectories"
    return trajectories[key][traj_idx]


def _eef_matrix(traj: Dict[str, Any], idx: int) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from the EEF pose at *idx*."""
    from scipy.spatial.transform import Rotation as R

    mat = np.eye(4)
    mat[:3, 3] = traj["eef_positions"][idx]
    mat[:3, :3] = R.from_quat(traj["eef_quaternions"][idx]).as_matrix()
    return mat
