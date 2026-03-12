"""Cost computation for the segment-assignment planner.

Implements the two-term cost model:

.. math::
    C(A_{i,j}) = \\alpha \\, C_1(A_{i,j}) + \\beta \\, C_2(A_{i,j})

where :math:`C_1` is a smoothness term (squared joint displacement) and
:math:`C_2` is a manipulability reciprocal term.

Public functions
----------------
compute_segment_cost
    Cost of a single trajectory segment.
compute_cost_matrix
    Per-segment cost for every ``(robot, trajectory)`` state.
compute_switching_cost
    Cost of switching arms at a given segment boundary.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)
from bimanual_manipulation.planning.utils.manipulability import (
    compute_trajectory_manipulability,
)
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation.utils.primitives import DualArmHandoverPlanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment cost
# ---------------------------------------------------------------------------

def compute_segment_cost(
    trajectory: RobotTrajectory,
    start_idx: int,
    end_idx: int,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    jacobian_fn: Optional[Callable] = None,
) -> Tuple[float, bool]:
    """Compute the task cost for one trajectory segment ``[start_idx, end_idx]``.

    Cost = α · C₁(smoothness) + β · C₂(manipulability).

    Args:
        trajectory: Robot trajectory with feasibility and joint data.
        start_idx: First waypoint index.
        end_idx: Last waypoint index (inclusive).
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        jacobian_fn: Optional ``(q) -> J`` for Yoshikawa manipulability.

    Returns:
        ``(cost, is_feasible)`` — cost is ``inf`` when infeasible.
    """
    n_waypoints = trajectory.n_waypoints
    if start_idx < 0 or end_idx >= n_waypoints or start_idx >= end_idx:
        logger.debug(
            "Segment [%d, %d] out of range for trajectory with %d waypoints",
            start_idx, end_idx, n_waypoints,
        )
        return np.inf, False

    segment_feasible = trajectory.is_feasible[start_idx : end_idx + 1]
    segment_joints = trajectory.joint_configs[start_idx : end_idx + 1]

    if len(segment_feasible) == 0 or not np.all(segment_feasible):
        logger.debug(
            "Segment [%d, %d] infeasible: %d/%d waypoints feasible",
            start_idx, end_idx,
            int(np.sum(segment_feasible)), len(segment_feasible),
        )
        return np.inf, False
    if len(segment_joints) == 0:
        return np.inf, False

    if segment_joints.ndim == 1:
        segment_joints = segment_joints.reshape(1, -1)

    # C1: smoothness — squared joint displacement (avoid sqrt in norm)
    smoothness = 0.0
    if segment_joints.shape[0] >= 2:
        diff = segment_joints[-1] - segment_joints[0]
        smoothness = float(diff @ diff)

    # C2: manipulability
    manip_vals = compute_trajectory_manipulability(
        segment_joints,
        use_jacobian=(jacobian_fn is not None),
        jacobian_fn=jacobian_fn,
    )
    manip_cost = float(np.sum(1.0 / (manip_vals + epsilon)))

    cost = alpha * smoothness + beta * manip_cost
    logger.debug(
        "Segment [%d, %d]: cost=%.4f (smoothness=%.4f, manip=%.4f)",
        start_idx, end_idx, cost, alpha * smoothness, beta * manip_cost,
    )
    return cost, True


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------

def compute_cost_matrix(
    robot0_trajectories: List[RobotTrajectory],
    robot1_trajectories: List[RobotTrajectory],
    n_segments: int,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """Build the per-segment cost for every ``(robot, trajectory)`` state.

    Args:
        robot0_trajectories: Robot 0 grasp trajectories.
        robot1_trajectories: Robot 1 grasp trajectories.
        n_segments: Number of trajectory segments.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
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
        n_wp = traj.n_waypoints
        feasible_segs = 0
        for seg in range(min(n_segments, n_wp - 1)):
            cost, ok = compute_segment_cost(
                traj, seg, seg + 1,
                alpha=alpha, beta=beta, epsilon=epsilon, jacobian_fn=jac_fn,
            )
            if ok:
                cost_matrix[seg, si] = cost
                feasible_segs += 1
        logger.debug(
            "State %d (robot=%d, traj=%d): %d/%d feasible segments",
            si, robot_id, traj_idx, feasible_segs, min(n_segments, n_wp - 1),
        )

    finite_count = int(np.isfinite(cost_matrix).sum())
    logger.info(
        "Cost matrix built: %d segments x %d states, %d/%d finite entries",
        n_segments, n_states, finite_count, n_segments * n_states,
    )
    return cost_matrix, trajectory_info


# ---------------------------------------------------------------------------
# Handover primitive cost
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Handover primitive helpers
# ---------------------------------------------------------------------------

def _solve_all_ik(
    poses: List[StampedPose],
    robot_id: int,
    ik_solver_fn: Callable,
    q_warm_start: Optional[np.ndarray] = None,
) -> Tuple[List[List[np.ndarray]], int]:
    """Solve IK for every pose, returning ALL candidate configs per pose.

    Returns:
        ``(all_candidates, n_failed)`` where ``all_candidates[i]`` is a
        list of valid joint configs for pose *i* (empty on failure).
    """
    all_candidates: List[List[np.ndarray]] = []
    q_prev = q_warm_start
    n_failed = 0
    for p in poses:
        try:
            result = ik_solver_fn(p, robot_id, q_warm_start=q_prev)
            if result is None:
                n_failed += 1
                all_candidates.append([])
                return all_candidates, n_failed
            candidates = result if isinstance(result, list) else [result]
            if q_prev is not None and len(candidates) > 1:
                candidates = sorted(
                    candidates,
                    key=lambda q: float(np.linalg.norm(q - q_prev)),
                )
            all_candidates.append(candidates)
            q_prev = candidates[0]
        except Exception:
            n_failed += 1
            all_candidates.append([])
    return all_candidates, n_failed


def _select_collision_free_configs(
    candidates_list: List[List[np.ndarray]],
    other_config: np.ndarray,
    this_robot_id: int,
    collision_check_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> Tuple[List[np.ndarray], int]:
    """Pick one config per waypoint, preferring collision-free candidates.

    Each waypoint's candidates are checked against a single fixed
    *other_config* (the opposing arm's config during that phase).

    Returns:
        ``(selected_configs, n_collision_steps)``
    """
    selected: List[np.ndarray] = []
    n_collision_steps = 0
    for cands in candidates_list:
        if not cands:
            continue
        found_free = False
        for q_cand in cands:
            q0, q1 = (q_cand, other_config) if this_robot_id == 0 else (other_config, q_cand)
            if collision_check_fn(q0, q1):
                selected.append(q_cand)
                found_free = True
                break
        if not found_free:
            selected.append(cands[0])
            n_collision_steps += 1
    return selected, n_collision_steps


def _compute_smoothness(configs: List[np.ndarray]) -> float:
    """Squared joint displacement along a config sequence."""
    if len(configs) < 2:
        return 0.0
    arr = np.array(configs)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(diffs * diffs))


def _compute_manipulability_cost(
    configs: List[np.ndarray],
    robot_id: int,
    epsilon: float,
    jacobian_fns: Optional[Dict[int, Callable]],
) -> float:
    """Reciprocal manipulability cost for one arm's config sequence."""
    if not configs:
        return 0.0
    arr = np.array(configs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    jac_fn = jacobian_fns.get(robot_id) if jacobian_fns else None
    manip_vals = compute_trajectory_manipulability(
        arr,
        use_jacobian=(jac_fn is not None),
        jacobian_fn=jac_fn,
    )
    return float(np.sum(1.0 / (manip_vals + epsilon)))


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
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    transfer_robot_id: int = 0,
    receiver_robot_id: int = 1,
    q_warm_start_transfer: Optional[np.ndarray] = None,
    q_warm_start_receiver: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """Evaluate cost of a handover primitive using the same C1 + C2 model.

    Args:
        jacobian_fns: Optional ``{robot_id: (q) -> J}`` for per-arm
            Yoshikawa manipulability.
        collision_check_fn: Optional ``(q_arm0, q_arm1) -> collision_free``
            to verify inter-arm clearance at critical handover configurations.
        transfer_robot_id: Robot ID of the arm performing the transfer (holding).
        receiver_robot_id: Robot ID of the arm receiving the object.
        q_warm_start_transfer: Joint config of the transfer arm at the
            handover point in the trajectory, used to warm-start IK.
        q_warm_start_receiver: Joint config of the receiver arm at the
            handover point in the trajectory, used to warm-start IK.
    """
    transfer_poses: List[StampedPose] = []
    receiver_poses: List[StampedPose] = []
    for key in ("transfer_hold", "transfer_retreat"):
        transfer_poses.extend(handover_trajectory.get(key, []))
    for key in ("receiver_pre", "receiver_approach"):
        receiver_poses.extend(handover_trajectory.get(key, []))

    if not transfer_poses and not receiver_poses:
        return np.inf, False

    # --- IK (early-out on any failure) ---
    transfer_candidates, n_fail_t = _solve_all_ik(
        transfer_poses, transfer_robot_id, ik_solver_fn, q_warm_start_transfer,
    ) if transfer_poses else ([], 0)
    if n_fail_t > 0:
        logger.debug("Handover infeasible: transfer arm (robot%d) — %d/%d IK failures",
                      transfer_robot_id, n_fail_t, len(transfer_poses))
        return np.inf, False

    receiver_candidates, n_fail_r = _solve_all_ik(
        receiver_poses, receiver_robot_id, ik_solver_fn, q_warm_start_receiver,
    ) if receiver_poses else ([], 0)
    if n_fail_r > 0:
        logger.debug("Handover infeasible: receiver arm (robot%d) — %d/%d IK failures",
                      receiver_robot_id, n_fail_r, len(receiver_poses))
        return np.inf, False

    # --- Collision-aware config selection ---
    transfer_configs, receiver_configs, n_collision_steps = _select_handover_configs(
        handover_trajectory, transfer_candidates, receiver_candidates,
        transfer_robot_id, receiver_robot_id, collision_check_fn,
    )

    # --- Cost ---
    smoothness = _compute_smoothness(transfer_configs) + _compute_smoothness(receiver_configs)
    manip_cost = (
        _compute_manipulability_cost(transfer_configs, transfer_robot_id, epsilon, jacobian_fns)
        + _compute_manipulability_cost(receiver_configs, receiver_robot_id, epsilon, jacobian_fns)
    )
    cost = alpha * smoothness + beta * manip_cost
    if n_collision_steps > 0:
        cost += constants.HANDOVER_COLLISION_STEP_PENALTY * n_collision_steps

    logger.debug(
        "Handover primitive cost: smoothness=%.4f, manip=%.4f, total=%.4f "
        "(transfer=robot%d, receiver=robot%d, collisions=%d)",
        smoothness, manip_cost, cost, transfer_robot_id, receiver_robot_id,
        n_collision_steps,
    )
    return cost, True


def _select_handover_configs(
    handover_trajectory: Dict[str, List[StampedPose]],
    transfer_candidates: List[List[np.ndarray]],
    receiver_candidates: List[List[np.ndarray]],
    transfer_robot_id: int,
    receiver_robot_id: int,
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]],
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Select one config per waypoint for both arms, preferring collision-free.

    Phase 1: receiver approaches while transfer holds at hold pose.
    Phase 2: transfer retreats while receiver stays at final approach.

    Returns:
        ``(transfer_configs, receiver_configs, n_collision_steps)``
    """
    if collision_check_fn is None or not transfer_candidates or not receiver_candidates:
        return (
            [c[0] for c in transfer_candidates if c],
            [c[0] for c in receiver_candidates if c],
            0,
        )

    n_collision_steps = 0
    n_hold = len(handover_trajectory.get("transfer_hold", []))
    transfer_configs = [c[0] for c in transfer_candidates]

    # Phase 1: receiver vs transfer hold
    q_transfer_hold = transfer_configs[0]
    receiver_configs, n_col_p1 = _select_collision_free_configs(
        receiver_candidates, q_transfer_hold,
        receiver_robot_id, collision_check_fn,
    )
    n_collision_steps += n_col_p1

    # Phase 2: transfer retreat vs receiver final
    if receiver_configs:
        q_receiver_final = receiver_configs[-1]
        retreat_candidates = transfer_candidates[n_hold:]
        if retreat_candidates:
            retreat_configs, n_col_p2 = _select_collision_free_configs(
                retreat_candidates, q_receiver_final,
                transfer_robot_id, collision_check_fn,
            )
            n_collision_steps += n_col_p2
            transfer_configs = transfer_configs[:n_hold] + retreat_configs

    if n_collision_steps > 0:
        logger.debug(
            "Handover has %d colliding steps (transfer=robot%d, receiver=robot%d)",
            n_collision_steps, transfer_robot_id, receiver_robot_id,
        )
    return transfer_configs, receiver_configs, n_collision_steps


# ---------------------------------------------------------------------------
# Switching cost
# ---------------------------------------------------------------------------

def compute_switching_cost(
    prev_robot: int,
    curr_robot: int,
    segment_idx: int,
    trajectories: TrajectoryBundle,
    prev_state_idx: int,
    curr_state_idx: int,
    trajectory_info: Dict[int, Tuple[int, int]],
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    handover_clearance_fn: Optional[Callable] = None,
) -> float:
    """Compute the cost of switching from one arm to another at *segment_idx*.

    Returns 0 when ``prev_robot == curr_robot``, ``inf`` when the transfer is
    blocked by *transfer_feasibility*, environment clearance, or an inter-arm
    collision, or the handover primitive cost otherwise.
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
    obj_traj = trajectories.object_trajectory
    if segment_idx >= obj_traj.n_waypoints:
        return np.inf

    obj_pose = StampedPose(
        position=obj_traj.positions[segment_idx],
        quaternion=obj_traj.quaternions[segment_idx],
        frame_id=_frame_id_at(obj_traj, segment_idx),
    )

    # Early-out: skip handover if environment obstacles are too close
    if handover_clearance_fn is not None and not handover_clearance_fn(obj_pose):
        logger.debug(
            "Handover blocked at seg %d: environment clearance check failed",
            segment_idx,
        )
        return np.inf

    prev_traj = trajectories.get_trajectory(prev_robot_id, prev_traj_idx)
    curr_traj = trajectories.get_trajectory(curr_robot_id, curr_traj_idx)

    if segment_idx >= prev_traj.n_waypoints:
        return np.inf
    if segment_idx >= curr_traj.n_waypoints:
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
        approach_steps=constants.COST_EVAL_APPROACH_STEPS,
        ramp_steps=constants.COST_EVAL_RAMP_STEPS,
        retreat_steps=constants.COST_EVAL_RETREAT_STEPS,
    )
    h_traj = h_planner.generate_trajectory()

    # Extract warm-start joint configs from the trajectories at the
    # handover segment so IK solutions stay close to the planned path.
    q_ws_transfer = None
    q_ws_receiver = None
    if (segment_idx < prev_traj.n_waypoints
            and prev_traj.is_feasible[segment_idx]):
        q_ws_transfer = prev_traj.joint_configs[segment_idx]
    if (segment_idx < curr_traj.n_waypoints
            and curr_traj.is_feasible[segment_idx]):
        q_ws_receiver = curr_traj.joint_configs[segment_idx]

    cost, ok = _compute_handover_primitive_cost(
        h_traj, alpha, beta, epsilon, ik_solver_fn, jacobian_fns,
        collision_check_fn=collision_check_fn,
        transfer_robot_id=prev_robot_id,
        receiver_robot_id=curr_robot_id,
        q_warm_start_transfer=q_ws_transfer,
        q_warm_start_receiver=q_ws_receiver,
    )

    if ok:
        logger.debug(
            "Switching cost at seg %d (robot %d→%d): %.4f",
            segment_idx, prev_robot, curr_robot, cost,
        )
        return cost

    # _compute_handover_primitive_cost returns (inf, False) only when
    # ALL IK calls failed for at least one arm — a truly infeasible
    # handover at this location.
    logger.debug(
        "Handover at seg %d (robot %d→%d) is truly infeasible — "
        "all IK calls failed for at least one arm.",
        segment_idx, prev_robot, curr_robot,
    )
    return np.inf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_id_at(obj_traj: ObjectTrajectory, idx: int) -> str:
    """Extract frame_id from object trajectory at *idx*."""
    fids = obj_traj.frame_ids
    if fids is None:
        return constants.WORLD_FRAME
    if isinstance(fids, np.ndarray):
        return str(fids[idx]) if idx < len(fids) else constants.WORLD_FRAME
    return str(fids)


def _eef_matrix(traj: RobotTrajectory, idx: int) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from the EEF pose at *idx*."""
    mat = np.eye(4)
    mat[:3, 3] = traj.eef_positions[idx]
    mat[:3, :3] = R.from_quat(traj.eef_quaternions[idx]).as_matrix()
    return mat
