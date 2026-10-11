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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)
from bimanual_manipulation.planning.ik.manipulability import (
    compute_trajectory_manipulability,
)
from bimanual_manipulation.planning.assignment.utils import (
    compute_smoothness as _compute_smoothness,
    compute_manipulability_cost as _compute_manipulability_cost,
    frame_id_at as _frame_id_at,
    eef_matrix as _eef_matrix,
)
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation.utils.primitives import (
    DualArmHandoverPlanner,
    HandoverTrajectory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handover result container
# ---------------------------------------------------------------------------

@dataclass
class HandoverResult:
    """Rich result from a handover primitive evaluation.

    Attributes:
        cost: Handover cost ``alpha * C1 + beta * C2``, or ``inf`` when
            infeasible.
        feasible: ``True`` when IK succeeded for every pose and no
            collision was detected.
        transfer_robot_id: Robot handing the object over.
        receiver_robot_id: Robot receiving the object.
        segment_idx: Segment index at which the handover is evaluated
            (``-1`` until annotated by :func:`compute_switching_cost`).
        prev_traj_idx: Trajectory index of the transferring robot.
        curr_traj_idx: Trajectory index of the receiving robot.
        transfer_configs: Selected joint configs for the transferring arm
            (hold followed by retreat).
        receiver_configs: Selected joint configs for the receiving arm
            (pre-grasp followed by approach).
        n_collision_steps: Number of waypoints with no collision-free
            candidate.
        transfer_collision_mask: Per-waypoint collision flags for the
            transferring arm (``True`` = in collision).
        receiver_collision_mask: Per-waypoint collision flags for the
            receiving arm (``True`` = in collision).
        rejection_reason: Short reason string when infeasible, else ``""``.
        object_pose: Object pose at the handover segment.
    """

    cost: float
    feasible: bool
    transfer_robot_id: int = 0
    receiver_robot_id: int = 1
    segment_idx: int = -1
    prev_traj_idx: int = -1
    curr_traj_idx: int = -1
    transfer_configs: List[np.ndarray] = field(default_factory=list)
    receiver_configs: List[np.ndarray] = field(default_factory=list)
    n_collision_steps: int = 0
    transfer_collision_mask: List[bool] = field(default_factory=list)
    receiver_collision_mask: List[bool] = field(default_factory=list)
    rejection_reason: str = ""
    object_pose: Optional[StampedPose] = None


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

    Cost = α · C₁(smoothness) + β · C₂(manipulability), where C₁ is the
    squared joint displacement between the first and last waypoint of the
    segment and C₂ is ``sum(1 / (w + epsilon))`` over all waypoints in the
    segment.  Without *jacobian_fn* the fast proxy manipulability from
    :func:`compute_manipulability_simple` is used.

    Args:
        trajectory: Robot trajectory with feasibility and joint data.
        start_idx: First waypoint index.
        end_idx: Last waypoint index (inclusive); must be ``> start_idx``.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        jacobian_fn: Optional ``(q) -> J`` for Yoshikawa manipulability.

    Returns:
        ``(cost, is_feasible)`` — cost is ``inf`` and *is_feasible* is
        ``False`` when the index range is invalid or any waypoint in the
        segment is infeasible.
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

    States are numbered with all robot-0 trajectories first, followed by
    all robot-1 trajectories.  Segment ``seg`` spans waypoints
    ``[seg, seg + 1]`` and is scored with :func:`compute_segment_cost`.

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

        * ``cost_matrix`` — ``(n_segments, n_states)`` array where
          ``cost_matrix[seg, state_idx]`` is the task cost (``inf`` when
          the segment is infeasible or beyond the trajectory's length).
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
# Handover primitive helpers
# ---------------------------------------------------------------------------

def _solve_all_ik(
    poses: List[StampedPose],
    robot_id: int,
    ik_solver_fn: Callable,
    q_warm_start: Optional[np.ndarray] = None,
) -> Tuple[List[List[np.ndarray]], int]:
    """Solve IK for every pose, returning ALL candidate configs per pose.

    Each pose is warm-started from the best candidate of the previous
    pose, and candidates are sorted by distance to that warm start.
    Exceptions raised by *ik_solver_fn* are counted as failures.

    When :data:`constants.HANDOVER_EARLY_EXIT_ON_IK_FAILURE` is ``True``
    iteration stops at the first failure, avoiding unnecessary IK calls.

    Args:
        poses: Target EEF poses, in order.
        robot_id: Robot to solve IK for.
        ik_solver_fn: ``(pose, robot_id, q_warm_start=...) -> result``
            where *result* is ``None``, a single config, a list of
            configs, or a tuple whose first element is one of those.
        q_warm_start: Optional warm start for the first pose.

    Returns:
        ``(all_candidates, n_failed)`` where ``all_candidates[i]`` is a
        list of valid joint configs for pose *i* (empty on failure).
        On early exit *all_candidates* is shorter than *poses*.
    """
    early_exit = constants.HANDOVER_EARLY_EXIT_ON_IK_FAILURE
    all_candidates: List[List[np.ndarray]] = []
    q_prev = q_warm_start
    n_failed = 0
    for p in poses:
        try:
            raw = ik_solver_fn(p, robot_id, q_warm_start=q_prev)
            # Accept either plain result or (result, residual) tuple
            if isinstance(raw, tuple):
                result = raw[0]
            else:
                result = raw
            if result is None:
                n_failed += 1
                all_candidates.append([])
                if early_exit:
                    return all_candidates, n_failed
                continue
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
            if early_exit:
                return all_candidates, n_failed
    return all_candidates, n_failed


def _select_collision_free_configs(
    candidates_list: List[List[np.ndarray]],
    other_config: np.ndarray,
    this_robot_id: int,
    collision_check_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> Tuple[List[np.ndarray], List[bool]]:
    """Pick one config per waypoint, preferring collision-free candidates.

    Each waypoint's candidates are checked against a single fixed
    *other_config* (the opposing arm's config during that phase).

    When no candidate is collision-free the first candidate is selected
    and the step is flagged as colliding.  Waypoints with no candidates
    are skipped.  When :data:`constants.HANDOVER_EARLY_EXIT_ON_COLLISION`
    is ``True`` iteration stops at the first colliding waypoint.

    Args:
        candidates_list: Per-waypoint lists of candidate configs for
            this arm.
        other_config: Fixed config of the opposing arm.
        this_robot_id: Robot the candidates belong to (decides argument
            order for *collision_check_fn*).
        collision_check_fn: ``(q_arm0, q_arm1) -> collision_free``.

    Returns:
        ``(selected_configs, collision_mask)`` where *collision_mask* is a
        list of bools (``True`` = that step is in collision).
    """
    early_exit = constants.HANDOVER_EARLY_EXIT_ON_COLLISION
    selected: List[np.ndarray] = []
    collision_mask: List[bool] = []
    for cands in candidates_list:
        if not cands:
            continue
        found_free = False
        for q_cand in cands:
            q0, q1 = (q_cand, other_config) if this_robot_id == 0 else (other_config, q_cand)
            if collision_check_fn(q0, q1):
                selected.append(q_cand)
                collision_mask.append(False)
                found_free = True
                break
        if not found_free:
            selected.append(cands[0])
            collision_mask.append(True)
            if early_exit:
                return selected, collision_mask
    return selected, collision_mask


# ---------------------------------------------------------------------------
# Handover primitive cost
# ---------------------------------------------------------------------------

def _compute_handover_primitive_cost(
    handover_trajectory: HandoverTrajectory,
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
) -> HandoverResult:
    """Evaluate cost of a handover primitive using the same C1 + C2 model.

    The transferring arm follows ``transfer_hold + transfer_retreat`` and
    the receiving arm follows ``receiver_pre + receiver_approach``.  Any
    IK failure or any colliding step makes the handover infeasible
    (``inf`` cost); there is no soft penalty.  The early-exit flags
    :data:`constants.HANDOVER_EARLY_EXIT_ON_IK_FAILURE` and
    :data:`constants.HANDOVER_EARLY_EXIT_ON_COLLISION` only stop IK /
    collision checking at the first failure to save time.

    On success, C1 is the sum of squared consecutive joint displacements
    of both arms (:func:`compute_smoothness`) and C2 the summed reciprocal
    manipulability of both arms (:func:`compute_manipulability_cost`).

    Args:
        handover_trajectory: Handover EEF pose sequences for both arms.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        ik_solver_fn: IK callable (see :func:`_solve_all_ik`).
        jacobian_fns: Optional ``{robot_id: (q) -> J}`` mapping.
        collision_check_fn: Optional ``(q_arm0, q_arm1) -> collision_free``.
            When ``None`` no collision checking is done.
        transfer_robot_id: Robot handing the object over.
        receiver_robot_id: Robot receiving the object.
        q_warm_start_transfer: IK warm start for the transferring arm.
        q_warm_start_receiver: IK warm start for the receiving arm.

    Returns:
        :class:`HandoverResult` with cost, feasibility, resolved joint
        configs, and collision info.
    """
    transfer_poses: List[StampedPose] = (
        handover_trajectory.transfer_hold + handover_trajectory.transfer_retreat
    )
    receiver_poses: List[StampedPose] = (
        handover_trajectory.receiver_pre + handover_trajectory.receiver_approach
    )

    if not transfer_poses and not receiver_poses:
        return HandoverResult(
            cost=np.inf, feasible=False,
            transfer_robot_id=transfer_robot_id,
            receiver_robot_id=receiver_robot_id,
            rejection_reason="empty_trajectory",
        )

    # --- IK ---
    transfer_candidates, n_fail_t = _solve_all_ik(
        transfer_poses, transfer_robot_id, ik_solver_fn, q_warm_start_transfer,
    ) if transfer_poses else ([], 0)
    if n_fail_t > 0:
        n_seen_t = len(transfer_candidates)
        logger.debug("Handover infeasible: transfer arm (robot%d) — %d/%d IK failures (%d/%d seen)",
                      transfer_robot_id, n_fail_t, len(transfer_poses), n_seen_t, len(transfer_poses))
        if constants.HANDOVER_EARLY_EXIT_ON_IK_FAILURE:
            return HandoverResult(
                cost=np.inf, feasible=False,
                transfer_robot_id=transfer_robot_id,
                receiver_robot_id=receiver_robot_id,
                rejection_reason=f"ik_fail_transfer({n_fail_t}_failed@{n_seen_t}/{len(transfer_poses)}_seen)",
            )

    receiver_candidates, n_fail_r = _solve_all_ik(
        receiver_poses, receiver_robot_id, ik_solver_fn, q_warm_start_receiver,
    ) if receiver_poses else ([], 0)
    if n_fail_r > 0:
        n_seen_r = len(receiver_candidates)
        logger.debug("Handover infeasible: receiver arm (robot%d) — %d/%d IK failures (%d/%d seen)",
                      receiver_robot_id, n_fail_r, len(receiver_poses), n_seen_r, len(receiver_poses))
        if constants.HANDOVER_EARLY_EXIT_ON_IK_FAILURE:
            return HandoverResult(
                cost=np.inf, feasible=False,
                transfer_robot_id=transfer_robot_id,
                receiver_robot_id=receiver_robot_id,
                rejection_reason=f"ik_fail_receiver({n_fail_r}_failed@{n_seen_r}/{len(receiver_poses)}_seen)",
            )

    if n_fail_r > 0 or n_fail_t > 0:
        logger.debug(
            "Handover IK failures: transfer=robot%d %d/%d, receiver=robot%d %d/%d",
            transfer_robot_id, n_fail_t, len(transfer_poses),
            receiver_robot_id, n_fail_r, len(receiver_poses),
        )
        return HandoverResult(
            cost=np.inf, feasible=False,
            transfer_robot_id=transfer_robot_id,
            receiver_robot_id=receiver_robot_id,
            rejection_reason=f"ik_fail_transfer({n_fail_t}/{len(transfer_poses)}) "
                             f"receiver({n_fail_r}/{len(receiver_poses)})",
        )
    
    # --- Collision-aware config selection ---
    transfer_configs, receiver_configs, n_collision_steps, t_col_mask, r_col_mask = (
        _select_handover_configs(
            handover_trajectory, transfer_candidates, receiver_candidates,
            transfer_robot_id, receiver_robot_id, collision_check_fn,
        )
    )

    # Early-exit on collision
    if n_collision_steps > 0:
        n_hold = len(handover_trajectory.transfer_hold)
        n_total_checkable = len(receiver_candidates) + max(0, len(transfer_candidates) - n_hold)
        n_checked = len(r_col_mask) + max(0, len(t_col_mask) - n_hold)
        logger.debug(
            "Handover infeasible: %d collisions (%d/%d checked, transfer=robot%d, receiver=robot%d)",
            n_collision_steps, n_checked, n_total_checkable,
            transfer_robot_id, receiver_robot_id,
        )
        return HandoverResult(
            cost=np.inf, feasible=False,
            transfer_robot_id=transfer_robot_id,
            receiver_robot_id=receiver_robot_id,
            transfer_configs=transfer_configs,
            receiver_configs=receiver_configs,
            n_collision_steps=n_collision_steps,
            transfer_collision_mask=t_col_mask,
            receiver_collision_mask=r_col_mask,
            rejection_reason=f"collision({n_collision_steps}_colliding@{n_checked}/{n_total_checkable}_checked)",
        )

    # --- Cost ---
    smoothness = _compute_smoothness(transfer_configs) + _compute_smoothness(receiver_configs)
    manip_cost = (
        _compute_manipulability_cost(transfer_configs, transfer_robot_id, epsilon, jacobian_fns)
        + _compute_manipulability_cost(receiver_configs, receiver_robot_id, epsilon, jacobian_fns)
    )
    cost = alpha * smoothness + beta * manip_cost

    logger.debug(
        "Handover primitive cost: smoothness=%.4f, manip=%.4f, total=%.4f "
        "(transfer=robot%d, receiver=robot%d, collisions=%d)",
        smoothness, manip_cost, cost, transfer_robot_id, receiver_robot_id,
        n_collision_steps,
    )
    return HandoverResult(
        cost=cost, feasible=True,
        transfer_robot_id=transfer_robot_id,
        receiver_robot_id=receiver_robot_id,
        transfer_configs=transfer_configs,
        receiver_configs=receiver_configs,
        n_collision_steps=n_collision_steps,
        transfer_collision_mask=t_col_mask,
        receiver_collision_mask=r_col_mask,
    )


def _select_handover_configs(
    handover_trajectory: HandoverTrajectory,
    transfer_candidates: List[List[np.ndarray]],
    receiver_candidates: List[List[np.ndarray]],
    transfer_robot_id: int,
    receiver_robot_id: int,
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]],
) -> Tuple[List[np.ndarray], List[np.ndarray], int, List[bool], List[bool]]:
    """Select one config per waypoint for both arms, preferring collision-free.

    Phase 1: receiver approaches while transfer holds at hold pose (each
    receiver step is checked against the first transfer hold config).
    Phase 2: transfer retreats while receiver stays at final approach.
    Transfer hold steps are never collision-checked.  When
    *collision_check_fn* is ``None`` or either candidate list is empty,
    the first candidate per waypoint is taken with no checks.

    Args:
        handover_trajectory: Handover trajectory (used for the number of
            transfer hold poses).
        transfer_candidates: Per-waypoint IK candidates for the
            transferring arm (hold then retreat).
        receiver_candidates: Per-waypoint IK candidates for the
            receiving arm.
        transfer_robot_id: Robot handing the object over.
        receiver_robot_id: Robot receiving the object.
        collision_check_fn: Optional ``(q_arm0, q_arm1) -> collision_free``.

    Returns:
        ``(transfer_configs, receiver_configs, n_collision_steps,
        transfer_collision_mask, receiver_collision_mask)``
    """
    if collision_check_fn is None or not transfer_candidates or not receiver_candidates:
        t_cfgs = [c[0] for c in transfer_candidates if c]
        r_cfgs = [c[0] for c in receiver_candidates if c]
        return t_cfgs, r_cfgs, 0, [False] * len(t_cfgs), [False] * len(r_cfgs)

    n_collision_steps = 0
    n_hold = len(handover_trajectory.transfer_hold)
    transfer_configs = [c[0] for c in transfer_candidates]
    transfer_mask: List[bool] = [False] * len(transfer_configs)

    # Phase 1: receiver vs transfer hold
    q_transfer_hold = transfer_configs[0]
    receiver_configs, receiver_mask = _select_collision_free_configs(
        receiver_candidates, q_transfer_hold,
        receiver_robot_id, collision_check_fn,
    )
    n_collision_steps += sum(receiver_mask)

    if n_collision_steps > 0 and constants.HANDOVER_EARLY_EXIT_ON_COLLISION:
        return transfer_configs, receiver_configs, n_collision_steps, transfer_mask, receiver_mask
    
    # Phase 2: transfer retreat vs receiver final
    if receiver_configs:
        q_receiver_final = receiver_configs[-1]
        retreat_candidates = transfer_candidates[n_hold:]
        if retreat_candidates:
            retreat_configs, retreat_mask = _select_collision_free_configs(
                retreat_candidates, q_receiver_final,
                transfer_robot_id, collision_check_fn,
            )
            n_collision_steps += sum(retreat_mask)
            transfer_configs = transfer_configs[:n_hold] + retreat_configs
            transfer_mask = transfer_mask[:n_hold] + retreat_mask

    return transfer_configs, receiver_configs, n_collision_steps, transfer_mask, receiver_mask


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
    handover_callback: Optional[Callable[[HandoverResult], None]] = None,
) -> float:
    """Compute the cost of switching from one arm to another at *segment_idx*.

    Returns 0 when ``prev_robot == curr_robot``; ``inf`` when the transfer
    is blocked by *transfer_feasibility* (checked in both directions),
    *segment_idx* is beyond the object or either robot trajectory, the
    environment-clearance check fails, either trajectory is infeasible at
    *segment_idx*, or the handover primitive is infeasible (IK failure or
    collision); :data:`constants.ASSIGNMENT_FALLBACK_SWITCH_COST` when
    *ik_solver_fn* is ``None``; and the handover primitive cost otherwise.

    When *handover_callback* is provided it is called with a
    :class:`HandoverResult` after every evaluated handover primitive (both
    successful and rejected); early ``inf`` returns before primitive
    evaluation do not trigger it.

    Args:
        prev_robot: Robot id active before the switch.
        curr_robot: Robot id active after the switch.
        segment_idx: Segment (waypoint) index at which the handover occurs.
        trajectories: Bundle containing the object and robot trajectories.
        prev_state_idx: DP state index before the switch.
        curr_state_idx: DP state index after the switch.
        trajectory_info: ``{state_idx: (robot_id, traj_idx)}`` mapping.
        transfer_feasibility: Optional
            ``{(r_from, t_from, r_to, t_to): feasible}`` constraints.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        ik_solver_fn: IK callable used to evaluate the handover primitive.
        jacobian_fns: Optional ``{robot_id: (q) -> J}`` mapping.
        collision_check_fn: Optional ``(q_arm0, q_arm1) -> collision_free``.
        handover_clearance_fn: Optional ``(obj_pose) -> bool`` returning
            ``True`` when the handover location is clear of obstacles.
        handover_callback: Optional callback receiving each
            :class:`HandoverResult`.

    Returns:
        Switching cost (``0.0``, a finite cost, or ``inf``).
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

    # Compute grasp transforms: G = T_ee⁻¹ @ T_obj  (ee ← object)
    T_obj = obj_pose.to_matrix()
    T_ee_prev = _eef_matrix(prev_traj, segment_idx)
    T_ee_curr = _eef_matrix(curr_traj, segment_idx)

    grasp_transfer = StampedPose.from_matrix(
        np.linalg.inv(T_ee_prev) @ T_obj, "ee"
    )
    grasp_receiver = StampedPose.from_matrix(
        np.linalg.inv(T_ee_curr) @ T_obj, "ee"
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

    # Abort if either arm's trajectory is infeasible at the handover segment
    # — the arm can't physically reach that pose, so the handover is moot.
    if not prev_traj.is_feasible[segment_idx]:
        logger.debug(
            "Handover blocked at seg %d: transfer traj (robot%d, traj%d) infeasible",
            segment_idx, prev_robot_id, prev_traj_idx,
        )
        return np.inf
    if not curr_traj.is_feasible[segment_idx]:
        logger.debug(
            "Handover blocked at seg %d: receiver traj (robot%d, traj%d) infeasible",
            segment_idx, curr_robot_id, curr_traj_idx,
        )
        return np.inf

    # Extract warm-start joint configs from the trajectories at the
    # handover segment so IK solutions stay close to the planned path.
    q_ws_transfer = prev_traj.joint_configs[segment_idx]
    q_ws_receiver = curr_traj.joint_configs[segment_idx]

    result = _compute_handover_primitive_cost(
        h_traj, alpha, beta, epsilon, ik_solver_fn, jacobian_fns,
        collision_check_fn=collision_check_fn,
        transfer_robot_id=prev_robot_id,
        receiver_robot_id=curr_robot_id,
        q_warm_start_transfer=q_ws_transfer,
        q_warm_start_receiver=q_ws_receiver,
    )
    # Annotate result with trajectory metadata
    result.segment_idx = segment_idx
    result.prev_traj_idx = prev_traj_idx
    result.curr_traj_idx = curr_traj_idx
    result.object_pose = obj_pose

    if handover_callback is not None:
        handover_callback(result)

    if result.feasible:
        return result.cost

    logger.debug(
        "Handover at seg %d (robot %d→%d) is truly infeasible — %s",
        segment_idx, prev_robot, curr_robot, result.rejection_reason,
    )
    return np.inf
