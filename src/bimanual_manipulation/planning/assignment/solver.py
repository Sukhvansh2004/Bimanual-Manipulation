"""Viterbi-style DP solver for optimal robot-to-segment assignment.

Uses a cost matrix (built by :mod:`cost`) and switching costs to find
the globally optimal sequence ``(robot_id, trajectory_idx)`` for every
trajectory segment.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.planning.assignment.models import (
    TrajectoryBundle,
)
from bimanual_manipulation.planning.assignment.result import (
    AssignmentResult,
)
from bimanual_manipulation.planning.assignment.cost import (
    compute_cost_matrix,
    compute_switching_cost,
    HandoverResult,
)
from bimanual_manipulation.planning.utils.assignment_utils import (
    normalize_trajectory_configs,
    apply_beam_pruning as _apply_beam_pruning,
)
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation.utils.primitives import extract_grasp_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DP table initialisation
# ---------------------------------------------------------------------------

def _initialize_dp(
    cost_matrix: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    initial_robot: Optional[int],
    beam_width: Optional[int],
    step_feasibility: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create DP tables and populate segment 0.

    Returns:
        ``(J_star, prev_state, switch_cost_cache)``

    Raises:
        ValueError: If no state is feasible at segment 0.
    """
    n_segments, n_states = cost_matrix.shape

    J_star = np.full((n_segments, n_states), np.inf)
    prev_state = np.full((n_segments, n_states), -1, dtype=int)
    switch_cost_cache = np.zeros((n_segments, n_states))

    for si in range(n_states):
        robot_id, traj_idx = trajectory_info[si]
        if initial_robot is not None and robot_id != initial_robot:
            continue
        if np.isinf(cost_matrix[0, si]):
            continue
        # Segment 0 spans waypoints [0, 1]; block states whose joint
        # step from waypoint 0→1 exceeds the threshold.
        if step_feasibility is not None:
            step_ok = step_feasibility.get((robot_id, traj_idx))
            if step_ok is not None and len(step_ok) > 1 and not step_ok[1]:
                logger.debug(
                    "Segment 0 state %d (robot%d traj%d): blocked — "
                    "joint step exceeds threshold",
                    si, robot_id, traj_idx,
                )
                continue
        J_star[0, si] = cost_matrix[0, si]

    n_alive = int(np.sum(~np.isinf(J_star[0])))
    logger.info("Segment 0 initialised: %d/%d alive states", n_alive, n_states)
    if n_alive == 0:
        raise ValueError(
            "No feasible assignment found — all trajectories infeasible "
            "at segment 0."
        )

    if beam_width is not None and beam_width > 0:
        _apply_beam_pruning(J_star, 0, beam_width)

    return J_star, prev_state, switch_cost_cache


# ---------------------------------------------------------------------------
# Same-robot transition (Phase 1)
# ---------------------------------------------------------------------------

def _best_same_robot_transition(
    alive_prev: List[Tuple[float, int]],
    cr: int,
    ct: int,
    trajectory_info: Dict[int, Tuple[int, int]],
) -> Tuple[float, int, float]:
    """Find the best same-robot predecessor for state ``(cr, ct)``.

    Returns:
        ``(best_cost, best_prev_state, switching_cost=0.0)``
    """
    for acc_cost, ps in alive_prev:
        pr, pt = trajectory_info[ps]
        if pr == cr and pt == ct:
            return acc_cost, ps, 0.0
    return np.inf, -1, 0.0


# ---------------------------------------------------------------------------
# Cross-robot transitions (Phase 2)
# ---------------------------------------------------------------------------

def _best_cross_robot_transition(
    alive_prev: List[Tuple[float, int]],
    cr: int,
    best_cost: float,
    i: int,
    trajectories: TrajectoryBundle,
    cs: int,
    trajectory_info: Dict[int, Tuple[int, int]],
    *,
    transfer_feasibility,
    alpha: float,
    beta: float,
    epsilon: float,
    ik_solver_fn,
    jacobian_fns,
    collision_check_fn,
    handover_clearance_fn,
    handover_callback: Optional[Callable[[HandoverResult], None]],
) -> Tuple[float, int, float, int, int]:
    """Evaluate cross-robot handover transitions for segment *i*, state *cs*.

    Returns:
        ``(best_cost, best_prev, best_sw, n_ik_calls, n_pruned)``
    """
    n_ik = 0
    n_pruned = 0

    best_prev = -1
    best_sw = 0.0

    for idx, (acc_cost, ps) in enumerate(alive_prev):
        if acc_cost >= best_cost:
            n_pruned += len(alive_prev) - idx
            break
        pr, _ = trajectory_info[ps]
        if pr == cr:
            continue

        n_ik += 1
        sw = compute_switching_cost(
            pr, cr, i, trajectories, ps, cs, trajectory_info,
            transfer_feasibility=transfer_feasibility,
            alpha=alpha, beta=beta, epsilon=epsilon,
            ik_solver_fn=ik_solver_fn, jacobian_fns=jacobian_fns,
            collision_check_fn=collision_check_fn,
            handover_clearance_fn=handover_clearance_fn,
            handover_callback=handover_callback,
        )

        pt_prev = trajectory_info[ps][1]
        ct_curr = trajectory_info[cs][1]

        if np.isinf(sw):
            logger.debug(
                "Seg %d: handover robot%d→robot%d traj(%d→%d) rejected (inf)",
                i, pr, cr, pt_prev, ct_curr,
            )
            continue

        total = acc_cost + sw
        logger.debug(
            "Seg %d: handover robot%d→robot%d traj(%d→%d) total_cost=%.4f (handover cost=%.4f, acc_cost=%.4f)",
            i, pr, cr, pt_prev, ct_curr, total, sw, acc_cost,
        )
        if total < best_cost:
            best_cost = total
            best_prev = ps
            best_sw = sw

    return best_cost, best_prev, best_sw, n_ik, n_pruned


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward_pass(
    cost_matrix: np.ndarray,
    J_star: np.ndarray,
    prev_state: np.ndarray,
    switch_cost_cache: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    trajectories: TrajectoryBundle,
    *,
    transfer_feasibility,
    alpha: float,
    beta: float,
    epsilon: float,
    ik_solver_fn,
    jacobian_fns,
    collision_check_fn,
    handover_clearance_fn,
    handover_callback: Optional[Callable[[HandoverResult], None]],
    beam_width: Optional[int],
    handover_stride: int,
    step_feasibility: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> None:
    """Run the Viterbi forward pass, filling *J_star* in-place."""
    n_segments, n_states = cost_matrix.shape
    t_dp_start = time.time()
    total_ik_calls = 0
    total_pruned = 0

    for i in range(1, n_segments):
        t_seg_start = time.time()
        seg_ik = 0
        seg_pruned = 0

        alive_prev = sorted(
            ((float(J_star[i - 1, ps]), ps)
             for ps in range(n_states)
             if not np.isinf(J_star[i - 1, ps])),
        )

        for cs in range(n_states):
            if np.isinf(cost_matrix[i, cs]):
                continue

            cr, ct = trajectory_info[cs]

            # Phase 1: same-robot
            best_cost, best_prev, best_sw = _best_same_robot_transition(
                alive_prev, cr, ct, trajectory_info,
            )

            # Reject same-robot continuation when the joint step from
            # waypoint i to i+1 exceeds the threshold.  Cross-robot
            # transitions are unaffected because the handover primitive
            # allows the receiving robot to reconfigure.
            if best_prev != -1 and step_feasibility is not None:
                step_ok = step_feasibility.get((cr, ct))
                wp = i + 1  # segment i spans waypoints [i, i+1]
                if step_ok is not None and wp < len(step_ok) and not step_ok[wp]:
                    logger.debug(
                        "Seg %d state %d (robot%d traj%d): same-robot "
                        "transition blocked — joint step exceeds threshold",
                        i, cs, cr, ct,
                    )
                    best_cost = np.inf
                    best_prev = -1
                    best_sw = 0.0

            # Phase 2: cross-robot (only at stride boundaries)
            if i % handover_stride == 0:
                xr_cost, xr_prev, xr_sw, n_ik, n_pr = (
                    _best_cross_robot_transition(
                        alive_prev, cr, best_cost, i,
                        trajectories, cs, trajectory_info,
                        transfer_feasibility=transfer_feasibility,
                        alpha=alpha, beta=beta, epsilon=epsilon,
                        ik_solver_fn=ik_solver_fn,
                        jacobian_fns=jacobian_fns,
                        collision_check_fn=collision_check_fn,
                        handover_clearance_fn=handover_clearance_fn,
                        handover_callback=handover_callback,
                    )
                )
                seg_ik += n_ik
                seg_pruned += n_pr
                if xr_cost < best_cost:
                    logger.debug(
                        "Seg %d: cross-robot transition to state %d (robot%d traj%d) is better: %.4f < %.4f",
                        i, cs, cr, ct, xr_cost, best_cost,
                    )
                    best_cost = xr_cost
                    best_prev = xr_prev
                    best_sw = xr_sw

            if best_prev != -1:
                J_star[i, cs] = best_cost + cost_matrix[i, cs]
                prev_state[i, cs] = best_prev
                switch_cost_cache[i, cs] = best_sw

        if beam_width is not None and beam_width > 0:
            _apply_beam_pruning(J_star, i, beam_width)

        n_alive = int(np.sum(~np.isinf(J_star[i])))
        t_seg = time.time() - t_seg_start
        total_ik_calls += seg_ik
        total_pruned += seg_pruned

        elapsed = time.time() - t_dp_start
        remaining = n_segments - 1 - i
        eta = (elapsed / i) * remaining if i > 0 else 0

        logger.info(
            "Segment %d/%d: %.1fs (IK=%d, pruned=%d, alive=%d) ETA %.0fs",
            i, n_segments - 1, t_seg, seg_ik, seg_pruned, n_alive, eta,
        )

        if n_alive == 0:
            logger.error(
                "All states died at segment %d/%d.", i, n_segments - 1,
            )
            break

    logger.info(
        "Forward pass complete: total IK evals=%d, pruned=%d, elapsed=%.1fs",
        total_ik_calls, total_pruned, time.time() - t_dp_start,
    )


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------

def _backtrack(
    cost_matrix: np.ndarray,
    J_star: np.ndarray,
    prev_state: np.ndarray,
    switch_cost_cache: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    trajectories: TrajectoryBundle,
    normalized_configs_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> AssignmentResult:
    """Extract the optimal sequence from the filled DP tables.

    When *normalized_configs_cache* is provided, the result includes a
    ``normalized_configs`` array of shape ``(n_waypoints, DOF)``
    containing the active robot's joint configuration at each waypoint,
    with angles adjusted for continuity within joint limits.

    Raises:
        ValueError: If no feasible terminal state exists.
    """
    n_segments, n_states = cost_matrix.shape

    best_final_cost = np.inf
    best_final_state = -1
    for si in range(n_states):
        if J_star[n_segments - 1, si] < best_final_cost:
            best_final_cost = J_star[n_segments - 1, si]
            best_final_state = si

    if best_final_state == -1:
        raise ValueError(
            "No feasible assignment found — check trajectory feasibility."
        )

    logger.info(
        "Best terminal cost=%.4f at state %d",
        best_final_cost, best_final_state,
    )

    sequence: List[int] = []
    traj_paths: List[str] = []
    task_costs_list: List[float] = []
    switch_costs_list: List[float] = []
    segment_keys: List[Tuple[int, int]] = []  # (robot_id, traj_idx) per segment

    cs = best_final_state
    for i in range(n_segments - 1, -1, -1):
        robot_id, traj_idx = trajectory_info[cs]
        sequence.append(robot_id)
        segment_keys.append((robot_id, traj_idx))
        traj_obj = trajectories.get_trajectory(robot_id, traj_idx)
        traj_paths.append(
            traj_obj.source_path
            or f"robot{robot_id}_trajectory_{traj_idx}.npz"
        )
        task_costs_list.append(float(cost_matrix[i, cs]))

        if i > 0:
            switch_costs_list.append(float(switch_cost_cache[i, cs]))
            cs = prev_state[i, cs]
        else:
            switch_costs_list.append(0.0)

    sequence.reverse()
    traj_paths.reverse()
    task_costs_list.reverse()
    switch_costs_list.reverse()
    segment_keys.reverse()

    handover_points = [
        i for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]
    ]

    cost_summary = np.full((n_segments, 2), np.inf)
    traj_summary = np.full((n_segments, 2), -1, dtype=int)
    for seg in range(n_segments):
        for si, (rid, tidx) in trajectory_info.items():
            if cost_matrix[seg, si] < cost_summary[seg, rid]:
                cost_summary[seg, rid] = cost_matrix[seg, si]
                traj_summary[seg, rid] = tidx

    # --- Build normalised config sequence for the assigned path ---
    n_waypoints = n_segments + 1
    first_key = next(iter(normalized_configs_cache))
    dof = normalized_configs_cache[first_key].shape[1]
    assigned_configs = np.full((n_waypoints, dof), np.nan)

    for seg_idx, (rid, tidx) in enumerate(segment_keys):
        norm = normalized_configs_cache.get((rid, tidx))
        if norm is not None and seg_idx < norm.shape[0]:
            assigned_configs[seg_idx] = norm[seg_idx]
            if seg_idx == n_segments - 1 and seg_idx + 1 < norm.shape[0]:
                assigned_configs[seg_idx + 1] = norm[seg_idx + 1]

    logger.info(
        "Built normalised config sequence: shape=%s, "
        "nan_waypoints=%d/%d",
        assigned_configs.shape,
        int(np.any(np.isnan(assigned_configs), axis=1).sum()),
        n_waypoints,
    )

    # --- Build per-waypoint execution data for the assigned path ---
    n_waypoints = n_segments + 1
    eef_pos = np.zeros((n_waypoints, 3))
    eef_quat = np.zeros((n_waypoints, 4))
    eef_fids: List[str] = [""] * n_waypoints
    obj_pos = np.zeros((n_waypoints, 3))
    obj_quat = np.zeros((n_waypoints, 4))
    obj_fids: List[str] = [""] * n_waypoints
    is_feasible_arr = np.ones(n_waypoints, dtype=bool)

    for seg_idx, (rid, tidx) in enumerate(segment_keys):
        traj = trajectories.get_trajectory(rid, tidx)
        if seg_idx < traj.n_waypoints:
            eef_pos[seg_idx] = traj.eef_positions[seg_idx]
            eef_quat[seg_idx] = traj.eef_quaternions[seg_idx]
            eef_fids[seg_idx] = str(traj.eef_frame_ids[seg_idx])
            obj_pos[seg_idx] = traj.object_positions[seg_idx]
            obj_quat[seg_idx] = traj.object_quaternions[seg_idx]
            obj_fids[seg_idx] = str(traj.object_frame_ids[seg_idx])
            is_feasible_arr[seg_idx] = bool(traj.is_feasible[seg_idx])
        # Fill last waypoint from last segment's next waypoint
        if seg_idx == n_segments - 1:
            wp = seg_idx + 1
            if wp < traj.n_waypoints:
                eef_pos[wp] = traj.eef_positions[wp]
                eef_quat[wp] = traj.eef_quaternions[wp]
                eef_fids[wp] = str(traj.eef_frame_ids[wp])
                obj_pos[wp] = traj.object_positions[wp]
                obj_quat[wp] = traj.object_quaternions[wp]
                obj_fids[wp] = str(traj.object_frame_ids[wp])
                is_feasible_arr[wp] = bool(traj.is_feasible[wp])

    # --- Build per-unique-trajectory grasp transforms ---
    unique_paths = sorted(set(traj_paths))
    g_positions = np.zeros((len(unique_paths), 3))
    g_quaternions = np.zeros((len(unique_paths), 4))
    g_frame_ids: List[str] = [""] * len(unique_paths)

    # Map trajectory paths to (robot_id, traj_idx) for first occurrence
    path_to_key: Dict[str, Tuple[int, int]] = {}
    for seg_idx, (rid, tidx) in enumerate(segment_keys):
        p = traj_paths[seg_idx]
        if p not in path_to_key:
            path_to_key[p] = (rid, tidx)

    for gi, gpath in enumerate(unique_paths):
        rid, tidx = path_to_key[gpath]
        traj = trajectories.get_trajectory(rid, tidx)
        eef0 = StampedPose(
            position=traj.eef_positions[0],
            quaternion=traj.eef_quaternions[0],
            frame_id=str(traj.eef_frame_ids[0]),
        )
        obj0 = StampedPose(
            position=traj.object_positions[0],
            quaternion=traj.object_quaternions[0],
            frame_id=str(traj.object_frame_ids[0]),
        )
        grasp = extract_grasp_transform(eef0, obj0)
        g_positions[gi] = grasp.position
        g_quaternions[gi] = grasp.quaternion
        g_frame_ids[gi] = grasp.frame_id

    logger.info(
        "Built execution data: %d waypoints, %d grasp transforms",
        n_waypoints, len(unique_paths),
    )

    # --- Build handover transfer configs ---
    # At each handover point, store the *transferring* arm's normalised
    # config at that waypoint so execution can warm-start the hold/retreat.
    ho_transfer = np.full((len(handover_points), dof), np.nan)
    for hi, hp in enumerate(handover_points):
        # The segment *before* the handover was assigned to the transferor.
        prev_rid, prev_tidx = segment_keys[hp - 1]
        norm = normalized_configs_cache.get((prev_rid, prev_tidx))
        if norm is not None and hp < norm.shape[0]:
            ho_transfer[hi] = norm[hp]

    logger.info(
        "Handover transfer configs: %d handovers, nan=%d/%d",
        len(handover_points),
        int(np.any(np.isnan(ho_transfer), axis=1).sum()),
        len(handover_points),
    )

    return AssignmentResult(
        sequence=sequence,
        trajectory_paths=traj_paths,
        total_cost=best_final_cost,
        task_costs=np.array(task_costs_list),
        switching_costs=np.array(switch_costs_list),
        handover_points=handover_points,
        cost_matrix=cost_summary,
        trajectory_matrix=traj_summary,
        normalized_configs=assigned_configs,
        eef_positions=eef_pos,
        eef_quaternions=eef_quat,
        eef_frame_ids=eef_fids,
        object_positions=obj_pos,
        object_quaternions=obj_quat,
        object_frame_ids=obj_fids,
        is_feasible=is_feasible_arr,
        grasp_paths=unique_paths,
        grasp_positions=g_positions,
        grasp_quaternions=g_quaternions,
        grasp_frame_ids=g_frame_ids,
        handover_transfer_configs=ho_transfer,
    )


# ---------------------------------------------------------------------------
# Internal orchestrator
# ---------------------------------------------------------------------------

def _solve_optimal_assignment(
    cost_matrix: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    trajectories: TrajectoryBundle,
    transfer_feasibility=None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn=None,
    jacobian_fns=None,
    collision_check_fn=None,
    handover_clearance_fn=None,
    handover_callback: Optional[Callable[[HandoverResult], None]] = None,
    initial_robot: Optional[int] = None,
    beam_width: Optional[int] = None,
    handover_stride: int = 1,
    normalized_configs_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    step_feasibility: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> AssignmentResult:
    """Solve the optimal assignment with a Viterbi-style DP.

    Delegates to :func:`_initialize_dp`, :func:`_forward_pass`, and
    :func:`_backtrack`.
    """
    n_segments, n_states = cost_matrix.shape
    logger.info(
        "DP solver: %d segments x %d states, beam_width=%s, "
        "handover_stride=%d",
        n_segments, n_states,
        beam_width if beam_width is not None else "unlimited",
        handover_stride,
    )

    J_star, prev_state, switch_cost_cache = _initialize_dp(
        cost_matrix, trajectory_info, initial_robot, beam_width,
        step_feasibility=step_feasibility,
    )

    _forward_pass(
        cost_matrix, J_star, prev_state, switch_cost_cache,
        trajectory_info, trajectories,
        transfer_feasibility=transfer_feasibility,
        alpha=alpha, beta=beta, epsilon=epsilon,
        ik_solver_fn=ik_solver_fn,
        jacobian_fns=jacobian_fns,
        collision_check_fn=collision_check_fn,
        handover_clearance_fn=handover_clearance_fn,
        handover_callback=handover_callback,
        beam_width=beam_width,
        handover_stride=handover_stride,
        step_feasibility=step_feasibility,
    )

    return _backtrack(
        cost_matrix, J_star, prev_state, switch_cost_cache,
        trajectory_info, trajectories,
        normalized_configs_cache=normalized_configs_cache,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_optimal_robot_assignment(
    trajectories: TrajectoryBundle,
    transfer_feasibility=None,
    initial_robot: Optional[int] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn=None,
    jacobian_fns=None,
    collision_check_fn=None,
    handover_clearance_fn=None,
    handover_callback: Optional[Callable[[HandoverResult], None]] = None,
    beam_width: Optional[int] = None,
    handover_stride: int = 1,
    joint_limits_lo: np.ndarray = constants.FRANKA_JOINT_LIMITS_LO,
    joint_limits_hi: np.ndarray = constants.FRANKA_JOINT_LIMITS_HI,
    max_joint_step: float = constants.ASSIGNMENT_MAX_JOINT_STEP,
) -> AssignmentResult:
    """Compute the globally optimal robot-to-segment assignment.

    Args:
        trajectories: Typed trajectory bundle for both arms.
        transfer_feasibility: Optional ``{(from_robot, from_traj, to_robot,
            to_traj): bool}`` — *False* blocks that handover.
        initial_robot: Force the first segment to a specific robot.
        alpha: Smoothness weight.
        beta: Manipulability weight.
        epsilon: Numerical floor for manipulability reciprocal.
        ik_solver_fn: ``(pose, robot_id) -> q`` for handover cost evaluation.
        jacobian_fns: ``{robot_id: (q) -> J}`` per-robot Jacobian functions.
        collision_check_fn: ``(q_arm0, q_arm1) -> collision_free``.
        handover_clearance_fn: ``(object_pose) -> bool`` — environment
            clearance filter.
        handover_callback: Called with a :class:`HandoverResult` after every
            evaluated handover attempt.  Useful for visualization or logging.
        beam_width: Max surviving DP states per segment.
        handover_stride: Evaluate handovers every *handover_stride* segments.
        joint_limits_lo: ``(DOF,)`` lower joint position limits.  Defaults to
            Franka Panda limits from
            :data:`constants.FRANKA_JOINT_LIMITS_LO`.
        joint_limits_hi: ``(DOF,)`` upper joint position limits.  Defaults to
            Franka Panda limits from
            :data:`constants.FRANKA_JOINT_LIMITS_HI`.
        max_joint_step: Maximum per-joint angle change (rad) between
            consecutive waypoints.  Transitions exceeding this after
            angle normalisation are marked infeasible.  Defaults to
            :data:`constants.ASSIGNMENT_MAX_JOINT_STEP`.

    Returns:
        :class:`AssignmentResult` with the optimal sequence, cost breakdown,
        and ``normalized_configs`` for direct use by the execution controller.

    Raises:
        ValueError: If no feasible assignment exists.
    """
    r0 = trajectories.robot0_trajectories
    r1 = trajectories.robot1_trajectories
    n_segments = trajectories.n_segments

    logger.info(
        "Computing assignment for %d segments (%d R0 + %d R1 trajectories, "
        "alpha=%.2f, beta=%.2f, beam=%s, stride=%d)",
        n_segments, len(r0), len(r1), alpha, beta,
        beam_width if beam_width is not None else "unlimited",
        handover_stride,
    )

    # --- Pre-normalise trajectory joint configs and check step feasibility --
    normalized_configs_cache: Dict[Tuple[int, int], np.ndarray] = {}
    step_feasibility: Dict[Tuple[int, int], np.ndarray] = {}
    # Track original configs by object identity so that shared trajectory
    # objects (e.g. same object in both robot lists) are handled safely.
    originals: Dict[int, np.ndarray] = {}  # id(traj) -> original joint_configs

    for robot_id, trajs in [(0, r0), (1, r1)]:
        for traj_idx, traj in enumerate(trajs):
            traj_obj_id = id(traj)
            if traj_obj_id not in originals:
                originals[traj_obj_id] = traj.joint_configs

            # Always normalise from the true original so that repeated
            # processing of a shared trajectory object is idempotent.
            norm, step_ok = normalize_trajectory_configs(
                originals[traj_obj_id], traj.is_feasible,
                joint_limits_lo, joint_limits_hi, max_joint_step,
            )
            normalized_configs_cache[(robot_id, traj_idx)] = norm
            step_feasibility[(robot_id, traj_idx)] = step_ok
            # Temporarily replace with normalised configs so that
            # cost computation (smoothness) uses the adjusted angles.
            traj.joint_configs = norm

    n_infeasible_total = sum(
        int((~ok[1:]).sum()) for ok in step_feasibility.values()
    )
    if n_infeasible_total:
        logger.info(
            "Joint-step normalisation: %d segment transitions across all "
            "trajectories exceed the max-step threshold — same-robot "
            "continuations at those segments will be blocked (cross-robot "
            "handovers remain allowed).",
            n_infeasible_total,
        )

    try:
        cost_matrix, traj_info = compute_cost_matrix(
            r0, r1, n_segments,
            alpha=alpha, beta=beta, epsilon=epsilon, jacobian_fns=jacobian_fns,
        )

        logger.info("Cost matrix: %d segments x %d states", *cost_matrix.shape)

        result = _solve_optimal_assignment(
            cost_matrix, traj_info, trajectories,
            transfer_feasibility=transfer_feasibility,
            alpha=alpha, beta=beta, epsilon=epsilon,
            ik_solver_fn=ik_solver_fn, jacobian_fns=jacobian_fns,
            collision_check_fn=collision_check_fn,
            handover_clearance_fn=handover_clearance_fn,
            handover_callback=handover_callback,
            initial_robot=initial_robot,
            beam_width=beam_width,
            handover_stride=handover_stride,
            normalized_configs_cache=normalized_configs_cache,
            step_feasibility=step_feasibility,
        )
    finally:
        # Restore original joint configs so the caller's objects aren't
        # permanently mutated.
        for trajs in (r0, r1):
            for traj in trajs:
                orig = originals.get(id(traj))
                if orig is not None:
                    traj.joint_configs = orig

    logger.info(
        "Assignment solved: total_cost=%.4f, handovers=%d",
        result.total_cost, len(result.handover_points),
    )
    return result
