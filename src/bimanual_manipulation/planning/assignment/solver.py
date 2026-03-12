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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beam-pruning helper
# ---------------------------------------------------------------------------

def _apply_beam_pruning(
    J_star: np.ndarray, seg_idx: int, beam_width: int,
) -> None:
    """Keep only the top *beam_width* states at *seg_idx*; set rest to inf."""
    row = J_star[seg_idx]
    finite_indices = np.where(~np.isinf(row))[0]
    if len(finite_indices) <= beam_width:
        return
    sorted_by_cost = finite_indices[np.argsort(row[finite_indices])]
    row[sorted_by_cost[beam_width:]] = np.inf


# ---------------------------------------------------------------------------
# DP table initialisation
# ---------------------------------------------------------------------------

def _initialize_dp(
    cost_matrix: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    initial_robot: Optional[int],
    beam_width: Optional[int],
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
        robot_id, _ = trajectory_info[si]
        if initial_robot is not None and robot_id != initial_robot:
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

    for acc_cost, ps in alive_prev:
        if acc_cost >= best_cost:
            n_pruned += 1
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
) -> AssignmentResult:
    """Extract the optimal sequence from the filled DP tables.

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

    cs = best_final_state
    for i in range(n_segments - 1, -1, -1):
        robot_id, traj_idx = trajectory_info[cs]
        sequence.append(robot_id)
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

    return AssignmentResult(
        sequence=sequence,
        trajectory_paths=traj_paths,
        total_cost=best_final_cost,
        task_costs=np.array(task_costs_list),
        switching_costs=np.array(switch_costs_list),
        handover_points=handover_points,
        cost_matrix=cost_summary,
        trajectory_matrix=traj_summary,
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
    )

    return _backtrack(
        cost_matrix, J_star, prev_state, switch_cost_cache,
        trajectory_info, trajectories,
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

    Returns:
        :class:`AssignmentResult` with the optimal sequence and cost breakdown.

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
    )

    logger.info(
        "Assignment solved: total_cost=%.4f, handovers=%d",
        result.total_cost, len(result.handover_points),
    )
    return result
