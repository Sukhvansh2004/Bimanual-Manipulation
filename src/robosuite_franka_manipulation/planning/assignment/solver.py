"""Viterbi-style DP solver for optimal robot-to-segment assignment.

Uses a cost matrix (built by :mod:`cost`) and switching costs to find
the globally optimal sequence ``(robot_id, trajectory_idx)`` for every
trajectory segment.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.assignment.models import (
    TrajectoryBundle,
)
from robosuite_franka_manipulation.planning.assignment.result import (
    AssignmentResult,
)
from robosuite_franka_manipulation.planning.assignment.cost import (
    compute_cost_matrix,
    compute_switching_cost,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Viterbi DP solver (internal)
# ---------------------------------------------------------------------------

def _solve_optimal_assignment(
    cost_matrix: np.ndarray,
    trajectory_info: Dict[int, Tuple[int, int]],
    trajectories: TrajectoryBundle,
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    initial_robot: Optional[int] = None,
) -> AssignmentResult:
    """Solve the optimal assignment with a Viterbi-style DP.

    Enforces that within the same arm the trajectory index must stay constant
    (only arm-switches may change trajectory).  Switching costs are cached
    during the forward pass to avoid recomputation during backtracking.
    """
    n_segments, n_states = cost_matrix.shape

    J_star = np.full((n_segments, n_states), np.inf)
    prev_state = np.full((n_segments, n_states), -1, dtype=int)
    switch_cost_cache = np.zeros((n_segments, n_states))

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
            best_sw = 0.0

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
                    collision_check_fn=collision_check_fn,
                )
                if np.isinf(sw):
                    continue

                total = J_star[i - 1, ps] + sw
                if total < best_cost:
                    best_cost = total
                    best_prev = ps
                    best_sw = sw

            if best_prev != -1:
                J_star[i, cs] = best_cost + cost_matrix[i, cs]
                prev_state[i, cs] = best_prev
                switch_cost_cache[i, cs] = best_sw

    # Find best terminal state
    best_final_cost = np.inf
    best_final_state = -1
    for si in range(n_states):
        if J_star[n_segments - 1, si] < best_final_cost:
            best_final_cost = J_star[n_segments - 1, si]
            best_final_state = si

    if best_final_state == -1:
        logger.error("No feasible assignment found across %d states", n_states)
        raise ValueError("No feasible assignment found — check trajectory feasibility.")

    logger.debug(
        "Forward pass complete: best terminal cost=%.4f at state %d",
        best_final_cost, best_final_state,
    )

    # Backtrack using cached switching costs (append + reverse for O(n))
    sequence: List[int] = []
    traj_indices: List[int] = []
    task_costs_list: List[float] = []
    switch_costs_list: List[float] = []

    cs = best_final_state
    for i in range(n_segments - 1, -1, -1):
        robot_id, traj_idx = trajectory_info[cs]
        sequence.append(robot_id)
        traj_indices.append(traj_idx)
        task_costs_list.append(float(cost_matrix[i, cs]))

        if i > 0:
            switch_costs_list.append(float(switch_cost_cache[i, cs]))
            cs = prev_state[i, cs]
        else:
            switch_costs_list.append(0.0)

    sequence.reverse()
    traj_indices.reverse()
    task_costs_list.reverse()
    switch_costs_list.reverse()

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
    trajectories: TrajectoryBundle,
    transfer_feasibility: Optional[Dict[Tuple[int, int, int, int], bool]] = None,
    initial_robot: Optional[int] = None,
    alpha: float = constants.ASSIGNMENT_ALPHA,
    beta: float = constants.ASSIGNMENT_BETA,
    epsilon: float = constants.ASSIGNMENT_EPSILON,
    ik_solver_fn: Optional[Callable] = None,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
    collision_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
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
        jacobian_fns: ``{robot_id: (q) -> J}`` per-robot Jacobian functions
            for Yoshikawa manipulability.
        collision_check_fn: Optional ``(q_arm0, q_arm1) -> collision_free``
            for verifying inter-arm clearance at handover meeting points.

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
        collision_check_fn=collision_check_fn,
        initial_robot=initial_robot,
    )

    logger.info(
        "Assignment solved: total_cost=%.4f, handovers=%d",
        result.total_cost, len(result.handover_points),
    )
    return result
