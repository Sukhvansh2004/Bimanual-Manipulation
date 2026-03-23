"""Utility helpers for the trajectory-segment assignment planner.

Pure-function utilities extracted from the assignment subpackage so that
they can be reused and tested independently of the core DP solver and
cost-matrix construction logic.

Functions
---------
normalize_joint_angle
    Shift a single angle by 2π multiples to minimise distance to a
    reference while respecting joint limits.
normalize_trajectory_configs
    Normalize a full ``(N, DOF)`` trajectory and check step feasibility.
compute_smoothness
    Squared joint displacement along a config sequence.
compute_manipulability_cost
    Reciprocal manipulability cost for one arm's config sequence.
frame_id_at
    Extract the frame id from an object trajectory at a given index.
eef_matrix
    Build a 4×4 homogeneous matrix from a robot trajectory's EEF pose.
apply_beam_pruning
    Keep only the top-k states in a DP cost row; set the rest to inf.
save_assignment_result
    Persist an :class:`AssignmentResult` to an ``.npz`` file.
load_assignment_result
    Load an :class:`AssignmentResult` from an ``.npz`` file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.planning.ik.manipulability import (
    compute_trajectory_manipulability,
)

if TYPE_CHECKING:
    from bimanual_manipulation.planning.assignment.models import (
        ObjectTrajectory,
        RobotTrajectory,
    )
    from bimanual_manipulation.planning.assignment.result import AssignmentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint-angle normalisation
# ---------------------------------------------------------------------------

def normalize_joint_angle(
    q: float, q_ref: float, lo: float, hi: float,
) -> float:
    """Shift *q* by multiples of 2π to minimise ``|q - q_ref|`` within ``[lo, hi]``.

    Only integer multiples of 2π are tried so that the resulting
    configuration is physically equivalent.  If no shift improves the
    distance while respecting limits, *q* is returned unchanged.
    """
    best_q = q
    best_diff = abs(q - q_ref)
    for k in (-1, 1, -2, 2):
        q_shifted = q + k * 2.0 * np.pi
        if lo <= q_shifted <= hi:
            diff = abs(q_shifted - q_ref)
            if diff < best_diff:
                best_q = q_shifted
                best_diff = diff
    return best_q


def normalize_trajectory_configs(
    joint_configs: np.ndarray,
    is_feasible: np.ndarray,
    joint_limits_lo: np.ndarray = constants.FRANKA_JOINT_LIMITS_LO,
    joint_limits_hi: np.ndarray = constants.FRANKA_JOINT_LIMITS_HI,
    max_joint_step: float = constants.ASSIGNMENT_MAX_JOINT_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize consecutive joint configurations and check step feasibility.

    For each consecutive pair of feasible waypoints, each joint angle is
    shifted by an integer multiple of 2π to minimise the displacement
    while staying within joint position limits.  Any remaining per-joint
    displacement that exceeds *max_joint_step* after normalisation causes
    that transition to be flagged as infeasible.

    Args:
        joint_configs: ``(N, DOF)`` joint angles in radians.
        is_feasible: ``(N,)`` boolean feasibility per waypoint.
        joint_limits_lo: ``(DOF,)`` lower joint limits (defaults to Franka).
        joint_limits_hi: ``(DOF,)`` upper joint limits (defaults to Franka).
        max_joint_step: Maximum per-joint angle change (rad) between
            consecutive waypoints.  Defaults to
            :data:`constants.ASSIGNMENT_MAX_JOINT_STEP`.

    Returns:
        ``(normalized_configs, step_feasible)``

        * ``normalized_configs`` — ``(N, DOF)`` with adjusted angles.
        * ``step_feasible`` — ``(N,)`` boolean.  ``True`` at index *i*
          means the transition from waypoint *i − 1* to *i* is within the
          step threshold.  Index 0 is always ``True``.
    """
    N, DOF = joint_configs.shape
    normalized = joint_configs.copy()
    step_feasible = np.ones(N, dtype=bool)

    for i in range(1, N):
        if not is_feasible[i] or not is_feasible[i - 1]:
            continue
        feasible = True
        for j in range(DOF):
            normalized[i, j] = normalize_joint_angle(
                normalized[i, j], normalized[i - 1, j],
                float(joint_limits_lo[j]), float(joint_limits_hi[j]),
            )
            if abs(normalized[i, j] - normalized[i - 1, j]) > max_joint_step:
                feasible = False
        if not feasible:
            step_feasible[i] = False
            logger.debug(
                "Waypoint %d: joint step exceeds threshold after "
                "normalisation — marking transition infeasible",
                i,
            )

    return normalized, step_feasible


# ---------------------------------------------------------------------------
# Smoothness & manipulability helpers
# ---------------------------------------------------------------------------

def compute_smoothness(configs: List[np.ndarray]) -> float:
    """Squared joint displacement along a config sequence."""
    if len(configs) < 2:
        return 0.0
    arr = np.array(configs)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(diffs * diffs))


def compute_manipulability_cost(
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
# Trajectory accessors
# ---------------------------------------------------------------------------

def frame_id_at(obj_traj: ObjectTrajectory, idx: int) -> str:
    """Extract frame_id from object trajectory at *idx*."""
    fids = obj_traj.frame_ids
    if fids is None:
        return constants.WORLD_FRAME
    if isinstance(fids, np.ndarray):
        return str(fids[idx]) if idx < len(fids) else constants.WORLD_FRAME
    return str(fids)


def eef_matrix(traj: RobotTrajectory, idx: int) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from the EEF pose at *idx*."""
    mat = np.eye(4)
    mat[:3, 3] = traj.eef_positions[idx]
    mat[:3, :3] = R.from_quat(traj.eef_quaternions[idx]).as_matrix()
    return mat


# ---------------------------------------------------------------------------
# DP beam-pruning
# ---------------------------------------------------------------------------

def apply_beam_pruning(
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
# Assignment result NPZ persistence
# ---------------------------------------------------------------------------

def save_assignment_result(result: AssignmentResult, filepath: str) -> None:
    """Persist an :class:`AssignmentResult` to an ``.npz`` file."""
    np.savez(
        filepath,
        sequence=np.array(result.sequence),
        trajectory_paths=np.array(result.trajectory_paths, dtype=str),
        total_cost=result.total_cost,
        task_costs=result.task_costs,
        switching_costs=result.switching_costs,
        handover_points=np.array(result.handover_points),
        cost_matrix=result.cost_matrix,
        trajectory_matrix=result.trajectory_matrix,
        normalized_configs=result.normalized_configs,
        eef_positions=result.eef_positions,
        eef_quaternions=result.eef_quaternions,
        eef_frame_ids=np.array(result.eef_frame_ids, dtype=str),
        object_positions=result.object_positions,
        object_quaternions=result.object_quaternions,
        object_frame_ids=np.array(result.object_frame_ids, dtype=str),
        is_feasible=result.is_feasible,
        grasp_paths=np.array(result.grasp_paths, dtype=str),
        grasp_positions=result.grasp_positions,
        grasp_quaternions=result.grasp_quaternions,
        grasp_frame_ids=np.array(result.grasp_frame_ids, dtype=str),
        handover_transfer_configs=result.handover_transfer_configs,
    )
    logger.info("Assignment result saved to %s", filepath)


def load_assignment_result(filepath: str) -> AssignmentResult:
    """Load an :class:`AssignmentResult` from an ``.npz`` file."""
    from bimanual_manipulation.planning.assignment.result import (
        AssignmentResult as _AssignmentResult,
    )

    data = np.load(filepath, allow_pickle=True)

    return _AssignmentResult(
        sequence=data["sequence"].tolist(),
        trajectory_paths=data["trajectory_paths"].tolist(),
        total_cost=float(data["total_cost"]),
        task_costs=data["task_costs"],
        switching_costs=data["switching_costs"],
        handover_points=data["handover_points"].tolist(),
        cost_matrix=data["cost_matrix"],
        trajectory_matrix=data["trajectory_matrix"],
        normalized_configs=data["normalized_configs"],
        eef_positions=data["eef_positions"],
        eef_quaternions=data["eef_quaternions"],
        eef_frame_ids=data["eef_frame_ids"].tolist(),
        object_positions=data["object_positions"],
        object_quaternions=data["object_quaternions"],
        object_frame_ids=data["object_frame_ids"].tolist(),
        is_feasible=data["is_feasible"],
        grasp_paths=data["grasp_paths"].tolist(),
        grasp_positions=data["grasp_positions"],
        grasp_quaternions=data["grasp_quaternions"],
        grasp_frame_ids=data["grasp_frame_ids"].tolist(),
        handover_transfer_configs=data["handover_transfer_configs"],
    )
