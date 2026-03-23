"""Optimal trajectory-segment assignment planner (subpackage).

Finds the minimum-cost assignment of robot arms to object-trajectory
segments via a Viterbi-style DP, with smoothness + manipulability costs
and handover switching costs.

Submodules
----------
models
    :class:`ObjectTrajectory`, :class:`RobotTrajectory`, and
    :class:`TrajectoryBundle` typed trajectory containers.
result
    :class:`AssignmentResult` dataclass and NPZ persistence helpers.
cost
    Segment-cost and cost-matrix construction, handover-primitive
    evaluation, and switching-cost computation.
solver
    Viterbi DP solver and public :func:`compute_optimal_robot_assignment`.
factories
    Environment-bound closure factories for IK, Jacobian, and
    collision-check callables.
trajectory_io
    Disk I/O for robot/object trajectory NPZ files.
"""

from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)
from bimanual_manipulation.planning.assignment.result import (
    AssignmentResult,
)
from bimanual_manipulation.planning.assignment.cost import (
    compute_segment_cost,
    compute_cost_matrix,
    compute_switching_cost,
    HandoverResult,
)
from bimanual_manipulation.planning.utils.assignment_utils import (
    normalize_joint_angle,
    normalize_trajectory_configs,
    compute_smoothness,
    compute_manipulability_cost,
    frame_id_at,
    eef_matrix,
    apply_beam_pruning,
    save_assignment_result,
    load_assignment_result,
)
from bimanual_manipulation.planning.assignment.solver import (
    compute_optimal_robot_assignment,
)
from bimanual_manipulation.planning.assignment.factories import (
    make_ik_solver_fn,
    make_jacobian_fns,
    make_collision_check_fn,
    make_handover_clearance_fn,
)
from bimanual_manipulation.planning.assignment.trajectory_io import (
    load_all_trajectories,
)

__all__ = [
    "ObjectTrajectory",
    "RobotTrajectory",
    "TrajectoryBundle",
    "AssignmentResult",
    "save_assignment_result",
    "load_assignment_result",
    "compute_segment_cost",
    "compute_cost_matrix",
    "compute_switching_cost",
    "normalize_joint_angle",
    "normalize_trajectory_configs",
    "compute_smoothness",
    "compute_manipulability_cost",
    "frame_id_at",
    "eef_matrix",
    "apply_beam_pruning",
    "HandoverResult",
    "compute_optimal_robot_assignment",
    "make_ik_solver_fn",
    "make_jacobian_fns",
    "make_collision_check_fn",
    "make_handover_clearance_fn",
    "load_all_trajectories",
]
