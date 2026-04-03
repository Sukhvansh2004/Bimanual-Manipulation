"""Planning utilities package.

Low-level shared utilities: MuJoCo helpers, trajectory manipulation,
and assignment planning.
"""

from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_joint_dof_indices,
    get_joint_limits,
    get_eef_site_name,
    get_qpos,
    set_qpos,
    get_native_mj_object,
    compute_site_error,
    get_robot_collision_geoms,
    get_all_dof_indices_except,
    get_object_collision_geoms,
    get_environment_collision_geoms,
    get_frozen_dof_indices,
)
from bimanual_manipulation.planning.utils.trajectory_utils import (
    save_trajectory,
    load_trajectory,
    WaypointFeasibilityResult,
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

__all__ = [
    # MuJoCo utilities
    "get_arm_joint_ids",
    "get_joint_qpos_indices",
    "get_joint_dof_indices",
    "get_joint_limits",
    "get_eef_site_name",
    "get_qpos",
    "set_qpos",
    "get_native_mj_object",
    "compute_site_error",
    "get_robot_collision_geoms",
    "get_all_dof_indices_except",
    "get_object_collision_geoms",
    "get_environment_collision_geoms",
    "get_frozen_dof_indices",
    # Trajectory utilities
    "save_trajectory",
    "load_trajectory",
    "WaypointFeasibilityResult",
    # Assignment utilities
    "normalize_joint_angle",
    "normalize_trajectory_configs",
    "compute_smoothness",
    "compute_manipulability_cost",
    "frame_id_at",
    "eef_matrix",
    "apply_beam_pruning",
    "save_assignment_result",
    "load_assignment_result",
]
