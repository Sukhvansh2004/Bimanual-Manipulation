"""Planning utilities package.

Low-level shared utilities: collision, MuJoCo helpers,
and trajectory manipulation.

IK solvers and manipulability utilities have moved to
:mod:`planning.ik`.

Visualization and RRT-related utilities have moved to their own
subpackages:

* :mod:`planning.rrt` — RRT planners, path smoothing, RRT visualizer
* :mod:`planning.visualization` — trajectory / mesh visualization
"""

from bimanual_manipulation.planning.utils.collision_utils import (
    in_collision,
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
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
from bimanual_manipulation.utils.trajectory_math import (
    resample_joint_trajectory,
    quintic_scalar,
    plan_quintic_joint_trajectory,
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
    # Collision utilities
    "in_collision",
    "get_geom_ids_by_prefix",
    "get_robot_geom_mask",
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
    "resample_joint_trajectory",
    "quintic_scalar",
    "plan_quintic_joint_trajectory",
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
