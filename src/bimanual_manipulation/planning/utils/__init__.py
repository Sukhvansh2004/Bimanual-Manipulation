"""Planning utilities package.

Low-level shared utilities: IK, collision, MuJoCo helpers,
trajectory manipulation, and manipulability computation.

Visualization and RRT-related utilities have moved to their own
subpackages:

* :mod:`planning.rrt` — RRT planners, path smoothing, RRT visualizer
* :mod:`planning.visualization` — trajectory / mesh visualization
"""

from bimanual_manipulation.planning.utils.ik_utils import (
    enumerate_ik_solutions_for_arm,
)
from bimanual_manipulation.planning.utils.eaik_solver import (
    enumerate_ik_solutions_eaik,
    solve_ik_eaik,
    compute_arm_jacobian,
)
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
from bimanual_manipulation.planning.utils.manipulability import (
    compute_manipulability_simple,
    compute_manipulability_from_jacobian,
    compute_trajectory_manipulability,
    compute_average_manipulability,
)

__all__ = [
    # IK utilities
    "enumerate_ik_solutions_for_arm",
    "enumerate_ik_solutions_eaik",
    "solve_ik_eaik",
    "compute_arm_jacobian",
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
    # Manipulability utilities
    "compute_manipulability_simple",
    "compute_manipulability_from_jacobian",
    "compute_trajectory_manipulability",
    "compute_average_manipulability",
]
