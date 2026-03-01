"""Planning utilities package."""

from robosuite_franka_manipulation.planning.utils.ik_utils import (
    enumerate_ik_solutions_for_arm,
    solve_ik_mink,
    unique_solutions
)
from robosuite_franka_manipulation.planning.utils.collision_utils import (
    in_collision,
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
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
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    save_trajectory,
    load_trajectory,
    WaypointFeasibilityResult,
    resample_joint_trajectory,
    quintic_scalar,
    plan_quintic_joint_trajectory,
)
from robosuite_franka_manipulation.planning.utils.manipulability import (
    compute_manipulability_simple,
    compute_manipulability_from_jacobian,
    compute_trajectory_manipulability,
    compute_average_manipulability,
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import (
    visualize_trajectory
)
from robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer import (
    RobotMeshVisualizer
)
from robosuite_franka_manipulation.planning.utils.object_mesh_visualizer import (
    ObjectMeshVisualizer
)
from robosuite_franka_manipulation.planning.utils.rrt_visualizer import (
    RRTVisualizer
)

__all__ = [
    # IK utilities
    'enumerate_ik_solutions_for_arm',
    'solve_ik_mink',
    'unique_solutions',
    # Collision utilities
    'in_collision',
    'get_geom_ids_by_prefix',
    'get_robot_geom_mask',
    # MuJoCo utilities
    'get_arm_joint_ids',
    'get_joint_qpos_indices',
    'get_joint_dof_indices',
    'get_joint_limits',
    'get_eef_site_name',
    'get_qpos',
    'set_qpos',
    'get_native_mj_object',
    'compute_site_error',
    'get_robot_collision_geoms',
    'get_all_dof_indices_except',
    'get_object_collision_geoms',
    'get_environment_collision_geoms',
    # Trajectory utilities
    'save_trajectory',
    'load_trajectory',
    'WaypointFeasibilityResult',
    'resample_joint_trajectory',
    'quintic_scalar',
    'plan_quintic_joint_trajectory',
    # Visualization utilities
    'visualize_trajectory',
    'RobotMeshVisualizer',
    'ObjectMeshVisualizer',
    'RRTVisualizer',
    # Manipulability utilities
    'compute_manipulability_simple',
    'compute_manipulability_from_jacobian',
    'compute_trajectory_manipulability',
    'compute_average_manipulability',
]
