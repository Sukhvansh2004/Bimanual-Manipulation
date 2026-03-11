"""
Utilities module for bimanual_manipulation.

This module contains reusable utility functions organized by domain:
- se3: SE(3) Lie algebra utilities for trajectory planning
- pointcloud: Point cloud processing utilities
- env_utils: Environment helpers (stepping, YCB objects, primitives)
- mesh_utils: Mesh loading and surface sampling
- primitives: High-level manipulation primitives (handover, grasp, release)
- suction_weld: Suction gripper weld constraints
- meshcat_mj_utils: Meshcat ↔ MuJoCo bridge (not re-exported)
"""

from bimanual_manipulation.utils.se3 import (
    # Skew-symmetric utilities
    skew,
    unskew,
    # se(3) Lie algebra utilities  
    se3_to_matrix,
    matrix_to_se3,
    # SE(3) Lie group utilities
    SE3_log,
    SE3_exp,
    # Pose conversions
    pose_to_SE3,
    SE3_to_pose,
    # Distance metrics
    compute_se3_distance,
    # Interpolation
    interpolate_SE3,
)

from bimanual_manipulation.utils.pointcloud import (
    downsample_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    compute_point_cloud_bounds,
    filter_point_cloud_by_range,
    validate_point_cloud,
    deproject_depth_to_point_cloud,
    parse_camera_intrinsics,
    ensure_depth_2d,
)

from bimanual_manipulation.utils.env_utils import (
    env_step,
    set_robot_config,
    register_mesh_extraction_env,
    reset_action_persistence,
    get_action_persistence_manager,
    ActionPersistenceManager,
    # Collision monitoring
    start_collision_monitor,
    stop_collision_monitor,
    get_collision_count,
    # YCB utilities
    get_available_ycb_objects,
    list_ycb_objects,
    resolve_mesh_path,
    # Object specification helpers
    create_ycb_object,
    create_primitive_object,
)

from bimanual_manipulation.utils.mesh_utils import load_mesh_and_sample

from bimanual_manipulation.utils.primitives import (
    DualArmHandoverPlanner,
    ObjectGraspPrimitive,
    ReleaseRetreatPrimitive,
    compute_ee_pose,
    extract_grasp_transform,
)

from bimanual_manipulation.utils.suction_weld import (
    apply_suction_weld,
    clear_suction_cache,
)

from bimanual_manipulation.utils.trajectory_math import (
    plan_quintic_joint_trajectory,
    quintic_scalar,
    resample_joint_trajectory,
)

__all__ = [
    # SE(3) utilities
    'skew',
    'unskew',
    'se3_to_matrix',
    'matrix_to_se3',
    'SE3_log',
    'SE3_exp',
    'pose_to_SE3',
    'SE3_to_pose',
    'compute_se3_distance',
    'interpolate_SE3',
    # Point cloud utilities
    'downsample_point_cloud',
    'estimate_pose_from_point_cloud',
    'transform_point_cloud',
    'compute_point_cloud_bounds',
    'filter_point_cloud_by_range',
    'validate_point_cloud',
    'deproject_depth_to_point_cloud',
    'parse_camera_intrinsics',
    'ensure_depth_2d',
    # Env utils
    'env_step',
    'set_robot_config',
    'register_mesh_extraction_env',
    'reset_action_persistence',
    'get_action_persistence_manager',
    'ActionPersistenceManager',
    'start_collision_monitor',
    'stop_collision_monitor',
    'get_collision_count',
    # YCB utilities
    'get_available_ycb_objects',
    'list_ycb_objects',
    'resolve_mesh_path',
    # Object specification helpers
    'create_ycb_object',
    'create_primitive_object',
    # Mesh utils
    'load_mesh_and_sample',
    # Motion primitives
    'DualArmHandoverPlanner',
    'ObjectGraspPrimitive',
    'ReleaseRetreatPrimitive',
    'compute_ee_pose',
    'extract_grasp_transform',
    # Suction weld utilities
    'apply_suction_weld',
    'clear_suction_cache',
    # Trajectory math
    'plan_quintic_joint_trajectory',
    'quintic_scalar',
    'resample_joint_trajectory',
]
