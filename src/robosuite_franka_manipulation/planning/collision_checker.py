"""Collision and feasibility checker for grasps using the DualArmEnvironment.

Replaces synthetic geometric checks with actual MuJoCo physics and IK checks.
"""

import numpy as np
from typing import List, Tuple, Union
import logging
from robosuite_franka_manipulation.planning.utils.feasability_checks import check_dual_arm_pose_feasibility
from robosuite_franka_manipulation.planning.utils.ik_utils import enumerate_ik_solutions_for_arm
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    set_qpos,
    get_arm_joint_ids,
    get_joint_qpos_indices
)
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation import constants
import mujoco
from contextlib import nullcontext

# Configure logging
logger = logging.getLogger(__name__)

def _get_native_mj_object(obj):
    """Helper to unwrap Robosuite MjModel/MjData wrappers to native MuJoCo objects."""
    if hasattr(obj, '_data'):
        return obj._data
    if hasattr(obj, '_model'):
        return obj._model
    return obj

def check_single_grasp_feasibility(
    env: DualArmEnvironment,
    grasp: StampedPose,
    robot_id: int,
    arm: str = "right",
    env_lock = None,
    tf_manager: Union[TransformManager, None] = None
) -> bool:
    """
    Check if a single grasp is feasible (reachable via IK and collision-free).
    
    Args:
        env: DualArmEnvironment
        grasp: GraspSchema or StampedPose (must be in WORLD frame or convertible).
        robot_id: ID of robot (0 or 1)
        env_lock: Optional threading.RLock
        tf_manager: Optional TransformManager
        
    Returns:
        True if feasible
    """
    # GraspSchema/StampedPose from GraspGenerator are already in World Frame (or StampedPose handles frame)
    # enumerate_ik_solutions_for_arm handles StampedPose input.
    
    # Get IK solutions
    # Note: enumerate_ik_solutions_for_arm expects target_pose to be StampedPose or dict.
    # GraspSchema inherits from StampedPose, so it's valid.
    
    solutions = enumerate_ik_solutions_for_arm(
        env,
        robot_id=robot_id,
        target_pose=grasp,
        arm=arm,
        max_solutions=constants.DEFAULT_IK_MAX_SOLUTIONS,
        env_lock=env_lock,
        tf_manager=tf_manager
    )
    
    if not solutions:
        return False
        
    ctx = env_lock if env_lock is not None else nullcontext()
    
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids = get_arm_joint_ids(model, robot_id)
        qpos_idx = get_joint_qpos_indices(model, jids)
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        feasible = False
        try:
            for q in solutions:
                set_qpos(data, qpos_idx, q)
                mujoco.mj_forward(_get_native_mj_object(model), _get_native_mj_object(data))
                
                if not in_collision(env, env_lock=None): # Already locked by ctx
                    feasible = True
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(_get_native_mj_object(model), _get_native_mj_object(data))
        
    return feasible

def find_feasible_grasp_pairs(
    env: DualArmEnvironment,
    grasps_robot0: List[GraspSchema],
    grasps_robot1: List[GraspSchema],
    env_lock = None,
    tf_manager: Union[TransformManager, None] = None
) -> List[Tuple[GraspSchema, GraspSchema]]:
    """
    Find feasible pairs of grasps (one for each robot).
    
    Args:
        env: DualArmEnvironment
        grasps_robot0: List of grasps for robot 0
        grasps_robot1: List of grasps for robot 1
        env_lock: Optional threading.RLock
        tf_manager: Optional TransformManager
        
    Returns:
        List of (grasp0, grasp1) tuples that are simultaneously feasible.
    """
    feasible_pairs = []
    
    # Pre-filter single grasps to reduce N*M checks
    logger.info(f"Pre-filtering {len(grasps_robot0)} grasps for Robot 0...")
    valid_grasps0 = [g for g in grasps_robot0 if check_single_grasp_feasibility(env, g, 0, env_lock=env_lock, tf_manager=tf_manager)]
    logger.info(f"  -> {len(valid_grasps0)} valid grasps for Robot 0")
    
    logger.info(f"Pre-filtering {len(grasps_robot1)} grasps for Robot 1...")
    valid_grasps1 = [g for g in grasps_robot1 if check_single_grasp_feasibility(env, g, 1, env_lock=env_lock, tf_manager=tf_manager)]
    logger.info(f"  -> {len(valid_grasps1)} valid grasps for Robot 1")
    
    if not valid_grasps0 or not valid_grasps1:
        return []
        
    logger.info(f"Checking {len(valid_grasps0) * len(valid_grasps1)} pairs...")
    
    # Iterate pairs
    count = 0
    for g0 in valid_grasps0:
        for g1 in valid_grasps1:
            # Check pair feasibility
            # This runs IK for both and checks collision for pairs of solutions
            # check_dual_arm_pose_feasibility accepts StampedPose
            results = check_dual_arm_pose_feasibility(
                env, 
                eef_pose_robot0=g0,
                eef_pose_robot1=g1,
                max_pairs=constants.DEFAULT_FEASIBILITY_MAX_PAIRS,
                env_lock=env_lock,
                tf_manager=tf_manager
            )
            
            # results is list of [q0, q1, feasible]
            is_feasible = any(r[2] for r in results)
            
            if is_feasible:
                feasible_pairs.append((g0, g1))
            
            count += 1
            if count % 10 == 0:
                logger.debug(f"Processed {count} pairs...")
                
    logger.info(f"Found {len(feasible_pairs)} feasible pairs.")
    return feasible_pairs