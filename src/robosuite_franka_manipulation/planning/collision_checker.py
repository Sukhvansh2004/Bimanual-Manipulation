"""Collision and feasibility checker for grasps using the DualArmEnvironment.

Replaces synthetic geometric checks with actual MuJoCo physics and IK checks.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
import logging
from robosuite_franka_manipulation.planning.utils.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction
)
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    set_qpos,
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object
)
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation import constants
import mujoco
from contextlib import nullcontext
from scipy.spatial.transform import Rotation as R

# Configure logging
logger = logging.getLogger(__name__)


def check_single_grasp_feasibility(
    env: DualArmEnvironment,
    grasp: StampedPose,
    robot_id: int,
    tf_manager: TransformManager,
    arm: str = "right",
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    env_lock = None,
) -> bool:
    """
    Check if a single grasp is feasible (reachable via IK and collision-free).
    
    Args:
        env: DualArmEnvironment
        grasp: GraspSchema or StampedPose (must be in WORLD frame or convertible).
        robot_id: ID of robot (0 or 1)
        tf_manager: TransformManager
        arm: Arm side
        object_name: Name of object to ignore in collisions.
        ignore_names: List of geom name prefixes to ignore.
        env_lock: Optional threading.RLock
        
    Returns:
        True if feasible
    """
    # Prepare ignore names for collision checking
    current_ignore_names = list(ignore_names) if ignore_names else []
    if object_name and object_name not in current_ignore_names:
        current_ignore_names.append(object_name)

    # Prepare Target Pose
    target_pose = apply_ik_orientation_correction(grasp)

    # Get IK solutions for Flange site
    site_name = f"gripper{robot_id}_{arm}_ft_frame"
    
    solutions = enumerate_ik_solutions_for_arm(
        env,
        robot_id=robot_id,
        target_pose=target_pose,
        arm=arm,
        site_name=site_name,
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
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                
                if not in_collision(env, env_lock=None, ignore_names=current_ignore_names): # Already locked by ctx
                    feasible = True
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
        
    return feasible

def find_feasible_grasp_pairs(
    env: DualArmEnvironment,
    grasps_robot0: List[GraspSchema],
    grasps_robot1: List[GraspSchema],
    tf_manager: TransformManager,
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    env_lock = None,
) -> List[Tuple[GraspSchema, GraspSchema]]:
    """
    Find feasible pairs of grasps (one for each robot).
    
    Args:
        env: DualArmEnvironment
        grasps_robot0: List of grasps for robot 0
        grasps_robot1: List of grasps for robot 1
        tf_manager: TransformManager
        object_name: Name of object to ignore in collisions.
        ignore_names: List of geom name prefixes to ignore.
        env_lock: Optional threading.RLock
        
    Returns:
        List of (grasp0, grasp1) tuples that are simultaneously feasible.
    """
    feasible_pairs = []
    
    # Pre-filter single grasps to reduce N*M checks
    logger.info(f"Pre-filtering {len(grasps_robot0)} grasps for Robot 0...")
    valid_grasps0 = [g for g in grasps_robot0 if check_single_grasp_feasibility(
        env, g, 0, object_name=object_name, ignore_names=ignore_names, env_lock=env_lock, tf_manager=tf_manager
    )]
    logger.info(f"  -> {len(valid_grasps0)} valid grasps for Robot 0")
    
    logger.info(f"Pre-filtering {len(grasps_robot1)} grasps for Robot 1...")
    valid_grasps1 = [g for g in grasps_robot1 if check_single_grasp_feasibility(
        env, g, 1, object_name=object_name, ignore_names=ignore_names, env_lock=env_lock, tf_manager=tf_manager
    )]
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
            
            results = check_dual_arm_pose_feasibility(
                env, 
                eef_pose_robot0=g0,
                eef_pose_robot1=g1,
                object_name=object_name,
                ignore_names=ignore_names,
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


def check_dual_arm_pose_feasibility(
    env: DualArmEnvironment,
    eef_pose_robot0: StampedPose,
    eef_pose_robot1: StampedPose,
    tf_manager: TransformManager,
    *,
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    input_frame: str = constants.WORLD_FRAME,
    env_lock=None,
    arm: str = "right",
    # IK discretization / search params
    num_seed_configs: int = constants.IK_NUM_SEED_CONFIGS,
    elbow_discretization: int = constants.IK_ELBOW_DISCRETIZATION,
    max_solutions_per_arm: int = constants.IK_NUM_SEED_CONFIGS,
    # IK solver params
    pos_tol: float = constants.IK_POS_TOLERANCE,
    ori_tol: float = constants.IK_ORI_TOLERANCE,
    # Collision params
    contact_dist_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
    # Pairing caps
    max_pairs: int = constants.DUAL_ARM_MAX_PAIRS,
) -> List[List[Union[np.ndarray, bool]]]:
    f"""Check feasibility for *all* (discretized) dual-arm joint solutions at given EEF poses.

    This function:
    1) Runs IK for each arm from multiple seed configurations to obtain a set of
       joint solutions for robot0 and robot1.
    2) Forms joint-solution pairs (cartesian product) up to ``max_pairs``.
    3) For each pair, sets the MuJoCo qpos, runs forward kinematics, and checks
       for contacts involving either robot.

    Args:
        env: Robosuite environment (or wrapper) exposing ``env.sim.model`` and ``env.sim.data``.
        eef_pose_robot0: Desired EEF pose for robot0, in ``input_frame``.
        eef_pose_robot1: Desired EEF pose for robot1, in ``input_frame``.
        tf_manager: TransformManager for frame conversions.
        input_frame: Frame the input poses are expressed in (default: "world").
        env_lock: Optional threading.RLock used to guard sim state.

    Returns:
        A list of entries: ``[q_arm0 ({constants.FRANKA_ARM_DOF},), q_arm1 ({constants.FRANKA_ARM_DOF},), feasible_bool]``.

    Notes
    - If either arm has no IK solutions, returns an empty list.
    - Feasible means: IK converged for both arms AND no MuJoCo contact involving
      either robot is present at that configuration.
    """
    
    # Prepare ignore names for collision checking
    current_ignore_names = list(ignore_names) if ignore_names else []
    if object_name and object_name not in current_ignore_names:
        current_ignore_names.append(object_name)

    # Prepare targets
    target0 = apply_ik_orientation_correction(eef_pose_robot0)
    target1 = apply_ik_orientation_correction(eef_pose_robot1)

    q_solutions_0 = enumerate_ik_solutions_for_arm(
        env,
        robot_id=0,
        target_pose=target0,
        arm=arm,
        site_name=f"gripper0_{arm}_ft_frame", # Target Flange
        tf_manager=tf_manager,
        env_lock=env_lock,
        num_seed_configs=num_seed_configs,
        elbow_discretization=elbow_discretization,
        max_solutions=max_solutions_per_arm,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )

    q_solutions_1 = enumerate_ik_solutions_for_arm(
        env,
        robot_id=1,
        target_pose=target1,
        arm=arm,
        site_name=f"gripper1_{arm}_ft_frame", # Target Flange
        tf_manager=tf_manager,
        env_lock=env_lock,
        num_seed_configs=num_seed_configs,
        elbow_discretization=elbow_discretization,
        max_solutions=max_solutions_per_arm,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )

    if len(q_solutions_0) == 0 or len(q_solutions_1) == 0:
        return []

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        jids0 = get_arm_joint_ids(model, 0)
        jids1 = get_arm_joint_ids(model, 1)
        qpos_idx0 = get_joint_qpos_indices(model, jids0)
        qpos_idx1 = get_joint_qpos_indices(model, jids1)

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        results: List[List[Union[np.ndarray, bool]]] = []
        try:
            pair_count = 0
            for q0 in q_solutions_0:
                for q1 in q_solutions_1:
                    pair_count += 1
                    if pair_count > int(max_pairs):
                        break

                    data.qpos[:] = qpos_backup
                    data.qvel[:] = qvel_backup
                    set_qpos(data, qpos_idx0, np.asarray(q0, dtype=float))
                    set_qpos(data, qpos_idx1, np.asarray(q1, dtype=float))
                    
                    # Unwrap for native call
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))

                    feasible = not in_collision(
                        env, 
                        contact_dist_threshold=contact_dist_threshold, 
                        env_lock=None,
                        ignore_names=current_ignore_names
                    )
                    results.append([np.asarray(q0, dtype=float), np.asarray(q1, dtype=float), bool(feasible)])

                if pair_count > int(max_pairs):
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            
            # Unwrap for native call
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))

        return results