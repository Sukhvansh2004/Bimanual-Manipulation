"""
Trajectory verification utilities.
"""

import numpy as np
import logging
from typing import List, Optional, Union
import mujoco

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils.ik_utils import enumerate_ik_solutions_for_arm
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    set_qpos,
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object
)
from robosuite.utils import transform_utils as T
from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)

def check_trajectory_feasibility(
    env: DualArmEnvironment,
    robot_id: int,
    eef_trajectory: List[np.ndarray],
    object_trajectory: Optional[List[np.ndarray]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None
) -> Optional[List[np.ndarray]]:
    """
    Check if an end-effector trajectory is feasible (IK reachable + Collision Free).
    
    Args:
        env: Environment instance
        robot_id: Robot ID (0 or 1)
        eef_trajectory: List of 4x4 matrices representing EEF poses in World frame.
        object_trajectory: Optional list of 4x4 matrices for object poses.
        object_name: Name of the object (required if object_trajectory is provided).
        check_collisions: Whether to check collisions.
        env_lock: Optional threading.RLock
        
    Returns:
        List of joint configurations (qpos) if feasible, None otherwise.
    """
    if object_trajectory and len(object_trajectory) != len(eef_trajectory):
        logger.error("Object trajectory length mismatch")
        return None

    # Locking context
    from contextlib import nullcontext
    ctx = env_lock if env_lock is not None else nullcontext()

    # We need to lock while accessing/modifying sim state
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        # Get joint indices for this robot
        jids = get_arm_joint_ids(model, robot_id)
        qpos_idx = get_joint_qpos_indices(model, jids)
        
        # Backup state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        # Object joint index if needed
        obj_joint_addr = None
        if object_name:
            try:
                # Assuming object is a free joint
                obj_joint_id = model.joint(f"{object_name}_joint0").id
                obj_qpos_addr = model.jnt_qposadr[obj_joint_id]
                obj_joint_addr = obj_qpos_addr
            except Exception:
                pass

        joint_trajectory = []
        
        try:
            for i, pose_mat in enumerate(eef_trajectory):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose[:3, 3]
                    # Robosuite T.mat2quat returns w,x,y,z. MuJoCo free joint expects pos, quat(w,x,y,z).
                    quat = T.mat2quat(obj_pose[:3, :3]) 
                    
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Solve IK
                pos = pose_mat[:3, 3]
                quat_wxyz = T.mat2quat(pose_mat[:3, :3])
                quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
                
                # Pass env_lock=None because we are already holding the lock via `with ctx:`
                # BUT wait, does enumerate_ik re-acquire? RLock allows re-acquire.
                # Ideally we pass `env_lock` to it if it expects it to be released or if it uses it.
                # ik_utils uses `with env_lock:` internally. So passing it is fine (re-entrant).
                
                solutions = enumerate_ik_solutions_for_arm(
                    env=env,
                    robot_id=robot_id,
                    target_pose=(pos, quat_xyzw),
                    target_frame="world",
                    arm="right",
                    num_seed_configs=8,
                    max_solutions=1,
                    env_lock=env_lock 
                )
                
                if not solutions:
                    return None
                
                q_sol = solutions[0]
                
                # 3. Check Collision
                if check_collisions:
                    set_qpos(data, qpos_idx, q_sol)
                    mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
                    
                    if in_collision(env, env_lock=None): # Already locked
                        return None
                
                joint_trajectory.append(q_sol)
                
            return joint_trajectory
            
        finally:
            # Restore state
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
