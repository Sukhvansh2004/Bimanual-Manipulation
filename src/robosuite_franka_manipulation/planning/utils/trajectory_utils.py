"""
Trajectory verification utilities.
"""

import numpy as np
import logging
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass
import mujoco
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction
)
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    set_qpos,
    get_native_mj_object
)
from robosuite.utils import transform_utils as T
from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)

from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager, Transform


@dataclass
class WaypointFeasibilityResult:
    """Result of feasibility check for a single trajectory waypoint."""
    waypoint_index: int
    eef_pose: StampedPose
    joint_config: Optional[np.ndarray]  # None if IK failed
    is_feasible: bool  # True if IK succeeded and no collision
    ik_success: bool  # True if IK found a solution
    collision_free: bool  # True if no collision detected

def resolve_ignore_names(object_name: Optional[str], ignore_names: Optional[List[str]]) -> List[str]:
    """Helper to resolve ignore names list including object body geoms."""
    current_ignore_names = list(ignore_names) if ignore_names else []
    if object_name and object_name not in current_ignore_names:
        current_ignore_names.append(object_name)
        
    return current_ignore_names

def process_waypoint(
    env: DualArmEnvironment,
    robot_id: int,
    waypoint_index: int,
    pose: StampedPose,
    qpos_idx: np.ndarray,
    check_collisions: bool,
    ignore_names: List[str],
    last_config: Optional[np.ndarray],
    tf_manager: TransformManager,
    arm: str = "right"
) -> Tuple[WaypointFeasibilityResult, Optional[np.ndarray]]:
    """Process a single waypoint: IK + Collision Check."""
    
    # 1. Prepare Target Pose
    target_pose = apply_ik_orientation_correction(pose)
        
    site_name = f"gripper{robot_id}_{arm}_ft_frame"

    # 2. Solve IK
    solutions = enumerate_ik_solutions_for_arm(
        env=env,
        robot_id=robot_id,
        target_pose=target_pose,
        arm=arm,
        site_name=site_name,
        q_warm_start=last_config,
        num_seed_configs=constants.TRAJ_IK_NUM_SEED_CONFIGS,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS,
        env_lock=None, # Caller holds lock
        tf_manager=tf_manager
    )
    
    if not solutions:
        logger.debug(f"  WP {waypoint_index}: IK failed")
        return WaypointFeasibilityResult(
            waypoint_index=waypoint_index,
            eef_pose=pose,
            joint_config=None,
            is_feasible=False,
            ik_success=False,
            collision_free=True # Unknown
        ), None

    # 3. Check Collisions
    model = env.sim.model
    data = env.sim.data
    
    for q_sol in solutions:
        collision_free = True
        if check_collisions:
            set_qpos(data, qpos_idx, q_sol)
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
            collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names)
        
        if collision_free:
            return WaypointFeasibilityResult(
                waypoint_index=waypoint_index,
                eef_pose=pose,
                joint_config=q_sol,
                is_feasible=True,
                ik_success=True,
                collision_free=True
            ), q_sol # New last_config

    # If no collision-free solution found, return the first IK solution as failure
    logger.debug(f"  WP {waypoint_index}: IK succeeded but all in COLLISION")
    first_sol = solutions[0]
    return WaypointFeasibilityResult(
        waypoint_index=waypoint_index,
        eef_pose=pose,
        joint_config=first_sol,
        is_feasible=False,
        ik_success=True,
        collision_free=False
    ), first_sol # Return as potential warm start attempt


def process_dual_waypoint(
    env: DualArmEnvironment,
    waypoint_index: int,
    pose0: StampedPose,
    pose1: StampedPose,
    qpos_idx0: np.ndarray,
    qpos_idx1: np.ndarray,
    check_collisions: bool,
    ignore_names: List[str],
    last_config0: Optional[np.ndarray],
    last_config1: Optional[np.ndarray],
    tf_manager: TransformManager,
    arm: str = "right"
) -> Tuple[WaypointFeasibilityResult, WaypointFeasibilityResult, Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single dual-arm waypoint: IK for both + simultaneous Collision Check."""
    
    # 1. Prepare Target Poses
    target0 = apply_ik_orientation_correction(pose0)
    target1 = apply_ik_orientation_correction(pose1)
        
    site0 = f"gripper0_{arm}_ft_frame"
    site1 = f"gripper1_{arm}_ft_frame"

    # 2. Solve IK for both arms
    solutions0 = enumerate_ik_solutions_for_arm(
        env=env,
        robot_id=0,
        target_pose=target0,
        arm=arm,
        site_name=site0,
        q_warm_start=last_config0,
        num_seed_configs=constants.TRAJ_IK_NUM_SEED_CONFIGS,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS,
        env_lock=None, 
        tf_manager=tf_manager
    )
    
    solutions1 = enumerate_ik_solutions_for_arm(
        env=env,
        robot_id=1,
        target_pose=target1,
        arm=arm,
        site_name=site1,
        q_warm_start=last_config1,
        num_seed_configs=constants.TRAJ_IK_NUM_SEED_CONFIGS,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS,
        env_lock=None, 
        tf_manager=tf_manager
    )
    
    # Handle cases where one or both arms fail IK
    model = env.sim.model
    data = env.sim.data

    if not solutions0 or not solutions1:
        res0_ik = bool(solutions0)
        res1_ik = bool(solutions1)
        
        # Robot 0 check: Iterate all solutions against partner's last known config
        best_q0 = None
        res0_col = False
        if solutions0:
            for q0 in solutions0:
                if check_collisions:
                    set_qpos(data, qpos_idx0, q0)
                    if last_config1 is not None:
                        set_qpos(data, qpos_idx1, last_config1)
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                    if not in_collision(env, env_lock=None, ignore_names=ignore_names):
                        best_q0 = q0
                        res0_col = True
                        break
                else:
                    best_q0 = q0
                    res0_col = True
                    break
            if best_q0 is None:
                best_q0 = solutions0[0]
                res0_col = False

        # Robot 1 check: Iterate all solutions against partner's last known config
        best_q1 = None
        res1_col = False
        if solutions1:
            for q1 in solutions1:
                if check_collisions:
                    set_qpos(data, qpos_idx1, q1)
                    if last_config0 is not None:
                        set_qpos(data, qpos_idx0, last_config0)
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                    if not in_collision(env, env_lock=None, ignore_names=ignore_names):
                        best_q1 = q1
                        res1_col = True
                        break
                else:
                    best_q1 = q1
                    res1_col = True
                    break
            if best_q1 is None:
                best_q1 = solutions1[0]
                res1_col = False

        res0 = WaypointFeasibilityResult(
            waypoint_index=waypoint_index, eef_pose=pose0, joint_config=best_q0,
            is_feasible=(res0_ik and res0_col), ik_success=res0_ik, collision_free=res0_col
        )
        res1 = WaypointFeasibilityResult(
            waypoint_index=waypoint_index, eef_pose=pose1, joint_config=best_q1,
            is_feasible=(res1_ik and res1_col), ik_success=res1_ik, collision_free=res1_col
        )
        return res0, res1, best_q0, best_q1

    # 3. Check Collisions for pairs of solutions
    # To be efficient, we might not check all pairs if we find a feasible one
    pair_count = 0
    for q0 in solutions0:
        for q1 in solutions1:
            pair_count += 1
            collision_free = True
            if check_collisions:
                set_qpos(data, qpos_idx0, q0)
                set_qpos(data, qpos_idx1, q1)
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names)
            
            if collision_free:
                res0 = WaypointFeasibilityResult(
                    waypoint_index=waypoint_index, eef_pose=pose0, joint_config=q0,
                    is_feasible=True, ik_success=True, collision_free=True
                )
                res1 = WaypointFeasibilityResult(
                    waypoint_index=waypoint_index, eef_pose=pose1, joint_config=q1,
                    is_feasible=True, ik_success=True, collision_free=True
                )
                return res0, res1, q0, q1

    # If no collision-free pair found, return the first solution pair as failure
    logger.debug(f"  WP {waypoint_index}: Checked {pair_count} IK pairs, all in COLLISION")
    res0 = WaypointFeasibilityResult(
        waypoint_index=waypoint_index, eef_pose=pose0, joint_config=solutions0[0],
        is_feasible=False, ik_success=True, collision_free=False
    )
    res1 = WaypointFeasibilityResult(
        waypoint_index=waypoint_index, eef_pose=pose1, joint_config=solutions1[0],
        is_feasible=False, ik_success=True, collision_free=False
    )
    return res0, res1, solutions0[0], solutions1[0]


def save_trajectory(
    output_dir: str, 
    filename: str,
    eef_trajectory: Optional[List[StampedPose]] = None,
    object_trajectory: Optional[List[StampedPose]] = None,
    waypoint_results: Optional[List[WaypointFeasibilityResult]] = None,
) -> None:
    """Save trajectory to npz file with optional feasibility results.
    
    Args:
        output_dir: Output directory
        filename: Output filename
        eef_trajectory: List of end-effector StampedPose objects (optional if waypoint_results provided)
        object_trajectory: List of object StampedPose objects (optional)
        waypoint_results: List of WaypointFeasibilityResult with joint configs and feasibility (optional)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    data_dict = {}
    
    # Save EEF trajectory poses
    if eef_trajectory:
        data_dict['eef_positions'] = np.array([p.position for p in eef_trajectory])
        data_dict['eef_quaternions'] = np.array([p.quaternion for p in eef_trajectory])
        data_dict['eef_frame_ids'] = np.array([p.frame_id for p in eef_trajectory])
    
    # Save object trajectory poses
    if object_trajectory:
        data_dict['object_positions'] = np.array([p.position for p in object_trajectory])
        data_dict['object_quaternions'] = np.array([p.quaternion for p in object_trajectory])
        data_dict['object_frame_ids'] = np.array([p.frame_id for p in object_trajectory])
    
    # Save waypoint results with feasibility info
    if waypoint_results:
        # Feasibility flags (1 = feasible, 0 = infeasible)
        data_dict['is_feasible'] = np.array([int(r.is_feasible) for r in waypoint_results], dtype=np.int32)
        data_dict['ik_success'] = np.array([int(r.ik_success) for r in waypoint_results], dtype=np.int32)
        data_dict['collision_free'] = np.array([int(r.collision_free) for r in waypoint_results], dtype=np.int32)
        
        # Joint configurations (NaN for failed IK)
        joint_configs = []
        for r in waypoint_results:
            if r.joint_config is not None:
                joint_configs.append(r.joint_config)
            else:
                # Use NaN array for failed IK
                joint_configs.append(np.full(constants.FRANKA_ARM_DOF, np.nan))
        data_dict['joint_configs'] = np.array(joint_configs)
        
        # EEF poses from results (in case eef_trajectory not separately provided)
        if not eef_trajectory:
            data_dict['eef_positions'] = np.array([r.eef_pose.position for r in waypoint_results])
            data_dict['eef_quaternions'] = np.array([r.eef_pose.quaternion for r in waypoint_results])
            data_dict['eef_frame_ids'] = np.array([r.eef_pose.frame_id for r in waypoint_results])
    
    # Save as npz
    np.savez(filepath, **data_dict)
    logger.info(f"Trajectory saved to {filepath}")

def load_trajectory(
    filepath: str,
) -> Tuple[List[StampedPose], Optional[List[StampedPose]], Optional[np.ndarray], Optional[List[WaypointFeasibilityResult]]]:
    """Load trajectory from npz file.
    
    Args:
        filepath: Path to the npz file
        
    Returns:
        Tuple of (eef_trajectory, object_trajectory, joint_configs, waypoint_results)
        eef_trajectory: List of StampedPose
        object_trajectory: Optional list of StampedPose
        joint_configs: Optional np.ndarray of shape (N, DOF)
        waypoint_results: Optional List of WaypointFeasibilityResult
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
    data = np.load(filepath, allow_pickle=True)
    
    eef_trajectory = []
    if 'eef_positions' in data and 'eef_quaternions' in data:
        positions = data['eef_positions']
        quaternions = data['eef_quaternions']
        frame_ids = data['eef_frame_ids'] if 'eef_frame_ids' in data else [constants.WORLD_FRAME] * len(positions)
        
        for p, q, f in zip(positions, quaternions, frame_ids):
            eef_trajectory.append(StampedPose(position=p, quaternion=q, frame_id=str(f)))
            
    object_trajectory = []
    if 'object_positions' in data and 'object_quaternions' in data:
        positions = data['object_positions']
        quaternions = data['object_quaternions']
        frame_ids = data['object_frame_ids'] if 'object_frame_ids' in data else [constants.WORLD_FRAME] * len(positions)
        
        for p, q, f in zip(positions, quaternions, frame_ids):
            object_trajectory.append(StampedPose(position=p, quaternion=q, frame_id=str(f)))
    else:
        object_trajectory = None
        
    joint_configs = data['joint_configs'] if 'joint_configs' in data else None
    
    waypoint_results = None
    if 'is_feasible' in data:
        waypoint_results = []
        for i in range(len(eef_trajectory)):
            res = WaypointFeasibilityResult(
                waypoint_index=i,
                eef_pose=eef_trajectory[i],
                joint_config=joint_configs[i] if joint_configs is not None else None,
                is_feasible=bool(data['is_feasible'][i]),
                ik_success=bool(data['ik_success'][i]) if 'ik_success' in data else True,
                collision_free=bool(data['collision_free'][i]) if 'collision_free' in data else True
            )
            waypoint_results.append(res)
    
    return eef_trajectory, object_trajectory, joint_configs, waypoint_results
