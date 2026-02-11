#!/usr/bin/env python3
"""
Dual-arm planning script for shelf manipulation task.

This script:
1. Initializes the DualArmEnvironment with a shelf and a plate.
2. Generates grasps for both robots and pairs them (N x M).
3. Evaluates simultaneous feasibility for each grasp pair using multiprocessing.
4. Saves feasible dual-arm trajectories to disk.
"""

import numpy as np
import time
import threading
import logging
import meshcat
import meshcat.geometry as g
import argparse
import sys
import os
import multiprocessing
from functools import partial

import meshcat.transformations as tf
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.utils import create_ycb_object
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose

# Services and Planning
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.services_core import ServiceManager
from robosuite_franka_manipulation.planning.grasp_generator import GraspGenerator

# Planning Modules
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_end_effector_trajectory, 
    generate_object_trajectory,
    check_dual_trajectory_feasibility
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import save_trajectory
from robosuite_franka_manipulation.planning.utils import (
    visualize_trajectory,
    RobotMeshVisualizer
)

def setup_logging(level: int = logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

logger = logging.getLogger(__name__)

def evaluate_pair(pair_info, ycb_objects, object_trajectory):
    """
    Worker function to evaluate a single pair of grasps.
    Each worker creates its own environment instance.
    """
    i, j, g0, g1 = pair_info
    
    # Create local env for this process
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    
    tf_manager = TransformManager(env)
    
    # Generate EEF trajectories
    traj0_eef = generate_end_effector_trajectory(object_trajectory, g0, tf_manager=tf_manager)
    traj1_eef = generate_end_effector_trajectory(object_trajectory, g1, tf_manager=tf_manager)
    
    # Check simultaneous feasibility
    results0, results1, all_feasible = check_dual_trajectory_feasibility(
        env=env,
        eef_trajectory0=traj0_eef,
        eef_trajectory1=traj1_eef,
        object_trajectory=object_trajectory,
        object_name="plate_target",
        tf_manager=tf_manager
    )
    
    env.close()
    
    return {
        'pair_idx': (i, j),
        'traj0_eef': traj0_eef,
        'traj1_eef': traj1_eef,
        'results0': results0,
        'results1': results1,
        'all_feasible': all_feasible,
        'g0': g0,
        'g1': g1
    }

def main() -> None:
    # Default values
    default_target_pos = [-0.55, 0.05, 1.35]
    default_target_quat = [0.70710678, 0.0, 0.0, 0.70710678]
    
    parser = argparse.ArgumentParser(description="Generate and visualize dual-arm shelf manipulation trajectory")
    parser.add_argument("--save-traj", action="store_true", help="Automatically save trajectory")
    parser.add_argument("--output-dir", type=str, default=constants.DATA_DIR, help="Output directory for data")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count() // 2, help="Number of parallel workers")
    
    # Pose arguments
    parser.add_argument("--target-pos", type=float, nargs=3, default=default_target_pos, help="Target position [x, y, z]")
    parser.add_argument("--target-quat", type=float, nargs=4, default=default_target_quat, help="Target quaternion [x, y, z, w]")
    
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger.info("Initializing Dual Trajectory Generation...")
    
    # 1. Setup Environment and Planning Context
    # ----------------------------------------
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678]
    plate_pos = [0.0, -0.1, 0.85]
    plate_quat = [0, 0, 0, 1]
    
    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat),
        create_ycb_object(ycb_name="plate", instance_name="plate_target", position=plate_pos, quaternion=plate_quat)
    ]
    
    # Create main env for grasp generation and object trajectory planning
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    env_lock = threading.RLock()
    tf_manager = TransformManager(env, env_lock=env_lock)
    
    # 2. Determine Start and Target Poses
    # -----------------------------------
    plate_pose_mat = env.get_object_pose("plate_target")
    start_pos = plate_pose_mat[:3, 3]
    start_quat = R.from_matrix(plate_pose_mat[:3, :3]).as_quat()
    start_pose_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME)
    
    target_pose_stamped = StampedPose(position=np.array(args.target_pos), quaternion=np.array(args.target_quat), frame_id=constants.WORLD_FRAME)
    
    # 3. Generate Object Trajectory
    # -----------------------------
    object_trajectory = generate_object_trajectory(
        start_pose_stamped, 
        target_pose_stamped, 
        num_steps=constants.OBJECT_TRAJ_NUM_STEPS,
        tf_manager=tf_manager
    )
    
    # 4. Generate Grasps
    # ------------------
    logger.info("Generating grasps...")
    service_manager = ServiceManager()
    service_manager.start()
    grasp_service = GraspService(tf_manager=tf_manager)
    grasp_service.start()
    grasp_generator = GraspGenerator(env=env)
    
    all_grasps = grasp_generator.generate_grasps(
        object_name="plate_target", 
        gripper_types=["finger", "suction"], 
        num_grasps=40,
        env_lock=env_lock
    )
    
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)[:4]
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)[:4]
    
    logger.info(f"Using {len(grasps_0)} Finger grasps and {len(grasps_1)} Suction grasps.")
    
    # 5. Evaluate Pairs in Parallel
    # -----------------------------
    grasp_pairs = []
    for i, g0 in enumerate(grasps_0):
        for j, g1 in enumerate(grasps_1):
            grasp_pairs.append((i, j, g0, g1))
            
    logger.info(f"Evaluating {len(grasp_pairs)} pairs using {args.num_workers} workers...")
    
    # Use multiprocessing pool
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        eval_func = partial(evaluate_pair, ycb_objects=ycb_objects, object_trajectory=object_trajectory)
        results = pool.map(eval_func, grasp_pairs)
    
    # 6. Save Feasible Trajectories
    # -----------------------------
    if args.save_traj or input("Save feasible trajectories? [y/N]: ").strip().lower() in ('y', 'yes'):
        # Save object trajectory
        save_trajectory(args.output_dir, "object_trajectory.npz", object_trajectory=object_trajectory)
        
        for idx, res in enumerate(results):
            pair_id = f"{res['pair_idx'][0]}_{res['pair_idx'][1]}"
            # Save robot 0
            save_trajectory(
                args.output_dir, f"dual_robot0_pair_{pair_id}.npz",
                eef_trajectory=res['traj0_eef'], object_trajectory=object_trajectory, waypoint_results=res['results0']
            )
            # Save robot 1
            save_trajectory(
                args.output_dir, f"dual_robot1_pair_{pair_id}.npz",
                eef_trajectory=res['traj1_eef'], object_trajectory=object_trajectory, waypoint_results=res['results1']
            )
            
        logger.info(f"Saved {len(results)} dual-arm trajectory pairs to {args.output_dir}")

    # Cleanup
    grasp_service.stop()
    service_manager.stop()
    env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
