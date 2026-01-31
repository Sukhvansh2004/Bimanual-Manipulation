#!/usr/bin/env python3
"""
Planning script for shelf manipulation task.

This script:
1. Initializes the DualArmEnvironment with a shelf and a plate.
2. Plans a simple trajectory for the plate from the table to a rack slot in the shelf.
3. Visualizes the planning context and the generated trajectory using Meshcat.
4. Generates grasps for the object and computes/filters feasible EEF trajectories.
5. Saves the object and feasible robot trajectories to disk.
"""

import numpy as np
import time
import threading
import logging
import meshcat
import meshcat.geometry as g
import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict

import meshcat.transformations as tf
from grasp_gen.utils.meshcat_utils import make_frame

import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.utils import create_ycb_object
from robosuite_franka_manipulation.utils.se3 import interpolate_pose, pose_to_SE3, SE3_to_pose
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose

# Services and Planning
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.services_core import ServiceManager
from robosuite_franka_manipulation.planning.grasp_generator import GraspGenerator

# NEW: Planning Modules
from robosuite_franka_manipulation.planning.collision_checker import find_feasible_grasp_pairs
from robosuite_franka_manipulation.planning.trajectory_generator import generate_end_effector_trajectory
from robosuite_franka_manipulation.planning.utils.trajectory_utils import check_trajectory_feasibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

def generate_trajectory(
    start_pose: StampedPose, 
    target_pose: StampedPose, 
    num_steps: int = 50, 
    tf_manager: Optional[TransformManager] = None
) -> List[StampedPose]:
    """Generate a trajectory using SE(3) interpolation through waypoints.
    
    Args:
        start_pose: Start pose as StampedPose
        target_pose: Target pose as StampedPose
        num_steps: Number of steps for the trajectory
        tf_manager: Optional TransformManager for frame conversion
        
    Returns:
        List of StampedPose objects representing the trajectory
    """
    # Normalize frames to world if possible
    if tf_manager:
        if start_pose.frame_id != "world":
            start_pose = tf_manager.transform_pose(start_pose, "world")
        if target_pose.frame_id != "world":
            target_pose = tf_manager.transform_pose(target_pose, "world")
            
    # If still mismatch (and no manager), warn
    if start_pose.frame_id != target_pose.frame_id:
        logger.warning(f"Frame mismatch in generate_trajectory: {start_pose.frame_id} vs {target_pose.frame_id}")
        
    frame_id = start_pose.frame_id # Use common frame
    
    trajectory_points = []
    
    start_pos = start_pose.position
    start_quat = start_pose.quaternion
    
    end_pos = target_pose.position
    end_quat = target_pose.quaternion
    
    # Define waypoints in SE(3)
    # 1. Lift up (same orientation as start)
    lift_height = 0.2
    lift_pos = start_pos.copy()
    lift_pos[2] += lift_height
    lift_quat = start_quat.copy()
    
    # 2. Approach shelf (pre-align orientation with target)
    # Move to approach position, and start rotating towards target orientation
    approach_pos = end_pos.copy()
    approach_pos[0] += 0.3 # Stand back from shelf
    approach_quat = end_quat.copy() 
    
    # Define segments as (start_p, start_q, end_p, end_q, steps)
    segments = [
        (start_pos, start_quat, lift_pos, lift_quat, int(num_steps * 0.2)),
        (lift_pos, lift_quat, approach_pos, approach_quat, int(num_steps * 0.5)),
        (approach_pos, approach_quat, end_pos, end_quat, int(num_steps * 0.3))
    ]
    
    for p_start, q_start, p_end, q_end, steps in segments:
        for i in range(steps):
            t = i / float(steps)
            # Use SE(3) interpolation utility
            pos, quat = interpolate_pose(p_start, q_start, p_end, q_end, t)
            trajectory_points.append((pos, quat))
            
    # Ensure final point is included
    trajectory_points.append((end_pos, end_quat))
    
    # Convert to StampedPose list
    stamped_trajectory = []
    for pos, quat in trajectory_points:
        stamped_trajectory.append(StampedPose(
            position=pos,
            quaternion=quat,
            frame_id=frame_id
        ))
            
    return stamped_trajectory

def visualize_trajectory(vis: meshcat.Visualizer, trajectory: List[StampedPose], name: str = "trajectory", color: int = 0x00ff00) -> None:
    """Visualize trajectory as a line/points in Meshcat.
    
    Args:
        vis: Meshcat visualizer instance
        trajectory: List of StampedPose objects
        name: Name for the visualization group
        color: Color of the trajectory line
    """
    points = np.array([p.position for p in trajectory]).T
    
    # Draw line
    vis[name].set_object(g.Line(
        g.PointsGeometry(points),
        g.LineBasicMaterial(linewidth=5, color=color)
    ))
    
    # Draw key frames
    for i, pose in enumerate(trajectory):
        if i % 5 == 0: # Draw every 5th frame
            frame_name = f"{name}/frame_{i}"
            
            # Construct transform matrix
            mat = np.eye(4)
            mat[:3, 3] = pose.position
            # Convert quat [x,y,z,w] to matrix
            # Use T.convert_quat to handle xyzw -> wxyz for meshcat
            quat_wxyz = T.convert_quat(pose.quaternion, to="wxyz")
            r = tf.quaternion_matrix(quat_wxyz) 
            mat[:3, :3] = r[:3, :3]
            
            # Use make_frame from grasp_gen to draw axis
            make_frame(vis, frame_name, h=0.05, radius=0.002, o=1.0)
            vis[frame_name].set_transform(mat)

def save_trajectory(trajectory: List[StampedPose], output_dir: str, filename: str) -> None:
    """Save trajectory to npz file.
    
    Args:
        trajectory: List of StampedPose objects
        output_dir: Output directory
        filename: Output filename
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    # Extract data from StampedPose objects
    positions = np.array([p.position for p in trajectory])
    quaternions = np.array([p.quaternion for p in trajectory])
    frame_ids = np.array([p.frame_id for p in trajectory])
    
    # Save as npz
    np.savez(
        filepath,
        positions=positions,
        quaternions=quaternions,
        frame_ids=frame_ids
    )
    logger.info(f"Trajectory saved to {filepath}")

def main() -> None:
    # Default values
    default_target_pos = [-0.65, 0.05, 1.35]
    # Target Orientation: Vertical for slot
    # Rotate 90 deg around Y so it stands on edge facing X.
    # Quat 90 deg Y: [0, sin(45), 0, cos(45)] -> [0, 0.707, 0, 0.707] (xyzw)
    default_target_quat = [0.0, 0.70710678, 0.0, 0.70710678]
    
    parser = argparse.ArgumentParser(description="Generate and visualize shelf manipulation trajectory")
    parser.add_argument("--save-traj", action="store_true", help="Automatically save trajectory")
    parser.add_argument("--output-dir", type=str, default=constants.DATA_DIR, help="Output directory for data")
    parser.add_argument("--filename", type=str, default="shelf_trajectory.npz", help="Output filename")
    
    # Pose arguments
    parser.add_argument("--start-pos", type=float, nargs=3, help="Start position [x, y, z]")
    parser.add_argument("--start-quat", type=float, nargs=4, help="Start quaternion [x, y, z, w]")
    parser.add_argument("--target-pos", type=float, nargs=3, default=default_target_pos, help="Target position [x, y, z]")
    parser.add_argument("--target-quat", type=float, nargs=4, default=default_target_quat, help="Target quaternion [x, y, z, w]")
    
    args = parser.parse_args()

    logger.info("Initializing Trajectory Generation...")
    
    # 1. Setup Environment
    # --------------------
    # Shelf configuration
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678] # 90 deg Z
    
    # Plate configuration (on table)
    # Default plate pos if not overridden
    plate_pos = [0.0, -0.1, 0.85]
    plate_quat = [0, 0, 0, 1]
    
    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat),
        create_ycb_object(ycb_name="plate", instance_name="plate_target", position=plate_pos, quaternion=plate_quat)
    ]
    
    # Create env (headless for planning script, visualization handled by Meshcat)
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    
    # Lock for thread safety
    env_lock = threading.RLock()
    
    # Create TransformManager for pose queries
    tf_manager = TransformManager(env, env_lock=env_lock)
    
    # 2. Setup Visualization
    # ----------------------
    vis_config = MeshcatTFVisualizerConfig(
        fps=20.0, 
        clear=True, 
        labels=True, 
        visualize_scene=True # Visualize robot/scene meshes
    )
    
    # Start TF streaming in background
    stop_event = threading.Event()
    vis_thread = threading.Thread(
        target=stream_transforms_to_meshcat,
        kwargs={
            'env': env,
            'env_lock': env_lock,
            'auto_discover': True,
            'config': vis_config,
            'stop_event': stop_event,
            'tf_manager': tf_manager
        },
        daemon=True
    )
    vis_thread.start()
    
    # Connect to the same meshcat server for custom visualization
    vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
    
    logger.info("Visualization started. Open http://127.0.0.1:7000")
    
    # 3. Define Planning Problem
    # --------------------------
    
    # Determine Start Pose
    start_pos = None
    start_quat = None
    
    if args.start_pos and args.start_quat:
        start_pos = np.array(args.start_pos)
        start_quat = np.array(args.start_quat)
        logger.info(f"Using provided start pose: pos={start_pos}")
    else:
        logger.info("No start pose provided. Waiting for environment to settle...")
        time.sleep(1.0) # Wait for simulation/TF to stabilize
        
        # Query pose of plate_target
        try:
            plate_pose_mat = env.get_object_pose("plate_target")
            if plate_pose_mat is not None:
                start_pos = plate_pose_mat[:3, 3]
                # Convert matrix to quat [x, y, z, w]
                import scipy.spatial.transform as strans
                start_quat = strans.Rotation.from_matrix(plate_pose_mat[:3, :3]).as_quat()
                logger.info(f"Detected plate pose: pos={start_pos}")
            else:
                logger.warning("Could not find 'plate_target' in environment. Using default.")
        except Exception as e:
            logger.warning(f"Error querying object pose: {e}")
            
        if start_pos is None:
            start_pos = np.array(plate_pos)
            start_quat = np.array(plate_quat)
    
    start_pose_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id="world")
    
    # Determine Target Pose
    target_pos = np.array(args.target_pos)
    target_quat = np.array(args.target_quat)
    target_pose_stamped = StampedPose(position=target_pos, quaternion=target_quat, frame_id="world")
    
    logger.info(f"Planning from {start_pose_stamped.position} to {target_pose_stamped.position}")
    
    # 4. Generate Object Trajectory
    # ----------------------
    object_trajectory = generate_trajectory(
        start_pose_stamped, 
        target_pose_stamped, 
        num_steps=100,
        tf_manager=tf_manager
    )
    
    # 5. Visualize Object Trajectory
    # -----------------------
    visualize_trajectory(vis, object_trajectory, name="object_path", color=0x00ff00)
    
    # Visualize Target
    vis["target_pose"].set_object(g.Sphere(0.05), g.MeshBasicMaterial(color=0xff0000, opacity=0.5))
    T_target = np.eye(4)
    T_target[:3, 3] = target_pose_stamped.position
    
    # Convert target_quat (xyzw) to wxyz for meshcat
    target_quat_wxyz = T.convert_quat(target_pose_stamped.quaternion, to="wxyz")
    r = tf.quaternion_matrix(target_quat_wxyz)
    T_target[:3, :3] = r[:3, :3]
    vis["target_pose"].set_transform(T_target)
    
    logger.info("Object trajectory generated and visualized.")
    
    # 6. Generate Grasps and Feasible EEF Trajectories
    # ------------------------------------------------
    logger.info("Initializing Service Architecture...")
    
    # Initialize Service Manager
    service_manager = ServiceManager()
    service_manager.start()
    
    # Initialize Grasp Service (it registers itself)
    grasp_service = GraspService(
        env=env,
        env_lock=env_lock,
        tf_manager=tf_manager,
        visualize=False 
    )
    grasp_service.start()
    
    # Initialize Grasp Generator
    grasp_generator = GraspGenerator(env=env)
    
    # Step 6a: Get grasps for both robots together
    logger.info("Generating grasps for Robot 0 (Finger) and Robot 1 (Suction)...")
    all_grasps = grasp_generator.generate_grasps(
        object_name="plate_target", 
        gripper_types=["finger", "suction"], 
        num_grasps=40,
        env_lock=env_lock
    )
    
    grasps_0 = [g for g in all_grasps if g.gripper_type == 'finger']
    grasps_1 = [g for g in all_grasps if g.gripper_type == 'suction']
    
    logger.info(f"Received {len(grasps_0)} grasps for Finger and {len(grasps_1)} for Suction.")

    if not grasps_0 or not grasps_1:
        logger.error("Failed to generate grasps for one or both robots.")
    else:
        # Step 6b: Find feasible grasp pairs (Collision Free Grasp Configuration)
        logger.info("Finding feasible grasp pairs (checking mutual collisions)...")
        feasible_pairs = find_feasible_grasp_pairs(env, grasps_0, grasps_1, env_lock=env_lock)
        
        logger.info(f"Found {len(feasible_pairs)} feasible pairs.")
        
        # Step 6c: Generate and Check Trajectories for each pair
        valid_trajectories = [] # List of tuples (traj0, traj1)
        
        for i, (g0, g1) in enumerate(feasible_pairs):
            logger.info(f"Checking pair {i}...")
            
            # Generate EEF trajectories
            # PASS tf_manager to check/fix frames
            traj0_eef = generate_end_effector_trajectory(object_trajectory, g0, tf_manager=tf_manager)
            traj1_eef = generate_end_effector_trajectory(object_trajectory, g1, tf_manager=tf_manager)
            
            # Check Feasibility for Robot 0
            # Note: We check object collision by passing object_trajectory and object_name
            # This ensures the object doesn't collide with environment while moving
            logger.info("  Checking Robot 0 trajectory...")
            joint_traj0 = check_trajectory_feasibility(
                env=env,
                robot_id=0,
                eef_trajectory=traj0_eef,
                object_trajectory=object_trajectory,
                object_name="plate_target",
                env_lock=env_lock
            )
            
            if joint_traj0 is None:
                logger.info("  -> Robot 0 trajectory infeasible.")
                continue
                
            # Check Feasibility for Robot 1
            logger.info("  Checking Robot 1 trajectory...")
            joint_traj1 = check_trajectory_feasibility(
                env=env,
                robot_id=1,
                eef_trajectory=traj1_eef,
                object_trajectory=object_trajectory,
                object_name="plate_target",
                env_lock=env_lock
            )
            
            if joint_traj1 is None:
                logger.info("  -> Robot 1 trajectory infeasible.")
                continue
                
            logger.info("  -> Pair Valid!")
            
            valid_trajectories.append((traj0_eef, traj1_eef))
            
            # Visualize
            visualize_trajectory(vis, traj0_eef, name=f"robot0_traj_{i}", color=0x0000ff)
            visualize_trajectory(vis, traj1_eef, name=f"robot1_traj_{i}", color=0xff00ff)

    # 7. Save Trajectories
    # ---------------------------------------------
    should_save = args.save_traj
    
    if not should_save:
        try:
            if sys.stdin.isatty():
                user_input = input("Save trajectories? [y/N]: ").strip().lower()
                if user_input in ('y', 'yes'):
                    should_save = True
        except EOFError:
            pass
            
    if should_save and 'valid_trajectories' in locals() and valid_trajectories:
        # Save object trajectory
        save_trajectory(object_trajectory, args.output_dir, "object_trajectory.npz")
        
        # Save feasible robot trajectories
        for i, (traj0, traj1) in enumerate(valid_trajectories):
            save_trajectory(traj0, args.output_dir, f"robot0_trajectory_{i}.npz")
            save_trajectory(traj1, args.output_dir, f"robot1_trajectory_{i}.npz")
            
    logger.info("Closing simulation...")
    stop_event.set()
    vis_thread.join(timeout=2.0)
    
    # Stop Services
    if grasp_service:
        grasp_service.stop()
    if service_manager:
        service_manager.stop()
        
    env.close()
    sys.exit(0)

if __name__ == "__main__":
    main()