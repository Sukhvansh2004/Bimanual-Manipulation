#!/usr/bin/env python3
"""
Planning script for shelf manipulation task.

This script:
1. Initializes the DualArmEnvironment with a shelf and a plate.
2. Plans a simple trajectory for the plate from the table to a rack slot in the shelf.
3. Visualizes the planning context and the generated trajectory using Meshcat.
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
from typing import List

import meshcat.transformations as tf
from grasp_gen.utils.meshcat_utils import make_frame

import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.utils import create_ycb_object
from robosuite_franka_manipulation.utils.se3 import interpolate_pose
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

def generate_trajectory(start_pose: np.ndarray, target_pose: np.ndarray, num_steps: int = 50, frame_id: str = "world") -> List[StampedPose]:
    """Generate a trajectory using SE(3) interpolation through waypoints.
    
    Args:
        start_pose: Start pose as [x, y, z, qx, qy, qz, qw]
        target_pose: Target pose as [x, y, z, qx, qy, qz, qw]
        num_steps: Number of steps for the trajectory
        frame_id: Frame ID for the StampedPoses
        
    Returns:
        List of StampedPose objects representing the trajectory
    """
    trajectory_points = []
    
    start_pos = np.array(start_pose[:3])
    start_quat = np.array(start_pose[3:])
    
    end_pos = np.array(target_pose[:3])
    end_quat = np.array(target_pose[3:])
    
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

def save_trajectory(trajectory: List[StampedPose], output_dir: str = "data", filename: str = "trajectory_data.npz") -> None:
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
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for data")
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
    
    start_pose_vec = np.concatenate([start_pos, start_quat])
    
    # Determine Target Pose
    target_pos = np.array(args.target_pos)
    target_quat = np.array(args.target_quat)
    target_pose_vec = np.concatenate([target_pos, target_quat])
    
    logger.info(f"Planning from {start_pose_vec[:3]} to {target_pose_vec[:3]}")
    
    # 4. Generate Trajectory
    # ----------------------
    trajectory = generate_trajectory(start_pose_vec, target_pose_vec, num_steps=100)
    
    # 5. Visualize Trajectory
    # -----------------------
    visualize_trajectory(vis, trajectory, name="planned_path")
    
    # Visualize Target
    vis["target_pose"].set_object(g.Sphere(0.05), g.MeshBasicMaterial(color=0xff0000, opacity=0.5))
    T_target = np.eye(4)
    T_target[:3, 3] = target_pos
    
    # Convert target_quat (xyzw) to wxyz for meshcat
    target_quat_wxyz = T.convert_quat(target_quat, to="wxyz")
    r = tf.quaternion_matrix(target_quat_wxyz)
    T_target[:3, :3] = r[:3, :3]
    vis["target_pose"].set_transform(T_target)
    
    logger.info("Plan visualized.")
    
    # 6. Save Trajectory (Interactive or Automatic)
    # ---------------------------------------------
    should_save = args.save_traj
    
    if not should_save:
        try:
            # Check if running interactively
            if sys.stdin.isatty():
                user_input = input("Save trajectory? [y/N]: ").strip().lower()
                if user_input in ('y', 'yes'):
                    should_save = True
        except EOFError:
            pass # Non-interactive mode
            
    if should_save:
        save_trajectory(trajectory, args.output_dir, args.filename)
    
    logger.info("Closing simulation...")
    stop_event.set()
    vis_thread.join(timeout=2.0)
    env.close()
    sys.exit(0)

if __name__ == "__main__":
    main()
