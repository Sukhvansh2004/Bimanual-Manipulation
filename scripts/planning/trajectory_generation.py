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

import meshcat.transformations as tf
from grasp_gen.utils.meshcat_utils import make_frame

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

def generate_trajectory(start_pose, target_pose, num_steps=50):
    """Generate a trajectory using SE(3) interpolation through waypoints."""
    trajectory = []
    
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
            trajectory.append(np.concatenate([pos, quat]))
            
    # Ensure final point is included
    trajectory.append(np.concatenate([end_pos, end_quat]))
            
    return trajectory

def visualize_trajectory(vis, trajectory, name="trajectory", color=0x00ff00):
    """Visualize trajectory as a line/points in Meshcat."""
    points = np.array([p[:3] for p in trajectory]).T
    
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
            T = np.eye(4)
            T[:3, 3] = pose[:3]
            # Convert quat [x,y,z,w] to matrix
            r = tf.quaternion_matrix([pose[6], pose[3], pose[4], pose[5]]) # meshcat expects w,x,y,z
            T[:3, :3] = r[:3, :3]
            
            # Use make_frame from grasp_gen to draw axis
            make_frame(vis, frame_name, h=0.05, radius=0.002, o=1.0)
            vis[frame_name].set_transform(T)

def save_trajectory(trajectory, output_dir="data", filename="trajectory_data.npz"):
    """Save trajectory to npz file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    # Split trajectory into positions and quaternions
    traj_array = np.array(trajectory)
    positions = traj_array[:, :3]
    quaternions = traj_array[:, 3:]
    
    # Create StampedPose-like structure storage
    np.savez(
        filepath,
        positions=positions,
        quaternions=quaternions,
        frame_id="world"
    )
    logger.info(f"Trajectory saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize shelf manipulation trajectory")
    parser.add_argument("--save-traj", action="store_true", help="Automatically save trajectory")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for data")
    parser.add_argument("--filename", type=str, default="shelf_trajectory.npz", help="Output filename")
    args = parser.parse_args()

    logger.info("Initializing Trajectory Generation...")
    
    # 1. Setup Environment
    # --------------------
    # Shelf configuration
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678] # 90 deg Z
    
    # Plate configuration (on table)
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
            'stop_event': stop_event
        },
        daemon=True
    )
    vis_thread.start()
    
    # Connect to the same meshcat server for custom visualization
    vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
    
    logger.info("Visualization started. Open http://127.0.0.1:7000")
    time.sleep(2) # Wait for initial frames to populate
    
    # 3. Define Planning Problem
    # --------------------------
    # Start Pose: Plate's current pose
    start_pose = np.array(plate_pos + plate_quat)
    
    # Target Pose: Slot in the rack
    # Dividers at World Y: -0.2, -0.1, 0.0, 0.1, 0.2
    # Target slot between 0.0 and 0.1 -> Y = 0.05
    target_pos = [-0.65, 0.05, 1.35] # Corrected Y to avoid collision
    
    # Target Orientation: Vertical for slot
    # Rotate 90 deg around Y so it stands on edge facing X.
    target_quat = [0.0, 0.70710678, 0.0, 0.70710678]
    
    target_pose = np.array(target_pos + target_quat)
    
    logger.info(f"Planning from {start_pose[:3]} to {target_pose[:3]}")
    
    # 4. Generate Trajectory
    # ----------------------
    trajectory = generate_trajectory(start_pose, target_pose, num_steps=100)
    
    # 5. Visualize Trajectory
    # -----------------------
    visualize_trajectory(vis, trajectory, name="planned_path")
    
    # Visualize Target
    vis["target_pose"].set_object(g.Sphere(0.05), g.MeshBasicMaterial(color=0xff0000, opacity=0.5))
    T_target = np.eye(4)
    T_target[:3, 3] = target_pose[:3]
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
    
    logger.info("Press Ctrl+C to exit.")
    
    try:
        while True:
            # Step environment to keep it alive (and robots holding position)
            with env_lock:
                env.step(np.zeros(env.action_dim))
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_event.set()
        vis_thread.join()
        env.close()

if __name__ == "__main__":
    main()
