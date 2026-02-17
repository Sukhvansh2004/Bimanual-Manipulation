#!/usr/bin/env python3
"""
Visualize Robot Trajectory in MeshCat.

This script loads a trajectory file (.npz) and visualizes:
1. The end-effector path (line and frames).
2. Robot meshes at waypoints (green = feasible, red = infeasible).
3. Coordinate frames for object trajectory (if available).
"""

import argparse
import logging
import meshcat
import numpy as np
import time
import os
import threading

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object, create_primitive_object, register_mesh_extraction_env
from robosuite_franka_manipulation.transforms import (
    TransformManager, 
    stream_transforms_to_meshcat, 
    MeshcatTFVisualizerConfig
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import load_trajectory
from robosuite_franka_manipulation.planning.utils.visualization_utils import visualize_trajectory
from robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer import RobotMeshVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Visualize robot trajectory in MeshCat")
    parser.add_argument("path", type=str, help="Path to trajectory .npz file or directory")
    parser.add_argument("--robot-id", type=int, default=0, help="Robot ID (0 or 1)")
    parser.add_argument("--name", type=str, default=None, help="Name/Tag for the trajectory in MeshCat (defaults to file name)")
    parser.add_argument("--step", type=int, default=10, help="Step interval for robot mesh visualization")
    parser.add_argument("--robot-mesh-vis", action="store_true", help="Whether to visualize robot meshes at waypoints")
    parser.add_argument("--visualize-object-trajectory", action="store_true", help="Whether to visualize object trajectory (if available)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Opacity for robot meshes")
    parser.add_argument("--clear", action="store_true", help="Clear MeshCat before visualizing")
    parser.add_argument("--zmq-url", type=str, default=str(constants.MESHCAT_ZMQ_URL), help="MeshCat ZMQ URL")
    
    args = parser.parse_args()
    
    # 1. Initialize Visualizer
    vis = meshcat.Visualizer(zmq_url=args.zmq_url)
    if args.clear:
        vis.delete()
        logger.info("MeshCat cleared.")
        
    # 2. Collect Files
    if os.path.isdir(args.path):
        files = sorted([os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".npz")])
        logger.info(f"Found {len(files)} trajectory files in {args.path}")
    else:
        files = [args.path]

    # 3. Create Environment and TF Manager
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678]
    plate_pos = [0.0, 0.0, 0.85]
    plate_quat = [0, 0, 0, 1]

    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat)
    ]
    custom_objects = [
        create_primitive_object(obj_type='box', name='flatPlate', size=[0.3, 0.3, 0.02], position=plate_pos, quat=plate_quat)
    ]

    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        has_renderer=False, 
        has_offscreen_renderer=False, 
        use_camera_obs=False
    )
    register_mesh_extraction_env(env)
    tf_manager = TransformManager(env=env)

    # 4. Stream TFs and Scene to MeshCat
    logger.info("Starting TF and Scene stream to MeshCat...")
    env_lock = threading.Lock()
    stream_config = MeshcatTFVisualizerConfig(clear=False, visualize_scene=True)
    
    threading.Thread(
        target=stream_transforms_to_meshcat,
        kwargs={
            "env": env,
            "env_lock": env_lock,
            "tf_manager": tf_manager,
            "config": stream_config
        },
        daemon=True
    ).start()
    
    # 5. Process each file
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        traj_name = args.name if (args.name and len(files) == 1) else base_name
        
        logger.info(f"Processing '{file_path}' as '{traj_name}'...")
        
        try:
            eef_traj, obj_traj, joints, waypoint_results = load_trajectory(file_path)
        except Exception as e:
            logger.error(f"Failed to load trajectory from {file_path}: {e}")
            continue

        # Visualize EEF path
        visualize_trajectory(
            vis=vis,
            trajectory=eef_traj,
            waypoint_results=waypoint_results,
            name=traj_name,
            tf_manager=tf_manager,
        )

        # Visualize object trajectory
        if args.visualize_object_trajectory and obj_traj is not None:
            visualize_trajectory(
                vis=vis,
                trajectory=obj_traj,
                name=f"{traj_name}_object",
                tf_manager=tf_manager,
            )
        
        # Visualize Robot Meshes
        if waypoint_results and args.robot_mesh_vis:
            logger.info(f"  -> Visualizing robot {args.robot_id} meshes for '{traj_name}'...")
            mesh_vis = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager)
            mesh_vis.visualize_trajectory_configs(
                trajectory_id=traj_name,
                robot_id=args.robot_id,
                waypoint_results=waypoint_results,
                step_interval=args.step,
                opacity=args.opacity
            )

    logger.info("Visualization complete. Keep script running to view TFs and Scene.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main()
