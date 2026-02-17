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

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation.planning.utils.trajectory_utils import load_trajectory
from robosuite_franka_manipulation.planning.utils.visualization_utils import visualize_trajectory
from robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer import RobotMeshVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Visualize robot trajectory in MeshCat")
    parser.add_argument("file", type=str, help="Path to trajectory .npz file")
    parser.add_argument("--robot-id", type=int, default=0, help="Robot ID (0 or 1)")
    parser.add_argument("--name", type=str, default="trajectory", help="Name/Tag for the trajectory in MeshCat")
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
        
    # 2. Load Trajectory
    try:
        eef_traj, obj_traj, joints, waypoint_results = load_trajectory(args.file)
        logger.info(f"Loaded trajectory from {args.file} ({len(eef_traj)} waypoints)")
    except Exception as e:
        logger.error(f"Failed to load trajectory: {e}")
        return

    # 3. Create Environment and TF Manager (needed for robot geometry)
    env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    tf_manager = TransformManager(env=env)
    
    # 4. Visualize Trajectory
    logger.info(f"Visualizing EEF path as '{args.name}'...")
    visualize_trajectory(
        vis=vis,
        trajectory=eef_traj,
        waypoint_results=waypoint_results,
        name=args.name,
        tf_manager=tf_manager,
    )

    if args.visualize_object_trajectory and obj_traj is not None:
        logger.info(f"Visualizing object trajectory as '{args.name}_object'...")
        visualize_trajectory(
            vis=vis,
            trajectory=obj_traj,
            name=f"{args.name}_object",
            tf_manager=tf_manager,
        )
    
    # 5. Visualize Robot Meshes
    if waypoint_results and args.robot_mesh_vis:
        logger.info(f"Visualizing robot {args.robot_id} meshes every {args.step} steps...")
        mesh_vis = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager)
        
        # We need to make sure waypoint_results are consistent with the robot_id
        # Note: waypoint_results in .npz are usually for a specific robot saved.
        mesh_vis.visualize_trajectory_configs(
            trajectory_id=args.name,
            robot_id=args.robot_id,
            waypoint_results=waypoint_results,
            step_interval=args.step,
            opacity=args.opacity
        )

    logger.info("Visualization complete. Keep MeshCat open to view.")

if __name__ == "__main__":
    main()
