#!/usr/bin/env python3
"""Visualize robot trajectories in MeshCat.

Loads one trajectory ``.npz`` file (or every ``.npz`` in a directory) and
visualizes:
1. The end-effector path (line and frames).
2. Robot meshes at waypoints (green = feasible, red = infeasible), with
   ``--robot-mesh-vis``.
3. The object trajectory frames, with ``--visualize-object-trajectory``,
   and object meshes, with ``--object-mesh-vis`` (last loaded file only).

Usage:
    python scripts/utils/visualize_robot_trajectory.py traj.npz \\
        --experiment experiments/flatPlate_table_to_shelf.json --robot-mesh-vis
    python scripts/utils/visualize_robot_trajectory.py data/<run>/ \\
        --experiment experiments/flatPlate_table_to_shelf.json \\
        --visualize-object-trajectory --object-mesh-vis --step 5 --clear
"""

import argparse
import logging
import meshcat
import numpy as np
import time
import os
import threading

from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.object_factory import register_mesh_extraction_env
from bimanual_manipulation.transforms import (
    TransformManager, 
    stream_transforms_to_meshcat, 
    MeshcatTFVisualizerConfig
)
from bimanual_manipulation.experiment_config import load_experiment
from bimanual_manipulation.planning.trajectory.utils import load_trajectory
from bimanual_manipulation.planning.visualization.trajectory import visualize_trajectory
from bimanual_manipulation.planning.visualization import RobotMeshVisualizer, ObjectMeshVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Parse CLI args, load trajectories, draw them in MeshCat, and block until Ctrl-C.

    The robot ID is ``1`` if ``--robot-id 1`` is given; otherwise it is
    inferred from ``robot0``/``robot1`` in each file name (so
    ``--robot-id 0`` has no effect). Files that fail to load are logged and
    skipped.
    """
    parser = argparse.ArgumentParser(description="Visualize robot trajectory in MeshCat")
    parser.add_argument("path", type=str, help="Path to trajectory .npz file or directory")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    parser.add_argument("--robot-id", type=int, default=0, help="Robot ID (0 or 1)")
    parser.add_argument("--name", type=str, default=None, help="Name/Tag for the trajectory in MeshCat (defaults to file name)")
    parser.add_argument("--step", type=int, default=10, help="Step interval for robot mesh visualization")
    parser.add_argument("--robot-mesh-vis", action="store_true", help="Whether to visualize robot meshes at waypoints")
    parser.add_argument("--object-mesh-vis", action="store_true", help="Whether to visualize object meshes at waypoints")
    parser.add_argument("--object-name", type=str, default=None, help="Name of the object to visualize (defaults to experiment's manipulated_object)")
    parser.add_argument("--visualize-object-trajectory", action="store_true", help="Whether to visualize object trajectory (if available)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Opacity for robot meshes")
    parser.add_argument("--clear", action="store_true", help="Clear MeshCat before visualizing")
    parser.add_argument("--zmq-url", type=str, default=None, help="MeshCat ZMQ URL (defaults to experiment config or constants)")
    
    args = parser.parse_args()
    
    # 1. Load experiment configuration
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    ycb_objects = exp_cfg.ycb_objects
    custom_objects = exp_cfg.custom_objects
    object_name = args.object_name or exp_cfg.task.manipulated_object
    zmq_url = args.zmq_url or getattr(exp_cfg.visualization, 'zmq_url', None) or str(constants.MESHCAT_ZMQ_URL)

    # 2. Initialize Visualizer
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if args.clear:
        vis.delete()
        logger.info("MeshCat cleared.")
        
    # 3. Collect Files
    if os.path.isdir(args.path):
        files = sorted([os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".npz")])
        logger.info(f"Found {len(files)} trajectory files in {args.path}")
    else:
        files = [args.path]

    # 4. Create Environment and TF Manager
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        has_renderer=False, 
        has_offscreen_renderer=False, 
        use_camera_obs=False,
        table_full_size=exp_cfg.table_full_size,
        table_offset=exp_cfg.table_offset,
        robot_base_configs=exp_cfg.robot_base_configs,
    )
    for _ in range(10):
        env_step(env)

    register_mesh_extraction_env(env)
    tf_manager = TransformManager(env=env)

    # 5. Stream TFs and Scene to MeshCat
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
    
    obj_traj = None  # Initialize object trajectory variable for later use
    # 6. Process each file
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        traj_name = args.name if (args.name and len(files) == 1) else base_name
        
        # Determine robot_id
        if len(files) and args.robot_id == 1:
            robot_id = args.robot_id
        else:
            if "robot0" in base_name:
                robot_id = 0
            elif "robot1" in base_name:
                robot_id = 1
            else:
                robot_id = None
        
        logger.info(f"Processing '{file_path}' as '{traj_name}' (robot_id: {robot_id})...")
        
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

        # Visualize Robot Meshes
        if waypoint_results and args.robot_mesh_vis:
            if robot_id is not None:
                logger.info(f"  -> Visualizing robot {robot_id} meshes for '{traj_name}'...")
                mesh_vis = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager)
                mesh_vis.visualize_trajectory_configs(
                    trajectory_id=traj_name,
                    robot_id=robot_id,
                    waypoint_results=waypoint_results,
                    step_interval=args.step,
                    opacity=args.opacity
                )
            else:
                logger.warning(f"  -> Skipping robot mesh visualization for '{traj_name}': robot_id not found in filename.")

    # Visualize object trajectory
    if args.visualize_object_trajectory and obj_traj is not None:
        visualize_trajectory(
            vis=vis,
            trajectory=obj_traj,
            name=f"{object_name}_trajectory",
            tf_manager=tf_manager,
        )
    

    # Visualize Object Meshes
    if obj_traj is not None and args.object_mesh_vis:
        logger.info(f"  -> Visualizing object '{object_name}' meshes")
        obj_mesh_vis = ObjectMeshVisualizer(vis=vis, tf_manager=tf_manager)
        obj_mesh_vis.visualize_trajectory(
            object_name=object_name,
            object_trajectory=obj_traj,
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
