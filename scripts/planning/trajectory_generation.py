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
import sys

import meshcat.transformations as tf
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.perception.grasp_predictor import maximize_entropy_grasps
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.utils import create_ycb_object, env_step
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
    check_trajectory_feasibility
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

def main() -> None:
    # Default values
    default_target_pos = [-0.55, 0.05, 1.35]
    default_target_quat = [0.70710678, 0.0, 0.0, 0.70710678]
    
    parser = argparse.ArgumentParser(description="Generate and visualize shelf manipulation trajectory")
    parser.add_argument("--save-traj", action="store_true", help="Automatically save trajectory")
    parser.add_argument("--output-dir", type=str, default=constants.DATA_DIR, help="Output directory for data")
    parser.add_argument("--filename", type=str, default="shelf_trajectory.npz", help="Output filename")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Pose arguments
    parser.add_argument("--start-pos", type=float, nargs=3, help="Start position [x, y, z]")
    parser.add_argument("--start-quat", type=float, nargs=4, help="Start quaternion [x, y, z, w]")
    parser.add_argument("--target-pos", type=float, nargs=3, default=default_target_pos, help="Target position [x, y, z]")
    parser.add_argument("--target-quat", type=float, nargs=4, default=default_target_quat, help="Target quaternion [x, y, z, w]")
    
    # Visualization arguments
    parser.add_argument("--visualize-robot-meshes", action="store_true", default=False,
                        help="Visualize robot configurations as meshes along trajectories")
    parser.add_argument("--mesh-step-interval", type=int, default=constants.VIS_MESH_STEP_INTERVAL,
                        help=f"Visualize robot mesh every N waypoints (default: {constants.VIS_MESH_STEP_INTERVAL})")
    parser.add_argument("--mesh-opacity", type=float, default=constants.VIS_MESH_OPACITY,
                        help=f"Opacity for robot mesh visualization (0-1, default: {constants.VIS_MESH_OPACITY})")
    
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger.debug(f"Parsed arguments: {args}")
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
    # Allow environment to settle
    for _ in range(10):
        env_step(env)
    
    # Lock for thread safety
    env_lock = threading.RLock()
    
    # Create TransformManager for pose queries
    tf_manager = TransformManager(env, env_lock=env_lock)
    
    # 2. Setup Visualization
    # ----------------------
    logger.info("Setting up visualization")

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
                start_quat = R.from_matrix(plate_pose_mat[:3, :3]).as_quat()
                logger.info(f"Detected plate pose: pos={start_pos}")
            else:
                logger.warning("Could not find 'plate_target' in environment. Using default.")
        except Exception as e:
            logger.warning(f"Error querying object pose: {e}")
            
        if start_pos is None:
            start_pos = np.array(plate_pos)
            start_quat = np.array(plate_quat)
    
    start_pose_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME)
    
    # Determine Target Pose
    target_pos = np.array(args.target_pos)
    target_quat = np.array(args.target_quat)
    target_pose_stamped = StampedPose(position=target_pos, quaternion=target_quat, frame_id=constants.WORLD_FRAME)
    
    logger.info(f"Planning from {start_pose_stamped.position} to {target_pose_stamped.position}")
    
    # 4. Generate Object Trajectory
    # ----------------------
    object_trajectory = generate_object_trajectory(
        start_pose_stamped, 
        target_pose_stamped, 
        num_steps=constants.OBJECT_TRAJ_NUM_STEPS,
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
        tf_manager=tf_manager,
        visualize=True 
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
    
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)
    
    grasps_0 = maximize_entropy_grasps(grasps_0, num_grasps=10)
    grasps_1 = maximize_entropy_grasps(grasps_1, num_grasps=10)
    
    logger.info(f"Received {len(grasps_0)} grasps for Finger and {len(grasps_1)} for Suction.")

    # Store all trajectory results for each robot
    robot0_trajectory_results = []
    robot1_trajectory_results = []

    if not grasps_0 and not grasps_1:
        logger.error("Failed to generate any grasps.")
    else:
        # Step 6b: Process Robot 0 (Finger) Grasps
        logger.info("Processing Robot 0 (Finger) grasps...")
        for i, g0 in enumerate(grasps_0):
            logger.info(f"  Checking Robot 0 Grasp {i}...")
            # Generate EEF trajectory
            traj0_eef = generate_end_effector_trajectory(object_trajectory, g0, tf_manager=tf_manager)
            
            # Check feasibility (ignoring Robot 1)
            results0, all_feasible0 = check_trajectory_feasibility(
                env=env,
                robot_id=0,
                eef_trajectory=traj0_eef,
                object_trajectory=object_trajectory,
                object_name="plate_target",
                env_lock=env_lock,
                tf_manager=tf_manager,
                ignore_names=["robot1", "gripper1"]
            )
            
            feasible_count = sum(1 for r in results0 if r.is_feasible)
            logger.info(f"    -> {feasible_count}/{len(results0)} waypoints feasible (all_feasible={all_feasible0})")
            
            robot0_trajectory_results.append({
                'id': i,
                'traj_eef': traj0_eef,
                'results': results0,
                'all_feasible': all_feasible0,
                'grasp': g0
            })
            
            # Visualize trajectory line and status-coded markers
            visualize_trajectory(vis, traj0_eef, waypoint_results=results0, name=f"robot0_traj_{i}", color=0x0000ff)
            
            # Visualize robot mesh configurations along the trajectory
            if args.visualize_robot_meshes:
                mesh_visualizer = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager, base_path="robot0_configs")
                mesh_visualizer.visualize_trajectory_configs(
                    trajectory_id=f"grasp_{i}",
                    robot_id=0,
                    waypoint_results=results0,
                    step_interval=args.mesh_step_interval,
                    opacity=args.mesh_opacity
                )

        # Step 6c: Process Robot 1 (Suction) Grasps
        logger.info("Processing Robot 1 (Suction) grasps...")
        for i, g1 in enumerate(grasps_1):
            logger.info(f"  Checking Robot 1 Grasp {i}...")
            # Generate EEF trajectory
            traj1_eef = generate_end_effector_trajectory(object_trajectory, g1, tf_manager=tf_manager)
            
            # Check feasibility (ignoring Robot 0)
            results1, all_feasible1 = check_trajectory_feasibility(
                env=env,
                robot_id=1,
                eef_trajectory=traj1_eef,
                object_trajectory=object_trajectory,
                object_name="plate_target",
                env_lock=env_lock,
                tf_manager=tf_manager,
                ignore_names=["robot0", "gripper0"]
            )
            
            feasible_count = sum(1 for r in results1 if r.is_feasible)
            logger.info(f"    -> {feasible_count}/{len(results1)} waypoints feasible (all_feasible={all_feasible1})")
            
            robot1_trajectory_results.append({
                'id': i,
                'traj_eef': traj1_eef,
                'results': results1,
                'all_feasible': all_feasible1,
                'grasp': g1
            })
            
            # Visualize trajectory line and status-coded markers
            visualize_trajectory(vis, traj1_eef, waypoint_results=results1, name=f"robot1_traj_{i}", color=0xff00ff)
            
            # Visualize robot mesh configurations along the trajectory
            if args.visualize_robot_meshes:
                mesh_visualizer = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager, base_path="robot1_configs")
                mesh_visualizer.visualize_trajectory_configs(
                    trajectory_id=f"grasp_{i}",
                    robot_id=1,
                    waypoint_results=results1,
                    step_interval=args.mesh_step_interval,
                    opacity=args.mesh_opacity
                )

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
            
    if should_save and (robot0_trajectory_results or robot1_trajectory_results):
        # Save object trajectory
        save_trajectory(
            output_dir=args.output_dir,
            filename="object_trajectory.npz",
            object_trajectory=object_trajectory
        )
        
        # Save Robot 0 trajectories
        for traj_data in robot0_trajectory_results:
            i = traj_data['id']
            save_trajectory(
                output_dir=args.output_dir,
                filename=f"robot0_trajectory_{i}.npz",
                eef_trajectory=traj_data['traj_eef'],
                object_trajectory=object_trajectory,
                waypoint_results=traj_data['results']
            )
            
        # Save Robot 1 trajectories
        for traj_data in robot1_trajectory_results:
            i = traj_data['id']
            save_trajectory(
                output_dir=args.output_dir,
                filename=f"robot1_trajectory_{i}.npz",
                eef_trajectory=traj_data['traj_eef'],
                object_trajectory=object_trajectory,
                waypoint_results=traj_data['results']
            )
        
        logger.info(f"Saved {len(robot0_trajectory_results)} Robot 0 and {len(robot1_trajectory_results)} Robot 1 trajectories to {args.output_dir}")
    else:
        logger.info("No trajectories saved.")
        
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