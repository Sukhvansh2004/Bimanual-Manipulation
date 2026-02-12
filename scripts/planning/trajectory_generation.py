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
import multiprocessing
from functools import partial

import meshcat.transformations as tf
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.perception.grasp_predictor import maximize_entropy_grasps
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.utils import (
    create_ycb_object, 
    env_step, 
    create_primitive_object, 
    register_mesh_extraction_env
)
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
constants.ZMQ_MANAGER_TIMEOUT = 16 * 50 * 1000
constants.ZMQ_SERVICE_CALL_TIMEOUT = 15 * 50 * 1000

# Global variables for worker processes
worker_env = None
worker_tf_manager = None
worker_vis = None

def init_worker(ycb_objects, custom_objects):
    """Initialize worker process by creating a persistent environment and visualizer."""
    global worker_env, worker_tf_manager, worker_vis
    try:
        worker_env = make_dual_arm_env(
            ycb_objects=ycb_objects,
            custom_objects=custom_objects,
            has_renderer=False, 
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        worker_env.reset()
        # Allow environment to settle
        for _ in range(10):
            env_step(worker_env)
        worker_tf_manager = TransformManager(worker_env)
        worker_vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
        logging.info("Worker initialized: Environment and Visualizer created.")
    except Exception as e:
        logging.error(f"Worker initialization failed: {e}")

def evaluate_robot_trajectory(grasp_info, robot_id, object_trajectory, object_name, ignore_names, vis_args=None):
    """Worker function to evaluate a single robot trajectory."""
    global worker_env, worker_tf_manager, worker_vis
    
    if worker_env is None or worker_tf_manager is None:
        logging.error("Worker environment not properly initialized.")
        return None
        
    i, grasp = grasp_info
    
    try:
        # Generate EEF trajectory
        traj_eef = generate_end_effector_trajectory(object_trajectory, grasp, tf_manager=worker_tf_manager)
        
        # Check feasibility
        results, all_feasible = check_trajectory_feasibility(
            env=worker_env,
            robot_id=robot_id,
            eef_trajectory=traj_eef,
            object_trajectory=object_trajectory,
            object_name=object_name,
            tf_manager=worker_tf_manager,
            ignore_names=ignore_names
        )
        
        # Visualize in worker process
        if worker_vis:
            color = 0x0000ff if robot_id == 0 else 0xff00ff
            visualize_trajectory(
                worker_vis, 
                traj_eef, 
                waypoint_results=results, 
                name=f"robot{robot_id}_traj_{i}", 
                color=color,
                tf_manager=worker_tf_manager
            )
            
            if vis_args and vis_args.get('visualize_robot_meshes'):
                mesh_visualizer = RobotMeshVisualizer(vis=worker_vis, tf_manager=worker_tf_manager, base_path=f"robot{robot_id}_configs")
                mesh_visualizer.visualize_trajectory_configs(
                    trajectory_id=f"grasp_{i}",
                    robot_id=robot_id,
                    waypoint_results=results,
                    step_interval=vis_args.get('mesh_step_interval', 10),
                    opacity=vis_args.get('mesh_opacity', 0.2)
                )
        
        return {
            'id': i,
            'traj_eef': traj_eef,
            'results': results,
            'all_feasible': all_feasible,
            'grasp': grasp
        }
    except Exception as e:
        logging.error(f"Worker: Error evaluating robot {robot_id} grasp {i}: {e}")
        return None

def main() -> None:
    # Default values
    default_target_pos = [-0.55, 0.05, 1.35]
    default_target_quat = [0.70710678, 0.0, 0.0, 0.70710678]
    
    parser = argparse.ArgumentParser(description="Generate and visualize shelf manipulation trajectory")
    parser.add_argument("--save-traj", action="store_true", help="Automatically save trajectory")
    parser.add_argument("--output-dir", type=str, default=constants.DATA_DIR, help="Output directory for data")
    parser.add_argument("--filename", type=str, default="shelf_trajectory.npz", help="Output filename")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count() // 2, help="Number of parallel workers")
    
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
    plate_pos = [0.0, 0.0, 0.85]
    plate_quat = [0, 0, 0, 1]
    
    logger.info("Creating planning environment and objects...")
    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat)
    ]
    custom_objects = [
        create_primitive_object(obj_type='box', name='flatPlate', size=[0.3, 0.3, 0.02], position=plate_pos, quat=plate_quat)
    ]
    
    # Create env (headless for planning script, visualization handled by Meshcat)
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    # Allow environment to settle
    for _ in range(10):
        env_step(env)
    
    # Register for lazy mesh extraction
    register_mesh_extraction_env(env)
    
    # Lock for thread safety
    env_lock = threading.RLock()
    
    # Create TransformManager for pose queries
    tf_manager = TransformManager(env, env_lock=env_lock)
    
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
        
        # Query pose of flatPlate
        try:
            plate_pose_mat = env.get_object_pose("flatPlate")
            if plate_pose_mat is not None:
                start_pos = plate_pose_mat[:3, 3]
                # Convert matrix to quat [x, y, z, w]
                start_quat = R.from_matrix(plate_pose_mat[:3, :3]).as_quat()
                logger.info(f"Detected plate pose: pos={start_pos}")
            else:
                logger.warning("Could not find 'flatPlate' in environment. Using default.")
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
    
    logger.info("Object trajectory generated.")
    
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
        object_name="flatPlate", 
        gripper_types=["finger", "suction"], 
        num_grasps=40,
        env_lock=env_lock
    )
    
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)
    
    grasps_0 = maximize_entropy_grasps(grasps_0, num_grasps=10)
    grasps_1 = maximize_entropy_grasps(grasps_1, num_grasps=10)
    
    logger.info(f"Received {len(grasps_0)} grasps for Finger and {len(grasps_1)} for Suction.")

    # 6. Evaluate Robot Trajectories in Parallel
    # ------------------------------------------
    logger.info(f"Evaluating grasps using {args.num_workers} processes...")
    start_time = time.time()

    vis_args = {
        'visualize_robot_meshes': args.visualize_robot_meshes,
        'mesh_step_interval': args.mesh_step_interval,
        'mesh_opacity': args.mesh_opacity
    }

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(ycb_objects, custom_objects)
    ) as pool:
        
        # Evaluate Robot 0 Grasps
        logger.info("Processing Robot 0 (Finger) grasps...")
        robot0_tasks = [(i, g0) for i, g0 in enumerate(grasps_0)]
        eval_func0 = partial(
            evaluate_robot_trajectory, 
            robot_id=0, 
            object_trajectory=object_trajectory, 
            object_name="flatPlate", 
            ignore_names=["robot1", "gripper1"],
            vis_args=vis_args
        )
        robot0_trajectory_results = pool.map(eval_func0, robot0_tasks)
        robot0_trajectory_results = [r for r in robot0_trajectory_results if r is not None]

        # Evaluate Robot 1 Grasps
        logger.info("Processing Robot 1 (Suction) grasps...")
        robot1_tasks = [(i, g1) for i, g1 in enumerate(grasps_1)]
        eval_func1 = partial(
            evaluate_robot_trajectory, 
            robot_id=1, 
            object_trajectory=object_trajectory, 
            object_name="flatPlate", 
            ignore_names=["robot0", "gripper0"],
            vis_args=vis_args
        )
        robot1_trajectory_results = pool.map(eval_func1, robot1_tasks)
        robot1_trajectory_results = [r for r in robot1_trajectory_results if r is not None]

    duration = time.time() - start_time
    logger.info(f"Evaluation complete in {duration:.2f}s.")

    # 8. Collective Feasibility Analysis
    # ----------------------------------
    num_waypoints = len(object_trajectory)
    logger.info(f"Analyzing collective feasibility for {num_waypoints} object waypoints...")
    
    feasible_waypoints_count = 0
    for k in range(num_waypoints):
        r0_feasible = any(res['results'][k].is_feasible for res in robot0_trajectory_results)
        r1_feasible = any(res['results'][k].is_feasible for res in robot1_trajectory_results)
        
        if r0_feasible and r1_feasible:
            status = "FEASIBLE (BOTH)"
            feasible_waypoints_count += 1
        elif r0_feasible:
            status = "FEASIBLE (ROBOT 0 ONLY)"
        elif r1_feasible:
            status = "FEASIBLE (ROBOT 1 ONLY)"
        else:
            status = "INFEASIBLE (NONE)"
            
        logger.info(f"  Waypoint {k:2d}: {status}")

    logger.info(f"Collective analysis: {feasible_waypoints_count}/{num_waypoints} waypoints can be reached by AT LEAST one grasp from BOTH robots.")

    # 9. Save Trajectories
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
    
    # Stop Services
    if grasp_service:
        grasp_service.stop()
    if service_manager:
        service_manager.stop()
        
    env.close()
    sys.exit(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
