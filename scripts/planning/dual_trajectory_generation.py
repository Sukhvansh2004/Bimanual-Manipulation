#!/usr/bin/env python3
"""
Dual-arm planning script for shelf manipulation task.

This script:
1. Initializes the DualArmEnvironment with a shelf and a plate.
2. Generates grasps for both robots and pairs them (N x M).
3. Evaluates simultaneous feasibility for each grasp pair using multiprocessing.
   Optimization: Environments are created once per worker process.
4. Visualizes the planning context and generated trajectories in Meshcat.
5. Saves ALL evaluated trajectories to disk.
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

# Global variables for worker processes
worker_env = None
worker_tf_manager = None

def init_worker(ycb_objects):
    """Initialize worker process by creating a persistent environment."""
    global worker_env, worker_tf_manager
    try:
        worker_env = make_dual_arm_env(
            ycb_objects=ycb_objects,
            has_renderer=False, 
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        worker_env.reset()
        worker_tf_manager = TransformManager(worker_env)
        logging.info("Worker initialized: Environment created.")
    except Exception as e:
        logging.error(f"Worker initialization failed: {e}")

def evaluate_pair(pair_info, object_trajectory):
    """
    Worker function to evaluate a single pair of grasps using the persistent environment.
    """
    global worker_env, worker_tf_manager
    
    if worker_env is None or worker_tf_manager is None:
        logging.error("Worker environment not properly initialized.")
        return None
        
    i, j, g0, g1 = pair_info
    
    try:
        # Generate EEF trajectories
        traj0_eef = generate_end_effector_trajectory(object_trajectory, g0, tf_manager=worker_tf_manager)
        traj1_eef = generate_end_effector_trajectory(object_trajectory, g1, tf_manager=worker_tf_manager)
        
        # Check simultaneous feasibility
        results0, results1, all_feasible = check_dual_trajectory_feasibility(
            env=worker_env,
            eef_trajectory0=traj0_eef,
            eef_trajectory1=traj1_eef,
            object_trajectory=object_trajectory,
            object_name="plate_target",
            tf_manager=worker_tf_manager
        )
        
        if all_feasible:
            logging.info(f"Worker: Pair ({i}, {j}) is FEASIBLE")
        else:
            logging.info(f"Worker: Pair ({i}, {j}) is INFEASIBLE")

        return {
            'id0': i,
            'id1': j,
            'traj0_eef': traj0_eef,
            'traj1_eef': traj1_eef,
            'results0': results0,
            'results1': results1,
            'all_feasible': all_feasible,
            'grasp0': g0,
            'grasp1': g1
        }
    except Exception as e:
        logging.error(f"Worker: Error evaluating pair ({i}, {j}): {e}")
        return None

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
    logger.info("Initializing Dual Trajectory Generation...")
    
    # 1. Setup Environment
    # --------------------
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678]
    plate_pos = [0.0, -0.1, 0.85]
    plate_quat = [0, 0, 0, 1]
    
    logger.info("Creating planning environment and objects...")
    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat),
        create_ycb_object(ycb_name="plate", instance_name="plate_target", position=plate_pos, quaternion=plate_quat)
    ]
    
    # Environment for the main process (visualization and grasp generation)
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    
    env_lock = threading.RLock()
    tf_manager = TransformManager(env, env_lock=env_lock)
    
    # 2. Setup Visualization
    # ----------------------
    logger.info(f"Setting up Meshcat visualization at {constants.MESHCAT_ZMQ_URL}")
    vis_config = MeshcatTFVisualizerConfig(fps=20.0, clear=True, labels=True, visualize_scene=True)
    stop_event = threading.Event()
    vis_thread = threading.Thread(
        target=stream_transforms_to_meshcat,
        kwargs={'env': env, 'env_lock': env_lock, 'config': vis_config, 'stop_event': stop_event, 'tf_manager': tf_manager},
        daemon=True
    )
    vis_thread.start()
    
    vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
    logger.info("Visualization stream started.")
    
    # 3. Define Planning Problem
    # --------------------------
    start_pos = None
    start_quat = None
    if args.start_pos and args.start_quat:
        start_pos = np.array(args.start_pos)
        start_quat = np.array(args.start_quat)
        logger.info(f"Using manual start pose: {start_pos}")
    else:
        logger.info("Detecting plate pose from environment...")
        time.sleep(1.0)
        try:
            plate_pose_mat = env.get_object_pose("plate_target")
            if plate_pose_mat is not None:
                start_pos = plate_pose_mat[:3, 3]
                start_quat = R.from_matrix(plate_pose_mat[:3, :3]).as_quat()
                logger.info(f"Detected start pose: {start_pos}")
        except Exception as e:
            logger.warning(f"Failed to query plate pose: {e}")
            
        if start_pos is None:
            logger.info("Using default plate pose.")
            start_pos, start_quat = np.array(plate_pos), np.array(plate_quat)
    
    start_pose_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME)
    target_pose_stamped = StampedPose(position=np.array(args.target_pos), quaternion=np.array(args.target_quat), frame_id=constants.WORLD_FRAME)
    logger.info(f"Target pose: {target_pose_stamped.position}")
    
    # 4. Generate Object Trajectory
    # -----------------------------
    logger.info("Generating object trajectory...")
    object_trajectory = generate_object_trajectory(start_pose_stamped, target_pose_stamped, num_steps=constants.OBJECT_TRAJ_NUM_STEPS, tf_manager=tf_manager)
    visualize_trajectory(vis, object_trajectory, name="object_path", color=0x00ff00)
    
    # Visualize Target
    vis["target_pose"].set_object(g.Sphere(0.05), g.MeshBasicMaterial(color=0xff0000, opacity=0.5))
    T_target = np.eye(4)
    T_target[:3, 3] = target_pose_stamped.position
    T_target[:3, :3] = tf.quaternion_matrix(T.convert_quat(target_pose_stamped.quaternion, to="wxyz"))[:3, :3]
    vis["target_pose"].set_transform(T_target)
    
    # 5. Generate Grasps
    # ------------------
    logger.info("Connecting to Grasp Service...")
    service_manager = ServiceManager()
    service_manager.start()
    grasp_service = GraspService(tf_manager=tf_manager, visualize=True)
    grasp_service.start()
    grasp_generator = GraspGenerator(env=env)
    
    logger.info("Generating grasp candidates for Finger and Suction grippers...")
    all_grasps = grasp_generator.generate_grasps(object_name="plate_target", gripper_types=["finger", "suction"], num_grasps=40, env_lock=env_lock)
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)[:5]
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)[:5]
    
    logger.info(f"Selection: {len(grasps_0)} Finger grasps, {len(grasps_1)} Suction grasps.")

    # 6. Evaluate Pairs in Parallel
    # -----------------------------
    grasp_pairs = []
    for i, g0 in enumerate(grasps_0):
        for j, g1 in enumerate(grasps_1):
            grasp_pairs.append((i, j, g0, g1))
            
    logger.info(f"Evaluating {len(grasp_pairs)} grasp pairs using {args.num_workers} processes (optimized env creation)...")
    start_time = time.time()
    
    # Use multiprocessing pool with initializer to create environments once per process
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(ycb_objects,)
    ) as pool:
        eval_func = partial(evaluate_pair, object_trajectory=object_trajectory)
        all_results = pool.map(eval_func, grasp_pairs)
    
    # Filter out None results from workers
    all_results = [r for r in all_results if r is not None]
    
    duration = time.time() - start_time
    feasible_count = sum(1 for r in all_results if r['all_feasible'])
    logger.info(f"Evaluation complete in {duration:.2f}s.")
    logger.info(f"Results: {feasible_count}/{len(all_results)} pairs are fully feasible.")

    # 7. Visualization of Results
    # ---------------------------
    logger.info("Streaming evaluated trajectories to Meshcat...")
    for res in all_results:
        pair_id = f"pair_{res['id0']}_{res['id1']}"
        status = "FEASIBLE" if res['all_feasible'] else "INFEASIBLE"
        logger.debug(f"Visualizing {pair_id} ({status})")
        
        # Visualize Robot 0 Path
        visualize_trajectory(vis, res['traj0_eef'], waypoint_results=res['results0'], name=f"robot0_{pair_id}", color=0x0000ff)
        # Visualize Robot 1 Path
        visualize_trajectory(vis, res['traj1_eef'], waypoint_results=res['results1'], name=f"robot1_{pair_id}", color=0xff00ff)
        
        if args.visualize_robot_meshes and res['all_feasible']:
            logger.info(f"Rendering robot meshes for feasible pair {pair_id}")
            mesh_vis = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager, base_path=f"dual_configs/{pair_id}")
            mesh_vis.visualize_dual_arm_trajectory(
                trajectory_id="feasible",
                waypoint_results_robot0=res['results0'],
                waypoint_results_robot1=res['results1'],
                step_interval=args.mesh_step_interval,
                opacity=args.mesh_opacity
            )

    # 8. Save Trajectories
    # --------------------
    should_save = args.save_traj
    if not should_save and sys.stdin.isatty():
        try:
            should_save = input("\nSave ALL evaluated trajectories to disk? [y/N]: ").strip().lower() in ('y', 'yes')
        except EOFError:
            pass
    
    if should_save:
        logger.info(f"Saving {len(all_results)} trajectory pairs to {args.output_dir}...")
        save_trajectory(args.output_dir, "object_trajectory.npz", object_trajectory=object_trajectory)
        for res in all_results:
            pair_id = f"{res['id0']}_{res['id1']}"
            save_trajectory(args.output_dir, f"dual_robot0_pair_{pair_id}.npz", eef_trajectory=res['traj0_eef'], object_trajectory=object_trajectory, waypoint_results=res['results0'])
            save_trajectory(args.output_dir, f"dual_robot1_pair_{pair_id}.npz", eef_trajectory=res['traj1_eef'], object_trajectory=object_trajectory, waypoint_results=res['results1'])
        logger.info("All trajectories saved.")
    else:
        logger.info("Saving skipped.")

    # Cleanup
    logger.info("Shutting down...")
    stop_event.set()
    vis_thread.join(timeout=2.0)
    grasp_service.stop()
    service_manager.stop()
    env.close()
    logger.info("Done.")
    sys.exit(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
