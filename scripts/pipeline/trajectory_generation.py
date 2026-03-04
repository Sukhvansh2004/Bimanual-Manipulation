#!/usr/bin/env python3
"""
Planning script for manipulation tasks driven by experiment JSON configs.

This script:
1. Loads an experiment configuration from a JSON file (or falls back to CLI defaults).
2. Initializes the DualArmEnvironment with the specified objects.
3. Plans an object trajectory from start to target pose.
4. Generates grasps and evaluates/filters feasible EEF trajectories.
5. Visualizes everything in Meshcat and optionally saves trajectories to disk.

Usage:
    python trajectory_generation.py --experiment experiments/flatPlate_table_to_shelf.json
    python trajectory_generation.py --experiment experiments/flatPlate_shelf_to_box.json --save-traj
"""

import numpy as np
import os
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
from robosuite_franka_manipulation.utils.env_utils import (
    env_step,
    register_mesh_extraction_env
)
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose

# Services and Planning
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.services_core import ServiceManager
from robosuite_franka_manipulation.planning.grasp_generator import GraspGenerator

# Planning Modules
from robosuite_franka_manipulation.planning.trajectory import (
    generate_end_effector_trajectory, 
    generate_object_trajectory,
)
from robosuite_franka_manipulation.planning.collision_checker import (
    check_trajectory_feasibility,
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import save_trajectory
from robosuite_franka_manipulation.planning.visualization.trajectory import visualize_trajectory
from robosuite_franka_manipulation.planning.visualization.robot_mesh import RobotMeshVisualizer
from robosuite_franka_manipulation.planning.visualization.object_mesh import ObjectMeshVisualizer
from robosuite_franka_manipulation.experiment_config import (
    load_experiment, detect_start_pose, get_collision_filter_fn
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

def init_worker(ycb_objects, custom_objects, log_level, collision_exclude_patterns=None, table_full_size=None, enable_meshcat=True):
    """Initialize worker process by creating a persistent environment and visualizer.
    
    Args:
        ycb_objects: YCB object specs for the environment.
        custom_objects: Custom object specs for the environment.
        log_level: Logging level.
        collision_exclude_patterns: List of substrings; geoms whose names contain
            any of these patterns (or are robot geoms) keep their collisions.
            All other geoms get contype=0 for faster collision checking.
        table_full_size: Optional tuple (x, y, z) to override the default table dimensions.
        enable_meshcat: Whether to create a Meshcat visualizer in the worker.
    """
    setup_logging(log_level)
    global worker_env, worker_tf_manager, worker_vis
    if collision_exclude_patterns is None:
        collision_exclude_patterns = []
    try:
        worker_env = make_dual_arm_env(
            ycb_objects=ycb_objects,
            custom_objects=custom_objects,
            has_renderer=False, 
            has_offscreen_renderer=False,
            use_camera_obs=False,
            table_full_size=table_full_size,
        )
        worker_env.reset()
        # Allow environment to settle
        for _ in range(10):
            env_step(worker_env)
        
        # Optimization: Disable collisions between environment geoms
        # Anything that isn't a robot or a target object should have contype=0
        _is_relevant = get_collision_filter_fn(
            type('CF', (), {'exclude_patterns': collision_exclude_patterns})()
        )
        model = worker_env.sim.model
        for i in range(model.ngeom):
            name = model.geom_id2name(i) or ""
            if not _is_relevant(name):
                model.geom_contype[i] = 0

        worker_tf_manager = TransformManager(worker_env)
        if enable_meshcat:
            worker_vis = meshcat.Visualizer(zmq_url=constants.MESHCAT_ZMQ_URL)
        else:
            worker_vis = None
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
    logger.info("WORKER: Evaluating Robot %d Grasp %d (Gripper: %s, Score: %.3f)", robot_id, i, grasp.gripper_type, grasp.score)

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
            
            if vis_args and vis_args.get('visualize_mesh'):
                sample_ratio = vis_args.get('mesh_sample_ratio', 0.5)
                step_interval = vis_args.get('mesh_step_interval', 10)
                opacity = vis_args.get('mesh_opacity', 0.2)
                
                # Robot mesh visualization
                mesh_visualizer = RobotMeshVisualizer(
                    vis=worker_vis, tf_manager=worker_tf_manager,
                    mesh_sample_ratio=sample_ratio
                )
                mesh_visualizer.visualize_trajectory_configs(
                    trajectory_id=f"grasp_{i}",
                    robot_id=robot_id,
                    waypoint_results=results,
                    step_interval=step_interval,
                    opacity=opacity
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
    parser = argparse.ArgumentParser(description="Generate and visualize manipulation trajectory")
    
    # Experiment configuration (required)
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    
    # Minimal CLI overrides – everything else lives in the experiment JSON
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (overrides JSON)")
    parser.add_argument("--save-traj", action="store_true", help="Force-save trajectories (overrides JSON)")
    parser.add_argument("--no-meshcat", action="store_true",
                        help="Disable all Meshcat visualization (useful for batch / headless runs)")
    
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load experiment configuration
    # ---------------------------------------------------------------
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)
    
    # Resolve runtime settings: CLI overrides > JSON
    rt_cfg = exp_cfg.runtime
    vis_cfg = exp_cfg.visualization
    out_cfg = exp_cfg.output
    
    debug = args.debug or rt_cfg.debug
    enable_meshcat = vis_cfg.enable_meshcat and not args.no_meshcat
    num_workers = rt_cfg.num_workers if rt_cfg.num_workers > 0 else multiprocessing.cpu_count() // 2

    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)

    logger.debug(f"Parsed arguments: {args}")
    
    logger.info("Initializing Trajectory Generation...")
    
    # 1. Setup Environment
    # --------------------
    ycb_objects = exp_cfg.ycb_objects
    custom_objects = exp_cfg.custom_objects
    object_name = exp_cfg.task.manipulated_object
    collision_exclude_patterns = exp_cfg.collision_filter.exclude_patterns
    table_full_size = exp_cfg.table_full_size
    
    logger.info("Creating planning environment with %d YCB + %d custom objects...",
                len(ycb_objects), len(custom_objects))
    
    # Create env (headless for planning script, visualization handled by Meshcat)
    env = make_dual_arm_env(
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        has_renderer=False, 
        has_offscreen_renderer=False,
        use_camera_obs=False,
        table_full_size=table_full_size,
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
    
    # 2. Setup Visualization (Scene Streaming)
    # ----------------------------------------
    stop_event = threading.Event()
    vis_thread = None
    vis = None
    if enable_meshcat:
        logger.info("Setting up global visualization stream...")
        vis_config = MeshcatTFVisualizerConfig(fps=20.0, clear=True, labels=True, visualize_scene=True)
        vis_thread = threading.Thread(
            target=stream_transforms_to_meshcat,
            kwargs={'env': env, 'env_lock': env_lock, 'config': vis_config, 'stop_event': stop_event, 'tf_manager': tf_manager},
            daemon=True
        )
        vis_thread.start()
        # Connect to the same meshcat server for worker coordination
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
    else:
        logger.info("Meshcat visualization disabled.")
    
    # 3. Define Planning Problem
    # --------------------------
    
    # Determine Start Pose  (experiment JSON > auto-detect from sim)
    start_pos = None
    start_quat = None
    
    if exp_cfg.task.start_pose is not None:
        start_pos = exp_cfg.task.start_pose.position
        start_quat = exp_cfg.task.start_pose.quaternion
        logger.info(f"Using experiment start pose: pos={start_pos}")
    else:
        logger.info("No start pose specified. Auto-detecting from simulation...")
        time.sleep(1.0)
        detected = detect_start_pose(env, object_name)
        if detected is not None:
            start_pos = detected.position
            start_quat = detected.quaternion
            logger.info(f"Detected object pose: pos={start_pos}")
        else:
            logger.error("Auto-detection failed and no start_pose in experiment JSON.")
            sys.exit(1)
    
    start_pose_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME)
    
    # Target Pose (from experiment JSON)
    target_pos = exp_cfg.task.target_pose.position
    target_quat = exp_cfg.task.target_pose.quaternion
    target_pose_stamped = StampedPose(position=target_pos, quaternion=target_quat, frame_id=constants.WORLD_FRAME)
    
    logger.info(f"Planning from {start_pose_stamped.position} to {target_pose_stamped.position}")
    
    # 4. Generate Object Trajectory
    # ----------------------
    plan_cfg = exp_cfg.planning
    traj_num_steps = plan_cfg.num_steps
    
    # RRT workspace bounds from experiment config (None → auto-compute in planner)
    rrt_bounds = plan_cfg.rrt_bounds
    if rrt_bounds is not None:
        logger.info(f"Using experiment RRT bounds: {rrt_bounds[0]} to {rrt_bounds[1]}")

    object_trajectory = generate_object_trajectory(
        start_pose_stamped, 
        target_pose_stamped, 
        num_steps=traj_num_steps,
        tf_manager=tf_manager,
        env=env,
        object_name=object_name,
        env_lock=env_lock,
        pos_bounds=rrt_bounds,
    )
    if vis is not None:
        visualize_trajectory(vis, object_trajectory, name="object_path", color=0x00ff00)
    
    # Visualize object meshes along trajectory (once, shared across all grasps)
    if vis is not None and vis_cfg.visualize_mesh:
        logger.info("Visualizing object meshes along trajectory...")
        obj_mesh_visualizer = ObjectMeshVisualizer(
            vis=vis, tf_manager=tf_manager,
            mesh_sample_ratio=vis_cfg.mesh_sample_ratio
        )
        obj_mesh_visualizer.visualize_trajectory(
            object_name=object_name,
            object_trajectory=object_trajectory,
            step_interval=vis_cfg.mesh_step_interval,
            opacity=vis_cfg.mesh_opacity
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
        visualize=False 
    )
    grasp_service.start()
    
    # Initialize Grasp Generator
    grasp_generator = GraspGenerator(env=env)
    
    # Step 6a: Get grasps for both robots together
    grasp_gripper_types = plan_cfg.gripper_types
    grasp_num = plan_cfg.num_grasps
    grasp_entropy_n = plan_cfg.num_entropy_grasps
    grasp_collision_mode = plan_cfg.collision_check
    
    logger.info("Generating grasps for %s (types: %s)...", object_name, grasp_gripper_types)
    all_grasps = grasp_generator.generate_grasps(
        object_name=object_name, 
        gripper_types=grasp_gripper_types, 
        collision_check=grasp_collision_mode,
        num_grasps=grasp_num,
        env_lock=env_lock
    )
    
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)
    
    grasps_0 = maximize_entropy_grasps(grasps_0, num_grasps=grasp_entropy_n)
    grasps_1 = maximize_entropy_grasps(grasps_1, num_grasps=grasp_entropy_n)
    
    logger.info(f"Received {len(grasps_0)} grasps for Finger and {len(grasps_1)} for Suction.")

    # 6. Evaluate Robot Trajectories in Parallel
    # ------------------------------------------
    logger.info(f"Evaluating grasps using {num_workers} processes...")
    start_time = time.time()

    vis_args = {
        'visualize_mesh': vis_cfg.visualize_mesh if enable_meshcat else False,
        'mesh_step_interval': vis_cfg.mesh_step_interval,
        'mesh_opacity': vis_cfg.mesh_opacity,
        'mesh_sample_ratio': vis_cfg.mesh_sample_ratio,
        'enable_meshcat': enable_meshcat,
    }

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(ycb_objects, custom_objects, log_level, collision_exclude_patterns, table_full_size, enable_meshcat)
    ) as pool:
        
        # Evaluate Robot 0 Grasps
        logger.info("Processing Robot 0 (Finger) grasps...")
        robot0_tasks = [(i, g0) for i, g0 in enumerate(grasps_0)]
        eval_func0 = partial(
            evaluate_robot_trajectory, 
            robot_id=0, 
            object_trajectory=object_trajectory, 
            object_name=object_name, 
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
            object_name=object_name, 
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
            feasible_waypoints_count += 1
        elif r1_feasible:
            status = "FEASIBLE (ROBOT 1 ONLY)"
            feasible_waypoints_count += 1
        else:
            status = "INFEASIBLE (NONE)"
            
        logger.info(f"  Waypoint {k:2d}: {status}")

    logger.info(f"Collective analysis: {feasible_waypoints_count}/{num_waypoints} waypoints can be reached by AT LEAST one grasp from BOTH robots.")

    # 9. Save Trajectories
    # ---------------------------------------------
    # Determine output directory: JSON output_dir > data/<experiment_name>
    if out_cfg.output_dir is not None:
        output_dir = out_cfg.output_dir
    else:
        output_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)
    
    should_save = args.save_traj or out_cfg.save_trajectories
    
    if not should_save:
        try:
            if sys.stdin.isatty():
                user_input = input("Save trajectories? [y/N]: ").strip().lower()
                if user_input in ('y', 'yes'):
                    should_save = True
        except EOFError:
            logger.error("No input available to confirm saving trajectories. Skipping save.")
            
    if should_save and (robot0_trajectory_results or robot1_trajectory_results):
        os.makedirs(output_dir, exist_ok=True)
        # Save object trajectory
        save_trajectory(
            output_dir=output_dir,
            filename="object_trajectory.npz",
            object_trajectory=object_trajectory
        )
        
        # Save Robot 0 trajectories
        for traj_data in robot0_trajectory_results:
            i = traj_data['id']
            save_trajectory(
                output_dir=output_dir,
                filename=f"robot0_trajectory_{i}.npz",
                eef_trajectory=traj_data['traj_eef'],
                object_trajectory=object_trajectory,
                waypoint_results=traj_data['results']
            )
            
        # Save Robot 1 trajectories
        for traj_data in robot1_trajectory_results:
            i = traj_data['id']
            save_trajectory(
                output_dir=output_dir,
                filename=f"robot1_trajectory_{i}.npz",
                eef_trajectory=traj_data['traj_eef'],
                object_trajectory=object_trajectory,
                waypoint_results=traj_data['results']
            )
        
        logger.info(f"Saved {len(robot0_trajectory_results)} Robot 0 and {len(robot1_trajectory_results)} Robot 1 trajectories to {output_dir}")
    else:
        logger.info("No trajectories saved.")
        
    logger.info("Closing simulation...")
    stop_event.set()
    if vis_thread is not None:
        vis_thread.join(timeout=2.0)
    
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
