#!/usr/bin/env python3
"""
Dual-arm planning script driven by experiment JSON configs.

This script:
1. Loads an experiment configuration from a JSON file.
2. Initializes the DualArmEnvironment with the specified objects.
3. Generates grasps for both robots and pairs them (N x M).
4. Evaluates simultaneous feasibility for each grasp pair using multiprocessing.
   Optimization: Environments are created once per worker process.
5. Visualizes the planning context and generated trajectories in Meshcat.
6. Saves ALL evaluated trajectories to disk.

Usage:
    python dual_trajectory_generation.py --experiment experiments/flatPlate_table_to_shelf.json
    python dual_trajectory_generation.py --experiment experiments/flatPlate_shelf_to_box.json --save-traj
"""

import numpy as np
import os
import time
import threading
import logging
import random
import meshcat
import meshcat.geometry as g
import argparse
import sys
import multiprocessing
from functools import partial

import meshcat.transformations as tf
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.object_factory import (
    create_ycb_object,
    create_primitive_object,
    register_mesh_extraction_env,
)
from bimanual_manipulation import constants
from bimanual_manipulation.transforms.stamped_pose import StampedPose

# Services and Planning
from bimanual_manipulation.services.grasp_service import GraspService
from bimanual_manipulation.services.services_core import ServiceManager
from bimanual_manipulation.services.grasp_generator import GraspGenerator

# Planning Modules
from bimanual_manipulation.planning.trajectory import (
    generate_end_effector_trajectory, 
    generate_object_trajectory,
)
from bimanual_manipulation.planning.collision_checker import (
    check_dual_trajectory_feasibility,
)
from bimanual_manipulation.perception.grasp_predictor import maximize_entropy_grasps
from bimanual_manipulation.planning.trajectory.utils import save_trajectory
from bimanual_manipulation.planning.visualization.trajectory import visualize_trajectory
from bimanual_manipulation.planning.visualization.robot_mesh import RobotMeshVisualizer
from bimanual_manipulation.experiment_config import (
    load_experiment, detect_start_pose, get_collision_filter_fn, ExperimentConfig
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

def init_worker(ycb_objects, custom_objects, log_level, collision_relevant_patterns=None, table_full_size=None):
    """Initialize worker process by creating a persistent environment.
    
    Args:
        ycb_objects: YCB object specs for the environment.
        custom_objects: Custom object specs for the environment.
        log_level: Logging level.
        collision_relevant_patterns: List of substrings; geoms whose names contain
            any of these patterns (or are robot geoms) keep their collisions.
            All other geoms get contype=0 for faster collision checking.
        table_full_size: Optional tuple (x, y, z) to override the default table dimensions.
    """
    setup_logging(log_level)
    global worker_env, worker_tf_manager
    if collision_relevant_patterns is None:
        collision_relevant_patterns = []
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
        _is_relevant = get_collision_filter_fn(
            type('CF', (), {'relevant_patterns': collision_relevant_patterns})()
        )
        model = worker_env.sim.model
        for i in range(model.ngeom):
            name = model.geom_id2name(i) or ""
            if not _is_relevant(name):
                model.geom_contype[i] = 0

        worker_tf_manager = TransformManager(worker_env)
        logging.info("Worker initialized: Environment created.")
    except Exception as e:
        logging.error(f"Worker initialization failed: {e}")

def evaluate_pair(pair_info, object_trajectory, object_name, collision_ignore_waypoints=None):
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
            object_name=object_name,
            tf_manager=worker_tf_manager,
            ignore_names=["geo_snout"],
            collision_ignore_waypoints=collision_ignore_waypoints,
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
    parser = argparse.ArgumentParser(description="Generate and visualize dual-arm manipulation trajectory")
    
    # Experiment configuration (required)
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    
    # Minimal CLI overrides – everything else lives in the experiment JSON
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (overrides JSON)")
    parser.add_argument("--save-traj", action="store_true", help="Force-save trajectories (overrides JSON)")
    parser.add_argument("--no-meshcat", action="store_true",
                        help="Disable all Meshcat visualization (useful for batch / headless runs)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (overrides JSON)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load experiment configuration
    # ---------------------------------------------------------------
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    # Set seeds if provided
    if args.seed is not None:
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Set random seeds to {args.seed}")

    # Resolve runtime settings: CLI overrides > JSON
    rt_cfg = exp_cfg.runtime
    vis_cfg = exp_cfg.visualization
    out_cfg = exp_cfg.output
    plan_cfg = exp_cfg.planning

    debug = args.debug or rt_cfg.debug
    enable_meshcat = vis_cfg.enable_meshcat and not args.no_meshcat
    
    if args.workers is not None:
        num_workers = args.workers
    else:
        num_workers = rt_cfg.num_workers if rt_cfg.num_workers > 0 else multiprocessing.cpu_count() // 2

    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)

    logger.debug(f"Parsed arguments: {args}")
    logger.info("Initializing Dual Trajectory Generation...")
    
    # 1. Setup Environment
    # --------------------
    ycb_objects = exp_cfg.ycb_objects
    custom_objects = exp_cfg.custom_objects
    object_name = exp_cfg.task.manipulated_object
    collision_relevant_patterns = exp_cfg.collision_filter.relevant_patterns
    table_full_size = exp_cfg.table_full_size
    
    logger.info("Creating planning environment with %d YCB + %d custom objects...",
                len(ycb_objects), len(custom_objects))
    
    # Environment for the main process (visualization and grasp generation)
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
    
    env_lock = threading.RLock()
    tf_manager = TransformManager(env, env_lock=env_lock)
    
    # 2. Setup Visualization
    # ----------------------
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
    else:
        logger.info("Meshcat visualization disabled.")
    
    if enable_meshcat:
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
        logger.info("Visualization stream started.")
    
    # 3. Define Planning Problem
    # --------------------------
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
    # -----------------------------
    logger.info("Generating object trajectory...")
    object_trajectory = generate_object_trajectory(
        start_pose_stamped, target_pose_stamped,
        num_steps=plan_cfg.num_steps,
        tf_manager=tf_manager,
        env=env,
        object_name=object_name,
        env_lock=env_lock,
    )
    if enable_meshcat:
        visualize_trajectory(vis, object_trajectory, name="object_path", color=0x00ff00)
    
    # Visualize Target
    if enable_meshcat:
        vis["target_pose"].set_object(g.Sphere(0.05), g.MeshBasicMaterial(color=0xff0000, opacity=0.5))
        T_target = np.eye(4)
        T_target[:3, 3] = target_pose_stamped.position
        T_target[:3, :3] = tf.quaternion_matrix(T.convert_quat(target_pose_stamped.quaternion, to="wxyz"))[:3, :3]
        vis["target_pose"].set_transform(T_target)
    
    logger.info("Object trajectory generated.")

    # 5. Generate Grasps
    # ------------------
    logger.info("Initializing Service Architecture...")
    service_manager = ServiceManager()
    service_manager.start()
    grasp_service = GraspService(tf_manager=tf_manager, visualize=True, use_dpp=plan_cfg.use_dpp)
    grasp_service.start()
    grasp_generator = GraspGenerator(env=env)
    
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
        env_lock=env_lock,
    )
    grasps_0 = sorted([g for g in all_grasps if g.gripper_type in ('finger', constants.GRIPPER_TYPE_MAPPING['finger'])], key=lambda g: g.score, reverse=True)
    grasps_1 = sorted([g for g in all_grasps if g.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction'])], key=lambda g: g.score, reverse=True)
    
    if plan_cfg.use_dpp:
        grasps_0 = maximize_entropy_grasps(grasps_0, num_grasps=grasp_entropy_n)
        grasps_1 = maximize_entropy_grasps(grasps_1, num_grasps=grasp_entropy_n)
    else:
        grasps_0 = grasps_0[:grasp_entropy_n]
        grasps_1 = grasps_1[:grasp_entropy_n]

    logger.info(f"Selection: {len(grasps_0)} Finger grasps, {len(grasps_1)} Suction grasps.")

    # 6. Evaluate Pairs in Parallel
    # -----------------------------
    grasp_pairs = []
    for i, g0 in enumerate(grasps_0):
        for j, g1 in enumerate(grasps_1):
            grasp_pairs.append((i, j, g0, g1))
            
    logger.info(f"Evaluating {len(grasp_pairs)} grasp pairs using {num_workers} processes...")
    start_time = time.time()
    
    # Use multiprocessing pool with initializer to create environments once per process
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(ycb_objects, custom_objects, log_level, collision_relevant_patterns, table_full_size)
    ) as pool:
        eval_func = partial(evaluate_pair, object_trajectory=object_trajectory, object_name=object_name, collision_ignore_waypoints=plan_cfg.collision_ignore_waypoints)
        all_results = pool.map(eval_func, grasp_pairs)
    
    # Filter out None results from workers
    all_results = [r for r in all_results if r is not None]
    
    duration = time.time() - start_time
    feasible_count = sum(1 for r in all_results if r['all_feasible'])
    logger.info(f"Evaluation complete in {duration:.2f}s.")
    logger.info(f"Results: {feasible_count}/{len(all_results)} pairs are fully feasible.")

    # 7. Visualization of Results
    # ---------------------------
    if enable_meshcat:
        logger.info("Streaming evaluated trajectories to Meshcat...")
        for res in all_results:
            pair_id = f"pair_{res['id0']}_{res['id1']}"
            status = "FEASIBLE" if res['all_feasible'] else "INFEASIBLE"
            logger.debug(f"Visualizing {pair_id} ({status})")
            
            # Visualize Robot 0 Path
            visualize_trajectory(vis, res['traj0_eef'], waypoint_results=res['results0'], name=f"robot0_{pair_id}", color=0x888888)
            # Visualize Robot 1 Path
            visualize_trajectory(vis, res['traj1_eef'], waypoint_results=res['results1'], name=f"robot1_{pair_id}", color=0x888888)
            
            if vis_cfg.visualize_mesh:
                logger.info(f"Rendering robot meshes for feasible pair {pair_id}")
                mesh_vis = RobotMeshVisualizer(vis=vis, tf_manager=tf_manager, base_path=f"dual_configs/{pair_id}")
                mesh_vis.visualize_dual_arm_trajectory(
                    trajectory_id="feasible",
                    waypoint_results_robot0=res['results0'],
                    waypoint_results_robot1=res['results1'],
                    step_interval=vis_cfg.mesh_step_interval,
                    opacity=vis_cfg.mesh_opacity
                )
    else:
        logger.info("Skipping trajectory visualization (meshcat disabled).")

    # 8. Save Trajectories
    # --------------------
    # Determine output directory: JSON output_dir > data/<experiment_name>
    if out_cfg.output_dir is not None:
        output_dir = out_cfg.output_dir
    else:
        output_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)

    should_save = args.save_traj or out_cfg.save_trajectories
    if not should_save and sys.stdin.isatty():
        try:
            should_save = input("\nSave ALL evaluated trajectories to disk? [y/N]: ").strip().lower() in ('y', 'yes')
        except EOFError:
            pass
    
    if should_save:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving {len(all_results)} trajectory pairs to {output_dir}...")
        save_trajectory(output_dir, "object_trajectory.npz", object_trajectory=object_trajectory)
        for res in all_results:
            pair_id = f"{res['id0']}_{res['id1']}"
            save_trajectory(output_dir, f"dual_robot0_pair_{pair_id}.npz", eef_trajectory=res['traj0_eef'], object_trajectory=object_trajectory, waypoint_results=res['results0'])
            save_trajectory(output_dir, f"dual_robot1_pair_{pair_id}.npz", eef_trajectory=res['traj1_eef'], object_trajectory=object_trajectory, waypoint_results=res['results1'])
        logger.info("All trajectories saved.")
    else:
        logger.info("Saving skipped.")

    # Cleanup
    logger.info("Shutting down...")
    stop_event.set()
    if vis_thread is not None:
        vis_thread.join(timeout=2.0)
    grasp_service.stop()
    service_manager.stop()
    env.close()

    logger.info("Done.")
    sys.exit(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
