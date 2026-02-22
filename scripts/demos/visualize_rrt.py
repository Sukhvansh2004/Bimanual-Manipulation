#!/usr/bin/env python3
"""
Demo: Visualize RRT planning for an experiment in Meshcat.

Loads an experiment JSON, creates the environment, detects (or uses)
the start pose, and runs the RRT-Connect object trajectory planner
with real-time Meshcat visualization of:

  * search bounds (grey wireframe box)
  * start (white) / goal (magenta) markers
  * tree-a edges (cyan) and tree-b edges (orange)
  * collision-rejected samples (small red dots)
  * raw RRT path (green) and smoothed path (yellow)

Usage:
    # Start meshcat-server in another terminal first
    python scripts/demos/visualize_rrt.py \
        --experiment experiments/flatPlate_table_to_shelf.json

    # Slow down drawing for real-time watching
    python scripts/demos/visualize_rrt.py \
        --experiment experiments/flatPlate_table_to_shelf.json \
        --delay 0.01

    # Override RRT parameters
    python scripts/demos/visualize_rrt.py \
        --experiment experiments/flatPlate_table_to_shelf.json \
        --max-iter 2000 --step-size 0.03
"""

import argparse
import logging
import sys
import time
import threading

import meshcat
import numpy as np

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import (
    TransformManager,
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.utils.env_utils import env_step
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.experiment_config import (
    load_experiment,
    detect_start_pose,
)
from robosuite_franka_manipulation.planning.object_trajectory_planner import (
    plan_object_trajectory,
    RRTVisualizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RRT-Connect planning in Meshcat"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment JSON config",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each Meshcat draw call (default: 0, set >0 for slow-mo)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override RRT max iterations (default: from constants)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Override RRT step size (default: from constants)",
    )
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=None,
        help="Override RRT goal bias (default: from constants)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip path smoothing",
    )
    parser.add_argument(
        "--zmq-url",
        type=str,
        default=None,
        help="Meshcat ZMQ URL (default: from experiment or constants)",
    )
    parser.add_argument(
        "--no-scene",
        action="store_true",
        help="Skip streaming the full MuJoCo scene to Meshcat (faster startup)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load experiment
    # ------------------------------------------------------------------
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    zmq_url = args.zmq_url or exp_cfg.visualization.zmq_url

    # ------------------------------------------------------------------
    # 2. Create environment
    # ------------------------------------------------------------------
    logger.info("Creating environment (%d YCB + %d custom objects)...",
                len(exp_cfg.ycb_objects), len(exp_cfg.custom_objects))

    env = make_dual_arm_env(
        ycb_objects=exp_cfg.ycb_objects,
        custom_objects=exp_cfg.custom_objects,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        table_full_size=exp_cfg.table_full_size,
    )
    env.reset()
    for _ in range(10):
        env_step(env)

    env_lock = threading.RLock()
    tf_manager = TransformManager(env, env_lock=env_lock)

    # ------------------------------------------------------------------
    # 3. (Optional) Stream MuJoCo scene to Meshcat
    # ------------------------------------------------------------------
    stop_event = threading.Event()
    if not args.no_scene:
        logger.info("Streaming scene to Meshcat (%s) ...", zmq_url)
        vis_cfg = MeshcatTFVisualizerConfig(
            fps=10.0, clear=True, labels=False, visualize_scene=True,
        )
        vis_thread = threading.Thread(
            target=stream_transforms_to_meshcat,
            kwargs={
                "env": env,
                "env_lock": env_lock,
                "config": vis_cfg,
                "stop_event": stop_event,
                "tf_manager": tf_manager,
            },
            daemon=True,
        )
        vis_thread.start()
        # Give it a moment to render the scene
        time.sleep(2.0)
    else:
        logger.info("Skipping scene streaming (--no-scene).")

    # ------------------------------------------------------------------
    # 4. Resolve start & target poses
    # ------------------------------------------------------------------
    object_name = exp_cfg.task.manipulated_object

    if exp_cfg.task.start_pose is not None:
        start_pos = exp_cfg.task.start_pose.position
        start_quat = exp_cfg.task.start_pose.quaternion
        logger.info("Using experiment start pose: %s", start_pos)
    else:
        logger.info("Auto-detecting start pose from simulation...")
        time.sleep(0.5)
        detected = detect_start_pose(env, object_name)
        if detected is None:
            logger.error("Could not detect start pose for '%s'. Exiting.", object_name)
            sys.exit(1)
        start_pos = detected.position
        start_quat = detected.quaternion
        logger.info("Detected start pose: %s", start_pos)

    target_pos = exp_cfg.task.target_pose.position
    target_quat = exp_cfg.task.target_pose.quaternion

    start_stamped = StampedPose(position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME)
    target_stamped = StampedPose(position=target_pos, quaternion=target_quat, frame_id=constants.WORLD_FRAME)

    logger.info("Planning: %s -> %s", start_pos, target_pos)

    # ------------------------------------------------------------------
    # 5. Create RRT Meshcat visualizer
    # ------------------------------------------------------------------
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    rrt_vis = RRTVisualizer(vis, delay=args.delay)

    # ------------------------------------------------------------------
    # 6. Run RRT planning
    # ------------------------------------------------------------------
    logger.info("Running RRT-Connect planner with Meshcat visualization...")
    t0 = time.time()

    trajectory, success = plan_object_trajectory(
        env=env,
        object_name=object_name,
        start_pose=start_stamped,
        target_pose=target_stamped,
        tf_manager=tf_manager,
        num_steps=exp_cfg.planning.num_steps,
        env_lock=env_lock,
        max_iterations=args.max_iter,
        step_size=args.step_size,
        goal_bias=args.goal_bias,
        smoothing_iterations=0 if args.no_smooth else None,
        rrt_vis=rrt_vis,
    )

    elapsed = time.time() - t0

    if success:
        logger.info("SUCCESS – trajectory with %d waypoints found in %.2fs", len(trajectory), elapsed)
    else:
        logger.warning("FAILED – RRT could not find a path (%.2fs). "
                       "Returned fallback interpolation.", elapsed)

    # ------------------------------------------------------------------
    # 7. Keep alive so the user can inspect Meshcat
    # ------------------------------------------------------------------
    logger.info("Visualization is live in Meshcat. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        env.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
