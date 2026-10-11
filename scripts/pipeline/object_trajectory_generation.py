#!/usr/bin/env python3
"""Standalone object trajectory generation with mesh visualization.

This script generates only the object trajectory (RRT-based path from start to
target pose) without any grasp generation or EEF trajectory evaluation. The
result is optionally saved to ``object_trajectory.npz`` in the experiment's
output directory (``output.output_dir`` or ``DATA_DIR/<experiment name>``).

Usage:
    python object_trajectory_generation.py --experiment experiments/flatPlate_table_to_shelf.json
    python object_trajectory_generation.py --experiment experiments/mug_inversion.json --save
    python object_trajectory_generation.py --experiment experiments/ycb_plate_table_to_shelf.json --no-meshcat
"""

import numpy as np
import os
import time
import threading
import logging
import argparse
import sys

import meshcat
import meshcat.geometry as g

from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.transforms import (
    TransformManager,
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.object_factory import register_mesh_extraction_env
from bimanual_manipulation import constants
from bimanual_manipulation.transforms.stamped_pose import StampedPose

from bimanual_manipulation.planning.trajectory import generate_object_trajectory
from bimanual_manipulation.planning.trajectory.utils import save_trajectory
from bimanual_manipulation.planning.visualization.trajectory import visualize_trajectory
from bimanual_manipulation.planning.visualization.object_mesh import ObjectMeshVisualizer
from bimanual_manipulation.experiment_config import (
    load_experiment,
    detect_start_pose,
)


def setup_logging(level: int = logging.INFO):
    """Configure root logging with a timestamped format.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.

    Returns:
        ``argparse.Namespace`` with ``experiment``, ``save``, ``no_meshcat``
        and ``debug``.
    """
    parser = argparse.ArgumentParser(
        description="Generate an object trajectory from an experiment config and visualize it."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment JSON config file.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the object trajectory to an NPZ file.",
    )
    parser.add_argument(
        "--no-meshcat",
        action="store_true",
        default=False,
        help="Disable Meshcat visualization.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main():
    """Generate, visualize and optionally save the object trajectory.

    Builds the dual-arm environment from the experiment config, resolves the
    start pose (from the config or by auto-detection), plans the object path
    with ``generate_object_trajectory`` and renders it in Meshcat if enabled.
    Without ``--save``, the user is prompted interactively before saving.

    Raises:
        SystemExit: If no start pose is configured and auto-detection fails.
    """
    args = parse_args()

    # ── Load experiment config ──────────────────────────────────────────
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    vis_cfg = exp_cfg.visualization
    out_cfg = exp_cfg.output
    plan_cfg = exp_cfg.planning
    rt_cfg = exp_cfg.runtime

    debug = args.debug or rt_cfg.debug
    enable_meshcat = vis_cfg.enable_meshcat and not args.no_meshcat

    setup_logging(logging.DEBUG if debug else logging.INFO)

    # ── Create environment ──────────────────────────────────────────────
    object_name = exp_cfg.task.manipulated_object
    logger.info(
        "Creating environment with %d YCB + %d custom objects...",
        len(exp_cfg.ycb_objects),
        len(exp_cfg.custom_objects),
    )

    env = make_dual_arm_env(
        ycb_objects=exp_cfg.ycb_objects,
        custom_objects=exp_cfg.custom_objects,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        table_full_size=exp_cfg.table_full_size,
        table_offset=exp_cfg.table_offset,
        robot_base_configs=exp_cfg.robot_base_configs,
    )
    env.reset()
    for _ in range(10):
        env_step(env)

    register_mesh_extraction_env(env)

    env_lock = threading.RLock()
    tf_manager = TransformManager(env, env_lock=env_lock)

    # ── Meshcat scene streaming ─────────────────────────────────────────
    stop_event = threading.Event()
    vis_thread = None
    vis = None

    if enable_meshcat:
        logger.info("Starting Meshcat scene stream...")
        vis_config = MeshcatTFVisualizerConfig(
            fps=20.0, clear=True, labels=True, visualize_scene=True
        )
        vis_thread = threading.Thread(
            target=stream_transforms_to_meshcat,
            kwargs={
                "env": env,
                "env_lock": env_lock,
                "config": vis_config,
                "stop_event": stop_event,
                "tf_manager": tf_manager,
            },
            daemon=True,
        )
        vis_thread.start()
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
    else:
        logger.info("Meshcat visualization disabled.")

    # ── Resolve start / target poses ────────────────────────────────────
    if exp_cfg.task.start_pose is not None:
        start_pos = exp_cfg.task.start_pose.position
        start_quat = exp_cfg.task.start_pose.quaternion
        logger.info("Using experiment start pose: pos=%s", start_pos)
    else:
        logger.info("No start pose specified. Auto-detecting from simulation...")
        time.sleep(1.0)
        detected = detect_start_pose(env, object_name)
        if detected is not None:
            start_pos = detected.position
            start_quat = detected.quaternion
            logger.info("Detected object pose: pos=%s", start_pos)
        else:
            logger.error("Auto-detection failed and no start_pose in experiment JSON.")
            sys.exit(1)

    start_pose = StampedPose(
        position=start_pos, quaternion=start_quat, frame_id=constants.WORLD_FRAME
    )

    target_pos = exp_cfg.task.target_pose.position
    target_quat = exp_cfg.task.target_pose.quaternion
    target_pose = StampedPose(
        position=target_pos, quaternion=target_quat, frame_id=constants.WORLD_FRAME
    )

    logger.info("Planning from %s to %s", start_pose.position, target_pose.position)

    # ── Generate object trajectory (RRT) ────────────────────────────────
    rrt_bounds = plan_cfg.rrt_bounds
    if rrt_bounds is not None:
        logger.info("RRT bounds: %s to %s", rrt_bounds[0], rrt_bounds[1])

    t0 = time.time()
    object_trajectory = generate_object_trajectory(
        start_pose,
        target_pose,
        num_steps=plan_cfg.num_steps,
        tf_manager=tf_manager,
        env=env,
        object_name=object_name,
        env_lock=env_lock,
        pos_bounds=rrt_bounds,
    )
    elapsed = time.time() - t0
    logger.info(
        "Object trajectory generated: %d waypoints in %.2f s", len(object_trajectory), elapsed
    )

    # ── Visualize ───────────────────────────────────────────────────────
    if vis is not None:
        # Trajectory line + markers
        visualize_trajectory(vis, object_trajectory, name="object_path", color=0x00ff00)

        # Object meshes along trajectory
        if vis_cfg.visualize_mesh:
            logger.info("Rendering object meshes along trajectory...")
            obj_mesh_viz = ObjectMeshVisualizer(
                vis=vis,
                tf_manager=tf_manager,
                mesh_sample_ratio=vis_cfg.mesh_sample_ratio,
            )
            obj_mesh_viz.visualize_trajectory(
                object_name=object_name,
                object_trajectory=object_trajectory,
                step_interval=vis_cfg.mesh_step_interval,
                opacity=vis_cfg.mesh_opacity,
            )

    # ── Save ────────────────────────────────────────────────────────────
    if args.save or input("Save object trajectory to NPZ? [y/N]: ").lower() == "y":
        if out_cfg.output_dir is not None:
            output_dir = out_cfg.output_dir
        else:
            output_dir = os.path.join(constants.DATA_DIR, exp_cfg.name)

        save_trajectory(
            output_dir=output_dir,
            filename="object_trajectory.npz",
            object_trajectory=object_trajectory,
        )
        logger.info("Saved object trajectory to %s/object_trajectory.npz", output_dir)


    logger.info("Done.")


if __name__ == "__main__":
    main()
