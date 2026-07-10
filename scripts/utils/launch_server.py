#!/usr/bin/env python3
"""
Launch the GPU service server for remote clients.

Starts the ServiceManager, FastSAM, and Grasp services so that a
client-only DualArmManipulationSystem on another machine (or one
without a GPU) can connect via ``server_host``/``server_port``.
Optionally starts the MoveToPose service as well.

Usage:
    python scripts/utils/launch_server.py
    python scripts/utils/launch_server.py --host 0.0.0.0 --port 6006
    python scripts/utils/launch_server.py --no-fastsam          # grasp only
    python scripts/utils/launch_server.py --no-grasp             # fastsam only
    python scripts/utils/launch_server.py --move-to-pose         # include MoveToPose
"""

import argparse
import logging
import signal
import sys
import threading
import time

from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.services.fastsam_service import FastSAMService
from bimanual_manipulation.services.grasp_service import GraspService
from bimanual_manipulation.services.services_core import ServiceManager
from bimanual_manipulation.services.services_core import ServiceClient
from bimanual_manipulation.services.move_to_pose_service import MoveToPoseService
from bimanual_manipulation.controllers.arm import RobosuiteArmController
from bimanual_manipulation.transforms import TransformManager
from bimanual_manipulation.utils.env_utils import env_step

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch GPU service server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address for the ServiceManager (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=constants.SERVICE_MANAGER_PORT,
                        help=f"ServiceManager port (default: {constants.SERVICE_MANAGER_PORT})")
    parser.add_argument("--no-fastsam", action="store_true",
                        help="Do not start FastSAM service")
    parser.add_argument("--no-grasp", action="store_true",
                        help="Do not start Grasp service")
    parser.add_argument("--move-to-pose", action="store_true",
                        help="Also start the MoveToPose service (requires arm controllers)")
    parser.add_argument("--detection-prompt", default="object",
                        help="Initial FastSAM detection prompt")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- Environment (headless, offscreen only for camera rendering) ----------
    logger.info("Creating headless environment for services...")
    env = make_dual_arm_env(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env.reset()
    for _ in range(10):
        env_step(env)

    env_lock = threading.RLock()
    tf_manager = TransformManager(env=env, auto_discover=True, env_lock=env_lock)

    # -- ServiceManager -------------------------------------------------------
    logger.info("Starting ServiceManager on %s:%d ...", args.host, args.port)
    manager = ServiceManager(host=args.host, port=args.port)
    manager.start()
    time.sleep(0.5)

    # -- GPU services ---------------------------------------------------------
    fastsam_service = None
    grasp_service = None
    move_service = None

    if not args.no_fastsam:
        logger.info("Starting FastSAM service...")
        fastsam_service = FastSAMService(
            tf_manager=tf_manager,
            manager_host=args.host,
            manager_port=args.port,
        )
        fastsam_service.start()
        time.sleep(0.5)

        client = ServiceClient(manager_host="localhost", manager_port=args.port)
        client.call("fastsam_service", {
            "command": "SET_PROMPT",
            "prompt": args.detection_prompt,
        })
        client.close()
        logger.info("FastSAM service started (prompt: '%s')", args.detection_prompt)

    if not args.no_grasp:
        logger.info("Starting Grasp service...")
        grasp_service = GraspService(
            tf_manager=tf_manager,
            manager_host=args.host,
            manager_port=args.port,
        )
        grasp_service.start()
        time.sleep(0.5)
        logger.info("Grasp service started")

    # -- MoveToPose service (optional) ----------------------------------------
    if args.move_to_pose:
        logger.info("Starting MoveToPose service...")
        left_arm = RobosuiteArmController(
            env, arm_id="right", robot_id=0, env_lock=env_lock,
            transform_manager=tf_manager,
        )
        right_arm = RobosuiteArmController(
            env, arm_id="right", robot_id=1, env_lock=env_lock,
            transform_manager=tf_manager,
        )
        move_service = MoveToPoseService(
            manager_host=args.host,
            manager_port=args.port,
        )
        move_service.register_controller(0, "right", left_arm)
        move_service.register_controller(1, "right", right_arm)
        move_service.start()
        time.sleep(0.5)
        logger.info("MoveToPose service started")

    # -- Wait for shutdown ----------------------------------------------------
    stop = threading.Event()

    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Server ready. Press Ctrl+C to stop.")
    stop.wait()

    # Cleanup
    if move_service:
        move_service.stop()
    if fastsam_service:
        fastsam_service.stop()
    if grasp_service:
        grasp_service.stop()
    manager.stop()
    env.close()
    logger.info("Server stopped.")


if __name__ == "__main__":
    main()
