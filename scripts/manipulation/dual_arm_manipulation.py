#!/usr/bin/env python3
"""
Start Services and Interactive Teleoperation
"""

import argparse
import threading
import time
import logging
import code
import atexit
import os
import signal
import sys
from typing import Any, Optional, List
import numpy as np

from robosuite_franka_manipulation import (
    DualArmManipulationSystem, detect, get_grasps, get_grasps_mesh, get_eef, 
    set_prompt, go_to_pose, load_trajectory_file, visualize_trajectory_data, 
    cancel_move, get_arm_status, follow_eef_trajectory,
    constants, StampedPose, create_ycb_object, env_step
)

# Configure logging
logging.basicConfig(
    level=constants.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start dual-arm manipulation system with services and interactive shell"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Default detection prompt"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable OpenCV visualization"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--meshcat-tf",
        action="store_true",
        help="Stream the environment TF tree to MeshCat (requires `meshcat-server`)"
    )
    parser.add_argument(
        "--no-fastsam",
        action="store_false",
        dest="use_fastsam",
        help="Disable FastSAM service",
        default=True,
    )
    parser.add_argument(
        "--no-graspgen",
        action="store_false",
        dest="use_graspgen",
        help="Disable GraspGen service",
        default=True,
    )
    
    args = parser.parse_args()
    
    ycb_objects = []
    ycb_objects.append(create_ycb_object(ycb_name="plate"))
    ycb_objects.append(create_ycb_object(ycb_name="shelf", position=[-0.7, 0.0, 0.0], quaternion=[0.0, 0.0, 0.70710678, 0.70710678]))

    # Create and start system
    global system
    system = DualArmManipulationSystem(
        visualize=args.visualize,
        debug=args.debug,
        detection_prompt=args.prompt,
        meshcat_tf=args.meshcat_tf,
        ycb_objects=ycb_objects,
        use_fastsam=args.use_fastsam,
        use_grasp_service=args.use_graspgen,
    )

    atexit.register(lambda: system.stop())

    def _handle_signal(signum: int, _frame: Any) -> None:
        system.request_shutdown(reason=f"signal {signal.Signals(signum).name}")

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass
    
    try:
        system.start(start_teleop_thread=False)
        time.sleep(1.0)
        
        banner = """
Interactive Python Shell - Dual Arm Manipulation System
========================================================
Helper functions:
  detect()                               - Detect objects
  get_grasps(detection, gripper_type)    - Generate grasps for detection
  get_eef()                              - Get end-effector coordinates
  set_prompt(text)                       - Change detection prompt
  go_to_pose(pose, robot_id, arm_id)     - Move an arm to a pose
  follow_eef_trajectory(robot_id, arm, traj) - Follow an EEF trajectory
  cancel_move(robot_id, arm_id)          - Cancel a move command
  get_arm_status(robot_id, arm_id)       - Get arm status (-1: moving, 1: success, 0: fail)
========================================================
"""
        
        client = system.service_client

        def shell_thread():
            namespace = {
                'client': client,
                'system': system,
                'detect': lambda: detect(system),
                'get_grasps': lambda d, g, n=200, c=None: get_grasps(system, d, g, n, c),
                'get_grasps_mesh': lambda p, o, n=200, g='suction', c=None, v=True: get_grasps_mesh(system, p, o, n, g, c, v),
                'get_eef': lambda: get_eef(system),
                'set_prompt': lambda p: set_prompt(system, p),
                'go_to_pose': lambda p, r, a, g=0.0, n=None: go_to_pose(system, p, r, a, g, n),
                'cancel_move': lambda r, a: cancel_move(system, r, a),
                'get_arm_status': lambda r, a: get_arm_status(system, r, a),
                'follow_eef_trajectory': lambda r, a, t, g=0.0, n=None: follow_eef_trajectory(system, r, a, t, g, n),
                'load_trajectory_file': load_trajectory_file,
                'visualize_trajectory_data': lambda t, w=None, n="trajectory": visualize_trajectory_data(system, t, w, n),
                'env_step': lambda a=None, r=None, arm=None: env_step(system.env, a, r, arm),
                'StampedPose': StampedPose,
            }
            code.interact(banner=banner, local=namespace)

        t = threading.Thread(target=shell_thread, daemon=True)
        t.start()
        system.run_teleop(blocking=True)

    except KeyboardInterrupt:
        return 0
    finally:
        system.stop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
