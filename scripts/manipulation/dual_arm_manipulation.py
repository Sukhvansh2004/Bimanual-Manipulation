#!/usr/bin/env python3
"""
Dual-arm manipulation script with perception integration.
This script performs object detection, grasp prediction, and coordinated dual-arm manipulation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict, Any, Optional

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
from robosuite_franka_manipulation.services import ServiceClient, ServiceManager, UnifiedPerceptionService
import cv2
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

def create_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Dict:
    """
    Creates a pose dict with position and quaternion.
    """
    quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
    return {"pos": np.array([x, y, z]), "quat": quat}

def perform_dual_arm_manipulation(env: Any, service_client: ServiceClient, task: str = "handoff", visualize: bool = False) -> Optional[bool]:
    """
    Perform dual-arm manipulation with perception.

    Args:
        env: Robosuite environment
        service_client: Client for communicating with services.
        task: Type of task ("handoff", "cooperative_lift", etc.)
        visualize: Whether to show visualizations.
    """
    print(f"Starting dual-arm manipulation task: {task}")
    print("Note: Manual teleoperation can be started separately using teleop_dual_arm.py")

    # Initialize controllers
    left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
    right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
    left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
    right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

    # Initialize visualization components if needed
    if visualize:
        object_detector_vis = ObjectDetector(env, visualize=True)
        grasp_predictor_vis = GraspPredictor(visualize=True)

    # Define ready poses
    PI = np.pi
    ready_pose_left = create_pose(-0.3, 0.3, 0.5, 0, PI/2, 0)
    ready_pose_right = create_pose(0.3, 0.3, 0.5, -PI/2, 0, PI/2)

    try:
        # Move to ready positions
        print("Moving to ready positions...")
        left_arm.move_to_pose(ready_pose_left["pos"], ready_pose_left["quat"])
        right_arm.move_to_pose(ready_pose_right["pos"], ready_pose_right["quat"])

        if task == "handoff":
            return perform_handoff_task(left_arm, right_arm, left_gripper, right_gripper,
                                      ready_pose_left, ready_pose_right, service_client, visualize,
                                      object_detector_vis if visualize else None, 
                                      grasp_predictor_vis if visualize else None, env)
        elif task == "cooperative_lift":
            return perform_cooperative_lift(left_arm, right_arm, left_gripper, right_gripper,
                                          ready_pose_left, ready_pose_right, service_client, visualize,
                                          object_detector_vis if visualize else None,
                                          grasp_predictor_vis if visualize else None, env)
        else:
            print(f"Unknown task: {task}")
            return False

    except Exception as e:
        print(f"Error during manipulation: {e}")
        return False

def perform_handoff_task(left_arm, right_arm, left_gripper, right_gripper,
                        ready_left, ready_right, service_client, visualize,
                        object_detector_vis, grasp_predictor_vis, env):
    """Perform handoff task: left arm picks, hands to right arm, right arm places."""
    print("Executing handoff task...")

    # Wait for objects to be detected
    objects = []
    for _ in range(10): # 10 retries
        response = service_client.call("perception_service", {"command": "GET_LATEST_OBJECTS"})
        if response and response.get("status") == "success" and response.get("objects"):
            objects = response["objects"]
            break
        time.sleep(0.5)

    if not objects:
        print("No objects detected!")
        return False

    obj = objects[0]  # Use first detected object
    object_pose = obj["pose"]
    print(f"Detected object at position: {object_pose['pos']}")

    # Get grasps for the object
    if not obj["grasps"]:
        print("No grasps predicted!")
        return False
    
    grasps = obj["grasps"]
    print(f"Predicted {len(grasps)} grasps. Using first grasp.")

    if visualize:
        obs = env._get_observations()
        rgb_image = obs.get(f'{object_detector_vis.camera_name}_image')
        object_detector_vis._visualize_fastsam_results(rgb_image, [np.array(o['mask']) for o in objects], "handoff")
        grasp_predictor_vis._visualize_grasps(np.array(obj['pcd']), grasps, object_pose)

    # Select best grasp (for now, first one)
    grasp_pose = grasps[0]
    grasp_pose['pos'] = np.array(grasp_pose['pos']) # ensure numpy array
    grasp_pose['quat'] = np.array(grasp_pose['quat']) # ensure numpy array

    # Define manipulation poses based on grasp
    pre_grasp = create_pose(grasp_pose["pos"][0] - 0.1, grasp_pose["pos"][1],
                           grasp_pose["pos"][2] + 0.1, 0, np.pi/2, 0)
    post_grasp = create_pose(grasp_pose["pos"][0], grasp_pose["pos"][1],
                            grasp_pose["pos"][2] + 0.2, 0, np.pi/2, 0)
    pre_handoff = create_pose(grasp_pose["pos"][0] + 0.25, grasp_pose["pos"][1],
                             grasp_pose["pos"][2] + 0.22, -np.pi/2, 0, np.pi/2)
    handoff = create_pose(grasp_pose["pos"][0] + 0.1, grasp_pose["pos"][1],
                         grasp_pose["pos"][2] + 0.22, -np.pi/2, 0, np.pi/2)
    place = create_pose(-grasp_pose["pos"][0], -grasp_pose["pos"][1], 0.3, 0, np.pi/2, np.pi)

    # 1. Left arm approach and grasp
    print("Left arm approaching object...")
    left_gripper.open()
    left_arm.move_to_pose(pre_grasp["pos"], pre_grasp["quat"])
    left_arm.move_to_pose(grasp_pose["pos"], grasp_pose["quat"])
    left_gripper.close()
    time.sleep(0.5)
    left_arm.move_to_pose(post_grasp["pos"], post_grasp["quat"])

    # 2. Handoff to right arm
    print("Handoff to right arm...")
    right_gripper.open()
    right_arm.move_to_pose(pre_handoff["pos"], pre_handoff["quat"])
    right_arm.move_to_pose(handoff["pos"], handoff["quat"])
    right_gripper.close()
    time.sleep(0.5)
    left_gripper.open()

    # 3. Left arm retract, right arm place
    print("Right arm placing object...")
    left_arm.move_to_pose(ready_left["pos"], ready_left["quat"])
    right_arm.move_to_pose(place["pos"], place["quat"])
    right_gripper.open()
    time.sleep(0.5)
    right_arm.move_to_pose(ready_right["pos"], ready_right["quat"])

    print("Handoff task completed successfully!")
    return True

def perform_cooperative_lift(left_arm, right_arm, left_gripper, right_gripper,
                           ready_left, ready_right, service_client, visualize,
                           object_detector_vis, grasp_predictor_vis, env):
    """Perform cooperative lift: both arms grasp and lift together."""
    print("Executing cooperative lift task...")

    # Wait for objects to be detected
    objects = []
    for _ in range(10): # 10 retries
        response = service_client.call("perception_service", {"command": "GET_LATEST_OBJECTS"})
        if response and response.get("status") == "success" and response.get("objects"):
            objects = response["objects"]
            break
        time.sleep(0.5)

    if not objects:
        print("No objects detected!")
        return False
        
    obj = objects[0]  # Use first detected object
    object_pose = obj["pose"]
    print(f"Detected object at position: {object_pose['pos']}")

    if not obj["grasps"]:
        print("No grasps predicted!")
        return False
    
    grasps = obj["grasps"]
    print(f"Predicted {len(grasps)} grasps. Using first grasp.")

    if visualize:
        obs = env._get_observations()
        rgb_image = obs.get(f'{object_detector_vis.camera_name}_image')
        object_detector_vis._visualize_fastsam_results(rgb_image, [np.array(o['mask']) for o in objects], "cooperative_lift")
        grasp_predictor_vis._visualize_grasps(np.array(obj['pcd']), grasps, object_pose)

    # Select best grasp (for now, first one)
    grasp_pose = grasps[0]
    grasp_pose['pos'] = np.array(grasp_pose['pos']) # ensure numpy array
    grasp_pose['quat'] = np.array(grasp_pose['quat']) # ensure numpy array

    # Define symmetric grasp poses
    left_grasp = create_pose(grasp_pose["pos"][0] - 0.05, grasp_pose["pos"][1],
                            grasp_pose["pos"][2], 0, np.pi/2, 0)
    right_grasp = create_pose(grasp_pose["pos"][0] + 0.05, grasp_pose["pos"][1],
                             grasp_pose["pos"][2], 0, np.pi/2, 0)
    lift_pose = create_pose(grasp_pose["pos"][0], grasp_pose["pos"][1],
                           grasp_pose["pos"][2] + 0.3, 0, np.pi/2, 0)

    # Both arms approach
    print("Both arms approaching object...")
    left_gripper.open()
    right_gripper.open()
    left_arm.move_to_pose(left_grasp["pos"], left_grasp["quat"])
    right_arm.move_to_pose(right_grasp["pos"], right_grasp["quat"])

    # Close grippers
    left_gripper.close()
    right_gripper.close()
    time.sleep(0.5)

    # Lift together
    print("Lifting object cooperatively...")
    left_arm.move_to_pose(lift_pose["pos"], lift_pose["quat"])
    right_arm.move_to_pose(lift_pose["pos"], lift_pose["quat"])

    time.sleep(1.0)

    # Return to ready
    print("Returning to ready positions...")
    left_arm.move_to_pose(ready_left["pos"], ready_left["quat"])
    right_arm.move_to_pose(ready_right["pos"], ready_right["quat"])

    # Open grippers
    left_gripper.open()
    right_gripper.open()

    print("Cooperative lift task completed successfully!")
    return True

def main() -> None:
    """Main function to run dual-arm manipulation."""
    import argparse

    parser = argparse.ArgumentParser(description="Dual-arm manipulation with perception")
    parser.add_argument("--task", type=str, default="handoff",
                       choices=["handoff", "cooperative_lift"],
                       help="Type of manipulation task")
    parser.add_argument("--visualize", action="store_true",
                       help="Enable visualization (if display available)")

    args = parser.parse_args()

    # Start service manager and services
    manager = ServiceManager()
    manager.start()

    env = make_dual_arm_env()
    env.reset()

    perception_service = UnifiedPerceptionService(env, visualize=False) # Visualization is handled in the script
    perception_service.start()
    
    # Wait for services to start
    time.sleep(2)

    try:
        service_client = ServiceClient()
        success = perform_dual_arm_manipulation(env, service_client, task=args.task, visualize=args.visualize)

        if success:
            print(f"\n {args.task.replace('_', ' ').title()} task completed successfully!")
        else:
            print(f"\n {args.task.replace('_', ' ').title()} task failed!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        perception_service.stop()
        manager.stop()
        env.close()

if __name__ == '__main__':
    main()