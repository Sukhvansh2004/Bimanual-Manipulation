#!/usr/bin/env python3
"""
Example script demonstrating the perception pipeline:
1. Teleop arm to desired position
2. Detect objects using FastSAM
3. Get grasp predictions

This is a helper script for testing and debugging the perception pipeline.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController
from robosuite_franka_manipulation.perception import ObjectDetector


def teleop_arm_to_viewing_position(env, arm_controller):
    """
    Manually teleop the arm to a good viewing position for object detection.
    This is a simplified version - in practice you'd use the full teleop interface.
    """
    print("Moving arm to viewing position...")

    # Example viewing pose - adjust as needed
    viewing_pos = np.array([0.3, 0.2, 0.3])  # Position in front of robot
    viewing_quat = R.from_euler('xyz', [0, np.pi/4, 0]).as_quat()  # Looking down slightly

    success = arm_controller.move_to_pose(viewing_pos, viewing_quat, timeout=100)
    if success:
        print("✓ Arm moved to viewing position")
    else:
        print("✗ Failed to move arm to viewing position")

    return success


def demonstrate_perception_pipeline():
    """
    Complete demonstration of the perception pipeline.
    """
    print("=== Perception Pipeline Demonstration ===\n")

    # 1. Setup environment
    print("1. Setting up environment...")
    env = make_dual_arm_handoff_env()
    env.reset()

    # 2. Setup controllers
    print("2. Setting up controllers...")
    arm_controller = RobosuiteArmController(env, arm_id="right", robot_id=0)

    # 3. Teleop arm to viewing position
    print("3. Moving arm to viewing position...")
    success = teleop_arm_to_viewing_position(env, arm_controller)
    if not success:
        print("Warning: Arm movement failed, continuing with current position")

    # 4. Setup perception
    print("4. Setting up perception...")
    detector = ObjectDetector(env)

    # 5. Detect objects
    print("5. Detecting objects...")
    objects = detector.detect_objects()

    print(f"Detected {len(objects)} object(s)")

    # 6. Process detections
    for i, obj in enumerate(objects):
        print(f"\nObject {i}:")
        print(f"  Name: {obj['name']}")
        print(f"  Pose: position={obj['pose']['pos']}, orientation={obj['pose']['quat']}")
        print(f"  Point cloud: {len(obj['pcd'])} points")
        print(f"  Grasps: {len(obj['grasps'])} predicted")

        if obj['grasps']:
            print("  Sample grasp poses:")
            for j, grasp in enumerate(obj['grasps'][:3]):  # Show first 3 grasps
                print(f"    Grasp {j}: pos={grasp.get('pos', 'N/A')}, quat={grasp.get('quat', 'N/A')}")

    # 7. Cleanup
    print("\n6. Cleaning up...")
    env.close()

    print("\n=== Demonstration Complete ===")


def test_individual_components():
    """
    Test individual components of the perception pipeline.
    """
    print("=== Component Testing ===\n")

    # Test environment
    print("Testing environment...")
    env = make_dual_arm_handoff_env()
    env.reset()
    print("✓ Environment OK")

    # Test controller
    print("Testing controller...")
    controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
    print("✓ Controller OK")

    # Test detector initialization
    print("Testing detector...")
    detector = ObjectDetector(env)
    print("✓ Detector OK")

    # Test trajectory planning
    print("Testing trajectory planning...")
    start_pos = np.array([0.3, 0.2, 0.1])
    start_quat = np.array([0, 0, 0, 1])
    target_pos = np.array([0.4, 0.3, 0.2])
    target_quat = np.array([0, 0, 0, 1])

    positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
        start_pos, start_quat, target_pos, target_quat
    )
    print(f"✓ Trajectory planned: {len(positions)} waypoints, {total_time:.2f}s duration")

    env.close()
    print("\n=== Component Testing Complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perception Pipeline Demonstration")
    parser.add_argument("--test-components", action="store_true",
                       help="Test individual components instead of full pipeline")
    parser.add_argument("--teleop-only", action="store_true",
                       help="Only demonstrate teleoperation")

    args = parser.parse_args()

    if args.test_components:
        test_individual_components()
    elif args.teleop_only:
        # Just setup and show teleop (would need full teleop interface)
        print("Teleop demonstration - use the main teleop script for full functionality")
    else:
        demonstrate_perception_pipeline()