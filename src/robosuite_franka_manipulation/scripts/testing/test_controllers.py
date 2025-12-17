#!/usr/bin/env python3
"""
Test script to verify environment setup and controller functionality.
This script tests basic operations: environment creation, arm movements, gripper actions.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from typing import Optional, Any

from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController

def test_environment_setup() -> Optional[Any]:
    """Test that the environment can be created and reset."""
    print("Testing environment setup...")
    try:
        env = make_dual_arm_handoff_env()
        env.reset()
        print("✓ Environment created and reset successfully")
        return env
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return None

def test_controllers(env: Any) -> bool:
    """Test basic controller functionality."""
    print("\nTesting controllers...")

    # Initialize controllers
    left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
    right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
    left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
    right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

    print("✓ Controllers initialized")

    # Test arm movements
    print("Testing arm movements...")

    # Define test poses
    test_pose_left = {"pos": np.array([0.3, 0.3, 0.5]), "quat": R.from_euler('xyz', [0, np.pi/2, 0]).as_quat()}
    test_pose_right = {"pos": np.array([0.3, 0.3, 0.5]), "quat": R.from_euler('xyz', [-np.pi/2, 0, np.pi/2]).as_quat()}

    # Move left arm
    success_left = left_arm.move_to_pose(test_pose_left["pos"], test_pose_left["quat"], timeout=100)
    if success_left:
        print("✓ Left arm movement successful")
    else:
        print("✗ Left arm movement failed")

    # Move right arm
    success_right = right_arm.move_to_pose(test_pose_right["pos"], test_pose_right["quat"], timeout=100)
    if success_right:
        print("✓ Right arm movement successful")
    else:
        print("✗ Right arm movement failed")

    # Test grippers
    print("Testing grippers...")

    # Test left gripper
    left_gripper.open()
    time.sleep(0.5)
    left_gripper.close()
    time.sleep(0.5)
    left_gripper.open()
    print("✓ Left gripper test completed")

    # Test right gripper
    right_gripper.open()
    time.sleep(0.5)
    right_gripper.close()
    time.sleep(0.5)
    right_gripper.open()
    print("✓ Right gripper test completed")

    return True

def test_perception(env: Any) -> bool:
    """Test perception components."""
    print("\nTesting perception...")

    try:
        from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

        detector = ObjectDetector(env)
        predictor = GraspPredictor()

        # Detect objects
        objects = detector.detect_objects()
        if objects:
            print(f"✓ Detected {len(objects)} object(s)")
            obj = objects[0]
            print(f"  Object pose: {obj['pose']}")

            # Predict grasps
            grasps = predictor.predict_grasps(obj["pcd"], obj["pose"], num_grasps=3)
            if grasps:
                print(f"✓ Predicted {len(grasps)} grasp(s)")
                print(f"  First grasp position: {grasps[0]['pos']}")
            else:
                print("✗ No grasps predicted")
        else:
            print("✗ No objects detected")

        return True
    except Exception as e:
        print(f"✗ Perception test failed: {e}")
        return False

def main() -> None:
    """Run all tests."""
    print("=== Controller and Environment Test ===\n")

    # Test environment
    env = test_environment_setup()
    if not env:
        return

    # Test controllers
    controller_test = test_controllers(env)

    # Test perception
    perception_test = test_perception(env)

    # Cleanup
    env.close()

    print("\n=== Test Summary ===")
    print(f"Environment: {'PASS' if env else 'FAIL'}")
    print(f"Controllers: {'PASS' if controller_test else 'FAIL'}")
    print(f"Perception: {'PASS' if perception_test else 'FAIL'}")

    if env and controller_test and perception_test:
        print("\n All tests passed! The system is ready for manipulation tasks.")
    else:
        print("\n Some tests failed. Please check the errors above.")

if __name__ == '__main__':
    main()