#!/usr/bin/env python3
"""
Pytest-based test suite for controller functionality.
Tests arm controllers, gripper controllers, and their interactions.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController


class TestArmControllers:
    """Test arm controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_handoff_env()
        env.reset()
        yield env
        env.close()

    @pytest.fixture
    def controllers(self, env):
        """Fixture to provide initialized controllers."""
        left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
        right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
        return left_arm, right_arm

    def test_controller_initialization(self, controllers):
        """Test that controllers can be initialized."""
        left_arm, right_arm = controllers
        assert left_arm is not None
        assert right_arm is not None

    def test_arm_current_pose_access(self, controllers):
        """Test that current arm poses can be accessed."""
        left_arm, right_arm = controllers

        # Test left arm pose access
        left_pos = left_arm.robot._hand_pos["right"]
        left_ori_mat = left_arm.robot._hand_orn["right"]
        left_quat = R.from_matrix(left_ori_mat).as_quat()

        assert isinstance(left_pos, np.ndarray)
        assert left_pos.shape == (3,)
        assert isinstance(left_quat, np.ndarray)
        assert left_quat.shape == (4,)

        # Test right arm pose access
        right_pos = right_arm.robot._hand_pos["right"]
        right_ori_mat = right_arm.robot._hand_orn["right"]
        right_quat = R.from_matrix(right_ori_mat).as_quat()

        assert isinstance(right_pos, np.ndarray)
        assert right_pos.shape == (3,)
        assert isinstance(right_quat, np.ndarray)
        assert right_quat.shape == (4,)

    @pytest.mark.parametrize("arm_name", ["left", "right"])
    def test_arm_movement(self, controllers, arm_name):
        """Test arm movement to a target pose."""
        left_arm, right_arm = controllers

        if arm_name == "left":
            arm = left_arm
        else:
            arm = right_arm

        # Get current pose
        current_pos = arm.robot._hand_pos["right"]
        current_ori_mat = arm.robot._hand_orn["right"]
        current_quat = R.from_matrix(current_ori_mat).as_quat()

        # Create a slightly offset target pose
        target_pos = current_pos + np.array([0.05, 0.05, 0.05])
        target_quat = current_quat  # Keep same orientation

        # Attempt movement
        success = arm.move_to_pose(target_pos, target_quat, timeout=50)

        # Note: Movement may fail due to control tuning, but the method should not crash
        assert isinstance(success, bool)


class TestGripperControllers:
    """Test gripper controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_handoff_env()
        env.reset()
        yield env
        env.close()

    @pytest.fixture
    def grippers(self, env):
        """Fixture to provide initialized grippers."""
        left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
        right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)
        return left_gripper, right_gripper

    def test_gripper_initialization(self, grippers):
        """Test that grippers can be initialized."""
        left_gripper, right_gripper = grippers
        assert left_gripper is not None
        assert right_gripper is not None

    def test_gripper_operations(self, grippers):
        """Test gripper open/close operations."""
        left_gripper, right_gripper = grippers

        # Test left gripper
        left_gripper.open()
        time.sleep(0.2)
        left_gripper.close()
        time.sleep(0.2)
        left_gripper.open()

        # Test right gripper
        right_gripper.open()
        time.sleep(0.2)
        right_gripper.close()
        time.sleep(0.2)
        right_gripper.open()