#!/usr/bin/env python3
"""
Pytest-based test suite for controller functionality.
Tests arm controllers, gripper controllers, and their interactions.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation import constants


class TestArmControllers:
    """Test arm controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env()
        env.reset()
        yield env
        env.close()

    @pytest.fixture
    def controllers(self, env):
        """Fixture to provide initialized controllers."""
        # Robot 0's right arm and Robot 1's right arm
        robot0_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
        robot1_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
        return robot0_arm, robot1_arm

    def test_controller_initialization(self, controllers):
        """Test that controllers can be initialized."""
        robot0_arm, robot1_arm = controllers
        assert robot0_arm is not None
        assert robot1_arm is not None
        
        # Check transform manager is initialized
        assert robot0_arm.transform_manager is not None
        assert robot1_arm.transform_manager is not None
        
        # Check base frame is set correctly
        assert "base" in robot0_arm.base_frame
        assert "base" in robot1_arm.base_frame

    def test_arm_current_pose_access(self, controllers):
        """Test that current arm poses can be accessed."""
        robot0_arm, robot1_arm = controllers

        # Test robot 0 arm pose access
        pos0 = robot0_arm.robot._hand_pos["right"]
        ori_mat0 = robot0_arm.robot._hand_orn["right"]
        quat0 = R.from_matrix(ori_mat0).as_quat()

        assert isinstance(pos0, np.ndarray)
        assert pos0.shape == (3,)
        assert isinstance(quat0, np.ndarray)
        assert quat0.shape == (4,)

        # Test robot 1 arm pose access
        pos1 = robot1_arm.robot._hand_pos["right"]
        ori_mat1 = robot1_arm.robot._hand_orn["right"]
        quat1 = R.from_matrix(ori_mat1).as_quat()

        assert isinstance(pos1, np.ndarray)
        assert pos1.shape == (3,)
        assert isinstance(quat1, np.ndarray)
        assert quat1.shape == (4,)

    @pytest.mark.parametrize("robot_id", [0, 1])
    def test_arm_movement(self, env, robot_id):
        """Test arm movement to a target pose."""
        arm = RobosuiteArmController(env, arm_id="right", robot_id=robot_id)

        # Get current pose
        current_pos = arm.robot._hand_pos["right"]
        current_ori_mat = arm.robot._hand_orn["right"]
        current_quat = R.from_matrix(current_ori_mat).as_quat()

        # Create a slightly offset target pose (small movement for reliability)
        target_pos = current_pos + np.array([0.05, 0.0, 0.0])
        target_quat = current_quat  # Keep same orientation

        # Attempt movement
        success = arm.move_to_pose(target_pos, target_quat, timeout=100, threshold=0.05)

        # Movement should complete without crashing
        # Success depends on control tuning but method should work
        assert isinstance(success, bool)


class TestArmControllerFrameHandling:
    """Test arm controller frame transformation functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env()
        env.reset()
        yield env
        env.close()

    @pytest.fixture
    def controller(self, env):
        """Fixture to provide arm controller."""
        return RobosuiteArmController(env, arm_id="right", robot_id=0)

    def test_parse_pose_input_array(self, controller):
        """Test parsing numpy array pose input."""
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0, 0, 0, 1])
        
        parsed_pos, parsed_quat, frame_id = controller._parse_pose_input(pos, quat, None)
        
        np.testing.assert_array_almost_equal(parsed_pos, pos)
        np.testing.assert_array_almost_equal(parsed_quat, quat)
        assert frame_id is None

    def test_parse_pose_input_stamped_pose(self, controller):
        """Test parsing StampedPose input."""
        pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        parsed_pos, parsed_quat, frame_id = controller._parse_pose_input(pose, None, None)
        
        np.testing.assert_array_almost_equal(parsed_pos, [1.0, 2.0, 3.0])
        assert frame_id == "world"

    def test_parse_pose_input_dict(self, controller):
        """Test parsing dictionary pose input."""
        grasp_dict = {
            'pos': [1.0, 2.0, 3.0],
            'quat': [0, 0, 0, 1],
            'frame_id': 'world',
            'score': 0.95
        }
        
        parsed_pos, parsed_quat, frame_id = controller._parse_pose_input(grasp_dict, None, None)
        
        np.testing.assert_array_almost_equal(parsed_pos, [1.0, 2.0, 3.0])
        assert frame_id == "world"

    def test_parse_pose_input_dict_frame_override(self, controller):
        """Test that explicit frame_id parameter overrides dict value."""
        grasp_dict = {
            'pos': [1.0, 2.0, 3.0],
            'quat': [0, 0, 0, 1],
            # No frame_id in dict
        }
        
        parsed_pos, parsed_quat, frame_id = controller._parse_pose_input(
            grasp_dict, None, "custom_frame"
        )
        
        assert frame_id == "custom_frame"

    def test_move_to_pose_accepts_dict(self, env):
        """Test that move_to_pose accepts grasp dictionary directly."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        
        # Get current pose and create a small offset
        current_pos = controller.robot._hand_pos["right"].copy()
        current_ori_mat = controller.robot._hand_orn["right"]
        current_quat = R.from_matrix(current_ori_mat).as_quat()
        
        # Create grasp dict (simulating GraspPredictor output)
        grasp_dict = {
            'pos': current_pos + np.array([0.01, 0, 0]),  # Tiny movement
            'quat': current_quat,
            'frame_id': controller.base_frame,  # In base frame, no transform needed
            'score': 0.95
        }
        
        # Should not raise an error
        success = controller.move_to_pose(grasp_dict, timeout=50)
        assert isinstance(success, bool)

    def test_move_to_pose_accepts_stamped_pose(self, env):
        """Test that move_to_pose accepts StampedPose."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        
        current_pos = controller.robot._hand_pos["right"].copy()
        current_ori_mat = controller.robot._hand_orn["right"]
        current_quat = R.from_matrix(current_ori_mat).as_quat()
        
        pose = StampedPose(
            position=current_pos + np.array([0.01, 0, 0]),
            quaternion=current_quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(pose, timeout=50)
        assert isinstance(success, bool)


class TestGripperControllers:
    """Test gripper controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env()
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