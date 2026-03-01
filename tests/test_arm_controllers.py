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
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
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
        assert robot0_arm.tf_manager is not None
        assert robot1_arm.tf_manager is not None
        
        # Check base frame is set correctly
        assert "base" in robot0_arm.base_frame
        assert "base" in robot1_arm.base_frame

    def test_arm_current_pose_access(self, controllers):
        """Test that current arm poses can be accessed."""
        robot0_arm, robot1_arm = controllers

        # Test robot 0 arm pose access
        pos0, quat0 = robot0_arm.get_eef_pose_in_base()

        assert isinstance(pos0, np.ndarray)
        assert pos0.shape == (3,)
        assert isinstance(quat0, np.ndarray)
        assert quat0.shape == (4,)

        # Test robot 1 arm pose access
        pos1, quat1 = robot1_arm.get_eef_pose_in_base()

        assert isinstance(pos1, np.ndarray)
        assert pos1.shape == (3,)
        assert isinstance(quat1, np.ndarray)
        assert quat1.shape == (4,)

    @pytest.mark.parametrize("robot_id", [0, 1])
    def test_arm_movement(self, env, robot_id):
        """Test arm movement to a small offset target pose within workspace."""
        arm = RobosuiteArmController(env, arm_id="right", robot_id=robot_id)

        # Get current pose
        current_pos, current_quat = arm.get_eef_pose_in_base()

        # Use a small offset that stays well within reachable workspace
        target_pos = current_pos + np.array([0.02, 0.02, -0.02])
        # Keep current orientation to isolate position tracking
        target_quat = current_quat.copy()

        target_pose = StampedPose(
            position=target_pos,
            quaternion=target_quat,
            frame_id=arm.base_frame
        )

        # Attempt movement
        success = arm.move_to_pose(target_pose)

        # Should converge for a small in-workspace offset
        assert success is True


class TestArmControllerFrameHandling:
    """Test arm controller frame transformation functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        yield env
        env.close()

    @pytest.fixture
    def controller(self, env):
        """Fixture to provide arm controller."""
        return RobosuiteArmController(env, arm_id="right", robot_id=0)

    def test_move_to_pose_accepts_dict(self, env):
        """Test that move_to_pose accepts grasp dictionary converted to StampedPose."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        
        # Get current pose and create a small offset
        current_pos, current_quat = controller.get_eef_pose_in_base()
        current_pos = current_pos.copy()
        
        # Create grasp dict (simulating GraspPredictor output)
        grasp_dict = {
            'pos': current_pos + np.array([0.01, 0, 0]),  # Tiny movement
            'quat': current_quat,
            'frame_id': controller.base_frame,  # In base frame, no transform needed
            'score': 0.95
        }
        
        # Convert to StampedPose
        grasp_pose = StampedPose(**grasp_dict)
        
        # Should not raise an error
        success = controller.move_to_pose(grasp_pose, num_steps=50)
        assert isinstance(success, bool)

    def test_move_to_pose_accepts_stamped_pose(self, env):
        """Test that move_to_pose accepts StampedPose."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        
        current_pos, current_quat = controller.get_eef_pose_in_base()
        current_pos = current_pos.copy()
        
        pose = StampedPose(
            position=current_pos + np.array([0.01, 0, 0]),
            quaternion=current_quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(pose, num_steps=50)
        assert isinstance(success, bool)


class TestGripperControllers:
    """Test gripper controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
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
        left_gripper.release()
        time.sleep(0.2)
        left_gripper.grasp()
        time.sleep(0.2)
        left_gripper.release()

        # Test right gripper
        right_gripper.release()
        time.sleep(0.2)
        right_gripper.grasp()
        time.sleep(0.2)
        right_gripper.release()