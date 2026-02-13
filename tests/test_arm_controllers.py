#!/usr/bin/env python3
"""
Pytest-based test suite for controller functionality.
Tests arm controllers, gripper controllers, and their interactions.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading
from unittest.mock import MagicMock, patch

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation import constants


class TestArmControllers:
    """Test arm controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
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
        """Test arm movement to a target pose."""
        arm = RobosuiteArmController(env, arm_id="right", robot_id=robot_id)

        # Get current pose
        current_pos, current_quat = arm.get_eef_pose_in_base()

        # Create a slightly offset target pose
        target_pos = current_pos + np.array([0.2, 0.3, 0.4])
        target_quat = np.array([0.86603, 0.28868, 0.28868, 0.28868])

        target_pose = StampedPose(
            position=target_pos,
            quaternion=target_quat,
            frame_id=arm.base_frame
        )

        # Attempt movement
        success = arm.move_to_pose(target_pose)

        # Movement should complete without crashing
        # Success depends on control tuning but method should work
        assert success==True


class TestArmControllerFrameHandling:
    """Test arm controller frame transformation functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
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


class TestArmControllerQuinticTrajectory:
    """Test quintic trajectory planning in arm controller."""

    @pytest.fixture
    def controller(self):
        """Create a minimal controller for testing trajectory planning."""
        controller = RobosuiteArmController.__new__(RobosuiteArmController)
        return controller

    def test_quintic_polynomial_basic(self, controller):
        """Test basic quintic polynomial generation."""
        s_traj = controller._quintic_polynomial(
            s0=0.0, sf=1.0, v0=0.0, vf=0.0, a0=0.0, af=0.0, num_points=50
        )
        
        assert len(s_traj) == 50
        assert abs(s_traj[0] - 0.0) < 1e-6
        assert abs(s_traj[-1] - 1.0) < 1e-6

    def test_quintic_polynomial_boundary_conditions(self, controller):
        """Test that quintic polynomial satisfies boundary conditions."""
        s_traj = controller._quintic_polynomial(
            s0=0.0, sf=1.0, v0=0.0, vf=0.0, a0=0.0, af=0.0, num_points=100
        )
        
        # Approximate velocity at endpoints (should be ~0)
        ds_start = (s_traj[1] - s_traj[0]) * 99  # Multiply by n-1 to approximate derivative
        ds_end = (s_traj[-1] - s_traj[-2]) * 99
        
        # Velocities should be small at endpoints
        assert abs(ds_start) < 0.1
        assert abs(ds_end) < 0.1


class TestArmControllerCancellation:
    """Test arm controller movement cancellation."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        yield env
        env.close()

    def test_cancel_movement(self, env):
        """Test that movement can be cancelled."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        
        # Cancel should work even if not moving
        controller.cancel_movement()
        
        assert controller._stop_event.is_set()


class TestArmControllerWithLock:
    """Test arm controller with environment lock."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        yield env
        env.close()

    def test_controller_with_lock(self, env):
        """Test controller creation with environment lock."""
        env_lock = threading.RLock()
        controller = RobosuiteArmController(
            env, arm_id="right", robot_id=0, env_lock=env_lock
        )
        
        assert controller.env_lock is env_lock
        
        # Test that pose access works with lock
        pos, quat = controller.get_eef_pose_in_base()
        assert pos is not None
        assert quat is not None


class TestGripperControllers:
    """Test gripper controller functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
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

    def test_gripper_width_parameter(self, env):
        """Test gripper operations with width parameter."""
        gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
        
        # Test different widths
        gripper.release(width=1.0)  # Fully open
        time.sleep(0.1)
        gripper.grasp(width=0.0)  # Fully closed
        time.sleep(0.1)
        gripper.release(width=0.5)  # Half open


class TestGripperControllerEdgeCases:
    """Test edge cases for gripper controller."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        yield env
        env.close()

    def test_gripper_type_detection(self, env):
        """Test that gripper type is correctly detected."""
        gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
        
        # Should have detected gripper type
        assert hasattr(gripper, 'is_suction')
        assert isinstance(gripper.is_suction, bool)

    def test_gripper_with_lock(self, env):
        """Test gripper controller with environment lock."""
        env_lock = threading.RLock()
        gripper = RobosuiteGripperController(
            env, arm_id="right", robot_id=0, env_lock=env_lock
        )
        
        assert gripper.env_lock is env_lock
        
        # Operations should work with lock
        gripper.release()
        time.sleep(0.1)
        gripper.grasp()