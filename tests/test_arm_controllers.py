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


from robosuite_franka_manipulation.utils.suction_weld import (
    apply_suction_weld, clear_suction_cache, SuctionWeldHandles
)

from robosuite_franka_manipulation.teleop.teleop import KeyboardTeleop, KeyboardOverride

class TestTeleop:
    """Test keyboard teleoperation logic."""

    @pytest.fixture
    def mock_teleop_env(self):
        env = MagicMock()
        env.action_dim = 14
        robot = MagicMock()
        robot.arms = ["right"]
        robot.action_dim = 7
        env.robots = [robot, robot]
        return env

    @patch('robosuite_franka_manipulation.teleop.teleop.pynput_keyboard.Listener')
    def test_keyboard_teleop_init(self, mock_listener, mock_teleop_env):
        """Test KeyboardTeleop initialization."""
        teleop = KeyboardTeleop(mock_teleop_env)
        assert teleop.device is not None
        assert mock_listener.called

    def test_keyboard_override(self, mock_teleop_env):
        """Test KeyboardOverride special handling."""
        device = KeyboardOverride(mock_teleop_env)
        from pynput.keyboard import Key
        # Space should be ignored (method returns None)
        assert device.on_press(Key.space) is None

    @patch('robosuite_franka_manipulation.teleop.teleop.pynput_keyboard.Listener')
    def test_teleop_key_press(self, mock_listener, mock_teleop_env):
        """Test _on_key_press callback."""
        teleop = KeyboardTeleop(mock_teleop_env)
        
        # Test 'q'
        key_q = MagicMock()
        key_q.char = 'q'
        teleop._on_key_press(key_q)
        assert teleop.exit_pressed is True
        
        # Test Space
        from pynput.keyboard import Key
        teleop._on_key_press(Key.space)
        assert teleop.gripper_toggle.is_set()

    @patch('robosuite_franka_manipulation.teleop.teleop.pynput_keyboard.Listener')
    @patch('time.sleep', side_effect=[None, Exception("Stop loop")]) # To break the loop
    def test_teleop_run_minimal(self, mock_sleep, mock_listener, mock_teleop_env):
        """Test a single iteration of the teleop run loop."""
        teleop = KeyboardTeleop(mock_teleop_env)
        teleop.device = MagicMock()
        teleop.device.active_robot = 0
        teleop.device.input2action.return_value = {"right_delta": np.zeros(6)}
        
        # We need to mock robot.create_action_vector
        mock_teleop_env.robots[0].create_action_vector.return_value = np.zeros(7)
        
        try:
            teleop.run()
        except Exception as e:
            if str(e) != "Stop loop":
                raise e
        
        assert mock_teleop_env.reset.called

class TestSuctionWeld:
    """Test suction weld utilities."""

    @pytest.fixture
    def mock_suction_env(self):
        env = MagicMock()
        env.action_dim = 14
        
        # Mock robots with suction grippers
        robot0 = MagicMock()
        robot0.arms = ["right"]
        robot0.action_dim = 7
        gripper0 = MagicMock()
        type(gripper0).__name__ = "PandaSuctionGripper"
        gripper0.correct_naming.side_effect = lambda x: f"robot0_right_{x}"
        robot0.gripper = {"right": gripper0}
        
        robot1 = MagicMock()
        robot1.arms = ["right"]
        robot1.action_dim = 7
        gripper1 = MagicMock()
        type(gripper1).__name__ = "PandaGripper"
        robot1.gripper = {"right": gripper1}
        
        env.robots = [robot0, robot1]
        
        # Mock sim/model
        env.sim = MagicMock()
        env.sim.model = MagicMock()
        env.sim.data = MagicMock()
        env.sim.model.ngeom = 10
        env.sim.model.geom_bodyid = np.zeros(10, dtype=int)
        env.sim.model.eq_type = np.ones(10) # 1 = WELD
        env.sim.model.eq_obj1id = np.zeros(10)
        env.sim.model.eq_obj2id = np.zeros(10)
        env.sim.model.eq_data = np.zeros((10, 11))
        
        env.sim.data.ncon = 1
        contact = MagicMock()
        contact.geom1 = [1] # snout
        contact.geom2 = [2] # object
        contact.dist = [0.01]
        env.sim.data.contact = contact
        # Use a real array so it can be updated
        env.sim.data.eq_active = np.zeros(10)
        
        env.object_body_ids = {"obj": 5}
        
        return env

    @patch('robosuite_franka_manipulation.utils.suction_weld._name2id')
    @patch('robosuite_franka_manipulation.utils.suction_weld._get_body_pose_wxyz')
    @patch('robosuite_franka_manipulation.utils.suction_weld._resolve_handles')
    @patch('robosuite_franka_manipulation.utils.suction_weld._is_valid_suction_target')
    @patch('robosuite_franka_manipulation.utils.suction_weld._get_contact_arrays')
    def test_apply_suction_weld_on(self, mock_contacts, mock_valid, mock_resolve, mock_pose, mock_name2id, mock_suction_env):
        """Test activating suction weld."""
        mock_name2id.side_effect = lambda sim, type, name: 1
        mock_pose.return_value = (np.zeros(3), np.array([1,0,0,0]))
        mock_valid.return_value = True
        
        # mock_contacts returns (g1, g2, dist)
        mock_contacts.return_value = (np.array([1]), np.array([2]), np.array([0.01]))
        
        handles = SuctionWeldHandles(valid=True, eq_id=1, snout_geom_id=1, eef_body_id=0)
        mock_resolve.return_value = handles
        
        mock_suction_env.sim.model.geom_bodyid[2] = 5 # Object body
        
        action = np.zeros(14)
        action[6] = 1.0 # Suction ON for robot 0
        
        clear_suction_cache()
        apply_suction_weld(mock_suction_env, action)
        
        assert mock_suction_env.sim.data.eq_active[1] == 1

    def test_apply_suction_weld_off(self, mock_suction_env):
        """Test deactivating suction weld."""
        action = np.zeros(14)
        action[6] = -1.0 # Suction OFF
        
        # Manually inject handle into cache to simulate active weld
        from robosuite_franka_manipulation.utils.suction_weld import _SUCTION_CACHE
        model_key = id(mock_suction_env.sim.model)
        _SUCTION_CACHE[model_key] = {(0, "right"): SuctionWeldHandles(valid=True, eq_id=1, welded_body_id=5)}
        
        apply_suction_weld(mock_suction_env, action)
        assert mock_suction_env.sim.data.eq_active[1] == 0

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
        target_pos = current_pos + np.array([0.05, 0.05, 0.05])
        target_quat = current_quat.copy()

        target_pose = StampedPose(
            position=target_pos,
            quaternion=target_quat,
            frame_id=arm.base_frame
        )

        # Attempt movement
        success = arm.move_to_pose(target_pose, num_steps=10)

        # Movement should complete without crashing
        assert isinstance(success, bool)

    def test_move_to_pose_with_approach(self, env):
        """Test move_to_pose with approach=True."""
        arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
        current_pos, current_quat = arm.get_eef_pose_in_base()
        
        target_pose = StampedPose(
            position=current_pos + np.array([0.05, 0.05, 0.05]),
            quaternion=current_quat,
            frame_id=arm.base_frame
        )
        
        success = arm.move_to_pose(target_pose, approach=True, approach_distance=0.02, num_steps=10)
        assert isinstance(success, bool)

    def test_follow_trajectory(self, env):
        """Test following a pre-defined trajectory."""
        arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
        current_pos, current_quat = arm.get_eef_pose_in_base()
        
        trajectory = []
        for i in range(5):
            pose = StampedPose(
                position=current_pos + np.array([0.01 * i, 0, 0]),
                quaternion=current_quat,
                frame_id=arm.base_frame
            )
            trajectory.append(pose)
            
        success = arm.follow_trajectory(trajectory, threshold=0.1)
        assert isinstance(success, bool)

    @patch("meshcat.Visualizer")
    def test_visualize_trajectory(self, mock_visualizer, env):
        """Test trajectory visualization (mocked)."""
        mock_vis_instance = MagicMock()
        mock_visualizer.return_value = mock_vis_instance
        
        arm = RobosuiteArmController(env, arm_id="right", robot_id=0, visualize=True)
        current_pos, current_quat = arm.get_eef_pose_in_base()
        
        target_pose = StampedPose(
            position=current_pos + np.array([0.02, 0.02, 0.02]),
            quaternion=current_quat,
            frame_id=arm.base_frame
        )
        
        # This will trigger visualization
        arm.move_to_pose(target_pose, num_steps=5)
        
        assert arm._meshcat_vis is not None
        assert mock_vis_instance.__getitem__.called


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

    def test_cancel_during_movement(self, env):
        """Test that movement can be cancelled during execution."""
        controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
        current_pos, current_quat = controller.get_eef_pose_in_base()
        
        target_pose = StampedPose(
            position=current_pos + np.array([0.1, 0.1, 0.1]),
            quaternion=current_quat,
            frame_id=controller.base_frame
        )
        
        # Run movement in a thread
        success_container = []
        def move():
            success = controller.move_to_pose(target_pose, num_steps=100)
            success_container.append(success)
            
        move_thread = threading.Thread(target=move)
        move_thread.start()
        
        # Wait a bit then cancel
        time.sleep(0.5)
        controller.cancel_movement()
        move_thread.join()
        
        assert len(success_container) > 0
        assert success_container[0] is False
        assert controller.last_status == 0


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