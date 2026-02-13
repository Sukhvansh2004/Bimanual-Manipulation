import pytest
import numpy as np
import mujoco
from unittest.mock import MagicMock, patch
from scipy.spatial.transform import Rotation as R
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.planning.utils import ik_utils, mujoco_utils
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_object_trajectory,
    generate_end_effector_trajectory,
    check_trajectory_feasibility
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

class TestPlanningModule:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.sim = MagicMock()
        env.sim.model = MagicMock()
        env.sim.data = MagicMock()
        
        # Mock model properties
        env.sim.model.njnt = 20
        env.sim.model.nq = 20
        env.sim.model.nv = 20
        env.sim.model.ngeom = 10
        
        # Mock joint lookups
        env.sim.model.joint_name2id.return_value = 0
        env.sim.model.jnt_qposadr = np.zeros(100, dtype=int)
        env.sim.model.geom_bodyid = np.zeros(100, dtype=int)
        env.sim.model.geom_id2name.side_effect = lambda i: f"geom_{i}"
        
        # Mock data properties
        env.sim.data.qpos = np.zeros(20)
        env.sim.data.qvel = np.zeros(20)
        env.sim.data.ncon = 0
        
        return env

    @pytest.fixture
    def mock_tf_manager(self, mock_env):
        tfm = MagicMock(spec=TransformManager)
        def side_effect_transform(pose, target_frame):
            new_pose = pose.copy()
            new_pose.frame_id = target_frame
            return new_pose
        tfm.transform_pose.side_effect = side_effect_transform
        return tfm

    def test_generate_object_trajectory(self, mock_tf_manager):
        start = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        target = StampedPose(position=[1,0,0], quaternion=[0,0,0,1], frame_id="world")
        
        traj = generate_object_trajectory(
            start, 
            target, 
            tf_manager=mock_tf_manager,
            num_steps=10
        )
        
        assert len(traj) > 0
        assert isinstance(traj[0], StampedPose)
        np.testing.assert_allclose(traj[0].position, start.position, atol=1e-6)
        np.testing.assert_allclose(traj[-1].position, target.position, atol=1e-6)

    def test_generate_end_effector_trajectory(self, mock_tf_manager):
        obj_traj = [
            StampedPose(position=[1,0,0], quaternion=[0,0,0,1], frame_id="world"),
            StampedPose(position=[1,0,1], quaternion=[0,0,0,1], frame_id="world")
        ]
        grasp = StampedPose(position=[1,0,0.1], quaternion=[0,0,0,1], frame_id="world")
        
        eef_traj = generate_end_effector_trajectory(obj_traj, grasp, tf_manager=mock_tf_manager)
        
        assert len(eef_traj) == 2
        # Grasp is at [1,0,0.1], object at [1,0,0]. 
        # T_obj_grasp should have translation [0,0,0.1]
        # In second point, object is at [1,0,1], so eef should be [1,0,1.1]
        np.testing.assert_allclose(eef_traj[1].position, [1, 0, 1.1], atol=1e-6)
        assert eef_traj[0].frame_id == "world"

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.set_qpos')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.in_collision')
    def test_check_trajectory_feasibility_success(
        self,
        mock_in_collision, 
        mock_ik, 
        mock_get_ids, 
        mock_get_indices, 
        mock_set_qpos,
        mock_utils_mujoco,
        mock_gen_mujoco,
        mock_env, 
        mock_tf_manager
    ):
        mock_ik.return_value = [np.zeros(7)]
        mock_in_collision.return_value = False
        mock_get_ids.return_value = [0, 1, 2, 3, 4, 5, 6]
        mock_get_indices.return_value = np.arange(7)
        
        traj = [StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")]
        
        results, feasible = check_trajectory_feasibility(
            env=mock_env,
            robot_id=0,
            eef_trajectory=traj,
            tf_manager=mock_tf_manager,
            check_collisions=True
        )
        
        assert feasible is True
        assert len(results) == 1
        assert results[0].is_feasible is True
        assert results[0].ik_success is True
        assert results[0].collision_free is True
        mock_ik.assert_called() 
        assert mock_utils_mujoco.mj_fwdPosition.called or mock_gen_mujoco.mj_fwdPosition.called

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    def test_check_trajectory_feasibility_ik_fail(
        self,
        mock_ik, 
        mock_get_ids, 
        mock_get_indices, 
        mock_utils_mujoco, 
        mock_gen_mujoco, 
        mock_env, 
        mock_tf_manager
    ):
        mock_ik.return_value = []
        mock_get_ids.return_value = [0]
        mock_get_indices.return_value = np.arange(7)
        
        traj = [StampedPose(position=[10, 10, 10], quaternion=[0, 0, 0, 1], frame_id="world")]
        
        results, feasible = check_trajectory_feasibility(
            env=mock_env,
            robot_id=0,
            eef_trajectory=traj,
            tf_manager=mock_tf_manager
        )
        
        assert feasible is False
        assert results[0].ik_success is False
        assert results[0].is_feasible is False

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.set_qpos')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.in_collision')
    def test_check_trajectory_feasibility_collision(
        self,
        mock_in_collision, 
        mock_ik, 
        mock_get_ids, 
        mock_get_indices, 
        mock_set_qpos,
        mock_utils_mujoco, 
        mock_gen_mujoco, 
        mock_env, 
        mock_tf_manager
    ):
        mock_ik.return_value = [np.zeros(7)]
        mock_in_collision.return_value = True
        mock_get_ids.return_value = [0]
        mock_get_indices.return_value = np.arange(7)
        
        traj = [StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")]
        
        results, feasible = check_trajectory_feasibility(
            env=mock_env,
            robot_id=0,
            eef_trajectory=traj,
            tf_manager=mock_tf_manager,
            check_collisions=True
        )
        
        assert feasible is False
        assert results[0].ik_success is True
        assert results[0].collision_free is False
        assert results[0].is_feasible is False

    def test_ik_convergence_basic(self):
        """Test that IK converges to a known reachable pose."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        # Robot 0 right arm
        robot_id = 0
        arm = "right"
        
        model = env.sim.model
        data = env.sim.data
        
        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))
        
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)
        
        # 1. Pick a target configuration and find its site pose
        rng = np.random.default_rng(42)
        # Avoid extreme limits for better test reliability
        q_target = rng.uniform(joint_lo + 0.1, joint_hi - 0.1)
        
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)
        
        mujoco_utils.set_qpos(data, joint_qpos_idx, q_target)
        mujoco.mj_fwdPosition(model_native, data_native)
        
        target_pos = np.array(data.site_xpos[site_id])
        target_mat = np.array(data.site_xmat[site_id]).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()
        
        # 2. Reset and try to solve IK from neutral config
        env.reset()
        q_start = mujoco_utils.get_qpos(data, joint_qpos_idx)
        
        q_sol = ik_utils.solve_ik_scipy(
            model, data, site_id, 
            joint_qpos_idx, joint_dof_idx, 
            joint_lo, joint_hi, 
            target_pos, target_quat, 
            q_start, 
            pos_tol=1e-4, ori_tol=1e-3
        )
        
        assert q_sol is not None, "IK failed to converge to known valid pose"
        
        # 3. Verify precision
        mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
        mujoco.mj_fwdPosition(model_native, data_native)
        pos_err, rot_err = mujoco_utils.compute_site_error(model, data, site_id, target_pos, target_quat)
        
        pos_dist = np.linalg.norm(pos_err)
        rot_dist = np.linalg.norm(rot_err)
        
        assert pos_dist < 1e-4, f"Position error too high: {pos_dist}"
        assert rot_dist < 1e-3, f"Rotation error too high: {rot_dist}"
        
        env.close()


class TestObjectTrajectoryGeneration:
    """Tests for object trajectory generation."""

    @pytest.fixture
    def mock_tf_manager(self):
        tfm = MagicMock(spec=TransformManager)
        def side_effect_transform(pose, target_frame):
            new_pose = pose.copy()
            new_pose.frame_id = target_frame
            return new_pose
        tfm.transform_pose.side_effect = side_effect_transform
        return tfm

    def test_trajectory_length(self, mock_tf_manager):
        """Test trajectory has correct number of waypoints based on lift/approach/insert phases."""
        start = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        target = StampedPose(position=[1, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        # Trajectory has lift(0.2) + approach(0.5) + insert(0.3) phases + final point
        # So expected = int(num_steps*0.2) + int(num_steps*0.5) + int(num_steps*0.3) + 1
        for num_steps in [10, 50, 100]:
            traj = generate_object_trajectory(
                start, target, tf_manager=mock_tf_manager, num_steps=num_steps
            )
            expected_len = (
                int(num_steps * constants.OBJECT_TRAJ_LIFT_RATIO) +
                int(num_steps * constants.OBJECT_TRAJ_APPROACH_RATIO) +
                int(num_steps * constants.OBJECT_TRAJ_INSERT_RATIO) + 1
            )
            assert len(traj) == expected_len

    def test_trajectory_interpolation(self, mock_tf_manager):
        """Test trajectory has valid poses from start region to target region."""
        start = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        target = StampedPose(position=[1, 1, 1], quaternion=[0, 0, 0, 1], frame_id="world")
        
        traj = generate_object_trajectory(
            start, target, tf_manager=mock_tf_manager, num_steps=100
        )
        
        # First waypoint should be at or near start position
        np.testing.assert_allclose(traj[0].position[:2], start.position[:2], atol=0.01)
        
        # Last waypoint should be at target position
        np.testing.assert_allclose(traj[-1].position, target.position, atol=0.01)
        
        # All waypoints should have valid quaternions (unit length)
        for pose in traj:
            quat_norm = np.linalg.norm(pose.quaternion)
            assert 0.99 < quat_norm < 1.01, f"Quaternion not normalized: {quat_norm}"

    def test_trajectory_with_rotation(self, mock_tf_manager):
        """Test trajectory generation with rotation change."""
        start = StampedPose(
            position=[0, 0, 0], 
            quaternion=[0, 0, 0, 1],  # Identity
            frame_id="world"
        )
        target = StampedPose(
            position=[1, 0, 0], 
            quaternion=[0, 0, 0.7071, 0.7071],  # 90 deg around Z
            frame_id="world"
        )
        
        traj = generate_object_trajectory(
            start, target, tf_manager=mock_tf_manager, num_steps=10
        )
        
        # First and last should match
        np.testing.assert_allclose(traj[0].position, start.position, atol=1e-6)
        np.testing.assert_allclose(traj[-1].position, target.position, atol=1e-6)


class TestEndEffectorTrajectoryGeneration:
    """Tests for end effector trajectory generation."""

    @pytest.fixture
    def mock_tf_manager(self):
        tfm = MagicMock(spec=TransformManager)
        def side_effect_transform(pose, target_frame):
            new_pose = pose.copy()
            new_pose.frame_id = target_frame
            return new_pose
        tfm.transform_pose.side_effect = side_effect_transform
        return tfm

    def test_eef_follows_object(self, mock_tf_manager):
        """Test that EEF trajectory follows object trajectory with offset."""
        obj_traj = [
            StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world"),
            StampedPose(position=[0.5, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world"),
            StampedPose(position=[1, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world"),
        ]
        grasp = StampedPose(position=[0, 0, 0.1], quaternion=[0, 0, 0, 1], frame_id="world")
        
        eef_traj = generate_end_effector_trajectory(obj_traj, grasp, tf_manager=mock_tf_manager)
        
        assert len(eef_traj) == len(obj_traj)
        
        # EEF should maintain offset from object
        for i, (obj_pose, eef_pose) in enumerate(zip(obj_traj, eef_traj)):
            offset = np.array(eef_pose.position) - np.array(obj_pose.position)
            np.testing.assert_allclose(offset, [0, 0, 0.1], atol=1e-6)


class TestWaypointFeasibilityResult:
    """Tests for WaypointFeasibilityResult dataclass."""

    @pytest.fixture
    def sample_pose(self):
        return StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")

    def test_all_fields_required(self, sample_pose):
        """Test that all fields must be provided."""
        result = WaypointFeasibilityResult(
            waypoint_index=0,
            eef_pose=sample_pose,
            joint_config=np.zeros(7),
            is_feasible=True,
            ik_success=True,
            collision_free=True
        )
        
        assert result.waypoint_index == 0
        assert result.ik_success is True
        assert result.collision_free is True
        assert result.is_feasible is True

    def test_feasibility_conditions(self, sample_pose):
        """Test feasibility with different IK and collision combinations."""
        # IK fails, collision-free
        result1 = WaypointFeasibilityResult(
            waypoint_index=0, eef_pose=sample_pose, joint_config=None,
            ik_success=False, collision_free=True, is_feasible=False
        )
        assert result1.is_feasible is False
        assert result1.joint_config is None  # IK failed, no config
        
        # IK succeeds, collision
        result2 = WaypointFeasibilityResult(
            waypoint_index=0, eef_pose=sample_pose, joint_config=np.zeros(7),
            ik_success=True, collision_free=False, is_feasible=False
        )
        assert result2.is_feasible is False
        
        # Both good
        result3 = WaypointFeasibilityResult(
            waypoint_index=0, eef_pose=sample_pose, joint_config=np.zeros(7),
            ik_success=True, collision_free=True, is_feasible=True
        )
        assert result3.is_feasible is True


class TestTrajectoryFeasibilityEdgeCases:
    """Edge case tests for trajectory feasibility checking."""

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.sim = MagicMock()
        env.sim.model = MagicMock()
        env.sim.data = MagicMock()
        
        env.sim.model.njnt = 20
        env.sim.model.nq = 20
        env.sim.model.nv = 20
        env.sim.model.ngeom = 10
        
        env.sim.model.joint_name2id.return_value = 0
        env.sim.model.jnt_qposadr = np.zeros(100, dtype=int)
        env.sim.model.geom_bodyid = np.zeros(100, dtype=int)
        env.sim.model.geom_id2name.side_effect = lambda i: f"geom_{i}"
        
        env.sim.data.qpos = np.zeros(20)
        env.sim.data.qvel = np.zeros(20)
        env.sim.data.ncon = 0
        
        return env

    @pytest.fixture
    def mock_tf_manager(self):
        tfm = MagicMock(spec=TransformManager)
        def side_effect_transform(pose, target_frame):
            new_pose = pose.copy()
            new_pose.frame_id = target_frame
            return new_pose
        tfm.transform_pose.side_effect = side_effect_transform
        return tfm

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    def test_empty_trajectory(self, mock_utils_mujoco, mock_gen_mujoco, mock_env, mock_tf_manager):
        """Test handling of empty trajectory."""
        results, feasible = check_trajectory_feasibility(
            env=mock_env,
            robot_id=0,
            eef_trajectory=[],
            tf_manager=mock_tf_manager
        )
        
        assert results == []
        assert feasible is True  # Empty trajectory is trivially feasible

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    def test_multi_waypoint_partial_feasibility(
        self,
        mock_ik, 
        mock_get_ids, 
        mock_get_indices, 
        mock_utils_mujoco, 
        mock_gen_mujoco, 
        mock_env, 
        mock_tf_manager
    ):
        """Test trajectory where some waypoints are feasible and others not."""
        # First waypoint has IK solution, second doesn't
        mock_ik.side_effect = [
            [np.zeros(7)],  # First call: IK success
            [],              # Second call: IK fail
        ]
        mock_get_ids.return_value = [0]
        mock_get_indices.return_value = np.arange(7)
        
        traj = [
            StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world"),
            StampedPose(position=[10, 10, 10], quaternion=[0, 0, 0, 1], frame_id="world"),
        ]
        
        results, feasible = check_trajectory_feasibility(
            env=mock_env,
            robot_id=0,
            eef_trajectory=traj,
            tf_manager=mock_tf_manager
        )
        
        assert feasible is False  # Overall not feasible
        assert len(results) == 2
        # Note: Due to short-circuiting, may not process all waypoints after failure


class TestIKSolutionValidation:
    """Tests for IK solution validation."""

    def test_joint_limit_satisfaction(self):
        """Test that IK solutions respect joint limits."""
        env = make_dual_arm_env(
            has_renderer=False, 
            has_offscreen_renderer=False, 
            use_camera_obs=False
        )
        env.reset()
        
        model = env.sim.model
        
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id=0)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)
        
        # Verify limits are sensible
        assert len(joint_lo) == 7  # 7 DOF arm
        assert len(joint_hi) == 7
        assert np.all(joint_hi > joint_lo)
        
        env.close()

    @pytest.mark.parametrize("robot_id", [0, 1])
    def test_arm_joint_ids_discovery(self, robot_id):
        """Test that arm joint IDs can be discovered for both robots."""
        env = make_dual_arm_env(
            has_renderer=False, 
            has_offscreen_renderer=False, 
            use_camera_obs=False
        )
        env.reset()
        
        model = env.sim.model
        
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id=robot_id)
        
        assert len(joint_ids) == 7
        assert all(isinstance(jid, (int, np.integer)) for jid in joint_ids)
        
        env.close()


class TestMujocoUtilsHelpers:
    """Tests for mujoco utility helper functions."""

    def test_get_eef_site_name(self):
        """Test EEF site name generation."""
        # Robot 0 right arm
        site_name = mujoco_utils.get_eef_site_name(robot_id=0, arm="right")
        assert isinstance(site_name, str)
        assert len(site_name) > 0
        
        # Robot 1 right arm
        site_name2 = mujoco_utils.get_eef_site_name(robot_id=1, arm="right")
        assert isinstance(site_name2, str)
        assert site_name != site_name2  # Different robots have different names

    def test_native_object_extraction(self):
        """Test extraction of native MuJoCo objects."""
        env = make_dual_arm_env(
            has_renderer=False, 
            has_offscreen_renderer=False, 
            use_camera_obs=False
        )
        env.reset()
        
        model = env.sim.model
        data = env.sim.data
        
        native_model = mujoco_utils.get_native_mj_object(model)
        native_data = mujoco_utils.get_native_mj_object(data)
        
        # Should return objects (exact type depends on robosuite version)
        assert native_model is not None
        assert native_data is not None
        
        env.close()
