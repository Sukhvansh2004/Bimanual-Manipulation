import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from robosuite_franka_manipulation import constants
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
