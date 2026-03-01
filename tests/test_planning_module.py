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
        """Test that mink-based IK converges to a known reachable pose."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        try:
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
            
            # 2. Solve IK from the target config as warm-start
            env.reset()
            
            full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()
            
            q_sol = ik_utils.solve_ik_mink(
                model_native, full_qpos, site_name,
                joint_qpos_idx, joint_dof_idx,
                target_pos, target_quat,
                q_target,  # warm-start with the ground-truth config
                pos_tol=1e-4, ori_tol=1e-3,
            )
            
            assert q_sol is not None, "mink IK failed to converge to known valid pose"
            
            # 3. Verify precision via FK
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = mujoco_utils.compute_site_error(model, data, site_id, target_pos, target_quat)
            
            pos_dist = np.linalg.norm(pos_err)
            rot_dist = np.linalg.norm(rot_err)
            
            assert pos_dist < 1e-4, f"Position error too high: {pos_dist}"
            assert rot_dist < 1e-3, f"Rotation error too high: {rot_dist}"
        finally:
            env.close()

    def test_enumerate_ik_solutions(self):
        """Test that enumerate_ik_solutions_for_arm returns valid solutions."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        try:
            robot_id = 0
            arm = "right"
            tf_manager = TransformManager(env)
            
            model = env.sim.model
            data = env.sim.data
            
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
            site_id = int(model.site_name2id(site_name))
            joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
            joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
            joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)
            
            model_native = mujoco_utils.get_native_mj_object(model)
            data_native = mujoco_utils.get_native_mj_object(data)
            
            # Use a known reachable configuration to create target pose
            rng = np.random.default_rng(42)
            q_target = rng.uniform(joint_lo + 0.1, joint_hi - 0.1)
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_target)
            mujoco.mj_fwdPosition(model_native, data_native)
            
            target_pos = np.array(data.site_xpos[site_id])
            target_mat = np.array(data.site_xmat[site_id]).reshape(3, 3)
            target_quat = R.from_matrix(target_mat).as_quat()
            
            # Create StampedPose in world frame (site poses are in world)
            target_pose = StampedPose(
                position=target_pos.tolist(),
                quaternion=target_quat.tolist(),
                frame_id=constants.WORLD_FRAME,
            )
            
            env.reset()
            
            solutions = ik_utils.enumerate_ik_solutions_for_arm(
                env,
                robot_id=robot_id,
                target_pose=target_pose,
                tf_manager=tf_manager,
                arm=arm,
                site_name=site_name,
                q_warm_start=q_target,
                max_solutions=3,
                num_seed_configs=4,
                elbow_discretization=2,
            )
            
            assert len(solutions) > 0, "enumerate_ik_solutions_for_arm returned no solutions"
            
            # Verify each solution is within joint limits
            for q_sol in solutions:
                assert q_sol.shape == (7,), f"Wrong shape: {q_sol.shape}"
                assert np.all(q_sol >= joint_lo - 1e-6), "Solution below joint limits"
                assert np.all(q_sol <= joint_hi + 1e-6), "Solution above joint limits"
        finally:
            env.close()

    def test_compute_arm_jacobian(self):
        """Test that mink-based Jacobian has correct shape and is non-degenerate."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        try:
            for robot_id in [0, 1]:
                J = ik_utils.compute_arm_jacobian(env, robot_id=robot_id, arm="right")
                
                # Shape check
                assert J.shape == (6, 7), f"Expected (6, 7), got {J.shape}"
                
                # Non-zero check (arm should not be at a singularity after reset)
                assert np.linalg.norm(J) > 0, "Jacobian is all zeros"
                
                # Rank check — a 7-DOF arm should have rank 6 generically
                rank = np.linalg.matrix_rank(J, tol=1e-6)
                assert rank >= 5, f"Jacobian rank too low: {rank}"
        finally:
            env.close()

    def test_mink_constants_centralized(self):
        """Verify all mink constants are accessible from constants module."""
        required = [
            'MINK_QP_SOLVER', 'MINK_POSITION_COST', 'MINK_ORIENTATION_COST',
            'MINK_FRAME_GAIN', 'MINK_LM_DAMPING', 'MINK_POSTURE_COST',
            'MINK_DAMPING_COST', 'MINK_DT', 'MINK_IK_DT', 'MINK_IK_MAX_ITERS',
            'MINK_IK_STALL_WINDOW', 'MINK_IK_STALL_TOL',
            'MINK_COLLISION_MIN_DIST', 'MINK_COLLISION_DETECT_DIST',
            'MINK_COLLISION_GAIN',
        ]
        for name in required:
            assert hasattr(constants, name), f"Missing constant: {name}"
        
        # Verify legacy scipy constants are removed
        removed = [
            'IK_TASK_WEIGHT', 'IK_MANIPULABILITY_WEIGHT',
            'IK_JOINT_LIMIT_WEIGHT', 'USE_GRAD_SCIPY_IK',
            'IK_DAMPING', 'IK_STEP_SIZE',
        ]
        for name in removed:
            assert not hasattr(constants, name), f"Legacy constant still present: {name}"
