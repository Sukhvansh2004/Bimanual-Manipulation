import pytest
import numpy as np
import mujoco
from unittest.mock import MagicMock, patch
from scipy.spatial.transform import Rotation as R
from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.planning.ik import ik_utils
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.planning.ik.eaik_solver import (
    compute_arm_jacobian,
    _fk_error,
    _sweep_values_for_joint,
    clear_robot_cache,
)
from bimanual_manipulation.planning.ik.scipy_solver import (
    enumerate_ik_solutions_scipy,
    _fk_objective_and_grad,
    _generate_seeds as scipy_generate_seeds,
)
from bimanual_manipulation.planning.ik.mink_solver import (
    enumerate_ik_solutions_mink,
    _build_target_se3,
    _generate_seeds as mink_generate_seeds,
)
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.planning.trajectory.generator import (
    generate_object_trajectory,
    generate_end_effector_trajectory,
)
from bimanual_manipulation.planning.collision_checker import (
    check_trajectory_feasibility,
)
from bimanual_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

# General FK tolerance used for cross-backend IK verification.
# This is the loosest tolerance across all backends (mink/scipy = 5e-3).
_GENERAL_FK_POS_TOL = 5e-3
_GENERAL_FK_ROT_TOL = 5e-3

# Backend identifiers for parametrization
_ALL_BACKENDS = ["eaik", "scipy", "mink"]

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

    @patch('bimanual_manipulation.planning.collision_checker.mujoco')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.set_qpos')
    @patch('bimanual_manipulation.planning.collision_checker.get_joint_qpos_indices')
    @patch('bimanual_manipulation.planning.collision_checker.get_arm_joint_ids')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.in_collision')
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

    @patch('bimanual_manipulation.planning.collision_checker.mujoco')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('bimanual_manipulation.planning.collision_checker.get_joint_qpos_indices')
    @patch('bimanual_manipulation.planning.collision_checker.get_arm_joint_ids')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
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

    @patch('bimanual_manipulation.planning.collision_checker.mujoco')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.mujoco')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.set_qpos')
    @patch('bimanual_manipulation.planning.collision_checker.get_joint_qpos_indices')
    @patch('bimanual_manipulation.planning.collision_checker.get_arm_joint_ids')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('bimanual_manipulation.planning.utils.trajectory_utils.in_collision')
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
        """Test that default (EAIK) IK finds valid solutions for the robot home pose."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        try:
            robot_id = 0
            arm = "right"
            
            model = env.sim.model
            data = env.sim.data
            
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
            joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
            joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
            joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)

            model_native = mujoco_utils.get_native_mj_object(model)
            data_native = mujoco_utils.get_native_mj_object(data)

            # Use the robot's current (home) configuration target
            site_id = int(model.site_name2id(site_name))
            target_pos = np.array(data.site_xpos[site_id])
            target_mat = np.array(data.site_xmat[site_id]).reshape(3, 3)
            target_quat = R.from_matrix(target_mat).as_quat()
            
            tf_manager = TransformManager(env)
            target_pose = StampedPose(
                position=target_pos.tolist(),
                quaternion=target_quat.tolist(),
                frame_id=constants.WORLD_FRAME,
            )
            
            solutions = ik_utils.enumerate_ik_solutions_for_arm(
                env,
                robot_id=robot_id,
                target_pose=target_pose,
                tf_manager=tf_manager,
                arm=arm,
                site_name=site_name,
                max_solutions=5,
            )
            
            assert len(solutions) > 0, "IK returned no solutions for home pose"
            
            for q_sol in solutions:
                assert q_sol.shape == (constants.FRANKA_ARM_DOF,), f"Wrong shape: {q_sol.shape}"
                assert np.all(q_sol >= joint_lo - 1e-6), "Solution below joint limits"
                assert np.all(q_sol <= joint_hi + 1e-6), "Solution above joint limits"

                # FK verification: set solution and check achieved pose
                mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
                mujoco.mj_fwdPosition(model_native, data_native)
                pos_err, rot_err = _fk_error(
                    data_native, site_id, target_pos, R.from_quat(target_quat).as_matrix(),
                )
                assert pos_err <= _GENERAL_FK_POS_TOL, (
                    f"FK pos error {pos_err:.6f} m > {_GENERAL_FK_POS_TOL}"
                )
                assert rot_err <= _GENERAL_FK_ROT_TOL, (
                    f"FK rot error {rot_err:.6f} rad > {_GENERAL_FK_ROT_TOL}"
                )
        finally:
            env.close()

    def test_enumerate_ik_solutions(self):
        """Test that enumerate_ik_solutions_for_arm returns FK-accurate solutions."""
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
                max_solutions=3,
            )
            
            assert len(solutions) > 0, "enumerate_ik_solutions_for_arm returned no solutions"
            
            # Verify each solution: joint limits + FK accuracy
            for q_sol in solutions:
                assert q_sol.shape == (7,), f"Wrong shape: {q_sol.shape}"
                assert np.all(q_sol >= joint_lo - 1e-6), "Solution below joint limits"
                assert np.all(q_sol <= joint_hi + 1e-6), "Solution above joint limits"

                # FK verification
                mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
                mujoco.mj_fwdPosition(model_native, data_native)
                pos_err, rot_err = _fk_error(
                    data_native, site_id, target_pos, R.from_quat(target_quat).as_matrix(),
                )
                assert pos_err <= _GENERAL_FK_POS_TOL, (
                    f"FK pos error {pos_err:.6f} m exceeds tolerance"
                )
                assert rot_err <= _GENERAL_FK_ROT_TOL, (
                    f"FK rot error {rot_err:.6f} rad exceeds tolerance"
                )
        finally:
            env.close()

    def test_compute_arm_jacobian(self):
        """Test that MuJoCo-based Jacobian has correct shape and is non-degenerate."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        
        try:
            for robot_id in [0, 1]:
                J = compute_arm_jacobian(env, robot_id=robot_id, arm="right")
                
                # Shape check
                assert J.shape == (6, 7), f"Expected (6, 7), got {J.shape}"
                
                # Non-zero check (arm should not be at a singularity after reset)
                assert np.linalg.norm(J) > 0, "Jacobian is all zeros"
                
                # Rank check — a 7-DOF arm should have rank 6 generically
                rank = np.linalg.matrix_rank(J, tol=1e-6)
                assert rank >= 5, f"Jacobian rank too low: {rank}"
        finally:
            env.close()


# =========================================================================
# Multi-backend parametrized IK tests
# =========================================================================

def _set_backend(monkeypatch, backend: str):
    """Patch the IK backend constant and clear EAIK robot cache."""
    monkeypatch.setattr(constants, "IK_SOLVER_BACKEND", backend)
    clear_robot_cache()


@pytest.fixture(scope="module")
def dual_arm_env():
    """Shared environment for all multi-backend IK tests (expensive to create)."""
    env = make_dual_arm_env(
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    env.reset()
    yield env
    env.close()


class TestIKSolverBackends:
    """General IK tests parametrized over all three backends (eaik, scipy, mink)."""

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_home_pose_solutions(self, monkeypatch, backend, dual_arm_env):
        """Each backend finds valid IK solutions for the robot home pose."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        site_id = int(model.site_name2id(site_name))
        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        tf_manager = TransformManager(env)
        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        solutions = ik_utils.enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=5,
        )

        assert len(solutions) > 0, f"[{backend}] returned no solutions for home pose"

        for q_sol in solutions:
            assert q_sol.shape == (constants.FRANKA_ARM_DOF,)
            assert np.all(q_sol >= joint_lo - 1e-6), f"[{backend}] below joint limits"
            assert np.all(q_sol <= joint_hi + 1e-6), f"[{backend}] above joint limits"

            # FK verification
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = _fk_error(
                data_native, site_id, target_pos, R.from_quat(target_quat).as_matrix(),
            )
            assert pos_err <= _GENERAL_FK_POS_TOL, (
                f"[{backend}] FK pos error {pos_err:.6f} m"
            )
            assert rot_err <= _GENERAL_FK_ROT_TOL, (
                f"[{backend}] FK rot error {rot_err:.6f} rad"
            )

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_random_reachable_target(self, monkeypatch, backend, dual_arm_env):
        """Each backend finds FK-accurate solutions for a random reachable target."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        # Set a random valid configuration and read its FK pose
        rng = np.random.default_rng(42)
        q_target = rng.uniform(joint_lo + 0.1, joint_hi - 0.1)
        mujoco_utils.set_qpos(data, joint_qpos_idx, q_target)
        mujoco.mj_fwdPosition(model_native, data_native)

        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        env.reset()
        tf_manager = TransformManager(env)

        solutions = ik_utils.enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=3,
        )

        assert len(solutions) > 0, f"[{backend}] returned no solutions for reachable target"

        for q_sol in solutions:
            assert q_sol.shape == (7,)
            assert np.all(q_sol >= joint_lo - 1e-3), (
                f"[{backend}] solution below joint limits"
            )
            assert np.all(q_sol <= joint_hi + 1e-3), (
                f"[{backend}] solution above joint limits"
            )

            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = _fk_error(
                data_native, site_id, target_pos, R.from_quat(target_quat).as_matrix(),
            )
            assert pos_err <= _GENERAL_FK_POS_TOL, (
                f"[{backend}] FK pos error {pos_err:.6f} m"
            )
            assert rot_err <= _GENERAL_FK_ROT_TOL, (
                f"[{backend}] FK rot error {rot_err:.6f} rad"
            )

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_both_robots(self, monkeypatch, backend, dual_arm_env):
        """Each backend solves IK for both robot0 and robot1."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        tf_manager = TransformManager(env)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        for robot_id in [0, 1]:
            arm = "right"
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
            site_id = int(model.site_name2id(site_name))
            joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
            joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
            joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)

            target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
            target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
            target_quat = R.from_matrix(target_mat).as_quat()

            target_pose = StampedPose(
                position=target_pos.tolist(),
                quaternion=target_quat.tolist(),
                frame_id=constants.WORLD_FRAME,
            )

            solutions = ik_utils.enumerate_ik_solutions_for_arm(
                env,
                robot_id=robot_id,
                target_pose=target_pose,
                tf_manager=tf_manager,
                arm=arm,
                site_name=site_name,
                max_solutions=3,
            )

            assert len(solutions) > 0, (
                f"[{backend}] no solutions for robot{robot_id} home pose"
            )

            for q_sol in solutions:
                assert np.all(q_sol >= joint_lo - 1e-6)
                assert np.all(q_sol <= joint_hi + 1e-6)

                mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
                mujoco.mj_fwdPosition(model_native, data_native)
                pos_err, rot_err = _fk_error(
                    data_native, site_id, target_pos,
                    R.from_quat(target_quat).as_matrix(),
                )
                assert pos_err <= _GENERAL_FK_POS_TOL, (
                    f"[{backend}] robot{robot_id}: pos err {pos_err:.6f} m"
                )
                assert rot_err <= _GENERAL_FK_ROT_TOL, (
                    f"[{backend}] robot{robot_id}: rot err {rot_err:.6f} rad"
                )

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_unreachable_target(self, monkeypatch, backend, dual_arm_env):
        """Each backend returns no solutions for a target far outside the workspace."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        tf_manager = TransformManager(env)
        target_pose = StampedPose(
            position=[100.0, 100.0, 100.0],
            quaternion=[0, 0, 0, 1],
            frame_id=constants.WORLD_FRAME,
        )

        solutions = ik_utils.enumerate_ik_solutions_for_arm(
            env,
            robot_id=0,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm="right",
            max_solutions=3,
        )

        assert len(solutions) == 0, (
            f"[{backend}] Expected no solutions for unreachable target, got {len(solutions)}"
        )

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_warm_start_proximity(self, monkeypatch, backend, dual_arm_env):
        """Solutions with q_reference are sorted by proximity to the reference."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        tf_manager = TransformManager(env)
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))

        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        q_ref = np.array(data.qpos, dtype=float).ravel()[joint_qpos_idx]

        solutions = ik_utils.enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=5,
            q_reference=q_ref,
        )

        assert len(solutions) > 0, f"[{backend}] No warm-started solutions"

        if len(solutions) > 1:
            dist_first = np.linalg.norm(solutions[0] - q_ref)
            dist_last = np.linalg.norm(solutions[-1] - q_ref)
            assert dist_first <= dist_last + 1e-10, (
                f"[{backend}] solutions not sorted by proximity to q_reference"
            )

    @pytest.mark.parametrize("backend", _ALL_BACKENDS)
    def test_jacobian_shape_and_rank(self, monkeypatch, backend, dual_arm_env):
        """Each backend's compute_arm_jacobian returns a 6x7 non-degenerate Jacobian."""
        _set_backend(monkeypatch, backend)
        env = dual_arm_env
        env.reset()

        for robot_id in [0, 1]:
            J = ik_utils.compute_arm_jacobian(env, robot_id=robot_id, arm="right")
            assert J.shape == (6, 7), f"[{backend}] Expected (6,7), got {J.shape}"
            assert np.linalg.norm(J) > 0, f"[{backend}] Jacobian is all zeros"
            rank = np.linalg.matrix_rank(J, tol=1e-6)
            assert rank >= 5, f"[{backend}] Jacobian rank too low: {rank}"


# =========================================================================
# EAIK-specific tests
# =========================================================================

class TestEAIKSolver:
    """Tests specific to the EAIK analytical IK backend."""

    def test_fk_error_at_current_pose(self):
        """_fk_error returns near-zero errors for the current site pose."""
        env = make_dual_arm_env(
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        )
        env.reset()
        try:
            data_native = mujoco_utils.get_native_mj_object(env.sim.data)
            model = env.sim.model
            site_name = mujoco_utils.get_eef_site_name(0, arm="right")
            site_id = int(model.site_name2id(site_name))

            pos = np.array(data_native.site_xpos[site_id], dtype=float)
            mat = np.array(data_native.site_xmat[site_id], dtype=float).reshape(3, 3)
            R_target_mat = R.from_matrix(mat).as_matrix()

            pos_err, rot_err = _fk_error(data_native, site_id, pos, R_target_mat)
            assert pos_err < 1e-10, f"Expected ~0 pos error, got {pos_err}"
            assert rot_err < 1e-6, f"Expected ~0 rot error, got {rot_err}"

            # Perturbed target → non-zero error
            pos_err2, _ = _fk_error(data_native, site_id, pos + [0.05, 0, 0], R_target_mat)
            assert pos_err2 > 0.04, f"Expected significant pos error, got {pos_err2}"
        finally:
            env.close()

    def test_sweep_values_full_range(self):
        """_sweep_values_for_joint covers the full range without q_reference."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])
        vals = _sweep_values_for_joint(0, lo, hi, num_sweep=8, q_reference=None)
        assert len(vals) == 8
        assert vals[0] == pytest.approx(lo[0])
        assert vals[-1] == pytest.approx(hi[0])
        assert np.all(np.diff(vals) > 0)

    def test_sweep_values_warm_start(self):
        """_sweep_values_for_joint is centred on q_reference within the window."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])
        q_ref = np.zeros(7)
        vals = _sweep_values_for_joint(0, lo, hi, num_sweep=8, q_reference=q_ref)
        half = constants.EAIK_WARM_START_WINDOW
        assert np.all(vals >= -half - 1e-10)
        assert np.all(vals <= half + 1e-10)
        assert np.any(np.isclose(vals, 0.0, atol=1e-8))

    def test_sweep_values_empty_for_invalid_limits(self):
        """_sweep_values_for_joint returns empty for invalid limits."""
        lo = np.array([1.0])
        hi = np.array([0.0])
        vals = _sweep_values_for_joint(0, lo, hi, num_sweep=8, q_reference=None)
        assert len(vals) == 0

    def test_eaik_constants_centralized(self):
        """All EAIK constants are accessible and valid."""
        required = {
            'EAIK_NUM_SWEEP_VALUES': int,
            'EAIK_WARM_START_WINDOW': float,
            'EAIK_FK_POS_TOLERANCE': float,
            'EAIK_FK_ROT_TOLERANCE': float,
        }
        for name, expected_type in required.items():
            assert hasattr(constants, name), f"Missing EAIK constant: {name}"
            val = getattr(constants, name)
            assert isinstance(val, expected_type)
            assert val > 0

        assert constants.EAIK_FK_POS_TOLERANCE <= 0.01
        assert constants.EAIK_FK_ROT_TOLERANCE <= 0.1

    def test_eaik_tight_fk_tolerance(self, dual_arm_env):
        """EAIK solutions meet the tighter EAIK-specific FK tolerances."""
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        tf_manager = TransformManager(env)
        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        from bimanual_manipulation.planning.ik.eaik_solver import (
            enumerate_ik_solutions_eaik,
        )

        solutions = enumerate_ik_solutions_eaik(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=5,
        )

        assert len(solutions) > 0, "EAIK returned no solutions"

        for q_sol in solutions:
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = _fk_error(
                data_native, site_id, target_pos,
                R.from_quat(target_quat).as_matrix(),
            )
            assert pos_err <= constants.EAIK_FK_POS_TOLERANCE, (
                f"EAIK FK pos error {pos_err:.6f} m > {constants.EAIK_FK_POS_TOLERANCE}"
            )
            assert rot_err <= constants.EAIK_FK_ROT_TOLERANCE, (
                f"EAIK FK rot error {rot_err:.6f} rad > {constants.EAIK_FK_ROT_TOLERANCE}"
            )


# =========================================================================
# Scipy-specific tests
# =========================================================================

class TestScipySolver:
    """Tests specific to the Scipy L-BFGS-B IK backend."""

    def test_scipy_constants_centralized(self):
        """All Scipy IK constants are accessible and valid."""
        required = {
            'SCIPY_IK_NUM_SEEDS': int,
            'SCIPY_IK_MAX_ITERS': int,
            'SCIPY_IK_POS_TOLERANCE': float,
            'SCIPY_IK_ROT_TOLERANCE': float,
        }
        for name, expected_type in required.items():
            assert hasattr(constants, name), f"Missing Scipy constant: {name}"
            val = getattr(constants, name)
            assert isinstance(val, expected_type)
            assert val > 0

    def test_scipy_generate_seeds_with_reference(self):
        """Scipy seed generation includes q_reference as first seed."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])
        q_ref = np.zeros(7)

        seeds = scipy_generate_seeds(q_ref, lo, hi, n_seeds=8)
        assert len(seeds) == 8
        np.testing.assert_array_equal(seeds[0], q_ref)
        for s in seeds:
            assert s.shape == (7,)

    def test_scipy_generate_seeds_uniform(self):
        """Scipy seed generation without reference uses uniform random seeds."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])

        seeds = scipy_generate_seeds(None, lo, hi, n_seeds=10)
        assert len(seeds) == 10
        for s in seeds:
            assert s.shape == (7,)
            assert np.all(s >= lo)
            assert np.all(s <= hi)

    def test_scipy_generate_seeds_with_current(self):
        """Scipy seed generation includes q_current when distinct from q_reference."""
        lo = np.zeros(7) - 3.0
        hi = np.zeros(7) + 3.0
        q_ref = np.ones(7) * 0.5
        q_cur = np.ones(7) * (-0.5)

        seeds = scipy_generate_seeds(q_ref, lo, hi, n_seeds=8, q_current=q_cur)
        assert len(seeds) == 8
        np.testing.assert_array_equal(seeds[0], q_ref)
        # q_current should be second since it's different from q_ref
        np.testing.assert_array_equal(seeds[1], q_cur)

    def test_scipy_objective_gradient_shape(self, dual_arm_env):
        """_fk_objective_and_grad returns (scalar, 7-vector)."""
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        site_id = mujoco.mj_name2id(
            model_native, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

        target_pos = np.array(data_native.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data_native.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()
        R_target = R.from_quat(target_quat)

        q_arm = full_qpos[joint_qpos_idx].copy()
        cost, grad = _fk_objective_and_grad(
            q_arm, model_native, data_native, site_id,
            joint_qpos_idx, joint_dof_idx, full_qpos,
            target_pos, R_target,
        )

        assert isinstance(cost, float)
        assert grad.shape == (7,)
        # At current pose, cost should be near zero
        assert cost < 1e-10, f"Cost at own pose should be ~0, got {cost}"

    def test_scipy_direct_solve(self, dual_arm_env):
        """Scipy solver directly finds solutions for the home pose."""
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        tf_manager = TransformManager(env)
        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        solutions = enumerate_ik_solutions_scipy(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=3,
        )

        assert len(solutions) > 0, "Scipy solver returned no solutions"

        for q_sol in solutions:
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = _fk_error(
                data_native, site_id, target_pos,
                R.from_quat(target_quat).as_matrix(),
            )
            assert pos_err <= constants.SCIPY_IK_POS_TOLERANCE, (
                f"Scipy FK pos error {pos_err:.6f} m"
            )
            assert rot_err <= constants.SCIPY_IK_ROT_TOLERANCE, (
                f"Scipy FK rot error {rot_err:.6f} rad"
            )


# =========================================================================
# Mink-specific tests
# =========================================================================

class TestMinkSolver:
    """Tests specific to the Mink QP-based IK backend."""

    def test_mink_constants_centralized(self):
        """All mink constants are accessible from constants module."""
        required = [
            'MINK_QP_SOLVER', 'MINK_POSITION_COST', 'MINK_ORIENTATION_COST',
            'MINK_FRAME_GAIN', 'MINK_LM_DAMPING', 'MINK_POSTURE_COST',
            'MINK_DAMPING_COST', 'MINK_DT', 'MINK_IK_DT',
            'MINK_COLLISION_MIN_DIST', 'MINK_COLLISION_DETECT_DIST',
            'MINK_COLLISION_GAIN',
            'MINK_IK_NUM_SEEDS', 'MINK_IK_MAX_ITERS',
            'MINK_IK_POS_TOLERANCE', 'MINK_IK_ROT_TOLERANCE',
            'MINK_IK_STALL_WINDOW', 'MINK_IK_STALL_TOL',
        ]
        for name in required:
            assert hasattr(constants, name), f"Missing mink constant: {name}"

    def test_mink_build_target_se3(self):
        """_build_target_se3 correctly encodes position and orientation."""
        import mink

        pos = np.array([1.0, 2.0, 3.0])
        quat = R.from_euler('z', 90, degrees=True).as_quat()

        se3 = _build_target_se3(pos, quat)
        T = se3.as_matrix()

        np.testing.assert_allclose(T[:3, 3], pos, atol=1e-10)
        expected_rot = R.from_quat(quat).as_matrix()
        np.testing.assert_allclose(T[:3, :3], expected_rot, atol=1e-10)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_mink_generate_seeds_with_reference(self):
        """Mink seed generation includes q_reference as first seed."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])
        q_ref = np.zeros(7)

        seeds = mink_generate_seeds(q_ref, lo, hi, n_seeds=8)
        assert len(seeds) == 8
        np.testing.assert_array_equal(seeds[0], q_ref)

    def test_mink_generate_seeds_uniform(self):
        """Mink seed generation without reference uses uniform random seeds."""
        lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        hi = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.75, 2.8])

        seeds = mink_generate_seeds(None, lo, hi, n_seeds=10)
        assert len(seeds) == 10
        for s in seeds:
            assert s.shape == (7,)
            assert np.all(s >= lo)
            assert np.all(s <= hi)

    def test_mink_generate_seeds_with_current(self):
        """Mink seed generation includes q_current when distinct from q_reference."""
        lo = np.zeros(7) - 3.0
        hi = np.zeros(7) + 3.0
        q_ref = np.ones(7) * 0.5
        q_cur = np.ones(7) * (-0.5)

        seeds = mink_generate_seeds(q_ref, lo, hi, n_seeds=8, q_current=q_cur)
        assert len(seeds) == 8
        np.testing.assert_array_equal(seeds[0], q_ref)
        np.testing.assert_array_equal(seeds[1], q_cur)

    def test_mink_direct_solve(self, dual_arm_env):
        """Mink solver directly finds solutions for the home pose."""
        env = dual_arm_env
        env.reset()

        model = env.sim.model
        data = env.sim.data
        robot_id = 0
        arm = "right"

        site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
        site_id = int(model.site_name2id(site_name))
        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        target_pos = np.array(data.site_xpos[site_id], dtype=float).copy()
        target_mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        target_quat = R.from_matrix(target_mat).as_quat()

        tf_manager = TransformManager(env)
        target_pose = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=constants.WORLD_FRAME,
        )

        solutions = enumerate_ik_solutions_mink(
            env,
            robot_id=robot_id,
            target_pose=target_pose,
            tf_manager=tf_manager,
            arm=arm,
            site_name=site_name,
            max_solutions=3,
        )

        assert len(solutions) > 0, "Mink solver returned no solutions"

        for q_sol in solutions:
            mujoco_utils.set_qpos(data, joint_qpos_idx, q_sol)
            mujoco.mj_fwdPosition(model_native, data_native)
            pos_err, rot_err = _fk_error(
                data_native, site_id, target_pos,
                R.from_quat(target_quat).as_matrix(),
            )
            assert pos_err <= constants.MINK_IK_POS_TOLERANCE, (
                f"Mink FK pos error {pos_err:.6f} m"
            )
            assert rot_err <= constants.MINK_IK_ROT_TOLERANCE, (
                f"Mink FK rot error {rot_err:.6f} rad"
            )
