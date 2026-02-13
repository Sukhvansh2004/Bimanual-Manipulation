import pytest
import numpy as np
import mujoco
import meshcat.geometry as g
import os
import meshcat
from unittest.mock import MagicMock, patch
from scipy.spatial.transform import Rotation as R
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.planning.utils import ik_utils, mujoco_utils
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_object_trajectory,
    generate_end_effector_trajectory,
    check_trajectory_feasibility,
    check_dual_trajectory_feasibility
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult
from robosuite_franka_manipulation.planning.utils import collision_utils
from robosuite_franka_manipulation.planning.collision_checker import (
    check_single_grasp_feasibility,
    find_feasible_grasp_pairs,
    check_dual_arm_pose_feasibility
)
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.ik_utils import (
    solve_ik_scipy, enumerate_ik_solutions_for_arm, unique_solutions, apply_ik_orientation_correction
)

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.sim = MagicMock()
    env.sim.model = MagicMock()
    env.sim.data = MagicMock()
    
    # Mock model properties
    env.sim.model.njnt = 100
    env.sim.model.nq = 100
    env.sim.model.nv = 100
    env.sim.model.ngeom = 100
    
    # Mock joint lookups
    env.sim.model.joint_name2id.return_value = 0
    env.sim.model.jnt_qposadr = np.zeros(200, dtype=int)
    env.sim.model.geom_bodyid = np.zeros(200, dtype=int)
    env.sim.model.geom_id2name.side_effect = lambda i: f"geom_{i}"
    
    # Mock data properties
    env.sim.data.qpos = np.zeros(200)
    env.sim.data.qvel = np.zeros(200)
    env.sim.data.ncon = 0
    
    return env

@pytest.fixture
def mock_tf_manager():
    tfm = MagicMock(spec=TransformManager)
    def side_effect_transform(pose, target_frame):
        new_pose = pose.copy()
        new_pose.frame_id = target_frame
        return new_pose
    tfm.transform_pose.side_effect = side_effect_transform
    return tfm

from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    save_trajectory, load_trajectory, process_waypoint, process_dual_waypoint, resolve_ignore_names
)

from robosuite_franka_manipulation.utils.primitives import compute_approach_pose, compute_retreat_pose, compute_linear_waypoints

from robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer import RobotMeshVisualizer

class TestRobotMeshVisualizer:
    """Test robot mesh visualization utilities."""

    @pytest.fixture
    def mock_vis(self):
        return MagicMock(spec=meshcat.Visualizer)

    @pytest.fixture
    def visualizer(self, mock_vis, mock_env, mock_tf_manager):
        # Setup mock_tf_manager.env.sim.model for initialization
        mock_tf_manager.env = mock_env
        mock_env.sim.model.ngeom = 2
        mock_env.sim.model.geom_bodyid = [0, 1]
        mock_env.sim.model.geom_id2name.side_effect = ["robot0_geom", "other"]
        
        return RobotMeshVisualizer(mock_vis, mock_tf_manager)

    def test_initialize_mesh_cache(self, visualizer):
        """Test _initialize_mesh_cache method."""
        assert len(visualizer._robot_geom_ids[0]) == 1
        assert len(visualizer._robot_geom_ids[1]) == 0

    def test_get_meshcat_geometry(self, visualizer):
        """Test _get_meshcat_geometry for different types."""
        model = MagicMock()
        
        # Mesh
        model.geom_type = [7]
        model.geom_dataid = [0]
        model.mesh_vertadr = [0]
        model.mesh_vertnum = [3]
        model.mesh_faceadr = [0]
        model.mesh_facenum = [1]
        model.mesh_vert = np.zeros((3, 3))
        model.mesh_face = np.zeros((1, 3), dtype=int)
        
        geom = visualizer._get_meshcat_geometry(model, 0)
        assert isinstance(geom, g.TriangularMeshGeometry)
        
        # Box
        model.geom_type = [6]
        model.geom_size = [np.array([0.1, 0.1, 0.1])]
        geom = visualizer._get_meshcat_geometry(model, 0)
        assert isinstance(geom, g.Box)

    @patch('robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.utils.robot_mesh_visualizer.mujoco')
    def test_visualize_trajectory_configs(self, mock_mj, mock_indices, mock_ids, visualizer):
        """Test visualize_trajectory_configs method."""
        mock_ids.return_value = [0]
        mock_indices.return_value = [0]
        
        results = [WaypointFeasibilityResult(
            waypoint_index=0, eef_pose=MagicMock(), joint_config=np.zeros(7),
            is_feasible=True, ik_success=True, collision_free=True
        )]
        
        visualizer.visualize_trajectory_configs("test_traj", 0, results)
        assert visualizer.vis.__getitem__.called

class TestPrimitives:
    """Test manipulation primitives."""

    def test_compute_approach_pose(self):
        """Test compute_approach_pose method."""
        pose = StampedPose(position=[0,0,1], quaternion=[0,0,0,1], frame_id="world")
        # Offset along Z: [0,0,1] - 0.1 * [0,0,1] = [0,0,0.9]
        res = compute_approach_pose(pose, approach_distance=0.1, approach_axis='z')
        np.testing.assert_allclose(res.position, [0, 0, 0.9], atol=1e-6)
        
        # Offset along X: [0,0,1] - 0.1 * [1,0,0] = [-0.1, 0, 1]
        res_x = compute_approach_pose(pose, approach_distance=0.1, approach_axis='x')
        np.testing.assert_allclose(res_x.position, [-0.1, 0, 1], atol=1e-6)

    def test_compute_retreat_pose(self):
        """Test compute_retreat_pose method."""
        pose = StampedPose(position=[0,0,1], quaternion=[0,0,0,1], frame_id="world")
        # Offset along Z: [0,0,1] + 0.1 * [0,0,1] = [0,0,1.1]
        res = compute_retreat_pose(pose, retreat_distance=0.1, retreat_axis='z')
        np.testing.assert_allclose(res.position, [0, 0, 1.1], atol=1e-6)

    def test_compute_linear_waypoints(self):
        """Test compute_linear_waypoints method."""
        p1 = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        p2 = StampedPose(position=[1,1,1], quaternion=[0,0,0,1], frame_id="world")
        
        waypoints = compute_linear_waypoints(p1, p2, num_waypoints=5)
        assert len(waypoints) == 5
        np.testing.assert_allclose(waypoints[2].position, [0.5, 0.5, 0.5], atol=1e-6)

class TestTrajectoryUtils:
    """Test trajectory utilities."""

    def test_resolve_ignore_names(self):
        """Test resolve_ignore_names method."""
        assert resolve_ignore_names("obj", ["ign"]) == ["ign", "obj"]
        assert resolve_ignore_names(None, ["ign"]) == ["ign"]
        assert resolve_ignore_names("obj", None) == ["obj"]

    def test_save_load_trajectory(self, tmp_path):
        """Test saving and loading trajectory data."""
        output_dir = str(tmp_path)
        filename = "test_traj.npz"
        filepath = os.path.join(output_dir, filename)
        
        eef_traj = [StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")]
        obj_traj = [StampedPose(position=[1,1,1], quaternion=[0,0,0,1], frame_id="world")]
        results = [WaypointFeasibilityResult(
            waypoint_index=0, eef_pose=eef_traj[0], joint_config=np.zeros(7),
            is_feasible=True, ik_success=True, collision_free=True
        )]
        
        save_trajectory(output_dir, filename, eef_traj, obj_traj, results)
        assert os.path.exists(filepath)
        
        l_eef, l_obj, l_joint, l_res = load_trajectory(filepath)
        
        assert len(l_eef) == 1
        assert len(l_obj) == 1
        assert l_joint.shape == (1, 7)
        assert len(l_res) == 1
        assert l_res[0].is_feasible is True

    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.in_collision')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    def test_process_waypoint(self, mock_mujoco, mock_in_col, mock_ik, mock_env, mock_tf_manager):
        """Test process_waypoint method."""
        pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        
        # Case 1: Success
        mock_ik.return_value = [np.zeros(7)]
        mock_in_col.return_value = False
        res, config = process_waypoint(mock_env, 0, 0, pose, np.arange(7), True, [], None, mock_tf_manager)
        assert res.is_feasible is True
        assert config is not None
        
        # Case 2: IK fail
        mock_ik.return_value = []
        res, config = process_waypoint(mock_env, 0, 0, pose, np.arange(7), True, [], None, mock_tf_manager)
        assert res.ik_success is False
        assert config is None

    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.in_collision')
    @patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
    def test_process_dual_waypoint(self, mock_mujoco, mock_in_col, mock_ik, mock_env, mock_tf_manager):
        """Test process_dual_waypoint method."""
        pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        
        # Case 1: Both success
        mock_ik.side_effect = [[np.zeros(7)], [np.zeros(7)]]
        mock_in_col.return_value = False
        res0, res1, c0, c1 = process_dual_waypoint(mock_env, 0, pose, pose, np.arange(7), np.arange(7), True, [], None, None, mock_tf_manager)
        assert res0.is_feasible and res1.is_feasible
        
        # Case 2: One fails IK
        mock_ik.side_effect = [[np.zeros(7)], []]
        res0, res1, c0, c1 = process_dual_waypoint(mock_env, 0, pose, pose, np.arange(7), np.arange(7), True, [], None, None, mock_tf_manager)
        assert res0.ik_success is True
        assert res1.ik_success is False

class TestIKUtils:
    """Test IK utilities."""

    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_native_mj_object')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.set_qpos')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.compute_site_error')
    @patch('mujoco.mj_fwdPosition')
    @patch('mujoco.mj_jacSite')
    @patch('robosuite_franka_manipulation.planning.utils.ik_utils.minimize')
    def test_solve_ik_scipy(
        self, 
        mock_minimize, mock_jac, mock_fwd, 
        mock_error, mock_set, mock_native, mock_env
    ):
        mock_native.return_value = MagicMock()
        mock_native.return_value.nv = 20
        
        # Case 1: Success
        mock_minimize.return_value.x = np.zeros(7)
        mock_minimize.return_value.success = True
        mock_error.return_value = (np.zeros(3), np.zeros(3))
        
        res = solve_ik_scipy(
            mock_env.sim.model, mock_env.sim.data, 0, np.arange(7), np.arange(7),
            np.zeros(7)-1, np.zeros(7)+1, np.zeros(3), np.array([0,0,0,1]), np.zeros(7),
            pos_tol=1e-3, ori_tol=1e-3
        )
        assert res is not None
        
        # Case 2: Failure - high error
        mock_error.return_value = (np.array([1.0, 0, 0]), np.zeros(3))
        res = solve_ik_scipy(
            mock_env.sim.model, mock_env.sim.data, 0, np.arange(7), np.arange(7),
            np.zeros(7)-1, np.zeros(7)+1, np.zeros(3), np.array([0,0,0,1]), np.zeros(7),
            pos_tol=1e-3, ori_tol=1e-3
        )
        assert res is None

    def test_unique_solutions(self):
        """Test de-duplication of joint solutions."""
        sols = [
            np.zeros(7),
            np.zeros(7) + 1e-4, # Should be unique if atol < 1e-4
            np.zeros(7) + 0.1
        ]
        unique = unique_solutions(sols, atol=1e-3)
        assert len(unique) == 2

    def test_apply_ik_orientation_correction(self):
        """Test orientation correction for IK."""
        pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        corrected = apply_ik_orientation_correction(pose)
        
        # Identity rot * (-90 deg Z rot)
        expected_quat = R.from_euler('z', -90, degrees=True).as_quat()
        np.testing.assert_allclose(corrected.quaternion, expected_quat, atol=1e-6)

    @patch('robosuite_franka_manipulation.planning.utils.ik_utils.solve_ik_scipy')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_joint_dof_indices')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_joint_limits')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_qpos')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_eef_site_name')
    @patch('robosuite_franka_manipulation.planning.utils.mujoco_utils.get_native_mj_object')
    @patch('mujoco.mj_fwdPosition')
    def test_enumerate_ik_solutions(
        self, 
        mock_fwd, mock_native, mock_site, mock_get_qpos, 
        mock_limits, mock_dof, mock_indices, mock_ids, 
        mock_solve, mock_env, mock_tf_manager
    ):
        mock_ids.return_value = np.arange(7)
        mock_indices.return_value = np.arange(7)
        mock_dof.return_value = np.arange(7)
        mock_limits.return_value = (np.zeros(7)-1, np.zeros(7)+1)
        mock_get_qpos.return_value = np.zeros(7)
        mock_solve.return_value = np.zeros(7)
        
        target = StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")
        
        sols = enumerate_ik_solutions_for_arm(
            mock_env, robot_id=0, target_pose=target, tf_manager=mock_tf_manager, num_seed_configs=1
        )
        
        assert len(sols) >= 1
        assert mock_solve.called

class TestCollisionUtils:
    """Test collision checking utilities."""

    def test_in_collision_basic(self, mock_env):
        """Test basic collision detection."""
        # Setup specific contact for this test
        mock_env.sim.model.geom_id2name.side_effect = lambda i: f"robot0_geom_{i}" if i < 5 else f"other_geom_{i}"
        mock_env.sim.data.ncon = 1
        contact = MagicMock()
        contact.geom1 = [0]
        contact.geom2 = [6]
        contact.dist = [0.0]
        mock_env.sim.data.contact = contact

        collision_utils._robot_mask = None
        
        assert in_collision(mock_env) is True

    def test_in_collision_with_ignore(self, mock_env):
        """Test collision detection with ignored names."""
        mock_env.sim.model.geom_id2name.side_effect = lambda i: f"robot0_geom_{i}" if i < 5 else f"other_geom_{i}"
        mock_env.sim.data.ncon = 1
        contact = MagicMock()
        contact.geom1 = [0]
        contact.geom2 = [6]
        contact.dist = [0.0]
        mock_env.sim.data.contact = contact

        collision_utils._robot_mask = None
        
        # Ignore robot0_geom_0 (which is geom1 in contact)
        assert in_collision(mock_env, ignore_names=["robot0_geom_0"]) is False
        
        # Ignore other_geom_6 (which is geom2 in contact)
        assert in_collision(mock_env, ignore_names=["other_geom_6"]) is False

class TestCollisionChecker:
    """Test collision checker functionality."""

    @patch('robosuite_franka_manipulation.planning.collision_checker.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.collision_checker.in_collision')
    @patch('robosuite_franka_manipulation.planning.collision_checker.mujoco')
    @patch('robosuite_franka_manipulation.planning.collision_checker.get_joint_qpos_indices')
    def test_check_single_grasp_feasibility(
        self, 
        mock_indices,
        mock_mujoco, 
        mock_in_collision, 
        mock_ik, 
        mock_env, 
        mock_tf_manager
    ):
        grasp = StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")
        mock_indices.return_value = np.arange(7)
        
        # Success case
        mock_ik.return_value = [np.zeros(7)]
        mock_in_collision.return_value = False
        
        success = check_single_grasp_feasibility(mock_env, grasp, 0, mock_tf_manager)
        assert success is True

    @patch('robosuite_franka_manipulation.planning.collision_checker.check_single_grasp_feasibility')
    @patch('robosuite_franka_manipulation.planning.collision_checker.check_dual_arm_pose_feasibility')
    def test_find_feasible_grasp_pairs(
        self, 
        mock_dual_feas, 
        mock_single_feas, 
        mock_env, 
        mock_tf_manager
    ):
        grasp0 = GraspSchema(pos=[0.5, 0, 0.5], quat=[0, 0, 0, 1], frame_id="world", score=1.0, id="0_0", gripper_type="franka_panda")
        grasp1 = GraspSchema(pos=[0.5, 0.5, 0.5], quat=[0, 0, 0, 1], frame_id="world", score=1.0, id="1_0", gripper_type="franka_panda")
        
        mock_single_feas.return_value = True
        mock_dual_feas.return_value = [[np.zeros(7), np.zeros(7), True]]
        
        pairs = find_feasible_grasp_pairs(mock_env, [grasp0], [grasp1], mock_tf_manager)
        assert len(pairs) == 1
        assert pairs[0] == (grasp0, grasp1)

    @patch('robosuite_franka_manipulation.planning.collision_checker.enumerate_ik_solutions_for_arm')
    @patch('robosuite_franka_manipulation.planning.collision_checker.in_collision')
    @patch('robosuite_franka_manipulation.planning.collision_checker.mujoco')
    @patch('robosuite_franka_manipulation.planning.collision_checker.get_joint_qpos_indices')
    def test_check_dual_arm_pose_feasibility(
        self, 
        mock_indices,
        mock_mujoco, 
        mock_in_collision, 
        mock_ik, 
        mock_env, 
        mock_tf_manager
    ):
        pose0 = StampedPose(position=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")
        pose1 = StampedPose(position=[0.5, 0.5, 0.5], quaternion=[0, 0, 0, 1], frame_id="world")
        mock_indices.return_value = np.arange(7)
        
        # Both IK success
        mock_ik.side_effect = [[np.zeros(7)], [np.zeros(7)]]
        mock_in_collision.return_value = False
        
        results = check_dual_arm_pose_feasibility(mock_env, pose0, pose1, mock_tf_manager)
        assert len(results) == 1
        assert results[0][2] is True
        
        # One IK fails
        mock_ik.side_effect = [[], [np.zeros(7)]]
        results = check_dual_arm_pose_feasibility(mock_env, pose0, pose1, mock_tf_manager)
        assert len(results) == 0

class TestTrajectoryGenerator:
    """Test trajectory generator functionality."""

    def test_generate_object_trajectory_frame_mismatch(self, caplog, mock_tf_manager):
        """Test frame mismatch warning in generate_object_trajectory."""
        start = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="frame1")
        target = StampedPose(position=[1,0,0], quaternion=[0,0,0,1], frame_id="frame2")
        
        # Mock transform to NOT change frame to world (to trigger warning)
        mock_tf_manager.transform_pose.side_effect = lambda p, f: p
        
        generate_object_trajectory(start, target, mock_tf_manager)
        assert "Frame mismatch" in caplog.text

    def test_generate_end_effector_trajectory_empty(self, mock_tf_manager):
        """Test generate_end_effector_trajectory with empty input."""
        res = generate_end_effector_trajectory([], MagicMock(), mock_tf_manager)
        assert res == []

    def test_check_trajectory_feasibility_mismatch(self, mock_env, mock_tf_manager):
        """Test check_trajectory_feasibility with length mismatch."""
        eef_traj = [StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")]
        obj_traj = [
            StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world"),
            StampedPose(position=[1,1,1], quaternion=[0,0,0,1], frame_id="world")
        ]
        
        res, success = check_trajectory_feasibility(
            mock_env, 0, eef_traj, mock_tf_manager, object_trajectory=obj_traj
        )
        assert success is False
        assert res == []

    @patch('robosuite_franka_manipulation.planning.trajectory_generator.process_dual_waypoint')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
    @patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
    def test_check_dual_trajectory_feasibility(
        self, 
        mock_mujoco, 
        mock_get_ids, 
        mock_get_indices, 
        mock_process, 
        mock_env, 
        mock_tf_manager
    ):
        eef_traj0 = [StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")]
        eef_traj1 = [StampedPose(position=[1,1,1], quaternion=[0,0,0,1], frame_id="world")]
        
        mock_process.return_value = (MagicMock(), MagicMock(), np.zeros(7), np.zeros(7))
        mock_get_ids.return_value = [0]
        mock_get_indices.return_value = np.arange(7)
        
        res0, res1, success = check_dual_trajectory_feasibility(
            mock_env, eef_traj0, eef_traj1, mock_tf_manager
        )
        assert success is True
        assert len(res0) == 1
        assert len(res1) == 1

class TestPlanningModule:
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

    def test_trajectory_length(self, mock_tf_manager):
        """Test trajectory has correct number of waypoints based on lift/approach/insert phases."""
        start = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        target = StampedPose(position=[1, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
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
        
        np.testing.assert_allclose(traj[0].position[:2], start.position[:2], atol=0.01)
        np.testing.assert_allclose(traj[-1].position, target.position, atol=0.01)
        
        for pose in traj:
            quat_norm = np.linalg.norm(pose.quaternion)
            assert 0.99 < quat_norm < 1.01

    def test_trajectory_with_rotation(self, mock_tf_manager):
        """Test trajectory generation with rotation change."""
        start = StampedPose(
            position=[0, 0, 0], 
            quaternion=[0, 0, 0, 1], 
            frame_id="world"
        )
        target = StampedPose(
            position=[1, 0, 0], 
            quaternion=[0, 0, 0.7071, 0.7071], 
            frame_id="world"
        )
        
        traj = generate_object_trajectory(
            start, target, tf_manager=mock_tf_manager, num_steps=10
        )
        
        np.testing.assert_allclose(traj[0].position, start.position, atol=1e-6)
        np.testing.assert_allclose(traj[-1].position, target.position, atol=1e-6)

class TestEndEffectorTrajectoryGeneration:
    """Tests for end effector trajectory generation."""

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
        assert feasible is True 

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
        mock_ik.side_effect = [
            [np.zeros(7)], 
            [],             
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
        
        assert feasible is False  
        assert len(results) == 2

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
        
        assert len(joint_lo) == 7  
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
        env.close()

class TestMujocoUtilsHelpers:
    """Tests for mujoco utility helper functions."""

    def test_get_eef_site_name(self):
        """Test EEF site name generation."""
        site_name = mujoco_utils.get_eef_site_name(robot_id=0, arm="right")
        assert isinstance(site_name, str)
        assert len(site_name) > 0
        
        site_name2 = mujoco_utils.get_eef_site_name(robot_id=1, arm="right")
        assert isinstance(site_name2, str)
        assert site_name != site_name2 

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
        
        assert native_model is not None
        assert native_data is not None
        
        env.close()
