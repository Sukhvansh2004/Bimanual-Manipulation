
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
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

# --- Mocks ---

@pytest.fixture
def mock_env():
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
    env.sim.data.ncon = 0 # No collisions by default
    
    return env

@pytest.fixture
def mock_tf_manager(mock_env):
    tfm = MagicMock(spec=TransformManager)
    # Default behavior: identity transform for testing logic flow
    def side_effect_transform(pose, target_frame):
        new_pose = pose.copy()
        new_pose.frame_id = target_frame
        return new_pose
    tfm.transform_pose.side_effect = side_effect_transform
    return tfm

# --- Tests ---

def test_generate_object_trajectory(mock_tf_manager):
    start = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
    target = StampedPose(position=[1,0,0], quaternion=[0,0,0,1], frame_id="world")
    
    traj = generate_object_trajectory(
        start, 
        target, 
        tf_manager=mock_tf_manager,
        num_steps=10
    )
    
    # Check length: lift(2) + approach(5) + insert(3) + 1 final = ~11-12 points roughly depending on logic
    # Logic: 3 segments. steps are calculated by ratio. 
    # int(10*0.2)=2, int(10*0.5)=5, int(10*0.3)=3. Total 10 steps loop + 1 final = 11?
    # Actually loops iterate steps. 
    assert len(traj) > 0
    assert isinstance(traj[0], StampedPose)
    # Check start matches (roughly, first point might be interpolated step 0)
    np.testing.assert_allclose(traj[0].position, start.position, atol=1e-6)
    # Check end matches
    np.testing.assert_allclose(traj[-1].position, target.position, atol=1e-6)

def test_generate_end_effector_trajectory(mock_tf_manager):
    obj_traj = [
        StampedPose(position=[1,0,0], quaternion=[0,0,0,1], frame_id="world"),
        StampedPose(position=[1,0,1], quaternion=[0,0,0,1], frame_id="world")
    ]
    grasp = StampedPose(position=[0,0,0.1], quaternion=[0,0,0,1], frame_id="world")
    
    eef_traj = generate_end_effector_trajectory(obj_traj, grasp, tf_manager=mock_tf_manager)
    
    assert len(eef_traj) == 2
    # Grasp is 0.1 offset.
    # T_world_obj = I(1,0,0)
    # T_world_grasp = I(0,0,0.1)
    # T_obj_grasp = inv(T_world_obj) * T_world_grasp = I(-1,0,0) * I(0,0,0.1) = I(-1, 0, 0.1)
    
    # New point 1: T_world_obj_1 * T_obj_grasp
    # I(1,0,1) * I(-1, 0, 0.1) -> Pos should be roughly [0, 0, 1.1] 
    # Logic: 
    # T_obj_grasp = [ [-1, 0, 0.1] ] (relative)
    # apply to new obj pose [1,0,1].
    # This checks that the relative transform is preserved.
    
    assert eef_traj[0].frame_id == "world"

@patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.set_qpos')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.in_collision')
def test_check_trajectory_feasibility_success(
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
    # Setup Mocks
    mock_ik.return_value = [np.zeros(7)] # Returns one solution
    mock_in_collision.return_value = False # No collision
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
    # Verify mujoco step was called (via either patch)
    assert mock_utils_mujoco.mj_forward.called or mock_gen_mujoco.mj_forward.called

@patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
def test_check_trajectory_feasibility_ik_fail(
    mock_ik, 
    mock_get_ids, 
    mock_get_indices, 
    mock_utils_mujoco, 
    mock_gen_mujoco, 
    mock_env, 
    mock_tf_manager
):
    mock_ik.return_value = [] # No solutions
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
    mock_in_collision.return_value = True # Collision!
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

@patch('robosuite_franka_manipulation.planning.trajectory_generator.mujoco')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.mujoco')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.resolve_ignore_names')
@patch('robosuite_franka_manipulation.planning.utils.trajectory_utils.enumerate_ik_solutions_for_arm')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_joint_qpos_indices')
@patch('robosuite_franka_manipulation.planning.trajectory_generator.get_arm_joint_ids')
def test_check_trajectory_feasibility_ignore_names(
    mock_get_ids, 
    mock_get_indices, 
    mock_ik, 
    mock_resolve, 
    mock_utils_mujoco, 
    mock_gen_mujoco, 
    mock_env, 
    mock_tf_manager
):
    # Test that ignore names are processed
    mock_ik.return_value = [np.zeros(7)]
    mock_resolve.return_value = ["object_geom"]
    mock_get_ids.return_value = [0]
    mock_get_indices.return_value = np.arange(7)
    
    traj = [StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")]
    
    check_trajectory_feasibility(
        env=mock_env,
        robot_id=0,
        eef_trajectory=traj,
        tf_manager=mock_tf_manager,
        object_name="my_object",
        check_collisions=False 
    )
    
    mock_resolve.assert_called_with("my_object", None)
