"""Environment stepping and action persistence utilities.

Provides :func:`env_step` (the primary simulation-step function),
:class:`ActionPersistenceManager`, and :func:`set_robot_config`.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.utils.monitors import _collision_monitor, _object_monitor
from bimanual_manipulation.utils.real_robot_bridge import (
    get_real_robot_bridge,
    is_real_robot_env,
    reset_real_robot_bridge,
)
from bimanual_manipulation.utils.suction_weld import apply_suction_weld

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Step Utilities
# =============================================================================

class ActionPersistenceManager:
    """
    Manages action persistence for different controller types.
    
    For delta controllers: No arm action persistence needed (zero = no movement)
    For absolute controllers: Persist arm actions to prevent abrupt movements on zero input
    
    Follows the same pattern as gripper action persistence but for arm actions.
    """
    
    def __init__(self):
        self.previous_gripper_actions: Dict[str, np.ndarray] = {}
        self.previous_arm_actions: Dict[str, np.ndarray] = {}
    
    def get_controller_info(self, robot) -> tuple:
        """
        Extract controller name and input type from robot's part_controllers.
        
        Args:
            robot: The robot instance
            
        Returns:
            (controller_name, input_type): Tuple of controller configuration
        """
        controller_name = constants.DEFAULT_CONTROLLER_NAME
        input_type = constants.DEFAULT_INPUT_TYPE
        
        try:
            # Get part controllers - they're usually nested in composite_controller
            if hasattr(robot, 'part_controllers') and robot.part_controllers:
                for arm_name, controller in robot.part_controllers.items():
                    if hasattr(controller, 'name'):
                        controller_name = controller.name
                    if hasattr(controller, 'input_type'):
                        input_type = controller.input_type
                    break  # Assume same controller for all arms
        except Exception as e:
            logger.debug(f"Could not extract controller info: {e}")
        
        return controller_name, input_type
    
    def is_absolute_mode(self, robot) -> bool:
        """Check if robot is using absolute input mode."""
        _, input_type = self.get_controller_info(robot)
        return input_type == constants.InputType.ABSOLUTE
    
    def get_arm_action_dim(self, robot, arm_name: str) -> int:
        """
        Get the action dimension for arm control (excluding gripper).
        
        Args:
            robot: The robot instance
            arm_name: Name of the arm
            
        Returns:
            Action dimension for arm control
        """
        controller_name, _ = self.get_controller_info(robot)
        return constants.CONTROLLER_ACTION_DIMS.get(
            controller_name, 
            constants.CONTROLLER_ACTION_DIMS[constants.ControllerName.OSC_POSE]
        )
    
    def read_current_arm_state(self, env: DualArmEnvironment, robot_id: int, arm_name: str) -> np.ndarray:
        """
        Read current arm state from environment for absolute mode persistence.
        
        For OSC_POSE: Returns [x, y, z, rx, ry, rz] in robot base frame
        For JOINT_POSITION: Returns joint positions
        
        Args:
            env: The robosuite environment
            robot_id: ID of the robot
            arm_name: Name of the arm
            
        Returns:
            Current arm state as action format
        """
        robot = env.robots[robot_id]
        controller_name, _ = self.get_controller_info(robot)
        
        if controller_name == constants.ControllerName.JOINT_POSITION:
            # Get joint positions
            return self._read_joint_positions(env, robot, arm_name)
        else:
            # Default to OSC_POSE - get EEF pose
            return self._read_eef_pose(env, robot, arm_name)
    
    def _read_joint_positions(self, env: DualArmEnvironment, robot, arm_name: str) -> np.ndarray:
        """Read current joint positions for an arm."""
        try:
            # Get joint indices for this arm
            if hasattr(robot, f'{arm_name}_arm_joint_indexes'):
                joint_indices = getattr(robot, f'{arm_name}_arm_joint_indexes')
            elif hasattr(robot, 'arm_joint_indexes'):
                joint_indices = robot.arm_joint_indexes
            else:
                joint_indices = robot.joint_indexes
            
            return env.sim.data.qpos[joint_indices].copy()
        except Exception as e:
            logger.warning(f"Could not read joint positions: {e}")
            return constants.FRANKA_HOME_JOINT_POSITION.copy()
    
    def _read_eef_pose(self, env: DualArmEnvironment, robot, arm_name: str) -> np.ndarray:
        """Read current EEF pose as [x, y, z, rx, ry, rz] in base frame."""
        try:
            
            # Get EEF pose in world frame from robot internals
            eef_site_name = robot.robot_model.eef_name.get(arm_name, robot.robot_model.eef_name.get('right'))
            eef_site_id = env.sim.model.site_name2id(eef_site_name)
            eef_pos_world = env.sim.data.site_xpos[eef_site_id].copy()
            eef_rot_mat_world = env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
            
            # Get robot base pose
            base_body_name = robot.robot_model.root_body
            base_body_id = env.sim.model.body_name2id(base_body_name)
            base_pos_world = env.sim.data.body_xpos[base_body_id].copy()
            base_rot_mat_world = env.sim.data.body_xmat[base_body_id].reshape(3, 3)
            
            # Transform EEF to base frame: p_base = R_base^T @ (p_world - base_pos)
            eef_pos_base = base_rot_mat_world.T @ (eef_pos_world - base_pos_world)
            eef_rot_mat_base = base_rot_mat_world.T @ eef_rot_mat_world
            
            # Convert rotation matrix to axis-angle (rotation vector)
            rot_vec = R.from_matrix(eef_rot_mat_base).as_rotvec()
            
            return np.concatenate([eef_pos_base, rot_vec])
        except Exception as e:
            logger.warning(f"Could not read EEF pose: {e}")
            # Return neutral pose
            return np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
    
    def is_zero_action(self, action: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Check if an action is effectively zero."""
        return np.allclose(action, 0.0, atol=tolerance)


# Global action persistence manager
_action_manager = ActionPersistenceManager()


def env_step(env: DualArmEnvironment, action: Any = None, robot_id: Optional[int] = None, arm: Union[str, List[str], None] = None):
    """Handles full action construction with gripper and arm action persistence.
    
    This function manages action persistence for different controller modes:
    - For delta controllers: No arm persistence needed (zero = no movement)
    - For absolute controllers: Persists arm actions to prevent abrupt movements on zero input
    - For grippers: Always persists state (both delta and absolute modes)
    
    Args:
        env: The robosuite environment (MjSim)
        action: The action vector. Can be:
                - Full environment action vector (size env.action_dim)
                - Single robot action vector (size robot.action_dim), if robot_id is provided
        robot_id: ID of the robot currently being controlled. Required if passing single robot action.
        arm: Name(s) of the arm(s) currently being controlled. Required if robot_id is provided.
             Can be a single string (e.g. "right") or a list of strings (e.g. ["right", "left"]).
    """
    global _action_manager
    
    if robot_id is not None and arm is None:
        raise ValueError("The 'arm' argument must be specified when 'robot_id' is provided.")

    full_action = np.zeros(env.action_dim)
    
    # Phase 1: Place input action into full_action vector
    if action is not None:
        input_action = np.array(action, copy=True)
        is_full_action = (len(input_action) == env.action_dim)
        
        if is_full_action:
            full_action = input_action
        elif robot_id is not None:
            # Construct full action from single robot action
            current_idx = 0
            for i, robot in enumerate(env.robots):
                robot_dim = robot.action_dim
                if i == robot_id:
                    if len(input_action) != robot_dim:
                        raise ValueError(f"Action dimension {len(input_action)} does not match expected dimension {robot_dim} for robot {robot_id}")
                    full_action[current_idx:current_idx + robot_dim] = input_action
                    break
                current_idx += robot_dim
        else:
            raise ValueError("Provided partial action without robot_id, or action dimension mismatch.")

    # Phase 2: Handle arm action persistence for absolute mode controllers
    # Phase 3: Handle gripper action persistence (always)
    current_idx = 0
    for i, robot in enumerate(env.robots):
        robot_dim = robot.action_dim
        is_absolute = _action_manager.is_absolute_mode(robot)
        
        # Determine gripper layout
        gripper_dofs = []
        gripper_arms = []
        for arm_name in robot.arms:
            if arm_name in robot.gripper:
                gripper_dofs.append(robot.gripper[arm_name].dof)
                gripper_arms.append(arm_name)
        
        total_gripper_dim = sum(gripper_dofs)
        arm_action_dim = robot_dim - total_gripper_dim
        
        # Arm action indices in full_action
        arm_start = current_idx
        arm_end = current_idx + arm_action_dim
        
        # Gripper block starts after arm actions
        gripper_block_start = arm_end
        
        # Process arm actions for each arm of this robot
        for arm_name in robot.arms:
            arm_store_key = f"robot_{i}_{arm_name}_arm"
            
            # Determine if this arm should be updated or restored
            should_update_arm = False
            if action is not None:
                if robot_id is not None:
                    if i == robot_id:
                        if isinstance(arm, list):
                            should_update_arm = arm_name in arm
                        else:
                            should_update_arm = (arm == arm_name)
                else:
                    # Full action passed without robot_id -> update everything
                    should_update_arm = True
            
            if is_absolute:
                # Absolute mode: handle arm action persistence
                arm_action = full_action[arm_start:arm_end]
                
                if should_update_arm:
                    # Check if incoming action is zero
                    if _action_manager.is_zero_action(arm_action):
                        # Use stored action or read current state
                        if arm_store_key in _action_manager.previous_arm_actions:
                            full_action[arm_start:arm_end] = _action_manager.previous_arm_actions[arm_store_key]
                        else:
                            # Read current state from environment
                            current_state = _action_manager.read_current_arm_state(env, i, arm_name)
                            full_action[arm_start:arm_end] = current_state
                            _action_manager.previous_arm_actions[arm_store_key] = current_state
                    else:
                        # Non-zero action: update storage
                        _action_manager.previous_arm_actions[arm_store_key] = arm_action.copy()
                else:
                    # Restore from storage for non-controlled arms
                    if arm_store_key in _action_manager.previous_arm_actions:
                        full_action[arm_start:arm_end] = _action_manager.previous_arm_actions[arm_store_key]
                    else:
                        # Initialize with current state
                        current_state = _action_manager.read_current_arm_state(env, i, arm_name)
                        full_action[arm_start:arm_end] = current_state
                        _action_manager.previous_arm_actions[arm_store_key] = current_state
        
        # Process gripper actions
        current_gripper_offset = 0
        for g_idx, g_dof in enumerate(gripper_dofs):
            g_arm_name = gripper_arms[g_idx]
            g_start = gripper_block_start + current_gripper_offset
            g_end = g_start + g_dof
            
            gripper_store_key = f"robot_{i}_{g_arm_name}_gripper"
            
            # Determine if this gripper should be updated or restored
            should_update_gripper = False
            if action is not None:
                if robot_id is not None:
                    if i == robot_id:
                        if isinstance(arm, list):
                            should_update_gripper = g_arm_name in arm
                        else:
                            should_update_gripper = (arm == g_arm_name)
                else:
                    should_update_gripper = True
            
            if should_update_gripper:
                # Update storage from input
                _action_manager.previous_gripper_actions[gripper_store_key] = full_action[g_start:g_end].copy()
            else:
                # Restore from storage
                if gripper_store_key in _action_manager.previous_gripper_actions:
                    full_action[g_start:g_end] = _action_manager.previous_gripper_actions[gripper_store_key]
                else:
                    # Initialize storage
                    _action_manager.previous_gripper_actions[gripper_store_key] = full_action[g_start:g_end].copy()
            
            current_gripper_offset += g_dof

        current_idx += robot_dim

    obs = env.step(full_action)
    apply_suction_weld(env, full_action)

    # Mirror the executed action onto the real hardware. Gated two ways: opt-in via
    # BIMANUAL_REAL_ROBOT (else get_real_robot_bridge() is None -> no-op for all the offline
    # planning/visualization callers), AND per-env -- only the env explicitly designated with
    # mark_real_robot_env() mirrors, so the pipeline's separate planning_env never does.
    real_bridge = get_real_robot_bridge()
    if real_bridge is not None and is_real_robot_env(env):
        real_bridge.mirror(env, full_action)

    # Collision monitoring (no-op when monitor is disabled)
    _collision_monitor.check(env)

    # Object trajectory monitoring (no-op when monitor is disabled)
    _object_monitor.check(env)

    return obs


def reset_action_persistence():
    """Reset the action persistence manager state. Call after env.reset()."""
    global _action_manager
    _action_manager = ActionPersistenceManager()
    # After a reset the sim teleports to its start config; force the real arms to ease in to
    # the new configuration on the next mirrored command instead of jumping.
    reset_real_robot_bridge()


def get_action_persistence_manager() -> ActionPersistenceManager:
    """Get the global action persistence manager for inspection/debugging."""
    return _action_manager


def set_robot_config(env: DualArmEnvironment, robot_id: int, arm_id: str, joint_config: Union[np.ndarray, List[float]]):
    """
    Sets the robot to a specific joint configuration in MuJoCo.
    
    Args:
        env: The robosuite environment.
        robot_id: ID of the robot (index in env.robots).
        arm_id: ID of the arm ('right' or 'left').
        joint_config: Joint configuration (list or numpy array).
    """
    if robot_id >= len(env.robots):
        raise ValueError(f"Robot ID {robot_id} out of range (total robots: {len(env.robots)})")
    
    robot = env.robots[robot_id]
    joint_config = np.array(joint_config)
    
    # Determine which joints to set
    if isinstance(robot.robot_model.arm_joints, dict):
        # Two-arm robot (e.g. Baxter)
        if arm_id not in robot.robot_model.arm_joints:
            raise ValueError(f"Arm '{arm_id}' not found in robot {robot_id}. Available arms: {list(robot.robot_model.arm_joints.keys())}")
        
        # Get indices for this specific arm
        attr = f"{arm_id}_arm_joint_indexes"
        if hasattr(robot, attr):
            joint_indices = getattr(robot, attr)
        else:
            # Fallback: compute from joint names
            joint_names = robot.robot_model.arm_joints[arm_id]
            joint_indices = [env.sim.model.get_joint_qpos_addr(x) for x in joint_names]
    else:
        # Single-arm robot (e.g. Panda)
        if arm_id not in robot.arms:
            logger.warning(f"Arm '{arm_id}' requested for single-arm robot {robot_id} which has arms {robot.arms}. Using default arm.")
        
        # For single arm robots in robosuite, arm_joint_indexes is usually what we want
        if hasattr(robot, "arm_joint_indexes"):
            joint_indices = robot.arm_joint_indexes
        else:
            joint_indices = robot.joint_indexes
            
    if len(joint_config) != len(joint_indices):
        raise ValueError(f"Joint configuration length {len(joint_config)} does not match robot arm DOF {len(joint_indices)}")
    
    # Set the positions in MuJoCo
    env.sim.data.qpos[joint_indices] = joint_config
    
    # Update the simulation state to reflect changes
    env.sim.forward()
    
    logger.debug(f"Set robot {robot_id} arm {arm_id} to config: {joint_config}")
