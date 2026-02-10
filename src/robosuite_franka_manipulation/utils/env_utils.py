import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils.suction_weld import apply_suction_weld

logger = logging.getLogger(__name__)


# =============================================================================
# YCB Dataset Utilities
# =============================================================================

def get_available_ycb_objects() -> List[str]:
    """
    Get a list of all available YCB objects.
    
    Returns:
        List of YCB object names (e.g., ['apple', 'banana', 'mug', ...])
    """
    if not constants.YCB_DATASET_PATH.exists():
        logger.warning(f"YCB dataset not found at {constants.YCB_DATASET_PATH}")
        return []
    
    objects = []
    # Search for XML files in the YCB directory
    for xml_file in constants.YCB_DATASET_PATH.glob("*.xml"):
        objects.append(xml_file.stem)
    return sorted(objects)


def list_ycb_objects() -> None:
    """Print all available YCB objects to console."""
    objects = get_available_ycb_objects()
    print(f"\nAvailable YCB Objects ({len(objects)} total):")
    print("-" * 40)
    for i, obj in enumerate(objects):
        print(f"  {i+1:3d}. {obj}")
    print("-" * 40)


def resolve_mesh_path(object_name: str) -> Optional[str]:
    """
    Resolve the mesh path for a given object name.
    
    Args:
        object_name: Name of the object (e.g., 'apple', 'apple_0')
        
    Returns:
        Path to the mesh file if found, None otherwise.
    """
    candidates = [object_name]
    if "_" in object_name:
        candidates.append(object_name.rsplit('_', 1)[0]) # Remove suffix (e.g. apple_0 -> apple)
    
    for name in candidates:
        ycb_obj_dir = constants.YCB_DATASET_PATH / name
        if ycb_obj_dir.exists():
            # Check for textured.obj (standard YCB)
            mesh_path = ycb_obj_dir / "textured.obj"
            if mesh_path.exists():
                return str(mesh_path)
                    
    return None


# =============================================================================
# Object Specification Helpers
# =============================================================================

def create_ycb_object(
    ycb_name: str,
    instance_name: Optional[str] = None,
    position: Optional[List[float]] = None,
    quaternion: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Create a YCB object specification for the environment.
    
    Args:
        ycb_name: Name of the YCB object (e.g., 'apple', 'mug', 'banana')
        instance_name: Optional unique instance name (defaults to ycb_name)
        position: Optional [x, y, z] position for the object (default: on table)
        quaternion: Optional [x, y, z, w] quaternion for orientation (default: identity)
    
    Returns:
        Dictionary with object specification
    """
    if instance_name is None:
        instance_name = f"{ycb_name}_0"
    
    return {
        'type': 'ycb',
        'ycb_name': ycb_name,
        'instance_name': instance_name,
        'name': instance_name,
        'pos': position if position else constants.DEFAULT_OBJECT_POSITION.copy(),
        'quat': quaternion if quaternion else constants.DEFAULT_OBJECT_QUATERNION.copy(),
    }


def create_primitive_object(
    obj_type: str,
    name: str,
    size: List[float],
    position: Optional[List[float]] = None,
    quat: Optional[List[float]] = None,
    rgba: Optional[List[float]] = None,
    density: float = 500.0,
) -> Dict[str, Any]:
    """
    Create a primitive object specification for the environment.
    
    Args:
        obj_type: Type of primitive ('box', 'cylinder', 'ball', 'capsule')
        name: Unique name for the object
        size: Size parameters (varies by type)
        position: Optional [x, y, z] position for the object (default: on table)
        quat: Optional [x, y, z, w] quaternion for orientation (default: identity)
        rgba: Optional [r, g, b, a] color values
        density: Object density in kg/m³
    
    Returns:
        Dictionary with object specification
    """
    return {
        'type': obj_type,
        'name': name,
        'size': size,
        'pos': position if position else constants.DEFAULT_OBJECT_POSITION.copy(),
        'quat': quat if quat else [0.0, 0.0, 0.0, 1.0],
        'rgba': rgba if rgba else [0.5, 0.5, 0.5, 1.0],
        'density': density,
    }


# =============================================================================
# Environment Step Utilities
# =============================================================================

def env_step(env: DualArmEnvironment, action: Any = None, robot_id: Optional[int] = None, arm: Union[str, List[str], None] = None):
    """Handles full action construction and gripper state persistence.
    
    Args:
        env: The robosuite environment (MjSim)
        action: The action vector. Can be:
                - Full environment action vector (size env.action_dim)
                - Single robot action vector (size robot.action_dim), if robot_id is provided
        robot_id: ID of the robot currently being controlled. Required if passing single robot action.
        arm: Name(s) of the arm(s) currently being controlled. Required if robot_id is provided.
             Can be a single string (e.g. "right") or a list of strings (e.g. ["right", "left"]).
    """
    
    if robot_id is not None and arm is None:
        raise ValueError("The 'arm' argument must be specified when 'robot_id' is provided.")

    # Initialize gripper state storage if not present
    if not hasattr(env_step, "previous_gripper_actions"):
        env_step.previous_gripper_actions = {}

    full_action = np.zeros(env.action_dim)
    
    if action is not None:
        input_action = np.array(action, copy=True)
        
        # Determine if input is full action or partial robot action
        is_full_action = (len(input_action) == env.action_dim)
        
        if is_full_action:
            full_action = input_action
        elif robot_id is not None:
            # Construct full action from single robot action
            current_idx = 0
            robot_start_idx = 0
            robot_end_idx = 0
            
            for i, robot in enumerate(env.robots):
                robot_dim = robot.action_dim
                if i == robot_id:
                    robot_start_idx = current_idx
                    robot_end_idx = current_idx + robot_dim
                    break
                current_idx += robot_dim
                
            if len(input_action) != (robot_end_idx - robot_start_idx):
                 raise ValueError(f"Action dimension {len(input_action)} does not match expected dimension {robot_end_idx - robot_start_idx} for robot {robot_id}")
                 
            full_action[robot_start_idx:robot_end_idx] = input_action
        else:
            raise ValueError("Provided partial action without robot_id, or action dimension mismatch.")

    # Update/Restore Gripper States
    current_idx = 0
    for i, robot in enumerate(env.robots):
        robot_dim = robot.action_dim
        
        # Determine gripper layout
        # We assume gripper actions are at the end of the robot action vector,
        # ordered by robot.arms
        
        gripper_dofs = []
        gripper_arms = []
        
        # Iterate in order of robot.arms to match action vector structure
        for arm_name in robot.arms:
            if arm_name in robot.gripper:
                gripper_dofs.append(robot.gripper[arm_name].dof)
                gripper_arms.append(arm_name)

        total_gripper_dim = sum(gripper_dofs)
        
        # Start index of the gripper block for this robot in the FULL action vector
        # The gripper block is at the end of this robot's segment
        robot_gripper_block_start = current_idx + robot_dim - total_gripper_dim
        
        current_gripper_offset = 0
        for g_idx, g_dof in enumerate(gripper_dofs):
            g_arm_name = gripper_arms[g_idx]
            
            # Indices for this specific gripper in full_action
            g_start = robot_gripper_block_start + current_gripper_offset
            g_end = g_start + g_dof
            
            store_key = f"robot_{i}_{g_arm_name}"
            
            # Check if we should update or restore
            should_update = False
            
            if action is not None:
                if robot_id is not None:
                    if i == robot_id:
                        # Check if this arm is in the 'arm' argument
                        if isinstance(arm, list):
                            if g_arm_name in arm:
                                should_update = True
                        elif arm == g_arm_name:
                            should_update = True
                else:
                    # Full action passed without robot_id -> update everything
                    should_update = True
            
            if should_update:
                # Update storage from full_action (which has the input)
                val = full_action[g_start:g_end]
                env_step.previous_gripper_actions[store_key] = val
            else:
                # Restore from storage
                if store_key in env_step.previous_gripper_actions:
                    full_action[g_start:g_end] = env_step.previous_gripper_actions[store_key]
                else:
                    # Init storage
                    val = full_action[g_start:g_end]
                    env_step.previous_gripper_actions[store_key] = val
            
            current_gripper_offset += g_dof

        current_idx += robot_dim

    obs = env.step(full_action)
    apply_suction_weld(env, full_action)
    return obs


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