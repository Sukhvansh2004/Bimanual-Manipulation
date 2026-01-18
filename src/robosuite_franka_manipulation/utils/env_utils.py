import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
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


# =============================================================================
# Object Specification Helpers
# =============================================================================

def create_ycb_object(
    ycb_name: str,
    instance_name: Optional[str] = None,
    position: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Create a YCB object specification for the environment.
    
    Args:
        ycb_name: Name of the YCB object (e.g., 'apple', 'mug', 'banana')
        instance_name: Optional unique instance name (defaults to ycb_name)
        position: Optional [x, y, z] position for the object (default: on table)
    
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
    }


def create_primitive_object(
    obj_type: str,
    name: str,
    size: List[float],
    position: Optional[List[float]] = None,
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
        'rgba': rgba if rgba else [0.5, 0.5, 0.5, 1.0],
        'density': density,
    }


# =============================================================================
# Environment Step Utilities
# =============================================================================

def env_step(env: DualArmEnvironment, action: Any, robot_id: Optional[int] = None, arm: Optional[str] = None):
    """Handles full action construction and gripper state persistence.
    
    Args:
        env: The robosuite environment (MjSim)
        action: The action vector. Can be:
                - Full environment action vector (size env.action_dim)
                - Single robot action vector (size robot.action_dim), if robot_id is provided
        retries: Number of retries for RuntimeError
        delay: Delay between retries
        robot_id: ID of the robot currently being controlled. Required if passing single robot action.
        arm: Name of the arm currently being controlled (unused in logic but kept for interface consistency)
    """
    
    # Initialize gripper state storage if not present
    if not hasattr(env_step, "previous_gripper_actions"):
        env_step.previous_gripper_actions = {}

    input_action = np.array(action, copy=True)
    full_action = np.zeros(env.action_dim)
    
    # Determine if input is full action or partial robot action
    is_full_action = (len(input_action) == env.action_dim)
    
    if is_full_action:
        full_action = input_action
    elif robot_id is not None:
        # Construct full action from single robot action
        # We need to find where this robot's action fits
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
            
        # Verify sizes match
        if len(input_action) != (robot_end_idx - robot_start_idx):
             raise ValueError(f"Action dimension {len(input_action)} does not match expected dimension {robot_end_idx - robot_start_idx} for robot {robot_id}")
             
        full_action[robot_start_idx:robot_end_idx] = input_action
    else:
        # Not full action and no robot_id?
        # Maybe it's a single robot env and action matches?
        if len(input_action) == env.action_dim:
             full_action = input_action
        else:
             raise ValueError("Provided partial action without robot_id, or action dimension mismatch.")

    # Update/Restore Gripper States
    # We iterate over robots to identify their segments in the full_action vector
    if hasattr(env, "robots"):
        current_idx = 0
        for i, robot in enumerate(env.robots):
            robot_dim = robot.action_dim
            
            # Determine gripper dimension
            # robot.gripper is a dict, we need to access the first gripper object
            gripper_dim = list(robot.gripper.values())[0].dof
            
            # Calculate indices for this robot's gripper in the FULL action vector
            gripper_start_idx = current_idx + robot_dim - gripper_dim
            gripper_end_idx = current_idx + robot_dim
            
            # Key for storage
            store_key = f"robot_{i}"
            
            # Check if this robot was the one actively controlled
            # If input was full action, we assume active control could be anywhere, 
            # BUT if robot_id is provided, we treat that as the "active" one for updating state.
            # If is_full_action is True and robot_id is None, we might overwrite all states?
            # Or maybe we should update ALL states from full action?
            # Let's say: 
            # 1. If robot_id is provided, update that robot's state, restore others.
            # 2. If robot_id is None (full action passed directly), update ALL states from input.
            
            if robot_id is not None:
                if i == robot_id:
                    # Update storage from the constructed full_action (which has the new input)
                    val = full_action[gripper_start_idx:gripper_end_idx]
                    env_step.previous_gripper_actions[store_key] = val
                else:
                    # Restore from storage if available
                    if store_key in env_step.previous_gripper_actions:
                         full_action[gripper_start_idx:gripper_end_idx] = env_step.previous_gripper_actions[store_key]
                    else:
                         # Init storage from the zeros (or whatever is in full_action currently)
                         # Since full_action was init with zeros (if partial), this defaults to 0.
                         val = full_action[gripper_start_idx:gripper_end_idx]
                         env_step.previous_gripper_actions[store_key] = val
            else:
                # robot_id is None, assume full control (e.g. from existing code that sends full action)
                # We save all states for next time
                val = full_action[gripper_start_idx:gripper_end_idx]
                env_step.previous_gripper_actions[store_key] = val

            current_idx += robot_dim

    obs = env.step(full_action)
    apply_suction_weld(env, full_action)
    return obs