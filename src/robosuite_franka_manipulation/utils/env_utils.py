import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from robosuite.utils.binding_utils import MjSim
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

def safe_env_step(env: MjSim, action: Any, retries: int = 3, delay: float = 0.01):
    """Safely call ``env.step(action)`` and retry on intermittent
    RuntimeError: "dictionary changed size during iteration"."""
    
    action_copy = None
    try:
        import numpy as _np
        if isinstance(action, _np.ndarray):
            action_copy = _np.array(action, copy=True)
        else:
            try:
                action_copy = _np.array(action, copy=True)
            except Exception:
                action_copy = action
    except Exception:
        action_copy = action

    last_exception = None
    for attempt in range(retries):
        try:
            obs = env.step(action_copy)
            apply_suction_weld(env, action_copy)
            return obs
        except RuntimeError as e:
            last_exception = e
            msg = str(e)
            if "dictionary changed size during iteration" in msg:
                logger.warning(
                    "Detected intermittent RuntimeError in env.step(): '%s' (attempt %d/%d). Retrying...",
                    msg, attempt + 1, retries,
                )
                time.sleep(delay * (attempt + 1))
                continue
            raise
    
    if last_exception is not None:
        logger.error("Final attempt to call env.step still failed: %s", last_exception)
    obs = env.step(action_copy)
    apply_suction_weld(env, action_copy)
    return obs