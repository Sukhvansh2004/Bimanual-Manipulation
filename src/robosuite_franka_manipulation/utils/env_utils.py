import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from robosuite.models.objects import MujocoXMLObject
from robosuite_franka_manipulation import constants

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


class YCBObject(MujocoXMLObject):
    """
    Custom MuJoCo object class for loading YCB dataset objects.
    
    This class wraps YCB dataset XML files to be compatible with robosuite's
    object placement system.
    """
    
    def __init__(
        self,
        name: str,
        ycb_object_name: str,
        joints: Optional[List[dict]] = None,
        obj_type: str = "all",
        duplicate_collision_geoms: bool = True,
    ):
        """
        Initialize a YCB object.
        
        Args:
            name: Unique name for this object instance
            ycb_object_name: Name of the YCB object (e.g., 'apple', 'mug')
            joints: List of joint specifications for the object
            obj_type: Type of object ('collision', 'visual', or 'all')
            duplicate_collision_geoms: Whether to duplicate collision geometries
        """
        self.ycb_object_name = ycb_object_name
        
        # Find the XML file for this YCB object
        xml_path = constants.YCB_DATASET_PATH / f"{ycb_object_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"YCB object XML not found: {xml_path}")
        
        # Default joint configuration (free joint for manipulation)
        if joints is None:
            # "default" joint in robosuite automatically adds a freejoint
            # However, for MujocoXMLObject we often pass explicit dicts if needed.
            # Passing "default" to the parent constructor is also supported by some classes,
            # but providing the dict here is safer for XML objects.
            joints = [{"type": "free", "damping": "0.0005"}]
        
        super().__init__(
            str(xml_path),
            name=name,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
    
    @property
    def bottom_offset(self):
        """Return the bottom offset of the object for placement."""
        # Get the bounding box from geoms and compute bottom offset
        bottom_site = self.worldbody.find(f"./body/site[@name='{self.naming_prefix}bottom_site']")
        if bottom_site is not None:
            return np.array([0, 0, float(bottom_site.get("pos", "0 0 0").split()[2])])
        # Default: assume object center is at origin, offset slightly
        return np.array([0, 0, -0.02])
    
    @property
    def top_offset(self):
        """Return the top offset of the object."""
        return np.array([0, 0, 0.05])
    
    @property
    def horizontal_radius(self):
        """Return the horizontal radius for placement."""
        return 0.05


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
        instance_name = ycb_name
    
    return {
        'type': 'ycb',
        'ycb_name': ycb_name,
        'name': instance_name,
        'pos': position if position else constants.DEFAULT_TABLE_CENTER.copy(),
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
        'pos': position if position else constants.DEFAULT_TABLE_CENTER.copy(),
        'rgba': rgba if rgba else [0.5, 0.5, 0.5, 1.0],
        'density': density,
    }


# =============================================================================
# Environment Step Utilities
# =============================================================================

def safe_env_step(env: Any, action: Any, retries: int = 3, delay: float = 0.01):
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
            return env.step(action_copy)
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
    return env.step(action_copy)