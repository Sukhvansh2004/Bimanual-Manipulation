import time
import logging
import os
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
        bottom_site = self.worldbody.find("./body/site[@name='{}bottom_site']".format(self.naming_prefix))
        if bottom_site is not None:
            return np.array([0, 0, float(bottom_site.get("pos", "0 0 0").split()[2])])
        # Default: assume object center is at origin
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


def generate_object_xml(obj_spec: Dict[str, Any]) -> str:
    """
    Generate MuJoCo XML snippet for a primitive object.
    
    Args:
        obj_spec: Object specification dictionary
        
    Returns:
        XML string for the object
    """
    obj_type = obj_spec.get('type', 'box')
    name = obj_spec.get('name', 'custom_object')
    size = obj_spec.get('size', [0.05, 0.05, 0.05])
    pos = obj_spec.get('pos', [0.0, 0.0, 0.85])
    rgba = obj_spec.get('rgba', [0.5, 0.5, 0.5, 1.0])
    density = obj_spec.get('density', 500.0)
    
    pos_str = ' '.join(map(str, pos))
    rgba_str = ' '.join(map(str, rgba))
    
    if obj_type == 'box':
        size_str = ' '.join(map(str, [s/2 for s in size[:3]]))  # MuJoCo uses half-sizes
        geom_type = 'box'
    elif obj_type == 'cylinder':
        size_str = ' '.join(map(str, size[:2]))  # radius, half-height
        geom_type = 'cylinder'
    elif obj_type == 'ball' or obj_type == 'sphere':
        size_str = str(size[0])  # radius
        geom_type = 'sphere'
    elif obj_type == 'capsule':
        size_str = ' '.join(map(str, size[:2]))  # radius, half-length
        geom_type = 'capsule'
    else:
        logger.warning(f"Unknown object type: {obj_type}, defaulting to box")
        size_str = ' '.join(map(str, [s/2 for s in size[:3]]))
        geom_type = 'box'
    
    xml = f'''
    <body name="{name}" pos="{pos_str}">
        <freejoint name="{name}_joint"/>
        <geom name="{name}_geom" type="{geom_type}" size="{size_str}" 
              rgba="{rgba_str}" density="{density}" 
              contype="1" conaffinity="1"
              friction="1 0.005 0.0001" condim="4"/>
        <site name="{name}_site" pos="0 0 0" size="0.002" rgba="1 0 0 0"/>
    </body>
    '''
    return xml


# =============================================================================
# Environment Step Utilities
# =============================================================================

def safe_env_step(env: Any, action: Any, retries: int = 3, delay: float = 0.01):
    """Safely call ``env.step(action)`` and retry on intermittent
    RuntimeError: "dictionary changed size during iteration" which can
    occasionally arise inside robosuite's internal robot control logic.

    This keeps the retry/mitigation inside our codebase (no external
    patching of robosuite). The function makes a defensive copy of the
    action before passing it to the environment.
    """
    action_copy = None
    try:
        # Defensive copy for numpy arrays and array-like objects
        import numpy as _np  # local import to avoid top-level dependency surprises
        if isinstance(action, _np.ndarray):
            action_copy = _np.array(action, copy=True)
        else:
            # Fallback: try to create a numpy copy, otherwise use the original
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
                    msg,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay * (attempt + 1))
                continue
            # Re-raise non-matching runtime errors
            raise
    # Final attempt (let it raise if it still fails)
    if last_exception is not None:
        logger.error("Final attempt to call env.step still failed: %s", last_exception)
    return env.step(action_copy)
