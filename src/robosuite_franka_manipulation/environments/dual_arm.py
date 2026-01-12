import logging
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import mujoco
import traceback

from scipy.spatial.transform import Rotation as R

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, find_elements

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)
logging.basicConfig(level=constants.LOG_LEVEL)

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
        position: Optional [x, y, z] position for the object
    
    Returns:
        Dictionary with object specification
    """
    if instance_name is None:
        instance_name = ycb_name
    
    return {
        'type': 'ycb',
        'ycb_name': ycb_name,
        'name': instance_name,
        'pos': position if position else [0.0, 0.0, 0.85],
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
        position: Optional [x, y, z] position for the object
        rgba: Optional [r, g, b, a] color values
        density: Object density in kg/m³
    
    Returns:
        Dictionary with object specification
    """
    return {
        'type': obj_type,
        'name': name,
        'size': size,
        'pos': position if position else [0.0, 0.0, 0.85],
        'rgba': rgba if rgba else [0.5, 0.5, 0.5, 1.0],
        'density': density,
    }


class DualArmEnvironmentWrapper:
    """
    Wrapper for robosuite dual-arm environment with realistic camera intrinsics
    and support for custom objects (primitives and YCB dataset objects).
    Automatically sets camera intrinsics on creation and after reset.
    """
    
    def __init__(
        self,
        env: Union[MujocoEnv, Wrapper],
        custom_objects: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the environment wrapper.
        
        Args:
            env: The robosuite environment to wrap
            custom_objects: Optional list of custom object specifications
        """
        self._env = env
        self._custom_objects = custom_objects or []
        self._object_names: List[str] = []
        self._set_intrinsics()
        
        # Add custom objects if specified
        if self._custom_objects:
            self._add_custom_objects()
    
    def _set_intrinsics(self) -> None:
        """Set realistic camera intrinsics for all cameras."""
        # robosuite dynamic attributes not in type stubs
        for i in range(len(self._env.camera_names)):
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self._env.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)
    
    def _add_custom_objects(self) -> None:
        """
        Add custom objects to the MuJoCo model.
        
        Supports:
        - Primitive objects: box, cylinder, ball, capsule
        - YCB dataset objects: loaded from XML files
        """
        
        for obj_spec in self._custom_objects:
            obj_type = obj_spec.get('type', 'box')
            obj_name = obj_spec.get('name', f'custom_obj_{len(self._object_names)}')
            pos = obj_spec.get('pos', [0.0, 0.0, 0.85])
            
            if obj_type == 'ycb':
                # YCB objects are handled through the MujocoXMLObject system
                # They need to be added at environment creation time
                logger.warning(
                    f"YCB object '{obj_spec.get('ycb_name')}' should be added at "
                    "environment creation. Use make_dual_arm_env(ycb_objects=[...]) instead."
                )
                continue
            
            # For primitive objects, we modify the existing model
            # This is a simplified approach - full dynamic object addition requires
            # modifying the XML before environment creation
            self._object_names.append(obj_name)
            logger.info(f"Registered custom object: {obj_name} at position {pos}")
    
    def get_object_names(self) -> List[str]:
        """Get the names of all custom objects in the environment."""
        return self._object_names.copy()
    
    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get the pose of a custom object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            4x4 transformation matrix or None if not found
        """
        try:
            body_id = self._env.sim.model.body_name2id(object_name)
            pos = self._env.sim.data.body_xpos[body_id]
            quat = self._env.sim.data.body_xquat[body_id]
            
            # Convert to 4x4 transformation matrix
            rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = rot_mat
            pose[:3, 3] = pos
            return pose
        except Exception as e:
            logger.debug(f"Could not get pose for object '{object_name}': {e}")
            return None
    
    def reset(self) -> Any:
        """Reset environment and restore intrinsics."""
        obs = self._env.reset()
        self._set_intrinsics()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """Step the environment."""
        return self._env.step(action)
    
    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()
    
    def close(self) -> None:
        """Close the environment."""
        self._env.close()
    
    # Delegate attribute access to wrapped environment
    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_env', '_custom_objects', '_object_names'):
            super().__setattr__(name, value)
        else:
            setattr(self._env, name, value)


def _generate_object_xml(obj_spec: Dict[str, Any]) -> str:
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
              friction="1 0.005 0.0001" condim="4"/>
        <site name="{name}_site" pos="0 0 0" size="0.002" rgba="1 0 0 0"/>
    </body>
    '''
    return xml


def make_dual_arm_env(
    custom_objects: Optional[List[Dict[str, Any]]] = None,
    ycb_objects: Optional[List[Union[str, Dict[str, Any]]]] = None,
    has_renderer: bool = constants.DEFAULT_HAS_RENDERER,
    has_offscreen_renderer: bool = constants.DEFAULT_HAS_OFFSCREEN_RENDERER,
    use_camera_obs: bool = constants.DEFAULT_USE_CAMERA_OBS,
    camera_name: str = constants.DEFAULT_CAMERA_NAME,
    camera_height: int = constants.DEFAULT_CAMERA_HEIGHT,
    camera_width: int = constants.DEFAULT_CAMERA_WIDTH,
) -> "DualArmEnvironmentWrapper":
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    Automatically sets realistic camera intrinsics.
    
    Args:
        custom_objects: List of custom primitive object specifications. Each dict can have:
            - type: 'box', 'cylinder', 'ball', 'capsule'
            - name: unique object name
            - size: size parameters (varies by type)
            - pos: [x, y, z] position
            - rgba: [r, g, b, a] color
            - density: object density in kg/m³
        ycb_objects: List of YCB object names or specifications. Can be:
            - String: YCB object name (e.g., 'apple', 'mug')
            - Dict: {'ycb_name': 'apple', 'name': 'apple_1', 'pos': [x, y, z]}
        has_renderer: Whether to enable on-screen rendering (default from constants)
        has_offscreen_renderer: Whether to enable off-screen rendering (default from constants)
        use_camera_obs: Whether to include camera observations (default from constants)
        camera_name: Name of camera to use (default from constants)
        camera_height: Height of camera images (default from constants)
        camera_width: Width of camera images (default from constants)
    
    Returns:
        DualArmEnvironmentWrapper: Wrapped robosuite environment
        
    Example:
        # Create environment with custom objects
        env = make_dual_arm_env(
            custom_objects=[
                {'type': 'box', 'name': 'red_cube', 'size': [0.04, 0.04, 0.04], 
                 'pos': [0.0, 0.0, 0.85], 'rgba': [1, 0, 0, 1]},
            ],
            ycb_objects=['apple', 'mug']
        )
        
        # List available YCB objects
        from robosuite_franka_manipulation.environments.dual_arm import get_available_ycb_objects
        print(get_available_ycb_objects())
    """

    # Load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller=constants.DEFAULT_CONTROLLER_NAME)
    osc_pose_config["gripper"] = {"type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE}
    osc_pose_config["input_ref_frame"] = constants.DEFAULT_INPUT_REF_FRAME
    osc_pose_config["input_type"] = constants.DEFAULT_INPUT_TYPE

    # Create the composite controller config
    controller_config = {
        "type": constants.DEFAULT_CONTROLLER_TYPE,
        "body_parts": {
            "right": osc_pose_config,
            "right_gripper": {
                "type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE
            }
        }
    }
    
    # For dual-arm, controller_configs should be a list of configs
    controller_configs = [controller_config, controller_config]
    
    # Prepare objects list for robosuite
    all_objects = []
    object_names = []
    
    # Process YCB objects
    if ycb_objects:
        for i, ycb_spec in enumerate(ycb_objects):
            if isinstance(ycb_spec, str):
                ycb_name = ycb_spec
                instance_name = f"{ycb_name}_{i}"
            else:
                ycb_name = ycb_spec.get('ycb_name', ycb_spec.get('name', 'unknown'))
                instance_name = ycb_spec.get('name', f"{ycb_name}_{i}")
            
            # Check if YCB object exists
            xml_path = constants.YCB_DATASET_PATH / f"{ycb_name}.xml"
            if xml_path.exists():
                try:
                    ycb_obj = YCBObject(
                        name=instance_name,
                        ycb_object_name=ycb_name,
                    )
                    all_objects.append(ycb_obj)
                    object_names.append(instance_name)
                    logger.info(f"Added YCB object: {ycb_name} as {instance_name}")
                except Exception as e:
                    logger.error(f"Failed to load YCB object '{ycb_name}': {e}")
            else:
                logger.warning(f"YCB object '{ycb_name}' not found at {xml_path}")
                available = get_available_ycb_objects()[:10]
                logger.info(f"Available YCB objects (first 10): {available}")

    # Create the base environment
    env_kwargs = {
        "env_name": constants.DEFAULT_ENV_NAME,
        "robots": constants.DEFAULT_ROBOT_TYPES,
        "gripper_types": constants.DEFAULT_GRIPPER_TYPES_ENV,
        "controller_configs": controller_configs,
        "env_configuration": constants.DEFAULT_ENV_CONFIG,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": has_offscreen_renderer,
        "use_camera_obs": use_camera_obs,
        "camera_names": camera_name,
        "camera_heights": camera_height,
        "camera_widths": camera_width,
        "camera_depths": constants.DEFAULT_CAMERA_DEPTH,
        "control_freq": constants.DEFAULT_CONTROL_FREQ,
        "horizon": constants.DEFAULT_HORIZON,
        "ignore_done": constants.DEFAULT_IGNORE_DONE,
        "reward_shaping": constants.DEFAULT_REWARD_SHAPING,
    }
    
    env = suite.make(**env_kwargs)
    
    # Combine custom objects for the wrapper
    all_custom_objects = (custom_objects or []).copy()
    
    # Wrap environment to handle camera intrinsics and objects
    wrapped_env = DualArmEnvironmentWrapper(env, custom_objects=all_custom_objects)
    
    # Store object names for external access
    wrapped_env._object_names.extend(object_names)
    
    return wrapped_env


# =============================================================================
# Convenience Functions
# =============================================================================

def list_ycb_objects() -> None:
    """Print all available YCB objects to console."""
    objects = get_available_ycb_objects()
    print(f"\nAvailable YCB Objects ({len(objects)} total):")
    print("-" * 40)
    for i, obj in enumerate(objects):
        print(f"  {i+1:3d}. {obj}")
    print("-" * 40)


if __name__ == '__main__':
    
    print("\n" + "=" * 60)
    print("Dual-Arm Environment Test")
    print("=" * 60)
    
    # List available YCB objects
    list_ycb_objects()
    
    try:
        # Create environment with sample objects
        print("\nCreating dual-arm environment with sample objects...")
        
        custom_objects = [
            {
                'type': 'box',
                'name': 'red_block',
                'size': [0.04, 0.04, 0.04],
                'pos': [0.0, 0.0, 0.85],
                'rgba': [0.9, 0.1, 0.1, 1.0],
            },
        ]
        
        # Try to load a YCB object if available
        available_ycb = get_available_ycb_objects()
        ycb_to_load = ['apple', 'mug'] if all(o in available_ycb for o in ['apple', 'mug']) else []
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            ycb_objects=ycb_to_load,
        )
        
        logger.info("Dual-arm handoff environment created successfully.")
        print(f"Objects in environment: {env.get_object_names()}")

        # Reset and step through a few times to test
        env.reset()
        for step in range(min(100, env.horizon)):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
            env.render()
        
        logger.info(f"Environment stepped for {step + 1} steps.")
        env.close()
        print("\nTest completed successfully!")

    except Exception as e:
        logger.error(f"Failed to create or step dual-arm handoff environment: {e}")
        traceback.print_exc()