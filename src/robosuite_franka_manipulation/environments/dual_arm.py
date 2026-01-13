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

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils.env_utils import (
    get_available_ycb_objects,
    list_ycb_objects,
    generate_object_xml,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=constants.LOG_LEVEL)


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
        ycb_objects: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the environment wrapper.
        
        Args:
            env: The robosuite environment to wrap
            custom_objects: Optional list of custom primitive object specifications
            ycb_objects: Optional list of YCB object specifications
        """
        self._env = env
        self._custom_objects = custom_objects or []
        self._ycb_objects = ycb_objects or []
        self._object_names: List[str] = []
        self._set_intrinsics()
        
        # Add all objects in a single XML modification
        if self._custom_objects or self._ycb_objects:
            self._add_all_objects()
    
    def _set_intrinsics(self) -> None:
        """Set realistic camera intrinsics for all cameras."""
        # robosuite dynamic attributes not in type stubs
        for i in range(len(self._env.camera_names)):
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self._env.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)
    
    def _add_all_objects(self) -> None:
        """
        Add all custom objects (primitives and YCB) in a single XML modification.
        This ensures all objects are added together without resetting in between.
        """
        try:
            # Get current XML from robosuite's model
            current_xml = self._env.model.get_xml()
            
            # Parse the XML
            root = ET.fromstring(current_xml)
            worldbody = root.find('worldbody')
            asset = root.find('asset')
            
            if worldbody is None:
                raise ValueError("Could not find worldbody in model XML")
            
            # Create asset section if it doesn't exist
            if asset is None:
                asset = ET.SubElement(root, 'asset')
            
            # Add primitive objects
            for obj_spec in self._custom_objects:
                obj_type = obj_spec.get('type', 'box')
                obj_name = obj_spec.get('name', f'custom_obj_{len(self._object_names)}')
                pos = obj_spec.get('pos', [0.0, 0.0, 0.90])
                
                if obj_type == 'ycb':
                    logger.warning(f"Use ycb_objects parameter for YCB objects, not custom_objects")
                    continue
                
                # Generate XML for primitive object
                object_xml = generate_object_xml(obj_spec)
                
                # Parse and add to worldbody
                obj_element = ET.fromstring(f"<root>{object_xml}</root>")
                for body in obj_element.findall('body'):
                    worldbody.append(body)
                
                self._object_names.append(obj_name)
                logger.info(f"Added custom object: {obj_name} at position {pos}")
            
            # Add YCB objects
            for ycb_obj in self._ycb_objects:
                xml_path = ycb_obj.get('xml_path')
                instance_name = ycb_obj.get('instance_name')
                pos = ycb_obj.get('pos', [0.0, 0.0, 0.90])
                ycb_name = ycb_obj.get('ycb_name', 'unknown')
                
                if not xml_path or not Path(xml_path).exists():
                    logger.warning(f"YCB XML not found: {xml_path}")
                    continue
                
                # Parse the YCB XML file
                ycb_tree = ET.parse(xml_path)
                ycb_root = ycb_tree.getroot()
                
                # Get asset directory
                asset_dir = str(Path(xml_path).parent)
                
                # Add assets from YCB XML
                ycb_assets = ycb_root.find('asset')
                if ycb_assets is not None:
                    for mesh in ycb_assets.findall('mesh'):
                        new_mesh = ET.SubElement(asset, 'mesh')
                        mesh_name = mesh.get('name', 'mesh')
                        new_mesh.set('name', f"{instance_name}_{mesh_name}")
                        mesh_file = mesh.get('file', '')
                        full_mesh_path = os.path.join(asset_dir, mesh_file)
                        new_mesh.set('file', full_mesh_path)
                    
                    for texture in ycb_assets.findall('texture'):
                        new_tex = ET.SubElement(asset, 'texture')
                        tex_name = texture.get('name', 'texture')
                        new_tex.set('name', f"{instance_name}_{tex_name}")
                        tex_file = texture.get('file', '')
                        full_tex_path = os.path.join(asset_dir, tex_file)
                        new_tex.set('file', full_tex_path)
                        new_tex.set('type', texture.get('type', '2d'))
                    
                    for material in ycb_assets.findall('material'):
                        new_mat = ET.SubElement(asset, 'material')
                        mat_name = material.get('name', 'material')
                        new_mat.set('name', f"{instance_name}_{mat_name}")
                        tex_ref = material.get('texture', '')
                        if tex_ref:
                            new_mat.set('texture', f"{instance_name}_{tex_ref}")
                
                # Create body for YCB object
                ycb_body_src = ycb_root.find('.//worldbody/body')
                if ycb_body_src is not None:
                    new_body = ET.SubElement(worldbody, 'body')
                    new_body.set('name', instance_name)
                    new_body.set('pos', ' '.join(map(str, pos)))
                    
                    # Add freejoint
                    freejoint = ET.SubElement(new_body, 'freejoint')
                    freejoint.set('name', f"{instance_name}_joint")
                    
                    # Copy geoms with renamed references
                    for geom in ycb_body_src.findall('geom'):
                        new_geom = ET.SubElement(new_body, 'geom')
                        geom_name = geom.get('name', 'geom')
                        new_geom.set('name', f"{instance_name}_{geom_name}")
                        
                        geom_type = geom.get('type', 'mesh')
                        new_geom.set('type', geom_type)
                        
                        mesh_ref = geom.get('mesh', '')
                        if mesh_ref:
                            new_geom.set('mesh', f"{instance_name}_{mesh_ref}")
                        
                        mat_ref = geom.get('material', '')
                        if mat_ref:
                            new_geom.set('material', f"{instance_name}_{mat_ref}")
                        
                        for attr in ['pos', 'euler', 'contype', 'conaffinity', 'group', 'rgba']:
                            val = geom.get(attr)
                            if val:
                                new_geom.set(attr, val)
                        
                        # Ensure collision geoms have proper collision attributes
                        if geom_type == 'mesh':
                             # Set default collision attributes if not specified
                            if not new_geom.get('contype'):
                                new_geom.set('contype', '1')
                            if not new_geom.get('conaffinity'):
                                new_geom.set('conaffinity', '1')
                    
                    # Copy inertial properties
                    inertial_src = ycb_body_src.find('inertial')
                    if inertial_src is not None:
                        new_inertial = ET.SubElement(new_body, 'inertial')
                        for attr in ['pos', 'euler', 'mass', 'diaginertia']:
                            val = inertial_src.get(attr)
                            if val:
                                new_inertial.set(attr, val)
                
                self._object_names.append(instance_name)
                logger.info(f"Added YCB object '{ycb_name}' as '{instance_name}'")
            
            # Convert back to string and reload
            new_xml = ET.tostring(root, encoding='unicode')
            self._env.reset_from_xml_string(new_xml)
            
            logger.info(f"Successfully added {len(self._object_names)} objects to simulation")
            
        except Exception as e:
            logger.error(f"Failed to add objects: {e}")
            traceback.print_exc()
    
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
        """
        Reset environment, restore intrinsics, and re-add custom objects.
        
        Note: robosuite's reset reloads the original model, so we need to 
        re-inject our custom objects after each reset.
        """
        obs = self._env.reset()
        self._set_intrinsics()
        
        # Re-add custom objects since reset reloads the original model
        if self._custom_objects or self._ycb_objects:
            # Clear object names and re-add
            self._object_names = []
            self._add_all_objects()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """Step the environment."""
        return self._env.step(action)
    
    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()
    
    def set_camera_free(self) -> None:
        """
        Set the viewer camera to 'Free' mode for manual control.
        
        This allows the user to freely rotate and pan the camera view
        using mouse controls instead of being locked to a fixed camera.
        Must be called after the first render() to ensure viewer is initialized.
        """
        if hasattr(self._env, 'viewer') and self._env.viewer is not None:
            # In robosuite's viewer, set_camera(-1) sets free camera mode
            # camera_id < 0 means free camera, >= 0 means fixed camera
            try:
                self._env.viewer.set_camera(-1)
                logger.debug("Viewer camera set to Free mode")
            except Exception as e:
                logger.debug(f"Could not set camera to free mode: {e}")
    
    def close(self) -> None:
        """Close the environment."""
        self._env.close()
    
    # Delegate attribute access to wrapped environment
    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_env', '_custom_objects', '_ycb_objects', '_object_names'):
            super().__setattr__(name, value)
        else:
            setattr(self._env, name, value)


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
    
    # Prepare YCB objects info for post-creation injection
    ycb_objects_to_add = []
    object_names = []
    
    # Process YCB objects
    if ycb_objects:
        for i, ycb_spec in enumerate(ycb_objects):
            if isinstance(ycb_spec, str):
                ycb_name = ycb_spec
                instance_name = f"{ycb_name}_{i}"
                # Default position on table: x offset based on index, y at table center, z above table
                pos = [-0.1 + 0.1 * i, -0.45, 0.90]
            else:
                ycb_name = ycb_spec.get('ycb_name', ycb_spec.get('name', 'unknown'))
                instance_name = ycb_spec.get('name', f"{ycb_name}_{i}")
                pos = ycb_spec.get('pos', [-0.1 + 0.1 * i, -0.45, 0.90])
            
            # Check if YCB object exists
            xml_path = constants.YCB_DATASET_PATH / f"{ycb_name}.xml"
            if xml_path.exists():
                ycb_objects_to_add.append({
                    'ycb_name': ycb_name,
                    'instance_name': instance_name,
                    'xml_path': xml_path,
                    'pos': pos,
                })
                object_names.append(instance_name)
                logger.info(f"Queued YCB object: {ycb_name} as {instance_name}")
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
    
    # Wrap environment to handle camera intrinsics and objects
    # Pass both custom objects and YCB objects to the wrapper for unified handling
    wrapped_env = DualArmEnvironmentWrapper(
        env, 
        custom_objects=custom_objects,
        ycb_objects=ycb_objects_to_add,
    )
    
    return wrapped_env


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
            action_dim = env.action_dim
            action = np.random.uniform(-0.3, 0.3, action_dim)
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