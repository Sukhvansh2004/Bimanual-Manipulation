import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.models.objects import BoxObject, CylinderObject, BallObject, CapsuleObject
from robosuite.controllers import load_part_controller_config

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils.env_utils import (
    get_available_ycb_objects,
    list_ycb_objects,
    YCBObject,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=constants.LOG_LEVEL)


class DualArmEnvironment(TwoArmHandover):
    """
    Custom dual-arm environment that extends TwoArmHandover to support 
    custom primitive and YCB objects using standard robosuite APIs.
    """
    
    def __init__(
        self,
        custom_objects: Optional[List[Dict[str, Any]]] = None,
        ycb_objects: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the custom environment.
        
        Args:
            custom_objects: List of custom primitive object specifications
            ycb_objects: List of YCB object specifications
            **kwargs: Arguments passed to the TwoArmHandover base class
        """
        self._custom_object_specs = custom_objects or []
        self._ycb_object_specs = ycb_objects or []
        
        # Store (name, object_instance, position) tuples
        self._added_objects: List[Tuple[str, Any, np.ndarray]] = []
        
        super().__init__(**kwargs)
    
    def _load_model(self):
        """
        Overridden to load custom objects into the model.
        """
        super()._load_model()
        
        self._added_objects = []
        
        # 1. Create and add primitive objects
        for obj_spec in self._custom_object_specs:
            obj_type = obj_spec.get('type', 'box')
            name = obj_spec.get('name', f'custom_obj_{len(self._added_objects)}')
            pos = np.array(obj_spec.get('pos', [0.0, 0.0, 0.90]))
            size = obj_spec.get('size', [0.05, 0.05, 0.05])
            rgba = obj_spec.get('rgba', [0.5, 0.5, 0.5, 1.0])
            density = obj_spec.get('density', 500.0)
            
            if obj_type == 'box':
                # BoxObject takes half-sizes
                obj = BoxObject(
                    name=name,
                    size=[s / 2.0 for s in size], 
                    rgba=rgba,
                    density=density,
                    joints="default" # Adds a free joint
                )
            elif obj_type == 'cylinder':
                obj = CylinderObject(
                    name=name,
                    size=size,
                    rgba=rgba,
                    density=density,
                    joints="default"
                )
            elif obj_type in ('ball', 'sphere'):
                # BallObject takes radius as a 1-element list
                radius = size[0] if isinstance(size, (list, tuple)) else size
                obj = BallObject(
                    name=name,
                    size=[radius],
                    rgba=rgba,
                    density=density,
                    joints="default"
                )
            elif obj_type == 'capsule':
                obj = CapsuleObject(
                    name=name,
                    size=size,
                    rgba=rgba,
                    density=density,
                    joints="default"
                )
            else:
                logger.warning(f"Unknown object type: {obj_type}, skipping.")
                continue
                
            self.model.merge_objects([obj])
            self._added_objects.append((name, obj, pos))
            logger.info(f"Added primitive object: {name}")

        # 2. Create and add YCB objects
        for ycb_spec in self._ycb_object_specs:
            # Handle both string and dict specifications (normalized in make_dual_arm_env)
            if isinstance(ycb_spec, dict):
                ycb_name = ycb_spec.get('ycb_name', 'unknown')
                instance_name = ycb_spec.get('instance_name', ycb_spec.get('name', f"{ycb_name}_0"))
                pos = np.array(ycb_spec.get('pos', [0.0, 0.0, 0.90]))
            else:
                continue

            try:
                # YCBObject is a MujocoXMLObject, which is a valid robosuite object
                obj = YCBObject(
                    name=instance_name,
                    ycb_object_name=ycb_name
                )
                self.model.merge_objects([obj])
                self._added_objects.append((instance_name, obj, pos))
                logger.info(f"Added YCB object: {instance_name} ({ycb_name})")
            except Exception as e:
                logger.error(f"Failed to load YCB object {ycb_name}: {e}")

    def _reset_internal(self):
        """
        Overridden to set initial positions of custom objects.
        """
        super()._reset_internal()
        
        # Set realistic camera intrinsics
        self._set_intrinsics()
        
        # Place all custom objects
        for name, obj, pos in self._added_objects:
            # Robosuite objects with 'default' joints usually have a joint named {name}_joint0
            # We look up the joint address in the simulation model
            joint_name = f"{name}_joint0"
            
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
                qpos_addr = self.sim.model.jnt_qposadr[joint_id]
                
                # Set position (x,y,z) and orientation (w,x,y,z - identity)
                # qpos for free joint is 7 dims: [x, y, z, qw, qx, qy, qz] (MuJoCo convention is w-first)
                initial_qpos = np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0])
                
                self.sim.data.qpos[qpos_addr:qpos_addr+7] = initial_qpos
                
            except ValueError:
                logger.warning(f"Could not find joint '{joint_name}' for object '{name}'. Object might be fixed or name mismatch.")
            except Exception as e:
                logger.error(f"Error placing object {name}: {e}")

    def _set_intrinsics(self) -> None:
        """Set realistic camera intrinsics for all cameras."""
        # This mirrors the logic from the previous wrapper
        for i in range(len(self.camera_names)):
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)

    def get_object_names(self) -> List[str]:
        """Get the names of all custom objects in the environment."""
        return [name for name, _, _ in self._added_objects]

    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get the pose of a custom object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            4x4 transformation matrix or None if not found
        """
        try:
            body_id = self.sim.model.body_name2id(object_name)
            pos = self.sim.data.body_xpos[body_id]
            quat = self.sim.data.body_xquat[body_id] # [w, x, y, z]
            
            # Convert to 4x4 transformation matrix
            # SciPy expects [x, y, z, w]
            rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = rot_mat
            pose[:3, 3] = pos
            return pose
        except Exception as e:
            logger.debug(f"Could not get pose for object '{object_name}': {e}")
            return None

    def set_camera_free(self) -> None:
        """Set the viewer camera to 'Free' mode."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.set_camera(-1)
            except Exception:
                pass


def make_dual_arm_env(
    custom_objects: Optional[List[Dict[str, Any]]] = None,
    ycb_objects: Optional[List[Union[str, Dict[str, Any]]]] = None,
    has_renderer: bool = constants.DEFAULT_HAS_RENDERER,
    has_offscreen_renderer: bool = constants.DEFAULT_HAS_OFFSCREEN_RENDERER,
    use_camera_obs: bool = constants.DEFAULT_USE_CAMERA_OBS,
    camera_name: str = constants.DEFAULT_CAMERA_NAME,
    camera_height: int = constants.DEFAULT_CAMERA_HEIGHT,
    camera_width: int = constants.DEFAULT_CAMERA_WIDTH,
) -> DualArmEnvironment:
    """
    Creates a DualArmEnvironment configured for a dual-arm Franka handoff task.
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
    
    # Normalize YCB object specs
    normalized_ycb_objects = []
    if ycb_objects:
        for i, ycb_spec in enumerate(ycb_objects):
            if isinstance(ycb_spec, str):
                ycb_name = ycb_spec
                instance_name = f"{ycb_name}_{i}"
                pos = [-0.1 + 0.1 * i, -0.45, 0.90]
            else:
                ycb_name = ycb_spec.get('ycb_name', ycb_spec.get('name', 'unknown'))
                instance_name = ycb_spec.get('name', f"{ycb_name}_{i}")
                pos = ycb_spec.get('pos', [-0.1 + 0.1 * i, -0.45, 0.90])
            
            # Check if YCB object exists
            xml_path = constants.YCB_DATASET_PATH / f"{ycb_name}.xml"
            if xml_path.exists():
                normalized_ycb_objects.append({
                    'ycb_name': ycb_name,
                    'instance_name': instance_name,
                    'pos': pos,
                })
            else:
                logger.warning(f"YCB object '{ycb_name}' not found at {xml_path}")

    # Environment configuration arguments
    env_kwargs = {
        # 'env_name' is not needed as we instantiate the class directly, 
        # but 'robots', 'env_configuration' etc are passed to TwoArmHandover
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

    # Instantiate our custom environment class directly
    env = DualArmEnvironment(
        custom_objects=custom_objects,
        ycb_objects=normalized_ycb_objects,
        **env_kwargs
    )
    
    return env


if __name__ == '__main__':
    
    print("\n" + "=" * 60)
    print("Dual-Arm Environment Test (Updated)")
    print("=" * 60)
    
    list_ycb_objects()
    
    try:
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
        
        available_ycb = get_available_ycb_objects()
        ycb_to_load = ['apple', 'mug'] if all(o in available_ycb for o in ['apple', 'mug']) else []
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            ycb_objects=ycb_to_load,
        )
        
        logger.info("Dual-arm handoff environment created successfully.")
        print(f"Objects in environment: {env.get_object_names()}")

        env.reset()
        for step in range(min(50, env.horizon)):
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
        import traceback
        traceback.print_exc()