import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R

import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, BallObject, CapsuleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.controllers import load_part_controller_config

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)


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
            name: Unique instance name for this object (used in simulation)
            ycb_object_name: Name of the YCB object file (e.g., 'apple', 'mug')
            joints: List of joint specifications (default: free joint)
            obj_type: Type of object ('collision', 'visual', or 'all')
            duplicate_collision_geoms: Whether to duplicate collision geometries
        """
        self.ycb_object_name = ycb_object_name
        
        # Find the XML file for this YCB object
        xml_path = constants.YCB_DATASET_PATH / f"{ycb_object_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"YCB object XML not found: {xml_path}")
        
        # Use default free joint if not specified
        # This lets robosuite handle joint creation properly
        super().__init__(
            str(xml_path),
            name=name,
            joints="default" if joints is None else joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
    
    @property
    def bottom_offset(self):
        """Return the bottom offset of the object for placement."""
        bottom_site = self.worldbody.find(f"./body/site[@name='{self.naming_prefix}bottom_site']")
        if bottom_site is not None:
            return np.array([0, 0, float(bottom_site.get("pos", "0 0 0").split()[2])])
        return np.array([0, 0, -0.02])
    
    @property
    def top_offset(self):
        """Return the top offset of the object."""
        return np.array([0, 0, 0.05])
    
    @property
    def horizontal_radius(self):
        """Return the horizontal radius for placement."""
        return 0.05


class DualArmEnvironment(TwoArmEnv):
    """
    Custom dual-arm environment that extends TwoArmEnv to support 
    custom primitive and YCB objects using standard robosuite APIs.
    Refactored to match TwoArmHandover structure.
    """
    
    def __init__(
        self,
        custom_objects: Optional[List[Dict[str, Any]]] = None,
        ycb_objects: Optional[List[Dict[str, Any]]] = None,
        # Default args matching TwoArmHandover where applicable
        env_configuration=constants.DEFAULT_ENV_CONFIG,
        table_full_size=constants.DEFAULT_TABLE_SIZE,
        table_friction=constants.DEFAULT_TABLE_FRICTION,
        use_object_obs=constants.DEFAULT_USE_OBJECT_OBS,
        **kwargs
    ):
        """
        Initialize the custom environment.
        """
        self._custom_object_specs = custom_objects or []
        self._ycb_object_specs = ycb_objects or []
        
        # Store (name, object_instance, initial_pos, initial_quat, is_static) tuples
        self._added_objects: List[Tuple[str, Any, np.ndarray, np.ndarray, bool]] = []
        
        # Settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = [0, 0, 1.0] # Default offset
        self.use_object_obs = use_object_obs
        
        super().__init__(
            env_configuration=env_configuration,
            **kwargs
        )
    
    def _load_model(self):
        """
        Overridden to load custom objects into the model using ManipulationTask.
        """
        # 1. Initialize robots via super class
        super()._load_model()

        # Override robot init_qpos to keep EEF well above the table surface
        for robot in self.robots:
            robot.init_qpos = constants.FRANKA_HIGH_INIT_QPOS.copy()

        # 2. Adjust robot base positions
        if self.env_configuration in constants.ROBOT_CONFIGURATIONS:
            configs = constants.ROBOT_CONFIGURATIONS[self.env_configuration]
            
            for i, robot in enumerate(self.robots):
                if i >= len(configs):
                    break
                
                cfg = configs[i]
                
                # Get default mounting position on table edge (usually center of edge)
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                
                # Convert config quaternion to rotation matrix
                # Config quat is [x, y, z, w]
                # robosuite T.quat2mat expects [w, x, y, z], scipy R.from_quat expects [x, y, z, w]
                # Let's use scipy to match the convention in constants.py
                quat_xyzw = cfg['quaternion']
                rot_obj = R.from_quat(quat_xyzw)
                rot_mat = rot_obj.as_matrix()
                
                # Rotate the mounting point vector (e.g. for opposed config)
                xpos = rot_mat @ np.array(xpos)
                
                # Add offset
                xpos += cfg['position']
                
                # Set base position and orientation
                robot.robot_model.set_base_xpos(xpos)
                # set_base_ori expects euler angles (radians), not a rotation matrix
                robot.robot_model.set_base_ori(rot_obj.as_euler('xyz'))
        else:
            raise ValueError(f"Unknown env_configuration '{self.env_configuration}', skipping base setup.")

        # 3. Load Table Arena
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # 4. Create Custom and YCB Objects
        self._added_objects = []  # list of (name, obj, pos, quat, is_static)
        mujoco_objects = [] # List to pass to ManipulationTask

        # Helper to process object specs
        def _process_objects(specs, is_ycb=False):
            for spec in specs:
                if not isinstance(spec, dict): 
                    continue
                
                # Extract position
                pos = np.array(spec.get('pos', constants.DEFAULT_OBJECT_POSITION.copy()))
                # Extract orientation
                quat = np.array(spec.get('quat', constants.DEFAULT_OBJECT_QUATERNION.copy()))
                # Check if object should be fixed in place (no free joint)
                obj_is_static = spec.get('is_static', False)
                joints_arg = [] if obj_is_static else "default"
                
                if is_ycb:

                    ycb_name = spec.get('ycb_name', 'unknown')
                    name = spec.get('instance_name', spec.get('name', f"{ycb_name}_0"))

                    # Many YCB XMLs already prefix geoms/sites with the object name (e.g. `plate_visual_geom`).
                    # If the robosuite instance name matches that prefix exactly, robosuite's internal naming
                    # can end up expecting double-prefixed geoms (e.g. `plate_plate_visual_geom`) during ID mapping.
                    # Make the instance name distinct from the raw YCB asset prefix to avoid that failure.
                    if name == ycb_name:
                        name = f"{ycb_name}_0"

                    # Ensure uniqueness across all objects in the scene.
                    existing_names = {n for n, _, _, _, _ in self._added_objects}
                    if name in existing_names:
                        suffix = 1
                        candidate = f"{name}_{suffix}"
                        while candidate in existing_names:
                            suffix += 1
                            candidate = f"{name}_{suffix}"
                        name = candidate
                    try:
                        obj = YCBObject(name=name, ycb_object_name=ycb_name, joints=joints_arg)
                        logger.info(f"Added YCB object: {name} ({ycb_name}){' [static]' if obj_is_static else ''}")
                    except Exception as e:
                        logger.error(f"Failed to load YCB object {ycb_name}: {e}")
                        continue
                else:
                    # Primitive objects
                    obj_type = spec.get('type', 'box')
                    name = spec.get('name', f'custom_obj_{len(mujoco_objects)}')
                    size = spec.get('size', [0.05, 0.05, 0.05])
                    rgba = spec.get('rgba', [0.5, 0.5, 0.5, 1.0])
                    density = spec.get('density', 500.0)
                    friction = spec.get('friction', None)
                    
                    if obj_type == 'box':
                        obj = BoxObject(name=name, size=[s/2.0 for s in size], rgba=rgba, density=density, friction=friction, joints=joints_arg)
                    elif obj_type == 'cylinder':
                        obj = CylinderObject(name=name, size=size, rgba=rgba, density=density, friction=friction, joints=joints_arg)
                    elif obj_type in ('ball', 'sphere'):
                        radius = size[0] if isinstance(size, (list, tuple)) else size
                        obj = BallObject(name=name, size=[radius], rgba=rgba, density=density, friction=friction, joints=joints_arg)
                    elif obj_type == 'capsule':
                        obj = CapsuleObject(name=name, size=size, rgba=rgba, density=density, friction=friction, joints=joints_arg)
                    else:
                        logger.warning(f"Unknown object type: {obj_type}")
                        continue
                    
                    logger.info(f"Added primitive object: {name}{' [static]' if obj_is_static else ''}")

                mujoco_objects.append(obj)
                self._added_objects.append((name, obj, pos, quat, obj_is_static))

        # Process both lists
        _process_objects(self._custom_object_specs, is_ycb=False)
        _process_objects(self._ycb_object_specs, is_ycb=True)

        logger.debug(f"Processed custom and YCB objects for environment. Count: {len(mujoco_objects)}")
        # 5. Create ManipulationTask (Handles merging of arena, robots, and objects)
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        
        # Setup references for all added objects
        # Use robosuite's standard pattern: obj.root_body returns the correctly prefixed name
        # See: robosuite/environments/manipulation/lift.py, pick_place.py, nut_assembly.py
        self.object_body_ids = {}
        for name, obj, _, _, _ in self._added_objects:
            try:
                self.object_body_ids[name] = self.sim.model.body_name2id(obj.root_body)
                logger.debug(f"Object '{name}' -> body '{obj.root_body}' -> id {self.object_body_ids[name]}")
            except ValueError as e:
                logger.warning(f"Could not find body for '{name}' (root_body='{obj.root_body}'): {e}")

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            sensors = []
            names = []

            # Create sensors for each object (only if body ID was found)
            for name, obj, _, _, _ in self._added_objects:
                # Skip objects that weren't found in the model
                if name not in self.object_body_ids:
                    logger.warning(f"Skipping observables for '{name}' - body ID not found")
                    continue
                
                # We need to capture the current value of 'name' in the closure
                # Using a factory function is cleaner
                
                def get_pos_sensor(n):
                    return lambda obs_cache: np.array(self.sim.data.body_xpos[self.object_body_ids[n]])

                def get_quat_sensor(n):
                    return lambda obs_cache: T.convert_quat(self.sim.data.body_xquat[self.object_body_ids[n]], to="xyzw")

                # Position sensor
                sensor_name_pos = f"{name}_pos"
                sensor_fn_pos = get_pos_sensor(name)
                # Manually set __name__ which Robosuite uses for keys
                sensor_fn_pos.__name__ = sensor_name_pos 
                
                # Orientation sensor
                sensor_name_quat = f"{name}_quat"
                sensor_fn_quat = get_quat_sensor(name)
                sensor_fn_quat.__name__ = sensor_name_quat

                # Register
                sensors.append(sensor(modality=modality)(sensor_fn_pos))
                sensors.append(sensor(modality=modality)(sensor_fn_quat))
                names.append(sensor_name_pos)
                names.append(sensor_name_quat)

            # Create Observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Overridden to set initial positions of custom objects.
        """
        super()._reset_internal()
        
        # Set realistic camera intrinsics
        self._set_intrinsics()
        
        # Place all custom objects
        # Note: We are using the explicit 'pos' from specs rather than a sampler
        # to preserve the user's configuration intent.
        for name, obj, pos, quat, obj_is_static in self._added_objects:
            if obj_is_static:
                # Static objects have no joint – set body position directly
                try:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    quat_wxyz = T.convert_quat(quat, to="wxyz")
                    self.sim.model.body_pos[body_id] = pos
                    self.sim.model.body_quat[body_id] = quat_wxyz
                    logger.debug(f"Placed static object '{name}' via body pos/quat.")
                except Exception as e:
                    logger.warning(f"Could not place static object '{name}': {e}")
            else:
                joint_name = f"{name}_joint0"
                try:
                    joint_id = self.sim.model.joint_name2id(joint_name)
                    qpos_addr = self.sim.model.jnt_qposadr[joint_id]
                    
                    # Set position (x,y,z) and orientation (w,x,y,z)
                    # MuJoCo convention is w-first for quats
                    # quat in spec is [x, y, z, w]
                    quat_wxyz = T.convert_quat(quat, to="wxyz")
                    initial_qpos = np.concatenate([pos, quat_wxyz])
                    
                    self.sim.data.qpos[qpos_addr:qpos_addr+7] = initial_qpos
                    
                except ValueError:
                    logger.warning(f"Could not find joint '{joint_name}' for object '{name}'.")
                except Exception as e:
                    logger.error(f"Error placing object {name}: {e}")

    def _set_intrinsics(self) -> None:
        """Set realistic camera intrinsics for all cameras."""
        for i in range(len(self.camera_names)):
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)

    def get_object_names(self) -> List[str]:
        """Get the names of all custom objects in the environment."""
        return [name for name, _, _, _, _ in self._added_objects]

    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get the pose of a custom object.
        """
        body_name_candidates = [f"{object_name}_main", object_name]
        
        for body_name in body_name_candidates:
            try:
                body_id = self.sim.model.body_name2id(body_name)
                pos = self.sim.data.body_xpos[body_id]
                quat = self.sim.data.body_xquat[body_id]  # [w, x, y, z]
                
                # Convert to 4x4 transformation matrix (Scipy expects x, y, z, w)
                quat_xyzw = T.convert_quat(quat, to="xyzw")
                rot_mat = R.from_quat(quat_xyzw).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = rot_mat
                pose[:3, 3] = pos
                return pose
            except Exception:
                continue
        return None
    
    def reward(self, action=None):
        """
        Dummy reward function required by base class.
        """
        return 0.0

    def _check_success(self):
        """
        Dummy success check required by base class.
        """
        return False


def make_dual_arm_env(
    custom_objects: Optional[List[Dict[str, Any]]] = None,
    ycb_objects: Optional[List[Union[str, Dict[str, Any]]]] = None,
    has_renderer: bool = constants.DEFAULT_HAS_RENDERER,
    has_offscreen_renderer: bool = constants.DEFAULT_HAS_OFFSCREEN_RENDERER,
    use_camera_obs: bool = constants.DEFAULT_USE_CAMERA_OBS,
    camera_name: str = constants.DEFAULT_CAMERA_NAME,
    camera_height: int = constants.DEFAULT_CAMERA_HEIGHT,
    camera_width: int = constants.DEFAULT_CAMERA_WIDTH,
    controller_name: str = constants.DEFAULT_CONTROLLER_NAME,
    input_type: str = constants.DEFAULT_INPUT_TYPE,
    table_full_size=None,
) -> DualArmEnvironment:
    """
    Creates a DualArmEnvironment configured for a dual-arm Franka task.
    
    Args:
        custom_objects: List of custom object specifications.
        ycb_objects: List of YCB object specifications (name string or dict).
        has_renderer: Whether to enable on-screen rendering.
        has_offscreen_renderer: Whether to enable off-screen rendering.
        use_camera_obs: Whether to include camera observations.
        camera_name: Name of the camera to use.
        camera_height: Height of camera images.
        camera_width: Width of camera images.
        controller_name: Controller type. Options:
            - 'OSC_POSE': Operational Space Control for end-effector pose
            - 'JOINT_POSITION': Joint position control
        input_type: Input mode for the controller. Options:
            - 'delta': Incremental/relative control
            - 'absolute': Target/absolute control
            
    Returns:
        DualArmEnvironment: The configured environment.
        
    Example:
        # Absolute JOINT_POSITION (default)
        env = make_dual_arm_env()
        
        # Absolute OSC_POSE
        env = make_dual_arm_env(controller_name='OSC_POSE', input_type='absolute')
        
        # Delta JOINT_POSITION
        env = make_dual_arm_env(controller_name='JOINT_POSITION', input_type='delta')
        
        # Absolute JOINT_POSITION
        env = make_dual_arm_env(controller_name='JOINT_POSITION', input_type='absolute')
    """
    # Validate controller configuration
    if controller_name not in constants.SUPPORTED_CONTROLLER_NAMES:
        raise ValueError(f"Unsupported controller_name: '{controller_name}'. "
                        f"Must be one of {constants.SUPPORTED_CONTROLLER_NAMES}")
    if input_type not in constants.SUPPORTED_INPUT_TYPES:
        raise ValueError(f"Unsupported input_type: '{input_type}'. "
                        f"Must be one of {constants.SUPPORTED_INPUT_TYPES}")

    # Load the part controller config for the specified controller
    part_config = load_part_controller_config(default_controller=controller_name)
    part_config["gripper"] = {"type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE}
    part_config["input_ref_frame"] = constants.DEFAULT_INPUT_REF_FRAME
    part_config["input_type"] = input_type
    
    part_config["kp"] = constants.DEFAULT_KP
    part_config["damping_ratio"] = constants.DEFAULT_DAMPING_RATIO
    part_config["kp_limits"] = constants.DEFAULT_KP_LIMITS
    part_config["damping_ratio_limits"] = constants.DEFAULT_DAMPING_RATIO_LIMITS

    # Create the composite controller config
    controller_config = {
        "type": constants.DEFAULT_CONTROLLER_TYPE,
        "body_parts": {
            "right": part_config,
            "right_gripper": {
                "type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE
            }
        }
    }
    
    # For dual-arm, controller_configs should be a list of configs
    controller_configs = [controller_config, controller_config]
    
    logger.info(f"Creating dual-arm environment with controller: {controller_name}, input_type: {input_type}")
    
    # Normalize YCB object specs
    normalized_ycb_objects = []
    if ycb_objects:
        for i, ycb_spec in enumerate(ycb_objects):
            if isinstance(ycb_spec, str):
                ycb_name = ycb_spec
                instance_name = f"{ycb_name}_{i}"
                # Use safer default position closer to table center (Y=0.0)
                pos = [-0.1 + 0.1 * i, 0.0, 1.10]
                ycb_spec = {'ycb_name': ycb_name, 'instance_name': instance_name, 'pos': pos}
            
            ycb_name = ycb_spec.get('ycb_name', ycb_spec.get('name', 'unknown'))
            xml_path = constants.YCB_DATASET_PATH / f"{ycb_name}.xml"
            if xml_path.exists():
                normalized_ycb_objects.append(ycb_spec)
            else:
                logger.warning(f"YCB object '{ycb_name}' not found at {xml_path}")

    # Environment configuration arguments
    env_kwargs = {
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
        "render_camera": constants.DEFAULT_RENDER_CAMERA,
    }

    # Optional table size override
    if table_full_size is not None:
        env_kwargs["table_full_size"] = tuple(table_full_size)

    env = DualArmEnvironment(
        custom_objects=custom_objects,
        ycb_objects=normalized_ycb_objects,
        **env_kwargs
    )
    
    return env


if __name__ == '__main__':
    logger.info("Starting Dual-Arm Environment Test...")
    try:
        custom_objects = [{
            'type': 'box', 'name': 'red_block', 'size': [0.04, 0.04, 0.04],
            'pos': [0.0, 0.0, 0.85], 'rgba': [0.9, 0.1, 0.1, 1.0],
        }]
        
        env = make_dual_arm_env(custom_objects=custom_objects)
        
        logger.info(f"Environment created. Objects: {env.get_object_names()}")
        env.reset()
        for step in range(10):
            env.step(np.random.uniform(-0.1, 0.1, env.action_dim))
        env.close()
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)