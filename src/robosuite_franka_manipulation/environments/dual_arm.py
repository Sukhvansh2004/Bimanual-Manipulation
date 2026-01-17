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
        
        # Store (name, object_instance, initial_pos) tuples
        self._added_objects: List[Tuple[str, Any, np.ndarray]] = []
        
        # Settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = [0, 0, 0.8] # Default offset
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

        # 2. Adjust robot base positions (Logic from TwoArmHandover)
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other
                for robot, rotation, offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # 3. Load Table Arena
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # 4. Create Custom and YCB Objects
        self._added_objects = []
        mujoco_objects = [] # List to pass to ManipulationTask

        # Helper to process object specs
        def _process_objects(specs, is_ycb=False):
            for spec in specs:
                if not isinstance(spec, dict): 
                    continue
                
                # Extract common properties
                pos = np.array(spec.get('pos', [0.0, 0.0, 0.90]))
                
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
                    existing_names = {n for n, _, _ in self._added_objects}
                    if name in existing_names:
                        suffix = 1
                        candidate = f"{name}_{suffix}"
                        while candidate in existing_names:
                            suffix += 1
                            candidate = f"{name}_{suffix}"
                        name = candidate
                    try:
                        obj = YCBObject(name=name, ycb_object_name=ycb_name)
                        logger.info(f"Added YCB object: {name} ({ycb_name})")
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
                    
                    if obj_type == 'box':
                        obj = BoxObject(name=name, size=[s/2.0 for s in size], rgba=rgba, density=density, joints="default")
                    elif obj_type == 'cylinder':
                        obj = CylinderObject(name=name, size=size, rgba=rgba, density=density, joints="default")
                    elif obj_type in ('ball', 'sphere'):
                        radius = size[0] if isinstance(size, (list, tuple)) else size
                        obj = BallObject(name=name, size=[radius], rgba=rgba, density=density, joints="default")
                    elif obj_type == 'capsule':
                        obj = CapsuleObject(name=name, size=size, rgba=rgba, density=density, joints="default")
                    else:
                        logger.warning(f"Unknown object type: {obj_type}")
                        continue
                    
                    logger.info(f"Added primitive object: {name}")

                mujoco_objects.append(obj)
                self._added_objects.append((name, obj, pos))

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
        for name, obj, _ in self._added_objects:
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
            for name, obj, _ in self._added_objects:
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
        for name, obj, pos in self._added_objects:
            joint_name = f"{name}_joint0"
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
                qpos_addr = self.sim.model.jnt_qposadr[joint_id]
                
                # Set position (x,y,z) and orientation (w,x,y,z - identity)
                # MuJoCo convention is w-first for quats
                initial_qpos = np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0])
                
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
        return [name for name, _, _ in self._added_objects]

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
) -> DualArmEnvironment:
    """
    Creates a DualArmEnvironment configured for a dual-arm Franka task.
    """

    # Load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller=constants.DEFAULT_CONTROLLER_NAME)
    osc_pose_config["gripper"] = {"type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE}
    osc_pose_config["input_ref_frame"] = constants.DEFAULT_INPUT_REF_FRAME
    osc_pose_config["input_type"] = constants.DEFAULT_INPUT_TYPE
    
    # Increase impedance (stiffness and damping) for better resistance to external forces
    # Higher kp = stiffer position control, higher damping_ratio = more stable
    osc_pose_config["kp"] = constants.DEFAULT_KP  # Position stiffness (default is ~150)
    osc_pose_config["damping_ratio"] = constants.DEFAULT_DAMPING_RATIO  # Critical damping (default is ~1.0)
    osc_pose_config["kp_limits"] = constants.DEFAULT_KP_LIMITS  # Allow higher stiffness values
    osc_pose_config["damping_ratio_limits"] = constants.DEFAULT_DAMPING_RATIO_LIMITS  # Allow higher damping

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