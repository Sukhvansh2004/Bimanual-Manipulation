import logging
import numpy as np
import os
import trimesh
import atexit
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation import constants
from bimanual_manipulation.utils.suction_weld import apply_suction_weld
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# =============================================================================
# Global Mesh Extraction Registry
# =============================================================================

class MeshExtractionManager:
    """Manages lazy extraction of meshes from environment and automatic cleanup."""
    def __init__(self):
        self.env = None
        self.temp_dir = None
        
    def register_env(self, env: DualArmEnvironment):
        self.env = env
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="meshes_")
            atexit.register(self.cleanup)
            logger.debug(f"Registered mesh extraction temp directory: {self.temp_dir}")

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up mesh extraction temp directory: {self.temp_dir}")
            self.temp_dir = None

    def get_mesh_path(self, object_name: str) -> Optional[str]:
        if self.temp_dir is None:
            # We don't have a temp dir yet, possibly because no env was registered
            # but we still want to try to register one if we can find it? 
            # No, caller should register_env first.
            return None
        
        # Check if already extracted
        mesh_path = os.path.join(self.temp_dir, f"{object_name}.obj")
        if os.path.exists(mesh_path):
            return mesh_path
            
        # Try to extract lazily
        if self.env is not None:
            # We try visual geom first
            visual_geom = f"{object_name}_visual"
            success = extract_mesh_from_geom(self.env, visual_geom, mesh_path)
            if not success:
                # Fallback to collision geom or main geom name
                success = extract_mesh_from_geom(self.env, object_name, mesh_path)
            
            if success:
                return mesh_path
        
        return None

_mesh_manager = MeshExtractionManager()

def register_mesh_extraction_env(env: DualArmEnvironment):
    """Register an environment for lazy mesh extraction."""
    _mesh_manager.register_env(env)


# =============================================================================
# MuJoCo Mesh Extraction Utilities
# =============================================================================

def extract_mesh_from_geom(env: DualArmEnvironment, geom_name: str, output_path: str) -> bool:
    """
    Extracts mesh data from a MuJoCo geom and saves it to a file.
    
    Args:
        env: Environment instance.
        geom_name: Name of the geom to extract.
        output_path: Path to save the mesh file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        model = env.sim.model
        # Check if geom exists
        try:
            geom_id = model.geom_name2id(geom_name)
        except Exception:
            # Try to find by partial match if exact name fails
            found = False
            for i in range(model.ngeom):
                name = model.geom_id2name(i)
                if name and geom_name in name:
                    geom_id = i
                    found = True
                    break
            if not found:
                return False

        geom_type = model.geom_type[geom_id]
        
        mesh = None
        if geom_type == 7: # Mesh
            mesh_id = model.geom_dataid[geom_id]
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
            faces = model.mesh_face[face_adr : face_adr + face_num]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        elif geom_type == 6: # Box
            size = model.geom_size[geom_id] # half-extents
            mesh = trimesh.creation.box(extents=size * 2)
            
        elif geom_type == 5: # Cylinder
            radius, half_height = model.geom_size[geom_id][:2]
            mesh = trimesh.creation.cylinder(radius=radius, height=half_height * 2)
            
        elif geom_type == 2: # Sphere
            radius = model.geom_size[geom_id][0]
            mesh = trimesh.creation.icosphere(radius=radius)
            
        if mesh is not None:
            mesh.export(output_path)
            logger.info(f"Extracted mesh for geom '{geom_name}' to {output_path}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Failed to extract mesh for geom '{geom_name}': {e}")
        return False


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
    # 1. Check if it's a YCB object first (standard path)
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
    
    # 2. If not YCB, try lazy extraction from environment
    return _mesh_manager.get_mesh_path(object_name)


# =============================================================================
# Object Specification Helpers
# =============================================================================

def create_ycb_object(
    ycb_name: str,
    instance_name: Optional[str] = None,
    position: Optional[List[float]] = None,
    quaternion: Optional[List[float]] = None,
    is_static: bool = False,
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
        'is_static': is_static,
    }


def create_primitive_object(
    obj_type: str,
    name: str,
    size: List[float],
    position: Optional[List[float]] = None,
    quat: Optional[List[float]] = None,
    rgba: Optional[List[float]] = None,
    density: float = 500.0,
    friction: Optional[List[float]] = None,
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
        friction: Optional [slide, spin, roll] MuJoCo friction coefficients
    
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
        'friction': friction,
        'density': density,
    }


# =============================================================================
# Environment Step Utilities
# =============================================================================

class ActionPersistenceManager:
    """
    Manages action persistence for different controller types.
    
    For delta controllers: No arm action persistence needed (zero = no movement)
    For absolute controllers: Persist arm actions to prevent abrupt movements on zero input
    
    Follows the same pattern as gripper action persistence but for arm actions.
    """
    
    def __init__(self):
        self.previous_gripper_actions: Dict[str, np.ndarray] = {}
        self.previous_arm_actions: Dict[str, np.ndarray] = {}
    
    def get_controller_info(self, robot) -> tuple:
        """
        Extract controller name and input type from robot's part_controllers.
        
        Args:
            robot: The robot instance
            
        Returns:
            (controller_name, input_type): Tuple of controller configuration
        """
        controller_name = constants.DEFAULT_CONTROLLER_NAME
        input_type = constants.DEFAULT_INPUT_TYPE
        
        try:
            # Get part controllers - they're usually nested in composite_controller
            if hasattr(robot, 'part_controllers') and robot.part_controllers:
                for arm_name, controller in robot.part_controllers.items():
                    if hasattr(controller, 'name'):
                        controller_name = controller.name
                    if hasattr(controller, 'input_type'):
                        input_type = controller.input_type
                    break  # Assume same controller for all arms
        except Exception as e:
            logger.debug(f"Could not extract controller info: {e}")
        
        return controller_name, input_type
    
    def is_absolute_mode(self, robot) -> bool:
        """Check if robot is using absolute input mode."""
        _, input_type = self.get_controller_info(robot)
        return input_type == constants.InputType.ABSOLUTE
    
    def get_arm_action_dim(self, robot, arm_name: str) -> int:
        """
        Get the action dimension for arm control (excluding gripper).
        
        Args:
            robot: The robot instance
            arm_name: Name of the arm
            
        Returns:
            Action dimension for arm control
        """
        controller_name, _ = self.get_controller_info(robot)
        return constants.CONTROLLER_ACTION_DIMS.get(
            controller_name, 
            constants.CONTROLLER_ACTION_DIMS[constants.ControllerName.OSC_POSE]
        )
    
    def read_current_arm_state(self, env: DualArmEnvironment, robot_id: int, arm_name: str) -> np.ndarray:
        """
        Read current arm state from environment for absolute mode persistence.
        
        For OSC_POSE: Returns [x, y, z, rx, ry, rz] in robot base frame
        For JOINT_POSITION: Returns joint positions
        
        Args:
            env: The robosuite environment
            robot_id: ID of the robot
            arm_name: Name of the arm
            
        Returns:
            Current arm state as action format
        """
        robot = env.robots[robot_id]
        controller_name, _ = self.get_controller_info(robot)
        
        if controller_name == constants.ControllerName.JOINT_POSITION:
            # Get joint positions
            return self._read_joint_positions(env, robot, arm_name)
        else:
            # Default to OSC_POSE - get EEF pose
            return self._read_eef_pose(env, robot, arm_name)
    
    def _read_joint_positions(self, env: DualArmEnvironment, robot, arm_name: str) -> np.ndarray:
        """Read current joint positions for an arm."""
        try:
            # Get joint indices for this arm
            if hasattr(robot, f'{arm_name}_arm_joint_indexes'):
                joint_indices = getattr(robot, f'{arm_name}_arm_joint_indexes')
            elif hasattr(robot, 'arm_joint_indexes'):
                joint_indices = robot.arm_joint_indexes
            else:
                joint_indices = robot.joint_indexes
            
            return env.sim.data.qpos[joint_indices].copy()
        except Exception as e:
            logger.warning(f"Could not read joint positions: {e}")
            return constants.FRANKA_HOME_JOINT_POSITION.copy()
    
    def _read_eef_pose(self, env: DualArmEnvironment, robot, arm_name: str) -> np.ndarray:
        """Read current EEF pose as [x, y, z, rx, ry, rz] in base frame."""
        try:
            
            # Get EEF pose in world frame from robot internals
            eef_site_name = robot.robot_model.eef_name.get(arm_name, robot.robot_model.eef_name.get('right'))
            eef_site_id = env.sim.model.site_name2id(eef_site_name)
            eef_pos_world = env.sim.data.site_xpos[eef_site_id].copy()
            eef_rot_mat_world = env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
            
            # Get robot base pose
            base_body_name = robot.robot_model.root_body
            base_body_id = env.sim.model.body_name2id(base_body_name)
            base_pos_world = env.sim.data.body_xpos[base_body_id].copy()
            base_rot_mat_world = env.sim.data.body_xmat[base_body_id].reshape(3, 3)
            
            # Transform EEF to base frame: p_base = R_base^T @ (p_world - base_pos)
            eef_pos_base = base_rot_mat_world.T @ (eef_pos_world - base_pos_world)
            eef_rot_mat_base = base_rot_mat_world.T @ eef_rot_mat_world
            
            # Convert rotation matrix to axis-angle (rotation vector)
            rot_vec = R.from_matrix(eef_rot_mat_base).as_rotvec()
            
            return np.concatenate([eef_pos_base, rot_vec])
        except Exception as e:
            logger.warning(f"Could not read EEF pose: {e}")
            # Return neutral pose
            return np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
    
    def is_zero_action(self, action: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Check if an action is effectively zero."""
        return np.allclose(action, 0.0, atol=tolerance)


# Global action persistence manager
_action_manager = ActionPersistenceManager()


# =============================================================================
# Collision Monitor — checks for collisions after every env_step when enabled
# =============================================================================

class CollisionMonitor:
    """Counts physics-based collision incidents during execution.

    When enabled, after every ``env_step`` the monitor checks MuJoCo contacts
    for inter-arm collisions AND arm-environment collisions (ignoring the
    manipulated object and any configured ``ignore_pairs``).

    Contacts with penetration depth shallower than
    ``min_penetration_depth`` are also excluded — this filters out light
    touches (e.g. grippers barely touching during handover) that do not
    represent dangerous collisions.

    Usage::

        start_collision_monitor("plate_0",
                                ignore_pairs=[("gripper0", "gripper1")])
        # ... execute trajectory ...
        n = stop_collision_monitor()
    """

    def __init__(self):
        self._enabled = False
        self._collision_count = 0
        self._manipulated_object: Optional[str] = None
        self._ignore_pairs: Optional[List[Tuple[str, str]]] = None
        self._min_penetration_depth: float = constants.COLLISION_MONITOR_MIN_DEPTH

    # ── public API ──────────────────────────────────────────────────────

    def start(
        self,
        manipulated_object: Optional[str] = None,
        ignore_pairs: Optional[List[Tuple[str, str]]] = None,
        min_penetration_depth: Optional[float] = None,
    ):
        """Enable monitoring and reset the counter.

        Args:
            manipulated_object: Geom-name prefix to ignore entirely.
            ignore_pairs: List of ``(prefix_a, prefix_b)`` geom-prefix
                pairs whose mutual contacts are ignored.
            min_penetration_depth: Minimum MuJoCo penetration depth
                (negative ``contact.dist``) to count.  ``None`` keeps
                the default from ``constants``.
        """
        self._enabled = True
        self._collision_count = 0
        self._manipulated_object = manipulated_object
        self._ignore_pairs = ignore_pairs
        if min_penetration_depth is not None:
            self._min_penetration_depth = min_penetration_depth
        else:
            self._min_penetration_depth = constants.COLLISION_MONITOR_MIN_DEPTH

    def stop(self) -> int:
        """Disable monitoring and return the accumulated count."""
        self._enabled = False
        count = self._collision_count
        return count

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def collision_count(self) -> int:
        return self._collision_count

    # ── internal ────────────────────────────────────────────────────────

    def check(self, env: DualArmEnvironment):
        """Called by ``env_step`` after each simulation step."""
        if not self._enabled:
            return
        # Deferred import to avoid circular dependencies
        from bimanual_manipulation.planning.utils.collision_utils import (
            in_collision,
        )
        ignore_names = [self._manipulated_object] if self._manipulated_object else None
        if in_collision(
            env,
            ignore_names=ignore_names,
            ignore_pairs=self._ignore_pairs,
            contact_dist_threshold=-self._min_penetration_depth,
        ):
            self._collision_count += 1


# Global collision monitor instance
_collision_monitor = CollisionMonitor()


def start_collision_monitor(
    manipulated_object: Optional[str] = None,
    ignore_pairs: Optional[List[Tuple[str, str]]] = None,
    min_penetration_depth: Optional[float] = None,
):
    """Enable collision monitoring in ``env_step``.

    Args:
        manipulated_object: Geom-name prefix of the grasped object to ignore
            (e.g. ``"plate_0"``, ``"mug_0"``).  Contacts involving geoms
            whose name starts with this prefix are excluded.
        ignore_pairs: List of ``(prefix_a, prefix_b)`` pairs whose 
            mutual contacts are ignored (e.g. gripper-gripper during
            handover).
        min_penetration_depth: Minimum penetration depth (m) to count.
            Defaults to ``constants.COLLISION_MONITOR_MIN_DEPTH``.
    """
    _collision_monitor.start(
        manipulated_object,
        ignore_pairs=ignore_pairs,
        min_penetration_depth=min_penetration_depth,
    )


def stop_collision_monitor() -> int:
    """Disable monitoring and return the number of steps that had collisions."""
    return _collision_monitor.stop()


def get_collision_count() -> int:
    """Return the current collision count without stopping the monitor."""
    return _collision_monitor.collision_count


# =============================================================================
# Object Trajectory Monitor — logs actual vs planned object pose during execution
# =============================================================================

class ObjectTrajectoryMonitor:
    """Tracks the manipulated object's actual trajectory vs the planned one.

    When enabled (via :func:`start_object_monitor`), ``env_step`` calls
    :meth:`check` after every simulation step.  Every
    ``constants.OBJECT_MONITOR_LOG_INTERVAL`` steps the monitor emits a
    DEBUG log with the object's current position and the error relative
    to the nearest planned waypoint.

    On :meth:`stop`, returns the full recorded actual trajectory, the
    planned trajectory, and aggregate error statistics.
    """

    def __init__(self):
        self._enabled = False
        self._object_name: Optional[str] = None
        self._step_count = 0
        self._planned_positions: Optional[np.ndarray] = None  # (N, 3)
        self._actual_positions: List[np.ndarray] = []          # recorded (M, 3)

    def start(
        self,
        object_name: str,
        planned_positions: Optional[np.ndarray] = None,
    ):
        """Enable monitoring for *object_name*.

        Args:
            object_name: MuJoCo object name (e.g. ``"flatPlate"``).
            planned_positions: Optional ``(N, 3)`` array of planned
                object xyz positions.
        """
        self._enabled = True
        self._object_name = object_name
        self._step_count = 0
        self._planned_positions = planned_positions
        self._actual_positions = []

    def stop(self) -> dict:
        """Disable monitoring and return a summary dict.

        Returns a dict with:
            actual_trajectory: (M, 3) recorded positions.
            planned_trajectory: (N, 3) planned positions (or None).
            mean_tracking_error: mean nearest-waypoint error (m).
            max_tracking_error: max nearest-waypoint error (m).
            total_steps: number of env steps recorded.
        """
        self._enabled = False

        actual = np.array(self._actual_positions) if self._actual_positions else np.empty((0, 3))
        planned = self._planned_positions

        mean_err = 0.0
        max_err = 0.0
        if planned is not None and len(planned) > 0 and len(actual) > 0:
            # For each actual position find the nearest planned waypoint
            errors = []
            for pos in actual:
                dists = np.linalg.norm(planned - pos, axis=1)
                errors.append(float(np.min(dists)))
            mean_err = float(np.mean(errors))
            max_err = float(np.max(errors))

        summary = {
            "actual_trajectory": actual,
            "planned_trajectory": planned,
            "mean_tracking_error": mean_err,
            "max_tracking_error": max_err,
            "total_steps": self._step_count,
        }

        if planned is not None and len(planned) > 0 and len(actual) > 0:
            logger.info(
                "Object monitor summary: mean_err=%.4f m, max_err=%.4f m "
                "over %d steps.",
                mean_err, max_err, self._step_count,
            )
        else:
            logger.info(
                "Object monitor summary: %d steps recorded (no planned trajectory for comparison).",
                self._step_count,
            )
        return summary

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(self, env: DualArmEnvironment):
        """Called by ``env_step`` after each simulation step."""
        if not self._enabled or self._object_name is None:
            return
        self._step_count += 1

        pose = env.get_object_pose(self._object_name)
        if pose is None:
            return
        pos = pose[:3, 3].copy()
        self._actual_positions.append(pos)

        # Periodic debug log
        if self._step_count % constants.OBJECT_MONITOR_LOG_INTERVAL == 0:
            msg = (
                "Object '%s' step %d: pos=[%.4f, %.4f, %.4f]"
            )
            args: list = [
                self._object_name, self._step_count,
                pos[0], pos[1], pos[2],
            ]
            if self._planned_positions is not None and len(self._planned_positions) > 0:
                dists = np.linalg.norm(self._planned_positions - pos, axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest_err = float(dists[nearest_idx])
                msg += ", nearest_planned_wp=%d (err=%.4f m)"
                args.extend([nearest_idx, nearest_err])
            logger.debug(msg, *args)


# Global object trajectory monitor instance
_object_monitor = ObjectTrajectoryMonitor()


def start_object_monitor(
    object_name: str,
    planned_positions: Optional[np.ndarray] = None,
):
    """Enable object trajectory monitoring in ``env_step``.

    Args:
        object_name: Name of the manipulated object.
        planned_positions: Optional ``(N, 3)`` planned xyz trajectory.
    """
    _object_monitor.start(object_name, planned_positions)


def stop_object_monitor() -> dict:
    """Disable monitoring and return a summary dict."""
    return _object_monitor.stop()


def env_step(env: DualArmEnvironment, action: Any = None, robot_id: Optional[int] = None, arm: Union[str, List[str], None] = None):
    """Handles full action construction with gripper and arm action persistence.
    
    This function manages action persistence for different controller modes:
    - For delta controllers: No arm persistence needed (zero = no movement)
    - For absolute controllers: Persists arm actions to prevent abrupt movements on zero input
    - For grippers: Always persists state (both delta and absolute modes)
    
    Args:
        env: The robosuite environment (MjSim)
        action: The action vector. Can be:
                - Full environment action vector (size env.action_dim)
                - Single robot action vector (size robot.action_dim), if robot_id is provided
        robot_id: ID of the robot currently being controlled. Required if passing single robot action.
        arm: Name(s) of the arm(s) currently being controlled. Required if robot_id is provided.
             Can be a single string (e.g. "right") or a list of strings (e.g. ["right", "left"]).
    """
    global _action_manager
    
    if robot_id is not None and arm is None:
        raise ValueError("The 'arm' argument must be specified when 'robot_id' is provided.")

    full_action = np.zeros(env.action_dim)
    
    # Phase 1: Place input action into full_action vector
    if action is not None:
        input_action = np.array(action, copy=True)
        is_full_action = (len(input_action) == env.action_dim)
        
        if is_full_action:
            full_action = input_action
        elif robot_id is not None:
            # Construct full action from single robot action
            current_idx = 0
            for i, robot in enumerate(env.robots):
                robot_dim = robot.action_dim
                if i == robot_id:
                    if len(input_action) != robot_dim:
                        raise ValueError(f"Action dimension {len(input_action)} does not match expected dimension {robot_dim} for robot {robot_id}")
                    full_action[current_idx:current_idx + robot_dim] = input_action
                    break
                current_idx += robot_dim
        else:
            raise ValueError("Provided partial action without robot_id, or action dimension mismatch.")

    # Phase 2: Handle arm action persistence for absolute mode controllers
    # Phase 3: Handle gripper action persistence (always)
    current_idx = 0
    for i, robot in enumerate(env.robots):
        robot_dim = robot.action_dim
        is_absolute = _action_manager.is_absolute_mode(robot)
        
        # Determine gripper layout
        gripper_dofs = []
        gripper_arms = []
        for arm_name in robot.arms:
            if arm_name in robot.gripper:
                gripper_dofs.append(robot.gripper[arm_name].dof)
                gripper_arms.append(arm_name)
        
        total_gripper_dim = sum(gripper_dofs)
        arm_action_dim = robot_dim - total_gripper_dim
        
        # Arm action indices in full_action
        arm_start = current_idx
        arm_end = current_idx + arm_action_dim
        
        # Gripper block starts after arm actions
        gripper_block_start = arm_end
        
        # Process arm actions for each arm of this robot
        for arm_name in robot.arms:
            arm_store_key = f"robot_{i}_{arm_name}_arm"
            
            # Determine if this arm should be updated or restored
            should_update_arm = False
            if action is not None:
                if robot_id is not None:
                    if i == robot_id:
                        if isinstance(arm, list):
                            should_update_arm = arm_name in arm
                        else:
                            should_update_arm = (arm == arm_name)
                else:
                    # Full action passed without robot_id -> update everything
                    should_update_arm = True
            
            if is_absolute:
                # Absolute mode: handle arm action persistence
                arm_action = full_action[arm_start:arm_end]
                
                if should_update_arm:
                    # Check if incoming action is zero
                    if _action_manager.is_zero_action(arm_action):
                        # Use stored action or read current state
                        if arm_store_key in _action_manager.previous_arm_actions:
                            full_action[arm_start:arm_end] = _action_manager.previous_arm_actions[arm_store_key]
                        else:
                            # Read current state from environment
                            current_state = _action_manager.read_current_arm_state(env, i, arm_name)
                            full_action[arm_start:arm_end] = current_state
                            _action_manager.previous_arm_actions[arm_store_key] = current_state
                    else:
                        # Non-zero action: update storage
                        _action_manager.previous_arm_actions[arm_store_key] = arm_action.copy()
                else:
                    # Restore from storage for non-controlled arms
                    if arm_store_key in _action_manager.previous_arm_actions:
                        full_action[arm_start:arm_end] = _action_manager.previous_arm_actions[arm_store_key]
                    else:
                        # Initialize with current state
                        current_state = _action_manager.read_current_arm_state(env, i, arm_name)
                        full_action[arm_start:arm_end] = current_state
                        _action_manager.previous_arm_actions[arm_store_key] = current_state
        
        # Process gripper actions
        current_gripper_offset = 0
        for g_idx, g_dof in enumerate(gripper_dofs):
            g_arm_name = gripper_arms[g_idx]
            g_start = gripper_block_start + current_gripper_offset
            g_end = g_start + g_dof
            
            gripper_store_key = f"robot_{i}_{g_arm_name}_gripper"
            
            # Determine if this gripper should be updated or restored
            should_update_gripper = False
            if action is not None:
                if robot_id is not None:
                    if i == robot_id:
                        if isinstance(arm, list):
                            should_update_gripper = g_arm_name in arm
                        else:
                            should_update_gripper = (arm == g_arm_name)
                else:
                    should_update_gripper = True
            
            if should_update_gripper:
                # Update storage from input
                _action_manager.previous_gripper_actions[gripper_store_key] = full_action[g_start:g_end].copy()
            else:
                # Restore from storage
                if gripper_store_key in _action_manager.previous_gripper_actions:
                    full_action[g_start:g_end] = _action_manager.previous_gripper_actions[gripper_store_key]
                else:
                    # Initialize storage
                    _action_manager.previous_gripper_actions[gripper_store_key] = full_action[g_start:g_end].copy()
            
            current_gripper_offset += g_dof

        current_idx += robot_dim

    obs = env.step(full_action)
    apply_suction_weld(env, full_action)

    # Collision monitoring (no-op when monitor is disabled)
    _collision_monitor.check(env)

    # Object trajectory monitoring (no-op when monitor is disabled)
    _object_monitor.check(env)

    return obs


def reset_action_persistence():
    """Reset the action persistence manager state. Call after env.reset()."""
    global _action_manager
    _action_manager = ActionPersistenceManager()


def get_action_persistence_manager() -> ActionPersistenceManager:
    """Get the global action persistence manager for inspection/debugging."""
    return _action_manager


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
