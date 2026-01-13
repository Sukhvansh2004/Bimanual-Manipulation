import logging
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Union

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)


class DualArmEnvironmentWrapper:
    """
    Wrapper for robosuite dual-arm environment with realistic camera intrinsics.
    Automatically sets camera intrinsics on creation and after reset.
    """
    
    def __init__(self, env: Union[MujocoEnv, Wrapper]):
        self._env = env
        self._set_intrinsics()
    
    def _set_intrinsics(self) -> None:
        """Set realistic camera intrinsics for all cameras."""
        # robosuite dynamic attributes not in type stubs
        for i in range(len(self._env.camera_names)):
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self._env.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)
    
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
        if name == '_env':
            super().__setattr__(name, value)
        else:
            setattr(self._env, name, value)


class CustomWorldEnvironment(MujocoEnv):
    """
    A custom MuJoCo environment that loads from a user-specified XML world file.
    
    This environment can load any MuJoCo XML file from the worlds folder and provides
    basic simulation capabilities (stepping, rendering, etc.) without requiring
    robosuite robot controllers.
    """
    
    def __init__(
        self,
        world_xml_path: str,
        has_renderer: bool = True,
        has_offscreen_renderer: bool = True,
        render_camera: str = "frontview",
        use_camera_obs: bool = True,
        camera_names: Union[str, List[str]] = "agentview",
        camera_heights: Union[int, List[int]] = 480,
        camera_widths: Union[int, List[int]] = 640,
        camera_depths: bool = True,
        control_freq: int = 20,
        horizon: int = 1000,
        ignore_done: bool = True,
        renderer: str = "mjviewer",
    ):
        """
        Initialize a custom world environment.
        
        Args:
            world_xml_path: Path to the MuJoCo XML world file (relative to worlds folder or absolute)
            has_renderer: Whether to enable onscreen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            render_camera: Camera name for rendering
            use_camera_obs: Whether to use camera observations
            camera_names: Name(s) of camera(s) for observations
            camera_heights: Height(s) of camera images
            camera_widths: Width(s) of camera images  
            camera_depths: Whether to include depth images
            control_freq: Control frequency in Hz
            horizon: Maximum episode length
            ignore_done: Whether to ignore episode termination
            renderer: Renderer type ('mjviewer' or 'mujoco')
        """
        self.world_xml_path = self._resolve_world_path(world_xml_path)
        
        # Ensure camera_names and dimensions are lists
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        if isinstance(camera_heights, int):
            camera_heights = [camera_heights] * len(camera_names)
        if isinstance(camera_widths, int):
            camera_widths = [camera_widths] * len(camera_names)
            
        # Store for property access
        self._camera_names = camera_names
        self._camera_heights = camera_heights
        self._camera_widths = camera_widths
        self._camera_depths = camera_depths
        self._use_camera_obs = use_camera_obs
        
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            renderer=renderer,
        )
    
    def _resolve_world_path(self, world_xml_path: str) -> Path:
        """Resolve the world XML path to an absolute path."""
        path = Path(world_xml_path)
        
        # If already absolute and exists, use it
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to worlds directory
        worlds_path = constants.WORLDS_DIR / world_xml_path
        if worlds_path.exists():
            return worlds_path
        
        # Try as-is (might be a relative path from cwd)
        if path.exists():
            return path.absolute()
        
        raise FileNotFoundError(
            f"Could not find world XML file: {world_xml_path}\n"
            f"Searched in: {constants.WORLDS_DIR}, and current directory"
        )
    
    def _load_model(self):
        """Load the MuJoCo model from the world XML file."""
        from robosuite.models import MujocoWorldBase
        
        # Load base world and merge with our custom XML
        self.model = MujocoWorldBase()
        
        # Load and merge the custom XML
        self._load_custom_world()
        
    def _load_custom_world(self):
        """Load the custom world XML and merge with base model."""
        # Read the XML file
        with open(self.world_xml_path, 'r') as f:
            custom_xml = f.read()
        
        # Merge into the world model by adding to the worldbody
        # We'll use a simple approach: load the XML and add its contents
        logger.info(f"Loading custom world from: {self.world_xml_path}")
        self.model.merge_assets(self._parse_xml_assets(custom_xml))
        self.model.worldbody.append(self._parse_xml_worldbody(custom_xml))
    
    def _parse_xml_assets(self, xml_str: str) -> Any:
        """Parse assets from XML string (placeholder - override if needed)."""
        # This is a simplified implementation
        # For complex assets, you may need to parse and add them properly
        return None
    
    def _parse_xml_worldbody(self, xml_str: str) -> Any:
        """Parse worldbody from XML string (placeholder - override if needed)."""
        # This is a simplified implementation
        return None

    def _setup_references(self):
        """Set up references to important objects in the simulation."""
        super()._setup_references()
        # Additional setup can be done here
    
    def _setup_observables(self):
        """Set up observable functions."""
        super()._setup_observables()
    
    def _reset_internal(self):
        """Reset internal state."""
        super()._reset_internal()
    
    def reward(self, action=None):
        """Calculate reward (returns 0 for basic environment)."""
        return 0.0
    
    def _check_success(self):
        """Check if task succeeded (always False for basic environment)."""
        return False
    
    @property
    def action_spec(self):
        """Return action specification."""
        low = np.array([-1.0])
        high = np.array([1.0])
        return low, high
    
    @property
    def action_dim(self):
        """Return action dimension."""
        return 1


def make_custom_world_env(
    world_xml_path: str
) -> "DualArmEnvironmentWrapper":
    """
    Create an environment from a custom MuJoCo world XML file.
    
    This is a simpler factory function that creates a basic MuJoCo environment
    from a custom XML file. For full robosuite dual-arm functionality with 
    controllers, use make_dual_arm_env instead.
    
    Args:
        world_xml_path: Path to the world XML file (relative to worlds folder or absolute)
        has_renderer: Enable onscreen rendering
        has_offscreen_renderer: Enable offscreen rendering  
        use_camera_obs: Include camera observations
        camera_names: Camera name(s) for observations
        camera_heights: Height(s) of camera images
        camera_widths: Width(s) of camera images
        camera_depths: Include depth images
        control_freq: Control frequency in Hz
        horizon: Maximum episode length
        
    Returns:
        DualArmEnvironmentWrapper wrapping the custom environment
    """
    env = CustomWorldEnvironment(
        world_xml_path=world_xml_path,
        has_renderer= constants.DEFAULT_HAS_RENDERER,
        has_offscreen_renderer= constants.DEFAULT_HAS_OFFSCREEN_RENDERER,
        use_camera_obs= constants.DEFAULT_USE_CAMERA_OBS,
        camera_names= constants.DEFAULT_CAMERA_NAME,
        camera_heights= constants.DEFAULT_CAMERA_HEIGHT,
        camera_widths= constants.DEFAULT_CAMERA_WIDTH,
        camera_depths= constants.DEFAULT_CAMERA_DEPTH,
        control_freq= constants.DEFAULT_CONTROL_FREQ,
        horizon= constants.DEFAULT_HORIZON,
    )
    return DualArmEnvironmentWrapper(env)


def make_dual_arm_env() -> "DualArmEnvironmentWrapper":
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    Automatically sets realistic camera intrinsics.
    
    Returns:
        DualArmEnvironmentWrapper wrapping the configured environment
    """
    # load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller=constants.DEFAULT_CONTROLLER_NAME)
    osc_pose_config["gripper"] = {"type": constants.DEFAULT_GRIPPER_CONTROLLER_TYPE}
    osc_pose_config["input_ref_frame"] = constants.DEFAULT_INPUT_REF_FRAME
    osc_pose_config["input_type"] = constants.DEFAULT_INPUT_TYPE

    # create the composite controller config
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

    env = suite.make(
        constants.DEFAULT_ENV_NAME,
        robots=constants.DEFAULT_ROBOT_TYPES,
        gripper_types=constants.DEFAULT_GRIPPER_TYPES_ENV,
        controller_configs=controller_configs,
        env_configuration=constants.DEFAULT_ENV_CONFIG,
        has_renderer=constants.DEFAULT_HAS_RENDERER,
        has_offscreen_renderer=constants.DEFAULT_HAS_OFFSCREEN_RENDERER,
        use_camera_obs=constants.DEFAULT_USE_CAMERA_OBS,
        camera_names=constants.DEFAULT_CAMERA_NAME,
        camera_heights=constants.DEFAULT_CAMERA_HEIGHT,
        camera_widths=constants.DEFAULT_CAMERA_WIDTH,
        camera_depths=constants.DEFAULT_CAMERA_DEPTH,
        control_freq=constants.DEFAULT_CONTROL_FREQ,
        horizon=constants.DEFAULT_HORIZON,
        ignore_done=constants.DEFAULT_IGNORE_DONE,
        reward_shaping=constants.DEFAULT_REWARD_SHAPING,
    )
    
    # Wrap environment to handle camera intrinsics automatically
    return DualArmEnvironmentWrapper(env)

if __name__ == '__main__':
    try:
        env = make_dual_arm_env()
        logger.info("Dual-arm handoff environment created successfully (headless).")

        # Reset and step through a few times to test
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
            env.render()
        logger.info(f"Environment stepped for {env.horizon} steps.")
        env.close()

    except Exception as e:
        logger.error(f"Failed to create or step dual-arm handoff environment: {e}")