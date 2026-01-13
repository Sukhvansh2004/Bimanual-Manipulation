import logging
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Union

import robosuite as suite
from robosuite.controllers import load_part_controller_config
import mujoco

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)


class DualArmEnvironmentWrapper:
    """
    Wrapper for robosuite dual-arm environment with realistic camera intrinsics.
    Automatically sets camera intrinsics on creation and after reset.
    """
    
    def __init__(self, env: Any):
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


class CustomWorldEnvironment:
    """
    A custom MuJoCo environment that loads from a user-specified XML world file.
    
    This environment can load any MuJoCo XML file from the worlds folder and provides
    basic simulation capabilities (stepping, rendering, etc.) without requiring
    robosuite robot controllers.
    
    This is a standalone environment that doesn't inherit from robosuite's MujocoEnv,
    allowing direct loading of custom MuJoCo XML files.
    """
    
    def __init__(
        self,
        world_xml_path: str,
        has_renderer: bool = True,
        render_camera: str = "agentview",
        camera_names: Union[str, List[str]] = "agentview",
        camera_height: int = 480,
        camera_width: int = 640,
        control_freq: int = 20,
        horizon: int = 1000,
    ):
        """
        Initialize a custom world environment.
        
        Args:
            world_xml_path: Path to the MuJoCo XML world file (relative to worlds folder or absolute)
            has_renderer: Whether to enable onscreen rendering
            render_camera: Camera name for rendering
            camera_names: Name(s) of camera(s) for observations
            camera_height: Height of camera images
            camera_width: Width of camera images
            control_freq: Control frequency in Hz
            horizon: Maximum episode length
        """
        import mujoco
        import mujoco.viewer
        
        self.world_xml_path = self._resolve_world_path(world_xml_path)
        self.has_renderer = has_renderer
        self.render_camera = render_camera
        self.control_freq = control_freq
        self.horizon = horizon
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # Ensure camera_names is a list
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        self.camera_names = camera_names
        
        # Load the MuJoCo model
        logger.info(f"Loading custom world from: {self.world_xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.world_xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Viewer for rendering
        self._viewer = None
        self._step_count = 0
        
        # Compute timestep
        self.timestep = self.model.opt.timestep
        self.steps_per_action = int(1.0 / (self.control_freq * self.timestep))
        
        logger.info(f"Custom world environment initialized:")
        logger.info(f"  - Bodies: {self.model.nbody}")
        logger.info(f"  - Geoms: {self.model.ngeom}")
        logger.info(f"  - Cameras: {self.model.ncam}")
        logger.info(f"  - Control freq: {self.control_freq} Hz")
        logger.info(f"  - Timestep: {self.timestep} s")
    
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
    
    def reset(self) -> dict:
        """Reset the environment."""
        import mujoco
        
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        
        # Step once to settle
        mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()
    
    def step(self, action: np.ndarray = None):
        """
        Step the environment.
        
        Args:
            action: Control action (optional, can be None for passive simulation)
            
        Returns:
            obs: Observation dictionary
            reward: Reward (always 0 for this basic environment)
            done: Whether episode is done
            info: Additional info dictionary
        """
        
        # Apply action if provided and we have actuators
        if action is not None and self.model.nu > 0:
            self.data.ctrl[:] = action[:self.model.nu]
        
        # Step simulation
        for _ in range(self.steps_per_action):
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        obs = self._get_obs()
        reward = 0.0
        done = self._step_count >= self.horizon
        info = {}
        
        return obs, reward, done, info
    
    def _get_obs(self) -> dict:
        """Get observation dictionary."""
        obs = {
            "time": np.array([self.data.time]),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }
        return obs
    
    def render(self):
        """Render the environment."""
        import mujoco.viewer
        
        if self.has_renderer:
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
    
    def close(self):
        """Close the environment."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    @property
    def action_dim(self):
        """Return action dimension (number of actuators)."""
        return max(1, self.model.nu)
    
    @property
    def action_spec(self):
        """Return action specification."""
        low = np.full(self.action_dim, -1.0)
        high = np.full(self.action_dim, 1.0)
        return low, high
    
    @property
    def name(self):
        """Return environment name."""
        return "CustomWorld"


def make_custom_world_env(
    world_xml_path: str,
    has_renderer: bool = constants.DEFAULT_HAS_RENDERER,
    render_camera: str = constants.DEFAULT_CAMERA_NAME,
    camera_names: Union[str, List[str]] = constants.DEFAULT_CAMERA_NAME,
    camera_height: int = constants.DEFAULT_CAMERA_HEIGHT,
    camera_width: int = constants.DEFAULT_CAMERA_WIDTH,
    control_freq: int = 20,
    horizon: int = 1000,
) -> CustomWorldEnvironment:
    """
    Create an environment from a custom MuJoCo world XML file.
    
    This is a simpler factory function that creates a basic MuJoCo environment
    from a custom XML file. This environment directly loads the MuJoCo XML
    without using robosuite's environment framework.
    
    For full robosuite dual-arm functionality with controllers, use 
    make_dual_arm_env instead.
    
    Args:
        world_xml_path: Path to the world XML file (relative to worlds folder or absolute)
        has_renderer: Enable onscreen rendering
        render_camera: Camera name for rendering
        camera_names: Camera name(s) for observations
        camera_height: Height of camera images
        camera_width: Width of camera images
        control_freq: Control frequency in Hz
        horizon: Maximum episode length
        
    Returns:
        CustomWorldEnvironment instance
    """
    env = CustomWorldEnvironment(
        world_xml_path=world_xml_path,
        has_renderer=has_renderer,
        render_camera=render_camera,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        control_freq=control_freq,
        horizon=horizon,
    )
    return env


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