import logging
import numpy as np
from typing import Any, Tuple, Union

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
        for i in range(len(self._env.camera_names)):  # type: ignore[attr-defined]
            fx = constants.DEFAULT_CAMERA_INTRINSICS[0, 0]
            fy = constants.DEFAULT_CAMERA_INTRINSICS[1, 1]
            cx = constants.DEFAULT_CAMERA_INTRINSICS[0, 2]
            cy = constants.DEFAULT_CAMERA_INTRINSICS[1, 2]
            self._env.sim.model.cam_intrinsic[i] = np.array([fx, fy, cx, cy], dtype=np.float32)  # type: ignore[attr-defined]
    
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

def make_dual_arm_env() -> "DualArmEnvironmentWrapper":
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    Automatically sets realistic camera intrinsics.
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
        for _ in range(env.horizon):  # type: ignore[arg-type]
            action = env.action_space.sample()  # type: ignore[attr-defined]
            obs, reward, done, info = env.step(action)
            if done:
                break
            env.render()
        logger.info(f"Environment stepped for {env.horizon} steps.")
        env.close()

    except Exception as e:
        logger.error(f"Failed to create or step dual-arm handoff environment: {e}")