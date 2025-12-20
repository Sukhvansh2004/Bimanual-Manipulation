import robosuite as suite
from robosuite.controllers import load_part_controller_config
import numpy as np
from typing import Any

from robosuite_franka_manipulation.constants import (
    DEFAULT_ENV_NAME, DEFAULT_ROBOT_TYPES, DEFAULT_GRIPPER_TYPES_ENV,
    DEFAULT_ENV_CONFIG, DEFAULT_HAS_RENDERER, DEFAULT_HAS_OFFSCREEN_RENDERER,
    DEFAULT_USE_CAMERA_OBS, DEFAULT_CAMERA_NAME, DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_DEPTH, DEFAULT_CONTROL_FREQ,
    DEFAULT_HORIZON, DEFAULT_REWARD_SHAPING, DEFAULT_IGNORE_DONE,
    DEFAULT_CONTROLLER_TYPE, DEFAULT_CONTROLLER_NAME, DEFAULT_GRIPPER_CONTROLLER_TYPE,
    DEFAULT_INPUT_REF_FRAME, DEFAULT_INPUT_TYPE
)

def make_dual_arm_env() -> Any:
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    """
    # load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller=DEFAULT_CONTROLLER_NAME)
    osc_pose_config["gripper"] = {"type": DEFAULT_GRIPPER_CONTROLLER_TYPE}
    osc_pose_config["input_ref_frame"] = DEFAULT_INPUT_REF_FRAME
    osc_pose_config["input_type"] = DEFAULT_INPUT_TYPE

    # create the composite controller config
    controller_config = {
        "type": DEFAULT_CONTROLLER_TYPE,
        "body_parts": {
            "right": osc_pose_config,
            "right_gripper": {
                "type": DEFAULT_GRIPPER_CONTROLLER_TYPE
            }
        }
    }
    
    # For dual-arm, controller_configs should be a list of configs
    controller_configs = [controller_config, controller_config]

    env = suite.make(
        DEFAULT_ENV_NAME,
        robots=DEFAULT_ROBOT_TYPES,
        gripper_types=DEFAULT_GRIPPER_TYPES_ENV,
        controller_configs=controller_configs,
        env_configuration=DEFAULT_ENV_CONFIG,
        has_renderer=DEFAULT_HAS_RENDERER,
        has_offscreen_renderer=DEFAULT_HAS_OFFSCREEN_RENDERER,
        use_camera_obs=DEFAULT_USE_CAMERA_OBS,
        camera_names=DEFAULT_CAMERA_NAME,
        camera_heights=DEFAULT_CAMERA_HEIGHT,
        camera_widths=DEFAULT_CAMERA_WIDTH,
        camera_depths=DEFAULT_CAMERA_DEPTH,
        control_freq=DEFAULT_CONTROL_FREQ,
        horizon=DEFAULT_HORIZON,
        ignore_done=DEFAULT_IGNORE_DONE,
        reward_shaping=DEFAULT_REWARD_SHAPING,
    )
    return env

if __name__ == '__main__':
    try:
        env = make_dual_arm_env()
        print("Dual-arm handoff environment created successfully (headless).")

        # Reset and step through a few times to test
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, info = env.step(action)
            if done:
                break
            env.render()
        print(f"Environment stepped for {env.horizon} steps.")
        env.close()

    except Exception as e:
        print(f"Failed to create or step dual-arm handoff environment: {e}")