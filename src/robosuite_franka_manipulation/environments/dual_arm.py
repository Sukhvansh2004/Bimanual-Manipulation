import robosuite as suite
from robosuite.controllers import load_part_controller_config
from typing import Any

from robosuite_franka_manipulation import constants

def make_dual_arm_env() -> Any:
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
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