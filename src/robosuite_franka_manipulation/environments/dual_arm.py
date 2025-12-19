import robosuite as suite
from robosuite.controllers import load_part_controller_config
import numpy as np
from typing import Any

def make_dual_arm_env() -> Any:
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    """
    # load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller="OSC_POSE")
    osc_pose_config["gripper"] = {"type": "GRIP"}
    osc_pose_config["input_ref_frame"] = "base"
    osc_pose_config["input_type"] = "delta"

    # create the composite controller config
    controller_config = {
        "type": "BASIC",
        "body_parts": {
            "right": osc_pose_config,
            "right_gripper": {
                "type": "GRIP"
            }
        }
    }
    
    # For dual-arm, controller_configs should be a list of configs
    controller_configs = [controller_config, controller_config]

    env = suite.make(
        "TwoArmHandover",            # Use the TwoArmHandover environment
        robots=["Panda", "Panda"],   # Specify two Panda robots
        gripper_types="default",     # Use default grippers for each robot
        controller_configs=controller_configs,
        env_configuration="opposed", # Robots face each other
        has_renderer=True,          # We need to render for teleoperation
        has_offscreen_renderer=True, # Enable offscreen rendering for camera observations
        use_camera_obs=True,         # Enable camera observations for perception
        camera_names="agentview", # Specify cameras to use
        camera_heights=128,          # Smaller height for faster rendering
        camera_widths=128,           # Smaller width for faster rendering
        camera_depths=True,          # Request depth observations
        control_freq=20,             # Control frequency
        horizon=1000,                # Longer horizon for sequence
        ignore_done=True,            # Keep environment alive
        reward_shaping=False,        # Disable reward shaping
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