"""
Teleoperate robot with keyboard.
This is a simplified version of the demo_device_control.py script, tailored for the dual-arm handoff environment.
"""

import argparse
import time

import numpy as np
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale rotation user inputs",
    )
    args = parser.parse_args()

    from robosuite.controllers import load_part_controller_config
    # load the OSC_POSE part controller config
    osc_pose_config = load_part_controller_config(default_controller="OSC_POSE")
    osc_pose_config["gripper"] = {"type": "GRIP"}

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

    # Create environment
    env = suite.make(
        "TwoArmHandover",
        robots=["Panda", "Panda"],
        env_configuration="opposed",
        controller_configs=controller_configs,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
    else:
        raise Exception(
            "Invalid device choice: this script only supports 'keyboard'."
        )

    while True:
        # Reset the environment
        obs = env.reset()
        env.render()

        # Initialize device control
        device.start_control()
        
        # Initialize variables that should be maintained between resets
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        # Loop until we get a reset from the input or the task completes
        while True:
            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action(mirror_actions=False)

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break
            
            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)
            # set arm actions
            for arm in active_robot.arms:
                # Assuming OSC_POSE is always used here, so we always take delta actions
                controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            # Step through the environment
            env.step(env_action)
            env.render()
