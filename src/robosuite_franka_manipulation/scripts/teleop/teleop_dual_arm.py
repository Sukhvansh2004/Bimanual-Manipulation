"""
Teleoperate robot with keyboard.
This script is now modular and uses the custom environment factory.
"""

import argparse
import time
import numpy as np
from copy import deepcopy

import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

# Import the custom environment factory
from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    args = parser.parse_args()

    # Create environment using the custom factory function
    env = make_dual_arm_handoff_env()

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env)

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(
        env=env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    env.viewer.add_keypress_callback(device.on_press)

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
            
            action_dict = deepcopy(input_ac_dict)
            # set arm actions
            for arm in active_robot.arms:
                # OSC controller is now set to delta mode for Keyboard
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            
            # Update gripper state for the active robot from the current action
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            # Step through the environment
            env.step(env_action)
            env.render()