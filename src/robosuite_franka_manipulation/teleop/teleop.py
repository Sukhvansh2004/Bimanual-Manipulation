"""
Teleoperation module for dual-arm Franka robots in robosuite.
Provides keyboard-based teleoperation using robosuite's Keyboard device.
"""

import numpy as np
from copy import deepcopy
import keyboard

from robosuite.devices import Keyboard

from robosuite_franka_manipulation import constants

class KeyboardTeleop:
    """
    Keyboard teleoperation for dual-arm manipulation using robosuite's Keyboard device.
    """
    def __init__(self, env, pos_sensitivity=None, rot_sensitivity=None):
        if pos_sensitivity is None:
            pos_sensitivity = constants.DEFAULT_POS_SENSITIVITY
        if rot_sensitivity is None:
            rot_sensitivity = constants.DEFAULT_ROT_SENSITIVITY
        self.env = env
        self.device = Keyboard(
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
        self.running = False

    def run(self):
        """
        Run teleoperation loop similar to robosuite's teleop script.
        """
        self.running = True
        print("Starting teleoperation with robosuite Keyboard device.")
        print("Controls: Follow robosuite's keyboard controls.")
        print("Press 'reset' or appropriate key to exit teleop.")

        while self.running:
            # Reset the environment
            obs = self.env.reset()
            self.env.render()

            # Initialize device control
            self.device.start_control()

            # Initialize variables that should be maintained between resets
            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in self.env.robots
            ]

            # Loop until we get a reset from the input or the task completes
            while True:
                # Set active robot
                active_robot = self.env.robots[self.device.active_robot]

                # Get the newest action
                input_ac_dict = self.device.input2action(mirror_actions=False)

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    break

                action_dict = deepcopy(input_ac_dict)
                # set arm actions
                for arm in active_robot.arms:
                    # OSC controller is now set to delta mode for Keyboard
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(self.env.robots)]
                env_action[self.device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)

                # Update gripper state for the active robot from the current action
                for gripper_ac in all_prev_gripper_actions[self.device.active_robot]:
                    all_prev_gripper_actions[self.device.active_robot][gripper_ac] = action_dict[gripper_ac]

                # Step through the environment
                self.env.step(env_action)
                self.env.render()

                # Check for switch back to auto
                if keyboard.is_pressed('a'):
                    self.running = False
                    break

        print("Teleoperation ended.")