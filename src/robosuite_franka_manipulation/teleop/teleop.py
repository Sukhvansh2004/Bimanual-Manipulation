"""
Teleoperation module for dual-arm Franka robots in robosuite.
Provides keyboard-based teleoperation using robosuite's Keyboard device.
"""

import numpy as np
from copy import deepcopy
from pynput import keyboard as pynput_keyboard

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
        self.exit_pressed = False
        
        # Setup keyboard listener for exit key
        self.listener = pynput_keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

    def _on_key_press(self, key):
        """
        Callback for keyboard listener to detect exit key.
        """
        try:
            if hasattr(key, 'char') and key.char == 'a':
                self.exit_pressed = True
                self.running = False
        except AttributeError:
            pass

    def run(self, env_lock=None):
        """Run teleoperation loop. If env_lock is provided, guard env access with it."""
        self.running = True
        print("Starting teleoperation with robosuite Keyboard device.")
        print("Controls: Follow robosuite's keyboard controls.")
        print("Press 'a' key to exit teleoperation.")

        while self.running:
            if env_lock:
                env_lock.acquire()
            try:
                obs = self.env.reset()
                self.env.render()
            finally:
                if env_lock:
                    env_lock.release()

            self.device.start_control()

            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in self.env.robots
            ]

            while True:
                active_robot = self.env.robots[self.device.active_robot]
                input_ac_dict = self.device.input2action(mirror_actions=False)

                if input_ac_dict is None:
                    break

                action_dict = deepcopy(input_ac_dict)
                for arm in active_robot.arms:
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]

                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(self.env.robots)]
                env_action[self.device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)

                for gripper_ac in all_prev_gripper_actions[self.device.active_robot]:
                    all_prev_gripper_actions[self.device.active_robot][gripper_ac] = action_dict[gripper_ac]

                if env_lock:
                    env_lock.acquire()
                try:
                    self.env.step(env_action)
                    self.env.render()
                finally:
                    if env_lock:
                        env_lock.release()

                if self.exit_pressed:
                    self.running = False
                    break

        self.listener.stop()
        print("Teleoperation ended.")