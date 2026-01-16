"""
Teleoperation module for dual-arm Franka robots in robosuite.
Provides keyboard-based teleoperation using robosuite's Keyboard device.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Any, Optional
from threading import Lock
from pynput import keyboard as pynput_keyboard
from scipy.spatial.transform import Rotation as R

from robosuite.devices import Keyboard
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils import safe_env_step

logger = logging.getLogger(__name__)

class KeyboardTeleop:
    """
    Keyboard teleoperation for dual-arm manipulation using robosuite's Keyboard device.
    """
    def __init__(self, env: DualArmEnvironment, pos_sensitivity: Optional[float] = None, rot_sensitivity: Optional[float] = None) -> None:
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

    def _on_key_press(self, key: Any) -> None:
        """
        Callback for keyboard listener to detect exit key.
        """
        try:
            if key.char == 'q':
                self.exit_pressed = True
        except AttributeError:
            pass

    def run(self, env_lock: Optional[Lock] = None) -> None:
        """Run teleoperation loop. If env_lock is provided, guard env access with it."""
        self.running = True
        logger.info("Starting teleoperation with robosuite Keyboard device.")
        logger.info("Controls: Follow robosuite's keyboard controls.")
        logger.info("Press 'q' key to exit teleoperation.")

        while self.running:
            if env_lock:
                env_lock.acquire()
            try:
                self.env.reset()
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
                
                # Get the currently active arm from the device
                active_arm = self.device.active_arm
                
                # For absolute input mode:
                # - Active arm: use the pre-computed absolute action from device
                # - Inactive arms: use their current achieved pose to keep them stationary
                for arm in active_robot.arms:
                    if arm == active_arm:
                        # Use the device's computed absolute action for the active arm
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        # For inactive arms, compute absolute pose from current achieved position
                        # This prevents them from jumping back to some internal goal state
                        eef_body = active_robot.robot_model.eef_name[arm]
                        pose4 = active_robot.pose_in_base_from_name(eef_body)
                        current_pos = pose4[:3, 3].copy()
                        current_ori_axisangle = R.from_matrix(pose4[:3, :3]).as_rotvec()
                        action_dict[arm] = np.concatenate([current_pos, current_ori_axisangle])

                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(self.env.robots)]
                env_action[self.device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)

                for gripper_ac in all_prev_gripper_actions[self.device.active_robot]:
                    all_prev_gripper_actions[self.device.active_robot][gripper_ac] = action_dict[gripper_ac]

                if env_lock:
                    env_lock.acquire()
                try:
                    # Use safe wrapper to mitigate intermittent RuntimeError from robosuite internals
                    safe_env_step(self.env, env_action)
                    self.env.render()
                finally:
                    if env_lock:
                        env_lock.release()

                if self.exit_pressed:
                    self.running = False
                    break

        self.listener.stop()
        self.device.listener.stop()
        logger.info("Teleoperation ended.")