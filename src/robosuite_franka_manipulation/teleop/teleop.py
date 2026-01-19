"""
Teleoperation module for dual-arm Franka robots in robosuite.
Provides keyboard-based teleoperation using robosuite's Keyboard device.
"""

from contextlib import nullcontext
import logging
import threading
from copy import deepcopy
from typing import Any, Optional
from threading import Lock
from pynput import keyboard as pynput_keyboard

from robosuite.devices import Keyboard
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils import env_step

logger = logging.getLogger(__name__)

class KeyboardTeleop:
    """
    Keyboard teleoperation for dual-arm manipulation using robosuite's Keyboard device.
    """
    def __init__(self, env: DualArmEnvironment, pos_sensitivity: Optional[float] = constants.DEFAULT_POS_SENSITIVITY, rot_sensitivity: Optional[float] = constants.DEFAULT_ROT_SENSITIVITY, env_lock: Lock = None) -> None:
        self.env = env
        self.env_lock = env_lock
        self.device = Keyboard(
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
        self.running = False
        self.exit_pressed = False
        self.step_event = threading.Event()
        
        # Setup keyboard listener for exit key and step triggering
        self.listener = pynput_keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

    def _on_key_press(self, key: Any) -> None:
        """
        Callback for keyboard listener to detect exit key and trigger step.
        """
        try:
            if hasattr(key, 'char') and key.char == 'q':
                self.exit_pressed = True
        except AttributeError:
            pass
        
        # Trigger a control step
        if self.running:
            self.step_event.set()

    def _render_loop(self) -> None:
        """Continuously render the environment."""
        while self.running:
            with self.env_lock if self.env_lock else nullcontext():
                env_step(self.env)
                self.env.render()

    def run(self) -> None:
        """Run teleoperation loop. If env_lock is provided, guard env access with it."""
        self.running = True
        logger.info("Starting teleoperation with robosuite Keyboard device.")
        logger.info("Controls: Follow robosuite's keyboard controls.")
        logger.info("Press 'q' key to exit teleoperation.")

        with self.env_lock if self.env_lock else nullcontext():
            self.env.reset()

        self.device.start_control()

        while self.running:
            self.step_event.clear()

            active_robot_idx = self.device.active_robot
            active_robot = self.env.robots[active_robot_idx]

            # Sync device grasp_states with previous environment steps
            if hasattr(env_step, "previous_gripper_actions"):
                for i, arm in enumerate(active_robot.arms):
                    store_key = f"robot_{active_robot_idx}_{arm}"
                    if store_key in env_step.previous_gripper_actions:
                        prev_val = env_step.previous_gripper_actions[store_key]
                        # Handle both scalar and array-like gripper states
                        if hasattr(prev_val, "__len__"):
                            prev_val = prev_val[0]
                        
                        # Use nested indexing: [robot_idx][arm_idx]
                        if active_robot_idx < len(self.device.grasp_states):
                            if i < len(self.device.grasp_states[active_robot_idx]):
                                self.device.grasp_states[active_robot_idx][i] = True if prev_val > 0 else False

            input_ac_dict = self.device.input2action(mirror_actions=False)

            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)
            for arm in active_robot.arms:
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]

            # Construct only the active robot's action vector
            robot_action = active_robot.create_action_vector(action_dict)

            with self.env_lock if self.env_lock else nullcontext():
                env_step(self.env, robot_action, robot_id=self.device.active_robot, arm=active_robot.arms)

            if self.exit_pressed:
                self.running = False
                break

        self.listener.stop()
        self.device.listener.stop()
        logger.info("Teleoperation ended.")