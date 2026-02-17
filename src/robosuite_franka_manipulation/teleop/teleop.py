"""
Teleoperation module for dual-arm Franka robots in robosuite.
Provides keyboard-based teleoperation using robosuite's Keyboard device.
"""

from contextlib import nullcontext
import logging
import threading
from copy import deepcopy
from typing import Any, Optional
from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Key
import numpy as np

from robosuite.devices import Keyboard
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils import env_step, get_action_persistence_manager, reset_action_persistence

import time

logger = logging.getLogger(__name__)

class KeyboardOverride(Keyboard):
    """
    A subclass of robosuite's Keyboard that explicitly ignores the Space key.
    This prevents robosuite from reacting to Space (e.g., resetting env),
    allowing the Teleop class to handle Space exclusively.
    """
    def on_press(self, key):
        if key == Key.space:
            return  # Do nothing if space is pressed
        
        # Pass all other keys to the original robosuite handler
        super().on_press(key)

class KeyboardTeleop:
    """
    Keyboard teleoperation for dual-arm manipulation using robosuite's Keyboard device.
    """
    def __init__(self, env: DualArmEnvironment, pos_sensitivity: Optional[float] = constants.DEFAULT_POS_SENSITIVITY, rot_sensitivity: Optional[float] = constants.DEFAULT_ROT_SENSITIVITY, env_lock: threading.RLock = None) -> None:
        self.env = env
        self.env_lock = env_lock
        self.device = KeyboardOverride(
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
        self.running = False
        self.exit_pressed = False
        self.gripper_toggle = threading.Event()

        # Detect controller type for each robot (assume same across arms)
        self._robot_controller_info = {}
        for ridx, robot in enumerate(env.robots):
            ctrl_name = constants.DEFAULT_CONTROLLER_NAME
            inp_type = constants.DEFAULT_INPUT_TYPE
            try:
                if hasattr(robot, 'part_controllers') and robot.part_controllers:
                    for _, ctrl in robot.part_controllers.items():
                        if hasattr(ctrl, 'name'):
                            ctrl_name = ctrl.name
                        if hasattr(ctrl, 'input_type'):
                            inp_type = ctrl.input_type
                        break
            except Exception:
                pass
            self._robot_controller_info[ridx] = (ctrl_name, inp_type)
            logger.debug(f"Teleop: robot {ridx} controller={ctrl_name}, input_type={inp_type}")

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
            elif key == Key.space:
                # Trigger a single environment step when spacebar is pressed
                self.gripper_toggle.set()

        except AttributeError:
            pass

    def _render_loop(self) -> None:
        """Continuously render the environment."""
        while self.running:
            with self.env_lock if self.env_lock else nullcontext():
                env_step(self.env)
                self.env.render()

    def _eef_delta_to_joint_delta(self, robot, arm_name: str, eef_delta: np.ndarray) -> np.ndarray:
        """
        Convert a 6D EEF delta (pos + ori) to a 7-DOF joint delta via the
        damped Jacobian pseudo-inverse.

        Args:
            robot: Robot instance from env.robots
            arm_name: Name of the arm (e.g. "right")
            eef_delta: 6-element array [dx, dy, dz, drx, dry, drz]

        Returns:
            7-element joint delta array
        """
        sim = self.env.sim

        # Resolve the EEF body from the site name
        eef_site = robot.robot_model.eef_name[arm_name]
        eef_site_id = sim.model.site_name2id(eef_site)
        eef_body_id = sim.model.site_bodyid[eef_site_id]
        eef_body_name = sim.model.body_id2name(eef_body_id)

        # Get arm joint names and their DOF indices (Jacobian columns)
        arm_joints = robot.robot_model.arm_joints
        if isinstance(arm_joints, dict):
            arm_joint_names = arm_joints[arm_name]
        else:
            arm_joint_names = arm_joints

        dof_indices = []
        for jn in arm_joint_names:
            jnt_id = sim.model.joint_name2id(jn)
            dof_indices.append(sim.model.jnt_dofadr[jnt_id])

        # Full body Jacobian (3, nv) → extract relevant columns
        jacp = sim.data.get_body_jacp(eef_body_name).reshape(3, -1)[:, dof_indices]
        jacr = sim.data.get_body_jacr(eef_body_name).reshape(3, -1)[:, dof_indices]
        J = np.vstack([jacp, jacr])  # (6, 7)

        # Damped pseudo-inverse for numerical stability at singularities
        damping = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))  # (7, 6)

        return J_pinv @ eef_delta[:6]

    def _get_joint_positions(self, robot, arm_name: str) -> np.ndarray:
        """Read current joint positions for an arm."""
        sim = self.env.sim
        if hasattr(robot, f'{arm_name}_arm_joint_indexes'):
            idx = getattr(robot, f'{arm_name}_arm_joint_indexes')
        elif hasattr(robot, 'arm_joint_indexes'):
            idx = robot.arm_joint_indexes
        else:
            idx = robot.joint_indexes
        return sim.data.qpos[idx].copy()

    def run(self) -> None:
        """Run teleoperation loop. If env_lock is provided, guard env access with it."""
        self.running = True
        logger.info("Starting teleoperation with robosuite Keyboard device.")
        logger.info("Controls: Follow robosuite's keyboard controls.")
        logger.info("Press 'q' key to exit teleoperation.")

        with self.env_lock if self.env_lock else nullcontext():
            self.env.reset()
            reset_action_persistence()  # Reset action persistence state on env reset

        self.device.start_control()

        while self.running:
            time.sleep(1/constants.DEFAULT_TELEOP_FREQ)

            active_robot_idx = self.device.active_robot
            active_robot = self.env.robots[active_robot_idx]

            with self.env_lock if self.env_lock else nullcontext():
                # Sync device grasp_states with previous environment steps
                action_manager = get_action_persistence_manager()
                for i, arm in enumerate(active_robot.arms):
                    # Use updated storage key format
                    store_key = f"robot_{active_robot_idx}_{arm}_gripper"
                    if store_key in action_manager.previous_gripper_actions:
                        prev_val = action_manager.previous_gripper_actions[store_key]
                        # Handle both scalar and array-like gripper states
                        if hasattr(prev_val, "__len__"):
                            prev_val = prev_val[0]
                        
                        # Use nested indexing: [robot_idx][arm_idx]
                        if active_robot_idx < len(self.device.grasp_states):
                            if i < len(self.device.grasp_states[active_robot_idx]):
                                self.device.grasp_states[active_robot_idx][i] = True if prev_val > 0 else False
                            if self.gripper_toggle.is_set():
                                # Toggle grasp state
                                self.device.grasp_states[active_robot_idx][i] = not self.device.grasp_states[active_robot_idx][i]
                                self.gripper_toggle.clear()

                input_ac_dict = self.device.input2action(mirror_actions=True)

                if input_ac_dict is None:
                    break

                action_dict = deepcopy(input_ac_dict)

                # Determine controller type for the active robot
                ctrl_name, inp_type = self._robot_controller_info.get(
                    active_robot_idx,
                    (constants.DEFAULT_CONTROLLER_NAME, constants.DEFAULT_INPUT_TYPE),
                )
                is_joint_ctrl = (ctrl_name == constants.ControllerName.JOINT_POSITION)
                is_absolute = (inp_type == constants.InputType.ABSOLUTE)

                for arm in active_robot.arms:
                    eef_delta = input_ac_dict[f"{arm}_delta"]

                    if is_joint_ctrl:
                        # Convert 6D EEF delta → 7D joint delta via Jacobian
                        joint_delta = self._eef_delta_to_joint_delta(
                            active_robot, arm, eef_delta
                        )
                        if is_absolute:
                            # Absolute mode: send current + delta as target
                            current_joints = self._get_joint_positions(active_robot, arm)
                            action_dict[arm] = current_joints + joint_delta
                        else:
                            action_dict[arm] = joint_delta
                    else:
                        action_dict[arm] = eef_delta

                # Construct only the active robot's action vector
                robot_action = active_robot.create_action_vector(action_dict)
                
                env_step(self.env, robot_action, robot_id=self.device.active_robot, arm=active_robot.arms)

            if self.exit_pressed:
                self.running = False
                break

        self.listener.stop()
        self.device.listener.stop()
        logger.info("Teleoperation ended.")