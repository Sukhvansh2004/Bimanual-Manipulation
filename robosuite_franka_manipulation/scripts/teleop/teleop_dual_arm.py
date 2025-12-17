#!/usr/bin/env python3
"""
Teleoperation script for dual-arm Franka robots in robosuite.
Uses robosuite's I/O devices for control.
"""

import numpy as np
import time
from typing import Any

from scipy.spatial.transform import Rotation

from robosuite.devices import Keyboard

from ...environments import make_dual_arm_handoff_env
from ...controllers import RobosuiteArmController, RobosuiteGripperController

class KeyboardTeleop:
    """
    Keyboard teleoperation for dual-arm manipulation using robosuite's I/O devices.
    """
    def __init__(self, env: Any) -> None:
        self.env = env
        self.left_arm = RobosuiteArmController(env, arm_id="left", robot_id=0)
        self.right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
        self.left_gripper = RobosuiteGripperController(env, arm_id="left", robot_id=0)
        self.right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

        # Control parameters
        self.step_size = 0.01  # meters
        self.rot_step = 0.05   # radians
        self.active_arm = "left"  # or "right"

        # Current target pose
        self.current_pos = np.array([0.0, 0.3, 0.5])
        self.current_euler = np.zeros(3)  # roll, pitch, yaw
        self.current_quat = np.array([0, 0, 0, 1])  # identity quaternion

        # Initialize robosuite keyboard device
        self.device = Keyboard(pos_sensitivity=self.step_size, rot_sensitivity=self.rot_step)

        print("Teleoperation initialized.")
        print("Controls:")
        print("  Movement (X/Y/Z): w/s (X), a/d (Y), q/e (Z)")
        print("  Rotation (Roll/Pitch/Yaw): i/k (roll), j/l (pitch), u/o (yaw)")
        print("  Gripper: space (close when pressed)")
        print("  Switch arm: r (right), t (left)")
        print("  Reset pose: z")
        print("  Quit: x")
        print(f"Active arm: {self.active_arm}")


    def move_arm(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
        """Move the active arm by the given deltas."""
        controller = self.left_arm if self.active_arm == "left" else self.right_arm

        # Update position
        self.current_pos += np.array([dx, dy, dz])

        # Update orientation
        self.current_euler += np.array([droll, dpitch, dyaw])
        rot = Rotation.from_euler('xyz', self.current_euler)
        self.current_quat = rot.as_quat()

        success = controller.move_to_pose(self.current_pos, self.current_quat, timeout=10)
        if success:
            print(f"Current pos: {self.current_pos}")
        else:
            print(f"Failed to move {self.active_arm} arm")

    def control_gripper(self, action):
        """Control the active arm's gripper."""
        gripper = self.left_gripper if self.active_arm == "left" else self.right_gripper

        if action == "open":
            gripper.open()
            print(f"{self.active_arm} gripper opened")
        elif action == "close":
            gripper.close()
            print(f"{self.active_arm} gripper closed")

    def reset_pose(self):
        """Reset to a safe pose."""
        self.current_pos = np.array([0.0, 0.3, 0.5])
        self.current_euler = np.zeros(3)
        self.current_quat = np.array([0, 0, 0, 1])
        self.move_arm()

    def switch_arm(self, arm):
        """Switch active arm."""
        self.active_arm = arm
        print(f"Switched to {arm} arm")

    def run(self):
        """Main teleoperation loop using robosuite's I/O devices."""
        print("Starting teleoperation. Use keyboard to control.")
        print("Movement: w/s (x), a/d (y), q/e (z)")
        print("Rotation: i/k (roll), j/l (pitch), u/o (yaw)")
        print("Gripper: space (close when pressed)")
        print("Switch arm: r (right), t (left)")
        print("Quit: x")
        print(f"Active arm: {self.active_arm}")

        import keyboard  # for additional keys

        last_switch_time = 0

        while True:
            current_time = time.time()

            # Handle arm switching with debounce
            if keyboard.is_pressed('r') and current_time - last_switch_time > 0.2:
                self.switch_arm("right")
                last_switch_time = current_time
            if keyboard.is_pressed('t') and current_time - last_switch_time > 0.2:
                self.switch_arm("left")
                last_switch_time = current_time

            # Quit
            if keyboard.is_pressed('x'):
                break

            # Reset pose
            if keyboard.is_pressed('z') and current_time - last_switch_time > 0.2:
                self.reset_pose()
                last_switch_time = current_time

            # Get controller state from robosuite device
            state = self.device.get_controller_state()

            # Handle movement
            dpos = state['dpos']
            drot = state['drot']
            if dpos is not None and np.linalg.norm(dpos) > 1e-6:
                self.move_arm(dx=dpos[0], dy=dpos[1], dz=dpos[2],
                              droll=drot[0], dpitch=drot[1], dyaw=drot[2])

            # Handle gripper
            if state['grasp']:
                self.control_gripper("close")
            else:
                self.control_gripper("open")

            time.sleep(0.01)  # Small delay to prevent excessive polling

        print("Teleoperation ended.")

def main() -> None:
    """Main function."""
    try:
        env = make_dual_arm_handoff_env()
        env.reset()

        teleop = KeyboardTeleop(env)
        teleop.run()

        env.close()

    except KeyboardInterrupt:
        print("\nTeleoperation interrupted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()