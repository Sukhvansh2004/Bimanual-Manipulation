import numpy as np
from typing import Any

class RobosuiteGripperController:
    """
    A robosuite-equivalent GripperController.
    This class abstracts the control of a robot gripper in a robosuite environment.
    """
    def __init__(self, env: Any, arm_id: str = "right", robot_id: int = 0) -> None:
        self.env = env
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_dim

    def open(self, width: float = 1.0) -> None:
        """
        Opens the gripper.
        Args:
            width (float): Target gripper width (normalized). 1.0 is fully open.
        """
        # In robosuite, gripper action: 1.0 is open, -1.0 is closed
        self._send_gripper_action(width)

    def close(self, width: float = 0.0) -> None:
        """
        Closes the gripper.
        Args:
            width (float): Target gripper width (normalized). 0.0 is fully closed.
        """
        # Map width to action: assuming 0-> -1, 1->1
        action_val = -1.0 + width * 2.0
        self._send_gripper_action(action_val)

    def _send_gripper_action(self, gripper_val: float) -> None:
        """
        Sends a gripper action while keeping arm still.
        """
        full_action = np.zeros(self.total_action_dim)
        start_idx = self.robot_id * self.action_dim_per_robot + 6
        full_action[start_idx] = gripper_val
        self.env.step(full_action)