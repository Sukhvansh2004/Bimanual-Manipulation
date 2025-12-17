import numpy as np

class RobosuiteGripperController:
    """
    A robosuite-equivalent GripperController.
    This class abstracts the control of a robot gripper in a robosuite environment.
    """
    def __init__(self, env, arm_id="right", robot_id=0):
        self.env = env
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_space.shape[0]

    def open(self, width=1.0):
        """
        Opens the gripper.
        Args:
            width (float): Target gripper width (normalized). 1.0 is fully open.
        """
        # In robosuite, gripper action: 1.0 is open, -1.0 is closed
        self._send_gripper_action(width)

    def close(self, width=0.0):
        """
        Closes the gripper.
        Args:
            width (float): Target gripper width (normalized). 0.0 is fully closed.
        """
        # Map width to action: assuming 0-> -1, 1->1
        action_val = -1.0 + width * 2.0
        self._send_gripper_action(action_val)

    def _send_gripper_action(self, gripper_val):
        """
        Sends a gripper action while keeping arm still.
        """
        full_action = np.zeros(self.total_action_dim)
        start_idx = self.robot_id * self.action_dim_per_robot + 6
        full_action[start_idx] = gripper_val
        self.env.step(full_action)