import robosuite as suite
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any, Optional

class RobosuiteArmController:
    """
    A robosuite-equivalent ArmController.
    This class abstracts the control of a single robot arm in a robosuite environment.
    It takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.
    """
    def __init__(self, env: Any, arm_id: str = "right", robot_id: int = 0) -> None:
        self.env = env
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.robot = self.env.robots[self.robot_id]
        self.dof = self.robot.dof

        # Action space: 6 pose + 1 gripper
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_dim

    def move_to_pose(self, target_pos: np.ndarray, target_quat: np.ndarray, threshold: float = 0.01, timeout: int = 100, gripper_action: float = 0.0) -> bool:
        """
        Moves the robot's end-effector to the target position and orientation.

        Args:
            target_pos (np.array): Desired 3D position (x, y, z).
            target_quat (np.array): Desired quaternion (x, y, z, w) for orientation.
            threshold (float): Position error threshold for convergence.
            timeout (int): Maximum number of environment steps to reach the pose.
            gripper_action (float): Gripper action to maintain (-1 closed, 1 open).
        """
        for i in range(timeout):
            current_pos = self.robot._hand_pos[self.arm_id]
            current_ori_mat = self.robot._hand_orn[self.arm_id]
            current_quat = R.from_matrix(current_ori_mat).as_quat()

            pos_error = target_pos - current_pos
            ori_error = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
            ori_error_vec = ori_error.as_rotvec()

            # Check convergence
            if np.linalg.norm(pos_error) < threshold and np.linalg.norm(ori_error_vec) < threshold:
                return True

            # Construct action for this robot
            robot_action = np.zeros(self.action_dim_per_robot)
            robot_action[:3] = np.clip(pos_error * 10, -1, 1)  # Position deltas
            robot_action[3:6] = np.clip(ori_error_vec * 2, -1, 1)  # Orientation deltas
            robot_action[6] = gripper_action  # Gripper

            # Construct full action
            full_action = np.zeros(self.total_action_dim)
            start_idx = self.robot_id * self.action_dim_per_robot
            end_idx = start_idx + self.action_dim_per_robot
            full_action[start_idx:end_idx] = robot_action

            # Step the environment
            self.env.step(full_action)

        return False