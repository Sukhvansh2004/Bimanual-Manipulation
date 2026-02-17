from contextlib import nullcontext
import threading
import numpy as np
from typing import Optional
import logging

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.utils import env_step

logger = logging.getLogger(__name__)

class RobosuiteGripperController:
    """
    A robosuite-equivalent GripperController.
    This class abstracts the control of a robot gripper in a robosuite environment.
    Supports both Parallel Jaw and Suction grippers.
    """
    def __init__(self, env: DualArmEnvironment, arm_id: str = "right", robot_id: int = 0, env_lock: Optional[threading.RLock] = None) -> None:
        self.env = env
        self.env_lock = env_lock
        self.arm_id = arm_id
        self.robot_id = robot_id
        # Dynamically determine per-robot action dim from the actual robot
        self.robot_action_dim = self.env.robots[self.robot_id].action_dim
        self.total_action_dim = self.env.action_dim
        
        # Detect gripper type
        self.is_suction = self._check_if_suction()
        if self.is_suction:
            logger.info(f"GripperController detected Suction/Vacuum gripper for robot {robot_id}")

    def _check_if_suction(self) -> bool:
        """Checks if the robot is equipped with a suction gripper based on model name."""
        try:
            # Robosuite robots list is accessible via env.robots
            robot = self.env.robots[self.robot_id]
            gripper_type = type(robot.gripper[self.arm_id]).__name__.lower()
            logger.debug(f"Detected gripper type for robot {self.robot_id}: {gripper_type}")
            return "vacuum" in gripper_type or "suction" in gripper_type
        except Exception as e:
            logger.warning(f"Could not detect gripper type: {e}")
        return False

    def release(self, width: float = 1.0) -> None:
        """
        Releases (opens) the gripper.
        Args:
            width (float): Target gripper width (normalized). 1.0 is fully open/release.
        """
        if self.is_suction:
            # For suction, "Open" means Release (Suction Off)
            self._send_gripper_action(-1)
        else:
            # For parallel jaw: map width [0.0, 1.0] to action [1.0, -1.0]
            # width=1.0 (fully open) -> action=-1.0
            # width=0.0 (closed) -> action=1.0
            action_val = 1.0 - width * 2.0
            logger.debug(f"Releasing gripper of arm {self.arm_id} of robot {self.robot_id} with width {width}, action value {action_val}")
            self._send_gripper_action(action_val)

    def grasp(self, width: float = 0.0) -> None:
        """
        Closes (grasps) the gripper.
        Args:
            width (float): Target gripper width (normalized). 0.0 is fully closed/grasp.
        """
        if self.is_suction:
            # For suction, "Close" means Grasp (Suction On)
            self._send_gripper_action(1)
        else:
            # For parallel jaw: map width [0.0, 1.0] to action [1.0, -1.0]
            # width=0.0 (fully closed) -> action=1.0
            # width=1.0 (fully open) -> action=-1.0
            action_val = 1.0 - width * 2.0
            logger.debug(f"Grasping gripper of arm {self.arm_id} of robot {self.robot_id} with width {width}, action value {action_val}")
            self._send_gripper_action(action_val)

    def _send_gripper_action(self, gripper_val: float) -> None:
        """
        Sends a gripper action while keeping arm still.
        """
        # Create single robot action
        robot_action = np.zeros(self.robot_action_dim)
        # Gripper index is at the end of the robot's action space
        # (Assuming gripper is the last dimension of robot_action_dim)
        robot_action[-1] = gripper_val
        
        with self.env_lock if self.env_lock else nullcontext():
            env_step(self.env, robot_action, robot_id=self.robot_id, arm=self.arm_id)