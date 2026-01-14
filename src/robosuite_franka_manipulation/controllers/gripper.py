from contextlib import nullcontext
import threading
import numpy as np
from typing import Union, Optional
import logging

from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper
from robosuite_franka_manipulation.utils import safe_env_step

logger = logging.getLogger(__name__)

class RobosuiteGripperController:
    """
    A robosuite-equivalent GripperController.
    This class abstracts the control of a robot gripper in a robosuite environment.
    Supports both Parallel Jaw and Suction grippers.
    """
    def __init__(self, env: Union[MujocoEnv, Wrapper], arm_id: str = "right", robot_id: int = 0, env_lock: Optional[threading.Lock] = None) -> None:
        self.env = env
        self.env_lock = env_lock
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_dim
        
        # Detect gripper type
        self.is_suction = self._check_if_suction()
        if self.is_suction:
            logger.info(f"GripperController detected Suction/Vacuum gripper for robot {robot_id}")

    def _check_if_suction(self) -> bool:
        """Checks if the robot is equipped with a suction gripper based on model name."""
        try:
            # Robosuite robots list is accessible via env.robots
            if hasattr(self.env, "robots") and len(self.env.robots) > self.robot_id:
                robot = self.env.robots[self.robot_id]
                if hasattr(robot, 'gripper') and robot.gripper is not None:
                    # Check the class name or the name attribute of the gripper model
                    # Common names: "VacuumGripper", "Suction", "Adhesion"
                    gripper_type = type(robot.gripper).__name__.lower()
                    return "vacuum" in gripper_type or "suction" in gripper_type
        except Exception as e:
            logger.warning(f"Could not detect gripper type: {e}")
        return False

    def open(self, width: float = 1.0) -> None:
        """
        Opens (releases) the gripper.
        Args:
            width (float): Target gripper width (normalized). 1.0 is fully open/release.
        """
        if self.is_suction:
            # For suction, "Open" means Release (Suction Off)
            # Robosuite Suction Action: < 0 is Off, > 0 is On.
            # We map "open" (width=1.0) to Action -1.0
            self._send_gripper_action(-1.0)
        else:
            # For parallel: 1.0 is open
            self._send_gripper_action(width)

    def close(self, width: float = 0.0) -> None:
        """
        Closes (grasps) the gripper.
        Args:
            width (float): Target gripper width (normalized). 0.0 is fully closed/grasp.
        """
        if self.is_suction:
            # For suction, "Close" means Grasp (Suction On)
            # Robosuite Suction Action: > 0 is On.
            # We map "close" (width=0.0) to Action 1.0
            self._send_gripper_action(1.0)
        else:
            # For parallel: 0.0 -> -1.0 (Closed)
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
        with self.env_lock if self.env_lock else nullcontext():
            safe_env_step(self.env, full_action)