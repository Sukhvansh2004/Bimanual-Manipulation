import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any, Tuple
import time

from robosuite_franka_manipulation import constants

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

    def plan_trapezoidal_trajectory(self, start_pos: np.ndarray, start_quat: np.ndarray, 
                                   target_pos: np.ndarray, target_quat: np.ndarray,
                                   max_vel: float = None, max_acc: float = None, 
                                   dt: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plan a trapezoidal velocity profile trajectory from start to target pose.
        
        Args:
            start_pos: Starting position
            start_quat: Starting quaternion
            target_pos: Target position
            target_quat: Target quaternion
            max_vel: Maximum velocity (m/s)
            max_acc: Maximum acceleration (m/s²)
            dt: Time step
            
        Returns:
            positions: Array of positions over time (N, 3)
            quaternions: Array of quaternions over time (N, 4)
            total_time: Total trajectory time
        """
        # Set defaults if not provided
        if max_vel is None:
            max_vel = constants.DEFAULT_MAX_VELOCITY
        if max_acc is None:
            max_acc = constants.DEFAULT_MAX_ACCELERATION
        if dt is None:
            dt = constants.DEFAULT_TIME_STEP
        
        # Calculate position distance
        pos_distance = np.linalg.norm(target_pos - start_pos)
        
        if pos_distance < constants.TRAJECTORY_MIN_DISTANCE:
            # No movement needed
            return np.array([target_pos]), np.array([target_quat]), 0.0
        
        # Calculate orientation distance (angle in radians)
        ori_diff = R.from_quat(target_quat) * R.from_quat(start_quat).inv()
        ori_distance = np.linalg.norm(ori_diff.as_rotvec())
        
        # Use position distance for timing, add orientation as a fraction
        total_distance = pos_distance + ori_distance * constants.TRAJECTORY_ORIENTATION_SCALING
        
        # Calculate trapezoidal profile timing with more reasonable parameters
        # Phase 1: Acceleration
        t_acc = max_vel / max_acc
        d_acc = 0.5 * max_acc * t_acc**2
        
        # Phase 3: Deceleration  
        t_dec = t_acc
        d_dec = d_acc
        
        # Phase 2: Constant velocity
        d_const = total_distance - d_acc - d_dec
        if d_const > 0:
            t_const = d_const / max_vel
        else:
            # Triangle profile
            t_acc = np.sqrt(total_distance / max_acc)
            t_const = 0
            t_dec = t_acc
        
        total_time = t_acc + t_const + t_dec
        n_steps = max(constants.TRAJECTORY_MIN_STEPS, int(np.ceil(total_time / dt)))  # At least 2 steps
        
        # Generate trajectory
        positions = []
        quaternions = []
        
        for i in range(n_steps + 1):
            t = i * dt
            
            # Calculate position along trapezoid
            if t <= t_acc:
                # Acceleration phase
                s = 0.5 * max_acc * t**2
            elif t <= t_acc + t_const:
                # Constant velocity phase
                s = d_acc + max_vel * (t - t_acc)
            else:
                # Deceleration phase
                t_dec_elapsed = t - (t_acc + t_const)
                s = d_acc + d_const + max_vel * t_dec_elapsed - 0.5 * max_acc * t_dec_elapsed**2
            
            # Clamp to [0, total_distance]
            s = np.clip(s, 0, total_distance)
            
            # Interpolate position
            if pos_distance > 0:
                pos_interp = start_pos + (target_pos - start_pos) * (s / total_distance)
            else:
                pos_interp = target_pos
            
            # Interpolate orientation (using rotation vector interpolation)
            if ori_distance > 0:
                start_rotvec = R.from_quat(start_quat).as_rotvec()
                target_rotvec = R.from_quat(target_quat).as_rotvec()
                rotvec_interp = start_rotvec + (target_rotvec - start_rotvec) * (s / total_distance)
                quat_interp = R.from_rotvec(rotvec_interp).as_quat()
            else:
                quat_interp = target_quat
            
            positions.append(pos_interp)
            quaternions.append(quat_interp)
        
        return np.array(positions), np.array(quaternions), total_time

    def move_to_pose(self, target_pos: np.ndarray, target_quat: np.ndarray, 
                    threshold: float = None, 
                    timeout: int = None, 
                    gripper_action: float = 0.0) -> bool:
        """
        Moves the robot's end-effector to the target position and orientation using a trapezoidal trajectory.

        Args:
            target_pos (np.array): Desired 3D position (x, y, z).
            target_quat (np.array): Desired quaternion (x, y, z, w) for orientation.
            threshold (float): Position error threshold for convergence.
            timeout (int): Maximum number of environment steps to reach the pose.
            gripper_action (float): Gripper action to maintain (-1 closed, 1 open).
        """
        # Set defaults if not provided
        if threshold is None:
            threshold = constants.TRAJECTORY_CONVERGENCE_THRESHOLD
        if timeout is None:
            timeout = constants.TRAJECTORY_MAX_TIMEOUT
        
        current_pos = self.robot._hand_pos[self.arm_id]
        current_ori_mat = self.robot._hand_orn[self.arm_id]
        current_quat = R.from_matrix(current_ori_mat).as_quat()

        positions, quaternions, total_time = self.plan_trapezoidal_trajectory(
            current_pos, current_quat, target_pos, target_quat
        )

        if len(positions) <= 1:
            return True # Already at target

        for i in range(min(len(positions), timeout)):
            target_pos_step = positions[i]
            target_quat_step = quaternions[i]

            current_pos = self.robot._hand_pos[self.arm_id]
            current_ori_mat = self.robot._hand_orn[self.arm_id]
            current_quat = R.from_matrix(current_ori_mat).as_quat()

            pos_error = target_pos_step - current_pos
            ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
            ori_error_vec = ori_error.as_rotvec()

            # Construct action for this robot
            robot_action = np.zeros(self.action_dim_per_robot)
            robot_action[:3] = np.clip(pos_error * constants.TRAJECTORY_POS_ERROR_SCALE, -1, 1)  # Position deltas
            robot_action[3:6] = np.clip(ori_error_vec * constants.TRAJECTORY_ORI_ERROR_SCALE, -1, 1)  # Orientation deltas
            robot_action[6] = gripper_action  # Gripper

            # Construct full action
            full_action = np.zeros(self.total_action_dim)
            start_idx = self.robot_id * self.action_dim_per_robot
            end_idx = start_idx + self.action_dim_per_robot
            full_action[start_idx:end_idx] = robot_action

            # Step the environment
            self.env.step(full_action)

        final_pos = self.robot._hand_pos[self.arm_id]
        final_ori_mat = self.robot._hand_orn[self.arm_id]
        final_quat = R.from_matrix(final_ori_mat).as_quat()
        pos_error = target_pos - final_pos
        ori_error = R.from_quat(target_quat) * R.from_quat(final_quat).inv()
        ori_error_vec = ori_error.as_rotvec()

        time.sleep(constants.TRAJECTORY_SLEEP_TIME)
        if np.linalg.norm(pos_error) < threshold and np.linalg.norm(ori_error_vec) < threshold:
            return True

        return False