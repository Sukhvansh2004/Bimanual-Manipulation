import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union, Optional

from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.utils.se3 import (
    pose_to_SE3,
    SE3_to_pose,
    SE3_log,
    SE3_exp,
    compute_se3_distance,
)


class RobosuiteArmController:
    """
    A robosuite-equivalent ArmController.
    This class abstracts the control of a single robot arm in a robosuite environment.
    It takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.
    """
    def __init__(self, env: Union[MujocoEnv, Wrapper], arm_id: str = "right", robot_id: int = 0, 
                 transform_manager: Optional[TransformManager] = None) -> None:
        self.env = env
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.robot = self.env.robots[self.robot_id]  # type: ignore[attr-defined]
        self.dof = self.robot.dof
        
        # Robot base frame name for transforms
        robot_name = getattr(self.robot, 'name', f'robot{robot_id}')
        self.base_frame = f"{robot_name}_base"

        # Action space: 6 pose + 1 gripper
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_dim
        
        # Transform manager for frame conversions
        # If not provided, create one (will be shared if passed from DualArmController)
        self.transform_manager = transform_manager or TransformManager(env)

    def plan_trapezoidal_trajectory(self, start_pos: np.ndarray, start_quat: np.ndarray, 
                                   target_pos: np.ndarray, target_quat: np.ndarray,
                                   max_vel: float = constants.DEFAULT_MAX_VELOCITY,
                                   max_acc: float = constants.DEFAULT_MAX_ACCELERATION, 
                                   dt: float = constants.DEFAULT_TIME_STEP) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plan a trapezoidal velocity profile trajectory from start to target pose in SE(3) space.
        
        Uses Lie algebra se(3) for proper coupled rotation-translation interpolation.
        This ensures smooth geodesic motion on the SE(3) manifold.
        
        The trajectory is computed as:
            T(s) = T_start @ exp(s * log(T_start^-1 @ T_target))
        
        where s follows a trapezoidal velocity profile from 0 to 1.
        
        Args:
            start_pos: Starting position [x, y, z]
            start_quat: Starting quaternion [x, y, z, w]
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [x, y, z, w]
            max_vel: Maximum velocity (m/s) - controls linear component
            max_acc: Maximum acceleration (m/s²)
            dt: Time step (s)
            
        Returns:
            positions: Array of positions over time (N, 3)
            quaternions: Array of quaternions over time (N, 4)
            total_time: Total trajectory time (s)
        """
        # Convert to SE(3) matrices
        T_start = pose_to_SE3(start_pos, start_quat)
        T_target = pose_to_SE3(target_pos, target_quat)
        
        # Compute relative transformation in the tangent space (Lie algebra)
        # T_rel = T_start^-1 @ T_target, so T_target = T_start @ T_rel
        T_start_inv = np.eye(4)
        T_start_inv[:3, :3] = T_start[:3, :3].T
        T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]
        
        T_rel = T_start_inv @ T_target
        
        # Map to Lie algebra se(3) - this is the "velocity" that takes us from start to target
        twist = SE3_log(T_rel)
        
        # Compute distance metrics for timing
        # Use translation component for velocity limiting, rotation for additional time if needed
        omega = twist[:3]  # Rotation vector
        v = twist[3:]      # Translation vector (in body frame)
        
        # Check if movement is needed
        total_se3_distance = compute_se3_distance(
            twist, 
            translation_weight=1.0, 
            rotation_weight=constants.TRAJECTORY_ORIENTATION_SCALING
        )
        
        if total_se3_distance < constants.TRAJECTORY_MIN_DISTANCE:
            return np.array([target_pos]), np.array([target_quat]), 0.0
        
        # Use weighted se(3) distance for velocity timing
        # This properly accounts for coupled rotation-translation motion
        effective_distance = total_se3_distance
        
        # Compute trapezoidal profile timing
        # Phase 1: Acceleration from 0 to max_vel
        t_acc = max_vel / max_acc
        d_acc = 0.5 * max_acc * t_acc**2
        
        # Phase 3: Deceleration from max_vel to 0
        t_dec = t_acc
        d_dec = d_acc
        
        # Phase 2: Constant velocity
        d_const = effective_distance - d_acc - d_dec
        
        if d_const > 0:
            # Full trapezoidal profile
            t_const = d_const / max_vel
        else:
            # Triangle profile (distance too short for constant velocity phase)
            t_acc = np.sqrt(effective_distance / max_acc)
            t_const = 0
            t_dec = t_acc
            d_acc = 0.5 * effective_distance
            d_dec = d_acc
            d_const = 0
        
        total_time = t_acc + t_const + t_dec
        n_steps = max(constants.TRAJECTORY_MIN_STEPS, int(np.ceil(total_time / dt)))
        
        # Generate trajectory waypoints
        positions = []
        quaternions = []
        
        for i in range(n_steps + 1):
            t = i * total_time / n_steps if n_steps > 0 else 0
            
            # Compute position along trapezoid (distance traveled at time t)
            if t <= t_acc:
                # Acceleration phase
                d = 0.5 * max_acc * t**2
            elif t <= t_acc + t_const:
                # Constant velocity phase
                d = d_acc + max_vel * (t - t_acc)
            else:
                # Deceleration phase
                t_in_dec = t - (t_acc + t_const)
                d = d_acc + d_const + max_vel * t_in_dec - 0.5 * max_acc * t_in_dec**2
            
            # Normalize to [0, 1] interpolation parameter
            s = np.clip(d / effective_distance, 0, 1) if effective_distance > 0 else 1.0
            
            # Interpolate in SE(3) using the exponential map
            # T(s) = T_start @ exp(s * twist)
            T_interp = T_start @ SE3_exp(s * twist)
            
            # Extract position and quaternion
            pos, quat = SE3_to_pose(T_interp)
            positions.append(pos)
            quaternions.append(quat)
        
        return np.array(positions), np.array(quaternions), total_time

    def move_to_pose(self, target_pose: StampedPose, 
                    threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD, 
                    timeout: int = constants.TRAJECTORY_MAX_TIMEOUT, 
                    gripper_action: float = 0.0) -> bool:
        """
        Moves the robot's end-effector to the target position and orientation using a trapezoidal trajectory.
        
        The pose will be automatically transformed to the robot's base frame if needed.

        Args:
            target_pose: Desired pose as StampedPose (includes position, quaternion, and frame_id)
            threshold: Position error threshold for convergence
            timeout: Maximum number of environment steps to reach the pose
            gripper_action: Gripper action to maintain (-1 closed, 1 open)
        
        Returns:
            bool: True if pose was reached within threshold, False otherwise
        """
        # Transform to robot base frame if needed
        if target_pose.frame_id != self.base_frame:
            target_pose = self.transform_manager.transform_pose(target_pose, self.base_frame)
        
        target_pos = np.asarray(target_pose.position)
        target_quat = np.asarray(target_pose.quaternion)
        
        current_pos = self.robot._hand_pos[self.arm_id]
        current_ori_mat = self.robot._hand_orn[self.arm_id]
        current_quat = R.from_matrix(current_ori_mat).as_quat()

        positions, quaternions, total_time = self.plan_trapezoidal_trajectory(
            current_pos, current_quat, target_pos, target_quat
        )

        if len(positions) <= 1:
            return True # Already at target

        # Closed-loop trajectory execution with waypoint tracking
        waypoint_idx = 1  # Start from second waypoint (first is current pos)
        steps_at_waypoint = 0
        max_steps_per_waypoint = constants.TRAJECTORY_STEPS_PER_WAYPOINT
        waypoint_threshold = constants.TRAJECTORY_WAYPOINT_THRESHOLD
        
        for step in range(timeout):
            target_pos_step = positions[waypoint_idx]
            target_quat_step = quaternions[waypoint_idx]

            current_pos = self.robot._hand_pos[self.arm_id]
            current_ori_mat = self.robot._hand_orn[self.arm_id]
            current_quat = R.from_matrix(current_ori_mat).as_quat()

            pos_error = target_pos_step - current_pos
            ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
            ori_error_vec = ori_error.as_rotvec()
            
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error_vec)
            
            # Check if we've reached the current waypoint or spent enough time
            steps_at_waypoint += 1
            if (pos_error_norm < waypoint_threshold and ori_error_norm < waypoint_threshold) or \
               steps_at_waypoint >= max_steps_per_waypoint:
                # Move to next waypoint
                waypoint_idx += 1
                steps_at_waypoint = 0
                if waypoint_idx >= len(positions):
                    break  # Finished all waypoints

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
        
        # Final convergence: Keep sending final target until converged or timeout
        final_convergence_steps = constants.TRAJECTORY_FINAL_CONVERGENCE_STEPS
        for _ in range(final_convergence_steps):
            current_pos = self.robot._hand_pos[self.arm_id]
            current_ori_mat = self.robot._hand_orn[self.arm_id]
            current_quat = R.from_matrix(current_ori_mat).as_quat()
            
            pos_error = target_pos - current_pos
            ori_error = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
            ori_error_vec = ori_error.as_rotvec()
            
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error_vec)
            
            # Check if converged
            if pos_error_norm < threshold and ori_error_norm < threshold:
                break
                
            # Construct action
            robot_action = np.zeros(self.action_dim_per_robot)
            robot_action[:3] = np.clip(pos_error * constants.TRAJECTORY_POS_ERROR_SCALE, -1, 1)
            robot_action[3:6] = np.clip(ori_error_vec * constants.TRAJECTORY_ORI_ERROR_SCALE, -1, 1)
            robot_action[6] = gripper_action
            
            full_action = np.zeros(self.total_action_dim)
            start_idx = self.robot_id * self.action_dim_per_robot
            end_idx = start_idx + self.action_dim_per_robot
            full_action[start_idx:end_idx] = robot_action
            
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