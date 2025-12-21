import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any, Tuple, Optional, Union, Dict
import time

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose


# ============================================================================
# SE(3) Lie Algebra Utilities for Trajectory Planning
# ============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    For vector v = [v1, v2, v3], returns:
        [[ 0, -v3,  v2],
         [ v3,  0, -v1],
         [-v2,  v1,  0]]
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """Extract 3D vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def se3_to_matrix(twist: np.ndarray) -> np.ndarray:
    """
    Convert 6D twist vector to 4x4 se(3) matrix.
    
    Twist format: [omega (3), v (3)] where omega is angular velocity
    and v is linear velocity.
    
    Returns:
        4x4 matrix: [[skew(omega), v],
                     [0, 0, 0,     0]]
    """
    omega = twist[:3]
    v = twist[3:]
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(omega)
    mat[:3, 3] = v
    return mat


def matrix_to_se3(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 se(3) matrix to 6D twist vector."""
    omega = unskew(mat[:3, :3])
    v = mat[:3, 3]
    return np.concatenate([omega, v])


def SE3_log(T: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm of an SE(3) transformation matrix.
    
    Maps SE(3) -> se(3) (Lie group to Lie algebra).
    Uses closed-form solution for numerical stability.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        6D twist vector [omega, v] in se(3)
    """
    R_mat = T[:3, :3]
    p = T[:3, 3]
    
    # Compute rotation angle
    cos_theta = (np.trace(R_mat) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    
    if theta < 1e-8:
        # Near identity: use first-order approximation
        omega = np.zeros(3)
        v = p
    elif np.abs(theta - np.pi) < 1e-8:
        # Near 180 degrees: special handling
        # Find axis from diagonal elements
        diag = np.diag(R_mat)
        idx = np.argmax(diag)
        axis = np.zeros(3)
        axis[idx] = 1.0
        
        # Reconstruct axis more carefully
        r_vec = np.sqrt((diag + 1) / 2)
        r_vec[idx] = np.sqrt((diag[idx] + 1) / 2)
        
        # Determine signs from off-diagonal elements
        if idx == 0:
            r_vec[1] = np.sign(R_mat[0, 1]) * r_vec[1] if R_mat[0, 1] != 0 else r_vec[1]
            r_vec[2] = np.sign(R_mat[0, 2]) * r_vec[2] if R_mat[0, 2] != 0 else r_vec[2]
        elif idx == 1:
            r_vec[0] = np.sign(R_mat[0, 1]) * r_vec[0] if R_mat[0, 1] != 0 else r_vec[0]
            r_vec[2] = np.sign(R_mat[1, 2]) * r_vec[2] if R_mat[1, 2] != 0 else r_vec[2]
        else:
            r_vec[0] = np.sign(R_mat[0, 2]) * r_vec[0] if R_mat[0, 2] != 0 else r_vec[0]
            r_vec[1] = np.sign(R_mat[1, 2]) * r_vec[1] if R_mat[1, 2] != 0 else r_vec[1]
        
        omega = theta * r_vec / (np.linalg.norm(r_vec) + 1e-10)
        
        # V_inv for theta = pi
        omega_hat = skew(omega)
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            V_inv = np.eye(3) - 0.5 * omega_hat + (1 / omega_norm**2) * (1 - omega_norm * np.sin(omega_norm) / (2 * (1 - np.cos(omega_norm)))) * omega_hat @ omega_hat
        else:
            V_inv = np.eye(3)
        v = V_inv @ p
    else:
        # General case
        omega_hat = (theta / (2 * np.sin(theta))) * (R_mat - R_mat.T)
        omega = unskew(omega_hat)
        
        # Compute V_inv analytically
        omega_norm = np.linalg.norm(omega)
        omega_hat_normalized = omega_hat / (omega_norm + 1e-10)
        V_inv = np.eye(3) - 0.5 * omega_hat + (1 / theta**2) * (1 - theta * np.sin(theta) / (2 * (1 - np.cos(theta)))) * omega_hat @ omega_hat
        v = V_inv @ p
    
    return np.concatenate([omega, v])


def SE3_exp(twist: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential of an se(3) twist vector.
    
    Maps se(3) -> SE(3) (Lie algebra to Lie group).
    Uses closed-form Rodriguez formula.
    
    Args:
        twist: 6D vector [omega, v] in se(3)
        
    Returns:
        4x4 homogeneous transformation matrix in SE(3)
    """
    omega = twist[:3]
    v = twist[3:]
    theta = np.linalg.norm(omega)
    
    T = np.eye(4)
    
    if theta < 1e-8:
        # Near zero rotation: use first-order approximation
        T[:3, :3] = np.eye(3) + skew(omega)
        T[:3, 3] = v
    else:
        # Rodriguez formula for rotation
        omega_hat = skew(omega)
        omega_hat_normalized = omega_hat / theta
        
        R_mat = np.eye(3) + np.sin(theta) * omega_hat_normalized + (1 - np.cos(theta)) * omega_hat_normalized @ omega_hat_normalized
        
        # V matrix for translation
        V = np.eye(3) + ((1 - np.cos(theta)) / theta**2) * omega_hat + ((theta - np.sin(theta)) / theta**3) * omega_hat @ omega_hat
        
        T[:3, :3] = R_mat
        T[:3, 3] = V @ v
    
    return T


def pose_to_SE3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Convert position and quaternion to SE(3) matrix.
    
    Args:
        pos: 3D position [x, y, z]
        quat: Quaternion [x, y, z, w] (scipy convention)
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def SE3_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SE(3) matrix to position and quaternion.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        (position, quaternion) where quaternion is [x, y, z, w]
    """
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat


def compute_se3_distance(twist: np.ndarray, translation_weight: float = 1.0, 
                         rotation_weight: float = 1.0) -> float:
    """
    Compute a weighted distance metric from an se(3) twist.
    
    This gives a measure of "how far" the transformation is,
    combining rotation and translation appropriately.
    
    Args:
        twist: 6D twist [omega, v]
        translation_weight: Weight for translation component
        rotation_weight: Weight for rotation component
        
    Returns:
        Weighted distance metric
    """
    omega = twist[:3]
    v = twist[3:]
    
    rot_dist = np.linalg.norm(omega)  # Rotation angle
    trans_dist = np.linalg.norm(v)
    
    return translation_weight * trans_dist + rotation_weight * rot_dist


# ============================================================================
# Arm Controller Class
# ============================================================================

class RobosuiteArmController:
    """
    A robosuite-equivalent ArmController.
    This class abstracts the control of a single robot arm in a robosuite environment.
    It takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.
    """
    def __init__(self, env: Any, arm_id: str = "right", robot_id: int = 0, 
                 transform_manager: Optional[TransformManager] = None) -> None:
        self.env = env
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.robot = self.env.robots[self.robot_id]
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
                                   max_vel: float = None, max_acc: float = None, 
                                   dt: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
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
        # Set defaults if not provided
        if max_vel is None:
            max_vel = constants.DEFAULT_MAX_VELOCITY
        if max_acc is None:
            max_acc = constants.DEFAULT_MAX_ACCELERATION
        if dt is None:
            dt = constants.DEFAULT_TIME_STEP
        
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

    def move_to_pose(self, target_pos: Union[np.ndarray, StampedPose, Dict], 
                    target_quat: Optional[np.ndarray] = None, 
                    threshold: float = None, 
                    timeout: int = None, 
                    gripper_action: float = 0.0,
                    frame_id: str = None) -> bool:
        """
        Moves the robot's end-effector to the target position and orientation using a trapezoidal trajectory.
        
        If frame_id is provided or target_pos is a StampedPose/dict with frame info,
        the pose will be transformed to the robot's base frame before execution.

        Args:
            target_pos: Desired 3D position (x, y, z), or StampedPose, or dict with 'pos', 'quat', 'frame_id'.
            target_quat: Desired quaternion (x, y, z, w) for orientation. Ignored if target_pos is StampedPose/dict.
            threshold (float): Position error threshold for convergence.
            timeout (int): Maximum number of environment steps to reach the pose.
            gripper_action (float): Gripper action to maintain (-1 closed, 1 open).
            frame_id (str): Frame of the target pose. If not robot base frame, will be transformed.
                           Defaults to robot base frame (no transform needed).
        
        Returns:
            bool: True if pose was reached within threshold, False otherwise.
        """
        # Set defaults if not provided
        if threshold is None:
            threshold = constants.TRAJECTORY_CONVERGENCE_THRESHOLD
        if timeout is None:
            timeout = constants.TRAJECTORY_MAX_TIMEOUT
        
        # Handle different input formats
        target_pos, target_quat, frame_id = self._parse_pose_input(target_pos, target_quat, frame_id)
        
        # Transform to robot base frame if needed
        # Only skip transform if pose is already in the robot's base frame
        if frame_id is not None and frame_id != self.base_frame:
            target_pos, target_quat = self._transform_to_base_frame(target_pos, target_quat, frame_id)
        
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

    def _parse_pose_input(self, target_pos: Union[np.ndarray, StampedPose, Dict], 
                          target_quat: Optional[np.ndarray], 
                          frame_id: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Parse different pose input formats into consistent position, quaternion, frame_id.
        
        Args:
            target_pos: Position array, StampedPose, or dict with pose info
            target_quat: Quaternion (if target_pos is array)
            frame_id: Frame ID (if not embedded in target_pos)
            
        Returns:
            Tuple of (position, quaternion, frame_id)
        """
        if isinstance(target_pos, StampedPose):
            return target_pos.position, target_pos.quaternion, target_pos.frame_id
        
        if isinstance(target_pos, dict):
            pos = np.asarray(target_pos['pos'])
            quat = np.asarray(target_pos['quat'])
            fid = target_pos.get('frame_id', frame_id)
            return pos, quat, fid
        
        # target_pos is already an array
        pos = np.asarray(target_pos)
        quat = np.asarray(target_quat) if target_quat is not None else np.array([0, 0, 0, 1])
        return pos, quat, frame_id

    def _transform_to_base_frame(self, position: np.ndarray, quaternion: np.ndarray, 
                                  source_frame: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a pose from source_frame to the robot's base frame.
        
        Args:
            position: Position in source_frame
            quaternion: Quaternion (x, y, z, w) in source_frame
            source_frame: The frame the pose is currently in
            
        Returns:
            Tuple of (position, quaternion) in robot base frame
        """
        # Create StampedPose for the input
        pose = StampedPose(
            position=position,
            quaternion=quaternion,
            frame_id=source_frame
        )
        
        # Transform to robot base frame
        transformed = self.transform_manager.transform_pose(pose, self.base_frame)
        
        return transformed.position, transformed.quaternion