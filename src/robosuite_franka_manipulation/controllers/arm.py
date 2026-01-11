from contextlib import nullcontext
import time

import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union, Optional

import meshcat
import meshcat.geometry as g

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
from robosuite_franka_manipulation.utils import safe_env_step
from robosuite_franka_manipulation.utils.primitives import compute_approach_pose
from grasp_gen.utils.meshcat_utils import make_frame
import logging  
# Configure logging
logging.basicConfig(level=constants.LOG_LEVEL)
logger = logging.getLogger(__name__)

class RobosuiteArmController:
    """
    A robosuite-equivalent ArmController.
    This class abstracts the control of a single robot arm in a robosuite environment.
    It takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.
    """
    def __init__(self, env: Union[MujocoEnv, Wrapper], arm_id: str = "right", robot_id: int = 0, 
                 transform_manager: Optional[TransformManager] = None, env_lock: Optional[threading.Lock] = None,
                 visualize: bool = False) -> None:
        self.env = env
        self.env_lock = env_lock
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.robot = self.env.robots[self.robot_id]  # type: ignore[attr-defined]
        self.dof = self.robot.dof
        self.visualize = visualize
        
        # Lazy-created MeshCat visualizer for trajectory visualization
        self._meshcat_vis: Optional[meshcat.Visualizer] = None
        
        # Robot base frame name for transforms
        robot_name = getattr(self.robot, 'name', f'robot{robot_id}')
        logger.debug(f"RobosuiteArmController initialized for robot '{robot_name}', arm '{arm_id}'")

        # Use a stable unique TF base frame (robosuite can reuse robot.name across robots)
        self.base_frame = f"robot{robot_id}_base"

        # Action space: 6 pose + 1 gripper
        self.action_dim_per_robot = 7
        self.total_action_dim = self.env.action_dim
        
        # Transform manager for frame conversions
        # If not provided, create one (will be shared if passed from DualArmController)
        if transform_manager:
            self.transform_manager = transform_manager
        else:
            self.transform_manager = TransformManager(env, env_lock=self.env_lock)

    def _eef_body_name(self) -> str:
        return self.robot.robot_model.eef_name[self.arm_id]

    def get_eef_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current end-effector pose expressed in the robot base frame.

        Uses robosuite's public API (`pose_in_base_from_name`) instead of accessing
        underscore-prefixed internals like `_hand_pos` / `_hand_orn`.

        Returns:
            (pos, quat): position (3,) and quaternion (x, y, z, w)
        """
        eef_body = self._eef_body_name()
        with self.env_lock if self.env_lock else nullcontext():
            pose4 = self.robot.pose_in_base_from_name(eef_body)

        pos = np.asarray(pose4[:3, 3], dtype=np.float64).copy()
        quat = R.from_matrix(np.asarray(pose4[:3, :3], dtype=np.float64)).as_quat()
        return pos, quat

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
                    gripper_action: float = 0.0,
                    approach: bool = False,
                    approach_distance: float = 0.1,
                    approach_axis: str = 'z') -> bool:
        """
        Moves the robot's end-effector to the target position and orientation using a trapezoidal trajectory.
        
        The pose will be automatically transformed to the robot's base frame if needed.

        Args:
            target_pose: Desired pose as StampedPose (includes position, quaternion, and frame_id)
            threshold: Position error threshold for convergence
            timeout: Maximum number of environment steps to reach the pose
            gripper_action: Gripper action to maintain (-1 closed, 1 open)
            approach: If True, first move to an approach pose offset along the target's local axis,
                     then move to the actual target. Useful for grasp approach motions.
            approach_distance: Distance to offset for approach (meters). Default 0.1m (10cm).
            approach_axis: Local axis of target frame to offset along ('x', 'y', or 'z').
                          Default 'z' (gripper approach direction).
        
        Returns:
            bool: True if pose was reached within threshold, False otherwise
        """
        # Handle approach motion: first move to approach pose, then to target
        if approach:
            logger.debug(f"Target pose before approach computation: pos={target_pose.position}, frame_id={target_pose.frame_id}")
            approach_pose = compute_approach_pose(
                target_pose, 
                approach_distance=approach_distance,
                approach_axis=approach_axis
            )
            logger.debug(f"Approach enabled: moving to approach pose first (offset {approach_distance}m along local {approach_axis}-axis)")
            logger.debug(f"Approach pose: pos={approach_pose.position}, frame_id={approach_pose.frame_id}")
            
            # Move to approach pose first (without approach, to avoid recursion)
            approach_success = self.move_to_pose(
                approach_pose,
                threshold=threshold,
                timeout=timeout,
                gripper_action=gripper_action,
                approach=False  # Don't recurse
            )
            
            if not approach_success:
                logger.warning("Failed to reach approach pose")
                # Continue to try reaching target anyway
            else:
                logger.debug("Approach pose reached successfully")
            
            logger.debug("Now moving to final target pose")
           
        
        # Transform to robot base frame if needed
        if target_pose.frame_id != self.base_frame:
            logger.debug(f"Transforming target pose from '{target_pose.frame_id}' to '{self.base_frame}'")
            target_pose = self.transform_manager.transform_pose(target_pose, self.base_frame)
        
        target_pos = np.asarray(target_pose.position)
        target_quat = np.asarray(target_pose.quaternion)

        # Get FRESH current pose right before planning
        current_pos, current_quat = self.get_eef_pose_in_base()
        
        logger.info(f"Planning trajectory for robot{self.robot_id}_{self.arm_id}:")
        logger.info(f"  Current EEF in base: pos={current_pos}")
        logger.info(f"  Target in base: pos={target_pos}")
        logger.info(f"  Distance to target: {np.linalg.norm(target_pos - current_pos):.4f}m")

        positions, quaternions, total_time = self.plan_trapezoidal_trajectory(
            current_pos, current_quat, target_pos, target_quat
        )

        # Visualize the SE(3) trajectory if enabled
        if self.visualize:
            self._clear_trajectory_visualization()
            self._visualize_se3_trajectory(positions, quaternions)

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

            current_pos, current_quat = self.get_eef_pose_in_base()

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
                if(pos_error_norm < waypoint_threshold and ori_error_norm < waypoint_threshold):
                    logger.debug(f"Reached waypoint {waypoint_idx} at step {step} with pos error {pos_error_norm:.4f}m, ori error {ori_error_norm:.4f}rad")
                else:
                    logger.debug(f"Max steps at waypoint {waypoint_idx} reached ({steps_at_waypoint}), moving to next waypoint")
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

            # Step the environment (use safe wrapper to handle intermittent runtime errors)
            with self.env_lock if self.env_lock else nullcontext():
                safe_env_step(self.env, full_action)
        
        # Final convergence: Keep sending final target until converged or timeout
        final_convergence_steps = constants.TRAJECTORY_FINAL_CONVERGENCE_STEPS
        for _ in range(final_convergence_steps):
            current_pos, current_quat = self.get_eef_pose_in_base()
            
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
            
            with self.env_lock if self.env_lock else nullcontext():
                safe_env_step(self.env, full_action)

        final_pos, final_quat = self.get_eef_pose_in_base()

        pos_error = target_pos - final_pos
        ori_error = R.from_quat(target_quat) * R.from_quat(final_quat).inv()
        ori_error_vec = ori_error.as_rotvec()

        time.sleep(constants.TRAJECTORY_SLEEP_TIME)
        if np.linalg.norm(pos_error) < threshold and np.linalg.norm(ori_error_vec) < threshold:
            return True

        return False
    
    def _clear_trajectory_visualization(self) -> None:
        """Clear all trajectory visualizations from MeshCat."""
        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
        
        vis = self._meshcat_vis
        
        # Clear the entire trajectory folder for this robot/arm
        try:
            vis[f"trajectory/robot{self.robot_id}_{self.arm_id}"].delete()
        except Exception:
            pass
    
    def _visualize_se3_trajectory(self, positions: np.ndarray, quaternions: np.ndarray,
                                   num_frames: int = 10, axis_length: float = 0.05,
                                   axis_radius: float = 0.003) -> None:
        """
        Visualize the SE(3) trajectory frames in MeshCat.
        
        Shows coordinate frames along the trajectory path to visualize the
        coupled rotation-translation motion in SE(3) space.
        
        Args:
            positions: Array of positions (N, 3) in robot base frame
            quaternions: Array of quaternions (N, 4) in (x, y, z, w) format, in robot base frame
            num_frames: Number of frames to display along trajectory (evenly spaced)
            axis_length: Length of coordinate frame axes
            axis_radius: Radius of coordinate frame axes
        """
        if len(positions) < 2:
            return
        
        # Create or reuse MeshCat visualizer
        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
        
        vis = self._meshcat_vis
        root = vis[f"trajectory/robot{self.robot_id}_{self.arm_id}"]
        
        # Clear previous trajectory visualization
        try:
            root.delete()
        except Exception:
            pass
        
        # Get transform from robot base frame to world frame
        # This is needed because trajectory is computed in base frame but MeshCat uses world frame
        # get_transform(target, source) returns transform FROM source TO target
        # We want: pose_in_world = T_world_from_base @ pose_in_base
        try:
            base_to_world = self.transform_manager.get_transform("world", self.base_frame)
            T_base_to_world = base_to_world.to_matrix()
        except Exception as e:
            logger.warning(f"Could not get base-to-world transform, visualizing in base frame: {e}")
            T_base_to_world = np.eye(4)
        
        # Visualize all waypoints in the trajectory
        n_waypoints = len(positions)
        
        # Store world-frame positions for the trajectory line
        world_positions = []
        
        # Draw frames at all waypoints
        for idx in range(n_waypoints):
            pos = positions[idx]
            quat = quaternions[idx]
            
            # Build SE(3) transform matrix in robot base frame
            T_pose_in_base = pose_to_SE3(pos, quat)
            
            # Transform to world frame: T_world = T_base_to_world @ T_pose_in_base
            T_pose_in_world = T_base_to_world @ T_pose_in_base
            
            # Store world position for trajectory line
            world_positions.append(T_pose_in_world[:3, 3])
            
            # Determine frame name based on position in trajectory
            if idx == 0:
                frame_name = "start"
            elif idx == n_waypoints - 1:
                frame_name = "end"
            else:
                frame_name = f"waypoint_{idx}"
            
            # Use make_frame from grasp_gen utils
            frame_path = f"{root.path}/{frame_name}"
            make_frame(
                vis,
                name=frame_path,
                h=axis_length,
                radius=axis_radius,
                o=0.9,  # opacity
            )
            vis[frame_path].set_transform(T_pose_in_world)
        
        # Draw trajectory line connecting the frames (in world coordinates)
        world_positions = np.array(world_positions)
        line_points = world_positions.T  # (3, N) for MeshCat
        colors = np.zeros((3, n_waypoints))
        
        # Gradient color: green -> blue -> red
        for i in range(n_waypoints):
            t = i / max(1, n_waypoints - 1)
            if t < 0.5:
                # Green to blue
                colors[0, i] = 0  # R
                colors[1, i] = 1 - 2*t  # G
                colors[2, i] = 2*t  # B
            else:
                # Blue to red
                colors[0, i] = 2*(t - 0.5)  # R
                colors[1, i] = 0  # G
                colors[2, i] = 1 - 2*(t - 0.5)  # B
        
        root["path"].set_object(
            g.Line(
                g.PointsGeometry(line_points, colors),
                g.LineBasicMaterial(vertexColors=True, linewidth=2)
            )
        )
        
        logger.debug(f"Visualized SE(3) trajectory with {n_waypoints} frames in world frame")