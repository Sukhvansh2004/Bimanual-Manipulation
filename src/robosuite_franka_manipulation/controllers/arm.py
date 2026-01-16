from contextlib import nullcontext
import time

import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional

import meshcat
import meshcat.geometry as g

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

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
    def __init__(self, env: DualArmEnvironment, arm_id: str = "right", robot_id: int = 0, 
                 transform_manager: Optional[TransformManager] = None, env_lock: Optional[threading.Lock] = None,
                 visualize: bool = False) -> None:
        self.env = env
        self.env_lock = env_lock
        self.arm_id = arm_id
        self.robot_id = robot_id
        # NOTE: Don't cache self.robot - always get fresh reference via self._get_robot()
        # to avoid stale references after env resets or state changes
        self.dof = self._get_robot().dof
        self.visualize = visualize
        
        # Lazy-created MeshCat visualizer for trajectory visualization
        self._meshcat_vis: Optional[meshcat.Visualizer] = None
        
        # Robot base frame name for transforms
        robot_name = getattr(self._get_robot(), 'name', f'robot{robot_id}')
        logger.debug(f"RobosuiteArmController initialized for robot '{robot_name}', arm '{arm_id}'")

        # Use a stable unique TF base frame (robosuite can reuse robot.name across robots)
        self.base_frame = f"robot{robot_id}_base"

        # Action space: 6 pose + 1 gripper
        self.action_dim_per_robot = constants.DEFAULT_ROBOT_ACTION_DIM
        self.total_action_dim = self.env.action_dim
        
        # Transform manager for frame conversions
        # If not provided, create one (will be shared if passed from DualArmController)
        if transform_manager:
            self.transform_manager = transform_manager
        else:
            self.transform_manager = TransformManager(env, env_lock=self.env_lock)

    def _get_robot(self):
        """Get fresh robot reference from environment.
        
        Always use this instead of a cached self.robot to ensure we have
        the current robot state after environment steps/resets.
        """
        return self.env.robots[self.robot_id]  # type: ignore[attr-defined]

    def _eef_body_name(self) -> str:
        return self._get_robot().robot_model.eef_name[self.arm_id]

    def get_eef_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current end-effector pose expressed in the robot base frame.

        Uses robosuite's public API (`pose_in_base_from_name`) instead of accessing
        underscore-prefixed internals like `_hand_pos` / `_hand_orn`.
        
        Gets a fresh robot reference each call to ensure current state.

        Returns:
            (pos, quat): position (3,) and quaternion (x, y, z, w)
        """
        robot = self._get_robot()  # Fresh reference
        eef_body = robot.robot_model.eef_name[self.arm_id]
        
        with self.env_lock if self.env_lock else nullcontext():
            pose4 = robot.pose_in_base_from_name(eef_body)

        pos = np.asarray(pose4[:3, 3], dtype=np.float64).copy()
        quat = R.from_matrix(np.asarray(pose4[:3, :3], dtype=np.float64)).as_quat()
        return pos, quat

    @staticmethod
    def _quintic_polynomial(s0: float, sf: float, v0: float = 0.0, vf: float = 0.0, 
                            a0: float = 0.0, af: float = 0.0, 
                            num_points: int = 100) -> np.ndarray:
        """
        Generate a 5th-order (quintic) polynomial trajectory.
        
        The quintic polynomial ensures smooth motion with:
        - Continuous position (C0 continuity)
        - Continuous velocity (C1 continuity) 
        - Continuous acceleration (C2 continuity)
        
        This results in jerk-limited motion which is smoother than trapezoidal profiles.
        
        Args:
            s0: Initial position (typically 0)
            sf: Final position (typically 1)
            v0: Initial velocity (default 0 - start from rest)
            vf: Final velocity (default 0 - end at rest)
            a0: Initial acceleration (default 0)
            af: Final acceleration (default 0)
            num_points: Number of trajectory points to generate
            
        Returns:
            s_trajectory: Array of interpolation parameters in [0, 1]
        """
        T = 1.0  # Normalize time to [0, 1]
        time = np.linspace(0, T, num_points)
        
        # Coefficient matrix for quintic polynomial boundary conditions:
        # s(0) = s0, s(T) = sf
        # s'(0) = v0, s'(T) = vf  
        # s''(0) = a0, s''(T) = af
        M = np.array([
            [0, 0, 0, 0, 0, 1],                      # s(0)
            [T**5, T**4, T**3, T**2, T, 1],          # s(T)
            [0, 0, 0, 0, 1, 0],                      # s'(0)
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],    # s'(T)
            [0, 0, 0, 2, 0, 0],                      # s''(0)
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]        # s''(T)
        ])
        
        # Boundary conditions vector
        b = np.array([s0, sf, v0, vf, a0, af])
        
        # Solve for polynomial coefficients [a5, a4, a3, a2, a1, a0]
        coeffs = np.linalg.solve(M, b)
        
        # Evaluate polynomial at each time step
        s_trajectory = np.zeros(num_points)
        for i, t in enumerate(time):
            T_vec = np.array([t**5, t**4, t**3, t**2, t, 1])
            s_trajectory[i] = T_vec @ coeffs
        
        return s_trajectory
    
    def plan_quintic_trajectory(self, start_pos: np.ndarray, start_quat: np.ndarray, 
                                target_pos: np.ndarray, target_quat: np.ndarray,
                                num_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plan a quintic polynomial trajectory from start to target pose in SE(3) space.
        
        Uses Lie algebra se(3) for proper coupled rotation-translation interpolation.
        This ensures smooth geodesic motion on the SE(3) manifold with continuous
        velocity and acceleration profiles (C2 continuity).
        
        The trajectory is computed as:
            T(s) = T_start @ exp(s * log(T_start^-1 @ T_target))
        
        where s follows a quintic polynomial from 0 to 1, ensuring:
        - Zero velocity at start and end (smooth stop)
        - Zero acceleration at start and end (jerk-limited)
        
        Args:
            start_pos: Starting position [x, y, z]
            start_quat: Starting quaternion [x, y, z, w]
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [x, y, z, w]
            num_points: Number of waypoints to generate (default from constants)
            
        Returns:
            positions: Array of positions over time (N, 3)
            quaternions: Array of quaternions over time (N, 4)
        """
        if num_points is None:
            num_points = constants.TRAJECTORY_NUM_POINTS
        
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
            return np.array([target_pos]), np.array([target_quat])
        
        # Generate quintic polynomial interpolation parameters s(t) in [0, 1]
        # Starting and ending at rest with zero acceleration
        s_trajectory = self._quintic_polynomial(
            s0=0.0, sf=1.0,  # Interpolate from start (0) to target (1)
            v0=0.0, vf=0.0,  # Zero velocity at endpoints (smooth stop)
            a0=0.0, af=0.0,  # Zero acceleration at endpoints (jerk-limited)
            num_points=num_points
        )
        
        # Generate trajectory waypoints by interpolating SE(3) using the quintic profile
        positions = []
        quaternions = []
        
        for s in s_trajectory:
            # Interpolate in SE(3) using the exponential map
            # T(s) = T_start @ exp(s * twist)
            T_interp = T_start @ SE3_exp(s * twist)
            
            # Extract position and quaternion
            pos, quat = SE3_to_pose(T_interp)
            positions.append(pos)
            quaternions.append(quat)
        
        return np.array(positions), np.array(quaternions)

    def move_to_pose(self, target_pose: StampedPose, 
                    threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD, 
                    gripper_action: float = 0.0,
                    approach: bool = False,
                    dt: float = constants.TRAJECTORY_TIME_STEP,
                    approach_distance: float = constants.DEFAULT_APPROACH_DISTANCE,
                    approach_axis: str = constants.DEFAULT_APPROACH_AXIS,
                    timeout: Optional[int] = None) -> bool:
        """
        Moves the robot's end-effector to the target position and orientation using a quintic polynomial trajectory.
        
        The pose will be automatically transformed to the robot's base frame if needed.
        Uses quintic polynomial interpolation in SE(3) space for smooth, jerk-limited motion.
        
        The OSC controller is sent delta commands based on the trajectory waypoints,
        with one action per waypoint to follow the smooth trajectory.

        Args:
            target_pose: Desired pose as StampedPose (includes position, quaternion, and frame_id)
            threshold: Position/orientation error threshold for convergence check
            gripper_action: Raw gripper action to maintain during movement.
                            - For Parallel Gripper: 1.0 (Open), -1.0 (Closed).
                            - For Suction Gripper: 1.0 (Release/Off), -1.0 (Grasp/On).
            approach: If True, first move to an approach pose offset along the target's local axis,
                     then move to the actual target. Useful for grasp approach motions.
            approach_distance: Distance to offset for approach (meters). Default 0.1m (10cm).
            approach_axis: Local axis of target frame to offset along ('x', 'y', or 'z').
                          Default 'z' (gripper approach direction).
            timeout: Maximum number of steps to execute. If None, uses trajectory waypoint count.
                    This is kept for backwards compatibility but is generally not needed.
        
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
        # Force MuJoCo to sync forward kinematics before reading
        with self.env_lock if self.env_lock else nullcontext():
            self.env.sim.forward()  # Ensure FK is up to date
        
        current_pos, current_quat = self.get_eef_pose_in_base()
        
        logger.info(f"Planning trajectory for robot{self.robot_id}_{self.arm_id}:")
        logger.info(f"  Current EEF in base: pos={current_pos}")
        logger.info(f"  Target in base: pos={target_pos}")
        logger.info(f"  Distance to target: {np.linalg.norm(target_pos - current_pos):.4f}m")

        positions, quaternions = self.plan_quintic_trajectory(
            current_pos, current_quat, target_pos, target_quat
        )

        # Visualize the SE(3) trajectory if enabled
        if self.visualize:
            self._clear_trajectory_visualization()
            self._visualize_se3_trajectory(positions, quaternions)

        if len(positions) <= 1:
            return True  # Already at target

        # Trajectory execution: send delta commands to follow the quintic trajectory
        # The OSC controller expects delta commands (since input_type='delta'),
        # so we compute the delta from current position to next waypoint
        num_waypoints = len(positions)
        max_steps = timeout if timeout is not None else num_waypoints * 2
        
        # Track which waypoint we're currently targeting
        current_waypoint_idx = 1  # Start from second waypoint (first is current position)
        steps_taken = 0
        
        while current_waypoint_idx < num_waypoints and steps_taken < max_steps:
            # Get target waypoint
            target_pos_step = positions[current_waypoint_idx]
            target_quat_step = quaternions[current_waypoint_idx]
            
            # Get current pose
            current_pos, current_quat = self.get_eef_pose_in_base()
            
            # Compute error to current waypoint target
            pos_delta = target_pos_step - current_pos
            ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
            ori_delta = ori_error.as_rotvec()
            
            # Check if we've reached this waypoint (within a tolerance)
            waypoint_pos_error = np.linalg.norm(pos_delta)
            waypoint_ori_error = np.linalg.norm(ori_delta)
            waypoint_threshold = threshold * 2  # More lenient for intermediate waypoints
            
            if waypoint_pos_error < waypoint_threshold and waypoint_ori_error < waypoint_threshold:
                # Move to next waypoint
                current_waypoint_idx += 1
                if current_waypoint_idx >= num_waypoints:
                    break
                continue
            
            # Send delta command toward current waypoint
            # Scale down large deltas to avoid instability
            max_pos_delta = constants.MAX_POS_DELTA
            max_ori_delta = constants.MAX_ORI_DELTA
            
            pos_delta_scaled = pos_delta
            if waypoint_pos_error > max_pos_delta:
                pos_delta_scaled = pos_delta * (max_pos_delta / waypoint_pos_error)
            
            ori_delta_scaled = ori_delta
            if waypoint_ori_error > max_ori_delta:
                ori_delta_scaled = ori_delta * (max_ori_delta / waypoint_ori_error)
            
            # Construct action with scaled delta
            robot_action = np.zeros(self.action_dim_per_robot)
            robot_action[:3] = pos_delta_scaled  # Position delta
            robot_action[3:6] = ori_delta_scaled  # Orientation delta
            robot_action[6] = gripper_action
            
            # Construct full action
            full_action = np.zeros(self.total_action_dim)
            start_idx = self.robot_id * self.action_dim_per_robot
            end_idx = start_idx + self.action_dim_per_robot
            full_action[start_idx:end_idx] = robot_action
            
            # Step the environment
            with self.env_lock if self.env_lock else nullcontext():
                safe_env_step(self.env, full_action)
            
            steps_taken += 1
            time.sleep(dt)  # Maintain control frequency
            
            # Log progress periodically
            if steps_taken % 20 == 0:
                logger.debug(f"Step {steps_taken}/{max_steps}, waypoint {current_waypoint_idx}/{num_waypoints-1}, "
                           f"pos_err={waypoint_pos_error:.4f}m, ori_err={waypoint_ori_error:.4f}rad")
        
        # Final check: verify we reached the target
        final_pos, final_quat = self.get_eef_pose_in_base()
        
        pos_error = target_pos - final_pos
        ori_error = R.from_quat(target_quat) * R.from_quat(final_quat).inv()
        ori_error_vec = ori_error.as_rotvec()
        
        pos_error_norm = np.linalg.norm(pos_error)
        ori_error_norm = np.linalg.norm(ori_error_vec)
        
        logger.info(f"Trajectory complete: final pos_err={pos_error_norm:.4f}m, ori_err={ori_error_norm:.4f}rad")
        
        if pos_error_norm < threshold and ori_error_norm < threshold:
            return True
        
        return False
    
    def _clear_trajectory_visualization(self) -> None:
        """Clear all trajectory visualizations from MeshCat."""
        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
        
        vis = self._meshcat_vis
        
        # Clear the entire trajectory folder for this robot/arm
        try:
            vis[f"trajectory_robot{self.robot_id}_{self.arm_id}"].delete()
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
        traj_path = f"trajectory_robot{self.robot_id}_{self.arm_id}"
        root = vis[traj_path]
        
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
            frame_path = f"{traj_path}/{frame_name}"
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