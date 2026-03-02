from contextlib import nullcontext
import time

import mink
from mink.lie import SE3 as MinkSE3, SO3 as MinkSO3
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, List

import meshcat
import meshcat.geometry as g

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.utils.se3 import (
    pose_to_SE3,
    SE3_log,
    compute_se3_distance,
    interpolate_pose,
)
from robosuite_franka_manipulation.utils import env_step
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    quintic_scalar,
    plan_quintic_joint_trajectory,
    velocity_profile_joint_trajectory,
)
from grasp_gen.utils.meshcat_utils import make_frame
import logging

# Configure logging
logger = logging.getLogger(__name__)

class RobosuiteArmController:
    """
    A robosuite-equivalent ArmController.
    This class abstracts the control of a single robot arm in a robosuite environment.
    It takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.
    
    Supports multiple controller types:
    - OSC_POSE (delta/absolute): Operational space control using EEF pose
    - JOINT_POSITION (delta/absolute): Joint position control
    """
    def __init__(self, env: DualArmEnvironment, arm_id: str = "right", robot_id: int = 0, 
                 transform_manager: Optional[TransformManager] = None, env_lock: Optional[threading.RLock] = None,
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

        # Detect controller configuration
        self._controller_name, self._input_type = self._detect_controller_config()
        logger.debug(f"Detected controller: {self._controller_name}, input_type: {self._input_type}")
        
        # Action dimensions based on controller type
        self._arm_action_dim = constants.CONTROLLER_ACTION_DIMS.get(
            self._controller_name, 
            constants.CONTROLLER_ACTION_DIMS[constants.ControllerName.OSC_POSE]
        )
        # Total action dim = arm action + gripper (usually 1)
        self.action_dim_per_robot = self._arm_action_dim + 1  # +1 for gripper
        self.total_action_dim = self.env.action_dim
        
        # Transform manager for frame conversions
        self.tf_manager = transform_manager if transform_manager else TransformManager(env, env_lock=self.env_lock)

        self._stop_event = threading.Event()
        self.is_moving = False
        self.last_status = 1  # 1 for success, 0 for failure
    
    def _detect_controller_config(self) -> Tuple[str, str]:
        """
        Detect the controller name and input type from the robot's configuration.
        
        Returns:
            (controller_name, input_type): Tuple of strings
        """
        controller_name = constants.DEFAULT_CONTROLLER_NAME
        input_type = constants.DEFAULT_INPUT_TYPE
        
        try:
            robot = self._get_robot()
            if hasattr(robot, 'part_controllers') and robot.part_controllers:
                for arm_name, controller in robot.part_controllers.items():
                    if hasattr(controller, 'name'):
                        controller_name = controller.name
                    if hasattr(controller, 'input_type'):
                        input_type = controller.input_type
                    break
        except Exception as e:
            logger.debug(f"Could not detect controller config: {e}")
        
        return controller_name, input_type
    
    @property
    def controller_name(self) -> str:
        """Get the current controller name."""
        return self._controller_name
    
    @property
    def input_type(self) -> str:
        """Get the current input type."""
        return self._input_type
    
    @property
    def is_absolute_mode(self) -> bool:
        """Check if controller is in absolute mode."""
        return self._input_type == constants.InputType.ABSOLUTE
    
    @property
    def is_joint_position_controller(self) -> bool:
        """Check if controller is JOINT_POSITION type."""
        return self._controller_name == constants.ControllerName.JOINT_POSITION

    def _get_robot(self):
        """Get fresh robot reference from environment.
        
        Always use this instead of a cached self.robot to ensure we have
        the current robot state after environment steps/resets.
        """
        return self.env.robots[self.robot_id]  # type: ignore[attr-defined]

    def _eef_body_name(self) -> str:
        return self._get_robot().robot_model.eef_name[self.arm_id]
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions of the arm.
        
        Returns:
            Array of joint positions (7 for Franka Panda)
        """
        robot = self._get_robot()
        try:
            if hasattr(robot, f'{self.arm_id}_arm_joint_indexes'):
                joint_indices = getattr(robot, f'{self.arm_id}_arm_joint_indexes')
            elif hasattr(robot, 'arm_joint_indexes'):
                joint_indices = robot.arm_joint_indexes
            else:
                joint_indices = robot.joint_indexes
            
            return self.env.sim.data.qpos[joint_indices].copy()
        except Exception as e:
            logger.warning(f"Could not read joint positions: {e}")
            return constants.FRANKA_HOME_JOINT_POSITION.copy()

    def get_eef_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current end-effector pose expressed in the robot base frame.

        Uses TransformManager instead of accessing underscore-prefixed internals.
        
        Returns:
            (pos, quat): position (3,) and quaternion (x, y, z, w)
        """
        eef_frame = f"robot{self.robot_id}_{self.arm_id}_eef"
        
        # We want pose of eef_frame in base_frame
        transform = self.tf_manager.get_transform(self.base_frame, eef_frame)
        pos = transform.translation.copy()
        quat = transform.rotation.copy()
        return pos, quat
    
    def _solve_ik_for_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve IK for a target EEF pose and return the best joint configuration.
        
        Uses the planning module's IK solver.
        The target pose is expected in the robot base frame and will be internally
        transformed to world frame for IK (which operates in world coords).
        
        Args:
            target_pos: Target EEF position [x, y, z] in robot base frame
            target_quat: Target EEF quaternion [x, y, z, w] in robot base frame  
            q_warm_start: Optional warm-start joint config (uses current if None).
                          Pass a known-good config (e.g. from planning NPZ) to
                          help IK converge for hard-to-reach poses.
            
        Returns:
            Joint configuration array (7,) or None if IK fails
        """
        # Create StampedPose in base frame; the IK solver will transform to world internally
        target_stamped = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=self.base_frame,
        )
        
        if q_warm_start is None:
            q_warm_start = self.get_joint_positions()
        
        # Lazy import to avoid circular dependency
        # (arm → planning → services → arm)
        from robosuite_franka_manipulation.planning.utils.ik_utils import (
            enumerate_ik_solutions_for_arm,
            apply_ik_orientation_correction,
        )

        # The eef body frame (used by TransformManager / get_eef_pose_in_base)
        # is rotated ~90° around Z relative to the MuJoCo ft_frame site that
        # the IK solver targets.  apply_ik_orientation_correction compensates
        # for this so the IK solution matches the body-frame convention.
        target_stamped = apply_ik_orientation_correction(target_stamped)

        site_name = f"gripper{self.robot_id}_{self.arm_id}_ft_frame"

        solutions = enumerate_ik_solutions_for_arm(
            self.env,
            robot_id=self.robot_id,
            target_pose=target_stamped,
            tf_manager=self.tf_manager,
            arm=self.arm_id,
            site_name=site_name,
            q_warm_start=q_warm_start,
            env_lock=self.env_lock,
            max_solutions=1,  # We only need one good solution
        )
        
        if not solutions:
            logger.warning(
                f"No IK solution found for robot{self.robot_id}_{self.arm_id} "
                f"at pos={target_pos}, quat={target_quat}"
            )
            return None
        
        return solutions[0]

    def construct_robot_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_action: float = 0.0,
        current_pos: Optional[np.ndarray] = None,
        current_quat: Optional[np.ndarray] = None,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct a robot action based on the current controller configuration.
        
        This method abstracts the action construction for different controller types
        and input modes (delta vs absolute).
        
        For OSC_POSE:
            - delta: computes [pos_delta, ori_delta, gripper]
            - absolute: uses [pos_target, ori_rotvec, gripper]
            
        For JOINT_POSITION:
            - Automatically solves IK to convert pose to joint targets
            - delta: computes [joint_delta, gripper]
            - absolute: uses [joint_target, gripper]
        
        Args:
            target_pos: Target position [x, y, z] in robot base frame
            target_quat: Target quaternion [x, y, z, w] in robot base frame
            gripper_action: Gripper action value
            current_pos: Current position (if None, reads from environment)
            current_quat: Current quaternion (if None, reads from environment)
            q_warm_start: Optional warm-start joint config for IK (uses current if None)
            
        Returns:
            Robot action array of appropriate dimension
        """
        if self.is_joint_position_controller:
            # Solve IK to convert pose target to joint target
            target_joints = self._solve_ik_for_pose(target_pos, target_quat, q_warm_start=q_warm_start)
            if target_joints is None:
                return None
            return self.construct_joint_action(target_joints, gripper_action)
        
        # Get current pose if not provided
        if current_pos is None or current_quat is None:
            current_pos, current_quat = self.get_eef_pose_in_base()
        
        robot_action = np.zeros(self.action_dim_per_robot)
        
        if self.is_absolute_mode:
            # Absolute mode: send target pose directly
            # OSC_POSE absolute expects [x, y, z, rx, ry, rz] where rotation is axis-angle
            target_rotvec = R.from_quat(target_quat).as_rotvec()
            robot_action[:3] = target_pos
            robot_action[3:6] = target_rotvec
        else:
            # Delta mode: compute difference from current to target
            pos_delta = target_pos - current_pos
            ori_error = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
            ori_delta = ori_error.as_rotvec()
            robot_action[:3] = pos_delta
            robot_action[3:6] = ori_delta
        
        robot_action[-1] = gripper_action  # Gripper action is always last
        return robot_action
    
    def construct_joint_action(
        self,
        target_joints: np.ndarray,
        gripper_action: float = 0.0,
        current_joints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct a robot action for JOINT_POSITION controller.
        
        Args:
            target_joints: Target joint positions [j1, ..., j7]
            gripper_action: Gripper action value
            current_joints: Current joint positions (if None, reads from environment)
            
        Returns:
            Robot action array for JOINT_POSITION controller
        """
        if not self.is_joint_position_controller:
            raise NotImplementedError(
                f"construct_joint_action requires JOINT_POSITION controller. "
                f"Current controller is: {self._controller_name}"
            )
        
        robot_action = np.zeros(self.action_dim_per_robot)
        
        if self.is_absolute_mode:
            # Absolute mode: send target joints directly
            robot_action[:7] = target_joints
        else:
            # Delta mode: compute difference from current to target
            if current_joints is None:
                current_joints = self.get_joint_positions()
            joint_delta = target_joints - current_joints
            robot_action[:7] = joint_delta
        
        robot_action[-1] = gripper_action
        return robot_action

    # ------------------------------------------------------------------
    # Mink-based reactive trajectory execution (JOINT_POSITION only)
    # ------------------------------------------------------------------

    def _execute_mink_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
    ) -> bool:
        """Execute a trajectory using mink's reactive differential IK.

        At each simulation tick the method:

        1. Sets the ``FrameTask`` target to the next waypoint (in world
           frame, with body↔site orientation correction applied).
        2. Calls ``mink.solve_ik`` — a single QP solve — to get the
           optimal joint velocity for the arm.
        3. Integrates the velocity to get target joint positions and
           sends them via ``construct_joint_action``.

        This replaces the per-step ``enumerate_ik_solutions_for_arm``
        approach with a lightweight single-QP call, giving smoother and
        faster trajectory tracking.

        Args:
            positions: Waypoint positions ``(N, 3)`` in **base** frame.
            quaternions: Waypoint quaternions ``(N, 4)`` xyzw in **base** frame.
            gripper_action: Gripper command to hold during execution.
            dt: Wall-clock time between steps (for drift-correcting sleep).
            threshold: Convergence threshold on final-waypoint error.

        Returns:
            ``True`` if the final waypoint was reached, ``False`` otherwise.
        """

        num_waypoints = len(positions)
        if num_waypoints <= 1:
            return True

        # ── Transform waypoints from base frame to world frame ────────
        try:
            base_to_world = self.tf_manager.get_transform(
                constants.WORLD_FRAME, self.base_frame
            )
            T_base_to_world = base_to_world.to_matrix()
        except Exception:
            T_base_to_world = np.eye(4)

        world_positions = np.zeros_like(positions)
        world_quaternions = np.zeros_like(quaternions)
        for i in range(num_waypoints):
            T_wp = pose_to_SE3(positions[i], quaternions[i])
            T_world = T_base_to_world @ T_wp
            world_positions[i] = T_world[:3, 3]
            world_quaternions[i] = R.from_matrix(T_world[:3, :3]).as_quat()

        # ── Apply body→site orientation correction (-90° Z) ──────────
        r_corr = R.from_euler('z', -90, degrees=True)
        for i in range(num_waypoints):
            r_wp = R.from_quat(world_quaternions[i])
            world_quaternions[i] = (r_wp * r_corr).as_quat()

        # ── Build mink objects ────────────────────────────────────────
        # Lazy import to avoid circular: arm → planning → services → arm
        from robosuite_franka_manipulation.planning.utils import mujoco_utils as _mu

        model = self.env.sim.model
        data = self.env.sim.data
        model_native = _mu.get_native_mj_object(model)

        joint_ids = _mu.get_arm_joint_ids(model, self.robot_id)
        arm_qpos_idx = _mu.get_joint_qpos_indices(model, joint_ids)
        arm_dof_idx = _mu.get_joint_dof_indices(model, joint_ids)

        site_name = f"gripper{self.robot_id}_{self.arm_id}_ft_frame"

        q_full = np.array(data.qpos, copy=True, dtype=float).ravel()
        config = mink.Configuration(model_native, q=q_full)

        frame_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=constants.MINK_POSITION_COST,
            orientation_cost=constants.MINK_ORIENTATION_COST,
            gain=constants.MINK_FRAME_GAIN,
            lm_damping=constants.MINK_LM_DAMPING,
        )

        posture_task = mink.PostureTask(model=model_native, cost=constants.MINK_POSTURE_COST)
        posture_task.set_target(q_full)

        damping_task = mink.DampingTask(model_native, cost=constants.MINK_DAMPING_COST)
        vel_limit = mink.VelocityLimit(model_native)
        config_limit = mink.ConfigurationLimit(model_native)

        from robosuite_franka_manipulation.planning.utils.mujoco_utils import get_frozen_dof_indices  # lazy to avoid circular import
        frozen_dofs = get_frozen_dof_indices(model_native, arm_dof_idx)
        freeze_task = mink.DofFreezingTask(model_native, dof_indices=frozen_dofs) if frozen_dofs else None

        tasks: list = [frame_task, posture_task, damping_task]
        if freeze_task is not None:
            tasks.append(freeze_task)
        limits = [vel_limit, config_limit]

        # ── Reactive execution loop ───────────────────────────────────
        max_steps = num_waypoints * 2
        current_wp_idx = 0
        steps_taken = 0
        start_time = time.time()

        while steps_taken < max_steps:
            if self._stop_event.is_set():
                logger.info("mink trajectory cancelled for robot%d_%s", self.robot_id, self.arm_id)
                self.last_status = 0
                return False

            # Advance waypoint
            if current_wp_idx < num_waypoints - 1:
                current_wp_idx += 1

            # Set mink FrameTask target
            wp_pos = world_positions[current_wp_idx]
            wp_quat = world_quaternions[current_wp_idx]
            target_so3 = MinkSO3.from_matrix(R.from_quat(wp_quat).as_matrix())
            target_se3 = MinkSE3.from_rotation_and_translation(target_so3, wp_pos)
            frame_task.set_target(target_se3)

            # Sync mink config with actual sim state
            q_actual = np.array(data.qpos, copy=True, dtype=float).ravel()
            config.update(q=q_actual)

            # Single QP solve
            try:
                vel = mink.solve_ik(
                    config, tasks, constants.MINK_DT,
                    solver=constants.MINK_QP_SOLVER,
                    limits=limits,
                )
            except Exception:
                # QP infeasible — use current joints (hold position)
                logger.debug("mink QP infeasible at step %d, holding position", steps_taken)
                vel = np.zeros(model_native.nv)

            # Integrate to get target joint positions
            q_next = config.integrate(vel, constants.MINK_DT)
            target_joints = q_next[arm_qpos_idx]

            # Send joint command
            robot_action = self.construct_joint_action(target_joints, gripper_action)
            with self.env_lock if self.env_lock else nullcontext():
                env_step(self.env, robot_action, robot_id=self.robot_id, arm=self.arm_id)

            steps_taken += 1

            # Drift-correcting sleep
            elapsed = time.time() - start_time
            target_time = steps_taken * dt
            time.sleep(max(0, target_time - elapsed))

            # Convergence check at final waypoint
            if current_wp_idx == num_waypoints - 1:
                current_pos, current_quat = self.get_eef_pose_in_base()
                final_pos = positions[-1]
                final_quat = quaternions[-1]
                pos_err = np.linalg.norm(final_pos - current_pos)
                ori_err = np.linalg.norm(
                    (R.from_quat(final_quat) * R.from_quat(current_quat).inv()).as_rotvec()
                )
                if pos_err < threshold and ori_err < threshold:
                    logger.debug("mink trajectory converged at step %d", steps_taken)
                    self.last_status = 1
                    return True

            # Log progress
            if steps_taken % 20 == 0:
                current_pos, current_quat = self.get_eef_pose_in_base()
                pos_err = np.linalg.norm(positions[current_wp_idx] - current_pos)
                logger.debug(
                    "mink step %d/%d, wp %d/%d, pos_err=%.4f",
                    steps_taken, max_steps, current_wp_idx, num_waypoints - 1, pos_err,
                )

        # Final report
        current_pos, current_quat = self.get_eef_pose_in_base()
        pos_err = np.linalg.norm(positions[-1] - current_pos)
        ori_err = np.linalg.norm(
            (R.from_quat(quaternions[-1]) * R.from_quat(current_quat).inv()).as_rotvec()
        )
        if pos_err < threshold and ori_err < threshold:
            self.last_status = 1
            logger.info(
                "ROBOT %d:%s - mink trajectory converged: pos_err=%.4f, ori_err=%.4f",
                self.robot_id, self.arm_id, pos_err, ori_err,
            )
            return True

        self.last_status = 0
        logger.info(
            "ROBOT %d:%s - mink trajectory did NOT converge: pos_err=%.4f, ori_err=%.4f",
            self.robot_id, self.arm_id, pos_err, ori_err,
        )
        return False

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
        
        # Convert to SE(3) matrices for distance check
        T_start = pose_to_SE3(start_pos, start_quat)
        T_target = pose_to_SE3(target_pos, target_quat)
        
        # Compute relative transformation to check distance
        T_start_inv = np.eye(4)
        T_start_inv[:3, :3] = T_start[:3, :3].T
        T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]
        
        T_rel = T_start_inv @ T_target
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
        s_trajectory = quintic_scalar(num_points)
        
        # Generate trajectory waypoints using the abstracted interpolation function
        positions = []
        quaternions = []
        
        for s in s_trajectory:
            pos, quat = interpolate_pose(start_pos, start_quat, target_pos, target_quat, s)
            positions.append(pos)
            quaternions.append(quat)
        
        return np.array(positions), np.array(quaternions)

    def smoothen_trajectory(self, trajectory: List[StampedPose], num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoothens a trajectory by applying a quintic polynomial timing law (velocity profile).
        The resulting trajectory will have 'num_steps' waypoints, starting and ending at rest
        with zero acceleration (jerk-limited).
        
        Args:
            trajectory: List of waypoints to smoothen.
            num_steps: Number of waypoints in the output trajectory.
            
        Returns:
            positions: Smoothened position array (N, 3).
            quaternions: Smoothened quaternion array (N, 4).
        """
        if not trajectory:
            return np.array([]), np.array([])
        
        # Ensure all waypoints are in the same frame (base_frame)
        base_trajectory = []
        for pose in trajectory:
            if pose.frame_id != self.base_frame:
                base_trajectory.append(self.tf_manager.transform_pose(pose, self.base_frame))
            else:
                base_trajectory.append(pose)
        
        if len(base_trajectory) == 1:
            pos = np.tile(base_trajectory[0].position, (num_steps, 1))
            quat = np.tile(base_trajectory[0].quaternion, (num_steps, 1))
            return pos, quat

        # 1. Extract positions and quaternions
        positions = np.array([p.position for p in base_trajectory])
        quats = np.array([p.quaternion for p in base_trajectory])
        
        # 2. Compute cumulative distance along the path for interpolation
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_dist = cumulative_dist[-1]
        
        if total_dist < 1e-8:
            # All positions are the same, use index-based normalization for rotation smoothening
            cumulative_dist = np.linspace(0, 1, len(base_trajectory))
            total_dist = 1.0

        # 3. Generate quintic polynomial interpolation parameters s(t) in [0, total_dist]
        s_profile = quintic_scalar(num_steps) * total_dist
        
        # 4. Interpolate poses along the path using the smoothened distance profile
        smooth_pos = []
        smooth_quat = []
        
        for s in s_profile:
            # Find the segment that contains distance s
            idx = np.searchsorted(cumulative_dist, s)
            
            if idx == 0:
                smooth_pos.append(positions[0])
                smooth_quat.append(quats[0])
            elif idx >= len(cumulative_dist):
                smooth_pos.append(positions[-1])
                smooth_quat.append(quats[-1])
            else:
                # Interpolate linearly for position and SLERP for rotation within the segment
                d0 = cumulative_dist[idx-1]
                d1 = cumulative_dist[idx]
                
                # Local interpolation parameter within segment [0, 1]
                alpha = (s - d0) / (d1 - d0) if (d1 - d0) > 1e-9 else 0.0
                
                p_interp, q_interp = interpolate_pose(positions[idx-1], quats[idx-1], positions[idx], quats[idx], alpha)
                
                smooth_pos.append(p_interp)
                smooth_quat.append(q_interp)
                
        return np.array(smooth_pos), np.array(smooth_quat)

    def cancel_movement(self) -> None:
        """Cancel the current movement."""
        logger.info(f"Cancelling movement for robot{self.robot_id}_{self.arm_id}")
        self._stop_event.set()

    def move_to_pose(self, target_pose: StampedPose, 
                    threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD, 
                    gripper_action: float = 0.0,
                    dt: float = constants.TRAJECTORY_TIME_STEP,
                    num_steps: int = constants.TRAJECTORY_NUM_POINTS,
                    q_warm_start: Optional[np.ndarray] = None) -> bool:
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
            dt: Time step between waypoints (seconds).
            num_steps: Number of waypoints to generate for the trajectory.
            q_warm_start: Optional warm-start joint config for IK (e.g. from planning NPZ).
                          Helps IK converge for hard-to-reach poses at workspace limits.
        
        Returns:
            bool: True if pose was reached within threshold, False otherwise
        """
        self.is_moving = True
        try:
            # Clear stop event at start
            self._stop_event.clear()

            # Transform to robot base frame if needed
            if target_pose.frame_id != self.base_frame:
                logger.debug(f"Transforming target pose from '{target_pose.frame_id}' to '{self.base_frame}'")
                target_pose = self.tf_manager.transform_pose(target_pose, self.base_frame)
            
            target_pos = np.asarray(target_pose.position)
            target_quat = np.asarray(target_pose.quaternion)
            
            current_pos, current_quat = self.get_eef_pose_in_base()
            
            logger.info(f"Planning trajectory for robot{self.robot_id}_{self.arm_id}:")
            logger.info(f"  Current EEF in base: pos={current_pos}")
            logger.info(f"  Target in base: pos={target_pos}")
            logger.info(f"  Distance to target: {np.linalg.norm(target_pos - current_pos):.4f}m")

            # Ensure at least 2 points (start/end)
            traj_num_points = max(2, num_steps)
            logger.debug(f"Adjusting trajectory points to {traj_num_points}")
                             
            positions, quaternions = self.plan_quintic_trajectory(
                current_pos, current_quat, target_pos, target_quat,
                num_points=traj_num_points
            )

            # Visualize the SE(3) trajectory if enabled
            if self.visualize:
                self._clear_trajectory_visualization()
                self._visualize_se3_trajectory(positions, quaternions)

            if len(positions) <= 1:
                self.last_status = 1
                return True  # Already at target

            # ── Mink reactive path for JOINT_POSITION controllers ─────
            # Uses a single mink QP per tick instead of full IK per step.
            if self.is_joint_position_controller:
                return self._execute_mink_trajectory(
                    positions, quaternions,
                    gripper_action=gripper_action,
                    dt=dt,
                    threshold=threshold,
                )

            # ── OSC path (delta/absolute operational space control) ────
            # Trajectory execution: commands are generated based on controller configuration
            # For delta mode: compute delta from current position to next waypoint
            # For absolute mode: send target position directly
            num_waypoints = len(positions)
            max_steps = num_waypoints * 2
            
            # Track which waypoint we're currently targeting
            # Start at 0, will increment to 1 on first step
            current_waypoint_idx = 0 
            steps_taken = 0
            start_time = time.time()
            
            # Sequential IK warm-starting: use caller's hint for the first
            # call, then chain through current joint positions so each IK
            # warm-starts from the previous solution (like the planning pipeline).
            _ik_ws = q_warm_start
            
            while steps_taken < max_steps:
                # Check for cancellation
                if self._stop_event.is_set():
                    logger.info(f"Movement cancelled for robot{self.robot_id}_{self.arm_id}")
                    self.last_status = 0
                    return False

                # Advance waypoint index to follow time-based trajectory
                if current_waypoint_idx < num_waypoints - 1:
                    current_waypoint_idx += 1
                    
                # Get target waypoint
                target_pos_step = positions[current_waypoint_idx]
                target_quat_step = quaternions[current_waypoint_idx]
                
                # Get current pose
                current_pos, current_quat = self.get_eef_pose_in_base()
                
                # Compute error (used for convergence check and delta mode)
                pos_delta = target_pos_step - current_pos
                ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
                ori_delta = ori_error.as_rotvec()
                
                # Construct action using helper method (handles delta/absolute mode)
                robot_action = self.construct_robot_action(
                    target_pos_step, target_quat_step,
                    gripper_action=gripper_action,
                    current_pos=current_pos, current_quat=current_quat,
                    q_warm_start=_ik_ws
                )
                
                if robot_action is None:
                    # IK failed for this waypoint; skip it
                    logger.debug(f"IK failed at waypoint {current_waypoint_idx}, skipping")
                    steps_taken += 1
                    continue
                
                # Step the environment
                with self.env_lock if self.env_lock else nullcontext():
                    env_step(self.env, robot_action, robot_id=self.robot_id, arm=self.arm_id)
                
                # Chain IK: next call uses current joints (≈ previous IK solution)
                _ik_ws = None
                
                steps_taken += 1
                
                # Drift-correcting sleep
                elapsed = time.time() - start_time
                target_time = steps_taken * dt
                sleep_time = max(0, target_time - elapsed)
                time.sleep(sleep_time)
                
                # Check convergence if we are targeting the final waypoint
                if current_waypoint_idx == num_waypoints - 1:
                    waypoint_pos_error = np.linalg.norm(pos_delta)
                    waypoint_ori_error = np.linalg.norm(ori_delta)
                    
                    if waypoint_pos_error < threshold and waypoint_ori_error < threshold:
                        logger.debug(f"Converged at step {steps_taken}")
                        break

                # Log progress periodically
                if steps_taken % 20 == 0:
                    current_pos_error = np.linalg.norm(pos_delta)
                    current_ori_error = np.linalg.norm(ori_delta)
                    logger.debug(f"Step {steps_taken}/{max_steps}, waypoint {current_waypoint_idx}/{num_waypoints-1}, "
                            f"pos_err={current_pos_error:.4f}m, ori_err={current_ori_error:.4f}rad")
            
            # Final check: verify we reached the target
            final_pos, final_quat = self.get_eef_pose_in_base()
            
            pos_error = target_pos - final_pos
            ori_error = R.from_quat(target_quat) * R.from_quat(final_quat).inv()
            ori_error_vec = ori_error.as_rotvec()
            
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error_vec)
            
            if pos_error_norm < threshold and ori_error_norm < threshold:
                self.last_status = 1
                logger.info(f"ROBOT: {self.robot_id}:{self.arm_id} - Trajectory converged: final pos_err={pos_error_norm:.4f}m, ori_err={ori_error_norm:.4f}rad")
                return True
            
            self.last_status = 0
            logger.info(f"ROBOT: {self.robot_id}:{self.arm_id} - Trajectory didn't converge: final pos_err={pos_error_norm:.4f}m, ori_err={ori_error_norm:.4f}rad")
            return False
        finally:
            self.is_moving = False

    def follow_trajectory(self, trajectory: List[StampedPose], 
                          gripper_action: float = 0.0,
                          dt: float = constants.TRAJECTORY_TIME_STEP,
                          threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
                          num_steps: Optional[int] = None,
                          smoothen: bool = False,
                          q_warm_start: Optional[np.ndarray] = None) -> bool:
        """
        Follows a pre-defined end-effector trajectory in constants.WORLD_FRAME.
        
        Args:
            trajectory: List of StampedPose waypoints in world frame.
            gripper_action: Gripper action to maintain.
            dt: Time step between waypoints.
            threshold: Convergence threshold for the final waypoint.
            num_steps: Optional maximum number of steps. Defaults to len(trajectory) * 2.
            smoothen: Whether to smoothen the trajectory using a quintic polynomial.
            q_warm_start: Optional warm-start joint config for IK at workspace limits.
            
        Returns:
            bool: True if final waypoint reached, False otherwise.
        """
        self.is_moving = True
        try:
            self._stop_event.clear()
            
            if not trajectory:
                self.last_status = 1
                return True

            logger.info(f"Following {'smoothened ' if smoothen else ''}pre-defined trajectory for robot{self.robot_id}_{self.arm_id} ({len(trajectory)} waypoints)")
            
            if smoothen:
                traj_num_steps = num_steps if num_steps is not None else len(trajectory)
                positions, quaternions = self.smoothen_trajectory(trajectory, num_steps=traj_num_steps)
            else:
                # Transform all waypoints to robot base frame
                base_trajectory = []
                for i, pose in enumerate(trajectory):
                    if pose.frame_id != self.base_frame:
                        base_pose = self.tf_manager.transform_pose(pose, self.base_frame)
                        base_trajectory.append(base_pose)
                    else:
                        base_trajectory.append(pose)
                
                # Extract positions and quaternions
                positions = np.array([p.position for p in base_trajectory])
                quaternions = np.array([p.quaternion for p in base_trajectory])
            
            if self.visualize:
                self._clear_trajectory_visualization()
                self._visualize_se3_trajectory(positions, quaternions)
            
            # ── Mink reactive path for JOINT_POSITION controllers ─────
            if self.is_joint_position_controller:
                return self._execute_mink_trajectory(
                    positions, quaternions,
                    gripper_action=gripper_action,
                    dt=dt,
                    threshold=threshold,
                )

            # ── OSC path ──────────────────────────────────────────────
            num_waypoints = len(positions)
            max_steps = num_steps * 2 if num_steps is not None else num_waypoints * 2
            
            current_waypoint_idx = 0
            steps_taken = 0
            start_time = time.time()
            
            # Sequential IK warm-starting (see move_to_pose for rationale)
            _ik_ws = q_warm_start
            
            while steps_taken < max_steps:
                if self._stop_event.is_set():
                    logger.info(f"Trajectory following cancelled for robot{self.robot_id}_{self.arm_id}")
                    self.last_status = 0
                    return False
                
                # Target the next waypoint in the trajectory
                if current_waypoint_idx < num_waypoints - 1:
                    current_waypoint_idx += 1
                
                target_pos_step = positions[current_waypoint_idx]
                target_quat_step = quaternions[current_waypoint_idx]
                
                current_pos, current_quat = self.get_eef_pose_in_base()
                
                # Compute error for convergence check
                pos_delta = target_pos_step - current_pos
                ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
                ori_delta = ori_error.as_rotvec()
                
                # Construct action using helper method (handles delta/absolute mode)
                robot_action = self.construct_robot_action(
                    target_pos_step, target_quat_step,
                    gripper_action=gripper_action,
                    current_pos=current_pos, current_quat=current_quat,
                    q_warm_start=_ik_ws
                )
                
                if robot_action is None:
                    # IK failed for this waypoint; skip it
                    logger.debug(f"IK failed at waypoint {current_waypoint_idx}, skipping")
                    steps_taken += 1
                    continue
                
                with self.env_lock if self.env_lock else nullcontext():
                    env_step(self.env, robot_action, robot_id=self.robot_id, arm=self.arm_id)
                
                # Chain IK: next call uses current joints (≈ previous IK solution)
                _ik_ws = None
                
                steps_taken += 1
                
                elapsed = time.time() - start_time
                target_time = steps_taken * dt
                sleep_time = max(0, target_time - elapsed)
                time.sleep(sleep_time)
                
                # Log progress periodically
                if steps_taken % 20 == 0:
                    logger.debug(
                        f"Step {steps_taken}/{max_steps}, waypoint {current_waypoint_idx}/{num_waypoints-1}, "
                        f"pos_err={np.linalg.norm(pos_delta):.4f}m, ori_err={np.linalg.norm(ori_delta):.4f}rad"
                    )

                # Check convergence if targeting the final waypoint
                if current_waypoint_idx == num_waypoints - 1:
                    if np.linalg.norm(pos_delta) < threshold and np.linalg.norm(ori_delta) < threshold:
                        logger.debug(f"ROBOT: {self.robot_id}:{self.arm_id} - Trajectory converged at step {steps_taken}")
                        self.last_status = 1
                        return True
            
            self.last_status = 0
            logger.info(f"ROBOT: {self.robot_id}:{self.arm_id} - Trajectory didn't converge after {steps_taken} steps, final pos_err={np.linalg.norm(pos_delta):.4f}m, ori_err={np.linalg.norm(ori_delta):.4f}rad")
            return False
        finally:
            self.is_moving = False

    def move_to_joint_config(
        self,
        target_joints: np.ndarray,
        gripper_action: float = 0.0,
        num_steps: int = constants.TRAJECTORY_NUM_POINTS,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.JOINT_CONVERGENCE_THRESHOLD,
    ) -> bool:
        """
        Move the arm to a target joint configuration using a quintic trajectory.

        Only supported when the controller is ``JOINT_POSITION``.

        Args:
            target_joints: Target joint angles (7,).
            gripper_action: Gripper action value.
            num_steps: Number of trajectory steps for the quintic plan.
            dt: Time step between steps.
            threshold: Max joint-space L-inf error for convergence (rad).

        Returns:
            True if converged, False otherwise.

        Raises:
            RuntimeError: If the controller is not JOINT_POSITION.
        """
        if not self.is_joint_position_controller:
            raise RuntimeError(
                "move_to_joint_config() requires a JOINT_POSITION controller. "
                f"Current controller: {self.controller_name}"
            )

        target_joints = np.asarray(target_joints, dtype=np.float64)
        if target_joints.shape != (7,):
            raise ValueError(f"target_joints must be shape (7,), got {target_joints.shape}")

        self.is_moving = True
        try:
            self._stop_event.clear()

            traj_num_steps = max(2, num_steps)  # Ensure at least start and end points

            current_joints = self.get_joint_positions()
            trajectory = plan_quintic_joint_trajectory(
                current_joints, target_joints, traj_num_steps
            )

            total_waypoints = len(trajectory)
            max_steps = total_waypoints * 2

            logger.info(
                f"ROBOT {self.robot_id}:{self.arm_id} - Moving to joint config "
                f"({total_waypoints} waypoints, max {max_steps} steps)"
            )

            current_wp_idx = 0
            steps_taken = 0
            start_time = time.time()

            while steps_taken < max_steps:
                if self._stop_event.is_set():
                    logger.info(
                        f"move_to_joint_config cancelled for "
                        f"robot{self.robot_id}_{self.arm_id}"
                    )
                    self.last_status = 0
                    return False

                if current_wp_idx < total_waypoints - 1:
                    current_wp_idx += 1

                target_wp = trajectory[current_wp_idx]
                robot_action = self.construct_joint_action(
                    target_wp, gripper_action=gripper_action
                )

                with self.env_lock if self.env_lock else nullcontext():
                    env_step(
                        self.env,
                        robot_action,
                        robot_id=self.robot_id,
                        arm=self.arm_id,
                    )

                steps_taken += 1

                elapsed = time.time() - start_time
                target_time = steps_taken * dt
                time.sleep(max(0, target_time - elapsed))

                # Log progress periodically
                if steps_taken % 20 == 0:
                    cur_joints = self.get_joint_positions()
                    cur_error = np.max(np.abs(cur_joints - target_joints))
                    logger.debug(
                        f"Step {steps_taken}/{max_steps}, waypoint {current_wp_idx}/{total_waypoints-1}, "
                        f"max joint err={cur_error:.4f} rad"
                    )

                # Convergence check at last waypoint
                if current_wp_idx == total_waypoints - 1:
                    cur_joints = self.get_joint_positions()
                    joint_error = np.max(np.abs(cur_joints - target_joints))
                    if joint_error < threshold:
                        self.last_status = 1
                        logger.info(
                            f"ROBOT {self.robot_id}:{self.arm_id} - "
                            f"Joint config converged at step {steps_taken} "
                            f"(max err={joint_error:.4f} rad)"
                        )
                        return True

            # Final convergence report
            final_joints = self.get_joint_positions()
            final_error = np.max(np.abs(final_joints - target_joints))
            self.last_status = 0
            logger.info(
                f"ROBOT {self.robot_id}:{self.arm_id} - "
                f"Joint config did NOT converge after {steps_taken} steps "
                f"(max err={final_error:.4f} rad)"
            )
            return False
        finally:
            self.is_moving = False

    def follow_joint_trajectory(
        self,
        joint_trajectory: List[np.ndarray],
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.JOINT_CONVERGENCE_THRESHOLD,
        num_steps: Optional[int] = None,
        smoothen: bool = False,
    ) -> bool:
        """
        Follow a pre-defined trajectory in joint space.

        Only supported when the controller is ``JOINT_POSITION``.

        Args:
            joint_trajectory: List of joint configurations, each shape (7,).
            gripper_action: Gripper action value.
            dt: Time step between waypoints.
            threshold: Convergence threshold (L-inf in radians) at the final
                waypoint.
            num_steps: Maximum number of environment steps.  Defaults to
                ``len(joint_trajectory) * 2``.
            smoothen: If True, re-plan using a quintic polynomial between the
                first and last waypoints (ignoring intermediate waypoints).

        Returns:
            True if the final waypoint is reached, False otherwise.

        Raises:
            RuntimeError: If the controller is not JOINT_POSITION.
        """
        if not self.is_joint_position_controller:
            raise RuntimeError(
                "follow_joint_trajectory() requires a JOINT_POSITION controller. "
                f"Current controller: {self.controller_name}"
            )

        self.is_moving = True
        try:
            self._stop_event.clear()

            if len(joint_trajectory) == 0:
                self.last_status = 1
                return True

            # Convert to numpy
            waypoints = [np.asarray(jt, dtype=np.float64) for jt in joint_trajectory]
            n_waypoints = len(waypoints)

            if smoothen:
                traj_steps = num_steps if num_steps is not None else n_waypoints
                if n_waypoints > 2:
                    # Multi-waypoint: apply quintic velocity profiling along
                    # the existing path (preserves shape, smooths timing)
                    joint_configs = velocity_profile_joint_trajectory(
                        np.array(waypoints), num_steps=traj_steps,
                    )
                else:
                    # Two-point: simple A→B quintic interpolation
                    current_joints = self.get_joint_positions()
                    joint_configs = plan_quintic_joint_trajectory(
                        current_joints, waypoints[-1], traj_steps
                    )
            else:
                joint_configs = np.array(waypoints)

            total_waypoints = len(joint_configs)
            max_steps = num_steps * 2 if num_steps is not None else total_waypoints * 2

            logger.info(
                f"Following joint trajectory for robot{self.robot_id}_{self.arm_id} "
                f"({total_waypoints} waypoints, max {max_steps} steps)"
            )

            current_wp_idx = 0
            steps_taken = 0
            start_time = time.time()

            while steps_taken < max_steps:
                if self._stop_event.is_set():
                    logger.info(
                        f"Joint trajectory cancelled for "
                        f"robot{self.robot_id}_{self.arm_id}"
                    )
                    self.last_status = 0
                    return False

                if current_wp_idx < total_waypoints - 1:
                    current_wp_idx += 1

                target_wp = joint_configs[current_wp_idx]
                robot_action = self.construct_joint_action(
                    target_wp, gripper_action=gripper_action
                )

                with self.env_lock if self.env_lock else nullcontext():
                    env_step(
                        self.env,
                        robot_action,
                        robot_id=self.robot_id,
                        arm=self.arm_id,
                    )

                steps_taken += 1

                elapsed = time.time() - start_time
                target_time = steps_taken * dt
                time.sleep(max(0, target_time - elapsed))

                # Log progress periodically
                if steps_taken % 20 == 0:
                    current_joints = self.get_joint_positions()
                    joint_error = np.max(np.abs(current_joints - target_wp))
                    logger.debug(
                        f"Step {steps_taken}/{max_steps}, waypoint {current_wp_idx}/{total_waypoints-1}, "
                        f"max joint err={joint_error:.4f} rad"
                    )

                # Convergence check at last waypoint
                if current_wp_idx == total_waypoints - 1:
                    current_joints = self.get_joint_positions()
                    joint_error = np.max(np.abs(current_joints - target_wp))
                    if joint_error < threshold:
                        self.last_status = 1
                        logger.info(
                            f"ROBOT {self.robot_id}:{self.arm_id} - "
                            f"Joint trajectory converged at step {steps_taken} "
                            f"(max err={joint_error:.4f} rad)"
                        )
                        return True

            # Final convergence report
            final_joints = self.get_joint_positions()
            final_error = np.max(np.abs(final_joints - joint_configs[-1]))
            self.last_status = 0
            logger.info(
                f"ROBOT {self.robot_id}:{self.arm_id} - "
                f"Joint trajectory did NOT converge after {steps_taken} steps "
                f"(max err={final_error:.4f} rad)"
            )
            return False
        finally:
            self.is_moving = False

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
                                   axis_length: float = 0.05, axis_radius: float = 0.003) -> None:
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
            base_to_world = self.tf_manager.get_transform(constants.WORLD_FRAME, self.base_frame)
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