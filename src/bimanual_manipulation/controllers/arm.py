"""Robosuite arm controller.

Provides a high-level interface for controlling a single robot arm in a
robosuite environment.  Supports OSC_POSE and JOINT_POSITION controllers
in both delta and absolute input modes.

Trajectory planning, execution, and visualization are delegated to
dedicated modules for modularity:

* :mod:`controllers.trajectory_planner` — quintic SE(3) planning
* :mod:`controllers.trajectory_executor` — mink/OSC/joint execution
* :mod:`controllers.visualization` — MeshCat trajectory frames
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.controllers.trajectory_executor import TrajectoryExecutor
from bimanual_manipulation.controllers.trajectory_planner import (
    plan_quintic_trajectory,
    smoothen_trajectory,
)
from bimanual_manipulation.controllers.visualization import TrajectoryVisualizer
from bimanual_manipulation.planning.utils.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction,
)
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.trajectory_math import (
    plan_quintic_joint_trajectory,
    velocity_profile_joint_trajectory,
)

if TYPE_CHECKING:
    from bimanual_manipulation.environments.dual_arm import DualArmEnvironment

logger = logging.getLogger(__name__)


class RobosuiteArmController:
    """High-level controller for a single robot arm in robosuite.

    Takes a target pose (position and orientation) and generates actions
    to move the end-effector to that pose.

    Supports multiple controller types:
    - OSC_POSE (delta/absolute): Operational space control using EEF pose
    - JOINT_POSITION (delta/absolute): Joint position control
    """

    def __init__(
        self,
        env: DualArmEnvironment,
        arm_id: str = "right",
        robot_id: int = 0,
        transform_manager: Optional[TransformManager] = None,
        env_lock: Optional[threading.RLock] = None,
        visualize: bool = False,
    ) -> None:
        self.env = env
        self.env_lock = env_lock
        self.arm_id = arm_id
        self.robot_id = robot_id
        self.dof = self._get_robot().dof
        self.visualize = visualize

        robot_name = getattr(self._get_robot(), "name", f"robot{robot_id}")
        logger.debug(
            "RobosuiteArmController initialized for robot '%s', arm '%s'",
            robot_name, arm_id,
        )

        self.base_frame = f"robot{robot_id}_base"

        self._controller_name, self._input_type = self._detect_controller_config()
        logger.debug(
            "Detected controller: %s, input_type: %s",
            self._controller_name, self._input_type,
        )

        self._arm_action_dim = constants.CONTROLLER_ACTION_DIMS.get(
            self._controller_name,
            constants.CONTROLLER_ACTION_DIMS[constants.ControllerName.OSC_POSE],
        )
        self.action_dim_per_robot = self._arm_action_dim + 1
        self.total_action_dim = self.env.action_dim

        self.tf_manager = (
            transform_manager
            if transform_manager
            else TransformManager(env, env_lock=self.env_lock)
        )

        self._stop_event = threading.Event()
        self.is_moving = False
        self.last_status = 1

        # Delegate trajectory execution and visualization
        self._executor = TrajectoryExecutor(self)
        self._visualizer = TrajectoryVisualizer(
            robot_id, arm_id, self.tf_manager, self.base_frame,
        )

    # ------------------------------------------------------------------
    # Controller detection
    # ------------------------------------------------------------------

    def _detect_controller_config(self) -> Tuple[str, str]:
        """Detect controller name and input type from the robot."""
        controller_name = constants.DEFAULT_CONTROLLER_NAME
        input_type = constants.DEFAULT_INPUT_TYPE

        try:
            robot = self._get_robot()
            if hasattr(robot, "part_controllers") and robot.part_controllers:
                for _arm_name, controller in robot.part_controllers.items():
                    if hasattr(controller, "name"):
                        controller_name = controller.name
                    if hasattr(controller, "input_type"):
                        input_type = controller.input_type
                    break
        except Exception as e:
            logger.debug("Could not detect controller config: %s", e)

        return controller_name, input_type

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def controller_name(self) -> str:
        return self._controller_name

    @property
    def input_type(self) -> str:
        return self._input_type

    @property
    def is_absolute_mode(self) -> bool:
        return self._input_type == constants.InputType.ABSOLUTE

    @property
    def is_joint_position_controller(self) -> bool:
        return self._controller_name == constants.ControllerName.JOINT_POSITION

    # ------------------------------------------------------------------
    # Robot accessors
    # ------------------------------------------------------------------

    def _get_robot(self):
        """Get fresh robot reference from environment."""
        return self.env.robots[self.robot_id]

    def _eef_body_name(self) -> str:
        return self._get_robot().robot_model.eef_name[self.arm_id]

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions of the arm (7,)."""
        robot = self._get_robot()
        try:
            if hasattr(robot, f"{self.arm_id}_arm_joint_indexes"):
                joint_indices = getattr(robot, f"{self.arm_id}_arm_joint_indexes")
            elif hasattr(robot, "arm_joint_indexes"):
                joint_indices = robot.arm_joint_indexes
            else:
                joint_indices = robot.joint_indexes
            return self.env.sim.data.qpos[joint_indices].copy()
        except Exception as e:
            logger.warning("Could not read joint positions: %s", e)
            return constants.FRANKA_HOME_JOINT_POSITION.copy()

    def get_eef_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current EEF pose in robot base frame.

        Returns:
            ``(pos, quat)`` — position ``(3,)`` and quaternion ``(x, y, z, w)``.
        """
        eef_frame = f"robot{self.robot_id}_{self.arm_id}_eef"
        transform = self.tf_manager.get_transform(self.base_frame, eef_frame)
        return transform.translation.copy(), transform.rotation.copy()

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _solve_ik_for_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solve IK for a target EEF pose using EAIK analytical solver.

        Args:
            target_pos: Target position ``[x, y, z]`` in robot base frame.
            target_quat: Target quaternion ``[x, y, z, w]`` in robot base frame.

        Returns:
            Joint configuration ``(7,)`` or ``None`` if IK fails.
        """
        target_stamped = StampedPose(
            position=target_pos.tolist(),
            quaternion=target_quat.tolist(),
            frame_id=self.base_frame,
        )

        target_stamped = apply_ik_orientation_correction(target_stamped)
        site_name = f"gripper{self.robot_id}_{self.arm_id}_ft_frame"

        solutions = enumerate_ik_solutions_for_arm(
            self.env,
            robot_id=self.robot_id,
            target_pose=target_stamped,
            tf_manager=self.tf_manager,
            arm=self.arm_id,
            site_name=site_name,
            env_lock=self.env_lock,
            max_solutions=1,
        )

        if not solutions:
            logger.warning(
                "No IK solution for robot%d_%s at pos=%s, quat=%s",
                self.robot_id, self.arm_id, target_pos, target_quat,
            )
            return None

        return solutions[0]

    # ------------------------------------------------------------------
    # Action construction
    # ------------------------------------------------------------------

    def construct_robot_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_action: float = 0.0,
        current_pos: Optional[np.ndarray] = None,
        current_quat: Optional[np.ndarray] = None,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Construct a robot action for the current controller configuration.

        For OSC_POSE: computes delta or absolute pose commands.
        For JOINT_POSITION: solves IK and constructs joint commands.

        Args:
            target_pos: Target position ``[x, y, z]`` in base frame.
            target_quat: Target quaternion ``[x, y, z, w]`` in base frame.
            gripper_action: Gripper action value.
            current_pos: Current position (reads from env if None).
            current_quat: Current quaternion (reads from env if None).
            q_warm_start: Optional warm-start for IK.

        Returns:
            Robot action array or ``None`` if IK fails.
        """
        if self.is_joint_position_controller:
            target_joints = self._solve_ik_for_pose(
                target_pos, target_quat,
            )
            if target_joints is None:
                return None
            return self.construct_joint_action(target_joints, gripper_action)

        if current_pos is None or current_quat is None:
            current_pos, current_quat = self.get_eef_pose_in_base()

        robot_action = np.zeros(self.action_dim_per_robot)

        if self.is_absolute_mode:
            target_rotvec = R.from_quat(target_quat).as_rotvec()
            robot_action[:3] = target_pos
            robot_action[3:6] = target_rotvec
        else:
            pos_delta = target_pos - current_pos
            ori_error = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
            robot_action[:3] = pos_delta
            robot_action[3:6] = ori_error.as_rotvec()

        robot_action[-1] = gripper_action
        return robot_action

    def construct_joint_action(
        self,
        target_joints: np.ndarray,
        gripper_action: float = 0.0,
        current_joints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Construct a JOINT_POSITION action with velocity clamping.

        Args:
            target_joints: Target joint positions ``[j1, ..., j7]``.
            gripper_action: Gripper action value.
            current_joints: Current joints (reads from env if None).

        Returns:
            Robot action array.
        """
        if not self.is_joint_position_controller:
            raise NotImplementedError(
                f"construct_joint_action requires JOINT_POSITION controller. "
                f"Current: {self._controller_name}"
            )

        if current_joints is None:
            current_joints = self.get_joint_positions()

        joint_delta = target_joints - current_joints
        max_delta = constants.MAX_JOINT_DELTA_PER_STEP
        clamped_delta = np.clip(joint_delta, -max_delta, max_delta)
        clamped_target = current_joints + clamped_delta

        robot_action = np.zeros(self.action_dim_per_robot)
        if self.is_absolute_mode:
            robot_action[:7] = clamped_target
        else:
            robot_action[:7] = clamped_delta
        robot_action[-1] = gripper_action
        return robot_action

    # ------------------------------------------------------------------
    # Movement cancellation
    # ------------------------------------------------------------------

    def cancel_movement(self) -> None:
        """Cancel the current movement."""
        logger.info("Cancelling movement for robot%d_%s", self.robot_id, self.arm_id)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # High-level motion commands
    # ------------------------------------------------------------------

    def move_to_pose(
        self,
        target_pose: StampedPose,
        threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        num_steps: int = constants.TRAJECTORY_NUM_POINTS,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> bool:
        """Move the EEF to a target pose via quintic trajectory.

        Args:
            target_pose: Desired pose (any frame).
            threshold: Convergence threshold.
            gripper_action: Gripper command during motion.
            dt: Time step between waypoints.
            num_steps: Number of trajectory waypoints.
            q_warm_start: Optional warm-start for IK.

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        self.is_moving = True
        try:
            self._stop_event.clear()

            if target_pose.frame_id != self.base_frame:
                target_pose = self.tf_manager.transform_pose(
                    target_pose, self.base_frame,
                )

            target_pos = np.asarray(target_pose.position)
            target_quat = np.asarray(target_pose.quaternion)

            current_pos, current_quat = self.get_eef_pose_in_base()

            logger.info(
                "Planning trajectory for robot%d_%s: "
                "current=%s, target=%s, dist=%.4f",
                self.robot_id, self.arm_id,
                current_pos, target_pos,
                np.linalg.norm(target_pos - current_pos),
            )

            traj_num_points = max(2, num_steps)
            positions, quaternions = plan_quintic_trajectory(
                current_pos, current_quat, target_pos, target_quat,
                num_points=traj_num_points,
            )

            if self.visualize:
                self._visualizer.clear()
                self._visualizer.visualize_se3_trajectory(positions, quaternions)

            if len(positions) <= 1:
                self.last_status = 1
                return True

            if self.is_joint_position_controller:
                return self._executor.execute_mink_trajectory(
                    positions, quaternions,
                    gripper_action=gripper_action,
                    dt=dt, threshold=threshold,
                )

            return self._executor.execute_osc_trajectory(
                positions, quaternions,
                gripper_action=gripper_action,
                dt=dt, threshold=threshold,
                q_warm_start=q_warm_start,
            )
        finally:
            self.is_moving = False

    def follow_trajectory(
        self,
        trajectory: List[StampedPose],
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
        num_steps: Optional[int] = None,
        smoothen: bool = False,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> bool:
        """Follow a pre-defined EEF trajectory.

        Args:
            trajectory: List of ``StampedPose`` waypoints.
            gripper_action: Gripper action during motion.
            dt: Time step between waypoints.
            threshold: Convergence threshold for final waypoint.
            num_steps: Optional max steps.
            smoothen: Apply quintic smoothing.
            q_warm_start: Optional warm-start for IK.

        Returns:
            ``True`` if reached, ``False`` otherwise.
        """
        self.is_moving = True
        try:
            self._stop_event.clear()

            if not trajectory:
                self.last_status = 1
                return True

            logger.info(
                "Following %strajectory for robot%d_%s (%d waypoints)",
                "smoothened " if smoothen else "",
                self.robot_id, self.arm_id, len(trajectory),
            )

            if smoothen:
                traj_num_steps = num_steps if num_steps is not None else len(trajectory)
                positions, quaternions = smoothen_trajectory(
                    trajectory, traj_num_steps,
                    self.tf_manager, self.base_frame,
                )
            else:
                base_trajectory = []
                for pose in trajectory:
                    if pose.frame_id != self.base_frame:
                        base_trajectory.append(
                            self.tf_manager.transform_pose(pose, self.base_frame),
                        )
                    else:
                        base_trajectory.append(pose)

                positions = np.array([p.position for p in base_trajectory])
                quaternions = np.array([p.quaternion for p in base_trajectory])

            if self.visualize:
                self._visualizer.clear()
                self._visualizer.visualize_se3_trajectory(positions, quaternions)

            if self.is_joint_position_controller:
                return self._executor.execute_mink_trajectory(
                    positions, quaternions,
                    gripper_action=gripper_action,
                    dt=dt, threshold=threshold,
                )

            return self._executor.execute_osc_trajectory(
                positions, quaternions,
                gripper_action=gripper_action,
                dt=dt, threshold=threshold,
                q_warm_start=q_warm_start,
            )
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
        """Move to a target joint configuration via quintic trajectory.

        Only supported with ``JOINT_POSITION`` controller.

        Args:
            target_joints: Target joint angles ``(7,)``.
            gripper_action: Gripper action value.
            num_steps: Trajectory steps.
            dt: Time step.
            threshold: Joint-space convergence threshold (rad).

        Returns:
            ``True`` if converged, ``False`` otherwise.

        Raises:
            RuntimeError: If not using JOINT_POSITION controller.
        """
        if not self.is_joint_position_controller:
            raise RuntimeError(
                f"move_to_joint_config() requires JOINT_POSITION controller. "
                f"Current: {self.controller_name}"
            )

        target_joints = np.asarray(target_joints, dtype=np.float64)
        if target_joints.shape != (7,):
            raise ValueError(
                f"target_joints must be shape (7,), got {target_joints.shape}"
            )

        self.is_moving = True
        try:
            self._stop_event.clear()
            traj_num_steps = max(2, num_steps)

            if traj_num_steps == 2:
                robot_action = self.construct_joint_action(
                    target_joints, gripper_action=gripper_action,
                )
                with self.env_lock if self.env_lock else nullcontext():
                    env_step(
                        self.env, robot_action,
                        robot_id=self.robot_id, arm=self.arm_id,
                    )
                return True

            current_joints = self.get_joint_positions()
            trajectory = plan_quintic_joint_trajectory(
                current_joints, target_joints, traj_num_steps,
            )

            logger.info(
                "ROBOT %d:%s - Moving to joint config (%d waypoints)",
                self.robot_id, self.arm_id, len(trajectory),
            )

            return self._executor.execute_joint_trajectory(
                trajectory, target_joints,
                gripper_action=gripper_action,
                dt=dt, threshold=threshold,
            )
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
        """Follow a pre-defined joint-space trajectory.

        Only supported with ``JOINT_POSITION`` controller.

        Args:
            joint_trajectory: List of joint configs, each ``(7,)``.
            gripper_action: Gripper action value.
            dt: Time step between waypoints.
            threshold: Convergence threshold (rad).
            num_steps: Max environment steps.
            smoothen: Apply quintic velocity profiling.

        Returns:
            ``True`` if converged, ``False`` otherwise.

        Raises:
            RuntimeError: If not using JOINT_POSITION controller.
        """
        if not self.is_joint_position_controller:
            raise RuntimeError(
                f"follow_joint_trajectory() requires JOINT_POSITION controller. "
                f"Current: {self.controller_name}"
            )

        self.is_moving = True
        try:
            self._stop_event.clear()

            if not joint_trajectory:
                self.last_status = 1
                return True

            waypoints = [np.asarray(jt, dtype=np.float64) for jt in joint_trajectory]
            n_waypoints = len(waypoints)

            if smoothen:
                traj_steps = num_steps if num_steps is not None else n_waypoints
                if n_waypoints > 2:
                    joint_configs = velocity_profile_joint_trajectory(
                        np.array(waypoints), num_steps=traj_steps,
                    )
                else:
                    current_joints = self.get_joint_positions()
                    joint_configs = plan_quintic_joint_trajectory(
                        current_joints, waypoints[-1], traj_steps,
                    )
            else:
                joint_configs = np.array(waypoints)

            max_steps_val = (
                num_steps * 2 if num_steps is not None else len(joint_configs) * 2
            )

            logger.info(
                "Following joint trajectory for robot%d_%s (%d waypoints)",
                self.robot_id, self.arm_id, len(joint_configs),
            )

            return self._executor.execute_joint_trajectory(
                joint_configs, waypoints[-1],
                gripper_action=gripper_action,
                dt=dt, threshold=threshold,
                max_steps=max_steps_val,
            )
        finally:
            self.is_moving = False
