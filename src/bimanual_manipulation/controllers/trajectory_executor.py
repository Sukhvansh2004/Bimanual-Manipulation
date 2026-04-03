"""Trajectory execution for arm controllers.

Provides mink-based reactive trajectory tracking (QP per tick) and
OSC-based trajectory following for end-effector pose trajectories.
Also provides joint-space trajectory execution.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import mink
from mink.lie import SE3 as MinkSE3, SO3 as MinkSO3
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.planning.utils import mujoco_utils as mu
from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_robot_collision_geoms,
    get_object_collision_geoms,
    get_environment_collision_geoms,
)
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.se3 import pose_to_SE3

if TYPE_CHECKING:
    from bimanual_manipulation.controllers.arm import RobosuiteArmController


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lightweight container for pre-built mink objects
# ------------------------------------------------------------------

@dataclass
class MinkContext:
    """All mink objects needed for reactive trajectory execution."""
    config: mink.Configuration
    tasks: list
    limits: list
    arm_qpos_idx: np.ndarray
    model_native: object
    world_positions: np.ndarray
    world_quaternions: np.ndarray


class TrajectoryExecutor:
    """Executes trajectories for a single arm using mink reactive tracking
    or joint-space stepping.

    This class is tightly coupled to a ``RobosuiteArmController`` instance
    and delegates action construction and environment stepping to it.
    """

    def __init__(self, controller: RobosuiteArmController) -> None:
        """
        Args:
            controller: The parent ``RobosuiteArmController``.
        """
        self._ctrl = controller

    # ------------------------------------------------------------------
    # Mink-based reactive trajectory execution
    # ------------------------------------------------------------------

    def _build_mink_context(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        q_warm_start: Optional[np.ndarray] = None,
        collision_check: bool = False,
        manipulated_object: Optional[str] = None,
    ) -> MinkContext:
        """Build all mink objects needed for reactive trajectory tracking.

        This transforms waypoints to world frame, constructs tasks
        (frame, posture, damping, freeze) and limits (velocity,
        configuration, optional collision avoidance), and returns them
        bundled in a :class:`MinkContext`.
        """
        ctrl = self._ctrl
        num_waypoints = len(positions)

        # ── Transform waypoints from base to world frame ──────────────
        try:
            base_to_world = ctrl.tf_manager.get_transform(
                constants.WORLD_FRAME, ctrl.base_frame,
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

        # Apply body→site orientation correction (-90° Z)
        r_corr = R.from_euler("z", -90, degrees=True)
        for i in range(num_waypoints):
            r_wp = R.from_quat(world_quaternions[i])
            world_quaternions[i] = (r_wp * r_corr).as_quat()

        # ── Build mink objects ────────────────────────────────────────
        model = ctrl.env.sim.model
        data = ctrl.env.sim.data
        model_native = mu.get_native_mj_object(model)

        joint_ids = mu.get_arm_joint_ids(model, ctrl.robot_id)
        arm_qpos_idx = mu.get_joint_qpos_indices(model, joint_ids)
        arm_dof_idx = mu.get_joint_dof_indices(model, joint_ids)

        site_name = f"gripper{ctrl.robot_id}_{ctrl.arm_id}_ft_frame"

        # Clip qpos to joint limits to avoid numerical errors
        # triggering mink's internal limit checks.
        q_full = np.array(data.qpos, copy=True, dtype=float).ravel()
        for j in range(model_native.njnt):
            if model_native.jnt_limited[j]:
                qadr = model_native.jnt_qposadr[j]
                if model_native.jnt_type[j] in [
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ]:
                    q_full[qadr] = np.clip(
                        q_full[qadr],
                        model_native.jnt_range[j, 0],
                        model_native.jnt_range[j, 1],
                    )
        config = mink.Configuration(model_native, q=q_full)

        # ── Tasks ─────────────────────────────────────────────────────
        frame_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=constants.MINK_POSITION_COST,
            orientation_cost=constants.MINK_ORIENTATION_COST,
            gain=constants.MINK_FRAME_GAIN,
            lm_damping=constants.MINK_LM_DAMPING,
        )

        posture_task = mink.PostureTask(
            model=model_native, cost=constants.MINK_POSTURE_COST,
        )
        if q_warm_start is not None:
            q_posture = q_full.copy()
            q_posture[arm_qpos_idx] = q_warm_start
            posture_task.set_target(q_posture)
        else:
            posture_task.set_target(q_full)

        damping_task = mink.DampingTask(
            model_native, cost=constants.MINK_DAMPING_COST,
        )

        frozen_dofs = mu.get_frozen_dof_indices(model_native, arm_dof_idx)
        freeze_task = (
            mink.DofFreezingTask(model_native, dof_indices=frozen_dofs)
            if frozen_dofs
            else None
        )

        tasks: list = [frame_task, posture_task, damping_task]
        if freeze_task is not None:
            tasks.append(freeze_task)

        # ── Limits ────────────────────────────────────────────────────
        vel_dict: dict = {}
        for jid in joint_ids:
            name = model_native.joint(jid).name
            try:
                k = int(name.rsplit("joint", 1)[1])
                vel_dict[name] = np.array([constants.FRANKA_VELOCITY_LIMITS[k - 1]])
            except (IndexError, ValueError):
                pass
        vel_limit = mink.VelocityLimit(model_native, velocities=vel_dict)
        config_limit = mink.ConfigurationLimit(model_native)
        limits = [vel_limit, config_limit]

        # ── Collision avoidance limit ─────────────────────────────────
        if collision_check:
            collision_limit = self._build_collision_limit(
                model_native, manipulated_object,
            )
            if collision_limit is not None:
                limits.append(collision_limit)

        return MinkContext(
            config=config,
            tasks=tasks,
            limits=limits,
            arm_qpos_idx=arm_qpos_idx,
            model_native=model_native,
            world_positions=world_positions,
            world_quaternions=world_quaternions,
        )

    def _build_collision_limit(
        self,
        model_native,
        manipulated_object: Optional[str],
    ) -> Optional[mink.CollisionAvoidanceLimit]:
        """Build a ``mink.CollisionAvoidanceLimit`` for the current robot."""
        ctrl = self._ctrl
        self_geoms = get_robot_collision_geoms(model_native, ctrl.robot_id)

        other_robot_ids = [
            i for i in range(len(ctrl.env.robots))
            if i != ctrl.robot_id
        ]
        other_geoms: list[str] = []
        for rid in other_robot_ids:
            other_geoms.extend(
                get_robot_collision_geoms(model_native, rid),
            )
        obj_geoms = get_object_collision_geoms(
            model_native, manipulated_object,
        )
        all_robot_ids = [ctrl.robot_id] + other_robot_ids
        env_geoms = get_environment_collision_geoms(
            model_native,
            robot_ids=all_robot_ids,
            object_name=manipulated_object,
        )

        geom_pairs: list[tuple] = []
        if self_geoms and other_geoms:
            geom_pairs.append((self_geoms, other_geoms))
        if self_geoms and obj_geoms:
            geom_pairs.append((self_geoms, obj_geoms))
        if self_geoms and env_geoms:
            geom_pairs.append((self_geoms, env_geoms))

        if not geom_pairs:
            logger.warning(
                "No collision geom pairs found — "
                "collision avoidance disabled for traj executor.",
            )
            return None

        logger.info(
            "mink CollisionAvoidanceLimit (traj executor): "
            "%d geom-pair groups (self=%d, other=%d, "
            "object=%d, env=%d geoms)",
            len(geom_pairs), len(self_geoms),
            len(other_geoms), len(obj_geoms),
            len(env_geoms),
        )
        return mink.CollisionAvoidanceLimit(
            model=model_native,
            geom_pairs=geom_pairs,
            gain=constants.MINK_COLLISION_GAIN,
            minimum_distance_from_collisions=constants.MINK_COLLISION_MIN_DIST,
            collision_detection_distance=constants.MINK_COLLISION_DETECT_DIST,
        )

    def _run_mink_loop(
        self,
        ctx: MinkContext,
        positions: np.ndarray,
        quaternions: np.ndarray,
        gripper_action: float,
        dt: float,
        threshold: float,
    ) -> bool:
        """Run the reactive mink tracking loop.

        Args:
            ctx: Pre-built mink context from :meth:`_build_mink_context`.
            positions: Original waypoint positions in **base** frame
                (used for convergence checking).
            quaternions: Original waypoint quaternions in **base** frame.
            gripper_action: Gripper command during execution.
            dt: Wall-clock time between steps.
            threshold: Convergence threshold on final-waypoint error.

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        ctrl = self._ctrl
        data = ctrl.env.sim.data
        num_waypoints = len(positions)
        max_steps = num_waypoints * 2
        current_wp_idx = 0
        steps_taken = 0
        start_time = time.time()

        while steps_taken < max_steps:
            if ctrl._stop_event.is_set():
                logger.info(
                    "mink trajectory cancelled for robot%d_%s",
                    ctrl.robot_id, ctrl.arm_id,
                )
                ctrl.last_status = 0
                return False

            if current_wp_idx < num_waypoints - 1:
                current_wp_idx += 1

            # Set frame task target
            wp_pos = ctx.world_positions[current_wp_idx]
            wp_quat = ctx.world_quaternions[current_wp_idx]
            target_so3 = MinkSO3.from_matrix(R.from_quat(wp_quat).as_matrix())
            target_se3 = MinkSE3.from_rotation_and_translation(target_so3, wp_pos)
            ctx.tasks[0].set_target(target_se3)  # tasks[0] is always frame_task

            # Sync config from simulation
            q_actual = np.array(data.qpos, copy=True, dtype=float).ravel()
            for j in range(ctx.model_native.njnt):
                if ctx.model_native.jnt_limited[j]:
                    qadr = ctx.model_native.jnt_qposadr[j]
                    if ctx.model_native.jnt_type[j] in [
                        mujoco.mjtJoint.mjJNT_HINGE,
                        mujoco.mjtJoint.mjJNT_SLIDE,
                    ]:
                        q_actual[qadr] = np.clip(
                            q_actual[qadr],
                            ctx.model_native.jnt_range[j, 0],
                            ctx.model_native.jnt_range[j, 1],
                        )
            ctx.config.update(q=q_actual)

            # Solve QP
            try:
                vel = mink.solve_ik(
                    ctx.config, ctx.tasks, constants.MINK_DT,
                    solver=constants.MINK_QP_SOLVER,
                    limits=ctx.limits,
                )
            except Exception as exc:
                logger.warning(
                    "mink.solve_ik failed at step %d, waypoint %d: %s",
                    steps_taken, current_wp_idx, str(exc),
                )
                vel = np.zeros(ctx.model_native.nv)

            q_next = ctx.config.integrate(vel, constants.MINK_DT)
            target_joints = q_next[ctx.arm_qpos_idx]

            robot_action = ctrl.construct_joint_action(
                target_joints, gripper_action,
            )
            with ctrl.env_lock if ctrl.env_lock else nullcontext():
                env_step(
                    ctrl.env, robot_action,
                    robot_id=ctrl.robot_id, arm=ctrl.arm_id,
                )

            steps_taken += 1

            elapsed = time.time() - start_time
            target_time = steps_taken * dt
            time.sleep(max(0, target_time - elapsed))

            # Convergence check at final waypoint
            if current_wp_idx == num_waypoints - 1:
                current_pos, current_quat = ctrl.get_eef_pose_in_base()
                pos_err = np.linalg.norm(positions[-1] - current_pos)
                ori_err = np.linalg.norm(
                    (R.from_quat(quaternions[-1]) * R.from_quat(current_quat).inv()).as_rotvec()
                )
                if pos_err < threshold and ori_err < threshold:
                    logger.debug("mink trajectory converged at step %d", steps_taken)
                    ctrl.last_status = 1
                    return True

            if steps_taken % 20 == 0:
                current_pos, current_quat = ctrl.get_eef_pose_in_base()
                pos_err = np.linalg.norm(
                    positions[current_wp_idx] - current_pos,
                )
                logger.debug(
                    "mink step %d/%d, wp %d/%d, pos_err=%.4f, arm_action_norm=%.4f, gripper_action=%d",
                    steps_taken, max_steps,
                    current_wp_idx, num_waypoints - 1, pos_err,
                    np.linalg.norm(robot_action[:-1]), robot_action[-1],
                )

        # Final report
        current_pos, current_quat = ctrl.get_eef_pose_in_base()
        pos_err = np.linalg.norm(positions[-1] - current_pos)
        ori_err = np.linalg.norm(
            (R.from_quat(quaternions[-1]) * R.from_quat(current_quat).inv()).as_rotvec()
        )
        if pos_err < threshold and ori_err < threshold:
            ctrl.last_status = 1
            logger.info(
                "ROBOT %d:%s - mink trajectory converged: pos_err=%.4f, ori_err=%.4f",
                ctrl.robot_id, ctrl.arm_id, pos_err, ori_err,
            )
            return True

        ctrl.last_status = 0
        logger.info(
            "ROBOT %d:%s - mink trajectory did NOT converge: pos_err=%.4f, ori_err=%.4f",
            ctrl.robot_id, ctrl.arm_id, pos_err, ori_err,
        )
        return False

    def execute_mink_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
        q_warm_start: Optional[np.ndarray] = None,
        collision_check: bool = False,
        manipulated_object: Optional[str] = None,
    ) -> bool:
        """Execute a trajectory using mink's reactive differential IK.

        Delegates to :meth:`_build_mink_context` for task/limit
        construction and :meth:`_run_mink_loop` for the reactive loop.

        Args:
            positions: Waypoint positions ``(N, 3)`` in **base** frame.
            quaternions: Waypoint quaternions ``(N, 4)`` xyzw in **base** frame.
            gripper_action: Gripper command during execution.
            dt: Wall-clock time between steps.
            threshold: Convergence threshold on final-waypoint error.
            q_warm_start: Optional warm-start joint config ``(7,)`` for the
                posture task.
            collision_check: If ``True``, add collision avoidance limits.
            manipulated_object: Name of the grasped object.

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        if len(positions) <= 1:
            return True

        ctx = self._build_mink_context(
            positions, quaternions,
            q_warm_start=q_warm_start,
            collision_check=collision_check,
            manipulated_object=manipulated_object,
        )
        return self._run_mink_loop(
            ctx, positions, quaternions,
            gripper_action=gripper_action,
            dt=dt,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # OSC-based trajectory execution
    # ------------------------------------------------------------------

    def execute_osc_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.TRAJECTORY_CONVERGENCE_THRESHOLD,
    ) -> bool:
        """Execute a trajectory using OSC (delta/absolute) commands.

        Args:
            positions: Waypoint positions ``(N, 3)`` in base frame.
            quaternions: Waypoint quaternions ``(N, 4)`` xyzw in base frame.
            gripper_action: Gripper command during execution.
            dt: Wall-clock time between steps.
            threshold: Convergence threshold.

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        ctrl = self._ctrl
        num_waypoints = len(positions)
        max_steps = num_waypoints * 2
        current_waypoint_idx = 0
        steps_taken = 0
        start_time = time.time()

        while steps_taken < max_steps:
            if ctrl._stop_event.is_set():
                logger.info(
                    "Movement cancelled for robot%d_%s",
                    ctrl.robot_id, ctrl.arm_id,
                )
                ctrl.last_status = 0
                return False

            if current_waypoint_idx < num_waypoints - 1:
                current_waypoint_idx += 1

            target_pos_step = positions[current_waypoint_idx]
            target_quat_step = quaternions[current_waypoint_idx]

            current_pos, current_quat = ctrl.get_eef_pose_in_base()

            pos_delta = target_pos_step - current_pos
            ori_error = R.from_quat(target_quat_step) * R.from_quat(current_quat).inv()
            ori_delta = ori_error.as_rotvec()

            robot_action = ctrl.construct_robot_action(
                target_pos_step, target_quat_step,
                gripper_action=gripper_action,
                current_pos=current_pos, current_quat=current_quat,
            )

            if robot_action is None:
                logger.debug(
                    "IK failed at waypoint %d, skipping", current_waypoint_idx,
                )
                steps_taken += 1
                continue

            with ctrl.env_lock if ctrl.env_lock else nullcontext():
                env_step(
                    ctrl.env, robot_action,
                    robot_id=ctrl.robot_id, arm=ctrl.arm_id,
                )

            steps_taken += 1

            elapsed = time.time() - start_time
            target_time = steps_taken * dt
            time.sleep(max(0, target_time - elapsed))

            if current_waypoint_idx == num_waypoints - 1:
                if np.linalg.norm(pos_delta) < threshold and np.linalg.norm(ori_delta) < threshold:
                    logger.debug("Converged at step %d", steps_taken)
                    break

            if steps_taken % 20 == 0:
                logger.debug(
                    "Step %d/%d, waypoint %d/%d, pos_err=%.4f, ori_err=%.4f",
                    steps_taken, max_steps,
                    current_waypoint_idx, num_waypoints - 1,
                    np.linalg.norm(pos_delta), np.linalg.norm(ori_delta),
                )

        # Final convergence check
        final_pos, final_quat = ctrl.get_eef_pose_in_base()
        target_pos = positions[-1]
        target_quat = quaternions[-1]
        pos_err = np.linalg.norm(target_pos - final_pos)
        ori_err = np.linalg.norm(
            (R.from_quat(target_quat) * R.from_quat(final_quat).inv()).as_rotvec()
        )

        if pos_err < threshold and ori_err < threshold:
            ctrl.last_status = 1
            logger.info(
                "ROBOT %d:%s - Trajectory converged: pos_err=%.4f, ori_err=%.4f",
                ctrl.robot_id, ctrl.arm_id, pos_err, ori_err,
            )
            return True

        ctrl.last_status = 0
        logger.info(
            "ROBOT %d:%s - Trajectory didn't converge: pos_err=%.4f, ori_err=%.4f",
            ctrl.robot_id, ctrl.arm_id, pos_err, ori_err,
        )
        return False

    # ------------------------------------------------------------------
    # Joint-space trajectory execution
    # ------------------------------------------------------------------

    def execute_joint_trajectory(
        self,
        joint_configs: np.ndarray,
        target_joints: np.ndarray,
        gripper_action: float = 0.0,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        threshold: float = constants.JOINT_CONVERGENCE_THRESHOLD,
        max_steps: Optional[int] = None,
    ) -> bool:
        """Execute a joint-space trajectory.

        Waypoint advancement is **time-based**: one waypoint per control step,
        matching the planned trajectory timing.  Robosuite's impedance (PD)
        controller naturally tracks the reference — if the robot lags, the
        position error grows and the PD drives harder (within actuator torque
        limits).  After all waypoints have been issued the final target is
        held for the remaining budget, giving the PD time to converge.

        Args:
            joint_configs: Waypoints ``(N, 7)``.
            target_joints: Final target joint configuration ``(7,)``.
            gripper_action: Gripper command during execution.
            dt: Wall-clock time between steps.
            threshold: Max joint-space L-inf error for convergence.
            max_steps: Maximum environment steps (default: ``2 * N``).

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        ctrl = self._ctrl
        total_waypoints = len(joint_configs)
        if max_steps is None:
            max_steps = total_waypoints * 2

        current_wp_idx = 0
        steps_taken = 0
        start_time = time.time()

        while steps_taken < max_steps:
            if ctrl._stop_event.is_set():
                logger.info(
                    "Joint trajectory cancelled for robot%d_%s",
                    ctrl.robot_id, ctrl.arm_id,
                )
                ctrl.last_status = 0
                return False

            # Time-based advancement: one waypoint per step, hold at final.
            current_wp_idx = min(steps_taken, total_waypoints - 1)

            target_wp = joint_configs[current_wp_idx]
            robot_action = ctrl.construct_joint_action(
                target_wp, gripper_action=gripper_action,
            )

            with ctrl.env_lock if ctrl.env_lock else nullcontext():
                env_step(
                    ctrl.env, robot_action,
                    robot_id=ctrl.robot_id, arm=ctrl.arm_id,
                )

            steps_taken += 1

            elapsed = time.time() - start_time
            target_time = steps_taken * dt
            time.sleep(max(0, target_time - elapsed))

            if steps_taken % 20 == 0:
                cur_joints = ctrl.get_joint_positions()
                cur_error = np.max(np.abs(cur_joints - target_wp))
                logger.debug(
                    "Step %d/%d, waypoint %d/%d, max joint err=%.4f rad",
                    steps_taken, max_steps,
                    current_wp_idx, total_waypoints - 1, cur_error,
                )

            if current_wp_idx == total_waypoints - 1:
                cur_joints = ctrl.get_joint_positions()
                joint_error = np.max(np.abs(cur_joints - target_joints))
                if joint_error < threshold:
                    ctrl.last_status = 1
                    logger.info(
                        "ROBOT %d:%s - Joint trajectory converged at step %d "
                        "(max err=%.4f rad)",
                        ctrl.robot_id, ctrl.arm_id, steps_taken, joint_error,
                    )
                    return True

        final_joints = ctrl.get_joint_positions()
        final_error = np.max(np.abs(final_joints - target_joints))
        ctrl.last_status = 0
        logger.info(
            "ROBOT %d:%s - Joint trajectory did NOT converge after %d steps "
            "(max err=%.4f rad)",
            ctrl.robot_id, ctrl.arm_id, steps_taken, final_error,
        )
        return False
