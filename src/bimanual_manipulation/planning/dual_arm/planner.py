"""Dual-Arm Parallel Planner — mink-based collision-aware concurrent execution.

Enables one arm (arm A) to follow a pre-planned manipulation trajectory
while the other arm (arm B) simultaneously moves to a target pose
(e.g. handover pre-pose) with **reactive collision avoidance**.

Architecture — powered by `mink <https://github.com/kevinzakka/mink>`_:

    At each simulation tick the planner:

    1.  Sets arm A's next waypoint from its pre-planned joint trajectory.
    2.  Builds a constrained QP (via ``mink.solve_ik``) that drives
        arm B's end-effector toward its target while satisfying:

        * **CollisionAvoidanceLimit** — inequality constraints that push
          arm B away from arm A's current geometry and the grasped object.
        * **DofFreezingTask** — equality constraint that pins arm A's DOFs
          so the QP only moves arm B.
        * **ConfigurationLimit / VelocityLimit** — physical joint limits.
        * **DampingTask** — minimum-norm regularisation for smooth motion.

    3.  Extracts arm B's joint velocity from the QP solution, integrates
        it, and sends both arms' commands via ``env_step``.

Usage example (see also ``dual_arm_handover_traj.py``)::

    planner = DualArmParallelPlanner(
        env        = system.env,
        env_lock   = system.env_lock,
        arm_a      = system.left_arm,
        arm_b      = system.right_arm,
        tf_manager = system.tf_manager,
    )
    result = planner.execute_parallel(
        traj_a    = joint_segment,        # pre-planned joints for arm A
        target_b  = handover_pre_pose,    # StampedPose for arm B
        gripper_a = 1.0,                  # keep gripper A closed
        gripper_b = -1.0,                 # open gripper B
    )
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import nullcontext
from typing import List, Optional, TYPE_CHECKING, Union

import mink
import numpy as np
import mujoco

from bimanual_manipulation import constants
from bimanual_manipulation.environments import DualArmEnvironment
from bimanual_manipulation.transforms.stamped_pose import StampedPose
from bimanual_manipulation.utils.env_utils import env_step
from bimanual_manipulation.utils.trajectory_math import (
    resample_joint_trajectory,
    plan_quintic_joint_trajectory,
)
from bimanual_manipulation.planning.trajectory.arm_planner import (
    plan_arm_trajectory,
)
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

from bimanual_manipulation.planning.dual_arm.result import (
    ParallelExecutionResult,
)
from bimanual_manipulation.planning.dual_arm.mink_setup import (
    MinkComponents,
    build_mink_components,
)
from bimanual_manipulation.planning.dual_arm.trajectory_helpers import (
    plan_pose_to_joint_trajectory,
    normalise_trajectory,
    eef_to_joint_trajectory,
)
from bimanual_manipulation.planning.ik.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction,
)

if TYPE_CHECKING:
    from bimanual_manipulation.controllers.arm import RobosuiteArmController

logger = logging.getLogger(__name__)


class DualArmParallelPlanner:
    """Collision-aware concurrent trajectory executor for two arms.

    Arm A follows a pre-planned joint trajectory (e.g. carrying an object
    to a handover point).  Arm B is driven to a target pose via mink's
    QP-based differential IK with inter-arm collision avoidance at every
    simulation tick.

    Parameters
    ----------
    env : DualArmEnvironment
        Shared robosuite simulation.
    arm_a, arm_b : RobosuiteArmController
        Controllers for the two arms.  Arm A has priority (pre-planned
        trajectory); arm B is reactively planned.
    env_lock : threading.RLock, optional
        Lock protecting ``env`` access concurrently.
    tf_manager : TransformManager, optional
        Shared TF manager.
    dt : float
        Time-step between waypoints (s).
    convergence_threshold : float
        Joint-error threshold for declaring convergence.
    collision_check : bool
        If ``True`` (default), add a ``CollisionAvoidanceLimit`` to the
        QP so that arm B actively avoids arm A and the grasped object.
    manipulated_object : str, optional
        Name of the grasped object (e.g. ``"mug_0"``).  When set, the
        collision avoidance limit includes object-vs-arm-B geom pairs
        (arm-A-vs-object contacts are automatically excluded because
        ``DofFreezingTask`` pins arm A and only arm B moves).
    """

    def __init__(
        self,
        env: DualArmEnvironment,
        arm_a: RobosuiteArmController,
        arm_b: RobosuiteArmController,
        env_lock=None,
        tf_manager=None,
        dt: float = constants.TRAJECTORY_TIME_STEP,
        convergence_threshold: float = constants.JOINT_CONVERGENCE_THRESHOLD,
        collision_check: bool = True,
        manipulated_object: Optional[str] = None,
        rrt_vis: Optional[RRTVisualizer] = None,
        planning_env: Optional[DualArmEnvironment] = None,
    ):
        self.env = env
        self.planning_env = planning_env or env
        self.env_lock = env_lock
        self.arm_a = arm_a
        self.arm_b = arm_b
        self.tf_manager = tf_manager or arm_a.tf_manager
        self.dt = dt
        self.convergence_threshold = convergence_threshold
        self.collision_check = collision_check
        self.manipulated_object = manipulated_object
        self.rrt_vis = rrt_vis

        # Lazy-initialised mink components (built on first execute)
        self._mink: Optional[MinkComponents] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_mink_init(self) -> None:
        """Build mink tasks, limits and constraints on first use.

        Deferred so that the planner can be constructed before the
        environment is fully stepped / stabilised.
        """
        if self._mink is not None:
            return
        self._mink = build_mink_components(
            self.env, self.arm_a, self.arm_b,
            collision_check=self.collision_check,
            manipulated_object=self.manipulated_object,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_parallel(
        self,
        traj_a: Union[List[np.ndarray], np.ndarray],
        target_b: Union[StampedPose, List[np.ndarray], np.ndarray],
        gripper_a: float = 0.0,
        gripper_b: float = 0.0,
        q_warm_start_b: Optional[np.ndarray] = None,
        num_steps_override: Optional[int] = None,
    ) -> ParallelExecutionResult:
        """Execute two arm motions simultaneously with collision avoidance.

        Arm A follows *traj_a* (pre-planned joint waypoints).  Arm B is
        driven reactively toward *target_b* using mink's QP-based
        differential IK with inter-arm collision constraints.

        Args:
            traj_a: Joint-space trajectory for arm A — (N, 7) array or
                list of (7,) arrays.
            target_b: Target for arm B.  Can be:

                * A ``StampedPose`` — IK + quintic planning will be done
                  as a *reference trajectory*; the mink QP then tracks
                  this reference while avoiding collisions.
                * A joint-space trajectory (list or array) — used as the
                  reference directly.

            gripper_a: Gripper action for arm A (1.0 = closed / grasping).
            gripper_b: Gripper action for arm B (-1.0 = open).
            q_warm_start_b: Optional IK warm-start for arm B.
            num_steps_override: Fixed step-count; both trajectories are
                resampled to this length.

        Returns:
            :class:`ParallelExecutionResult`.
        """
        t0 = time.time()
        self._ensure_mink_init()

        # ── Normalise arm A's trajectory ──────────────────────────────
        traj_a = normalise_trajectory(traj_a)
        n_a = len(traj_a)

        # ── Plan / normalise arm B's reference trajectory ─────────────
        with ThreadPoolExecutor(max_workers=1) as pool:
            future_b: Future = pool.submit(
                self._resolve_trajectory_b, target_b, n_a,
                q_warm_start_b, traj_a,
            )
            ref_traj_b = future_b.result()

        if ref_traj_b is None:
            logger.error(
                "Failed to plan reference trajectory for arm B "
                "(robot%d_%s). Falling back to arm-A-only execution.",
                self.arm_b.robot_id, self.arm_b.arm_id,
            )
            # Fall back: execute arm A's trajectory alone so it still
            # reaches the handover region, even though arm B couldn't
            # be moved simultaneously.
            logger.info(
                "Executing arm A trajectory alone (%d waypoints).",
                len(traj_a),
            )
            self.arm_a.follow_joint_trajectory(
                traj_a, smoothen=False,
                gripper_action=gripper_a,
            )
            err_a = self._joint_error(self.arm_a, traj_a[-1])
            result = ParallelExecutionResult(duration_s=time.time() - t0)
            result.success_a = err_a < self.convergence_threshold
            result.final_pos_error_a = err_a
            result.success_b = False
            result.final_pos_error_b = float("inf")
            result.collision_free = False
            return result

        # ── Synchronise lengths ───────────────────────────────────────
        target_len = num_steps_override or max(n_a, len(ref_traj_b))
        target_len = max(target_len, 2)

        if len(traj_a) != target_len:
            traj_a = resample_joint_trajectory(traj_a, target_len)
        if len(ref_traj_b) != target_len:
            ref_traj_b = resample_joint_trajectory(ref_traj_b, target_len)

        logger.info(
            "DualArmParallelPlanner: executing %d steps "
            "(arm_a=robot%d_%s, arm_b=robot%d_%s, collision_check=%s)",
            target_len,
            self.arm_a.robot_id, self.arm_a.arm_id,
            self.arm_b.robot_id, self.arm_b.arm_id,
            self.collision_check,
        )

        # ── Execute ───────────────────────────────────────────────────
        result = self._execute_mink_loop(
            traj_a, ref_traj_b, gripper_a, gripper_b, target_len,
        )
        result.duration_s = time.time() - t0
        logger.info(
            "DualArmParallelPlanner done in %.2fs — "
            "success_a=%s, success_b=%s "
            "(err_a=%.4f, err_b=%.4f, collision_free=%s)",
            result.duration_s, result.success_a, result.success_b,
            result.final_pos_error_a, result.final_pos_error_b,
            result.collision_free,
        )
        return result

    def execute_parallel_eef(
        self,
        traj_a: List[StampedPose],
        target_b: Union[StampedPose, List[np.ndarray], np.ndarray],
        gripper_a: float = 0.0,
        gripper_b: float = 0.0,
        q_warm_start_a: Optional[np.ndarray] = None,
        q_warm_start_b: Optional[np.ndarray] = None,
        num_steps_override: Optional[int] = None,
    ) -> ParallelExecutionResult:
        """Like :meth:`execute_parallel` but arm A receives an EEF trajectory.

        Each ``StampedPose`` is converted to joint space via sequential IK.
        """
        joint_traj_a = eef_to_joint_trajectory(
            self.arm_a, traj_a, q_warm_start=q_warm_start_a,
        )
        if joint_traj_a is None:
            logger.error("IK conversion failed for arm A's EEF trajectory.")
            return ParallelExecutionResult()

        return self.execute_parallel(
            traj_a=joint_traj_a,
            target_b=target_b,
            gripper_a=gripper_a,
            gripper_b=gripper_b,
            q_warm_start_b=q_warm_start_b,
            num_steps_override=num_steps_override,
        )

    # ------------------------------------------------------------------
    # Core execution loop — mink QP at every tick
    # ------------------------------------------------------------------

    def _execute_mink_loop(
        self,
        traj_a: np.ndarray,
        ref_traj_b: np.ndarray,
        gripper_a: float,
        gripper_b: float,
        num_steps: int,
    ) -> ParallelExecutionResult:
        """Synchronised execution with reactive collision avoidance.

        At each tick:
            1. Set arm A's next waypoint from *traj_a*.
            2. Update ``mink.Configuration`` with the full model state
               (arm A's planned joints + arm B's current joints).
            3. Set the ``FrameTask`` target from *ref_traj_b* (the
               reference B trajectory provides waypoint-by-waypoint
               targets — the QP then tracks them while satisfying
               collision constraints).
            4. ``solve_ik`` -> extract arm B's velocity -> integrate.
            5. Sends each arm's command via ``move_to_joint_config``.
        """
        ctx = self.env_lock if self.env_lock else nullcontext()
        max_steps = num_steps * 2
        wp_idx = 0
        steps_taken = 0
        start_time = time.time()
        qp_fail_count = 0

        result = ParallelExecutionResult()

        m = self._mink
        assert m is not None

        while steps_taken < max_steps:
            # ── Advance waypoint ──────────────────────────────────────
            if wp_idx < num_steps - 1:
                wp_idx += 1

            target_q_a = traj_a[wp_idx]
            ref_q_b = ref_traj_b[wp_idx]

            # ── Update mink Configuration with current sim state ──────
            q_full = self.env.sim.data.qpos.copy()
            q_full[m.qpos_idx_a] = target_q_a

            # Clip q_full to joint limits to avoid numerical errors
            # triggering mink's internal limit checks.
            nm = m.native_model
            for j in range(nm.njnt):
                if nm.jnt_limited[j]:
                    qadr = nm.jnt_qposadr[j]
                    if nm.jnt_type[j] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                        q_full[qadr] = np.clip(q_full[qadr], nm.jnt_range[j, 0], nm.jnt_range[j, 1])

            m.config.update(q=q_full)

            # ── Set FrameTask target for arm B ────────────────────────
            self._set_arm_b_target_from_joints(ref_q_b)

            # ── Set PostureTask target = reference config ─────────────
            if m.posture_task is not None:
                posture_q = q_full.copy()
                posture_q[m.qpos_idx_b] = ref_q_b
                m.posture_task.set_target(posture_q)

            # ── Solve the QP ──────────────────────────────────────────
            arm_b_joints = self._solve_mink_step()

            if arm_b_joints is None:
                # QP failed — hold arm B at its current sim position.
                # Using the reference waypoint (ref_q_b) would cause a
                # large jump if arm B is far behind the reference, leading
                # to wild oscillations.  Holding position is safe and lets
                # the QP resume once arm A moves away.
                arm_b_joints = q_full[m.qpos_idx_b].copy()
                qp_fail_count += 1
                logger.debug(
                    "mink QP failed at step %d — holding arm B position "
                    "(ref drift=%.3f rad).",
                    steps_taken,
                    float(np.max(np.abs(arm_b_joints - ref_q_b))),
                )

            # ── Move both arms via unified action (Single env step) ───
            action_a = self.arm_a.construct_joint_action(
                target_q_a, gripper_action=gripper_a,
            )
            action_b = self.arm_b.construct_joint_action(
                arm_b_joints, gripper_action=gripper_b,
            )

            # Construct full environment action vector
            full_action = np.zeros(self.env.action_dim)

            # Offset arm_a robot action
            offset_a = 0
            for i in range(self.arm_a.robot_id):
                offset_a += self.env.robots[i].action_dim
            full_action[offset_a : offset_a + len(action_a)] = action_a

            # Offset arm_b robot action
            offset_b = 0
            for i in range(self.arm_b.robot_id):
                offset_b += self.env.robots[i].action_dim
            full_action[offset_b : offset_b + len(action_b)] = action_b

            with ctx:
                env_step(self.env, full_action)

            steps_taken += 1

            # ── Drift-correcting sleep ────────────────────────────────
            elapsed = time.time() - start_time
            target_time = steps_taken * self.dt
            time.sleep(max(0, target_time - elapsed))

            # ── Convergence check ─────────────────────────────────────
            if wp_idx == num_steps - 1:
                err_a = self._joint_error(self.arm_a, traj_a[-1])
                err_b = self._joint_error(self.arm_b, ref_traj_b[-1])
                if (err_a < self.convergence_threshold
                        and err_b < self.convergence_threshold):
                    result.success_a = True
                    result.success_b = True
                    result.final_pos_error_a = err_a
                    result.final_pos_error_b = err_b
                    result.steps_taken = steps_taken
                    result.collision_free = qp_fail_count == 0
                    return result

            # ── Progress logging ──────────────────────────────────────
            if steps_taken % 50 == 0:
                logger.debug(
                    "Step %d/%d wp=%d err_a=%.4f err_b=%.4f qp_fails=%d",
                    steps_taken, max_steps, wp_idx,
                    self._joint_error(self.arm_a, traj_a[wp_idx]),
                    self._joint_error(self.arm_b, ref_traj_b[wp_idx]),
                    qp_fail_count,
                )

        # ── Timeout ───────────────────────────────────────────────────
        err_a = self._joint_error(self.arm_a, traj_a[-1])
        err_b = self._joint_error(self.arm_b, ref_traj_b[-1])
        result.success_a = err_a < self.convergence_threshold
        result.success_b = err_b < self.convergence_threshold
        result.final_pos_error_a = err_a
        result.final_pos_error_b = err_b
        result.steps_taken = steps_taken
        result.collision_free = qp_fail_count == 0
        logger.warning(
            "Timed out after %d steps "
            "(err_a=%.4f, err_b=%.4f, qp_fails=%d)",
            steps_taken, err_a, err_b, qp_fail_count,
        )
        return result

    # ------------------------------------------------------------------
    # Mink QP helpers
    # ------------------------------------------------------------------

    def _set_arm_b_target_from_joints(
        self, ref_q_b: np.ndarray,
    ) -> None:
        """Set the FrameTask target by FK-ing the reference joints."""
        m = self._mink
        assert m is not None

        q_backup = m.config.q.copy()
        q_ref = q_backup.copy()
        q_ref[m.qpos_idx_b] = ref_q_b
        m.config.update(q=q_ref)

        eef_body = self.arm_b._eef_body_name()
        target_se3 = m.config.get_transform_frame_to_world(
            eef_body, "body",
        )
        m.eef_task.set_target(target_se3)

        m.config.update(q=q_backup)

    def _solve_mink_step(self) -> Optional[np.ndarray]:
        """Run one mink solve_ik step and return arm B's target joints.

        Returns:
            (7,) joint configuration for arm B, or ``None`` if the
            QP solve fails.
        """
        m = self._mink
        assert m is not None

        tasks = [m.eef_task, m.damping_task]
        if m.posture_task is not None:
            tasks.append(m.posture_task)

        limits_full: list = []
        if m.vel_limit is not None:
            limits_full.append(m.vel_limit)
        if m.config_limit is not None:
            limits_full.append(m.config_limit)
        if m.collision_limit is not None:
            limits_full.append(m.collision_limit)

        constraints = []
        if m.freeze_task is not None:
            constraints.append(m.freeze_task)

        try:
            vel = mink.solve_ik(
                configuration=m.config,
                tasks=tasks,
                dt=constants.MINK_DT,
                solver=constants.MINK_QP_SOLVER,
                limits=limits_full if limits_full else None,
                constraints=constraints if constraints else None,
            )
            m.config.integrate_inplace(vel, constants.MINK_DT)
            return m.config.q[m.qpos_idx_b].copy()
        except Exception as exc_primary:
            logger.warning("Primary mink solve_ik failed: %s", exc_primary)

    # ------------------------------------------------------------------
    # Trajectory resolution
    # ------------------------------------------------------------------

    def _resolve_trajectory_b(
        self,
        target_b: Union[StampedPose, List[np.ndarray], np.ndarray],
        desired_length: int,
        q_warm_start: Optional[np.ndarray],
        traj_a: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Convert *target_b* into a (N, 7) reference joint trajectory."""
        if isinstance(target_b, StampedPose):
            return self._plan_collision_free_trajectory_b(
                target_b, desired_length, q_warm_start, traj_a,
            )
        traj = normalise_trajectory(target_b)
        # Single joint config → plan a collision-free path to it
        if len(traj) == 1:
            return self._plan_collision_free_trajectory_b_joints(
                traj[0], desired_length, traj_a,
            )
        return traj

    def _sync_planning_env(self) -> None:
        """Synchronize planning environment with execution environment."""
        if self.env is self.planning_env:
            return
        with self.env_lock if self.env_lock else nullcontext():
            self.planning_env.sim.data.qpos[:] = self.env.sim.data.qpos
            self.planning_env.sim.data.qvel[:] = self.env.sim.data.qvel
            self.planning_env.sim.forward()

    def _plan_collision_free_trajectory_b(
        self,
        target_pose: StampedPose,
        desired_length: int,
        q_warm_start: Optional[np.ndarray],
        traj_a: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Plan arm B's reference trajectory using RRT when possible.

        Tries multiple IK solutions for the goal configuration to
        increase the chance of finding one that is collision-free
        against arm A's trajectory.

        Returns ``None`` if RRT fails or IK cannot be solved.
        """
        self._sync_planning_env()

        # --- Resolve IK goal candidates ---
        # Get an initial solution via the normal pipeline (single best).
        quintic_traj, goal_q = plan_pose_to_joint_trajectory(
            self.arm_b, target_pose, desired_length,
            q_warm_start=q_warm_start,
        )
        if quintic_traj is None or goal_q is None:
            return None

        if traj_a is None or len(traj_a) == 0:
            return quintic_traj

        # --- Enumerate multiple IK solutions for robustness ---
        # The primary IK solution may be in collision with arm A at
        # some point during its trajectory.  Trying alternatives
        # greatly increases the chance of RRT success.
        goal_candidates = [goal_q]
        try:
            pose_for_ik = target_pose
            if pose_for_ik.frame_id != self.arm_b.base_frame:
                pose_for_ik = self.arm_b.tf_manager.transform_pose(
                    pose_for_ik, self.arm_b.base_frame,
                )
            corrected_pose = apply_ik_orientation_correction(pose_for_ik)
            site_name = f"gripper{self.arm_b.robot_id}_{self.arm_b.arm_id}_ft_frame"
            extra_solutions = enumerate_ik_solutions_for_arm(
                self.env,
                robot_id=self.arm_b.robot_id,
                target_pose=corrected_pose,
                tf_manager=self.arm_b.tf_manager,
                arm=self.arm_b.arm_id,
                site_name=site_name,
                env_lock=self.env_lock,
                max_solutions=constants.IK_MAX_SOLUTIONS,
                q_reference=q_warm_start,
            )
            for sol in extra_solutions:
                if not any(np.allclose(sol, g, atol=0.05) for g in goal_candidates):
                    goal_candidates.append(sol)
        except Exception as exc:
            logger.debug("Extra IK enumeration failed: %s", exc)

        if q_warm_start is not None and len(goal_candidates) > 1:
            goal_candidates.sort(
                key=lambda q: float(np.linalg.norm(q - q_warm_start)),
            )

        logger.info(
            "Arm B IK: %d goal candidate(s) for RRT.", len(goal_candidates),
        )

        traj_a_temporal: Optional[np.ndarray] = traj_a
        logger.info(
            "Temporal collision: using all %d arm A configs.",
            len(traj_a),
        )

        q_arm_a_ref = traj_a[-1]
        start_q = self.arm_b.get_joint_positions()

        # --- Try each IK goal candidate via RRT ---
        for idx, candidate_q in enumerate(goal_candidates):
            try:
                rrt_traj, rrt_success = plan_arm_trajectory(
                    env=self.planning_env,
                    robot_id_b=self.arm_b.robot_id,
                    start_q=start_q,
                    goal_q=candidate_q,
                    q_other_arm=q_arm_a_ref,
                    robot_id_a=self.arm_a.robot_id,
                    num_steps=desired_length,
                    env_lock=self.env_lock,
                    rrt_vis=self.rrt_vis,
                    q_other_arm_trajectory=traj_a_temporal,
                )
                if rrt_success:
                    logger.info(
                        "Arm B reference trajectory planned via "
                        "collision-free joint RRT (IK candidate %d/%d).",
                        idx + 1, len(goal_candidates),
                    )
                    return rrt_traj
            except Exception as exc:
                logger.error(
                    "Joint RRT planning failed for arm B "
                    "(candidate %d/%d): %s", idx + 1, len(goal_candidates), exc,
                )

        logger.warning(
            "No collision-free trajectory found for arm B — "
            "tried %d IK goal candidate(s). Returning None.",
            len(goal_candidates),
        )
        return None

    def _plan_collision_free_trajectory_b_joints(
        self,
        goal_q: np.ndarray,
        desired_length: int,
        traj_a: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Plan arm B's reference trajectory to a joint-space goal via RRT.

        Like :meth:`_plan_collision_free_trajectory_b` but takes a direct
        joint configuration (no IK needed).  Falls back to a quintic
        interpolation if RRT fails or arm A's trajectory is unavailable.
        """
        self._sync_planning_env()

        start_q = self.arm_b.get_joint_positions()

        if traj_a is None or len(traj_a) == 0:
            return plan_quintic_joint_trajectory(
                start_q, goal_q, desired_length,
            )

        q_arm_a_ref = traj_a[-1]

        try:
            rrt_traj, rrt_success = plan_arm_trajectory(
                env=self.planning_env,
                robot_id_b=self.arm_b.robot_id,
                start_q=start_q,
                goal_q=goal_q,
                q_other_arm=q_arm_a_ref,
                robot_id_a=self.arm_a.robot_id,
                num_steps=desired_length,
                env_lock=self.env_lock,
                rrt_vis=self.rrt_vis,
                q_other_arm_trajectory=traj_a,
            )
            if rrt_success:
                logger.info(
                    "Arm B joint-goal reference trajectory planned "
                    "via collision-free RRT.",
                )
                return rrt_traj
        except Exception as exc:
            logger.error(
                "Joint RRT planning failed for arm B joint goal: %s", exc,
            )

        logger.warning(
            "RRT failed for arm B joint goal — "
            "falling back to quintic trajectory.",
        )
        return plan_quintic_joint_trajectory(
            start_q, goal_q, desired_length,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_error(
        arm: RobosuiteArmController, target_q: np.ndarray,
    ) -> float:
        """Max absolute joint error."""
        return float(np.max(np.abs(arm.get_joint_positions() - target_q)))
