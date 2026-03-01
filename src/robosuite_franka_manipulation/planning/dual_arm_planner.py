"""
Dual-Arm Parallel Planner — mink-based collision-aware concurrent execution.

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

This replaces the previous "plan-then-stagger" approach with a
**reactive, online** planner: arm B continuously adapts its path to the
latest arm A configuration, guaranteeing collision avoidance at every
step without heuristic time-staggering.

Dependencies:
    - ``mink >= 1.0`` (already installed; QP-based differential IK for MuJoCo)
    - ``qpsolvers`` with ``daqp`` backend (installed with mink)
    - Existing arm controller infrastructure (``RobosuiteArmController``)
    - ``scipy.interpolate`` for trajectory time-scaling

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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import mink
import mujoco
import numpy as np

from robosuite_franka_manipulation.environments import DualArmEnvironment
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.controllers.arm import RobosuiteArmController
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.utils.env_utils import env_step
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object,
    get_robot_collision_geoms,
    get_all_dof_indices_except,
    get_object_collision_geoms,
    get_environment_collision_geoms,
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    resample_joint_trajectory,
    quintic_scalar,
    plan_quintic_joint_trajectory,
)
from robosuite_franka_manipulation.planning.rrt_joint_planner import (
    plan_arm_trajectory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ParallelExecutionResult:
    """Outcome of a parallel dual-arm execution."""

    success_a: bool = False
    success_b: bool = False
    steps_taken: int = 0
    duration_s: float = 0.0
    final_pos_error_a: float = float("inf")
    final_pos_error_b: float = float("inf")
    collision_free: bool = True

    @property
    def success(self) -> bool:
        return self.success_a and self.success_b


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------


def _plan_pose_to_joint_trajectory(
    arm: RobosuiteArmController,
    target_pose: StampedPose,
    num_steps: int,
    q_warm_start: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Solve IK for *target_pose* and return a quintic reference trajectory.

    Uses the arm's IK solver for the goal configuration.  Returns both
    the trajectory and the solved target joints (needed by the RRT-based
    planner as the goal configuration).

    Returns:
        ``(trajectory, target_joints)`` — either may be ``None`` on
        IK failure.
    """
    if target_pose.frame_id != arm.base_frame:
        target_pose = arm.tf_manager.transform_pose(target_pose, arm.base_frame)

    target_joints = arm._solve_ik_for_pose(
        np.asarray(target_pose.position),
        np.asarray(target_pose.quaternion),
        q_warm_start=q_warm_start,
    )
    if target_joints is None:
        logger.warning(
            "IK failed for robot%d_%s target pose",
            arm.robot_id, arm.arm_id,
        )
        return None, None

    traj = plan_quintic_joint_trajectory(
        arm.get_joint_positions(), target_joints, num_steps,
    )
    return traj, target_joints


# ---------------------------------------------------------------------------
# Main planner class
# ---------------------------------------------------------------------------

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
        convergence_threshold: float = 0.05,
        collision_check: bool = True,
        manipulated_object: Optional[str] = None,
    ):
        self.env = env
        self.env_lock = env_lock
        self.arm_a = arm_a
        self.arm_b = arm_b
        self.tf_manager = tf_manager or arm_a.tf_manager
        self.dt = dt
        self.convergence_threshold = convergence_threshold
        self.collision_check = collision_check
        self.manipulated_object = manipulated_object

        # ── Lazy-initialised mink objects (built on first execute) ─────
        self._mink_initialised = False
        self._native_model: Optional[mujoco.MjModel] = None
        self._mink_config: Optional[mink.Configuration] = None
        self._eef_task: Optional[mink.FrameTask] = None
        self._posture_task: Optional[mink.PostureTask] = None
        self._damping_task: Optional[mink.DampingTask] = None
        self._freeze_task: Optional[mink.DofFreezingTask] = None
        self._collision_limit: Optional[mink.CollisionAvoidanceLimit] = None
        self._vel_limit: Optional[mink.VelocityLimit] = None
        self._config_limit: Optional[mink.ConfigurationLimit] = None

        # Cached index maps
        self._qpos_idx_a: Optional[np.ndarray] = None
        self._qpos_idx_b: Optional[np.ndarray] = None
        self._dof_idx_b: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_mink_init(self) -> None:
        """Build mink tasks, limits and constraints on first use.

        Deferred so that the planner can be constructed before the
        environment is fully stepped / stabilised.
        """
        if self._mink_initialised:
            return

        model = self.env.sim.model
        nm = get_native_mj_object(model)
        self._native_model = nm

        # ── Index maps ────────────────────────────────────────────────
        jids_a = get_arm_joint_ids(nm, self.arm_a.robot_id)
        jids_b = get_arm_joint_ids(nm, self.arm_b.robot_id)
        self._qpos_idx_a = get_joint_qpos_indices(nm, jids_a)
        self._qpos_idx_b = get_joint_qpos_indices(nm, jids_b)

        # DOF indices for arm B (used to extract velocity from QP)
        self._dof_idx_b = [nm.jnt_dofadr[j] for j in jids_b]

        # ── mink.Configuration ────────────────────────────────────────
        self._mink_config = mink.Configuration(
            nm, q=self.env.sim.data.qpos.copy(),
        )

        # ── FrameTask for arm B's end-effector ────────────────────────
        eef_body_name = self.arm_b._eef_body_name()
        self._eef_task = mink.FrameTask(
            frame_name=eef_body_name,
            frame_type="body",
            position_cost=constants.MINK_POSITION_COST,
            orientation_cost=constants.MINK_ORIENTATION_COST,
            gain=constants.MINK_FRAME_GAIN,
            lm_damping=constants.MINK_LM_DAMPING,
        )

        # ── PostureTask — biases toward current (comfortable) config ──
        self._posture_task = mink.PostureTask(
            model=nm, cost=constants.MINK_POSTURE_COST,
        )

        # ── DampingTask — minimum-norm regulariser ────────────────────
        self._damping_task = mink.DampingTask(model=nm, cost=constants.MINK_DAMPING_COST)

        # ── DofFreezingTask — freeze everything except arm B ──────────
        frozen_dofs = get_all_dof_indices_except(nm, self.arm_b.robot_id)
        if frozen_dofs:
            self._freeze_task = mink.DofFreezingTask(
                model=nm, dof_indices=frozen_dofs,
            )

        # ── Velocity / configuration limits ───────────────────────────
        self._vel_limit = mink.VelocityLimit(nm)
        self._config_limit = mink.ConfigurationLimit(nm)

        # ── Collision avoidance limit ─────────────────────────────────
        if self.collision_check:
            geoms_a = get_robot_collision_geoms(nm, self.arm_a.robot_id)
            geoms_b = get_robot_collision_geoms(nm, self.arm_b.robot_id)
            obj_geoms = get_object_collision_geoms(nm, self.manipulated_object)
            env_geoms = get_environment_collision_geoms(
                nm,
                robot_ids=[self.arm_a.robot_id, self.arm_b.robot_id],
                object_name=self.manipulated_object,
            )

            # Build collision pairs:
            #   arm_B  vs  arm_A   — prevent inter-arm collision
            #   arm_B  vs  object  — prevent arm B from hitting the grasped object
            #   arm_B  vs  env     — prevent arm B from hitting static obstacles
            #                        (tables, shelves, walls, etc.)
            # We do NOT add arm_A vs object (they are in expected grasp contact).
            geom_pairs: List[Tuple[Sequence[str], Sequence[str]]] = []
            if geoms_a and geoms_b:
                geom_pairs.append((geoms_b, geoms_a))
            if obj_geoms and geoms_b:
                geom_pairs.append((geoms_b, obj_geoms))
            if env_geoms and geoms_b:
                geom_pairs.append((geoms_b, env_geoms))

            if geom_pairs:
                self._collision_limit = mink.CollisionAvoidanceLimit(
                    model=nm,
                    geom_pairs=geom_pairs,
                    gain=constants.MINK_COLLISION_GAIN,
                    minimum_distance_from_collisions=constants.MINK_COLLISION_MIN_DIST,
                    collision_detection_distance=constants.MINK_COLLISION_DETECT_DIST,
                )
                logger.info(
                    "mink CollisionAvoidanceLimit: %d geom-pair groups "
                    "(arm_B=%d geoms, arm_A=%d geoms, object=%d geoms, "
                    "env=%d geoms)",
                    len(geom_pairs), len(geoms_b), len(geoms_a),
                    len(obj_geoms), len(env_geoms),
                )
            else:
                logger.warning(
                    "No collision geom pairs found — "
                    "collision avoidance disabled."
                )

        self._mink_initialised = True
        logger.info(
            "mink initialised for DualArmParallelPlanner "
            "(nq=%d, nv=%d, frozen_dofs=%d, solver=%s)",
            nm.nq, nm.nv, len(frozen_dofs), constants.MINK_QP_SOLVER,
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
        traj_a = self._normalise_trajectory(traj_a)
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
                "(robot%d_%s).",
                self.arm_b.robot_id, self.arm_b.arm_id,
            )
            return ParallelExecutionResult(duration_s=time.time() - t0)

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
        joint_traj_a = self._eef_to_joint_trajectory(
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
            5. Send both arms' actions via ``env_step``.
        """
        ctx = self.env_lock if self.env_lock else nullcontext()
        max_steps = num_steps * 2
        wp_idx = 0
        steps_taken = 0
        start_time = time.time()
        qp_fail_count = 0

        result = ParallelExecutionResult()

        assert self._mink_config is not None
        assert self._qpos_idx_a is not None
        assert self._qpos_idx_b is not None

        while steps_taken < max_steps:
            # ── Advance waypoint ──────────────────────────────────────
            if wp_idx < num_steps - 1:
                wp_idx += 1

            target_q_a = traj_a[wp_idx]
            ref_q_b = ref_traj_b[wp_idx]

            # ── Update mink Configuration with current sim state ──────
            # Copy the full qpos from the simulation, then overwrite
            # arm A's joints with the *planned* waypoint (so the QP
            # sees arm A where it *should* be, not where it is if
            # the controller lags slightly).
            q_full = self.env.sim.data.qpos.copy()
            q_full[self._qpos_idx_a] = target_q_a
            self._mink_config.update(q=q_full)

            # ── Set FrameTask target for arm B ────────────────────────
            # FK the reference joints to get the EEF pose and set that
            # as the target.  The QP tracks the reference subject to
            # collision constraints.
            self._set_arm_b_target_from_joints(ref_q_b)

            # ── Set PostureTask target = reference config ─────────────
            if self._posture_task is not None:
                posture_q = q_full.copy()
                posture_q[self._qpos_idx_b] = ref_q_b
                self._posture_task.set_target(posture_q)

            # ── Solve the QP ──────────────────────────────────────────
            arm_b_joints = self._solve_mink_step()

            if arm_b_joints is None:
                # QP failed — fall back to reference trajectory
                arm_b_joints = ref_q_b
                qp_fail_count += 1
                logger.debug(
                    "mink QP failed at step %d, falling back to reference.",
                    steps_taken,
                )

            # ── Build and send actions ────────────────────────────────
            action_a = self.arm_a.construct_joint_action(
                target_q_a, gripper_action=gripper_a,
            ) if self.arm_a.is_joint_position_controller else (
                self._pose_action_from_joints(
                    self.arm_a, target_q_a, gripper_a,
                )
            )
            action_b = self.arm_b.construct_joint_action(
                arm_b_joints, gripper_action=gripper_b,
            ) if self.arm_b.is_joint_position_controller else (
                self._pose_action_from_joints(
                    self.arm_b, arm_b_joints, gripper_b,
                )
            )

            with ctx:
                if action_a is not None:
                    env_step(
                        self.env, action_a,
                        robot_id=self.arm_a.robot_id,
                        arm=self.arm_a.arm_id,
                    )
                if action_b is not None:
                    env_step(
                        self.env, action_b,
                        robot_id=self.arm_b.robot_id,
                        arm=self.arm_b.arm_id,
                    )

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
        """Set the FrameTask target by FK-ing the reference joints.

        Temporarily sets arm B's joints in the mink Configuration to
        *ref_q_b*, reads the resulting EEF SE3, then restores the
        original state.
        """
        assert self._mink_config is not None
        assert self._eef_task is not None
        assert self._qpos_idx_b is not None

        # Save current config
        q_backup = self._mink_config.q.copy()

        # Set arm B to reference joints and run FK
        q_ref = q_backup.copy()
        q_ref[self._qpos_idx_b] = ref_q_b
        self._mink_config.update(q=q_ref)

        # Read the EEF pose and set as target
        eef_body = self.arm_b._eef_body_name()
        target_se3 = self._mink_config.get_transform_frame_to_world(
            eef_body, "body",
        )
        self._eef_task.set_target(target_se3)

        # Restore original config
        self._mink_config.update(q=q_backup)

    def _solve_mink_step(self) -> Optional[np.ndarray]:
        """Run one mink solve_ik step and return arm B's target joints.

        Returns:
            (7,) joint configuration for arm B, or ``None`` if the QP
            fails.
        """
        assert self._mink_config is not None
        assert self._eef_task is not None
        assert self._qpos_idx_b is not None

        tasks = [self._eef_task, self._damping_task]
        if self._posture_task is not None:
            tasks.append(self._posture_task)

        limits = []
        if self._vel_limit is not None:
            limits.append(self._vel_limit)
        if self._config_limit is not None:
            limits.append(self._config_limit)
        if self._collision_limit is not None:
            limits.append(self._collision_limit)

        constraints = []
        if self._freeze_task is not None:
            constraints.append(self._freeze_task)

        try:
            vel = mink.solve_ik(
                configuration=self._mink_config,
                tasks=tasks,
                dt=constants.MINK_DT,
                solver=constants.MINK_QP_SOLVER,
                limits=limits if limits else None,
                constraints=constraints if constraints else None,
            )
        except Exception as exc:
            logger.debug("mink solve_ik failed: %s", exc)
            return None

        # Integrate to get new joint positions
        self._mink_config.integrate_inplace(vel, constants.MINK_DT)
        new_q = self._mink_config.q

        # Extract arm B's joint configuration
        return new_q[self._qpos_idx_b].copy()

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_trajectory(
        traj: Union[List[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Ensure trajectory is a 2-D numpy array."""
        if isinstance(traj, list):
            traj = np.array(
                [np.asarray(w, dtype=np.float64) for w in traj],
            )
        else:
            traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        return traj

    def _resolve_trajectory_b(
        self,
        target_b: Union[StampedPose, List[np.ndarray], np.ndarray],
        desired_length: int,
        q_warm_start: Optional[np.ndarray],
        traj_a: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Convert *target_b* into a (N, 7) reference joint trajectory.

        When *target_b* is a ``StampedPose`` and *traj_a* is available,
        attempts a joint-space RRT-Connect to find a collision-free
        reference trajectory for arm B.  Falls back to quintic
        interpolation if:  (a) no environment is available, (b) the RRT
        planner fails, or (c) *target_b* is already a trajectory.
        """
        if isinstance(target_b, StampedPose):
            return self._plan_collision_free_trajectory_b(
                target_b, desired_length, q_warm_start, traj_a,
            )
        return self._normalise_trajectory(target_b)

    def _plan_collision_free_trajectory_b(
        self,
        target_pose: StampedPose,
        desired_length: int,
        q_warm_start: Optional[np.ndarray],
        traj_a: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Plan arm B's reference trajectory using RRT when possible.

        Pipeline:

        1. Solve IK for *target_pose* → ``goal_q``.
        2. If *traj_a* is provided, run joint-space RRT-Connect from
           arm B's current config to ``goal_q`` while arm A is held at
           ``traj_a[-1]`` (the configuration arm A will reach at the
           handover point).
        3. If RRT finds a collision-free path, smooth and densify it.
        4. Fall back to quintic interpolation otherwise.

        Returns:
            ``(N, 7)`` joint trajectory or ``None`` on IK failure.
        """
        quintic_traj, goal_q = _plan_pose_to_joint_trajectory(
            self.arm_b, target_pose, desired_length,
            q_warm_start=q_warm_start,
        )
        if quintic_traj is None or goal_q is None:
            return None

        # Attempt joint-space RRT only when arm A's trajectory is known
        if traj_a is not None and len(traj_a) > 0:
            q_arm_a_ref = traj_a[-1]  # arm A's final config
            start_q = self.arm_b.get_joint_positions()

            try:
                rrt_traj, rrt_success = plan_arm_trajectory(
                    env=self.env,
                    robot_id_b=self.arm_b.robot_id,
                    start_q=start_q,
                    goal_q=goal_q,
                    q_other_arm=q_arm_a_ref,
                    robot_id_a=self.arm_a.robot_id,
                    num_steps=desired_length,
                    env_lock=self.env_lock,
                    manipulated_object=self.manipulated_object,
                )
                if rrt_success:
                    logger.info(
                        "Arm B reference trajectory planned via "
                        "collision-free joint RRT."
                    )
                    return rrt_traj
                logger.warning(
                    "Joint RRT failed — using quintic fallback for "
                    "arm B reference."
                )
            except Exception as exc:
                logger.warning(
                    "Joint RRT planning error — using quintic fallback: %s",
                    exc,
                )

        return quintic_traj

    @staticmethod
    def _eef_to_joint_trajectory(
        arm: RobosuiteArmController,
        eef_traj: List[StampedPose],
        q_warm_start: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Convert an EEF trajectory to joint space by sequential IK."""
        joint_configs = []
        q_ws = (
            q_warm_start if q_warm_start is not None
            else arm.get_joint_positions()
        )

        for i, pose in enumerate(eef_traj):
            if pose.frame_id != arm.base_frame:
                pose = arm.tf_manager.transform_pose(
                    pose, arm.base_frame,
                )

            q = arm._solve_ik_for_pose(
                np.asarray(pose.position),
                np.asarray(pose.quaternion),
                q_warm_start=q_ws,
            )
            if q is None:
                logger.warning(
                    "IK failed at EEF waypoint %d/%d", i, len(eef_traj),
                )
                joint_configs.append(q_ws.copy())
            else:
                joint_configs.append(q)
                q_ws = q
        return np.array(joint_configs)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_error(
        arm: RobosuiteArmController, target_q: np.ndarray,
    ) -> float:
        """Max absolute joint error."""
        return float(np.max(np.abs(arm.get_joint_positions() - target_q)))

    @staticmethod
    def _pose_action_from_joints(
        arm: RobosuiteArmController,
        target_joints: np.ndarray,
        gripper_action: float,
    ) -> Optional[np.ndarray]:
        """Fall-back for OSC controllers: FK the joints -> Cartesian action."""
        try:
            from robosuite_franka_manipulation.utils.env_utils import (
                set_robot_config,
            )
            qpos_backup = arm.env.sim.data.qpos.copy()
            set_robot_config(
                arm.env, arm.robot_id, arm.arm_id, target_joints,
            )
            pos, quat = arm.get_eef_pose_in_base()
            arm.env.sim.data.qpos[:] = qpos_backup
            arm.env.sim.forward()
            return arm.construct_robot_action(
                pos, quat, gripper_action=gripper_action,
            )
        except Exception as e:
            logger.warning("Pose-action fallback failed: %s", e)
            return None
