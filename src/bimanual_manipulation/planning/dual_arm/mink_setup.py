"""Mink task/limit/constraint builder for dual-arm collision avoidance.

Encapsulates the construction of all mink objects (``Configuration``,
``FrameTask``, ``PostureTask``, ``DampingTask``, ``DofFreezingTask``,
limits, and ``CollisionAvoidanceLimit``) into a single
:func:`build_mink_components` call that returns a :class:`MinkComponents`
bundle.

This keeps the mink setup logic in one place (SRP) and makes it testable
independently of the planner orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import mink
import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments import DualArmEnvironment
from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object,
    get_robot_collision_geoms,
    get_all_dof_indices_except,
    get_object_collision_geoms,
    get_environment_collision_geoms,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bimanual_manipulation.controllers.arm import RobosuiteArmController


@dataclass
class MinkComponents:
    """Bundle of all mink objects needed by the dual-arm planner."""

    native_model: mujoco.MjModel
    config: mink.Configuration
    eef_task: mink.FrameTask
    posture_task: mink.PostureTask
    damping_task: mink.DampingTask
    freeze_task: Optional[mink.DofFreezingTask]
    collision_limit: Optional[mink.CollisionAvoidanceLimit]
    vel_limit: mink.VelocityLimit
    config_limit: mink.ConfigurationLimit
    qpos_idx_a: np.ndarray
    qpos_idx_b: np.ndarray
    dof_idx_b: List[int]


def build_mink_components(
    env: DualArmEnvironment,
    arm_a: RobosuiteArmController,
    arm_b: RobosuiteArmController,
    collision_check: bool = True,
    manipulated_object: Optional[str] = None,
) -> MinkComponents:
    """Construct all mink tasks, limits and constraints.

    Parameters
    ----------
    env : DualArmEnvironment
        Shared robosuite simulation.
    arm_a, arm_b : RobosuiteArmController
        Arm A has priority (pre-planned); arm B is reactively planned.
    collision_check : bool
        Whether to add a ``CollisionAvoidanceLimit``.
    manipulated_object : str, optional
        Grasped object name (adds object-vs-arm-B geom pairs).

    Returns
    -------
    MinkComponents
        All mink objects ready for use by the planner loop.
    """
    model = env.sim.model
    nm = get_native_mj_object(model)

    # ── Index maps ────────────────────────────────────────────────────
    jids_a = get_arm_joint_ids(nm, arm_a.robot_id)
    jids_b = get_arm_joint_ids(nm, arm_b.robot_id)
    qpos_idx_a = get_joint_qpos_indices(nm, jids_a)
    qpos_idx_b = get_joint_qpos_indices(nm, jids_b)
    dof_idx_b = [nm.jnt_dofadr[j] for j in jids_b]

    # ── mink.Configuration ────────────────────────────────────────────
    mink_config = mink.Configuration(nm, q=env.sim.data.qpos.copy())

    # ── FrameTask for arm B's end-effector ────────────────────────────
    eef_body_name = arm_b._eef_body_name()
    eef_task = mink.FrameTask(
        frame_name=eef_body_name,
        frame_type="body",
        position_cost=constants.MINK_POSITION_COST,
        orientation_cost=constants.MINK_ORIENTATION_COST,
        gain=constants.MINK_FRAME_GAIN,
        lm_damping=constants.MINK_LM_DAMPING,
    )

    # ── PostureTask — biases toward current (comfortable) config ──────
    posture_task = mink.PostureTask(model=nm, cost=constants.MINK_POSTURE_COST)

    # ── DampingTask — minimum-norm regulariser ────────────────────────
    damping_task = mink.DampingTask(model=nm, cost=constants.MINK_DAMPING_COST)

    # ── DofFreezingTask — freeze everything except arm B ──────────────
    frozen_dofs = get_all_dof_indices_except(nm, arm_b.robot_id)
    freeze_task: Optional[mink.DofFreezingTask] = None
    if frozen_dofs:
        freeze_task = mink.DofFreezingTask(model=nm, dof_indices=frozen_dofs)

    # ── Velocity / configuration limits ───────────────────────────────
    vel_limit = mink.VelocityLimit(nm)
    config_limit = mink.ConfigurationLimit(nm)

    # ── Collision avoidance limit ─────────────────────────────────────
    collision_limit: Optional[mink.CollisionAvoidanceLimit] = None
    if collision_check:
        geoms_a = get_robot_collision_geoms(nm, arm_a.robot_id)
        geoms_b = get_robot_collision_geoms(nm, arm_b.robot_id)
        obj_geoms = get_object_collision_geoms(nm, manipulated_object)
        env_geoms = get_environment_collision_geoms(
            nm,
            robot_ids=[arm_a.robot_id, arm_b.robot_id],
            object_name=manipulated_object,
        )

        geom_pairs: List[Tuple[Sequence[str], Sequence[str]]] = []
        if geoms_a and geoms_b:
            geom_pairs.append((geoms_b, geoms_a))
        if obj_geoms and geoms_b:
            geom_pairs.append((geoms_b, obj_geoms))
        if env_geoms and geoms_b:
            geom_pairs.append((geoms_b, env_geoms))

        if geom_pairs:
            collision_limit = mink.CollisionAvoidanceLimit(
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

    logger.info(
        "mink initialised for DualArmParallelPlanner "
        "(nq=%d, nv=%d, frozen_dofs=%d, solver=%s)",
        nm.nq, nm.nv, len(frozen_dofs), constants.MINK_QP_SOLVER,
    )

    return MinkComponents(
        native_model=nm,
        config=mink_config,
        eef_task=eef_task,
        posture_task=posture_task,
        damping_task=damping_task,
        freeze_task=freeze_task,
        collision_limit=collision_limit,
        vel_limit=vel_limit,
        config_limit=config_limit,
        qpos_idx_a=qpos_idx_a,
        qpos_idx_b=qpos_idx_b,
        dof_idx_b=dof_idx_b,
    )
