"""
SE(3) RRT-Connect planner for collision-free object path planning.

Provides a ready-to-use SE(3) specialisation of the generic RRT-Connect
core (:func:`rrt_planner.rrt_connect_generic`) for planning collision-free
object trajectories through MuJoCo environments.

This module contains:

* **Object collision checking** — :func:`object_in_collision`,
  :func:`check_object_pose_collision`, :func:`_collision_free_segment`
* **SE(3) sampling / steering** — :func:`se3_pose_distance`,
  :func:`sample_random_pose`, :func:`steer`
* **SE(3) RRT-Connect** — :class:`RRTNode`, :class:`RRTResult`,
  :func:`rrt_connect_se3`

Usage example::

    result = rrt_connect_se3(
        env=env,
        object_name="mug_0",
        start_pos=start_pos,
        start_quat=start_quat,
        goal_pos=goal_pos,
        goal_quat=goal_quat,
    )
    if result.success:
        smoothed = smooth_path(env, "mug_0", result.path)
        dense = densify_path(smoothed, 200)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.rrt.core import rrt_connect_generic
from robosuite_franka_manipulation.planning.utils.collision_utils import (
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
from robosuite_franka_manipulation.planning.utils.mujoco_utils import get_native_mj_object
from robosuite_franka_manipulation.utils.se3 import (
    SE3_log,
    compute_se3_distance,
    interpolate_pose,
    pose_to_SE3,
)
from robosuite.utils import transform_utils as T

from robosuite_franka_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# Object Collision Checking
# ============================================================================

def object_in_collision(
    model,
    data,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    contact_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Check if the object collides with any non-robot geometry.

    Contacts between the object and robot arms/grippers are ignored.

    Args:
        model: MuJoCo model (native or wrapped).
        data: MuJoCo data (native or wrapped).
        object_geom_ids: Array of geom IDs belonging to the object.
        robot_mask: Boolean mask ``(ngeom,)`` — True for robot/gripper geoms
            (obtain via ``collision_utils.get_robot_geom_mask``).
        contact_threshold: Contact distance threshold.

    Returns:
        True if the object collides with any environment geometry.
    """
    ncon = int(data.ncon)
    if ncon <= 0:
        return False

    g1 = np.asarray(data.contact.geom1[:ncon])
    g2 = np.asarray(data.contact.geom2[:ncon])
    dist = np.asarray(data.contact.dist[:ncon])

    obj_mask = np.zeros(model.ngeom, dtype=bool)
    obj_mask[object_geom_ids] = True

    active = dist <= contact_threshold
    involves_obj = obj_mask[g1] | obj_mask[g2]
    active &= involves_obj

    other_is_robot = np.where(obj_mask[g1], robot_mask[g2], robot_mask[g1])
    active &= ~other_is_robot

    both_obj = obj_mask[g1] & obj_mask[g2]
    active &= ~both_obj

    return bool(np.any(active))


def check_object_pose_collision(
    env: DualArmEnvironment,
    object_name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    obj_joint_addr: int,
    env_lock=None,
) -> bool:
    """Set the object to a given pose and check for environment collisions.

    Args:
        env: The simulation environment.
        object_name: Name of the object.
        position: 3D position.
        quaternion: Quaternion [x, y, z, w] (scipy convention).
        object_geom_ids: Pre-computed object geom IDs.
        robot_mask: Boolean mask ``(ngeom,)`` for robot/gripper geoms.
        obj_joint_addr: qpos address for the object's free joint.
        env_lock: Optional threading lock.

    Returns:
        True if the object is in collision at this pose.
    """
    model = env.sim.model
    data = env.sim.data

    native_model = get_native_mj_object(model)
    native_data = get_native_mj_object(data)

    quat_wxyz = T.convert_quat(quaternion, to="wxyz")
    data.qpos[obj_joint_addr:obj_joint_addr + 3] = position
    data.qpos[obj_joint_addr + 3:obj_joint_addr + 7] = quat_wxyz

    mujoco.mj_fwdPosition(native_model, native_data)

    return object_in_collision(
        model, data, object_geom_ids, robot_mask,
        contact_threshold=constants.COLLISION_CONTACT_THRESHOLD,
    )


# ============================================================================
# SE(3) Distance & Sampling Utilities
# ============================================================================

def se3_pose_distance(
    pos1: np.ndarray, quat1: np.ndarray,
    pos2: np.ndarray, quat2: np.ndarray,
    rotation_weight: float = None,
) -> float:
    """Compute weighted distance between two SE(3) poses.

    Thin wrapper around :func:`utils.se3.compute_se3_distance` that accepts
    two poses instead of a pre-computed twist.

    Args:
        pos1, quat1: First pose (quaternion in [x,y,z,w]).
        pos2, quat2: Second pose.
        rotation_weight: Weight for rotation distance in radians
                         (default from constants).

    Returns:
        Scalar distance.
    """
    if rotation_weight is None:
        rotation_weight = constants.RRT_ROTATION_WEIGHT

    T1 = pose_to_SE3(pos1, quat1)
    T2 = pose_to_SE3(pos2, quat2)
    T_rel = np.linalg.inv(T1) @ T2
    twist = SE3_log(T_rel)
    return compute_se3_distance(twist, translation_weight=1.0, rotation_weight=rotation_weight)


def sample_random_pose(
    pos_bounds: np.ndarray,
    start_quat: np.ndarray,
    goal_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random SE(3) pose within workspace bounds.

    Position is sampled uniformly in the bounding box. Orientation is
    sampled by interpolating between start/goal orientation at a random
    fraction, then perturbing with a small random rotation.

    Args:
        pos_bounds: ``(2, 3)`` array of ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``.
        start_quat: Start orientation ``[x,y,z,w]``.
        goal_quat: Goal orientation ``[x,y,z,w]``.

    Returns:
        ``(position, quaternion)`` tuple.
    """
    pos = np.random.uniform(pos_bounds[0], pos_bounds[1])

    t = np.random.uniform(0.0, 1.0)
    _, base_quat = interpolate_pose(np.zeros(3), start_quat, np.zeros(3), goal_quat, t)
    r_base = R.from_quat(base_quat)

    noise_angle = np.random.uniform(0, constants.RRT_ORIENTATION_NOISE)
    noise_axis = np.random.randn(3)
    noise_axis /= np.linalg.norm(noise_axis) + 1e-10
    r_noise = R.from_rotvec(noise_angle * noise_axis)

    r_sample = r_base * r_noise
    return pos, r_sample.as_quat()


def steer(
    from_pos: np.ndarray, from_quat: np.ndarray,
    to_pos: np.ndarray, to_quat: np.ndarray,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Steer from one pose toward another, limited by *step_size*.

    If the distance is less than *step_size*, returns the target pose.

    Args:
        from_pos, from_quat: Current pose.
        to_pos, to_quat: Target pose.
        step_size: Maximum step distance.

    Returns:
        ``(position, quaternion)`` of the steered pose.
    """
    dist = se3_pose_distance(from_pos, from_quat, to_pos, to_quat)
    if dist <= step_size:
        return to_pos.copy(), to_quat.copy()

    t = step_size / dist
    pos, quat = interpolate_pose(from_pos, from_quat, to_pos, to_quat, t)
    return pos, quat


def _collision_free_segment(
    env: DualArmEnvironment,
    object_name: str,
    pos1: np.ndarray, quat1: np.ndarray,
    pos2: np.ndarray, quat2: np.ndarray,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    obj_joint_addr: int,
    resolution: float = None,
    env_lock=None,
    skip_endpoints: bool = False,
) -> bool:
    """Check if a straight-line SE(3) segment is collision free.

    Discretises the segment and checks each intermediate pose.

    Args:
        env: Environment.
        object_name: Object name.
        pos1, quat1: Start of segment.
        pos2, quat2: End of segment.
        object_geom_ids: Pre-computed geom IDs.
        robot_mask: Boolean mask ``(ngeom,)`` for robot/gripper geoms.
        obj_joint_addr: qpos address for the object's free joint.
        resolution: Check resolution (distance between checks).
        env_lock: Optional threading lock.
        skip_endpoints: If True, skip collision checks at the exact
            start and end of the segment.

    Returns:
        True if the segment is collision-free.
    """
    if resolution is None:
        resolution = constants.RRT_COLLISION_RESOLUTION

    dist = se3_pose_distance(pos1, quat1, pos2, quat2)
    n_checks = max(2, int(np.ceil(dist / resolution)))

    start_idx = 1 if skip_endpoints else 0
    end_idx = n_checks - 1 if skip_endpoints else n_checks

    for i in range(start_idx, end_idx + 1):
        t = i / float(n_checks)
        pos, quat = interpolate_pose(pos1, quat1, pos2, quat2, t)
        if check_object_pose_collision(
            env, object_name, pos, quat,
            object_geom_ids, robot_mask, obj_joint_addr,
            env_lock=env_lock,
        ):
            return False
    return True


# ============================================================================
# SE(3) RRT-Connect
# ============================================================================

@dataclass
class RRTNode:
    """A node in the RRT tree."""

    pos: np.ndarray
    quat: np.ndarray
    parent: int = -1


@dataclass
class RRTResult:
    """Result of the RRT-Connect planner."""

    success: bool
    path: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    iterations: int = 0
    tree_a_size: int = 0
    tree_b_size: int = 0


def rrt_connect_se3(
    env: DualArmEnvironment,
    object_name: str,
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    env_lock=None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    pos_bounds: np.ndarray = None,
    rrt_vis: Optional["RRTVisualizer"] = None,
) -> RRTResult:
    """Plan a collision-free path for an object using RRT-Connect in SE(3).

    Ignores collisions with robot arms and grippers.

    Args:
        env: The simulation environment.
        object_name: Name of the object to plan for.
        start_pos: Start position ``[x, y, z]``.
        start_quat: Start quaternion ``[x, y, z, w]``.
        goal_pos: Goal position.
        goal_quat: Goal quaternion.
        env_lock: Optional threading lock.
        max_iterations: Maximum RRT iterations.
        step_size: Step size for tree extension.
        goal_bias: Probability of sampling the goal.
        pos_bounds: Workspace bounds ``(2, 3)`` array.
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.

    Returns:
        :class:`RRTResult` with success flag and path.
    """
    if max_iterations is None:
        max_iterations = constants.RRT_MAX_ITERATIONS
    if step_size is None:
        step_size = constants.RRT_STEP_SIZE
    if goal_bias is None:
        goal_bias = constants.RRT_GOAL_BIAS
    if pos_bounds is None:
        pos_bounds = np.array(constants.RRT_WORKSPACE_BOUNDS)

    rotation_weight = constants.RRT_ROTATION_WEIGHT

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data

        object_geom_ids = get_geom_ids_by_prefix(model, object_name)
        robot_mask = get_robot_geom_mask(model)

        if len(object_geom_ids) == 0:
            logger.error("No geoms found for object '%s'", object_name)
            return RRTResult(success=False)

        try:
            obj_joint_id = model.joint(f"{object_name}_joint0").id
            obj_joint_addr = int(model.jnt_qposadr[obj_joint_id])
        except Exception:
            logger.error("Could not find joint for object '%s'", object_name)
            return RRTResult(success=False)

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # Fast path: try direct connection first
            if _collision_free_segment(
                env, object_name,
                start_pos, start_quat, goal_pos, goal_quat,
                object_geom_ids, robot_mask, obj_joint_addr,
                skip_endpoints=True,
            ):
                logger.info("Direct path is collision-free, no RRT needed.")
                return RRTResult(
                    success=True,
                    path=[(start_pos.copy(), start_quat.copy()),
                          (goal_pos.copy(), goal_quat.copy())],
                    iterations=0, tree_a_size=1, tree_b_size=1,
                )

            logger.info(
                "Direct path has collisions. Running RRT-Connect "
                "(max_iter=%d)...", max_iterations,
            )

            # ── Build domain callbacks ────────────────────────────────
            def _distance(s1, s2):
                return se3_pose_distance(
                    s1[0], s1[1], s2[0], s2[1], rotation_weight,
                )

            def _steer_cb(from_s, to_s, step):
                return steer(from_s[0], from_s[1], to_s[0], to_s[1], step)

            def _sample():
                return sample_random_pose(
                    pos_bounds, start_quat, goal_quat,
                )

            def _collision(s):
                return check_object_pose_collision(
                    env, object_name, s[0], s[1],
                    object_geom_ids, robot_mask, obj_joint_addr,
                )

            def _seg_free(s1, s2):
                return _collision_free_segment(
                    env, object_name,
                    s1[0], s1[1], s2[0], s2[1],
                    object_geom_ids, robot_mask, obj_joint_addr,
                    skip_endpoints=True,
                )

            def _copy(s):
                return (s[0].copy(), s[1].copy())

            # ── Visualisation hooks ───────────────────────────────────
            _on_extend = None
            _on_collision = None
            _on_iteration = None

            if rrt_vis is not None:
                rrt_vis.draw_start_goal(start_pos, goal_pos)
                rrt_vis.draw_bounds(pos_bounds)

                def _on_extend(parent_s, new_s, is_start):
                    rrt_vis.draw_tree_edge(
                        parent_s[0], new_s[0], is_tree_a=is_start,
                    )

                def _on_collision(s):
                    rrt_vis.draw_collision_sample(s[0])

                def _on_iteration(it, ta_size, tb_size):
                    rrt_vis.log_iteration(it, ta_size, tb_size)

            # ── Run generic RRT-Connect ───────────────────────────────
            success, path, iters, ta, tb = rrt_connect_generic(
                start_state=(start_pos.copy(), start_quat.copy()),
                goal_state=(goal_pos.copy(), goal_quat.copy()),
                distance_fn=_distance,
                steer_fn=_steer_cb,
                sample_fn=_sample,
                collision_fn=_collision,
                segment_free_fn=_seg_free,
                copy_fn=_copy,
                step_size=step_size,
                goal_bias=goal_bias,
                max_iterations=max_iterations,
                connect_max_steps=constants.RRT_CONNECT_MAX_STEPS,
                on_extend=_on_extend,
                on_collision=_on_collision,
                on_iteration=_on_iteration,
            )

            if success:
                logger.info(
                    "RRT-Connect found path in %d iterations "
                    "(tree sizes: %d, %d)", iters, ta, tb,
                )
                if rrt_vis is not None:
                    rrt_vis.draw_path(path)
            else:
                logger.warning(
                    "RRT-Connect failed after %d iterations.", max_iterations,
                )

            return RRTResult(
                success=success,
                path=path,
                iterations=iters,
                tree_a_size=ta,
                tree_b_size=tb,
            )

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
