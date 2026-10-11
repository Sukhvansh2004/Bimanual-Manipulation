"""Collision and feasibility checker for the DualArmEnvironment.

Provides:
* **Grasp feasibility** — IK + collision checks for single and dual-arm
  grasp poses.
* **Trajectory feasibility** — per-waypoint IK + collision checks along
  an end-effector trajectory (single-arm and dual-arm).

Replaces synthetic geometric checks with actual MuJoCo physics and IK checks.
"""

import logging
from typing import List, Tuple, Union, Optional

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.ik.ik_utils import (
    enumerate_ik_solutions_for_arm,
    apply_ik_orientation_correction,
)
from bimanual_manipulation.planning.trajectory.utils import (
    WaypointFeasibilityResult,
)
from bimanual_manipulation.schema import GraspSchema
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.utils.mujoco_collision import in_collision
from bimanual_manipulation.utils.mujoco_utils import (
    set_qpos,
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object,
)

from contextlib import nullcontext

from robosuite.utils import transform_utils as T

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Waypoint feasibility helpers (IK + collision)
# ---------------------------------------------------------------------------


def resolve_ignore_names(
    object_name: Optional[str],
    ignore_names: Optional[List[str]],
    robot_id: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Resolve ignore names and ignore pairs for collision checking.

    Object-gripper contacts for the *holding* robot are selectively ignored
    via ``ignore_pairs`` so that object-arm collisions are still detected.

    Args:
        object_name: Geom name prefix of the grasped object (e.g. ``"flatPlate"``).
        ignore_names: Extra geom prefixes to blanket-ignore (e.g. the *other*
            robot ``["robot1", "gripper1"]``).
        robot_id: The robot that is holding the object.  When provided, an
            ``ignore_pair`` of ``(object_name, "gripper{robot_id}")`` is
            generated so only the gripper-object contact is ignored.

    Returns:
        ``(ignore_names, ignore_pairs)`` — a copy of *ignore_names* (or an
        empty list) to pass to ``in_collision`` as ``ignore_names``, and
        the generated pairs to pass as ``ignore_pairs`` (empty unless both
        *object_name* and *robot_id* are given).
    """
    current_ignore_names = list(ignore_names) if ignore_names else []
    ignore_pairs: List[Tuple[str, str]] = []

    if object_name and robot_id is not None:
        gripper_prefix = f"gripper{robot_id}"
        ignore_pairs.append((object_name, gripper_prefix))

    return current_ignore_names, ignore_pairs


def process_waypoint(
    env: DualArmEnvironment,
    robot_id: int,
    waypoint_index: int,
    pose: StampedPose,
    qpos_idx: np.ndarray,
    check_collisions: bool,
    ignore_names: List[str],
    last_config: Optional[np.ndarray],
    tf_manager: TransformManager,
    arm: str = "right",
    ignore_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[WaypointFeasibilityResult, Optional[np.ndarray]]:
    """Process a single waypoint: IK + Collision Check.

    Enumerates IK solutions (ranked by proximity to *last_config*) and
    returns the first collision-free one.  Leaves the arm's qpos set to
    the last checked solution; the caller is responsible for restoring
    simulation state.

    Args:
        env: Simulation environment.
        robot_id: Robot to solve for.
        waypoint_index: Index stored in the result.
        pose: Target EEF pose (body-frame convention; the IK orientation
            correction is applied here).
        qpos_idx: qpos indices of the arm joints.
        check_collisions: If ``False``, the first IK solution is accepted.
        ignore_names: Geom name prefixes to ignore in collision checking.
        last_config: Previous config used to rank IK solutions.
        tf_manager: Transform manager for frame conversions.
        arm: Arm side used for the EEF site name.
        ignore_pairs: Geom prefix pairs to ignore in collision checking.

    Returns:
        ``(result, config)`` — on success the collision-free config; when
        all solutions collide, an infeasible result and the first
        solution; on IK failure, an infeasible result (with
        ``collision_free=True``) and ``None``.
    """
    target_pose = apply_ik_orientation_correction(pose)
    site_name = f"gripper{robot_id}_{arm}_ft_frame"

    solutions = enumerate_ik_solutions_for_arm(
        env=env,
        robot_id=robot_id,
        target_pose=target_pose,
        arm=arm,
        site_name=site_name,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS,
        env_lock=None,
        tf_manager=tf_manager,
        q_reference=last_config,
    )

    if not solutions:
        logger.debug(f"  WP {waypoint_index}: IK failed")
        return WaypointFeasibilityResult(
            waypoint_index=waypoint_index,
            eef_pose=pose,
            joint_config=None,
            is_feasible=False,
            ik_success=False,
            collision_free=True,
        ), None

    model = env.sim.model
    data = env.sim.data

    for q_sol in solutions:
        collision_free = True
        if check_collisions:
            set_qpos(data, qpos_idx, q_sol)
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
            collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names, ignore_pairs=ignore_pairs)

        if collision_free:
            return WaypointFeasibilityResult(
                waypoint_index=waypoint_index,
                eef_pose=pose,
                joint_config=q_sol,
                is_feasible=True,
                ik_success=True,
                collision_free=True,
            ), q_sol

    logger.debug(f"  WP {waypoint_index}: IK succeeded but all in COLLISION")
    first_sol = solutions[0]
    return WaypointFeasibilityResult(
        waypoint_index=waypoint_index,
        eef_pose=pose,
        joint_config=first_sol,
        is_feasible=False,
        ik_success=True,
        collision_free=False,
    ), first_sol


def process_dual_waypoint(
    env: DualArmEnvironment,
    waypoint_index: int,
    pose0: StampedPose,
    pose1: StampedPose,
    qpos_idx0: np.ndarray,
    qpos_idx1: np.ndarray,
    check_collisions: bool,
    ignore_names: List[str],
    last_config0: Optional[np.ndarray],
    last_config1: Optional[np.ndarray],
    tf_manager: TransformManager,
    arm: str = "right",
    ignore_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[WaypointFeasibilityResult, WaypointFeasibilityResult, Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single dual-arm waypoint: IK for both + simultaneous Collision Check.

    When both arms have IK solutions, all solution pairs are tried and the
    first collision-free pair is returned.  When only one arm has IK
    solutions, each arm with solutions is checked individually with the
    other arm at its *last_config* (when given); the arm without IK is
    reported with ``ik_success=False`` and ``collision_free=False``.
    Leaves arm qpos modified; the caller restores simulation state.

    Args:
        env: Simulation environment.
        waypoint_index: Index stored in the results.
        pose0: Target EEF pose for robot 0 (body-frame convention).
        pose1: Target EEF pose for robot 1 (body-frame convention).
        qpos_idx0: qpos indices of robot 0's arm joints.
        qpos_idx1: qpos indices of robot 1's arm joints.
        check_collisions: If ``False``, the first IK solutions are accepted.
        ignore_names: Geom name prefixes to ignore in collision checking.
        last_config0: Previous robot-0 config (IK ranking reference).
        last_config1: Previous robot-1 config (IK ranking reference).
        tf_manager: Transform manager for frame conversions.
        arm: Arm side used for the EEF site names.
        ignore_pairs: Geom prefix pairs to ignore in collision checking.

    Returns:
        ``(result0, result1, config0, config1)``.  Configs are the
        selected (or first, if all collide) solutions, or ``None`` for an
        arm without IK solutions.
    """
    target0 = apply_ik_orientation_correction(pose0)
    target1 = apply_ik_orientation_correction(pose1)

    site0 = f"gripper0_{arm}_ft_frame"
    site1 = f"gripper1_{arm}_ft_frame"

    solutions0 = enumerate_ik_solutions_for_arm(
        env=env, robot_id=0, target_pose=target0, arm=arm, site_name=site0,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS, env_lock=None,
        tf_manager=tf_manager, q_reference=last_config0,
    )
    solutions1 = enumerate_ik_solutions_for_arm(
        env=env, robot_id=1, target_pose=target1, arm=arm, site_name=site1,
        max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS, env_lock=None,
        tf_manager=tf_manager, q_reference=last_config1,
    )

    model = env.sim.model
    data = env.sim.data

    if not solutions0 or not solutions1:
        res0_ik = bool(solutions0)
        res1_ik = bool(solutions1)

        best_q0 = None
        res0_col = False
        if solutions0:
            for q0 in solutions0:
                if check_collisions:
                    set_qpos(data, qpos_idx0, q0)
                    if last_config1 is not None:
                        set_qpos(data, qpos_idx1, last_config1)
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                    if not in_collision(env, env_lock=None, ignore_names=ignore_names, ignore_pairs=ignore_pairs):
                        best_q0 = q0
                        res0_col = True
                        break
                else:
                    best_q0 = q0
                    res0_col = True
                    break
            if best_q0 is None:
                best_q0 = solutions0[0]
                res0_col = False

        best_q1 = None
        res1_col = False
        if solutions1:
            for q1 in solutions1:
                if check_collisions:
                    set_qpos(data, qpos_idx1, q1)
                    if last_config0 is not None:
                        set_qpos(data, qpos_idx0, last_config0)
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                    if not in_collision(env, env_lock=None, ignore_names=ignore_names, ignore_pairs=ignore_pairs):
                        best_q1 = q1
                        res1_col = True
                        break
                else:
                    best_q1 = q1
                    res1_col = True
                    break
            if best_q1 is None:
                best_q1 = solutions1[0]
                res1_col = False

        res0 = WaypointFeasibilityResult(
            waypoint_index=waypoint_index, eef_pose=pose0, joint_config=best_q0,
            is_feasible=(res0_ik and res0_col), ik_success=res0_ik, collision_free=res0_col,
        )
        res1 = WaypointFeasibilityResult(
            waypoint_index=waypoint_index, eef_pose=pose1, joint_config=best_q1,
            is_feasible=(res1_ik and res1_col), ik_success=res1_ik, collision_free=res1_col,
        )
        return res0, res1, best_q0, best_q1

    pair_count = 0
    for q0 in solutions0:
        for q1 in solutions1:
            pair_count += 1
            collision_free = True
            if check_collisions:
                set_qpos(data, qpos_idx0, q0)
                set_qpos(data, qpos_idx1, q1)
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names, ignore_pairs=ignore_pairs)

            if collision_free:
                res0 = WaypointFeasibilityResult(
                    waypoint_index=waypoint_index, eef_pose=pose0, joint_config=q0,
                    is_feasible=True, ik_success=True, collision_free=True,
                )
                res1 = WaypointFeasibilityResult(
                    waypoint_index=waypoint_index, eef_pose=pose1, joint_config=q1,
                    is_feasible=True, ik_success=True, collision_free=True,
                )
                return res0, res1, q0, q1

    logger.debug(f"  WP {waypoint_index}: Checked {pair_count} IK pairs, all in COLLISION")
    res0 = WaypointFeasibilityResult(
        waypoint_index=waypoint_index, eef_pose=pose0, joint_config=solutions0[0],
        is_feasible=False, ik_success=True, collision_free=False,
    )
    res1 = WaypointFeasibilityResult(
        waypoint_index=waypoint_index, eef_pose=pose1, joint_config=solutions1[0],
        is_feasible=False, ik_success=True, collision_free=False,
    )
    return res0, res1, solutions0[0], solutions1[0]


# ---------------------------------------------------------------------------
# Grasp and trajectory feasibility checks
# ---------------------------------------------------------------------------


def check_single_grasp_feasibility(
    env: DualArmEnvironment,
    grasp: StampedPose,
    robot_id: int,
    tf_manager: TransformManager,
    arm: str = "right",
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    env_lock = None,
) -> bool:
    """Check if a single grasp is feasible (reachable via IK and collision-free).

    Simulation qpos/qvel are restored afterwards.

    Args:
        env: Simulation environment.
        grasp: Grasp pose (:class:`GraspSchema` or :class:`StampedPose`)
            in any frame known to *tf_manager*.
        robot_id: ID of robot (0 or 1).
        tf_manager: Transform manager for frame conversions.
        arm: Arm side.
        object_name: Geom name prefix of the object; all of its contacts
            are ignored.
        ignore_names: Geom name prefixes to ignore.
        env_lock: Optional ``threading.RLock`` guarding the simulation.

    Returns:
        ``True`` if at least one IK solution is collision-free.
    """
    # Prepare ignore names for collision checking
    current_ignore_names = list(ignore_names) if ignore_names else []
    if object_name and object_name not in current_ignore_names:
        current_ignore_names.append(object_name)

    # Prepare Target Pose
    target_pose = apply_ik_orientation_correction(grasp)

    # Get IK solutions for Flange site
    site_name = f"gripper{robot_id}_{arm}_ft_frame"
    
    solutions = enumerate_ik_solutions_for_arm(
        env,
        robot_id=robot_id,
        target_pose=target_pose,
        arm=arm,
        site_name=site_name,
        max_solutions=constants.GRASP_IK_MAX_SOLUTIONS,
        env_lock=env_lock,
        tf_manager=tf_manager
    )
    
    if not solutions:
        return False
        
    ctx = env_lock if env_lock is not None else nullcontext()
    
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids = get_arm_joint_ids(model, robot_id)
        qpos_idx = get_joint_qpos_indices(model, jids)
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        feasible = False
        try:
            for q in solutions:
                set_qpos(data, qpos_idx, q)
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                
                if not in_collision(env, env_lock=None, ignore_names=current_ignore_names): # Already locked by ctx
                    feasible = True
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
        
    return feasible

def find_feasible_grasp_pairs(
    env: DualArmEnvironment,
    grasps_robot0: List[GraspSchema],
    grasps_robot1: List[GraspSchema],
    tf_manager: TransformManager,
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    env_lock = None,
) -> List[Tuple[GraspSchema, GraspSchema]]:
    """Find feasible pairs of grasps (one for each robot).

    Each grasp is first filtered individually with
    :func:`check_single_grasp_feasibility`; surviving pairs are then
    checked with :func:`check_dual_arm_pose_feasibility` (up to
    :data:`constants.DUAL_GRASP_FEASIBILITY_MAX_PAIRS` IK pairs each).

    Args:
        env: Simulation environment.
        grasps_robot0: Grasps for robot 0.
        grasps_robot1: Grasps for robot 1.
        tf_manager: Transform manager for frame conversions.
        object_name: Geom name prefix of the object; all of its contacts
            are ignored.
        ignore_names: Geom name prefixes to ignore.
        env_lock: Optional ``threading.RLock`` guarding the simulation.

    Returns:
        List of ``(grasp0, grasp1)`` tuples that are simultaneously feasible.
    """
    feasible_pairs = []
    
    # Pre-filter single grasps to reduce N*M checks
    logger.info(f"Pre-filtering {len(grasps_robot0)} grasps for Robot 0...")
    valid_grasps0 = [g for g in grasps_robot0 if check_single_grasp_feasibility(
        env, g, 0, object_name=object_name, ignore_names=ignore_names, env_lock=env_lock, tf_manager=tf_manager
    )]
    logger.info(f"  -> {len(valid_grasps0)} valid grasps for Robot 0")
    
    logger.info(f"Pre-filtering {len(grasps_robot1)} grasps for Robot 1...")
    valid_grasps1 = [g for g in grasps_robot1 if check_single_grasp_feasibility(
        env, g, 1, object_name=object_name, ignore_names=ignore_names, env_lock=env_lock, tf_manager=tf_manager
    )]
    logger.info(f"  -> {len(valid_grasps1)} valid grasps for Robot 1")
    
    if not valid_grasps0 or not valid_grasps1:
        return []
        
    logger.info(f"Checking {len(valid_grasps0) * len(valid_grasps1)} pairs...")
    
    # Iterate pairs
    count = 0
    for g0 in valid_grasps0:
        for g1 in valid_grasps1:
            # Check pair feasibility
            # This runs IK for both and checks collision for pairs of solutions
            
            results = check_dual_arm_pose_feasibility(
                env, 
                eef_pose_robot0=g0,
                eef_pose_robot1=g1,
                object_name=object_name,
                ignore_names=ignore_names,
                max_pairs=constants.DUAL_GRASP_FEASIBILITY_MAX_PAIRS,
                env_lock=env_lock,
                tf_manager=tf_manager
            )
            
            # results is list of [q0, q1, feasible]
            is_feasible = any(r[2] for r in results)
            
            if is_feasible:
                feasible_pairs.append((g0, g1))
            
            count += 1
            if count % 10 == 0:
                logger.debug(f"Processed {count} pairs...")
                
    logger.info(f"Found {len(feasible_pairs)} feasible pairs.")
    return feasible_pairs


def check_dual_arm_pose_feasibility(
    env: DualArmEnvironment,
    eef_pose_robot0: StampedPose,
    eef_pose_robot1: StampedPose,
    tf_manager: TransformManager,
    *,
    object_name: Optional[str] = None,
    ignore_names: Optional[List[str]] = None,
    input_frame: str = constants.WORLD_FRAME,
    env_lock=None,
    arm: str = "right",
    max_solutions_per_arm: int = constants.IK_MAX_SOLUTIONS,
    contact_dist_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
    max_pairs: int = constants.DUAL_ARM_MAX_PAIRS,
) -> List[List[Union[np.ndarray, bool]]]:
    """Check feasibility for *all* (discretized) dual-arm joint solutions at given EEF poses.

    This function:

    1) Enumerates IK solutions for each arm via
       :func:`enumerate_ik_solutions_for_arm` (up to
       *max_solutions_per_arm* each).
    2) Forms joint-solution pairs (cartesian product) up to ``max_pairs``.
    3) For each pair, sets the MuJoCo qpos, runs forward kinematics, and
       checks for contacts (via ``in_collision``) not covered by the
       ignore list.

    Simulation qpos/qvel are restored afterwards.

    Args:
        env: Robosuite environment (or wrapper) exposing ``env.sim.model`` and ``env.sim.data``.
        eef_pose_robot0: Desired EEF pose for robot0 (frame given by its
            ``frame_id``).
        eef_pose_robot1: Desired EEF pose for robot1 (frame given by its
            ``frame_id``).
        tf_manager: TransformManager for frame conversions.
        object_name: Geom name prefix of the object; all of its contacts
            are ignored.
        ignore_names: Geom name prefixes to ignore.
        input_frame: Unused; poses are converted using their own
            ``frame_id``.
        env_lock: Optional threading.RLock used to guard sim state.
        arm: Arm side used for the EEF site names.
        max_solutions_per_arm: Maximum IK solutions per arm.
        contact_dist_threshold: Contact distance threshold passed to
            ``in_collision``.
        max_pairs: Maximum number of solution pairs to check.

    Returns:
        A list of entries ``[q_arm0 (7,), q_arm1 (7,), feasible_bool]``,
        one per checked pair.  Empty if either arm has no IK solutions.
    """
    
    # Prepare ignore names for collision checking
    current_ignore_names = list(ignore_names) if ignore_names else []
    if object_name and object_name not in current_ignore_names:
        current_ignore_names.append(object_name)

    # Prepare targets
    target0 = apply_ik_orientation_correction(eef_pose_robot0)
    target1 = apply_ik_orientation_correction(eef_pose_robot1)

    q_solutions_0 = enumerate_ik_solutions_for_arm(
        env,
        robot_id=0,
        target_pose=target0,
        arm=arm,
        site_name=f"gripper0_{arm}_ft_frame",
        tf_manager=tf_manager,
        env_lock=env_lock,
        max_solutions=max_solutions_per_arm,
    )

    q_solutions_1 = enumerate_ik_solutions_for_arm(
        env,
        robot_id=1,
        target_pose=target1,
        arm=arm,
        site_name=f"gripper1_{arm}_ft_frame",
        tf_manager=tf_manager,
        env_lock=env_lock,
        max_solutions=max_solutions_per_arm,
    )

    if len(q_solutions_0) == 0 or len(q_solutions_1) == 0:
        return []

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        jids0 = get_arm_joint_ids(model, 0)
        jids1 = get_arm_joint_ids(model, 1)
        qpos_idx0 = get_joint_qpos_indices(model, jids0)
        qpos_idx1 = get_joint_qpos_indices(model, jids1)

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        results: List[List[Union[np.ndarray, bool]]] = []
        try:
            pair_count = 0
            for q0 in q_solutions_0:
                for q1 in q_solutions_1:
                    pair_count += 1
                    if pair_count > int(max_pairs):
                        break

                    data.qpos[:] = qpos_backup
                    data.qvel[:] = qvel_backup
                    set_qpos(data, qpos_idx0, np.asarray(q0, dtype=float))
                    set_qpos(data, qpos_idx1, np.asarray(q1, dtype=float))
                    
                    # Unwrap for native call
                    mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))

                    feasible = not in_collision(
                        env, 
                        contact_dist_threshold=contact_dist_threshold, 
                        env_lock=None,
                        ignore_names=current_ignore_names
                    )
                    results.append([np.asarray(q0, dtype=float), np.asarray(q1, dtype=float), bool(feasible)])

                if pair_count > int(max_pairs):
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            
            # Unwrap for native call
            mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))

        return results


# ---------------------------------------------------------------------------
# Trajectory feasibility checking
# ---------------------------------------------------------------------------

def check_trajectory_feasibility(
    env: DualArmEnvironment,
    robot_id: int,
    eef_trajectory: List[StampedPose],
    tf_manager: TransformManager,
    object_trajectory: Optional[List[StampedPose]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None,
    ignore_names: Optional[List[str]] = None,
    arm: str = "right",
    collision_ignore_waypoints: Optional[List[int]] = None,
    collision_ignore_waypoint_names: Optional[List[str]] = None,
    collision_ignore_waypoint_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[WaypointFeasibilityResult], bool]:
    """Check trajectory feasibility and return detailed per-waypoint results.

    Poses are converted to the world frame if the first pose is not
    already in it.  At each waypoint the object (if given) is moved via
    its ``{object_name}_joint0`` free joint, then IK + collision checking
    is done with :func:`process_waypoint`, warm-started from the previous
    feasible config.  Object-vs-``gripper{robot_id}`` contacts are always
    ignored when *object_name* is set.  Simulation qpos/qvel are restored
    afterwards.

    Args:
        env: Environment instance.
        robot_id: Robot ID (0 or 1).
        eef_trajectory: List of StampedPose objects representing EEF poses.
        tf_manager: TransformManager to handle frame conversions.
        object_trajectory: Optional list of StampedPose objects for object
            poses; must match *eef_trajectory* in length.
        object_name: Name of the object.  Without it (or if its free joint
            is not found) *object_trajectory* is ignored.
        check_collisions: Whether to check collisions.
        env_lock: Optional threading.RLock.
        ignore_names: Geom name prefixes to blanket-ignore in collision
            checking.
        arm: Arm side ("right" or "left").
        collision_ignore_waypoints: Waypoint indices to skip collision checking for.
            Useful for waypoints near tight clearances where collisions are expected.
        collision_ignore_waypoint_names: Geom name prefixes to additionally ignore
            at the waypoints listed in collision_ignore_waypoints. When None (and
            collision_ignore_waypoint_pairs is also None), all collision checking
            is skipped at those waypoints.
        collision_ignore_waypoint_pairs: Geom prefix pairs ``(a, b)`` to
            additionally ignore at the waypoints listed in
            collision_ignore_waypoints.
        
    Returns:
        Tuple of (list of WaypointFeasibilityResult, overall_feasible bool).
        ``([], False)`` if the object trajectory length does not match.
    """
    if object_trajectory and len(object_trajectory) != len(eef_trajectory):
        logger.error("Object trajectory length mismatch")
        return [], False

    # Ensure frames are World
    if eef_trajectory and eef_trajectory[0].frame_id != constants.WORLD_FRAME:
        eef_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in eef_trajectory]

    if object_trajectory and object_trajectory[0].frame_id != constants.WORLD_FRAME:
        object_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in object_trajectory]

    ctx = env_lock if env_lock is not None else nullcontext()

    results: List[WaypointFeasibilityResult] = []
    all_feasible = True

    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids = get_arm_joint_ids(model, robot_id)
        qpos_idx = get_joint_qpos_indices(model, jids)
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        # Object joint index if needed
        obj_joint_addr = None
        if object_name:
            try:
                obj_joint_id = model.joint(f"{object_name}_joint0").id
                obj_qpos_addr = model.jnt_qposadr[obj_joint_id]
                obj_joint_addr = obj_qpos_addr
            except Exception:
                pass

        try:
            last_config = None # Keep track of the last valid configuration for warm start
            
            # Resolve ignore names and pairs
            # Object-gripper contacts are selectively ignored; object-arm contacts are still detected
            current_ignore_names, current_ignore_pairs = resolve_ignore_names(object_name, ignore_names, robot_id=robot_id)
            logger.debug(f"Collision checking: ignore_names={current_ignore_names}, ignore_pairs={current_ignore_pairs}")

            ignore_set = set(collision_ignore_waypoints) if collision_ignore_waypoints else set()
            # Determine whether ignored waypoints use selective or blanket ignore
            has_selective_ignore = (
                collision_ignore_waypoint_names is not None
                or collision_ignore_waypoint_pairs is not None
            )

            for i, pose in enumerate(eef_trajectory):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Check Waypoint (skip collision check for ignored waypoints)
                wp_check_collisions = check_collisions
                wp_ignore_names = current_ignore_names
                wp_ignore_pairs = current_ignore_pairs
                if i in ignore_set:
                    if has_selective_ignore:
                        # Merge extra ignore names/pairs for this waypoint
                        wp_ignore_names = current_ignore_names + (collision_ignore_waypoint_names or [])
                        wp_ignore_pairs = current_ignore_pairs + (collision_ignore_waypoint_pairs or [])
                    else:
                        # No specific names/pairs → skip all collision checking
                        wp_check_collisions = False
                result, new_config = process_waypoint(
                    env=env,
                    robot_id=robot_id,
                    waypoint_index=i,
                    pose=pose,
                    qpos_idx=qpos_idx,
                    check_collisions=wp_check_collisions,
                    ignore_names=wp_ignore_names,
                    last_config=last_config,
                    tf_manager=tf_manager,
                    arm=arm,
                    ignore_pairs=wp_ignore_pairs,
                )
                
                results.append(result)
                
                if not result.is_feasible:
                    all_feasible = False
                
                if result.is_feasible and result.collision_free:
                    last_config = new_config
                else:
                    last_config = None
                    
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results, all_feasible


def check_dual_trajectory_feasibility(
    env: DualArmEnvironment,
    eef_trajectory0: List[StampedPose],
    eef_trajectory1: List[StampedPose],
    tf_manager: TransformManager,
    object_trajectory: Optional[List[StampedPose]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None,
    ignore_names: Optional[List[str]] = None,
    arm: str = "right",
    collision_ignore_waypoints: Optional[List[int]] = None,
    collision_ignore_waypoint_names: Optional[List[str]] = None,
    collision_ignore_waypoint_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[WaypointFeasibilityResult], List[WaypointFeasibilityResult], bool]:
    """Check simultaneous feasibility of two robot trajectories.

    At each waypoint the object (if given) is moved via its
    ``{object_name}_joint0`` free joint and both arms are checked jointly
    with :func:`process_dual_waypoint`.  Object contacts with both
    ``gripper0`` and ``gripper1`` are ignored.  Unlike
    :func:`check_trajectory_feasibility`, poses are not converted to the
    world frame here.  Simulation qpos/qvel are restored afterwards.

    Args:
        env: Environment instance.
        eef_trajectory0: EEF poses for robot 0.
        eef_trajectory1: EEF poses for robot 1 (same length).
        tf_manager: TransformManager to handle frame conversions.
        object_trajectory: Optional object poses (same length).
        object_name: Name of the object.  Without it (or if its free joint
            is not found) *object_trajectory* is ignored.
        check_collisions: Whether to check collisions.
        env_lock: Optional threading.RLock.
        ignore_names: Geom name prefixes to blanket-ignore.
        arm: Arm side ("right" or "left").
        collision_ignore_waypoints: Waypoint indices with relaxed
            collision checking.
        collision_ignore_waypoint_names: Extra geom prefixes to ignore at
            those waypoints.  When both this and
            *collision_ignore_waypoint_pairs* are ``None``, collision
            checking is skipped entirely at those waypoints.
        collision_ignore_waypoint_pairs: Extra geom prefix pairs to ignore
            at those waypoints.

    Returns:
        Tuple of (results0, results1, overall_feasible bool).
        ``([], [], False)`` on a trajectory length mismatch.
    """
    if len(eef_trajectory0) != len(eef_trajectory1):
        logger.error("EEF trajectories length mismatch")
        return [], [], False

    if object_trajectory and len(object_trajectory) != len(eef_trajectory0):
        logger.error("Object trajectory length mismatch")
        return [], [], False

    ctx = env_lock if env_lock is not None else nullcontext()

    results0: List[WaypointFeasibilityResult] = []
    results1: List[WaypointFeasibilityResult] = []
    all_feasible = True

    logger.debug(f"Checking dual-arm trajectory feasibility for {len(eef_trajectory0)} waypoints...")

    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids0 = get_arm_joint_ids(model, 0)
        jids1 = get_arm_joint_ids(model, 1)
        qpos_idx0 = get_joint_qpos_indices(model, jids0)
        qpos_idx1 = get_joint_qpos_indices(model, jids1)
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        # Object joint index if needed
        obj_joint_addr = None
        if object_name:
            try:
                obj_joint_id = model.joint(f"{object_name}_joint0").id
                obj_qpos_addr = model.jnt_qposadr[obj_joint_id]
                obj_joint_addr = obj_qpos_addr
            except Exception:
                pass

        try:
            last_config0 = None 
            last_config1 = None
            
            # Build set of waypoint indices to skip collision checking
            ignore_set = set(collision_ignore_waypoints) if collision_ignore_waypoints else set()
            has_selective_ignore = (
                collision_ignore_waypoint_names is not None
                or collision_ignore_waypoint_pairs is not None
            )
            
            # Resolve ignore names and pairs for dual-arm
            # Both robots hold the object at different times, so create pairs for both grippers
            current_ignore_names, current_ignore_pairs = resolve_ignore_names(object_name, ignore_names)
            # Add object-gripper pairs for both robots
            if object_name:
                current_ignore_pairs.extend([
                    (object_name, "gripper0"),
                    (object_name, "gripper1"),
                ])

            for i in range(len(eef_trajectory0)):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Check Waypoint for both
                wp_check_collisions = check_collisions
                wp_ignore_names = current_ignore_names
                wp_ignore_pairs = current_ignore_pairs
                if i in ignore_set:
                    if has_selective_ignore:
                        wp_ignore_names = current_ignore_names + (collision_ignore_waypoint_names or [])
                        wp_ignore_pairs = current_ignore_pairs + (collision_ignore_waypoint_pairs or [])
                    else:
                        wp_check_collisions = False
                res0, res1, new_config0, new_config1 = process_dual_waypoint(
                    env=env,
                    waypoint_index=i,
                    pose0=eef_trajectory0[i],
                    pose1=eef_trajectory1[i],
                    qpos_idx0=qpos_idx0,
                    qpos_idx1=qpos_idx1,
                    check_collisions=wp_check_collisions,
                    ignore_names=wp_ignore_names,
                    last_config0=last_config0,
                    last_config1=last_config1,
                    tf_manager=tf_manager,
                    arm=arm,
                    ignore_pairs=wp_ignore_pairs,
                )
                
                results0.append(res0)
                results1.append(res1)
                
                if not (res0.is_feasible and res1.is_feasible):
                    all_feasible = False
                    logger.debug(f"  Waypoint {i} INFEASIBLE (Robot 0: {res0.is_feasible}, Robot 1: {res1.is_feasible})")
                
                if new_config0 is not None: last_config0 = new_config0
                if new_config1 is not None: last_config1 = new_config1

            if all_feasible:
                logger.debug("Dual-arm trajectory is FULLY FEASIBLE.")
            else:
                feasible_count = sum(1 for r0, r1 in zip(results0, results1) if r0.is_feasible and r1.is_feasible)
                logger.debug(f"Dual-arm trajectory is INFEASIBLE ({feasible_count}/{len(eef_trajectory0)} waypoints feasible).")

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results0, results1, all_feasible