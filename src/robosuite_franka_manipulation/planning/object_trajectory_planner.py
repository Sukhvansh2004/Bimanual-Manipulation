"""
Collision-aware object trajectory planner using RRT-Connect in SE(3).

Plans a collision-free path for the object through the workspace,
ignoring contacts with robot arms and grippers (since the robots
will be carrying the object).

Uses MuJoCo for collision checking against static environment geometry
(tables, shelves, walls, etc.).
"""

import numpy as np
import mujoco
import logging
from typing import List, Optional, Tuple
from contextlib import nullcontext
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.utils.se3 import (
    interpolate_pose,
    pose_to_SE3,
    SE3_log,
    compute_se3_distance,
)
from robosuite_franka_manipulation.planning.utils.mujoco_utils import get_native_mj_object
from robosuite_franka_manipulation.planning.utils.collision_utils import (
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
from robosuite.utils import transform_utils as T

logger = logging.getLogger(__name__)


# ============================================================================
# Object Collision Checking
# ============================================================================

def object_in_collision(
    model,
    data,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    contact_threshold: float = 1e-6,
) -> bool:
    """
    Check if the object is in collision with any non-robot geometry.
    
    Looks for contacts that involve at least one object geom AND
    at least one non-robot geom (i.e., environment obstacles).
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

    # Build object membership mask
    obj_mask = np.zeros(model.ngeom, dtype=bool)
    obj_mask[object_geom_ids] = True

    # Contacts below distance threshold
    active = dist <= contact_threshold

    # Contact involves the object on at least one side
    involves_obj = obj_mask[g1] | obj_mask[g2]
    active &= involves_obj

    # Exclude contacts where the *other* geom is a robot geom
    other_is_robot = np.where(obj_mask[g1], robot_mask[g2], robot_mask[g1])
    active &= ~other_is_robot

    # Also exclude self-collisions within the object
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
    """
    Set the object to a given pose and check for collisions with the environment.
    
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

    # Set object pose
    quat_wxyz = T.convert_quat(quaternion, to="wxyz")
    data.qpos[obj_joint_addr:obj_joint_addr + 3] = position
    data.qpos[obj_joint_addr + 3:obj_joint_addr + 7] = quat_wxyz

    # Forward kinematics to compute contacts
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
    """
    Compute weighted distance between two SE(3) poses.
    
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
    """
    Sample a random SE(3) pose within workspace bounds.
    
    Position is sampled uniformly in the bounding box.
    Orientation is sampled by interpolating between start/goal orientation
    at a random fraction, then perturbing with a small random rotation.
    This biases the orientation search toward the reachable range.
    
    Args:
        pos_bounds: (2, 3) array of [[x_min, y_min, z_min], [x_max, y_max, z_max]].
        start_quat: Start orientation [x,y,z,w].
        goal_quat: Goal orientation [x,y,z,w].
        
    Returns:
        (position, quaternion) tuple.
    """
    pos = np.random.uniform(pos_bounds[0], pos_bounds[1])
    
    # Interpolate between start and goal orientation, then add noise
    t = np.random.uniform(0.0, 1.0)
    # Reuse the existing interpolate_pose for slerp (position output is ignored)
    _, base_quat = interpolate_pose(np.zeros(3), start_quat, np.zeros(3), goal_quat, t)
    r_base = R.from_quat(base_quat)
    
    # Add small random perturbation
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
    """
    Steer from one pose toward another, limited by step_size.
    
    If the distance is less than step_size, returns the target pose.
    Otherwise returns an intermediate pose at distance step_size.
    
    Args:
        from_pos, from_quat: Current pose.
        to_pos, to_quat: Target pose.
        step_size: Maximum step distance.
        
    Returns:
        (position, quaternion) of the steered pose.
    """
    dist = se3_pose_distance(from_pos, from_quat, to_pos, to_quat)
    if dist <= step_size:
        return to_pos.copy(), to_quat.copy()
    
    t = step_size / dist
    pos, quat = interpolate_pose(from_pos, from_quat, to_pos, to_quat, t)
    return pos, quat


# ============================================================================
# RRT-Connect Planner
# ============================================================================

@dataclass
class RRTNode:
    """A node in the RRT tree."""
    pos: np.ndarray
    quat: np.ndarray
    parent: int = -1  # Index of parent node in the tree


@dataclass 
class RRTResult:
    """Result of the RRT-Connect planner."""
    success: bool
    path: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    iterations: int = 0
    tree_a_size: int = 0
    tree_b_size: int = 0


def _nearest_node(
    tree: List[RRTNode],
    pos: np.ndarray,
    quat: np.ndarray,
    rotation_weight: float,
) -> int:
    """Find the index of the nearest node in the tree."""
    best_idx = 0
    best_dist = float("inf")
    for i, node in enumerate(tree):
        d = se3_pose_distance(node.pos, node.quat, pos, quat, rotation_weight)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _extract_path(tree_a: List[RRTNode], idx_a: int,
                  tree_b: List[RRTNode], idx_b: int,
                  a_is_start: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract the path from start to goal through the two trees."""
    # Trace back in tree_a
    path_a = []
    idx = idx_a
    while idx >= 0:
        node = tree_a[idx]
        path_a.append((node.pos.copy(), node.quat.copy()))
        idx = node.parent
    path_a.reverse()
    
    # Trace back in tree_b
    path_b = []
    idx = idx_b
    while idx >= 0:
        node = tree_b[idx]
        path_b.append((node.pos.copy(), node.quat.copy()))
        idx = node.parent
    # path_b goes from the connection point to the root of tree_b
    
    if a_is_start:
        # tree_a starts at start, tree_b starts at goal
        # path_a: start -> connection, path_b: connection -> goal (already correct after reverse)
        # But path_b is currently goal -> connection, so don't reverse
        return path_a + path_b
    else:
        # tree_a starts at goal, tree_b starts at start
        # path_b: start -> connection, path_a: connection -> goal
        path_b.reverse()
        return path_b + path_a[::-1]


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
    """
    Check if a straight-line SE(3) segment is collision free.
    
    Discretizes the segment and checks each intermediate pose.
    
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
            start (t=0) and end (t=1) of the segment.  Useful when the
            endpoints are known to be valid (e.g. the object resting on
            a surface that would register as a contact).
        
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
) -> RRTResult:
    """
    Plan a collision-free path for an object using RRT-Connect in SE(3).
    
    Ignores collisions with robot arms and grippers (the robots carry the object).
    Only detects collisions with static environment geometry.
    
    Args:
        env: The simulation environment.
        object_name: Name of the object to plan for.
        start_pos: Start position [x, y, z].
        start_quat: Start quaternion [x, y, z, w].
        goal_pos: Goal position [x, y, z].
        goal_quat: Goal quaternion [x, y, z, w].
        env_lock: Optional threading lock.
        max_iterations: Maximum RRT iterations.
        step_size: Step size for tree extension.
        goal_bias: Probability of sampling the goal.
        pos_bounds: Workspace bounds (2, 3) array.
        
    Returns:
        RRTResult with success flag and path.
    """
    # Defaults from constants
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
        
        # Pre-compute geom IDs
        object_geom_ids = get_geom_ids_by_prefix(model, object_name)
        robot_mask = get_robot_geom_mask(model)
        
        if len(object_geom_ids) == 0:
            logger.error(f"No geoms found for object '{object_name}'")
            return RRTResult(success=False)
        
        # Get object joint address
        try:
            obj_joint_id = model.joint(f"{object_name}_joint0").id
            obj_joint_addr = int(model.jnt_qposadr[obj_joint_id])
        except Exception:
            logger.error(f"Could not find joint for object '{object_name}'")
            return RRTResult(success=False)
        
        # Backup sim state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        try:
            # Skip collision checking at start/goal — the object is
            # physically at these poses (e.g. resting on a table/shelf)
            # so surface contacts are expected, not true collisions.
            
            # First try direct connection (fast path)
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
                    iterations=0,
                    tree_a_size=1,
                    tree_b_size=1,
                )
            
            logger.info(f"Direct path has collisions. Running RRT-Connect (max_iter={max_iterations})...")
            
            # Initialize trees
            tree_a = [RRTNode(pos=start_pos.copy(), quat=start_quat.copy())]
            tree_b = [RRTNode(pos=goal_pos.copy(), quat=goal_quat.copy())]
            a_is_start = True  # Track which tree is which
            
            for iteration in range(max_iterations):
                # Sample random pose (with goal bias)
                if np.random.random() < goal_bias:
                    if a_is_start:
                        sample_pos, sample_quat = goal_pos.copy(), goal_quat.copy()
                    else:
                        sample_pos, sample_quat = start_pos.copy(), start_quat.copy()
                else:
                    sample_pos, sample_quat = sample_random_pose(
                        pos_bounds, start_quat, goal_quat,
                    )
                
                # EXTEND tree_a toward sample
                nearest_idx = _nearest_node(tree_a, sample_pos, sample_quat, rotation_weight)
                nearest = tree_a[nearest_idx]
                new_pos, new_quat = steer(
                    nearest.pos, nearest.quat,
                    sample_pos, sample_quat,
                    step_size,
                )
                
                # Check collision for the extension
                if check_object_pose_collision(
                    env, object_name, new_pos, new_quat,
                    object_geom_ids, robot_mask, obj_joint_addr,
                ):
                    # Swap trees and continue
                    tree_a, tree_b = tree_b, tree_a
                    a_is_start = not a_is_start
                    continue
                
                # Check the edge is collision-free
                if not _collision_free_segment(
                    env, object_name,
                    nearest.pos, nearest.quat, new_pos, new_quat,
                    object_geom_ids, robot_mask, obj_joint_addr,
                ):
                    tree_a, tree_b = tree_b, tree_a
                    a_is_start = not a_is_start
                    continue
                
                # Add node to tree_a
                new_idx_a = len(tree_a)
                tree_a.append(RRTNode(pos=new_pos, quat=new_quat, parent=nearest_idx))
                
                # CONNECT tree_b toward the new node in tree_a
                connected = False
                connect_idx_b = -1
                
                nearest_b_idx = _nearest_node(tree_b, new_pos, new_quat, rotation_weight)
                current_pos = tree_b[nearest_b_idx].pos
                current_quat = tree_b[nearest_b_idx].quat
                parent_b_idx = nearest_b_idx
                
                for _ in range(constants.RRT_CONNECT_MAX_STEPS):
                    steered_pos, steered_quat = steer(
                        current_pos, current_quat,
                        new_pos, new_quat,
                        step_size,
                    )
                    
                    if check_object_pose_collision(
                        env, object_name, steered_pos, steered_quat,
                        object_geom_ids, robot_mask, obj_joint_addr,
                    ):
                        break
                    
                    if not _collision_free_segment(
                        env, object_name,
                        current_pos, current_quat, steered_pos, steered_quat,
                        object_geom_ids, robot_mask, obj_joint_addr,
                    ):
                        break
                    
                    connect_idx_b = len(tree_b)
                    tree_b.append(RRTNode(pos=steered_pos, quat=steered_quat, parent=parent_b_idx))
                    parent_b_idx = connect_idx_b
                    
                    # Check if we've reached the target node
                    if se3_pose_distance(steered_pos, steered_quat, new_pos, new_quat, rotation_weight) < step_size * 0.5:
                        connected = True
                        break
                    
                    current_pos = steered_pos
                    current_quat = steered_quat
                
                if connected:
                    logger.info(f"RRT-Connect found path in {iteration + 1} iterations "
                               f"(tree sizes: {len(tree_a)}, {len(tree_b)})")
                    
                    path = _extract_path(tree_a, new_idx_a, tree_b, connect_idx_b, a_is_start)
                    
                    return RRTResult(
                        success=True,
                        path=path,
                        iterations=iteration + 1,
                        tree_a_size=len(tree_a),
                        tree_b_size=len(tree_b),
                    )
                
                # Swap trees for balanced growth
                tree_a, tree_b = tree_b, tree_a
                a_is_start = not a_is_start
            
            logger.warning(f"RRT-Connect failed after {max_iterations} iterations.")
            return RRTResult(
                success=False,
                iterations=max_iterations,
                tree_a_size=len(tree_a),
                tree_b_size=len(tree_b),
            )
        
        finally:
            # Restore sim state
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))


# ============================================================================
# Path Post-Processing
# ============================================================================

def smooth_path(
    env: DualArmEnvironment,
    object_name: str,
    path: List[Tuple[np.ndarray, np.ndarray]],
    env_lock=None,
    max_smoothing_iterations: int = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Smooth an RRT path using iterative shortcutting.
    
    Randomly selects two waypoints and tries to connect them directly.
    If the direct connection is collision-free, removes the intermediate
    waypoints.
    
    Args:
        env: Environment.
        object_name: Object name.
        path: List of (position, quaternion) waypoints.
        env_lock: Optional threading lock.
        max_smoothing_iterations: Number of shortcut attempts.
        
    Returns:
        Smoothed path.
    """
    if max_smoothing_iterations is None:
        max_smoothing_iterations = constants.RRT_SMOOTHING_ITERATIONS
    
    if len(path) <= 2:
        return path
    
    ctx = env_lock if env_lock is not None else nullcontext()
    
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        object_geom_ids = get_geom_ids_by_prefix(model, object_name)
        robot_mask = get_robot_geom_mask(model)
        
        try:
            obj_joint_id = model.joint(f"{object_name}_joint0").id
            obj_joint_addr = int(model.jnt_qposadr[obj_joint_id])
        except Exception:
            return path
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        smoothed = list(path)
        
        try:
            for _ in range(max_smoothing_iterations):
                if len(smoothed) <= 2:
                    break
                
                # Pick two random indices
                i = np.random.randint(0, len(smoothed) - 1)
                j = np.random.randint(i + 1, len(smoothed))
                
                if j - i <= 1:
                    continue
                
                pos_i, quat_i = smoothed[i]
                pos_j, quat_j = smoothed[j]
                
                # Skip endpoint checks when the shortcut touches
                # the start (index 0) or goal (last index) poses.
                skip = (i == 0) or (j == len(smoothed) - 1)
                
                if _collision_free_segment(
                    env, object_name,
                    pos_i, quat_i, pos_j, quat_j,
                    object_geom_ids, robot_mask, obj_joint_addr,
                    skip_endpoints=skip,
                ):
                    # Remove intermediate waypoints
                    smoothed = smoothed[:i + 1] + smoothed[j:]
            
            logger.info(f"Path smoothing: {len(path)} -> {len(smoothed)} waypoints")
        
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return smoothed


def densify_path(
    path: List[Tuple[np.ndarray, np.ndarray]],
    num_points: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Resample a sparse path to have a fixed number of evenly-spaced waypoints.
    
    Uses SE(3) interpolation to produce smooth intermediate poses.
    
    Args:
        path: Sparse path from the planner.
        num_points: Desired number of output waypoints (including start and goal).
        
    Returns:
        Dense path with num_points waypoints.
    """
    if len(path) <= 1 or num_points <= 1:
        return path
    
    # Compute cumulative arc length (SE(3) distance)
    seg_lengths = []
    for i in range(len(path) - 1):
        d = se3_pose_distance(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        seg_lengths.append(d)
    
    total_length = sum(seg_lengths)
    if total_length < 1e-10:
        # Degenerate case: all waypoints are the same
        return [path[0]] * num_points
    
    # Cumulative parameter along the path [0, 1]
    cum_lengths = np.zeros(len(path))
    for i in range(1, len(path)):
        cum_lengths[i] = cum_lengths[i - 1] + seg_lengths[i - 1]
    cum_lengths /= total_length  # Normalize to [0, 1]
    
    # Generate evenly spaced parameter values
    dense = []
    for k in range(num_points):
        s = k / float(num_points - 1)
        
        # Find which segment this parameter falls in
        seg_idx = np.searchsorted(cum_lengths, s, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(path) - 2)
        
        # Local interpolation parameter within this segment
        seg_start = cum_lengths[seg_idx]
        seg_end = cum_lengths[seg_idx + 1]
        if seg_end - seg_start < 1e-10:
            t_local = 0.0
        else:
            t_local = (s - seg_start) / (seg_end - seg_start)
        
        pos, quat = interpolate_pose(
            path[seg_idx][0], path[seg_idx][1],
            path[seg_idx + 1][0], path[seg_idx + 1][1],
            t_local,
        )
        dense.append((pos, quat))
    
    return dense


# ============================================================================
# High-Level API
# ============================================================================

def plan_object_trajectory(
    env: DualArmEnvironment,
    object_name: str,
    start_pose: StampedPose,
    target_pose: StampedPose,
    tf_manager=None,
    num_steps: int = None,
    env_lock=None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    pos_bounds: np.ndarray = None,
    smoothing_iterations: int = None,
) -> Tuple[List[StampedPose], bool]:
    """
    Plan a collision-free object trajectory from start to target pose.
    
    Uses RRT-Connect in SE(3) with MuJoCo collision checking.
    Ignores collisions with robot arms/grippers.
    
    If the direct path is collision-free, returns the simple interpolated
    trajectory (fast path).
    
    Args:
        env: The simulation environment.
        object_name: Name of the object to plan for.
        start_pose: Start pose (StampedPose, will be converted to world frame).
        target_pose: Target pose (StampedPose).
        tf_manager: TransformManager for frame conversion.
        num_steps: Number of output trajectory waypoints.
        env_lock: Optional threading lock.
        max_iterations: Max RRT iterations (default from constants).
        step_size: RRT step size (default from constants).
        goal_bias: RRT goal bias (default from constants).
        pos_bounds: Workspace bounds (2, 3) array (default from constants).
        smoothing_iterations: Path smoothing iterations (default from constants).
        
    Returns:
        Tuple of (trajectory as list of StampedPose, planning_success bool).
        If planning fails, returns the simple interpolated trajectory as fallback
        with success=False.
    """
    if num_steps is None:
        num_steps = constants.OBJECT_TRAJ_NUM_STEPS
    
    # Normalize to world frame
    if tf_manager is not None:
        if start_pose.frame_id != constants.WORLD_FRAME:
            start_pose = tf_manager.transform_pose(start_pose, constants.WORLD_FRAME)
        if target_pose.frame_id != constants.WORLD_FRAME:
            target_pose = tf_manager.transform_pose(target_pose, constants.WORLD_FRAME)
    
    frame_id = start_pose.frame_id
    
    start_pos = np.asarray(start_pose.position, dtype=np.float64)
    start_quat = np.asarray(start_pose.quaternion, dtype=np.float64)
    goal_pos = np.asarray(target_pose.position, dtype=np.float64)
    goal_quat = np.asarray(target_pose.quaternion, dtype=np.float64)
    
    # Auto-compute position bounds from start/goal with margin
    if pos_bounds is None:
        all_pos = np.stack([start_pos, goal_pos])
        margin = np.array(constants.RRT_BOUNDS_MARGIN)
        auto_lo = all_pos.min(axis=0) - margin
        auto_hi = all_pos.max(axis=0) + margin
        # Enforce minimum z (don't plan below the table)
        auto_lo[2] = max(auto_lo[2], constants.RRT_MIN_Z)
        pos_bounds = np.array([auto_lo, auto_hi])
    
    logger.info(f"Planning object trajectory: {start_pos} -> {goal_pos}")
    logger.info(f"  Workspace bounds: {pos_bounds[0]} to {pos_bounds[1]}")
    
    # Run RRT-Connect
    result = rrt_connect_se3(
        env=env,
        object_name=object_name,
        start_pos=start_pos,
        start_quat=start_quat,
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        env_lock=env_lock,
        max_iterations=max_iterations,
        step_size=step_size,
        goal_bias=goal_bias,
        pos_bounds=pos_bounds,
    )
    
    if result.success:
        logger.info(f"RRT planning succeeded ({result.iterations} iterations, "
                    f"raw path length: {len(result.path)} waypoints)")
        
        # Smooth the path
        smoothed = smooth_path(
            env, object_name, result.path,
            env_lock=env_lock,
            max_smoothing_iterations=smoothing_iterations,
        )
        
        # Densify to the desired number of waypoints
        dense = densify_path(smoothed, num_steps + 1)
        
        # Convert to StampedPose
        trajectory = [
            StampedPose(position=pos, quaternion=quat, frame_id=frame_id)
            for pos, quat in dense
        ]
        
        return trajectory, True
    else:
        logger.warning("RRT planning failed — falling back to direct interpolation.")
        # Fallback: simple SE(3) interpolation
        trajectory = []
        for i in range(num_steps + 1):
            t = i / float(num_steps)
            pos, quat = interpolate_pose(start_pos, start_quat, goal_pos, goal_quat, t)
            trajectory.append(StampedPose(position=pos, quaternion=quat, frame_id=frame_id))
        
        return trajectory, False
