"""Contact-driven suction gripper emulation via MuJoCo weld equality constraints.

Quaternions read from / written to MuJoCo are ``[w, x, y, z]``; scipy
computations use ``[x, y, z, w]``.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import mujoco
import robosuite.utils.transform_utils as T
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite.utils.binding_utils import MjSim

from bimanual_manipulation import constants
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class SuctionWeldHandles:
    """Resolved MuJoCo IDs for one suction gripper's weld constraint.

    Attributes:
        valid: Whether all IDs were resolved.
        eq_id: Weld equality constraint ID.
        snout_geom_id: Suction-cup (``geo_snout``) geom ID.
        eef_body_id: EEF body the weld's body1 is set to.
        welded_body_id: Body currently welded (body2), or ``None`` if released.
    """
    valid: bool
    eq_id: int = -1
    snout_geom_id: int = -1
    eef_body_id: int = -1  # The body that the snout is attached to (body1 of weld)
    welded_body_id: Optional[int] = None

    def __repr__(self):
        """Return a multi-line string listing all handle fields.

        Returns:
            String representation.
        """
        return (
            f"""SuctionWeldHandles(
                    valid={self.valid}, 
                    eq_id={self.eq_id},
                    snout_geom_id={self.snout_geom_id},
                    eef_body_id={self.eef_body_id},
                    welded_body_id={self.welded_body_id}
                )"""
        )

# Cache keyed by model id() so we only resolve names once per model
_SUCTION_CACHE: Dict[int, Dict[tuple, SuctionWeldHandles]] = {}

# Cache for valid object body IDs per environment
_OBJECT_BODY_CACHE: Dict[int, Set[int]] = {}


def clear_suction_cache() -> None:
    """Clear all suction weld caches. Call this after env.reset() if needed.

    Clears both the per-model handle cache (including ``welded_body_id``
    state) and the valid-object body cache.
    """
    _SUCTION_CACHE.clear()
    _OBJECT_BODY_CACHE.clear()
    logger.debug("Cleared suction weld caches")


def is_object_grasped(env: DualArmEnvironment, robot_id: int, arm: str = "right") -> bool:
    """Return True if the suction weld for *robot_id* / *arm* is currently active.

    Checks the cached ``SuctionWeldHandles.welded_body_id`` — if it is not
    ``None``, the gripper has an active MuJoCo weld equality constraint
    attaching it to an object.

    Args:
        env: The simulation environment.
        robot_id: Robot index.
        arm: Arm name.

    Returns:
        ``False`` if no valid handles are cached for this gripper.
    """
    model_key = id(env.sim.model)
    per_model = _SUCTION_CACHE.get(model_key, {})
    cache_key = (int(robot_id), str(arm))
    handles = per_model.get(cache_key)
    if handles is None or not handles.valid:
        return False
    return handles.welded_body_id is not None


def _name2id(sim: MjSim, obj_type: str, name: str) -> int:
    """Get a MuJoCo element ID from its name.

    Bodies and geoms use the robosuite wrapper's ``*_name2id`` methods;
    equality constraints use native ``mj_name2id``.

    Args:
        sim: Simulation.
        obj_type: ``"body"``, ``"geom"`` or ``"equality"``; any other type is
            treated like ``"equality"``.
        name: Element name.

    Returns:
        Element ID; ``-1`` if an equality constraint is not found.

    Raises:
        ValueError: If a body or geom name is not found (raised by the
            robosuite wrapper).
    """
    model = sim.model

    type_map = {
        "body": "body",
        "geom": "geom",
        "equality": None,  # No public method for equality
    }
    
    obj_name = type_map.get(obj_type)
    if obj_name is None:
        # For equality constraints, the wrapper has no public helper; use native model
        mj_model = model if isinstance(model, mujoco.MjModel) else model._model
        return int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name))
    
    # Use wrapper's public name2id methods
    if obj_type == "body":
        return model.body_name2id(name)
    if obj_type == "geom":
        return model.geom_name2id(name)
    return -1


def _id2name(sim: MjSim, obj_type: str, idx: int) -> Optional[str]:
    """Get a MuJoCo element name from its ID.

    Args:
        sim: Simulation.
        obj_type: ``"body"``, ``"geom"`` or ``"equality"``.
        idx: Element ID.

    Returns:
        Element name, or ``None`` for an unknown ``obj_type`` or unnamed
        equality.
    """
    model = sim.model
    
    if obj_type == "body":
        return model.body_id2name(idx)
    if obj_type == "geom":
        return model.geom_id2name(idx)
    if obj_type == "equality":
        mj_model = model if isinstance(model, mujoco.MjModel) else model._model
        return mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, int(idx))
    return None


def _get_contact_arrays(sim: MjSim) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized contact info for the active contacts.

    Args:
        sim: Simulation.

    Returns:
        ``(geom1, geom2, dist)`` arrays of length ``ncon`` (empty if none).
    """
    data = sim.data
    ncon = int(data.ncon)
    if ncon <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    
    g1 = np.asarray(data.contact.geom1[:ncon], dtype=np.int32)
    g2 = np.asarray(data.contact.geom2[:ncon], dtype=np.int32)
    dist = np.asarray(data.contact.dist[:ncon], dtype=np.float64)
    return g1, g2, dist


def _get_valid_object_body_ids(env: DualArmEnvironment, sim: MjSim) -> Set[int]:
    """Get the set of body IDs that belong to manipulable objects in the environment.
    
    Uses robosuite's object tracking (object_body_ids) to identify valid targets.
    Only these bodies can be welded to - everything else is rejected. Static
    objects (``env.static_object_body_ids``) are excluded. Cached per model.

    Args:
        env: The simulation environment.
        sim: Simulation (its model's ``id`` is the cache key).

    Returns:
        Set of valid body IDs.
    """
    model_key = id(sim.model)
    if model_key in _OBJECT_BODY_CACHE:
        return _OBJECT_BODY_CACHE[model_key]
    
    valid_ids: Set[int] = set()
    
    for name, body_id in env.object_body_ids.items():
        if isinstance(body_id, (int, np.integer)) and body_id > 0:
            # Skip static objects (cannot be suctioned)
            if hasattr(env, 'static_object_body_ids') and body_id in env.static_object_body_ids:
                continue
            valid_ids.add(int(body_id))
            logger.debug(f"Valid suction target: '{name}' -> body_id {body_id}")

    _OBJECT_BODY_CACHE[model_key] = valid_ids
    logger.debug(f"Found {len(valid_ids)} valid suction target bodies: {valid_ids}")
    return valid_ids


def _is_valid_suction_target(env: DualArmEnvironment, sim: MjSim, body_id: int) -> bool:
    """Check if a body is a valid suction target (i.e., a manipulable object).

    Args:
        env: The simulation environment.
        sim: Simulation.
        body_id: Body ID.

    Returns:
        ``True`` if ``body_id > 0`` and it is a non-static tracked object.
    """
    if body_id <= 0:
        return False
    valid_ids = _get_valid_object_body_ids(env, sim)
    return body_id in valid_ids


def _set_equality_active(sim: MjSim, eq_id: int, active: bool) -> None:
    """Set a weld constraint's ``data.eq_active`` flag.

    Args:
        sim: Simulation.
        eq_id: Equality constraint ID.
        active: Whether to enable the constraint.
    """
    data = sim.data
    data.eq_active[eq_id] = 1 if active else 0


def _set_weld_bodies(sim: MjSim, eq_id: int, body1_id: int, body2_id: int) -> bool:
    """Set weld's body1 (eef) and body2 (object) using wrapper's public API.
    
    After robosuite merges XML, body IDs change. We must set BOTH
    eq_obj1id and eq_obj2id to ensure the weld connects the correct bodies.

    Args:
        sim: Simulation.
        eq_id: Equality constraint ID.
        body1_id: EEF body ID.
        body2_id: Object body ID.

    Returns:
        ``False`` if the constraint is not a weld, else ``True``.
    """
    model = sim.model
    success = True
    
    # Verify this is a weld constraint (mjEQ_WELD = 1)
    eq_type = int(model.eq_type[eq_id])
    if eq_type != 1:  # mjEQ_WELD = 1
        logger.error(f"Constraint {eq_id} is not a WELD (type={eq_type}, expected 1)")
        return False
    
    # Set body1 (the gripper's eef body)
    old_obj1 = int(model.eq_obj1id[eq_id])
    model.eq_obj1id[eq_id] = int(body1_id)
    # Always log body1 status for debugging
    logger.debug(f"eq_obj1id[{eq_id}]: was {old_obj1}, now {body1_id} (eef)")
    
    # Set body2 (the object being suctioned)
    old_obj2 = int(model.eq_obj2id[eq_id])
    model.eq_obj2id[eq_id] = int(body2_id)
    # Always log body2 status for debugging
    logger.debug(f"eq_obj2id[{eq_id}]: was {old_obj2}, now {body2_id} (object)")
    
    return success


def _debug_weld_state(sim: MjSim, eq_id: int) -> None:
    """Log a weld constraint's bodies, active flag, relpose, and solver params at DEBUG.

    Args:
        sim: Simulation.
        eq_id: Equality constraint ID.
    """
    model = sim.model
    data = sim.data
    
    eq_type = int(model.eq_type[eq_id])
    type_name = "WELD" if eq_type == 1 else f"type_{eq_type}"
    
    obj1_id = int(model.eq_obj1id[eq_id])
    obj2_id = int(model.eq_obj2id[eq_id])
    
    obj1_name = _id2name(sim, "body", obj1_id) or f"body_{obj1_id}"
    obj2_name = _id2name(sim, "body", obj2_id) or f"body_{obj2_id}"
    
    eq_active = int(data.eq_active[eq_id])
    
    anchor = model.eq_data[eq_id, 0:3]
    pos = model.eq_data[eq_id, 3:6]
    quat = model.eq_data[eq_id, 6:10]
    
    solref = model.eq_solref[eq_id]
    solimp = model.eq_solimp[eq_id]
    
    logger.debug(
        f"WELD STATE eq_id={eq_id}: type={type_name}, "
        f"body1={obj1_name}({obj1_id}), body2={obj2_name}({obj2_id}), "
        f"active={eq_active}, anchor={anchor}, relpose_pos={pos}, relpose_quat={quat}"
    )
    logger.debug(f"  solref={solref}, solimp={solimp}")


def _verify_weld_constraint(sim: MjSim, eq_id: int) -> bool:
    """Verify a weld constraint is properly configured.

    Args:
        sim: Simulation.
        eq_id: Equality constraint ID.

    Returns:
        True if the constraint is an active weld between two distinct
        non-world bodies; False otherwise (with an error/warning logged).
    """
    model = sim.model
    data = sim.data
    
    eq_type = int(model.eq_type[eq_id])
    if eq_type != 1:  # mjEQ_WELD = 1
        logger.error(f"eq_id={eq_id} is not a WELD constraint (type={eq_type})")
        return False
    
    active = int(data.eq_active[eq_id])
    if active != 1:
        logger.warning(f"eq_id={eq_id} is not active (eq_active={active})")
        return False
    
    body1 = int(model.eq_obj1id[eq_id])
    body2 = int(model.eq_obj2id[eq_id])
    if body1 <= 0 or body2 <= 0:
        logger.error(f"eq_id={eq_id} has invalid body IDs: body1={body1}, body2={body2}")
        return False
    if body1 == body2:
        logger.error(f"eq_id={eq_id} has same body for both: {body1}")
        return False
    
    return True


def _get_body_pose_wxyz(sim: MjSim, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat) for a body in world coordinates.

    Quaternion is MuJoCo's convention (w, x, y, z).

    Args:
        sim: Simulation.
        body_id: Body ID.

    Returns:
        ``(pos, quat_wxyz)`` arrays of shape ``(3,)`` and ``(4,)``.
    """
    data = sim.data
    
    pos = np.asarray(data.body_xpos[body_id], dtype=np.float64)
    quat = np.asarray(data.body_xquat[body_id], dtype=np.float64)
    return pos, quat


def _compute_relative_pose_wxyz(
    p1: np.ndarray, q1_wxyz: np.ndarray,
    p2: np.ndarray, q2_wxyz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose of body2 in body1's frame using scipy Rotation.
    
    Args:
        p1: Position of body1 in world frame.
        q1_wxyz: Quaternion of body1 in MuJoCo (w,x,y,z) convention.
        p2: Position of body2 in world frame.
        q2_wxyz: Quaternion of body2 in MuJoCo (w,x,y,z) convention.

    Returns:
        (p_rel, q_rel_wxyz): Relative position and quaternion (MuJoCo convention).
    """
    # Convert to scipy convention (x,y,z,w)
    q1_xyzw = T.convert_quat(q1_wxyz, to="xyzw")
    q2_xyzw = T.convert_quat(q2_wxyz, to="xyzw")
    
    # Create scipy Rotation objects
    rot1 = R.from_quat(q1_xyzw)
    rot2 = R.from_quat(q2_xyzw)
    
    # Relative rotation: q_rel = q1^-1 * q2
    rot_rel = rot1.inv() * rot2
    
    # Relative position: p_rel = R1^-1 * (p2 - p1)
    p_rel = rot1.inv().apply(p2 - p1)
    
    # Convert back to MuJoCo convention (w,x,y,z)
    q_rel_wxyz = T.convert_quat(rot_rel.as_quat(), to="wxyz")
    
    return p_rel, q_rel_wxyz


def _set_weld_relpose_current(sim: MjSim, eq_id: int, body1_id: int, body2_id: int) -> bool:
    """Set weld relpose to current relative pose (body2 in body1 frame).

    This prevents objects from snapping when the weld activates. Writes
    ``model.eq_data[eq_id, 0:10]`` (anchor zeroed, relpose position and
    ``wxyz`` quaternion).

    Args:
        sim: Simulation.
        eq_id: Weld constraint ID.
        body1_id: EEF body ID.
        body2_id: Object body ID.

    Returns:
        Always ``True``.
    """
    model = sim.model
    
    p1, q1 = _get_body_pose_wxyz(sim, body1_id)
    p2, q2 = _get_body_pose_wxyz(sim, body2_id)

    # Calculate relative pose: body2 in body1's frame using scipy Rotation
    p_rel, q_rel = _compute_relative_pose_wxyz(p1, q1, p2, q2)
    
    logger.debug(f"Setting weld relpose: pos={p_rel}, quat={q_rel}")

    # MuJoCo eq_data layout for WELD constraints:
    #   [0:3]  = anchor point in body1 frame (we keep at origin)
    #   [3:6]  = relpose position (body2 origin in body1 frame)
    #   [6:10] = relpose quaternion (body2 orientation relative to body1, wxyz format)
    #   [10]   = torquescale (optional)
    # Keep anchor at origin (constraint acts at body1 origin)
    model.eq_data[eq_id, 0:3] = np.zeros(3)
    # Set relpose position (indices 3:6)
    model.eq_data[eq_id, 3:6] = p_rel
    # Set relpose quaternion (indices 6:10)
    model.eq_data[eq_id, 6:10] = q_rel
    logger.debug(f"Set eq_data[{eq_id}]: anchor=[0,0,0], relpose_pos={p_rel}, relpose_quat={q_rel}")
    return True


def _resolve_handles(env: DualArmEnvironment, robot_id: int, arm: str, gripper: Any) -> Optional[SuctionWeldHandles]:
    """Resolve suction weld handles using robosuite's correct_naming method.
    
    Uses the gripper's correct_naming() to get properly prefixed element names,
    which is the canonical way robosuite handles XML element naming after merge.
    Looks up ``suction_constraint`` (equality), ``geo_snout`` (geom) and
    ``eef`` (body). Results are cached per ``(model, robot_id, arm)``.

    Args:
        env: The simulation environment.
        robot_id: Robot index.
        arm: Arm name.
        gripper: robosuite gripper model providing ``correct_naming``.

    Returns:
        Cached or newly resolved handles (``valid=False`` if the equality
        constraint is missing).

    Raises:
        ValueError: If the snout geom or EEF body name is not found.
    """

    sim = env.sim
    model = sim.model
    model_key = id(model)
    per_model = _SUCTION_CACHE.setdefault(model_key, {})
    cache_key = (int(robot_id), str(arm))
    if cache_key in per_model:
        return per_model[cache_key]

    # Use robosuite's correct_naming method - this is the proper way
    correct_naming = gripper.correct_naming

    # Get properly prefixed names using robosuite's naming system
    eq_name = correct_naming("suction_constraint")
    geom_name = correct_naming("geo_snout")
    eef_body_name = correct_naming("eef")
    
    logger.debug(f"Looking for suction elements: eq='{eq_name}', geom='{geom_name}', eef='{eef_body_name}'")

    # Look up IDs using native MuJoCo API
    eq_id = _name2id(sim, "equality", eq_name)
    snout_geom_id = _name2id(sim, "geom", geom_name)
    eef_body_id = _name2id(sim, "body", eef_body_name)

    valid = (eq_id != -1 and snout_geom_id != -1 and eef_body_id != -1)
    handles = SuctionWeldHandles(
        valid=valid,
        eq_id=eq_id,
        snout_geom_id=snout_geom_id,
        eef_body_id=eef_body_id,
    )
    
    logger.debug(f"Suction Weld Handles for robot {robot_id} arm {arm}: {handles}")

    per_model[cache_key] = handles
    return handles


def apply_suction_weld(env: DualArmEnvironment, action: Any) -> None:
    """Contact-driven suction weld.

    For each suction/vacuum gripper, the gripper command is the last entry of
    its robot's action block:

    - ``0.0``: neutral, weld state unchanged.
    - exactly ``-1.0``: suction OFF, deactivate the weld.
    - any other value: suction ON. Keep an existing weld active; otherwise,
      if the snout's closest contact (``dist < constants.SUCTION_DISTANCE``)
      is with a VALID OBJECT (from env.object_body_ids), retarget the weld
      to that body at the current relative pose and activate it. With no
      valid contact the weld is kept inactive (prevents self-welding).

    Calls ``sim.forward()`` if a weld was activated or released. All
    exceptions are logged and swallowed.

    Args:
        env: The simulation environment.
        action: Full environment action vector (size ``env.action_dim``).
    """
    try:
        robots = env.robots
        sim = env.sim
        model = sim.model

        total_action_dim = env.action_dim

        # Calculate action dimensions per robot (use actual robot dims)
        per_robot_dims = [r.action_dim for r in robots]
        # Fallback: even split
        if sum(per_robot_dims) != total_action_dim:
            if total_action_dim % len(robots) == 0:
                per_robot_dims = [total_action_dim // len(robots)] * len(robots)
            else:
                logger.warning("Cannot determine per-robot action dims; skipping suction weld.")
                return
        if any(d <= 0 for d in per_robot_dims):
            return

        action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
        
        if action_vec.size < total_action_dim:
            logger.warning(f"Action vector size {action_vec.size} is less than expected {total_action_dim}")
            return

        # Fetch contact arrays once per update
        g1_arr, g2_arr, dist_arr = _get_contact_arrays(sim)
        any_mutation = False

        for ridx, robot in enumerate(robots):
            arms = robot.arms
            grippers = robot.gripper

            for arm in list(arms):
                gripper = grippers[arm]

                # Check if this is a suction gripper
                gripper_type = type(gripper).__name__.lower()
                is_suction = ("suction" in gripper_type) or ("vacuum" in gripper_type)
                if not is_suction:
                    continue

                handles = _resolve_handles(env, ridx, arm, gripper)
                if not handles or not handles.valid:
                    logger.warning(f"Invalid suction weld handles for robot {ridx} arm {arm}")
                    continue

                # Get gripper action value
                this_robot_dim = per_robot_dims[ridx]
                robot_action_offset = sum(per_robot_dims[:ridx])
                gripper_action_index = robot_action_offset + (this_robot_dim - 1)
                if gripper_action_index < 0 or gripper_action_index >= action_vec.size:
                    continue
                gripper_action = float(action_vec[gripper_action_index])
                
                if (gripper_action == 0.0):
                    # Neutral action - do nothing
                    continue

                # --- SUCTION OFF: Deactivate weld ---
                if (gripper_action == -1.0):
                    if handles.welded_body_id is not None:
                        logger.info(f"Releasing suction weld on body {handles.welded_body_id}")
                        _set_equality_active(sim, handles.eq_id, False)
                        handles.welded_body_id = None
                        any_mutation = True
                    else:
                        # Make sure weld stays inactive
                        _set_equality_active(sim, handles.eq_id, False)
                    continue

                # --- SUCTION ON ---
                
                # If already welded, just keep the weld active
                if handles.welded_body_id is not None:
                    _set_equality_active(sim, handles.eq_id, True)
                    continue

                # Find the best contacted geom (closest contact with snout)
                target_geom_id = None
                if g1_arr.size > 0:
                    # Snout can be either geom1 or geom2
                    mask = (g1_arr == handles.snout_geom_id) | (g2_arr == handles.snout_geom_id)
                    mask &= (dist_arr < constants.SUCTION_DISTANCE)
                    
                    if np.any(mask):
                        # Get indices of contacts involving the snout within threshold
                        candidate_indices = np.where(mask)[0]
                        # Get the one with the minimum distance
                        best_idx = candidate_indices[np.argmin(dist_arr[candidate_indices])]
                        
                        # Identify the other geom in the contact
                        if g1_arr[best_idx] == handles.snout_geom_id:
                            target_geom_id = int(g2_arr[best_idx])
                        else:
                            target_geom_id = int(g1_arr[best_idx])

                if target_geom_id is None:
                    # No contact - keep weld inactive
                    _set_equality_active(sim, handles.eq_id, False)
                    continue

                # Get the body that owns this geom
                body2_id = int(model.geom_bodyid[target_geom_id])
                
                if not _is_valid_suction_target(env, sim, body2_id):
                    target_name = _id2name(sim, "body", body2_id) or f"body_{body2_id}"
                    logger.debug(f"Ignoring contact with non-object body: {target_name}")
                    _set_equality_active(sim, handles.eq_id, False)
                    continue

                # body1 is the eef (gripper), body2 is the object being suctioned
                body1_id = handles.eef_body_id
                
                if not _set_weld_bodies(sim, handles.eq_id, body1_id, body2_id):
                    logger.warning(f"Failed to set weld bodies: body1={body1_id}, body2={body2_id}")
                    _set_equality_active(sim, handles.eq_id, False)
                    continue

                # Set weld relpose to current relative pose to prevent snapping
                if body1_id != -1:
                    if not _set_weld_relpose_current(sim, handles.eq_id, body1_id, body2_id):
                        logger.warning("Failed to set weld relpose - weld may snap objects together")

                # Activate the weld
                target_name = _id2name(sim, "body", body2_id) or f"body_{body2_id}"
                body1_name = _id2name(sim, "body", body1_id) or f"body_{body1_id}"
                logger.info(f"Activating suction weld: {body1_name}({body1_id}) -> {target_name}({body2_id})")
                handles.welded_body_id = body2_id
                _set_equality_active(sim, handles.eq_id, True)
                
                # Debug: log full weld state and verify constraint
                _debug_weld_state(sim, handles.eq_id)
                if not _verify_weld_constraint(sim, handles.eq_id):
                    logger.error("Weld constraint verification FAILED!")
                
                any_mutation = True

        # Propagate constraint changes
        if any_mutation:
            sim.forward()

    except Exception as e:
        logger.exception("apply_suction_weld error (ignored): %s", e)