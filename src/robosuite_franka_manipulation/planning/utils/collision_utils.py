"""Collision checking utilities."""

from contextlib import nullcontext
import threading
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation import constants
import logging
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global caches for geom info
_robot_mask = None      # Boolean array of size ngeom
_geom_names = None      # List of geom names indexed by ID
_prefix_to_ids = {}     # Cache for name prefix -> list of geom IDs
_cached_model_id = None # id() of the model object used to build the caches

def _ensure_geom_caches(model):
    """Ensure global caches for geom names and robot memberships are initialized."""
    global _robot_mask, _geom_names, _prefix_to_ids, _cached_model_id
    if _robot_mask is None or _cached_model_id != id(model):
        _geom_names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        _robot_mask = np.zeros(model.ngeom, dtype=bool)
        for i, name in enumerate(_geom_names):
            if any(name.startswith(p) for p in ["robot0_", "gripper0_", "robot1_", "gripper1_"]):
                _robot_mask[i] = True
        _prefix_to_ids = {}  # Reset prefix cache
        _cached_model_id = id(model)

def _get_ids_for_prefixes(prefixes):
    """Get union of geom IDs that start with any of the given prefixes."""
    ids = set()
    for pref in prefixes:
        if pref not in _prefix_to_ids:
            _prefix_to_ids[pref] = [i for i, name in enumerate(_geom_names) if name.startswith(pref)]
        ids.update(_prefix_to_ids[pref])
    return list(ids)


# ── Public accessors for cached geom metadata ──────────────────────────────
# These allow other modules (e.g. object_trajectory_planner) to reuse the
# geom caches without duplicating the iteration/prefix-matching logic.

def get_geom_ids_by_prefix(model, prefix: str) -> np.ndarray:
    """Return an array of geom IDs whose names start with *prefix*.

    Initialises the internal geom caches on first call for a given model.
    """
    _ensure_geom_caches(model)
    return np.array(_get_ids_for_prefixes([prefix]), dtype=int)


def get_robot_geom_mask(model) -> np.ndarray:
    """Return a boolean mask of shape ``(ngeom,)`` that is True for robot/gripper geoms.

    Initialises the internal geom caches on first call for a given model.
    """
    _ensure_geom_caches(model)
    return _robot_mask


def in_collision(
    env: DualArmEnvironment,
    *,
    contact_dist_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
    env_lock: threading.RLock = None,
    ignore_names: Optional[List[str]] = None,
    ignore_pairs: Optional[List[Tuple[str, str]]] = None,
) -> bool:
    """Return True if either robot has any contacts below the threshold.

    Args:
        env: The simulation environment.
        contact_dist_threshold: Distance threshold for considering a contact.
        env_lock: Optional lock for thread safety.
        ignore_names: List of geom name prefixes to ignore entirely
            (contacts where *either* geom matches are discarded).
        ignore_pairs: List of ``(prefix_a, prefix_b)`` tuples.  A contact is
            discarded when one geom matches *prefix_a* AND the other matches
            *prefix_b*.  This allows selectively ignoring object-gripper
            contacts while still detecting object-arm contacts.
    """

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        ncon = int(data.ncon)
        if ncon <= 0:
            return False
            
        _ensure_geom_caches(model)

        # Vectorized check using numpy buffers
        g1 = np.asarray(data.contact.geom1[:ncon])
        g2 = np.asarray(data.contact.geom2[:ncon])
        dist = np.asarray(data.contact.dist[:ncon])

        # 1. Distance filter
        mask = dist <= contact_dist_threshold
        if not np.any(mask):
            return False

        # 2. Ignore names filter (blanket ignore)
        if ignore_names:
            ignore_ids = _get_ids_for_prefixes(ignore_names)
            if ignore_ids:
                ign_mask = np.zeros(model.ngeom, dtype=bool)
                ign_mask[ignore_ids] = True
                # A contact is valid ONLY if NEITHER geom is in the ignore set
                mask &= ~(ign_mask[g1] | ign_mask[g2])
                if not np.any(mask):
                    return False

        # 3. Ignore pairs filter (selective ignore between two groups)
        if ignore_pairs:
            for prefix_a, prefix_b in ignore_pairs:
                ids_a = _get_ids_for_prefixes([prefix_a])
                ids_b = _get_ids_for_prefixes([prefix_b])
                if ids_a and ids_b:
                    mask_a = np.zeros(model.ngeom, dtype=bool)
                    mask_b = np.zeros(model.ngeom, dtype=bool)
                    mask_a[ids_a] = True
                    mask_b[ids_b] = True
                    # Ignore contacts where (g1 in A and g2 in B) OR (g1 in B and g2 in A)
                    pair_mask = (mask_a[g1] & mask_b[g2]) | (mask_b[g1] & mask_a[g2])
                    mask &= ~pair_mask
                    if not np.any(mask):
                        return False

        # 4. Robot membership filter
        # Contact must involve at least one robot geom
        mask &= (_robot_mask[g1] | _robot_mask[g2])
        
        if np.any(mask):
            idx = np.where(mask)[0][0]
            logger.debug(f"Collision detected: {_geom_names[g1[idx]]} <-> {_geom_names[g2[idx]]} (dist={dist[idx]:.6e})")
            return True
            
        return False


def check_inter_arm_collision(
    env: DualArmEnvironment,
    *,
    contact_dist_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Return True if any robot0/gripper0 geom contacts a robot1/gripper1 geom.

    Unlike :func:`in_collision`, this checks **only** inter-arm contacts and
    ignores all robot-environment and robot-object contacts.  Intended for
    verifying that a dual-arm configuration is free of arm-arm interference.

    Note:
        Assumes ``mj_forward`` has already been called with the desired qpos.
        The caller is responsible for locking and state management.
    """
    model = env.sim.model
    data = env.sim.data
    ncon = int(data.ncon)
    if ncon <= 0:
        return False

    _ensure_geom_caches(model)

    g1 = np.asarray(data.contact.geom1[:ncon])
    g2 = np.asarray(data.contact.geom2[:ncon])
    dist = np.asarray(data.contact.dist[:ncon])

    mask = dist <= contact_dist_threshold
    if not np.any(mask):
        return False

    arm0_ids = _get_ids_for_prefixes(["robot0_", "gripper0_"])
    arm1_ids = _get_ids_for_prefixes(["robot1_", "gripper1_"])
    if not arm0_ids or not arm1_ids:
        return False

    arm0_mask = np.zeros(model.ngeom, dtype=bool)
    arm1_mask = np.zeros(model.ngeom, dtype=bool)
    arm0_mask[arm0_ids] = True
    arm1_mask[arm1_ids] = True

    inter_arm = mask & (
        (arm0_mask[g1] & arm1_mask[g2]) | (arm1_mask[g1] & arm0_mask[g2])
    )
    if np.any(inter_arm):
        idx = np.where(inter_arm)[0][0]
        logger.debug(
            "Inter-arm collision: %s <-> %s (dist=%.6e)",
            _geom_names[g1[idx]], _geom_names[g2[idx]], dist[idx],
        )
        return True
    return False
