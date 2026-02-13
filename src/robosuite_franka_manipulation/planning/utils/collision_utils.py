"""Collision checking utilities."""

from contextlib import nullcontext
import threading
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global caches for geom info
_robot_mask = None      # Boolean array of size ngeom
_geom_names = None      # List of geom names indexed by ID
_prefix_to_ids = {}     # Cache for name prefix -> list of geom IDs

def _ensure_geom_caches(model):
    """Ensure global caches for geom names and robot memberships are initialized."""
    global _robot_mask, _geom_names, _prefix_to_ids
    if _robot_mask is None or len(_robot_mask) != model.ngeom:
        _geom_names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        _robot_mask = np.zeros(model.ngeom, dtype=bool)
        for i, name in enumerate(_geom_names):
            if any(name.startswith(p) for p in ["robot0_", "gripper0_", "robot1_", "gripper1_"]):
                _robot_mask[i] = True
        _prefix_to_ids = {} # Reset prefix cache

def _get_ids_for_prefixes(prefixes):
    """Get union of geom IDs that start with any of the given prefixes."""
    ids = set()
    for pref in prefixes:
        if pref not in _prefix_to_ids:
            _prefix_to_ids[pref] = [i for i, name in enumerate(_geom_names) if name.startswith(pref)]
        ids.update(_prefix_to_ids[pref])
    return list(ids)

def in_collision(env: DualArmEnvironment, *, contact_dist_threshold: float = 1e-6, env_lock: threading.RLock = None, ignore_names: list = None) -> bool:
    """Return True if either robot has any contacts below the threshold, excluding ignored names."""

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

        # 2. Ignore names filter
        if ignore_names:
            ignore_ids = _get_ids_for_prefixes(ignore_names)
            if ignore_ids:
                ign_mask = np.zeros(model.ngeom, dtype=bool)
                ign_mask[ignore_ids] = True
                # A contact is valid ONLY if NEITHER geom is in the ignore set
                mask &= ~(ign_mask[g1] | ign_mask[g2])
                if not np.any(mask):
                    return False

        # 3. Robot membership filter
        # Contact must involve at least one robot geom
        mask &= (_robot_mask[g1] | _robot_mask[g2])
        
        if np.any(mask):
            idx = np.where(mask)[0][0]
            logger.info(f"Collision detected: {_geom_names[g1[idx]]} <-> {_geom_names[g2[idx]]} (dist={dist[idx]:.6e})")
            return True
            
        return False
