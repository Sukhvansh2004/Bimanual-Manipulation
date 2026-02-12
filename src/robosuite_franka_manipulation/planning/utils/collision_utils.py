"""Collision checking utilities."""

from contextlib import nullcontext
import threading
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils import mujoco_utils
import logging

logger = logging.getLogger(__name__)

# Global caches for geom info
_robot_geom_ids = None  # Set of geom IDs belonging to any robot
_geom_names = None      # List of geom names indexed by ID

def _ensure_geom_caches(model):
    """Ensure global caches for geom names and robot memberships are initialized."""
    global _robot_geom_ids, _geom_names
    if _robot_geom_ids is None or _geom_names is None:
        _geom_names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        _robot_geom_ids = {i for i, name in enumerate(_geom_names) 
                           if name.startswith("robot0_") or name.startswith("gripper0_") or 
                              name.startswith("robot1_") or name.startswith("gripper1_")}

def in_collision(env: DualArmEnvironment, *, contact_dist_threshold: float = 1e-6, env_lock: threading.RLock = None, ignore_names: list = None) -> bool:
    """Return True if either robot has any contacts below the threshold, excluding ignored names."""

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        _ensure_geom_caches(model)

        ncon = int(data.ncon)
        if ncon <= 0:
            return False

        # Pre-resolve ignore IDs to avoid string checks in loop
        ignore_ids = set()
        if ignore_names:
            for i, name in enumerate(_geom_names):
                if any(name.startswith(ign) for ign in ignore_names):
                    ignore_ids.add(i)

        for i in range(ncon):
            c = data.contact[i]
            if float(c.dist) > float(contact_dist_threshold):
                continue
            
            g1_id = int(c.geom1)
            g2_id = int(c.geom2)
            
            # Check if either geom is in ignore_ids
            if g1_id in ignore_ids or g2_id in ignore_ids:
                continue

            # We only care about robot-involving contacts
            if g1_id not in _robot_geom_ids and g2_id not in _robot_geom_ids:
                continue
            
            logger.debug(f"Collision between {_geom_names[g1_id]} and {_geom_names[g2_id]}")
            return True
        return False
