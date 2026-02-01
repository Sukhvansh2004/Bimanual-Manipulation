"""Collision checking utilities."""

from contextlib import nullcontext
import threading
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils import mujoco_utils
import logging

def in_collision(env: DualArmEnvironment, *, contact_dist_threshold: float = 1e-6, env_lock: threading.RLock =None, ignore_names: list = None) -> bool:
    """Return True if either robot has any contacts below the threshold, excluding ignored names."""

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        ncon = int(data.ncon)
        if ncon <= 0:
            return False

        if ignore_names is None:
            ignore_names = []

        for i in range(ncon):
            c = data.contact[i]
            if float(c.dist) > float(contact_dist_threshold):
                continue
            g1 = model.geom(int(c.geom1)).name or ""
            g2 = model.geom(int(c.geom2)).name or ""
            
            # Check if either geom is in ignore_names
            if any(name in g1 for name in ignore_names) or any(name in g2 for name in ignore_names):
                continue

            r1 = mujoco_utils.geom_belongs_to_robot(g1)
            r2 = mujoco_utils.geom_belongs_to_robot(g2)
            # We only care about robot-involving contacts
            if r1 is None and r2 is None:
                continue
            
            logger = logging.getLogger(__name__)
            logger.debug(f"Collision between {g1} and {g2}")
            return True
        return False
