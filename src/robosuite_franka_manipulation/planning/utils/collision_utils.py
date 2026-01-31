"""Collision checking utilities."""

from contextlib import nullcontext
import threading
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils import mujoco_utils


def in_collision(env: DualArmEnvironment, *, contact_dist_threshold: float = 1e-6, env_lock: threading.RLock =None) -> bool:
    """Return True if either robot has any contacts below the threshold."""

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        ncon = int(data.ncon)
        if ncon <= 0:
            return False

        for i in range(ncon):
            c = data.contact[i]
            if float(c.dist) > float(contact_dist_threshold):
                continue
            g1 = model.geom(int(c.geom1)).name or ""
            g2 = model.geom(int(c.geom2)).name or ""
            r1 = mujoco_utils.geom_belongs_to_robot(g1)
            r2 = mujoco_utils.geom_belongs_to_robot(g2)
            # We only care about robot-involving contacts
            if r1 is None and r2 is None:
                continue
            return True
        return False
