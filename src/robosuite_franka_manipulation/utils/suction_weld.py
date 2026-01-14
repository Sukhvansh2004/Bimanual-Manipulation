import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SuctionWeldHandles:
    valid: bool
    eq_id: int = -1
    snout_geom_id: int = -1
    welded_body_id: Optional[int] = None


# Cache keyed by model id() so we only resolve names once per model
_SUCTION_CACHE: Dict[int, Dict[int, SuctionWeldHandles]] = {}


def _name2id(sim: Any, obj_type: str, name: str) -> int:
    """Best-effort name->id across mujoco_py and native mujoco backends."""
    model = sim.model

    attr = f"{obj_type}_name2id"
    if hasattr(model, attr):
        try:
            return int(getattr(model, attr)(name))
        except Exception:
            return -1

    try:
        import mujoco  # type: ignore

        type_map = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
        }
        mjt = type_map.get(obj_type)
        if mjt is None:
            return -1
        return int(mujoco.mj_name2id(model, mjt, name))
    except Exception:
        return -1


def _id2name(sim: Any, obj_type: str, idx: int) -> Optional[str]:
    model = sim.model

    attr = f"{obj_type}_id2name"
    if hasattr(model, attr):
        try:
            return getattr(model, attr)(idx)
        except Exception:
            return None

    try:
        import mujoco  # type: ignore

        type_map = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
        }
        mjt = type_map.get(obj_type)
        if mjt is None:
            return None
        return mujoco.mj_id2name(model, mjt, int(idx))
    except Exception:
        return None


def _get_contact_pairs(sim: Any) -> List[Tuple[int, int, float]]:
    """Return list of (geom1, geom2, dist) for all contacts."""
    data = sim.data
    ncon = int(getattr(data, "ncon", 0))
    pairs: List[Tuple[int, int, float]] = []
    for i in range(ncon):
        c = data.contact[i]
        g1 = int(getattr(c, "geom1"))
        g2 = int(getattr(c, "geom2"))
        dist = float(getattr(c, "dist", 0.0))
        pairs.append((g1, g2, dist))
    return pairs


def _is_probably_robot_or_fixture_body(sim: Any, body_id: int) -> bool:
    name = (_id2name(sim, "body", body_id) or "").lower()
    if name == "":
        return True
    if "robot" in name or "gripper" in name or "panda" in name or "cobot_pump" in name:
        return True
    if "table" in name or "arena" in name or "world" in name or "floor" in name:
        return True
    return False


def _body_is_dynamic(sim: Any, body_id: int) -> bool:
    model = sim.model
    try:
        if hasattr(model, "body_jntnum"):
            return int(model.body_jntnum[body_id]) > 0
    except Exception:
        pass
    return True


def _set_equality_active(sim: Any, eq_id: int, active: bool) -> None:
    model = sim.model
    if not hasattr(model, "eq_active"):
        return
    model.eq_active[eq_id] = 1 if active else 0


def _set_weld_target_body(sim: Any, eq_id: int, body2_id: int) -> bool:
    """Set weld's second object (object being suctioned)."""
    model = sim.model
    if hasattr(model, "eq_obj2id"):
        try:
            model.eq_obj2id[eq_id] = int(body2_id)
            return True
        except Exception:
            return False
    return False


def _resolve_handles(env: Any, robot_id: int, gripper: Any) -> Optional[SuctionWeldHandles]:
    if not hasattr(env, "sim"):
        return None

    sim = env.sim
    model_key = id(sim.model)
    per_model = _SUCTION_CACHE.setdefault(model_key, {})
    if robot_id in per_model:
        return per_model[robot_id]

    # Prefer robosuite's own naming prefix if available; this makes us agnostic to
    # robot index and any XML prefixing rules.
    naming_prefix = getattr(gripper, "naming_prefix", None)
    prefix = str(naming_prefix) if naming_prefix else ""

    # Robosuite may already include arm in the prefix (e.g., gripper0_right_), so
    # only append the raw element names defined in the XML.
    eq_name_candidates = []
    geom_name_candidates = []
    if prefix:
        eq_name_candidates.extend([f"{prefix}suction_constraint"])
        geom_name_candidates.extend([f"{prefix}geo_snout"])

    # Backwards-compatible heuristics
    eq_name_candidates.extend([
        f"gripper{robot_id}_suction_constraint",
        f"gripper{robot_id}_right_suction_constraint",
        "suction_constraint",
    ])
    geom_name_candidates.extend([
        f"gripper{robot_id}_geo_snout",
        f"gripper{robot_id}_right_geo_snout",
        "geo_snout",
    ])

    eq_id = -1
    for n in eq_name_candidates:
        eq_id = _name2id(sim, "equality", n)
        if eq_id != -1:
            break

    snout_geom_id = -1
    for n in geom_name_candidates:
        snout_geom_id = _name2id(sim, "geom", n)
        if snout_geom_id != -1:
            break

    # Fallback: substring scan (works regardless of prefix conventions)
    if eq_id == -1:
        try:
            neq = int(getattr(sim.model, "neq", 0))
            for i in range(neq):
                nm = (_id2name(sim, "equality", i) or "")
                if "suction_constraint" in nm and f"gripper{robot_id}" in nm:
                    eq_id = i
                    break
                if eq_id == -1 and "suction_constraint" in nm:
                    eq_id = i
        except Exception:
            pass

    if snout_geom_id == -1:
        try:
            ngeom = int(getattr(sim.model, "ngeom", 0))
            for i in range(ngeom):
                nm = (_id2name(sim, "geom", i) or "")
                if "geo_snout" in nm and f"gripper{robot_id}" in nm:
                    snout_geom_id = i
                    break
                if snout_geom_id == -1 and "geo_snout" in nm:
                    snout_geom_id = i
        except Exception:
            pass

    handles = SuctionWeldHandles(valid=(eq_id != -1 and snout_geom_id != -1), eq_id=eq_id, snout_geom_id=snout_geom_id)
    per_model[robot_id] = handles
    return handles


def apply_suction_weld(env: Any, action: Any) -> None:
    """Contact-driven suction weld.

    - If suction is ON (gripper action > 0) and suction cup is in contact with a dynamic object,
      activate the MuJoCo weld equality and set its target body to that object.
    - If suction is OFF, deactivate the weld.

    This does *not* modify objects; it only uses contact queries + equality toggling.
    """
    try:
        if not hasattr(env, "sim") or not hasattr(env, "robots"):
            return

        robots = getattr(env, "robots")
        if not robots:
            return

        sim = env.sim
        model = sim.model

        total_action_dim = int(getattr(env, "action_dim", 0))
        if total_action_dim <= 0:
            return
        if len(robots) <= 0:
            return

        # Prefer even split; otherwise fall back to the common robosuite layout (7 DoF per robot)
        if total_action_dim % len(robots) == 0:
            per_robot_dim = total_action_dim // len(robots)
        else:
            per_robot_dim = 7
        if per_robot_dim <= 0:
            return

        try:
            action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
        except Exception:
            return
        if action_vec.size < total_action_dim:
            return

        contact_pairs = _get_contact_pairs(sim)
        any_mutation = False

        for ridx, robot in enumerate(robots):
            gripper = getattr(robot, "gripper", None)
            if gripper is None:
                continue

            gripper_type = type(gripper).__name__.lower()
            is_suction = ("suction" in gripper_type) or ("vacuum" in gripper_type)
            if not is_suction:
                continue

            handles = _resolve_handles(env, ridx, gripper)
            if not handles or not handles.valid:
                continue

            # Robosuite convention: last dim per robot is gripper
            gripper_action_index = ridx * per_robot_dim + (per_robot_dim - 1)
            if gripper_action_index < 0 or gripper_action_index >= action_vec.size:
                continue
            gripper_action = float(action_vec[gripper_action_index])
            suction_on = gripper_action > 0.0

            if not suction_on:
                if handles.welded_body_id is not None:
                    _set_equality_active(sim, handles.eq_id, False)
                    handles.welded_body_id = None
                    any_mutation = True
                else:
                    _set_equality_active(sim, handles.eq_id, False)
                continue

            # suction ON: keep existing weld
            if handles.welded_body_id is not None:
                _set_equality_active(sim, handles.eq_id, True)
                continue

            # select contacted geom other than the snout
            target_geom_id: Optional[int] = None
            best_dist = float("inf")
            for g1, g2, dist in contact_pairs:
                if g1 == handles.snout_geom_id:
                    candidate = g2
                elif g2 == handles.snout_geom_id:
                    candidate = g1
                else:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    target_geom_id = int(candidate)

            if target_geom_id is None:
                _set_equality_active(sim, handles.eq_id, False)
                continue

            body2_id = int(model.geom_bodyid[target_geom_id])
            if body2_id <= 0:
                _set_equality_active(sim, handles.eq_id, False)
                continue
            if _is_probably_robot_or_fixture_body(sim, body2_id):
                _set_equality_active(sim, handles.eq_id, False)
                continue
            if not _body_is_dynamic(sim, body2_id):
                _set_equality_active(sim, handles.eq_id, False)
                continue

            if not _set_weld_target_body(sim, handles.eq_id, body2_id):
                _set_equality_active(sim, handles.eq_id, False)
                continue

            handles.welded_body_id = body2_id
            _set_equality_active(sim, handles.eq_id, True)
            any_mutation = True

        if any_mutation and hasattr(sim, "forward"):
            try:
                sim.forward()
            except Exception:
                pass

    except Exception as e:
        logger.debug("apply_suction_weld error (ignored): %s", e)
