#!/usr/bin/env python3
"""
Experiment configuration loader.

Loads experiment setups from JSON files and provides structured access
to environment, task, planning, and execution parameters.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.utils.object_factory import create_ycb_object, create_primitive_object
from bimanual_manipulation.transforms.stamped_pose import StampedPose

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PoseConfig:
    """A 6-DOF pose (position + quaternion [x, y, z, w])."""
    position: np.ndarray
    quaternion: np.ndarray

    def to_stamped_pose(self, frame_id: str = constants.WORLD_FRAME) -> StampedPose:
        """Convert to a StampedPose.

        Args:
            frame_id: Frame of the pose.

        Returns:
            StampedPose with this position and quaternion.
        """
        return StampedPose(position=self.position, quaternion=self.quaternion, frame_id=frame_id)


@dataclass
class TaskConfig:
    """What to manipulate and where."""
    manipulated_object: str
    target_pose: PoseConfig
    start_pose: Optional[PoseConfig] = None


@dataclass
class PlanningConfig:
    """Grasp generation & trajectory planning knobs."""
    num_grasps: int = 200
    num_entropy_grasps: int = 20
    use_dpp: bool = True  # Use DPP entropy maximization for grasp selection; False → top-N by score
    gripper_types: List[str] = field(default_factory=lambda: ["finger", "suction"])
    collision_check: str = "Inter"
    num_steps: int = constants.OBJECT_TRAJ_NUM_STEPS
    lift_height: float = constants.OBJECT_TRAJ_LIFT_HEIGHT
    approach_offset: float = constants.OBJECT_TRAJ_APPROACH_OFFSET
    rrt_bounds: Optional[np.ndarray] = None  # (2, 3) array [[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]
    collision_ignore_waypoints: Optional[List[int]] = None  # Waypoint indices to skip collision checking
    collision_ignore_waypoint_names: Optional[List[str]] = None  # Geom prefixes to ignore at those waypoints
    collision_ignore_waypoint_pairs: Optional[List[Tuple[str, str]]] = None  # Geom prefix pairs to ignore at those waypoints


@dataclass
class CollisionFilterConfig:
    """Patterns for collision-filter optimisation in worker envs."""
    relevant_patterns: List[str] = field(default_factory=list)


@dataclass
class VisualizationConfig:
    """Meshcat / mesh visualization settings (used by both planning & execution)."""
    enable_meshcat: bool = True
    visualize_mesh: bool = False
    mesh_step_interval: int = constants.VIS_MESH_STEP_INTERVAL
    mesh_opacity: float = constants.VIS_MESH_OPACITY
    mesh_sample_ratio: float = 0.5
    zmq_url: str = constants.MESHCAT_ZMQ_URL


@dataclass
class OutputConfig:
    """Where and whether to save trajectory results."""
    save_trajectories: bool = False
    output_dir: Optional[str] = None  # None → data/<experiment_name>


@dataclass
class ExecutionConfig:
    """Execution parameters for the dual-arm handover script."""
    visualize: bool = True
    meshcat_tf: bool = True
    smoothen: bool = True
    assignment_path: Optional[str] = None  # path to assignment_results.npz; None → auto-discover in data dir


@dataclass
class RuntimeConfig:
    """Runtime / process-level settings."""
    debug: bool = False
    num_workers: int = 0  # 0 → auto (cpu_count // 2)


@dataclass
class AssignmentConfig:
    """Segment assignment planner settings."""
    alpha: float = constants.ASSIGNMENT_ALPHA
    beta: float = constants.ASSIGNMENT_BETA
    epsilon: float = constants.ASSIGNMENT_EPSILON
    initial_robot: Optional[int] = None
    transfer_feasibility: Optional[Dict[str, bool]] = None  # JSON keys are strings; decoded at use-time
    beam_width: Optional[int] = None  # DP beam width; None = unlimited (exact)
    handover_stride: int = 1  # only attempt handovers every K segments
    handover_clearance_radius: Optional[float] = None  # metres; None = disabled
    num_trajectories_robot0: Optional[int] = None  # max robot-0 trajectories; None = all
    num_trajectories_robot1: Optional[int] = None  # max robot-1 trajectories; None = all


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str
    description: str
    ycb_objects: List[Dict[str, Any]]
    custom_objects: List[Dict[str, Any]]
    task: TaskConfig
    planning: PlanningConfig
    collision_filter: CollisionFilterConfig
    visualization: VisualizationConfig
    output: OutputConfig
    execution: ExecutionConfig
    runtime: RuntimeConfig
    assignment: AssignmentConfig
    raw: Dict[str, Any]  # original JSON dict for pass-through
    table_full_size: Optional[Tuple[float, float, float]] = None  # (length_x, width_y, thickness_z)
    table_offset: Optional[Tuple[float, float, float]] = None  # (x, y, z) world offset of table top reference
    robot_base_configs: Optional[List[Dict[str, np.ndarray]]] = None  # per-robot base position/quaternion overrides
    gripper_grasp_offsets: Optional[Dict[str, Dict[str, Any]]] = None  # per-gripper offset overrides


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_pose(data: Optional[Dict]) -> Optional[PoseConfig]:
    """Parse a pose dict ``{"position": [...], "quaternion": [...]}``.

    Args:
        data: Pose dict, or ``None``.

    Returns:
        Parsed PoseConfig, or ``None`` if ``data`` is ``None``.

    Raises:
        KeyError: If ``position`` or ``quaternion`` is missing.
    """
    if data is None:
        return None
    return PoseConfig(
        position=np.array(data["position"], dtype=float),
        quaternion=np.array(data["quaternion"], dtype=float),
    )


def _build_ycb_objects(items: List[Dict]) -> List[Dict[str, Any]]:
    """Convert JSON ycb_objects list to env-ready dicts via ``create_ycb_object``.

    Args:
        items: JSON entries with ``ycb_name`` and optional ``instance_name``,
            ``position``, ``quaternion``, ``is_static``.

    Returns:
        Env-ready YCB object spec dicts.

    Raises:
        KeyError: If an entry lacks ``ycb_name``.
    """
    result = []
    for item in items:
        obj = create_ycb_object(
            ycb_name=item["ycb_name"],
            instance_name=item.get("instance_name"),
            position=item.get("position"),
            quaternion=item.get("quaternion"),
            is_static=item.get("is_static", False),
        )
        result.append(obj)
    return result


def _build_custom_objects(items: List[Dict]) -> List[Dict[str, Any]]:
    """Convert JSON custom_objects list to env-ready dicts via ``create_primitive_object``.

    Args:
        items: JSON entries with ``type``, ``name``, ``size`` and optional
            ``position``, ``quaternion``, ``rgba``, ``density``, ``friction``,
            ``is_static``.

    Returns:
        Env-ready primitive object spec dicts (``is_static`` forwarded).

    Raises:
        KeyError: If an entry lacks ``type``, ``name`` or ``size``.
    """
    result = []
    for item in items:
        obj = create_primitive_object(
            obj_type=item["type"],
            name=item["name"],
            size=item["size"],
            position=item.get("position"),
            quat=item.get("quaternion"),
            rgba=item.get("rgba"),
            density=item.get("density", 500.0),
            friction=item.get("friction"),
        )
        # Forward extra keys (e.g. is_static) so callers can use them
        for extra_key in ("is_static",):
            if extra_key in item:
                obj[extra_key] = item[extra_key]
        result.append(obj)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_experiment(path: str) -> ExperimentConfig:
    """Load an experiment configuration from a JSON file.

    Side effect: entries in ``gripper_grasp_offsets`` overwrite
    ``constants.GRIPPER_GRASP_OFFSETS`` globally. An ``rrt_bounds`` not of
    shape ``(2, 3)`` is ignored with a warning.

    Args:
        path: Path to the JSON experiment file (absolute or relative to project root).

    Returns:
        Fully parsed ``ExperimentConfig``.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: On missing required fields (e.g. ``task``).
        ValueError: On invalid field values.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(constants.get_project_root()) / resolved

    if not resolved.exists():
        raise FileNotFoundError(f"Experiment file not found: {resolved}")

    with open(resolved, "r") as f:
        raw: Dict[str, Any] = json.load(f)

    logger.info("Loading experiment '%s' from %s", raw.get("name", "?"), resolved)

    # --- environment ---
    env_cfg = raw.get("environment", {})
    ycb_objects = _build_ycb_objects(env_cfg.get("ycb_objects", []))
    custom_objects = _build_custom_objects(env_cfg.get("custom_objects", []))
    table_full_size = None
    if "table_full_size" in env_cfg:
        tfs = env_cfg["table_full_size"]
        table_full_size = (float(tfs[0]), float(tfs[1]), float(tfs[2]))
    table_offset = None
    if "table_offset" in env_cfg:
        to = env_cfg["table_offset"]
        table_offset = (float(to[0]), float(to[1]), float(to[2]))
    robot_base_configs = None
    if "robot_base_configs" in env_cfg:
        robot_base_configs = []
        for cfg in env_cfg["robot_base_configs"]:
            robot_base_configs.append({
                "position": np.array(cfg["position"], dtype=float),
                "quaternion": np.array(cfg["quaternion"], dtype=float),
            })

    # --- task ---
    task_raw = raw["task"]
    task = TaskConfig(
        manipulated_object=task_raw["manipulated_object"],
        target_pose=_parse_pose(task_raw["target_pose"]),
        start_pose=_parse_pose(task_raw.get("start_pose")),
    )

    # --- planning ---
    plan_raw = raw.get("planning", {})
    rrt_bounds = None
    rrt_bounds_raw = plan_raw.get("rrt_bounds")
    if rrt_bounds_raw is not None:
        rrt_bounds = np.array(rrt_bounds_raw, dtype=float)
        if rrt_bounds.shape != (2, 3):
            logger.warning("rrt_bounds must be shape (2,3) [[lo_x,lo_y,lo_z],[hi_x,hi_y,hi_z]], got %s — ignoring", rrt_bounds.shape)
            rrt_bounds = None

    planning = PlanningConfig(
        num_grasps=plan_raw.get("num_grasps", 200),
        num_entropy_grasps=plan_raw.get("num_entropy_grasps", 20),
        use_dpp=bool(plan_raw.get("use_dpp", True)),
        gripper_types=plan_raw.get("gripper_types", ["finger", "suction"]),
        collision_check=plan_raw.get("collision_check", "Inter"),
        num_steps=plan_raw.get("num_steps", constants.OBJECT_TRAJ_NUM_STEPS),
        lift_height=plan_raw.get("lift_height", constants.OBJECT_TRAJ_LIFT_HEIGHT),
        approach_offset=plan_raw.get("approach_offset", constants.OBJECT_TRAJ_APPROACH_OFFSET),
        rrt_bounds=rrt_bounds,
        collision_ignore_waypoints=plan_raw.get("collision_ignore_waypoints", None),
        collision_ignore_waypoint_names=plan_raw.get("collision_ignore_waypoint_names", None),
        collision_ignore_waypoint_pairs=[
            tuple(p) for p in plan_raw["collision_ignore_waypoint_pairs"]
        ] if plan_raw.get("collision_ignore_waypoint_pairs") else None,
    )

    # --- collision filter ---
    cf_raw = raw.get("collision_filter", {})
    collision_filter = CollisionFilterConfig(
        relevant_patterns=cf_raw.get("relevant_patterns", []),
    )

    # --- visualization ---
    vis_raw = raw.get("visualization", {})
    visualization = VisualizationConfig(
        enable_meshcat=vis_raw.get("enable_meshcat", True),
        visualize_mesh=vis_raw.get("visualize_mesh", False),
        mesh_step_interval=vis_raw.get("mesh_step_interval", constants.VIS_MESH_STEP_INTERVAL),
        mesh_opacity=vis_raw.get("mesh_opacity", constants.VIS_MESH_OPACITY),
        mesh_sample_ratio=vis_raw.get("mesh_sample_ratio", 0.5),
        zmq_url=vis_raw.get("zmq_url", constants.MESHCAT_ZMQ_URL),
    )

    # --- output ---
    out_raw = raw.get("output", {})
    output = OutputConfig(
        save_trajectories=out_raw.get("save_trajectories", False),
        output_dir=out_raw.get("output_dir", None),
    )

    # --- execution ---
    exec_raw = raw.get("execution", {})
    execution = ExecutionConfig(
        visualize=exec_raw.get("visualize", True),
        meshcat_tf=exec_raw.get("meshcat_tf", True),
        smoothen=exec_raw.get("smoothen", True),
        assignment_path=exec_raw.get("assignment_path", None),
    )

    # --- runtime ---
    rt_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        debug=rt_raw.get("debug", False),
        num_workers=rt_raw.get("num_workers", 0),
    )

    # --- assignment ---
    asgn_raw = raw.get("assignment", {})
    tf_feas_raw = asgn_raw.get("transfer_feasibility", None)
    assignment = AssignmentConfig(
        alpha=asgn_raw.get("alpha", constants.ASSIGNMENT_ALPHA),
        beta=asgn_raw.get("beta", constants.ASSIGNMENT_BETA),
        epsilon=asgn_raw.get("epsilon", constants.ASSIGNMENT_EPSILON),
        initial_robot=asgn_raw.get("initial_robot", None),
        transfer_feasibility=tf_feas_raw,
        beam_width=asgn_raw.get("beam_width", None),
        handover_stride=asgn_raw.get("handover_stride", 1),
        handover_clearance_radius=asgn_raw.get("handover_clearance_radius", None),
        num_trajectories_robot0=asgn_raw.get("num_trajectories_robot0", None),
        num_trajectories_robot1=asgn_raw.get("num_trajectories_robot1", None),
    )

    # --- gripper grasp offset overrides ---
    gripper_grasp_offsets = None
    go_raw = raw.get("gripper_grasp_offsets", None)
    if go_raw:
        gripper_grasp_offsets = {}
        for gripper_name, offset_data in go_raw.items():
            parsed = {
                "position": np.array(offset_data["position"], dtype=float),
                "quaternion": np.array(offset_data["quaternion"], dtype=float),
            }
            gripper_grasp_offsets[gripper_name] = parsed
            constants.GRIPPER_GRASP_OFFSETS[gripper_name] = parsed
            logger.info("Overriding GRIPPER_GRASP_OFFSETS[%s]: pos=%s quat=%s",
                        gripper_name, parsed["position"], parsed["quaternion"])

    return ExperimentConfig(
        name=raw.get("name", resolved.stem),
        description=raw.get("description", ""),
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        task=task,
        planning=planning,
        collision_filter=collision_filter,
        visualization=visualization,
        output=output,
        execution=execution,
        runtime=runtime,
        assignment=assignment,
        raw=raw,
        table_full_size=table_full_size,
        table_offset=table_offset,
        robot_base_configs=robot_base_configs,
        gripper_grasp_offsets=gripper_grasp_offsets,
    )


def detect_start_pose(env, object_name: str) -> Optional[PoseConfig]:
    """Query the simulation environment for the current pose of *object_name*.

    Args:
        env: Environment providing ``get_object_pose``.
        object_name: Object instance name.

    Returns:
        World-frame PoseConfig (quaternion xyzw), or ``None`` if the object
        cannot be found.
    """
    try:
        mat = env.get_object_pose(object_name)
        if mat is not None:
            pos = mat[:3, 3]
            quat = R.from_matrix(mat[:3, :3]).as_quat()  # [x, y, z, w]
            return PoseConfig(position=pos, quaternion=quat)
    except Exception as e:
        logger.warning("Could not detect start pose for '%s': %s", object_name, e)
    return None


def get_collision_filter_fn(collision_filter: CollisionFilterConfig):
    """Build a geom filter for collision checking.

    Args:
        collision_filter: Config whose ``relevant_patterns`` are matched as
            substrings of geom names.

    Returns:
        Callable ``(geom_name: str) -> bool`` that returns ``True`` when the
        geom should be **kept** (its name starts with ``robot0``,
        ``gripper0``, ``robot1`` or ``gripper1``, or contains a relevant
        pattern).
    """
    patterns = collision_filter.relevant_patterns

    def _is_relevant(name: str) -> bool:
        """Return ``True`` if ``name`` is a robot/gripper geom or matches a pattern."""
        is_robot = any(name.startswith(p) for p in ("robot0", "gripper0", "robot1", "gripper1"))
        is_task_object = any(pat in name for pat in patterns)
        return is_robot or is_task_object

    return _is_relevant


def list_experiments(experiments_dir: Optional[str] = None) -> List[str]:
    """Return a sorted list of experiment JSON files in *experiments_dir*.

    Args:
        experiments_dir: Directory to search; defaults to
            ``<project_root>/experiments``.

    Returns:
        Sorted ``*.json`` paths, or ``[]`` if the directory does not exist.
    """
    if experiments_dir is None:
        experiments_dir = str(Path(constants.get_project_root()) / "experiments")
    p = Path(experiments_dir)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.json"))
