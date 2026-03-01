"""
Visualization utilities for planning.
"""
import numpy as np
import logging
import meshcat
import meshcat.geometry as g
from typing import Any, Dict, List, Optional

from grasp_gen.utils.meshcat_utils import make_frame
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

logger = logging.getLogger(__name__)

def visualize_trajectory(
    vis: meshcat.Visualizer, 
    trajectory: List[StampedPose], 
    waypoint_results: Optional[List[WaypointFeasibilityResult]] = None,
    name: str = "trajectory", 
    color: int = 0x0000ff,
    marker_size: float = 0.01,
    tf_manager: Optional[TransformManager] = None
) -> None:
    """Visualize trajectory as a line and markers in Meshcat.
    
    Colors for markers:
    - Green: Feasible (IK success + No collision)
    - Red: Infeasible (In collision)
    - Orange: Unreachable (IK failed)
    - Blue: Default (no feasibility data)
    
    Args:
        vis: Meshcat visualizer instance
        trajectory: List of StampedPose objects
        waypoint_results: Optional list of feasibility results
        name: Name for the visualization group
        color: Default color of the trajectory line
        marker_size: Size of the waypoint spheres
        tf_manager: Optional TransformManager to ensure world frame
    """
    if not trajectory:
        return

    # Ensure all frames are in World frame
    world_trajectory = []
    for pose in trajectory:
        if pose.frame_id != constants.WORLD_FRAME:
            if tf_manager is None:
                raise ValueError(f"Pose is in frame '{pose.frame_id}', but no tf_manager provided to convert to world frame.")
            world_trajectory.append(tf_manager.transform_pose(pose, constants.WORLD_FRAME))
        else:
            world_trajectory.append(pose)
    
    trajectory = world_trajectory

    # Use planning root path for organization
    full_path = f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_TRAJECTORY_PATH}/{name}"
    
    # Delete existing visualization
    vis[full_path].delete()
    
    points = np.array([p.position for p in trajectory]).T
    
    # 1. Draw trajectory line
    vis[f"{full_path}/line"].set_object(g.Line(
        g.PointsGeometry(points),
        g.LineBasicMaterial(linewidth=2, color=color)
    ))
    
    # 2. Draw waypoints and frames
    for i, pose in enumerate(trajectory):
        waypoint_path = f"{full_path}/waypoints/wp_{i}"
        
        # Determine status color
        # Default to blue
        wp_color = 0x0000ff
        
        if waypoint_results and i < len(waypoint_results):
            res = waypoint_results[i]
            if not res.ik_success:
                wp_color = 0xffa500 # Orange (Unreachable)
            elif not res.collision_free:
                wp_color = 0xff0000 # Red (Infeasible/Collision)
            elif res.is_feasible:
                wp_color = 0x00ff00 # Green (Feasible)
        
        # Add sphere marker
        vis[f"{waypoint_path}/marker"].set_object(
            g.Sphere(marker_size),
            g.MeshBasicMaterial(color=wp_color)
        )
        
        # Add coordinate frame
        # Draw frame for every N waypoints or if it's start/end to avoid clutter
        if i % 10 == 0 or i == 0 or i == len(trajectory) - 1:
            frame_name = f"{waypoint_path}/frame"
            
            # Use make_frame from grasp_gen to draw axis
            make_frame(vis, frame_name, h=0.04, radius=0.0015, o=0.8)
            # The waypoint_path already has the pose transform, 
            # so the frame (child) should have identity relative to it.
            vis[frame_name].set_transform(np.eye(4))
        
        # Set transform for the whole waypoint group (marker + potentially frame)
        vis[waypoint_path].set_transform(pose.to_matrix())


def visualize_handover_plan(
    vis: meshcat.Visualizer,
    handover_data: dict,
    name: str = "handover",
    tf_manager: Optional[TransformManager] = None,
) -> None:
    """Visualize the handover plan produced by DualArmHandoverPlanner.

    Draws:
      - receiver_pre pose (cyan frame + sphere)
      - transfer_hold pose (magenta frame + sphere)
      - receiver_approach trajectory (line + frames)
      - transfer_retreat trajectory (line + frames)
      - object handover pose (yellow sphere, if inferable)

    Call *after* ``DualArmHandoverPlanner.generate_trajectory()`` and pass
    the returned dict.

    Args:
        vis: MeshCat visualizer instance.
        handover_data: Dict returned by ``DualArmHandoverPlanner.generate_trajectory()``.
        name: Group name in MeshCat.
        tf_manager: Optional TransformManager for frame conversion to world.
    """
    root = f"{constants.VIS_PLANNING_ROOT}/handover/{name}"
    vis[root].delete()

    def _to_world(pose: StampedPose) -> StampedPose:
        if pose.frame_id == constants.WORLD_FRAME or tf_manager is None:
            return pose
        return tf_manager.transform_pose(pose, constants.WORLD_FRAME)

    def _draw_pose(path: str, pose: StampedPose, color: int, sphere_r: float = 0.015,
                   frame_h: float = 0.08, frame_r: float = 0.003, label: str = ""):
        """Draw a coordinate frame + coloured sphere at *pose*."""
        wp = _to_world(pose)
        T = wp.to_matrix()
        vis[path].set_transform(T)
        vis[f"{path}/sphere"].set_object(
            g.Sphere(sphere_r), g.MeshBasicMaterial(color=color, opacity=0.85)
        )
        make_frame(vis, f"{path}/frame", h=frame_h, radius=frame_r, o=0.9)
        vis[f"{path}/frame"].set_transform(np.eye(4))  # relative to parent
        if label:
            logger.info(f"  {label}: pos={wp.position}, quat={wp.quaternion}")

    def _draw_traj(path: str, traj: List[StampedPose], color: int, line_color: int,
                   sphere_r: float = 0.008, frame_step: int = 10, label: str = ""):
        """Draw a trajectory as a line with periodic frames."""
        world_traj = [_to_world(p) for p in traj]
        points = np.array([p.position for p in world_traj])  # (N, 3)

        # Line
        vis[f"{path}/line"].set_object(
            g.Line(g.PointsGeometry(points.T), g.LineBasicMaterial(linewidth=2, color=line_color))
        )

        # Start / end markers + periodic frames
        for i, wp in enumerate(world_traj):
            if i == 0 or i == len(world_traj) - 1 or i % frame_step == 0:
                wp_path = f"{path}/wp_{i}"
                vis[wp_path].set_transform(wp.to_matrix())
                vis[f"{wp_path}/sphere"].set_object(
                    g.Sphere(sphere_r), g.MeshBasicMaterial(color=color, opacity=0.7)
                )
                make_frame(vis, f"{wp_path}/frame", h=0.04, radius=0.0015, o=0.7)
                vis[f"{wp_path}/frame"].set_transform(np.eye(4))

        if label:
            logger.info(f"  {label}: {len(traj)} waypoints, "
                        f"start={world_traj[0].position}, end={world_traj[-1].position}")

    logger.info(f"Visualizing handover plan '{name}':")

    # --- Key poses ---
    if "receiver_pre" in handover_data and handover_data["receiver_pre"]:
        _draw_pose(f"{root}/receiver_pre", handover_data["receiver_pre"][0],
                   color=0x00DDDD, label="receiver_pre")   # cyan

    if "transfer_hold" in handover_data and handover_data["transfer_hold"]:
        _draw_pose(f"{root}/transfer_hold", handover_data["transfer_hold"][0],
                   color=0xDD00DD, label="transfer_hold")  # magenta

    # --- Trajectories ---
    if "receiver_approach" in handover_data and handover_data["receiver_approach"]:
        _draw_traj(f"{root}/receiver_approach", handover_data["receiver_approach"],
                   color=0x00AAFF, line_color=0x00CCFF, label="receiver_approach")

    if "transfer_retreat" in handover_data and handover_data["transfer_retreat"]:
        _draw_traj(f"{root}/transfer_retreat", handover_data["transfer_retreat"],
                   color=0xFF6600, line_color=0xFF8800, label="transfer_retreat")

    logger.info(f"Handover plan '{name}' visualized in MeshCat.")


# ---------------------------------------------------------------------------
# Assignment / feasibility heatmap (matplotlib)
# ---------------------------------------------------------------------------

def visualize_trajectory_feasibility(
    trajectories: Dict[str, Any],
    save_path: Optional[str] = None,
) -> Any:
    """Render a heatmap of per-waypoint feasibility across all trajectories.

    Creates a single image with green (feasible), red (infeasible), and grey
    (missing) cells.

    Args:
        trajectories: Dict with ``'robot0_trajectories'``,
            ``'robot1_trajectories'``, and ``'object_trajectory'``.
        save_path: If provided, save the figure to this path.

    Returns:
        ``(fig, ax)`` matplotlib objects (or *None* if matplotlib unavailable).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping feasibility heatmap.")
        return None

    r0 = trajectories["robot0_trajectories"]
    r1 = trajectories["robot1_trajectories"]
    n_waypoints = len(trajectories["object_trajectory"]["positions"])
    n_r0, n_r1 = len(r0), len(r1)
    total = n_r0 + n_r1

    matrix = np.full((total, n_waypoints), np.nan)

    def _fill(trajs, offset):
        for t_idx, traj in enumerate(trajs):
            feasible = np.asarray(traj["is_feasible"])
            if feasible.ndim == 0:
                feasible = np.full(n_waypoints, feasible.item(), dtype=bool)
            for w in range(min(n_waypoints, len(feasible))):
                matrix[offset + t_idx, w] = float(feasible[w])

    _fill(r0, 0)
    _fill(r1, n_r0)

    fig, ax = plt.subplots(
        figsize=(max(12, n_waypoints * 0.3), max(8, total * 0.4))
    )
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
        interpolation="nearest", origin="upper",
    )
    ax.set_xlabel("Waypoint Index", fontsize=12)
    ax.set_ylabel("Trajectory Index", fontsize=12)
    ax.set_title(
        f"Trajectory Feasibility ({n_r0} R0 + {n_r1} R1, {n_waypoints} waypoints)",
        fontsize=14, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Feasibility (1=OK, 0=fail)", fontsize=10)

    labels = [f"R0-{i}" for i in range(n_r0)] + [f"R1-{i}" for i in range(n_r1)]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    if n_r0 > 0 and n_r1 > 0:
        ax.axhline(y=n_r0 - 0.5, color="blue", ls="--", lw=2, alpha=0.7)

    ax.set_xticks(np.arange(-0.5, n_waypoints, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, total, 1), minor=True)
    ax.grid(which="minor", color="black", ls="-", lw=0.5, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feasibility heatmap saved to %s", save_path)

    return fig, ax


def visualize_assignment_result(
    result: Any,
    save_path: Optional[str] = None,
) -> Any:
    """Render the optimal assignment sequence as a bar chart.

    Args:
        result: An :class:`AssignmentResult`.
        save_path: If provided, save the figure to this path.

    Returns:
        ``(fig, ax)`` or *None*.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping assignment plot.")
        return None

    n = len(result.sequence)
    colors = ["#1f77b4" if r == 0 else "#ff7f0e" for r in result.sequence]
    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 5))
    ax.bar(range(n), result.task_costs, color=colors, label="task cost")
    ax.bar(range(n), result.switching_costs, bottom=result.task_costs,
           color="red", alpha=0.5, label="switch cost")

    for hp in result.handover_points:
        ax.axvline(x=hp - 0.5, color="red", ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel("Segment")
    ax.set_ylabel("Cost")
    ax.set_title(f"Assignment (total={result.total_cost:.3f}, handovers={len(result.handover_points)})")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Assignment plot saved to %s", save_path)

    return fig, ax
