"""
Visualization utilities for grasp poses and grippers.

This module contains all visualization functions for:
- Drawing parallel jaw grippers
- Drawing suction grippers
- Drawing gripper cuboids
- Visualizing grasp poses
- Visualizing feasible/infeasible grasp pairs
"""

import numpy as np

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .collision_checker import SUCTION_GEOMETRY, PARALLEL_JAW_GEOMETRY


def draw_parallel_jaw_gripper(
    ax,
    position: np.ndarray,
    orientation: np.ndarray,
    gripper_opening: float,
    gripper_body_length: float = 2.0,
    jaw_length: float = 2.5,
    jaw_width: float = 0.3,
    jaw_height: float = 0.8,
):
    """
    Draw a parallel jaw gripper at the given pose.
    
    Args:
        ax: 3D axes object
        position: 3D position of gripper center (in plate frame)
        orientation: 3x3 rotation matrix (gripper frame orientation)
        gripper_opening: Distance between jaws (cm)
        gripper_body_length: Length of gripper body (cm)
        jaw_length: Length of each jaw (cm)
        jaw_width: Width of each jaw (cm)
        jaw_height: Height of each jaw (cm)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    # Gripper frame axes:
    # x-axis: along the edge
    # y-axis: opening direction (jaws open/close along this axis)
    # z-axis: approach direction (negative of approach_direction)
    
    R = orientation
    x_axis = R[:, 0]
    y_axis = R[:, 1]  # Opening direction
    z_axis = R[:, 2]  # Approach direction
    
    # Gripper body (small box behind the jaws)
    body_length = gripper_body_length
    body_width = gripper_opening + 0.5  # Slightly wider than opening
    body_height = jaw_height
    
    # Body center is offset along -z_axis (behind the jaws)
    body_center = position - 0.5 * body_length * z_axis
    
    # Create body vertices in gripper frame
    body_half_dims = np.array([body_length/2, body_width/2, body_height/2])
    body_vertices_local = np.array([
        [-body_half_dims[0], -body_half_dims[1], -body_half_dims[2]],
        [body_half_dims[0], -body_half_dims[1], -body_half_dims[2]],
        [body_half_dims[0], body_half_dims[1], -body_half_dims[2]],
        [-body_half_dims[0], body_half_dims[1], -body_half_dims[2]],
        [-body_half_dims[0], -body_half_dims[1], body_half_dims[2]],
        [body_half_dims[0], -body_half_dims[1], body_half_dims[2]],
        [body_half_dims[0], body_half_dims[1], body_half_dims[2]],
        [-body_half_dims[0], body_half_dims[1], body_half_dims[2]],
    ])
    
    # Transform to world frame
    body_vertices_world = (body_center + (R @ body_vertices_local.T).T)
    
    # Draw gripper body
    body_faces = [
        [body_vertices_world[0], body_vertices_world[1], body_vertices_world[2], body_vertices_world[3]],
        [body_vertices_world[4], body_vertices_world[5], body_vertices_world[6], body_vertices_world[7]],
        [body_vertices_world[0], body_vertices_world[1], body_vertices_world[5], body_vertices_world[4]],
        [body_vertices_world[2], body_vertices_world[3], body_vertices_world[7], body_vertices_world[6]],
        [body_vertices_world[0], body_vertices_world[3], body_vertices_world[7], body_vertices_world[4]],
        [body_vertices_world[1], body_vertices_world[2], body_vertices_world[6], body_vertices_world[5]],
    ]
    body_collection = Poly3DCollection(body_faces, alpha=0.7, facecolor='darkgray', edgecolor='black', linewidths=0.5)
    ax.add_collection3d(body_collection)
    
    # Draw two parallel jaws
    jaw_half_opening = gripper_opening / 2.0
    jaw_half_length = jaw_length / 2.0
    jaw_half_width = jaw_width / 2.0
    jaw_half_height = jaw_height / 2.0
    
    # Left jaw (negative y direction)
    left_jaw_center = position - jaw_half_opening * y_axis
    left_jaw_vertices_local = np.array([
        [-jaw_half_length, -jaw_half_width, -jaw_half_height],
        [jaw_half_length, -jaw_half_width, -jaw_half_height],
        [jaw_half_length, jaw_half_width, -jaw_half_height],
        [-jaw_half_length, jaw_half_width, -jaw_half_height],
        [-jaw_half_length, -jaw_half_width, jaw_half_height],
        [jaw_half_length, -jaw_half_width, jaw_half_height],
        [jaw_half_length, jaw_half_width, jaw_half_height],
        [-jaw_half_length, jaw_half_width, jaw_half_height],
    ])
    left_jaw_vertices_world = (left_jaw_center + (R @ left_jaw_vertices_local.T).T)
    
    left_jaw_faces = [
        [left_jaw_vertices_world[0], left_jaw_vertices_world[1], left_jaw_vertices_world[2], left_jaw_vertices_world[3]],
        [left_jaw_vertices_world[4], left_jaw_vertices_world[5], left_jaw_vertices_world[6], left_jaw_vertices_world[7]],
        [left_jaw_vertices_world[0], left_jaw_vertices_world[1], left_jaw_vertices_world[5], left_jaw_vertices_world[4]],
        [left_jaw_vertices_world[2], left_jaw_vertices_world[3], left_jaw_vertices_world[7], left_jaw_vertices_world[6]],
        [left_jaw_vertices_world[0], left_jaw_vertices_world[3], left_jaw_vertices_world[7], left_jaw_vertices_world[4]],
        [left_jaw_vertices_world[1], left_jaw_vertices_world[2], left_jaw_vertices_world[6], left_jaw_vertices_world[5]],
    ]
    left_jaw_collection = Poly3DCollection(left_jaw_faces, alpha=0.8, facecolor='steelblue', edgecolor='black', linewidths=0.5)
    ax.add_collection3d(left_jaw_collection)
    
    # Right jaw (positive y direction)
    right_jaw_center = position + jaw_half_opening * y_axis
    right_jaw_vertices_local = np.array([
        [-jaw_half_length, -jaw_half_width, -jaw_half_height],
        [jaw_half_length, -jaw_half_width, -jaw_half_height],
        [jaw_half_length, jaw_half_width, -jaw_half_height],
        [-jaw_half_length, jaw_half_width, -jaw_half_height],
        [-jaw_half_length, -jaw_half_width, jaw_half_height],
        [jaw_half_length, -jaw_half_width, jaw_half_height],
        [jaw_half_length, jaw_half_width, jaw_half_height],
        [-jaw_half_length, jaw_half_width, jaw_half_height],
    ])
    right_jaw_vertices_world = (right_jaw_center + (R @ right_jaw_vertices_local.T).T)
    
    right_jaw_faces = [
        [right_jaw_vertices_world[0], right_jaw_vertices_world[1], right_jaw_vertices_world[2], right_jaw_vertices_world[3]],
        [right_jaw_vertices_world[4], right_jaw_vertices_world[5], right_jaw_vertices_world[6], right_jaw_vertices_world[7]],
        [right_jaw_vertices_world[0], right_jaw_vertices_world[1], right_jaw_vertices_world[5], right_jaw_vertices_world[4]],
        [right_jaw_vertices_world[2], right_jaw_vertices_world[3], right_jaw_vertices_world[7], right_jaw_vertices_world[6]],
        [right_jaw_vertices_world[0], right_jaw_vertices_world[3], right_jaw_vertices_world[7], right_jaw_vertices_world[4]],
        [right_jaw_vertices_world[1], right_jaw_vertices_world[2], right_jaw_vertices_world[6], right_jaw_vertices_world[5]],
    ]
    right_jaw_collection = Poly3DCollection(right_jaw_faces, alpha=0.8, facecolor='steelblue', edgecolor='black', linewidths=0.5)
    ax.add_collection3d(right_jaw_collection)


def draw_suction_gripper(
    ax,
    position: np.ndarray,
    orientation: np.ndarray,
    suction_cup_radius: float = 1.0,
    cup_height: float = 0.5,
    base_radius: float = 0.8,
    base_height: float = 1.0,
):
    """
    Draw a suction gripper at the given pose.
    
    Args:
        ax: 3D axes object
        position: 3D position of suction cup center (in plate frame)
        orientation: 3x3 rotation matrix (gripper frame orientation)
        suction_cup_radius: Radius of the suction cup (cm)
        cup_height: Height of the suction cup (cm)
        base_radius: Radius of the gripper base (cm)
        base_height: Height of the gripper base (cm)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    R = orientation
    z_axis = R[:, 2]  # Approach direction (points toward plate)
    
    # Create a cylinder for the suction cup
    # We'll approximate with a polygon
    num_segments = 16
    angles = np.linspace(0, 2 * np.pi, num_segments + 1)
    
    # Suction cup (bottom, at position)
    cup_bottom_center = position
    cup_top_center = position - cup_height * z_axis
    
    # Create vertices for the cup cylinder
    cup_bottom_circle = []
    cup_top_circle = []
    for angle in angles[:-1]:
        # Local x and y in the plane perpendicular to z_axis
        # We need two perpendicular vectors in this plane
        if abs(z_axis[2]) < 0.9:
            local_x = np.array([1, 0, 0])
        else:
            local_x = np.array([0, 1, 0])
        local_x = local_x - np.dot(local_x, z_axis) * z_axis
        local_x = local_x / np.linalg.norm(local_x)
        local_y = np.cross(z_axis, local_x)
        
        # Point on circle
        circle_point = suction_cup_radius * (np.cos(angle) * local_x + np.sin(angle) * local_y)
        cup_bottom_circle.append(cup_bottom_center + circle_point)
        cup_top_circle.append(cup_top_center + circle_point)
    
    # Draw cup sides (cylinder)
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        face = [
            cup_bottom_circle[i],
            cup_bottom_circle[next_i],
            cup_top_circle[next_i],
            cup_top_circle[i],
        ]
        cup_face = Poly3DCollection([face], alpha=0.7, facecolor='orange', edgecolor='darkorange', linewidths=0.5)
        ax.add_collection3d(cup_face)
    
    # Draw cup bottom (circle)
    cup_bottom_face = Poly3DCollection([cup_bottom_circle], alpha=0.8, facecolor='orange', edgecolor='darkorange', linewidths=0.5)
    ax.add_collection3d(cup_bottom_face)
    
    # Draw cup top (circle)
    cup_top_face = Poly3DCollection([cup_top_circle], alpha=0.8, facecolor='lightcoral', edgecolor='darkorange', linewidths=0.5)
    ax.add_collection3d(cup_top_face)
    
    # Draw gripper base (cylinder above the cup)
    base_bottom_center = cup_top_center
    base_top_center = base_bottom_center - base_height * z_axis
    
    base_bottom_circle = []
    base_top_circle = []
    for angle in angles[:-1]:
        if abs(z_axis[2]) < 0.9:
            local_x = np.array([1, 0, 0])
        else:
            local_x = np.array([0, 1, 0])
        local_x = local_x - np.dot(local_x, z_axis) * z_axis
        local_x = local_x / np.linalg.norm(local_x)
        local_y = np.cross(z_axis, local_x)
        
        circle_point = base_radius * (np.cos(angle) * local_x + np.sin(angle) * local_y)
        base_bottom_circle.append(base_bottom_center + circle_point)
        base_top_circle.append(base_top_center + circle_point)
    
    # Draw base sides
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        face = [
            base_bottom_circle[i],
            base_bottom_circle[next_i],
            base_top_circle[next_i],
            base_top_circle[i],
        ]
        base_face = Poly3DCollection([face], alpha=0.6, facecolor='gray', edgecolor='black', linewidths=0.5)
        ax.add_collection3d(base_face)
    
    # Draw base top
    base_top_face = Poly3DCollection([base_top_circle], alpha=0.7, facecolor='darkgray', edgecolor='black', linewidths=0.5)
    ax.add_collection3d(base_top_face)


def draw_gripper_cuboid(
    ax,
    gripper_geometry,
    position: np.ndarray,
    orientation: np.ndarray,
    color: str = 'red',
    alpha: float = 0.3,
):
    """Draw a gripper as a cuboid for visualization."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    vertices = gripper_geometry.get_cuboid_vertices(position, orientation)
    
    # Define faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    
    cuboid = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidths=0.5)
    ax.add_collection3d(cuboid)


def draw_se3_frame(
    ax,
    position: np.ndarray,
    approach_direction: np.ndarray,
    frame_scale: float = 1.0,
    linewidth: float = 2.0,
):
    """
    Draw an SE3 coordinate frame where x-axis is in the approach direction.
    
    Args:
        ax: 3D axes object
        position: 3D position of the frame origin
        approach_direction: Direction vector (will be normalized, becomes x-axis)
        frame_scale: Scale factor for frame axes
        linewidth: Line width for frame axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    # Normalize approach direction to get x-axis
    x_axis = approach_direction / np.linalg.norm(approach_direction)
    
    # Create a perpendicular y-axis
    # Choose a vector that's not parallel to x_axis
    if abs(x_axis[2]) < 0.9:
        temp = np.array([0, 0, 1])
    else:
        temp = np.array([1, 0, 0])
    
    # Gram-Schmidt to get perpendicular y-axis
    y_axis = temp - np.dot(temp, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # z-axis is cross product of x and y
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Recompute y-axis to ensure right-handed frame
    y_axis = np.cross(z_axis, x_axis)
    
    # Draw axes (no labels here to avoid duplicate legend entries)
    ax.plot(
        [position[0], position[0] + x_axis[0] * frame_scale],
        [position[1], position[1] + x_axis[1] * frame_scale],
        [position[2], position[2] + x_axis[2] * frame_scale],
        'r-', linewidth=linewidth, alpha=0.8
    )
    ax.plot(
        [position[0], position[0] + y_axis[0] * frame_scale],
        [position[1], position[1] + y_axis[1] * frame_scale],
        [position[2], position[2] + y_axis[2] * frame_scale],
        'g-', linewidth=linewidth, alpha=0.8
    )
    ax.plot(
        [position[0], position[0] + z_axis[0] * frame_scale],
        [position[1], position[1] + z_axis[1] * frame_scale],
        [position[2], position[2] + z_axis[2] * frame_scale],
        'b-', linewidth=linewidth, alpha=0.8
    )
    
    # Draw origin point
    ax.scatter([position[0]], [position[1]], [position[2]], c='black', s=30, marker='o')


def draw_plate(ax, plate_width: float, plate_length: float, plate_thickness: float, transform: np.ndarray = None, alpha: float = 0.3, facecolor: str = 'lightblue'):
    """
    Draw the plate as a 3D box.
    
    Args:
        ax: 3D axes object
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        transform: Optional 4x4 transformation matrix to transform the plate pose
        alpha: Alpha transparency value (default: 0.3)
        facecolor: Face color of the plate (default: 'lightblue')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    half_w = plate_width / 2.0
    half_l = plate_length / 2.0
    half_t = plate_thickness / 2.0
    
    vertices = np.array([
        [-half_w, -half_l, -half_t],
        [half_w, -half_l, -half_t],
        [half_w, half_l, -half_t],
        [-half_w, half_l, -half_t],
        [-half_w, -half_l, half_t],
        [half_w, -half_l, half_t],
        [half_w, half_l, half_t],
        [-half_w, half_l, half_t],
    ])
    
    # Apply transformation if provided
    if transform is not None:
        # Convert to homogeneous coordinates
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        # Transform vertices
        vertices_transformed = (transform @ vertices_homogeneous.T).T
        vertices = vertices_transformed[:, :3]
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    
    plate = Poly3DCollection(faces, alpha=alpha, facecolor=facecolor, edgecolor='black', linewidths=1)
    ax.add_collection3d(plate)


def visualize_grasps(
    parallel_jaw_grasps: list = None,
    suction_grasps: list = None,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    show_approach_vectors: bool = False,
    show_grippers: bool = True,
    approach_scale: float = 2.0,
    max_grasps_to_show: int = None,
    save_path: str = None,
):
    """
    Visualize the plate and grasp poses in 3D with both parallel jaw and suction grippers.
    
    All poses are in the plate frame (centroid at origin).
    
    Args:
        parallel_jaw_grasps: List of GraspPose objects for parallel jaw gripper (all in plate frame)
        suction_grasps: List of GraspPose objects for suction gripper (all in plate frame)
        plate_width: Width of the plate in cm
        plate_length: Length of the plate in cm
        plate_thickness: Thickness of the plate in cm
        show_approach_vectors: If True, show approach direction arrows
        show_grippers: If True, show grippers at grasp positions
        approach_scale: Scale factor for approach direction arrows
        max_grasps_to_show: Maximum number of grasps to visualize per type (None = all)
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if parallel_jaw_grasps is None:
        parallel_jaw_grasps = []
    if suction_grasps is None:
        suction_grasps = []
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Limit number of grasps to show if specified
    parallel_jaw_to_show = parallel_jaw_grasps
    suction_to_show = suction_grasps
    
    if max_grasps_to_show is not None:
        if len(parallel_jaw_grasps) > max_grasps_to_show:
            parallel_jaw_to_show = parallel_jaw_grasps[:max_grasps_to_show]
            print(f"Visualizing {max_grasps_to_show} of {len(parallel_jaw_grasps)} parallel jaw grasps")
        if len(suction_grasps) > max_grasps_to_show:
            suction_to_show = suction_grasps[:max_grasps_to_show]
            print(f"Visualizing {max_grasps_to_show} of {len(suction_grasps)} suction grasps")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw parallel jaw grippers
    if show_grippers and parallel_jaw_to_show:
        for grasp in parallel_jaw_to_show:
            all_positions.append(grasp.position)
            draw_parallel_jaw_gripper(
                ax,
                grasp.position,
                grasp.orientation,
                grasp.gripper_opening,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Draw suction grippers
    if show_grippers and suction_to_show:
        for grasp in suction_to_show:
            all_positions.append(grasp.position)
            draw_suction_gripper(
                ax,
                grasp.position,
                grasp.orientation,
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
    
    # Draw approach vectors (optional)
    if show_approach_vectors:
        for grasp in parallel_jaw_to_show + suction_to_show:
            start = grasp.position
            end = start + approach_scale * grasp.approach_direction
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                'g-',
                linewidth=2,
                alpha=0.6
            )
            # Arrow head
            ax.scatter([end[0]], [end[1]], [end[2]], c='green', s=30, marker='^')
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    total_grasps = len(parallel_jaw_grasps) + len(suction_grasps)
    title = f'Grasp Poses Visualization (Plate Frame - Centroid at Origin)\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Parallel jaw grasps: {len(parallel_jaw_grasps)}, Suction grasps: {len(suction_grasps)} (Total: {total_grasps})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        # Add some padding
        max_range *= 1.2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        # Default view if no grasps
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    legend_elements = []
    if show_grippers:
        if parallel_jaw_to_show:
            legend_elements.append(
                plt.Line2D([0], [0], color='steelblue', linewidth=4, label='Parallel jaw grippers')
            )
        if suction_to_show:
            legend_elements.append(
                plt.Line2D([0], [0], color='orange', linewidth=4, label='Suction grippers')
            )
    if show_approach_vectors:
        legend_elements.append(
            plt.Line2D([0], [0], color='green', linewidth=2, label='Approach direction')
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_trajectories(
    all_trajectories: list,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_trajectories_to_show: int = 50,
    show_grippers: bool = False,
    trajectory_linewidth: float = 1.5,
    trajectory_alpha: float = 0.6,
    show_start_end: bool = True,
    save_path: str = None,
):
    """
    Visualize end effector trajectories for all feasible grasp pairs.
    
    Args:
        all_trajectories: List of trajectory dictionaries, each containing:
            - 'suction_grasp': suction grasp pose
            - 'parallel_jaw_grasp': parallel jaw grasp pose
            - 'suction_trajectory': list of 4x4 matrices for suction gripper
            - 'parallel_jaw_trajectory': list of 4x4 matrices for parallel jaw gripper
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_trajectories_to_show: Maximum number of trajectory pairs to visualize
        show_grippers: If True, show grippers at start/end positions
        trajectory_linewidth: Line width for trajectory paths
        trajectory_alpha: Alpha transparency for trajectory paths
        show_start_end: If True, mark start and end points
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not all_trajectories:
        print("No trajectories to visualize!")
        return None, None
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Limit number of trajectories to show
    trajectories_to_show = all_trajectories[:max_trajectories_to_show] if max_trajectories_to_show else all_trajectories
    if len(all_trajectories) > max_trajectories_to_show:
        print(f"Showing {max_trajectories_to_show} of {len(all_trajectories)} trajectory pairs")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw object trajectory first (so it appears behind other trajectories)
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        object_traj = trajectories_to_show[0]['object_trajectory']
        object_positions = np.array([T[:3, 3] for T in object_traj])
        all_positions.extend(object_positions)
        
        # Draw object trajectory with distinctive style (thick, dark, dashed)
        ax.plot(
            object_positions[:, 0],
            object_positions[:, 1],
            object_positions[:, 2],
            '--',
            color='black',
            linewidth=trajectory_linewidth * 2.5,
            alpha=0.9,
            markersize=5,
            label='Object trajectory',
            zorder=1  # Draw behind other trajectories
        )
        
        # Mark object start and end points
        if show_start_end:
            ax.scatter(
                [object_positions[0, 0]],
                [object_positions[0, 1]],
                [object_positions[0, 2]],
                c='black',
                s=150,
                marker='D',  # Diamond for object start
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object start'
            )
            ax.scatter(
                [object_positions[-1, 0]],
                [object_positions[-1, 1]],
                [object_positions[-1, 2]],
                c='black',
                s=200,
                marker='X',  # X for object end
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object end'
            )
    
    # Color cycle for different trajectories
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(trajectories_to_show)))
    
    # Draw trajectories
    for idx, traj_data in enumerate(trajectories_to_show):
        suction_traj = traj_data['suction_trajectory']
        parallel_jaw_traj = traj_data['parallel_jaw_trajectory']
        color = colors[idx % len(colors)]
        
        # Extract positions from transformation matrices
        suction_positions = np.array([T[:3, 3] for T in suction_traj])
        parallel_jaw_positions = np.array([T[:3, 3] for T in parallel_jaw_traj])
        
        all_positions.extend(suction_positions)
        all_positions.extend(parallel_jaw_positions)
        
        # Draw suction gripper trajectory
        ax.plot(
            suction_positions[:, 0],
            suction_positions[:, 1],
            suction_positions[:, 2],
            'o-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Suction {idx+1}' if idx < 10 else None
        )
        
        # Draw parallel jaw gripper trajectory
        ax.plot(
            parallel_jaw_positions[:, 0],
            parallel_jaw_positions[:, 1],
            parallel_jaw_positions[:, 2],
            's-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Parallel jaw {idx+1}' if idx < 10 else None
        )
        
        # Mark start and end points
        if show_start_end:
            # Start points
            ax.scatter(
                [suction_positions[0, 0]],
                [suction_positions[0, 1]],
                [suction_positions[0, 2]],
                c=color,
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[0, 0]],
                [parallel_jaw_positions[0, 1]],
                [parallel_jaw_positions[0, 2]],
                c=color,
                s=100,
                marker='s',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            
            # End points
            ax.scatter(
                [suction_positions[-1, 0]],
                [suction_positions[-1, 1]],
                [suction_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[-1, 0]],
                [parallel_jaw_positions[-1, 1]],
                [parallel_jaw_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
        
        # Optionally show grippers at start/end
        if show_grippers:
            # Start grippers
            suction_start_T = suction_traj[0]
            parallel_jaw_start_T = parallel_jaw_traj[0]
            
            draw_suction_gripper(
                ax,
                suction_start_T[:3, 3],
                suction_start_T[:3, :3],
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
            draw_parallel_jaw_gripper(
                ax,
                parallel_jaw_start_T[:3, 3],
                parallel_jaw_start_T[:3, :3],
                gripper_opening=2.2,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'End Effector Trajectories Visualization\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Total trajectory pairs: {len(all_trajectories)} (showing {len(trajectories_to_show)})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        max_range *= 1.2  # Add padding
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    legend_elements = []
    
    # Add object trajectory to legend if it was drawn
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', label='Object trajectory')
        )
        if show_start_end:
            legend_elements.extend([
                plt.Line2D([0], [0], color='black', marker='D', linestyle='None', markersize=10, markeredgecolor='white', markeredgewidth=2, label='Object start'),
                plt.Line2D([0], [0], color='black', marker='X', linestyle='None', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Object end'),
            ])
    
    legend_elements.extend([
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='o', linestyle='-', label='Suction gripper trajectories'),
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='s', linestyle='-', label='Parallel jaw gripper trajectories'),
    ])
    if show_start_end:
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Gripper start (circle)'),
            plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10, label='Gripper end (star)'),
        ])
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_all_grasps(
    all_pairs: list,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_pairs_to_show: int = 200,
    show_cuboids: bool = True,
    show_frames: bool = False,
    frame_scale: float = 1.0,
    save_path: str = None,
):
    """
    Visualize all grasp pairs without distinguishing feasible/infeasible.
    
    Args:
        all_pairs: List of (suction_grasp, parallel_jaw_grasp) tuples (all pairs)
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_pairs_to_show: Maximum number of pairs to visualize
        show_cuboids: If True, show gripper cuboids; if False, show detailed grippers
        show_frames: If True, show SE3 coordinate frames (x-axis = approach direction)
        frame_scale: Scale factor for frame axes
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not all_pairs:
        print("No pairs to visualize!")
        return None, None
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Limit number of pairs to show
    pairs_to_show = all_pairs[:max_pairs_to_show] if max_pairs_to_show else all_pairs
    if len(all_pairs) > max_pairs_to_show:
        print(f"Showing {max_pairs_to_show} of {len(all_pairs)} total pairs")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw all grasps (same color for all)
    for suction_grasp, parallel_jaw_grasp in pairs_to_show:
        all_positions.append(suction_grasp.position)
        all_positions.append(parallel_jaw_grasp.position)
        
        if show_frames:
            # Draw as SE3 frames with x-axis in approach direction
            draw_se3_frame(
                ax, suction_grasp.position, suction_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.5
            )
            draw_se3_frame(
                ax, parallel_jaw_grasp.position, parallel_jaw_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.5
            )
        elif show_cuboids:
            # Draw as cuboids with neutral colors
            draw_gripper_cuboid(
                ax, SUCTION_GEOMETRY, suction_grasp.position, suction_grasp.orientation,
                color='orange', alpha=0.3
            )
            draw_gripper_cuboid(
                ax, PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp.position, parallel_jaw_grasp.orientation,
                color='steelblue', alpha=0.3
            )
        else:
            # Draw detailed grippers
            draw_suction_gripper(
                ax,
                suction_grasp.position,
                suction_grasp.orientation,
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
            draw_parallel_jaw_gripper(
                ax,
                parallel_jaw_grasp.position,
                parallel_jaw_grasp.orientation,
                parallel_jaw_grasp.gripper_opening,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'All Grasp Pairs Visualization (Plate Frame - Centroid at Origin)\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Total pairs: {len(all_pairs)} (showing {len(pairs_to_show)})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        max_range *= 1.2  # Add padding
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    if show_frames:
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=2, label='X-axis (approach)'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Y-axis'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Z-axis'),
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], color='orange', linewidth=4, label='Suction grippers'),
            plt.Line2D([0], [0], color='steelblue', linewidth=4, label='Parallel jaw grippers'),
        ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_trajectories(
    all_trajectories: list,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_trajectories_to_show: int = 50,
    show_grippers: bool = False,
    trajectory_linewidth: float = 1.5,
    trajectory_alpha: float = 0.6,
    show_start_end: bool = True,
    save_path: str = None,
):
    """
    Visualize end effector trajectories for all feasible grasp pairs.
    
    Args:
        all_trajectories: List of trajectory dictionaries, each containing:
            - 'suction_grasp': suction grasp pose
            - 'parallel_jaw_grasp': parallel jaw grasp pose
            - 'suction_trajectory': list of 4x4 matrices for suction gripper
            - 'parallel_jaw_trajectory': list of 4x4 matrices for parallel jaw gripper
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_trajectories_to_show: Maximum number of trajectory pairs to visualize
        show_grippers: If True, show grippers at start/end positions
        trajectory_linewidth: Line width for trajectory paths
        trajectory_alpha: Alpha transparency for trajectory paths
        show_start_end: If True, mark start and end points
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not all_trajectories:
        print("No trajectories to visualize!")
        return None, None
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Limit number of trajectories to show
    trajectories_to_show = all_trajectories[:max_trajectories_to_show] if max_trajectories_to_show else all_trajectories
    if len(all_trajectories) > max_trajectories_to_show:
        print(f"Showing {max_trajectories_to_show} of {len(all_trajectories)} trajectory pairs")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw object trajectory first (so it appears behind other trajectories)
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        object_traj = trajectories_to_show[0]['object_trajectory']
        object_positions = np.array([T[:3, 3] for T in object_traj])
        all_positions.extend(object_positions)
        
        # Draw object trajectory with distinctive style (thick, dark, dashed)
        ax.plot(
            object_positions[:, 0],
            object_positions[:, 1],
            object_positions[:, 2],
            '--',
            color='black',
            linewidth=trajectory_linewidth * 2.5,
            alpha=0.9,
            markersize=5,
            label='Object trajectory',
            zorder=1  # Draw behind other trajectories
        )
        
        # Mark object start and end points
        if show_start_end:
            ax.scatter(
                [object_positions[0, 0]],
                [object_positions[0, 1]],
                [object_positions[0, 2]],
                c='black',
                s=150,
                marker='D',  # Diamond for object start
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object start'
            )
            ax.scatter(
                [object_positions[-1, 0]],
                [object_positions[-1, 1]],
                [object_positions[-1, 2]],
                c='black',
                s=200,
                marker='X',  # X for object end
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object end'
            )
    
    # Color cycle for different trajectories
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(trajectories_to_show)))
    
    # Draw trajectories
    for idx, traj_data in enumerate(trajectories_to_show):
        suction_traj = traj_data['suction_trajectory']
        parallel_jaw_traj = traj_data['parallel_jaw_trajectory']
        color = colors[idx % len(colors)]
        
        # Extract positions from transformation matrices
        suction_positions = np.array([T[:3, 3] for T in suction_traj])
        parallel_jaw_positions = np.array([T[:3, 3] for T in parallel_jaw_traj])
        
        all_positions.extend(suction_positions)
        all_positions.extend(parallel_jaw_positions)
        
        # Draw suction gripper trajectory
        ax.plot(
            suction_positions[:, 0],
            suction_positions[:, 1],
            suction_positions[:, 2],
            'o-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Suction {idx+1}' if idx < 10 else None
        )
        
        # Draw parallel jaw gripper trajectory
        ax.plot(
            parallel_jaw_positions[:, 0],
            parallel_jaw_positions[:, 1],
            parallel_jaw_positions[:, 2],
            's-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Parallel jaw {idx+1}' if idx < 10 else None
        )
        
        # Mark start and end points
        if show_start_end:
            # Start points
            ax.scatter(
                [suction_positions[0, 0]],
                [suction_positions[0, 1]],
                [suction_positions[0, 2]],
                c=color,
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[0, 0]],
                [parallel_jaw_positions[0, 1]],
                [parallel_jaw_positions[0, 2]],
                c=color,
                s=100,
                marker='s',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            
            # End points
            ax.scatter(
                [suction_positions[-1, 0]],
                [suction_positions[-1, 1]],
                [suction_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[-1, 0]],
                [parallel_jaw_positions[-1, 1]],
                [parallel_jaw_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
        
        # Optionally show grippers at start/end
        if show_grippers:
            # Start grippers
            suction_start_T = suction_traj[0]
            parallel_jaw_start_T = parallel_jaw_traj[0]
            
            draw_suction_gripper(
                ax,
                suction_start_T[:3, 3],
                suction_start_T[:3, :3],
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
            draw_parallel_jaw_gripper(
                ax,
                parallel_jaw_start_T[:3, 3],
                parallel_jaw_start_T[:3, :3],
                gripper_opening=2.2,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'End Effector Trajectories Visualization\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Total trajectory pairs: {len(all_trajectories)} (showing {len(trajectories_to_show)})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        max_range *= 1.2  # Add padding
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    legend_elements = []
    
    # Add object trajectory to legend if it was drawn
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', label='Object trajectory')
        )
        if show_start_end:
            legend_elements.extend([
                plt.Line2D([0], [0], color='black', marker='D', linestyle='None', markersize=10, markeredgecolor='white', markeredgewidth=2, label='Object start'),
                plt.Line2D([0], [0], color='black', marker='X', linestyle='None', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Object end'),
            ])
    
    legend_elements.extend([
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='o', linestyle='-', label='Suction gripper trajectories'),
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='s', linestyle='-', label='Parallel jaw gripper trajectories'),
    ])
    if show_start_end:
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Gripper start (circle)'),
            plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10, label='Gripper end (star)'),
        ])
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_feasible_grasps(
    feasible_pairs: list = None,
    infeasible_pairs: list = None,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_pairs_to_show: int = 50,
    max_infeasible_to_show: int = 50,
    show_cuboids: bool = True,
    show_frames: bool = False,
    frame_scale: float = 1.0,
    save_path: str = None,
):
    """
    Visualize both feasible and infeasible grasp pairs.
    
    Args:
        feasible_pairs: List of (suction_grasp, parallel_jaw_grasp) tuples
        infeasible_pairs: List of (suction_grasp, parallel_jaw_grasp, reason) tuples
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_pairs_to_show: Maximum number of feasible pairs to visualize
        max_infeasible_to_show: Maximum number of infeasible pairs to visualize
        show_cuboids: If True, show gripper cuboids; if False, show detailed grippers
        show_frames: If True, show SE3 coordinate frames (x-axis = approach direction)
        frame_scale: Scale factor for frame axes
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if feasible_pairs is None:
        feasible_pairs = []
    if infeasible_pairs is None:
        infeasible_pairs = []
    
    if not feasible_pairs and not infeasible_pairs:
        print("No pairs to visualize!")
        return None, None
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Limit number of pairs to show
    feasible_to_show = feasible_pairs[:max_pairs_to_show] if max_pairs_to_show and feasible_pairs else feasible_pairs
    infeasible_to_show = infeasible_pairs[:max_infeasible_to_show] if max_infeasible_to_show and infeasible_pairs else infeasible_pairs
    
    if len(feasible_pairs) > max_pairs_to_show:
        print(f"Showing {max_pairs_to_show} of {len(feasible_pairs)} feasible pairs")
    if len(infeasible_pairs) > max_infeasible_to_show:
        print(f"Showing {max_infeasible_to_show} of {len(infeasible_pairs)} infeasible pairs")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw infeasible grasps first (so feasible ones appear on top)
    for suction_grasp, parallel_jaw_grasp, reason in infeasible_to_show:
        all_positions.append(suction_grasp.position)
        all_positions.append(parallel_jaw_grasp.position)
        
        if show_frames:
            # Draw as SE3 frames (infeasible - use thinner lines)
            draw_se3_frame(
                ax, suction_grasp.position, suction_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.0
            )
            draw_se3_frame(
                ax, parallel_jaw_grasp.position, parallel_jaw_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.0
            )
        else:
            # Determine colors based on collision reason
            if reason == "plate_collision_suction":
                suction_color = 'red'
                parallel_jaw_color = 'darkred'
            elif reason == "plate_collision_parallel_jaw":
                suction_color = 'darkred'
                parallel_jaw_color = 'red'
            else:  # gripper_collision
                suction_color = 'crimson'
                parallel_jaw_color = 'crimson'
            
            if show_cuboids:
                # Draw as cuboids with lower alpha for infeasible
                draw_gripper_cuboid(
                    ax, SUCTION_GEOMETRY, suction_grasp.position, suction_grasp.orientation,
                    color=suction_color, alpha=0.2
                )
                draw_gripper_cuboid(
                    ax, PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp.position, parallel_jaw_grasp.orientation,
                    color=parallel_jaw_color, alpha=0.2
                )
            else:
                # Draw detailed grippers (simplified for infeasible)
                draw_gripper_cuboid(
                    ax, SUCTION_GEOMETRY, suction_grasp.position, suction_grasp.orientation,
                    color=suction_color, alpha=0.2
                )
                draw_gripper_cuboid(
                    ax, PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp.position, parallel_jaw_grasp.orientation,
                    color=parallel_jaw_color, alpha=0.2
                )
    
    # Draw feasible grasps
    for suction_grasp, parallel_jaw_grasp in feasible_to_show:
        all_positions.append(suction_grasp.position)
        all_positions.append(parallel_jaw_grasp.position)
        
        if show_frames:
            # Draw as SE3 frames with x-axis in approach direction
            draw_se3_frame(
                ax, suction_grasp.position, suction_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.5
            )
            draw_se3_frame(
                ax, parallel_jaw_grasp.position, parallel_jaw_grasp.approach_direction,
                frame_scale=frame_scale, linewidth=1.5
            )
        elif show_cuboids:
            # Draw as cuboids (faster, clearer for collision visualization)
            draw_gripper_cuboid(
                ax, SUCTION_GEOMETRY, suction_grasp.position, suction_grasp.orientation,
                color='orange', alpha=0.5
            )
            draw_gripper_cuboid(
                ax, PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp.position, parallel_jaw_grasp.orientation,
                color='steelblue', alpha=0.5
            )
        else:
            # Draw detailed grippers
            draw_suction_gripper(
                ax,
                suction_grasp.position,
                suction_grasp.orientation,
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
            draw_parallel_jaw_gripper(
                ax,
                parallel_jaw_grasp.position,
                parallel_jaw_grasp.orientation,
                parallel_jaw_grasp.gripper_opening,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'Grasp Pairs Visualization - Feasible & Infeasible (Plate Frame - Centroid at Origin)\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Feasible: {len(feasible_pairs)} (showing {len(feasible_to_show)}), '
    title += f'Infeasible: {len(infeasible_pairs)} (showing {len(infeasible_to_show)})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        max_range *= 1.2  # Add padding
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    legend_elements = []
    if show_frames:
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linewidth=2, label='X-axis (approach)'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Y-axis'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Z-axis'),
        ])
    else:
        if feasible_to_show:
            legend_elements.extend([
                plt.Line2D([0], [0], color='orange', linewidth=4, label='Suction grippers (feasible)'),
                plt.Line2D([0], [0], color='steelblue', linewidth=4, label='Parallel jaw grippers (feasible)'),
            ])
        if infeasible_to_show:
            legend_elements.extend([
                plt.Line2D([0], [0], color='red', linewidth=4, alpha=0.5, label='Infeasible (collision)'),
            ])
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_trajectories(
    all_trajectories: list,
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_trajectories_to_show: int = 50,
    show_grippers: bool = False,
    trajectory_linewidth: float = 1.5,
    trajectory_alpha: float = 0.6,
    show_start_end: bool = True,
    save_path: str = None,
):
    """
    Visualize end effector trajectories for all feasible grasp pairs.
    
    Args:
        all_trajectories: List of trajectory dictionaries, each containing:
            - 'suction_grasp': suction grasp pose
            - 'parallel_jaw_grasp': parallel jaw grasp pose
            - 'suction_trajectory': list of 4x4 matrices for suction gripper
            - 'parallel_jaw_trajectory': list of 4x4 matrices for parallel jaw gripper
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_trajectories_to_show: Maximum number of trajectory pairs to visualize
        show_grippers: If True, show grippers at start/end positions
        trajectory_linewidth: Line width for trajectory paths
        trajectory_alpha: Alpha transparency for trajectory paths
        show_start_end: If True, mark start and end points
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not all_trajectories:
        print("No trajectories to visualize!")
        return None, None
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limit number of trajectories to show
    trajectories_to_show = all_trajectories[:max_trajectories_to_show] if max_trajectories_to_show else all_trajectories
    if len(all_trajectories) > max_trajectories_to_show:
        print(f"Showing {max_trajectories_to_show} of {len(all_trajectories)} trajectory pairs")
    
    # Collect all positions for axis limits
    all_positions = []
    
    # Draw object trajectory first (so it appears behind other trajectories)
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        object_traj = trajectories_to_show[0]['object_trajectory']
        object_positions = np.array([T[:3, 3] for T in object_traj])
        all_positions.extend(object_positions)
        
        # Draw the plate at each waypoint with lighter color
        for T_object in object_traj:
            draw_plate(
                ax, 
                plate_width, 
                plate_length, 
                plate_thickness, 
                transform=T_object,
                alpha=0.15,  # Lighter/more transparent
                facecolor='lightgray'  # Lighter color
            )
        
        # Draw object trajectory with distinctive style (thick, dark, dashed)
        ax.plot(
            object_positions[:, 0],
            object_positions[:, 1],
            object_positions[:, 2],
            '--',
            color='black',
            linewidth=trajectory_linewidth * 2.5,
            alpha=0.9,
            markersize=5,
            label='Object trajectory',
            zorder=1  # Draw behind other trajectories
        )
        
        # Mark object start and end points
        if show_start_end:
            ax.scatter(
                [object_positions[0, 0]],
                [object_positions[0, 1]],
                [object_positions[0, 2]],
                c='black',
                s=150,
                marker='D',  # Diamond for object start
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object start'
            )
            ax.scatter(
                [object_positions[-1, 0]],
                [object_positions[-1, 1]],
                [object_positions[-1, 2]],
                c='black',
                s=200,
                marker='X',  # X for object end
                edgecolors='white',
                linewidths=2,
                alpha=1.0,
                zorder=6,
                label='Object end'
            )
    
    # Color cycle for different trajectories
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(trajectories_to_show)))
    
    # Draw trajectories
    for idx, traj_data in enumerate(trajectories_to_show):
        suction_traj = traj_data['suction_trajectory']
        parallel_jaw_traj = traj_data['parallel_jaw_trajectory']
        color = colors[idx % len(colors)]
        
        # Extract positions from transformation matrices
        suction_positions = np.array([T[:3, 3] for T in suction_traj])
        parallel_jaw_positions = np.array([T[:3, 3] for T in parallel_jaw_traj])
        
        all_positions.extend(suction_positions)
        all_positions.extend(parallel_jaw_positions)
        
        # Draw suction gripper trajectory
        ax.plot(
            suction_positions[:, 0],
            suction_positions[:, 1],
            suction_positions[:, 2],
            'o-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Suction {idx+1}' if idx < 10 else None
        )
        
        # Draw parallel jaw gripper trajectory
        ax.plot(
            parallel_jaw_positions[:, 0],
            parallel_jaw_positions[:, 1],
            parallel_jaw_positions[:, 2],
            's-',
            color=color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            markersize=3,
            label=f'Parallel jaw {idx+1}' if idx < 10 else None
        )
        
        # Mark start and end points
        if show_start_end:
            # Start points
            ax.scatter(
                [suction_positions[0, 0]],
                [suction_positions[0, 1]],
                [suction_positions[0, 2]],
                c=color,
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[0, 0]],
                [parallel_jaw_positions[0, 1]],
                [parallel_jaw_positions[0, 2]],
                c=color,
                s=100,
                marker='s',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            
            # End points
            ax.scatter(
                [suction_positions[-1, 0]],
                [suction_positions[-1, 1]],
                [suction_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.scatter(
                [parallel_jaw_positions[-1, 0]],
                [parallel_jaw_positions[-1, 1]],
                [parallel_jaw_positions[-1, 2]],
                c=color,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=5
            )
        
        # Optionally show grippers at start/end
        if show_grippers:
            # Start grippers
            suction_start_T = suction_traj[0]
            parallel_jaw_start_T = parallel_jaw_traj[0]
            
            draw_suction_gripper(
                ax,
                suction_start_T[:3, 3],
                suction_start_T[:3, :3],
                suction_cup_radius=1.0,
                cup_height=0.5,
                base_radius=0.8,
                base_height=1.0,
            )
            draw_parallel_jaw_gripper(
                ax,
                parallel_jaw_start_T[:3, 3],
                parallel_jaw_start_T[:3, :3],
                gripper_opening=2.2,
                gripper_body_length=2.0,
                jaw_length=2.5,
                jaw_width=0.3,
                jaw_height=0.8,
            )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'End Effector Trajectories Visualization\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Total trajectory pairs: {len(all_trajectories)} (showing {len(trajectories_to_show)})'
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        max_range *= 1.2  # Add padding
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-5, 5)
    
    # Add legend
    legend_elements = []
    
    # Add object trajectory to legend if it was drawn
    if trajectories_to_show and 'object_trajectory' in trajectories_to_show[0]:
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', label='Object trajectory')
        )
        if show_start_end:
            legend_elements.extend([
                plt.Line2D([0], [0], color='black', marker='D', linestyle='None', markersize=10, markeredgecolor='white', markeredgewidth=2, label='Object start'),
                plt.Line2D([0], [0], color='black', marker='X', linestyle='None', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Object end'),
            ])
    
    legend_elements.extend([
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='o', linestyle='-', label='Suction gripper trajectories'),
        plt.Line2D([0], [0], color='gray', linewidth=2, marker='s', linestyle='-', label='Parallel jaw gripper trajectories'),
    ])
    if show_start_end:
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Gripper start (circle)'),
            plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10, label='Gripper end (star)'),
        ])
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax
