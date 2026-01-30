#!/usr/bin/env python3
"""
Script to check trajectory feasibility against obstacles in the environment.

This script:
1. Reads all trajectories from all_trajectories.json
2. Accepts a set of obstacles (Cuboids) in the environment
3. Represents trajectories as a gripper moving in 3D space
4. Returns a feasibility matrix indicating which segments are feasible
5. Tracks which grasp is associated with each trajectory and its type (suction or parallel jaw)
6. Visualizes feasibility matrices, trajectories, and obstacles
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robosuite_franka_manipulation.planning.feasibility_checker import (
    Cuboid,
    check_trajectory_feasibility,
)
from robosuite_franka_manipulation.planning.visualization_utils import (
    HAS_MATPLOTLIB,
    draw_plate,
    draw_gripper_cuboid,
)
from robosuite_franka_manipulation.planning.collision_checker import (
    SUCTION_GEOMETRY,
    PARALLEL_JAW_GEOMETRY,
)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_trajectories(filepath: str) -> List[Dict[str, Any]]:
    """
    Load trajectories from JSON file.
    
    Args:
        filepath: Path to JSON file containing trajectory data
    
    Returns:
        List of trajectory dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def check_all_trajectories(
    trajectories_filepath: str,
    obstacles: List[Cuboid],
    num_samples_per_segment: int = 10,
    margin: float = 0.1,
) -> Dict[str, Any]:
    """
    Check feasibility of all trajectories against obstacles.
    
    Args:
        trajectories_filepath: Path to JSON file containing all trajectories
        obstacles: List of obstacle cuboids
        num_samples_per_segment: Number of samples to check per trajectory segment
        margin: Safety margin for collision detection (cm)
    
    Returns:
        Dictionary containing:
        - 'feasibility_matrices': List of feasibility matrices (one per trajectory)
        - 'metadata': List of metadata dictionaries (one per trajectory)
        - 'trajectory_info': List of trajectory indices and gripper types
    """
    print("Loading trajectories...")
    all_trajectories = load_trajectories(trajectories_filepath)
    print(f"Loaded {len(all_trajectories)} trajectory pairs")
    
    feasibility_matrices = []
    metadata_list = []
    trajectory_info = []
    
    print("\nChecking trajectory feasibility...")
    total_checks = len(all_trajectories) * 2  # Each pair has suction and parallel_jaw
    
    check_count = 0
    for traj_idx, trajectory_data in enumerate(all_trajectories):
        # Check suction trajectory
        suction_feasibility, suction_metadata = check_trajectory_feasibility(
            trajectory_data,
            obstacles,
            'suction',
            num_samples_per_segment,
            margin,
        )
        feasibility_matrices.append(suction_feasibility)
        metadata_list.append(suction_metadata)
        trajectory_info.append({
            'trajectory_index': traj_idx,
            'gripper_type': 'suction',
        })
        check_count += 1
        
        # Check parallel jaw trajectory
        parallel_jaw_feasibility, parallel_jaw_metadata = check_trajectory_feasibility(
            trajectory_data,
            obstacles,
            'parallel_jaw',
            num_samples_per_segment,
            margin,
        )
        feasibility_matrices.append(parallel_jaw_feasibility)
        metadata_list.append(parallel_jaw_metadata)
        trajectory_info.append({
            'trajectory_index': traj_idx,
            'gripper_type': 'parallel_jaw',
        })
        check_count += 1
        
        if check_count % 10 == 0:
            print(f"  Processed {check_count}/{total_checks} trajectories...")
    
    print(f"\nCompleted feasibility checking for {check_count} trajectories")
    
    return {
        'feasibility_matrices': feasibility_matrices,
        'metadata': metadata_list,
        'trajectory_info': trajectory_info,
        'obstacles': obstacles,
        'all_trajectories': all_trajectories,
    }


def draw_obstacle(ax, obstacle: Cuboid, color: str = 'red', alpha: float = 0.5):
    """Draw an obstacle cuboid."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    
    half_dims = obstacle.dimensions / 2.0
    local_vertices = np.array([
        [-half_dims[0], -half_dims[1], -half_dims[2]],
        [half_dims[0], -half_dims[1], -half_dims[2]],
        [half_dims[0], half_dims[1], -half_dims[2]],
        [-half_dims[0], half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1], half_dims[2]],
        [half_dims[0], -half_dims[1], half_dims[2]],
        [half_dims[0], half_dims[1], half_dims[2]],
        [-half_dims[0], half_dims[1], half_dims[2]],
    ])
    
    # Transform to world frame
    world_vertices = obstacle.center + (obstacle.orientation @ local_vertices.T).T
    
    # Define faces
    faces = [
        [world_vertices[0], world_vertices[1], world_vertices[2], world_vertices[3]],  # bottom
        [world_vertices[4], world_vertices[5], world_vertices[6], world_vertices[7]],  # top
        [world_vertices[0], world_vertices[1], world_vertices[5], world_vertices[4]],  # front
        [world_vertices[2], world_vertices[3], world_vertices[7], world_vertices[6]],  # back
        [world_vertices[0], world_vertices[3], world_vertices[7], world_vertices[4]],  # left
        [world_vertices[1], world_vertices[2], world_vertices[6], world_vertices[5]],  # right
    ]
    
    cuboid = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='darkred', linewidths=1.5)
    ax.add_collection3d(cuboid)


def visualize_feasibility_matrices(
    results: Dict[str, Any],
    max_trajectories_to_show: int = 20,
    save_path: str = None,
):
    """
    Visualize feasibility matrices as a heatmap.
    
    Args:
        results: Results dictionary from check_all_trajectories
        max_trajectories_to_show: Maximum number of trajectories to visualize
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Skipping visualization.")
        return None, None
    
    feasibility_matrices = results['feasibility_matrices']
    trajectory_info = results['trajectory_info']
    
    # Limit number of trajectories
    num_to_show = min(max_trajectories_to_show, len(feasibility_matrices))
    matrices_to_show = feasibility_matrices[:num_to_show]
    info_to_show = trajectory_info[:num_to_show]
    
    # Find maximum number of segments
    max_segments = max(len(feas) for feas in matrices_to_show) if matrices_to_show else 0
    
    # Create matrix for heatmap (trajectories x segments)
    heatmap_data = np.zeros((len(matrices_to_show), max_segments))
    for i, feas in enumerate(matrices_to_show):
        heatmap_data[i, :len(feas)] = feas.astype(float)
        # Pad with NaN for shorter trajectories
        if len(feas) < max_segments:
            heatmap_data[i, len(feas):] = np.nan
    
    fig, ax = plt.subplots(figsize=(14, max(8, num_to_show * 0.3)))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Trajectory Segment', fontsize=12)
    ax.set_ylabel('Trajectory Index', fontsize=12)
    ax.set_title('Trajectory Feasibility Matrix\n(Green=Feasible, Red=Infeasible)', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feasibility (1=Feasible, 0=Infeasible)', fontsize=10)
    
    # Add trajectory type labels on y-axis
    y_labels = []
    for info in info_to_show:
        label = f"{info['trajectory_index']} ({info['gripper_type'][:1]})"
        y_labels.append(label)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, max_segments, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrices_to_show), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feasibility matrix visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def visualize_trajectories_with_feasibility(
    results: Dict[str, Any],
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    max_trajectories_to_show: int = 20,
    save_path: str = None,
):
    """
    Visualize trajectories colored by feasibility, with obstacles.
    
    Args:
        results: Results dictionary from check_all_trajectories
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        max_trajectories_to_show: Maximum number of trajectories to visualize
        save_path: If provided, save the figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Skipping visualization.")
        return None, None
    
    feasibility_matrices = results['feasibility_matrices']
    trajectory_info = results['trajectory_info']
    obstacles = results['obstacles']
    all_trajectories = results['all_trajectories']
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the plate
    draw_plate(ax, plate_width, plate_length, plate_thickness)
    
    # Draw obstacles
    for i, obstacle in enumerate(obstacles):
        draw_obstacle(ax, obstacle, color='red', alpha=0.1)
    
    # Limit number of trajectories
    num_to_show = min(max_trajectories_to_show, len(feasibility_matrices))
    
    all_positions = []
    
    # Draw trajectories colored by feasibility
    for idx in range(num_to_show):
        feas = feasibility_matrices[idx]
        info = trajectory_info[idx]
        traj_idx = info['trajectory_index']
        gripper_type = info['gripper_type']
        
        # Get trajectory data
        traj_data = all_trajectories[traj_idx]
        if gripper_type == 'suction':
            trajectory = traj_data['suction_trajectory']
            gripper_geometry = SUCTION_GEOMETRY
        else:
            trajectory = traj_data['parallel_jaw_trajectory']
            gripper_geometry = PARALLEL_JAW_GEOMETRY
        
        # Convert to numpy arrays
        trajectory_matrices = [np.array(T) for T in trajectory]
        positions = np.array([T[:3, 3] for T in trajectory_matrices])
        all_positions.extend(positions)
        
        # Draw trajectory segments colored by feasibility
        for seg_idx in range(len(positions) - 1):
            start_pos = positions[seg_idx]
            end_pos = positions[seg_idx + 1]
            
            # Color based on feasibility
            if seg_idx < len(feas) and feas[seg_idx]:
                color = 'green'
                alpha = 0.8
                linewidth = 2.0
            else:
                color = 'red'
                alpha = 0.6
                linewidth = 2.5
            
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        
        # Mark start and end points
        ax.scatter(
            [positions[0, 0]],
            [positions[0, 1]],
            [positions[0, 2]],
            c='blue',
            s=100,
            marker='o',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9,
            zorder=5
        )
        ax.scatter(
            [positions[-1, 0]],
            [positions[-1, 1]],
            [positions[-1, 2]],
            c='purple',
            s=150,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9,
            zorder=5
        )
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    title = f'Trajectories with Feasibility Visualization\n'
    title += f'Plate: {plate_width}×{plate_length}×{plate_thickness} cm\n'
    title += f'Showing {num_to_show} trajectories (Green=Feasible, Red=Infeasible)'
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
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=3, label='Feasible segments'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Infeasible segments'),
        plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8, label='Start'),
        plt.Line2D([0], [0], color='purple', marker='*', linestyle='None', markersize=10, label='End'),
        plt.Line2D([0], [0], color='red', linewidth=4, alpha=0.6, label='Obstacles'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save feasibility results to a JSON file.
    
    Args:
        results: Results dictionary from check_all_trajectories
        filepath: Path to save the results
    """
    # Convert numpy arrays to lists for JSON serialization
    # Note: We don't save all_trajectories as it's large
    serializable_results = {
        'feasibility_matrices': [feas.tolist() for feas in results['feasibility_matrices']],
        'metadata': results['metadata'],
        'trajectory_info': results['trajectory_info'],
        'obstacles': [
            {
                'center': obs.center.tolist(),
                'dimensions': obs.dimensions.tolist(),
                'orientation': obs.orientation.tolist(),
            }
            for obs in results['obstacles']
        ],
    }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load feasibility results from a JSON file.
    
    Args:
        filepath: Path to load the results from
    
    Returns:
        Results dictionary with numpy arrays restored
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Restore numpy arrays
    results = {
        'feasibility_matrices': [np.array(feas) for feas in data['feasibility_matrices']],
        'metadata': data['metadata'],
        'trajectory_info': data['trajectory_info'],
        'obstacles': [
            Cuboid(
                center=np.array(obs['center']),
                dimensions=np.array(obs['dimensions']),
                orientation=np.array(obs['orientation']),
            )
            for obs in data['obstacles']
        ],
    }
    
    return results


def main():
    """Example usage of the trajectory feasibility checker."""
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    trajectories_filepath = os.path.join(data_dir, 'all_trajectories.json')
    
    # Define example obstacles (cuboids in the environment)
    # Format: center position [x, y, z], dimensions [width, length, height]
    obstacles = [
        Cuboid(
            center=np.array([40.0, -45.0, 63.0]),
            dimensions=np.array([30.0, 10.0, 5.0]),  # 10cm x 10cm x 5cm obstacle
        ),
        Cuboid(
            center=np.array([40.0, -45.0, 55.0]),
            dimensions=np.array([30.0, 10.0, 5.0]),  # 10cm x 10cm x 5cm obstacle
        ),
        Cuboid(
            center=np.array([0.0, 0.0, -1.0]),
            dimensions=np.array([100.0, 100.0, 2.0]),  # 3cm x 3cm x 6cm obstacle
        ),
        # Add more obstacles as needed
    ]
    
    print("Trajectory Feasibility Checker")
    print("=" * 60)
    print(f"Number of obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: center={obs.center}, dimensions={obs.dimensions}")
    print()
    
    # Check all trajectories
    results = check_all_trajectories(
        trajectories_filepath,
        obstacles,
        num_samples_per_segment=2,
        margin=0.1,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Feasibility Summary")
    print("=" * 60)
    
    total_trajectories = len(results['feasibility_matrices'])
    fully_feasible = 0
    partially_feasible = 0
    fully_infeasible = 0
    
    for i, feasibility in enumerate(results['feasibility_matrices']):
        num_feasible_segments = np.sum(feasibility)
        num_segments = len(feasibility)
        
        if num_feasible_segments == num_segments:
            fully_feasible += 1
        elif num_feasible_segments == 0:
            fully_infeasible += 1
        else:
            partially_feasible += 1
    
    print(f"Total trajectories checked: {total_trajectories}")
    print(f"  Fully feasible: {fully_feasible}")
    print(f"  Partially feasible: {partially_feasible}")
    print(f"  Fully infeasible: {fully_infeasible}")
    print()
    
    # Show some examples
    print("Example results (first 5 trajectories):")
    for i in range(min(5, total_trajectories)):
        feasibility = results['feasibility_matrices'][i]
        info = results['trajectory_info'][i]
        metadata = results['metadata'][i]
        
        num_feasible = np.sum(feasibility)
        num_segments = len(feasibility)
        
        print(f"\n  Trajectory {i+1}:")
        print(f"    Type: {info['gripper_type']}")
        print(f"    Trajectory pair index: {info['trajectory_index']}")
        print(f"    Feasible segments: {num_feasible}/{num_segments}")
        print(f"    Feasibility: {feasibility.tolist()}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("Creating visualizations...")
        print("=" * 60)
        
        # Visualize feasibility matrices
        feasibility_vis_path = os.path.join(data_dir, 'feasibility_matrices_visualization.png')
        visualize_feasibility_matrices(
            results,
            max_trajectories_to_show=50,
            save_path=feasibility_vis_path,
        )
        
        # Visualize trajectories with feasibility
        trajectory_vis_path = os.path.join(data_dir, 'trajectories_feasibility_visualization.png')
        visualize_trajectories_with_feasibility(
            results,
            plate_width=10.0,
            plate_length=20.0,
            plate_thickness=2.0,
            max_trajectories_to_show=20,
            save_path=trajectory_vis_path,
        )
    else:
        print("\nWarning: matplotlib not available. Skipping visualization.")
        print("Install matplotlib to enable visualization: pip install matplotlib")
    
    # Optionally save results
    results_filepath = os.path.join(data_dir, 'trajectory_feasibility_results.json')
    try:
        save_results(results, results_filepath)
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    results = main()
