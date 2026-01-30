#!/usr/bin/env python3
"""
Script to visualize end effector trajectories for all feasible grasp pairs.

This script:
1. Loads trajectory data from dummy_trajectory.json
2. Loads or generates feasible grasp pairs
3. Generates end effector trajectories for all pairs
4. Visualizes all trajectories
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bimanual.grasp_generator import PlateGraspGenerator, GraspPose
from bimanual.suction_grasp_generator import SuctionGraspGenerator
from bimanual.collision_checker import find_feasible_grasp_pairs
from bimanual.trajectory_generator import generate_trajectories_for_all_pairs
from bimanual.visualization_utils import visualize_trajectories, HAS_MATPLOTLIB


def load_grasps_from_file(filename: str) -> list:
    """Load grasps from a numpy file."""
    if not os.path.exists(filename):
        return None
    
    data = np.load(filename, allow_pickle=True)
    # Reconstruct GraspPose objects
    grasps = []
    positions = data['positions']
    orientations = data['orientations']
    approach_directions = data['approach_directions']
    gripper_openings = data['gripper_openings']
    
    for i in range(len(positions)):
        grasp = GraspPose(
            position=positions[i],
            orientation=orientations[i],
            approach_direction=approach_directions[i],
            gripper_opening=gripper_openings[i],
        )
        grasps.append(grasp)
    
    return grasps


def main():
    """Main function to generate and visualize trajectories."""
    print("End Effector Trajectory Visualization")
    print("=" * 60)
    
    plate_width = 10.0
    plate_length = 20.0
    plate_thickness = 2.0
    
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    trajectory_filepath = os.path.join(data_dir, 'dummy_trajectory.json')
    parallel_jaw_file = os.path.join(data_dir, 'parallel_jaw_grasps.npz')
    suction_file = os.path.join(data_dir, 'suction_grasps.npz')
    
    # Try to load existing grasps, otherwise generate them
    print("\nLoading or generating grasps...")
    
    # Load or generate parallel jaw grasps
    parallel_jaw_grasps = load_grasps_from_file(parallel_jaw_file)
    if parallel_jaw_grasps is None:
        print("Generating parallel jaw grasps...")
        parallel_jaw_generator = PlateGraspGenerator(
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            max_gripper_opening=5.0,
            grasp_offset=1.0,
        )
        parallel_jaw_grasps = parallel_jaw_generator.generate_all_grasps(
            num_grasps_per_edge=5,
            include_corners=True,
        )
        print(f"Generated {len(parallel_jaw_grasps)} parallel jaw grasps")
    else:
        print(f"Loaded {len(parallel_jaw_grasps)} parallel jaw grasps from file")
    
    # Load or generate suction grasps
    suction_grasps = load_grasps_from_file(suction_file)
    if suction_grasps is None:
        print("Generating suction grasps...")
        suction_generator = SuctionGraspGenerator(
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            suction_cup_radius=1.0,
            min_distance_from_edge=1.5,
        )
        suction_grasps = suction_generator.generate_all_grasps(
            num_grasps_x=5,
            num_grasps_y=5,
            include_top=True,
            include_bottom=True,
        )
        print(f"Generated {len(suction_grasps)} suction grasps")
    else:
        print(f"Loaded {len(suction_grasps)} suction grasps from file")
    
    # Find feasible pairs
    print("\n" + "=" * 60)
    print("Filtering feasible grasp pairs...")
    print("=" * 60)
    
    feasible_pairs = find_feasible_grasp_pairs(
        suction_grasps,
        parallel_jaw_grasps,
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        margin=0.5
    )
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total suction grasps: {len(suction_grasps)}")
    print(f"  Total parallel jaw grasps: {len(parallel_jaw_grasps)}")
    print(f"  Feasible pairs: {len(feasible_pairs)}")
    print(f"{'='*60}\n")
    
    # Generate trajectories
    print("=" * 60)
    print("Generating end effector trajectories...")
    print("=" * 60)
    
    all_trajectories = generate_trajectories_for_all_pairs(
        trajectory_filepath,
        feasible_pairs,
    )
    
    print(f"\n{'='*60}")
    print(f"Trajectory Generation Summary:")
    print(f"  Total trajectory pairs: {len(all_trajectories)}")
    if all_trajectories:
        print(f"  Waypoints per trajectory: {len(all_trajectories[0]['suction_trajectory'])}")
    print(f"{'='*60}\n")
    
    # Visualize trajectories
    if HAS_MATPLOTLIB and all_trajectories:
        print("=" * 60)
        print("Visualizing trajectories...")
        print("=" * 60)
        
        vis_save_path = os.path.join(data_dir, 'trajectories_visualization.png')
        visualize_trajectories(
            all_trajectories=all_trajectories,
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            max_trajectories_to_show=200,  # Show up to 50 trajectory pairs
            show_grippers=False,  # Set to True to show grippers at start/end
            trajectory_linewidth=1.5,
            trajectory_alpha=0.6,
            show_start_end=True,
            save_path=vis_save_path,
        )
    elif not all_trajectories:
        print("No trajectories to visualize!")
    else:
        print("\nWarning: matplotlib not available. Skipping visualization.")
        print("Install matplotlib to enable visualization: pip install matplotlib")
    # Save all_trajectories to a JSON file for later analysis or visualization
    import json
    def serialize_trajectory(obj):
        # Serialize numpy arrays to lists for JSON output
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # For GraspPose or similar, serialize their __dict__:
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    trajectories_save_path = os.path.join(data_dir, 'all_trajectories.json')
    try:
        with open(trajectories_save_path, 'w') as f:
            json.dump(all_trajectories, f, indent=2, default=serialize_trajectory)
        print(f"Saved all trajectories to {trajectories_save_path}")
    except Exception as e:
        print(f"Error saving trajectories: {e}")
    return all_trajectories


if __name__ == "__main__":
    main()
