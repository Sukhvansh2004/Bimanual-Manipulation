#!/usr/bin/env python3
"""
Script to visualize feasible grasp pairs after collision filtering.

This script:
1. Loads or generates grasps for both gripper types
2. Filters grasps that collide with the plate
3. Finds feasible pairs (no collision between grippers)
4. Visualizes only the feasible grasps
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robosuite_franka_manipulation.planning.grasp_generator import PlateGraspGenerator, GraspPose
from robosuite_franka_manipulation.planning.suction_grasp_generator import SuctionGraspGenerator
from robosuite_franka_manipulation.planning.collision_checker import (
    find_feasible_grasp_pairs,
    find_infeasible_grasp_pairs,
    SUCTION_GEOMETRY,
    PARALLEL_JAW_GEOMETRY,
    PlateGeometry,
)
from robosuite_franka_manipulation.planning.visualization_utils import (
    draw_gripper_cuboid,
    draw_parallel_jaw_gripper,
    draw_suction_gripper,
    visualize_all_grasps,
    visualize_feasible_grasps,
    HAS_MATPLOTLIB,
)


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
    """Main function to filter and visualize feasible grasps."""
    print("Feasible Grasp Pairs Filtering and Visualization")
    print("=" * 60)
    
    plate_width = 10.0
    plate_length = 20.0
    plate_thickness = 2.0
    
    # Try to load existing grasps, otherwise generate them
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    parallel_jaw_file = os.path.join(data_dir, 'parallel_jaw_grasps.npz')
    suction_file = os.path.join(data_dir, 'suction_grasps.npz')
    
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
    # Find infeasible pairs
    print("\n" + "=" * 60)
    print("Finding infeasible grasp pairs...")
    print("=" * 60)
    
    infeasible_pairs = find_infeasible_grasp_pairs(
        suction_grasps,
        parallel_jaw_grasps,
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        margin=0.1,
    )
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total suction grasps: {len(suction_grasps)}")
    print(f"  Total parallel jaw grasps: {len(parallel_jaw_grasps)}")
    print(f"  Total possible pairs: {len(suction_grasps) * len(parallel_jaw_grasps)}")
    print(f"  Feasible pairs: {len(feasible_pairs)}")
    print(f"  Infeasible pairs: {len(infeasible_pairs)}")
    print(f"{'='*60}\n")
    
    # Create all pairs list (combining feasible and infeasible)
    all_pairs = []
    for pair in feasible_pairs:
        all_pairs.append(pair)
    for pair_info in infeasible_pairs:
        # infeasible_pairs are (suction, parallel_jaw, reason), extract just the pair
        all_pairs.append((pair_info[0], pair_info[1]))
    
    # Parse command line arguments for visualization mode
    import argparse
    parser = argparse.ArgumentParser(description='Visualize feasible grasp pairs')
    parser.add_argument(
        '--show-frames',
        action='store_true',
        default=False,
        help='Visualize grasps as SE3 coordinate frames (x-axis = approach direction)'
    )
    parser.add_argument(
        '--frame-scale',
        type=float,
        default=1.0,
        help='Scale factor for frame axes (default: 1.0)'
    )
    args, _ = parser.parse_known_args()  # Use parse_known_args to avoid errors with unknown args
    
    # Visualize all grasps first
    if HAS_MATPLOTLIB and all_pairs:
        print("=" * 60)
        print("Step 1: Visualizing ALL grasp pairs...")
        print("=" * 60)
        all_grasps_save_path = os.path.join(data_dir, 'all_grasps_visualization.png')
        visualize_all_grasps(
            all_pairs=all_pairs,
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            max_pairs_to_show=10,  # Show up to 200 pairs
            show_cuboids=not args.show_frames,
            show_frames=args.show_frames,
            frame_scale=args.frame_scale,
            save_path=all_grasps_save_path,
        )
        
        print("\n" + "=" * 60)
        print("Step 2: Visualizing Feasible vs Infeasible grasp pairs...")
        print("=" * 60)
        vis_save_path = os.path.join(data_dir, 'grasps_visualization_feasible_infeasible.png')
        visualize_feasible_grasps(
            feasible_pairs=feasible_pairs,
            infeasible_pairs=infeasible_pairs,
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            max_pairs_to_show=500,  # Show up to 50 feasible pairs
            max_infeasible_to_show=1,  # Show up to 100 infeasible pairs
            show_cuboids=not args.show_frames,  # Use cuboids for clearer collision visualization
            show_frames=args.show_frames,
            frame_scale=args.frame_scale,
            save_path=vis_save_path,
        )
    elif not all_pairs:
        print("No pairs found to visualize!")
    else:
        print("\nWarning: matplotlib not available. Skipping visualization.")
        print("Install matplotlib to enable visualization: pip install matplotlib")
    
    return feasible_pairs


if __name__ == "__main__":
    main()
