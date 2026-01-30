#!/usr/bin/env python3
"""
Script to generate and visualize grasp poses for both parallel jaw and suction grippers.

Usage:
    python scripts/generate_grasps.py
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robosuite_franka_manipulation.planning.grasp_generator import PlateGraspGenerator, GraspPose
from robosuite_franka_manipulation.planning.suction_grasp_generator import SuctionGraspGenerator
from robosuite_franka_manipulation.planning.visualization_utils import (
    visualize_grasps,
    HAS_MATPLOTLIB,
)


def print_grasp_summary(grasps: list):
    """Print a summary of generated grasps."""
    print("=" * 60)
    print(f"Total grasps generated: {len(grasps)}")
    print("=" * 60)
    print()
    
    # Group by approach direction
    approach_groups = {}
    for grasp in grasps:
        approach_key = tuple(np.round(grasp.approach_direction, 2))
        if approach_key not in approach_groups:
            approach_groups[approach_key] = []
        approach_groups[approach_key].append(grasp)
    
    print(f"Number of unique approach directions: {len(approach_groups)}")
    print()
    
    # Print details for first few grasps
    print("Sample grasps (first 10):")
    print("-" * 60)
    for i, grasp in enumerate(grasps[:10]):
        print(f"\nGrasp {i+1}:")
        print(f"  Position (cm): [{grasp.position[0]:6.2f}, {grasp.position[1]:6.2f}, {grasp.position[2]:6.2f}]")
        print(f"  Approach direction: [{grasp.approach_direction[0]:5.2f}, {grasp.approach_direction[1]:5.2f}, {grasp.approach_direction[2]:5.2f}]")
        print(f"  Required gripper opening: {grasp.gripper_opening:.2f} cm")
        
        # Extract Euler angles for easier interpretation
        R = grasp.orientation
        # Using ZYX Euler angles
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        print(f"  Orientation (Euler ZYX, degrees): [{np.degrees(x):6.1f}, {np.degrees(y):6.1f}, {np.degrees(z):6.1f}]")


def save_grasps_to_file(grasps: list, filename: str = "grasp_poses.npz"):
    """Save grasps to a numpy file."""
    positions = np.array([g.position for g in grasps])
    orientations = np.array([g.orientation for g in grasps])
    approach_directions = np.array([g.approach_direction for g in grasps])
    gripper_openings = np.array([g.gripper_opening for g in grasps])
    
    np.savez(
        filename,
        positions=positions,
        orientations=orientations,
        approach_directions=approach_directions,
        gripper_openings=gripper_openings,
    )
    print(f"\nGrasps saved to {filename}")




def main():
    """Main function to generate and display grasps for both gripper types."""
    print("Plate Grasp Generator - Parallel Jaw & Suction Grippers")
    print("=" * 60)
    print("Plate dimensions: 10cm x 20cm x 2cm (width x length x thickness)")
    print("Frame at center of plate (centroid at origin)")
    print("Parallel jaw: Max opening 5cm, grasps from sides only")
    print("Suction: Grasps from top and bottom only")
    print()
    
    plate_width = 10.0
    plate_length = 20.0
    plate_thickness = 2.0
    
    # Create parallel jaw generator
    parallel_jaw_generator = PlateGraspGenerator(
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        max_gripper_opening=5.0,  # cm
        grasp_offset=1.0,      # cm - distance from plate edge
    )
    
    # Create suction gripper generator
    suction_generator = SuctionGraspGenerator(
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        suction_cup_radius=1.0,
        min_distance_from_edge=1.5,
    )
    
    # Generate parallel jaw grasps
    print("Generating parallel jaw grasp poses...")
    parallel_jaw_grasps = parallel_jaw_generator.generate_all_grasps(
        num_grasps_per_edge=5,
        include_corners=True,
    )
    print(f"Generated {len(parallel_jaw_grasps)} parallel jaw grasps")
    
    # Generate suction grasps
    print("Generating suction grasp poses...")
    suction_grasps = suction_generator.generate_all_grasps(
        num_grasps_x=5,
        num_grasps_y=5,
        include_top=True,
        include_bottom=True,
    )
    print(f"Generated {len(suction_grasps)} suction grasps")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Grasp Summary")
    print("=" * 60)
    print(f"Parallel jaw grasps: {len(parallel_jaw_grasps)}")
    print(f"Suction grasps: {len(suction_grasps)}")
    print(f"Total grasps: {len(parallel_jaw_grasps) + len(suction_grasps)}")
    print()
    
    # Save to files
    parallel_jaw_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'parallel_jaw_grasps.npz')
    suction_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'suction_grasps.npz')
    
    save_grasps_to_file(parallel_jaw_grasps, parallel_jaw_file)
    save_grasps_to_file(suction_grasps, suction_file)
    
    # Visualize grasps
    if HAS_MATPLOTLIB:
        print("Generating visualization...")
        vis_save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'grasp_visualization.png')
        visualize_grasps(
            parallel_jaw_grasps=parallel_jaw_grasps,
            suction_grasps=suction_grasps,
            plate_width=plate_width,
            plate_length=plate_length,
            plate_thickness=plate_thickness,
            show_approach_vectors=False,  # Set to True to show approach vectors
            show_grippers=True,
            approach_scale=2.0,
            max_grasps_to_show=None,  # Show all grasps
            save_path=vis_save_path,
        )
    else:
        print("\nWarning: matplotlib not available. Skipping visualization.")
        print("Install matplotlib to enable visualization: pip install matplotlib")
    
    return parallel_jaw_grasps, suction_grasps


if __name__ == "__main__":
    main()
