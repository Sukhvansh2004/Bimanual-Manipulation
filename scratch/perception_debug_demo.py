#!/usr/bin/env python3
"""
Debug and visualization demo for the perception system.
Shows FastSAM object detection and GraspGen grasp prediction with full visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robosuite as suite
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

def main():
    print("=== Dual-Arm Franka Perception Debug Demo ===")
    print("This demo shows FastSAM detection and GraspGen grasp prediction with visualization")
    print()

    # Create environment
    print("1. Creating environment...")
    env = suite.make(
        env_name='TwoArmPegInHole',
        robots=['Panda', 'Panda'],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview'],
        camera_heights=480,
        camera_widths=640,
    )

    # Reset to get initial observations
    obs = env.reset()
    print("✓ Environment created and reset")

    # Create perception components with debug and visualization enabled
    print("\n2. Creating perception components with debug+visualization...")
    detector = ObjectDetector(env, debug=True, visualize=True)
    predictor = GraspPredictor(debug=True, visualize=True)
    print("✓ Perception components created")

    # Run object detection
    print("\n3. Running object detection...")
    print("   (FastSAM visualization will appear in separate windows)")
    objects = detector.detect_objects(obs['agentview_image'])

    print(f"✓ Detected {len(objects)} objects")

    # Show details of detected objects
    for i, obj in enumerate(objects):
        print(f"\n   Object {i}:")
        print(f"   - Pose: {obj['pose']}")
        print(f"   - Point cloud: {len(obj['pcd'])} points")
        print(f"   - Grasps: {len(obj['grasps'])} predictions")

        if obj['grasps']:
            best_grasp = max(obj['grasps'], key=lambda g: g['score'])
            print(f"   - Best grasp score: {best_grasp['score']:.4f}")
    # Demonstrate direct grasp prediction if objects were found
    if objects:
        print("\n4. Demonstrating direct grasp prediction...")
        print("   (Point cloud and grasp visualizations will appear)")

        first_obj = objects[0]
        grasps = predictor.predict_grasps(
            first_obj['pcd'],
            first_obj['pose'],
            num_grasps=5
        )

        print(f"✓ Generated {len(grasps)} grasps for first object")

    print("\n=== Demo Complete ===")
    print("Close all matplotlib windows to exit")

if __name__ == "__main__":
    main()