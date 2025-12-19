#!/usr/bin/env python3
"""
Debug and visualization demo for the perception system.
Shows FastSAM object detection and GraspGen grasp prediction with full visualization.
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.services import ServiceManager, UnifiedPerceptionService, ServiceClient

def main():
    print("=== Dual-Arm Franka Perception Debug Demo (Service-Based) ===")
    print("This demo shows FastSAM detection and GraspGen grasp prediction with visualization, using the service architecture.")
    print()

    # 1. Start services
    print("1. Starting services...")
    manager = ServiceManager()
    manager.start()

    env = make_dual_arm_env()
    env.reset()
    
    perception_service = UnifiedPerceptionService(env, debug=True, visualize=True)
    perception_service.start()
    
    time.sleep(2) # Wait for services to start and for the first perception loop

    # 2. Create a service client
    service_client = ServiceClient()
    
    # 3. Run object detection via service
    print("\n2. Getting latest objects from perception service...")
    print("   (Visualization windows should appear if enabled in the service)")
    
    response = service_client.call("perception_service", {"command": "GET_LATEST_OBJECTS"})
    if not response or response.get("status") != "success" or not response.get("objects"):
        print("No objects detected!")
        objects = []
    else:
        objects = response["objects"]

    print(f"✓ Detected {len(objects)} objects")

    # 4. Show details of detected objects
    for i, obj in enumerate(objects):
        print(f"\n   Object {i}:")
        print(f"   - Pose: {obj['pose']}")
        print(f"   - Grasps: {len(obj['grasps'])} predictions")

        if obj['grasps']:
            best_grasp = max(obj['grasps'], key=lambda g: g['score'])
            print(f"   - Best grasp score: {best_grasp['score']:.4f}")

    print("\n=== Demo Complete ===")
    print("Press Ctrl+C to exit. Close visualization windows manually.")

    try:
        # Keep the script alive to see visualizations
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up...")
    finally:
        perception_service.stop()
        manager.stop()
        env.close()

if __name__ == "__main__":
    main()