#!/usr/bin/env python3
"""
Demo script to launch and visualize the dual-arm Franka environment.

This script creates a dual-arm robosuite environment with two Panda robots
in an opposed configuration and allows you to visualize and interact with it.

Usage:
    python scripts/demos/launch_env.py
    python scripts/demos/launch_env.py --with-objects
    python scripts/demos/launch_env.py --random-actions
"""

import argparse
import time
import numpy as np

from robosuite_franka_manipulation.environments import make_dual_arm_env


def print_env_info(env):
    """Print information about the environment."""
    print("\n" + "=" * 60)
    print("DUAL-ARM FRANKA ENVIRONMENT INFO")
    print("=" * 60)
    
    # Environment info
    print(f"\nEnvironment: {env.name if hasattr(env, 'name') else 'TwoArmHandover'}")
    print(f"Number of robots: {len(env.robots)}")
    print(f"Control frequency: {env.control_freq} Hz")
    print(f"Horizon: {env.horizon} steps")
    
    # Robot info
    for i, robot in enumerate(env.robots):
        print(f"\n--- Robot {i} ({['Left', 'Right'][i]} side) ---")
        print(f"  Robot name: {robot.name}")
        print(f"  Robot model: {robot.robot_model.name}")
        
        # Get EEF pose
        eef_name = robot.robot_model.eef_name["right"]
        pose = robot.pose_in_base_from_name(eef_name)
        pos = pose[:3, 3]
        print(f"  EEF name: {eef_name}")
        print(f"  EEF position (base frame): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Gripper info
        print(f"  Gripper type: {robot.gripper['right'].name if hasattr(robot.gripper['right'], 'name') else 'Panda Gripper'}")
    
    # Camera info
    print(f"\n--- Camera Configuration ---")
    print(f"  Camera names: {env.camera_names}")
    print(f"  Camera height: {env.camera_heights[0] if hasattr(env, 'camera_heights') else 'N/A'}")
    print(f"  Camera width: {env.camera_widths[0] if hasattr(env, 'camera_widths') else 'N/A'}")
    
    # Action space info
    print(f"\n--- Action Space ---")
    action_dim = env.action_dim
    print(f"  Total action dimension: {action_dim}")
    print(f"  Action spec (low): {env.action_spec[0][:6]}...")
    print(f"  Action spec (high): {env.action_spec[1][:6]}...")
    
    print("\n" + "=" * 60)


def get_sample_custom_objects():
    """Return a list of sample custom objects to add to the environment."""
    return [
        {
            'type': 'box',
            'name': 'red_block',
            'size': [0.04, 0.04, 0.04],  # 4cm cube
            'pos': [0.0, 0.0, 0.85],
            'rgba': [0.9, 0.1, 0.1, 1.0],  # Red
            'density': 500,
        },
        {
            'type': 'cylinder',
            'name': 'green_cylinder',
            'size': [0.025, 0.08],  # radius=2.5cm, height=8cm
            'pos': [0.1, 0.15, 0.85],
            'rgba': [0.1, 0.9, 0.1, 1.0],  # Green
            'density': 800,
        },
        {
            'type': 'ball',
            'name': 'blue_sphere',
            'size': [0.03],  # radius=3cm
            'pos': [-0.1, 0.1, 0.85],
            'rgba': [0.1, 0.1, 0.9, 1.0],  # Blue
            'density': 600,
        },
    ]


def run_random_actions(env, num_steps=200, render=True):
    """
    Run the environment with random actions to see the robots move.
    
    Args:
        env: The robosuite environment
        num_steps: Number of steps to run
        render: Whether to render the environment
    """
    print("\nRunning random actions...")
    print("Watch the robots move randomly!")
    print("(Press Ctrl+C to stop)\n")
    
    action_dim = env.action_dim
    
    try:
        for step in range(num_steps):
            # Generate small random actions (scaled down for smoother motion)
            action = np.random.uniform(-0.3, 0.3, action_dim)
            
            # Zero out gripper actions to keep them stable
            # Assuming action format: [pos(3), ori(3), gripper(1)] per arm
            # For two arms: [arm0_pos, arm0_ori, arm0_grip, arm1_pos, arm1_ori, arm1_grip]
            action[6] = 0  # Robot 0 gripper
            action[13] = 0  # Robot 1 gripper (if action dim is 14)
            
            obs, reward, done, info = env.step(action)
            
            if render:
                env.render()
            
            if step % 50 == 0:
                print(f"  Step {step}/{num_steps}")
                
    except KeyboardInterrupt:
        print("\nStopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch and visualize the dual-arm Franka environment"
    )
    parser.add_argument(
        '--with-objects',
        action='store_true',
        help='Add sample custom objects (box, cylinder, sphere) to the environment'
    )
    parser.add_argument(
        '--random-actions',
        action='store_true',
        help='Run random actions to see the robots move'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Number of random action steps (default: 200)'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (useful for headless testing)'
    )
    
    args = parser.parse_args()
    
    env = None
    try:
        print("Creating dual-arm Franka environment...")
        
        # Create environment with or without custom objects
        if args.with_objects:
            print("Adding custom objects to the scene...")
            custom_objects = get_sample_custom_objects()
            env = make_dual_arm_env(custom_objects=custom_objects)
        else:
            env = make_dual_arm_env()
        
        print("Resetting environment...")
        obs = env.reset()
        
        # Print environment information
        print_env_info(env)
        
        # Print observation keys
        print("\n--- Observation Keys ---")
        for key in sorted(obs.keys()):
            if isinstance(obs[key], np.ndarray):
                print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
            else:
                print(f"  {key}: {type(obs[key])}")
        
        if args.random_actions:
            # Run with random actions
            run_random_actions(env, num_steps=args.steps, render=not args.no_render)
        else:
            # Just visualize the environment
            print("\n" + "=" * 60)
            print("VISUALIZATION MODE")
            print("=" * 60)
            print("\nThe environment is now running!")
            print("You should see a window with two Panda robots.")
            print("\nControls:")
            print("  - Close the window or press Ctrl+C to exit")
            print("  - The robots will hold their initial pose")
            print("=" * 60 + "\n")
            
            # Main render loop
            step = 0
            while True:
                # Apply zero action to maintain pose
                action = np.zeros(env.action_dim)
                obs, reward, done, info = env.step(action)
                
                if not args.no_render:
                    env.render()
                
                step += 1
                time.sleep(0.02)  # ~50 FPS
                
    except KeyboardInterrupt:
        print("\n\nExiting... (Ctrl+C pressed)")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            print("Closing environment...")
            env.close()
            print("Done!")


if __name__ == '__main__':
    main()
