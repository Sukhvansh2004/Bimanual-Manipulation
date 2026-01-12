#!/usr/bin/env python3
"""
Demo script to launch and visualize the dual-arm Franka environment.

This script creates a dual-arm robosuite environment with two Panda robots
in an opposed configuration and allows you to visualize and interact with it.
Supports adding custom primitive objects and YCB dataset objects.

Usage:
    python scripts/demos/launch_env.py
    python scripts/demos/launch_env.py --with-objects
    python scripts/demos/launch_env.py --ycb apple mug banana
    python scripts/demos/launch_env.py --list-ycb
    python scripts/demos/launch_env.py --random-actions
    python scripts/demos/launch_env.py --ycb apple --with-objects --random-actions
"""

import argparse
import time
import numpy as np

from robosuite_franka_manipulation.environments import (
    make_dual_arm_env,
    get_available_ycb_objects,
    list_ycb_objects,
    create_primitive_object,
    create_ycb_object,
)


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
    
    # Custom objects info
    if hasattr(env, 'get_object_names'):
        obj_names = env.get_object_names()
        if obj_names:
            print(f"\n--- Custom Objects ---")
            for name in obj_names:
                pose = env.get_object_pose(name)
                if pose is not None:
                    pos = pose[:3, 3]
                    print(f"  {name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                else:
                    print(f"  {name}: (pose not available)")
    
    print("\n" + "=" * 60)


def get_sample_custom_objects():
    """Return a list of sample custom primitive objects to add to the environment."""
    return [
        create_primitive_object(
            obj_type='box',
            name='red_block',
            size=[0.04, 0.04, 0.04],  # 4cm cube
            position=[0.0, 0.0, 0.85],
            rgba=[0.9, 0.1, 0.1, 1.0],  # Red
            density=500,
        ),
        create_primitive_object(
            obj_type='cylinder',
            name='green_cylinder',
            size=[0.025, 0.04],  # radius=2.5cm, half-height=4cm
            position=[0.1, 0.15, 0.85],
            rgba=[0.1, 0.9, 0.1, 1.0],  # Green
            density=800,
        ),
        create_primitive_object(
            obj_type='ball',
            name='blue_sphere',
            size=[0.03],  # radius=3cm
            position=[-0.1, 0.1, 0.85],
            rgba=[0.1, 0.1, 0.9, 1.0],  # Blue
            density=600,
        ),
    ]


def get_sample_ycb_objects(ycb_names=None):
    """
    Get sample YCB object specifications.
    
    Args:
        ycb_names: List of YCB object names to use, or None for defaults
        
    Returns:
        List of YCB object specifications
    """
    available = get_available_ycb_objects()
    
    if not available:
        print("Warning: No YCB objects found in dataset.")
        return []
    
    if ycb_names is None:
        # Default objects (check availability)
        default_objects = ['apple', 'mug', 'banana']
        ycb_names = [obj for obj in default_objects if obj in available]
        if not ycb_names:
            # Fallback to first 3 available objects
            ycb_names = available[:3]
    
    objects = []
    for i, name in enumerate(ycb_names):
        if name not in available:
            print(f"Warning: YCB object '{name}' not found. Skipping.")
            continue
        
        # Position objects in a row
        pos = [0.0 + i * 0.1, 0.15, 0.85]
        objects.append(create_ycb_object(name, position=pos))
    
    return objects


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
            if action_dim >= 7:
                action[6] = 0  # Robot 0 gripper
            if action_dim >= 14:
                action[13] = 0  # Robot 1 gripper
            
            obs, reward, done, info = env.step(action)
            
            if render:
                env.render()
            
            if step % 50 == 0:
                print(f"  Step {step}/{num_steps}")
                
    except KeyboardInterrupt:
        print("\nStopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch and visualize the dual-arm Franka environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Launch basic environment
  %(prog)s --with-objects           # Add sample primitive objects
  %(prog)s --ycb apple mug          # Add YCB objects
  %(prog)s --list-ycb               # List available YCB objects
  %(prog)s --ycb-all 5              # Add first 5 YCB objects
  %(prog)s --random-actions         # Run with random robot movements
        """
    )
    parser.add_argument(
        '--with-objects',
        action='store_true',
        help='Add sample custom primitive objects (box, cylinder, sphere) to the environment'
    )
    parser.add_argument(
        '--ycb',
        nargs='+',
        metavar='OBJECT',
        help='Add specified YCB objects by name (e.g., --ycb apple mug banana)'
    )
    parser.add_argument(
        '--ycb-all',
        type=int,
        metavar='N',
        default=0,
        help='Add first N YCB objects from the dataset'
    )
    parser.add_argument(
        '--list-ycb',
        action='store_true',
        help='List all available YCB objects and exit'
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
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no on-screen renderer)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-ycb
    if args.list_ycb:
        list_ycb_objects()
        return
    
    # Prepare objects
    custom_objects = []
    ycb_objects = []
    
    if args.with_objects:
        custom_objects = get_sample_custom_objects()
        print(f"Adding {len(custom_objects)} custom primitive objects...")
    
    if args.ycb:
        ycb_objects = args.ycb
        print(f"Adding YCB objects: {ycb_objects}")
    
    if args.ycb_all > 0:
        available = get_available_ycb_objects()
        ycb_objects.extend(available[:args.ycb_all])
        print(f"Adding first {min(args.ycb_all, len(available))} YCB objects...")
    
    env = None
    try:
        print("\nCreating dual-arm Franka environment...")
        
        # Create environment with custom objects
        env = make_dual_arm_env(
            custom_objects=custom_objects if custom_objects else None,
            ycb_objects=ycb_objects if ycb_objects else None,
            has_renderer=not args.headless and not args.no_render,
        )
        
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
            if custom_objects or ycb_objects:
                print(f"Custom objects added: {len(custom_objects)} primitives, {len(ycb_objects)} YCB objects")
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
