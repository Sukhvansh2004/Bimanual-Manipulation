#!/usr/bin/env python3
"""
Demo script to launch dual-arm Franka environments.

This script demonstrates how to:
1. Create a standard dual-arm Franka environment (TwoArmHandover)
2. Load a custom world XML from the worlds folder
3. Visualize and interact with the environment

Usage:
    # Standard dual-arm environment
    python scripts/demos/launch_env.py
    
    # List available world files
    python scripts/demos/launch_env.py --list-worlds
    
    # With custom world XML
    python scripts/demos/launch_env.py --world dual_arm_ycb_scene.xml
    
    # With random actions to see robots move
    python scripts/demos/launch_env.py --random-actions
"""

import argparse
import time
import sys
import numpy as np

from robosuite.wrappers import VisualizationWrapper
from robosuite_franka_manipulation.constants import WORLDS_DIR

from robosuite_franka_manipulation.environments import (
    make_dual_arm_env,
    make_custom_world_env,
    CustomWorldEnvironment
)

def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_env_info(env):
    """Print detailed information about the environment."""
    print_banner("ENVIRONMENT INFORMATION")
    
    # Check if it's a custom world environment
    is_custom = isinstance(env, CustomWorldEnvironment)
    
    # Basic env info
    print(f"\n[Environment]")
    print(f"  Name: {env.name if hasattr(env, 'name') else 'Unknown'}")
    print(f"  Type: {'CustomWorldEnvironment' if is_custom else 'RobosuiteEnvironment'}")
    print(f"  Control Frequency: {env.control_freq} Hz")
    print(f"  Horizon: {env.horizon} steps")
    print(f"  Action Dimension: {env.action_dim}")
    
    if is_custom:
        # Custom world specific info
        print(f"\n[MuJoCo Model]")
        print(f"  Bodies: {env.model.nbody}")
        print(f"  Geoms: {env.model.ngeom}")
        print(f"  Cameras: {env.model.ncam}")
        print(f"  Actuators: {env.model.nu}")
        print(f"  World XML: {env.world_xml_path}")
    else:
        # Robot info for robosuite environments
        if hasattr(env, 'robots'):
            print(f"\n[Robots]")
            print(f"  Number of robots: {len(env.robots)}")
            for i, robot in enumerate(env.robots):
                print(f"\n  Robot {i}:")
                print(f"    Name: {robot.name}")
                print(f"    Model: {robot.robot_model.name}")
                
                # EEF info
                eef_name = robot.robot_model.eef_name.get("right", "unknown")
                print(f"    EEF Name: {eef_name}")
                
                try:
                    pose = robot.pose_in_base_from_name(eef_name)
                    pos = pose[:3, 3]
                    print(f"    EEF Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                except Exception as e:
                    print(f"    EEF Position: (unavailable)")
    
    # Camera info
    if hasattr(env, 'camera_names'):
        print(f"\n[Cameras]")
        print(f"  Names: {env.camera_names}")
        if hasattr(env, 'camera_heights'):
            print(f"  Heights: {env.camera_heights}")
        elif hasattr(env, 'camera_height'):
            print(f"  Height: {env.camera_height}")
        if hasattr(env, 'camera_widths'):
            print(f"  Widths: {env.camera_widths}")
        elif hasattr(env, 'camera_width'):
            print(f"  Width: {env.camera_width}")
    
    print()

def list_world_files():
    """List available world XML files."""
    print_banner("AVAILABLE WORLD FILES")
    
    if not WORLDS_DIR.exists():
        print(f"\n  Worlds directory not found: {WORLDS_DIR}")
        return
    
    xml_files = list(WORLDS_DIR.glob("*.xml"))
    
    if not xml_files:
        print(f"\n  No XML files found in: {WORLDS_DIR}")
        return
    
    print(f"\n  Found {len(xml_files)} world file(s) in {WORLDS_DIR}:\n")
    for xml_file in xml_files:
        print(f"    - {xml_file.name}")
    
    print(f"\n  Usage: --world {xml_files[0].name if xml_files else 'your_world.xml'}")
    print()


def run_visualization_loop(env, render: bool = True, random_actions: bool = False, 
                           num_steps: int = 500):
    """
    Run the main visualization loop.
    
    Args:
        env: The environment to visualize
        render: Whether to render the environment
        random_actions: Whether to apply random actions
        num_steps: Number of steps to run (for random actions mode)
    """
    if random_actions:
        print_banner("RANDOM ACTIONS MODE")
        print(f"\n  Running {num_steps} steps with random actions...")
        print("  Watch the robots move!")
        print("  Press Ctrl+C to stop\n")
        
        action_dim = env.action_dim
        
        try:
            for step in range(num_steps):
                # Generate small random actions
                action = np.random.uniform(-0.2, 0.2, action_dim)
                
                # Zero out gripper actions for stability
                # Action format: [arm_action(6), gripper(1)] per robot
                if action_dim >= 7:
                    action[6] = 0  # Robot 0 gripper
                if action_dim >= 14:
                    action[13] = 0  # Robot 1 gripper
                
                obs, reward, done, info = env.step(action)
                
                if render:
                    env.render()
                
                if step % 100 == 0:
                    print(f"  Step {step}/{num_steps}")
                    
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
    else:
        print_banner("VISUALIZATION MODE")
        print("\n  Environment is running!")
        print("  The robots will hold their initial pose.")
        print("\n  Controls:")
        print("    - Close the window to exit")
        print("    - Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                # Apply zero action to maintain pose
                action = np.zeros(env.action_dim)
                obs, reward, done, info = env.step(action)
                
                if render:
                    env.render()
                
                time.sleep(0.02)  # ~50 FPS
                
        except KeyboardInterrupt:
            print("\n  Stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch dual-arm environment with custom worlds and YCB objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--world',
        type=str,
        help='Path to custom world XML file (relative to worlds/ folder or absolute)'
    )

    parser.add_argument(
        '--list-worlds',
        action='store_true',
        help='List all available world files and exit'
    )
    
    # Visualization options
    parser.add_argument(
        '--random-actions',
        action='store_true',
        help='Run random actions to see the robots move'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of steps for random actions mode (default: 500)'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (headless mode)'
    )
    
    args = parser.parse_args()
    
    if args.list_worlds:
        list_world_files()
        return 0
    
    env = None
    try:
        print_banner("CREATING DUAL-ARM ENVIRONMENT")
        
        # Determine what to create
        if args.world:
            print(f"\n  Custom world: {args.world}")
            env = make_custom_world_env(args.world)
            print("  ✓ Custom world environment created!")
        else:
            print("\n  Using standard TwoArmHandover environment")
            env = make_dual_arm_env()
            # Wrap for visualization (only for robosuite environments)
            env = VisualizationWrapper(env)
        
        # Reset environment
        print("\n  Resetting environment...")
        obs = env.reset()
        print("  ✓ Environment ready!")
        
        # Print environment info
        print_env_info(env)
        
        # Print observation summary
        print("[Observations]")
        print(f"  Keys: {len(obs)} total")
        for key in sorted(obs.keys())[:5]:  # Show first 5
            if isinstance(obs[key], np.ndarray):
                print(f"    {key}: shape={obs[key].shape}")
            else:
                print(f"    {key}: {type(obs[key]).__name__}")
        if len(obs) > 5:
            print(f"    ... and {len(obs) - 5} more")
        print()
        
        # Run visualization loop
        run_visualization_loop(
            env, 
            render=not args.no_render,
            random_actions=args.random_actions,
            num_steps=args.steps
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExiting... (Ctrl+C pressed)")
        return 0
    except Exception as e:
        import traceback
        print(f"\n  Error: {e}")
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            print("\n  Closing environment...")
            env.close()
            print("  Done!")


if __name__ == '__main__':
    sys.exit(main())
