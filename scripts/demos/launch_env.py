#!/usr/bin/env python3
"""
Demo script to launch and visualize the dual-arm Franka environment.

This script creates a dual-arm robosuite environment with two Panda robots
in an opposed configuration. It demonstrates how to add custom primitive
objects (Box, Cylinder, etc.) and YCB dataset objects using the
refactored DualArmEnvironment.
"""

import argparse
import time
import numpy as np
import logging

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation.utils.env_utils import (
    get_available_ycb_objects,
    list_ycb_objects,
    create_primitive_object,
    create_ycb_object,
)
import robosuite_franka_manipulation.constants as constants

# Configure logging
logging.basicConfig(
    level=constants.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


def print_env_info(env):
    """Print information about the environment."""
    logger.info("\n" + "=" * 60)
    logger.info("DUAL-ARM FRANKA ENVIRONMENT INFO")
    logger.info("=" * 60)
    
    # Environment info
    logger.info(f"Environment: {env.name if hasattr(env, 'name') else 'TwoArmHandover'}")
    logger.info(f"Number of robots: {len(env.robots)}")
    logger.info(f"Control frequency: {env.control_freq} Hz")
    logger.info(f"Horizon: {env.horizon} steps")
    
    # Create local TransformManager for querying poses
    tf_manager = TransformManager(env)
    
    # Robot info
    for i, robot in enumerate(env.robots):
        logger.info(f"--- Robot {i} ({['Left', 'Right'][i]} side) ---")
        logger.info(f"  Robot name: {robot.name}")
        logger.info(f"  Robot model: {robot.robot_model.name}")
        
        # Get EEF pose using TransformManager
        eef_frame = f"robot{i}_right_eef"
        base_frame = f"robot{i}_base"
        
        try:
            pose = tf_manager.get_frame_pose(eef_frame, base_frame)
            pos = pose.position
            logger.info(f"  EEF name: {eef_frame}")
            logger.info(f"  EEF position (base frame): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        except Exception as e:
            logger.warning(f"  Could not get EEF pose for {eef_frame}: {e}")
    
    # Custom objects info
    if hasattr(env, 'get_object_names'):
        obj_names = env.get_object_names()
        if obj_names:
            logger.info(f"--- Custom Objects ({len(obj_names)}) ---")
            for name in obj_names:
                pose = env.get_object_pose(name)
                if pose is not None:
                    pos = pose[:3, 3]
                    logger.info(f"  {name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                else:
                    logger.info(f"  {name}: (pose not available)")
    
    logger.info("=" * 60)


def get_sample_custom_objects():
    """
    Return a list of sample custom primitive objects (dicts) to add to the environment.
    """
    return [
        create_primitive_object(
            obj_type='box',
            name='red_block',
            size=[0.04, 0.04, 0.04],  # 4cm cube
            position=[0.0, 0.0, 0.90],  # Center of table
            rgba=[0.9, 0.1, 0.1, 1.0],  # Red
            density=500,
        ),
        create_primitive_object(
            obj_type='cylinder',
            name='green_cylinder',
            size=[0.025, 0.04],  # radius=2.5cm, half-height=4cm
            position=[0.15, 0.0, 0.90],  # Right side of table
            rgba=[0.1, 0.9, 0.1, 1.0],  # Green
            density=800,
        ),
        create_primitive_object(
            obj_type='ball',
            name='blue_sphere',
            size=[0.03],  # radius=3cm
            position=[-0.15, 0.0, 0.90],  # Left side of table
            rgba=[0.1, 0.1, 0.9, 1.0],  # Blue
            density=600,
        ),
    ]


def get_sample_ycb_objects(ycb_names=None):
    """
    Get sample YCB object specifications as a list of dictionaries.
    
    Args:
        ycb_names: List of YCB object names to use, or None for defaults
        
    Returns:
        List of dicts, where each dict contains the spec for a YCB object.
    """
    available = get_available_ycb_objects()
    logger.debug(f"Received YCB names: {ycb_names}")
    
    if not available:
        logger.warning("No YCB objects found in dataset.")
        return []
    
    if ycb_names is None:
        # Default objects (check availability)
        default_objects = ['apple', 'mug', 'banana', 'mustard_bottle']
        ycb_names = [obj for obj in default_objects if obj in available]
        if not ycb_names:
            ycb_names = available[:3]
    
    objects = []
    for i, name in enumerate(ycb_names):
        if name not in available:
            logger.warning(f"YCB object '{name}' not found. Skipping.")
            continue
        
        # Position objects in a row on the table
        pos = [-0.1 + i * 0.3, 0.0, 0.90]
        quat = None
        
        # Special handling for shelf: place it on the floor, in front/behind the table
        if name == 'shelf':
            pos = [-0.8, 0.0, 0.0]
            quat = [0.0, 0.0, 0.70710678, 0.70710678]

        # Create the dictionary spec for this object
        obj_spec = create_ycb_object(
            ycb_name=name,
            instance_name=f"{name}_{i}",  # Unique instance name
            position=pos,
            quaternion=quat
        )
        objects.append(obj_spec)
    
    return objects


def run_random_actions(env, num_steps=200, render=True):
    """Run the environment with random actions."""
    logger.info("Running random actions...")
    
    action_dim = env.action_dim
    
    try:
        for _ in range(num_steps):
            action = np.random.uniform(-0.3, 0.3, action_dim)
            
            # Zero out gripper actions
            if action_dim >= 7:
                action[6] = 0
            if action_dim >= 14:
                action[13] = 0
            
            env.step(action)
            
            if render:
                env.render()
                
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Launch dual-arm Franka environment")
    parser.add_argument('--with-objects', action='store_true', help='Add sample primitive objects')
    parser.add_argument('--ycb', nargs='+', metavar='OBJECT', help='Add YCB objects by name')
    parser.add_argument('--ycb-all', type=int, default=0, help='Add N YCB objects')
    parser.add_argument('--list-ycb', action='store_true', help='List available YCB objects')
    parser.add_argument('--random-actions', action='store_true', help='Run random actions')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    
    args = parser.parse_args()
    
    if args.list_ycb:
        list_ycb_objects()
        return
    
    # Collect all object specifications (as lists of dicts)
    custom_objects = []
    ycb_objects = []
    
    if args.with_objects:
        custom_objects = get_sample_custom_objects()
        logger.info(f"Adding {len(custom_objects)} custom primitive objects.")
    
    if args.ycb:
        ycb_objects.extend(get_sample_ycb_objects(args.ycb))
        logger.info(f"Adding {len(args.ycb)} specific YCB objects.")
    
    if args.ycb_all > 0:
        available = get_available_ycb_objects()
        # Filter out ones we already added to avoid duplicates if desired, 
        # or just add more instances. Here we just add more.
        additional_count = min(args.ycb_all, len(available))
        additional_objs = get_sample_ycb_objects(available[:additional_count])
        ycb_objects.extend(additional_objs)
        logger.info(f"Adding first {additional_count} available YCB objects.")
    
    env = None
    try:
        logger.info("Creating dual-arm Franka environment...")
        
        # We explicitly pass the lists of dictionaries here
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            ycb_objects=ycb_objects,  # List of dicts
            has_renderer=not args.headless and not args.no_render,
        )
        
        logger.info("Resetting environment...")
        env.reset()
        
        if not args.no_render and not args.headless:
            env.render()
        
        print_env_info(env)
        
        if args.random_actions:
            run_random_actions(env, num_steps=args.steps, render=not args.no_render and not args.headless)
        else:
            logger.info("Visualization mode. Press Ctrl+C to exit.")
            while True:
                env.step(np.zeros(env.action_dim))
                if not args.no_render and not args.headless:
                    env.render()
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if env is not None:
            env.close()

if __name__ == '__main__':
    main()
