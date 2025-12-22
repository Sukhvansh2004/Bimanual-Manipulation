#!/usr/bin/env python3
"""Test script for transforms module."""

from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.environments import make_dual_arm_env

def main():
    print("Creating environment...")
    env = make_dual_arm_env()
    env.reset()
    
    print("\nCreating TransformManager...")
    tf_manager = TransformManager(env)
    
    print("\nKnown frames:")
    frames = tf_manager.get_known_frames()
    for frame in sorted(frames):
        print(f"  - {frame}")
    
    print("\nFrame tree:")
    tf_manager.print_frame_tree()
    
    # Find EEF frames
    eef_frames = [f for f in frames if 'eef' in f.lower()]
    print(f"\nFound EEF frames: {eef_frames}")
    
    if eef_frames:
        # Use the first EEF frame found
        eef_frame = eef_frames[0]
        print(f"\nGetting pose of '{eef_frame}' in 'world' frame...")
        
        try:
            eef_pose = tf_manager.get_frame_pose(eef_frame, "world")
            print(f"\nSuccess! EEF pose:")
            print(f"  Position: {eef_pose.position}")
            print(f"  Quaternion: {eef_pose.quaternion}")
            print(f"  Frame: {eef_pose.frame_id}")
        except Exception as e:
            print(f"\nError getting EEF pose: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo EEF frames found!")
    
    print("\nTest complete!")
    env.close()

if __name__ == "__main__":
    main()
