#!/usr/bin/env python3
"""
Live test for SE(3) trajectory planning with robosuite environment.

Runs with rendering DISABLED for faster execution.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# Disable rendering BEFORE importing the module
from robosuite_franka_manipulation import constants
constants.DEFAULT_HAS_RENDERER = False
# Note: Keep offscreen renderer on if using camera observations
# constants.DEFAULT_USE_CAMERA_OBS = False  # Would also disable cameras

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.controllers import RobosuiteArmController


def test_trajectory_execution():
    """Test SE(3) trajectory execution in robosuite."""
    print("Creating environment (headless mode, offscreen renderer enabled)...")
    
    env = make_dual_arm_env()
    env.reset()
    
    # Create arm controller for right arm
    controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
    
    # Get current end-effector pose
    current_pos = env.robots[0]._hand_pos["right"]
    current_ori_mat = env.robots[0]._hand_orn["right"]
    current_quat = R.from_matrix(current_ori_mat).as_quat()
    
    print(f"\nInitial pose:")
    print(f"  Position: {current_pos}")
    print(f"  Quaternion: {current_quat}")
    
    # Define target pose (move forward and rotate slightly)
    target_pos = current_pos + np.array([0.1, 0.05, 0.0])  # 10cm forward, 5cm right
    target_quat = (R.from_quat(current_quat) * R.from_euler('z', 15, degrees=True)).as_quat()
    
    print(f"\nTarget pose:")
    print(f"  Position: {target_pos}")
    print(f"  Quaternion: {target_quat}")
    
    # Plan trajectory
    print("\nPlanning SE(3) trajectory...")
    t_start = time.time()
    positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
        current_pos, current_quat, target_pos, target_quat,
        max_vel=0.1, max_acc=0.2, dt=0.02
    )
    t_plan = time.time() - t_start
    
    print(f"  Planning time: {t_plan*1000:.2f} ms")
    print(f"  Trajectory waypoints: {len(positions)}")
    print(f"  Expected duration: {total_time:.2f} s")
    
    # Analyze trajectory quality
    print("\n=== Trajectory Quality Analysis ===")
    
    # Position smoothness
    velocities = np.diff(positions, axis=0) / 0.02
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = np.max(speeds)
    
    accelerations = np.diff(velocities, axis=0) / 0.02
    max_acc = np.max(np.linalg.norm(accelerations, axis=1))
    
    print(f"  Max speed: {max_speed:.4f} m/s (limit: 0.1)")
    print(f"  Max acceleration: {max_acc:.4f} m/s² (limit: 0.2)")
    
    # Rotation smoothness  
    angular_vels = []
    for i in range(1, len(quaternions)):
        q1 = R.from_quat(quaternions[i-1])
        q2 = R.from_quat(quaternions[i])
        delta = (q2 * q1.inv()).as_rotvec()
        angular_vels.append(np.linalg.norm(delta) / 0.02)
    
    max_angular_vel = np.max(angular_vels) if angular_vels else 0
    print(f"  Max angular velocity: {np.degrees(max_angular_vel):.2f} deg/s")
    
    # Execute trajectory
    print("\n=== Executing Trajectory (headless) ===")
    t_start = time.time()
    
    from robosuite_franka_manipulation.transforms import StampedPose
    target_pose = StampedPose(
        position=target_pos,
        quaternion=target_quat,
        frame_id=controller.base_frame
    )
    
    success = controller.move_to_pose(target_pose)
    
    t_exec = time.time() - t_start
    print(f"  Execution time: {t_exec:.2f} s")
    print(f"  Success: {success}")
    
    # Check final pose
    final_pos = env.robots[0]._hand_pos["right"]
    final_ori_mat = env.robots[0]._hand_orn["right"]
    final_quat = R.from_matrix(final_ori_mat).as_quat()
    
    pos_error = np.linalg.norm(final_pos - target_pos)
    ori_error = np.linalg.norm((R.from_quat(target_quat) * R.from_quat(final_quat).inv()).as_rotvec())
    
    print(f"\nFinal errors:")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Orientation error: {np.degrees(ori_error):.2f} deg")
    
    env.close()
    
    return success


def test_multiple_motions():
    """Test multiple sequential motions."""
    print("\n" + "="*60)
    print("Testing Multiple Sequential Motions")
    print("="*60)
    
    env = make_dual_arm_env()
    env.reset()
    
    controller = RobosuiteArmController(env, arm_id="right", robot_id=0)
    
    # Define a sequence of motions
    motions = [
        ("Forward", np.array([0.05, 0.0, 0.0]), np.array([0, 0, 0, 1])),
        ("Up", np.array([0.0, 0.0, 0.05]), np.array([0, 0, 0, 1])),
        ("Rotate Z", np.array([0.0, 0.0, 0.0]), R.from_euler('z', 30, degrees=True).as_quat()),
        ("Diagonal", np.array([0.03, 0.03, -0.02]), R.from_euler('x', -10, degrees=True).as_quat()),
    ]
    
    all_success = True
    for name, delta_pos, delta_quat in motions:
        current_pos = env.robots[0]._hand_pos["right"]
        current_quat = R.from_matrix(env.robots[0]._hand_orn["right"]).as_quat()
        
        target_pos = current_pos + delta_pos
        if np.allclose(delta_quat, [0, 0, 0, 1]):
            target_quat = current_quat
        else:
            target_quat = (R.from_quat(current_quat) * R.from_quat(delta_quat)).as_quat()
        
        print(f"\n{name} motion...")
        
        target_pose = StampedPose(
            position=target_pos,
            quaternion=target_quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(target_pose)
        
        final_pos = env.robots[0]._hand_pos["right"]
        pos_error = np.linalg.norm(final_pos - target_pos) * 1000
        print(f"  Success: {success}, Position error: {pos_error:.2f} mm")
        
        all_success = all_success and success
    
    env.close()
    print(f"\nAll motions successful: {all_success}")
    return all_success


if __name__ == "__main__":
    print("="*60)
    print("SE(3) Trajectory Planning Live Test")
    print("="*60)
    
    test_trajectory_execution()
    test_multiple_motions()
    
    print("\n✓ Live tests completed!")
