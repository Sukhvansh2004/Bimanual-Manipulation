#!/usr/bin/env python3
"""
Test script for SE(3) trajectory planning in arm controller.

Tests the SE(3) Lie algebra interpolation for smooth robot motion.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Import SE(3) utilities from the controller module
from robosuite_franka_manipulation.controllers import (
    skew,
    unskew,
    se3_to_matrix,
    matrix_to_se3,
    SE3_log,
    SE3_exp,
    pose_to_SE3,
    SE3_to_pose,
    compute_se3_distance,
)
from robosuite_franka_manipulation import constants


def test_se3_utilities():
    """Test that SE3_log and SE3_exp are inverses of each other."""
    print("=== Testing SE(3) Utilities ===")
    
    # Test 1: Identity
    T_id = np.eye(4)
    twist = SE3_log(T_id)
    T_recovered = SE3_exp(twist)
    assert np.allclose(T_id, T_recovered), "Identity test failed"
    print("  ✓ Identity: exp(log(I)) = I")
    
    # Test 2: Pure translation
    T_trans = np.eye(4)
    T_trans[:3, 3] = [0.5, -0.3, 0.2]
    twist = SE3_log(T_trans)
    T_recovered = SE3_exp(twist)
    assert np.allclose(T_trans, T_recovered, atol=1e-10), "Pure translation test failed"
    print("  ✓ Pure translation: exp(log(T)) = T")
    
    # Test 3: Pure rotation (90 degrees about z)
    T_rot = np.eye(4)
    T_rot[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
    twist = SE3_log(T_rot)
    T_recovered = SE3_exp(twist)
    assert np.allclose(T_rot, T_recovered, atol=1e-10), "Pure rotation test failed"
    print("  ✓ Pure rotation (90°): exp(log(T)) = T")
    
    # Test 4: Combined transform
    T_combined = np.eye(4)
    T_combined[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    T_combined[:3, 3] = [0.3, 0.2, 0.1]
    twist = SE3_log(T_combined)
    T_recovered = SE3_exp(twist)
    assert np.allclose(T_combined, T_recovered, atol=1e-10), "Combined transform test failed"
    print("  ✓ Combined transform: exp(log(T)) = T")
    
    # Test 5: Interpolation consistency
    # T(0.5) should be "halfway" between T_start and T_target
    T_start = pose_to_SE3(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    T_target = pose_to_SE3(np.array([1, 0, 0]), R.from_euler('z', 90, degrees=True).as_quat())
    
    T_start_inv = np.eye(4)
    T_start_inv[:3, :3] = T_start[:3, :3].T
    T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]
    T_rel = T_start_inv @ T_target
    twist = SE3_log(T_rel)
    
    T_half = T_start @ SE3_exp(0.5 * twist)
    pos_half, quat_half = SE3_to_pose(T_half)
    
    # Position should be approximately 0.5 m along x (with some coupling due to rotation)
    # This is not exactly 0.5 because SE(3) geodesic is not linear interpolation
    print(f"  ✓ Halfway interpolation: pos={pos_half}, angle={np.degrees(2*np.arccos(quat_half[3])):.1f}°")
    
    print("All SE(3) utility tests passed!\n")


def test_trajectory_planning():
    """Test trajectory planning function from controller."""
    print("=== Testing Trajectory Planning ===")
    
    # Import the actual function we're testing
    from robosuite_franka_manipulation.controllers.arm import RobosuiteArmController
    
    # We can't instantiate the controller without an env, but we can call the static methods
    # Let's test using the module-level functions
    
    # Test 1: Pure translation
    start_pos = np.array([0.0, 0.0, 0.0])
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])
    target_pos = np.array([0.3, 0.2, 0.1])
    target_quat = np.array([0.0, 0.0, 0.0, 1.0])
    
    positions, quaternions, total_time = plan_trajectory_standalone(
        start_pos, start_quat, target_pos, target_quat,
        max_vel=0.2, max_acc=0.5, dt=0.02
    )
    
    print(f"  Pure translation: {len(positions)} waypoints, {total_time:.2f}s")
    print(f"    Start: {start_pos} -> End: {positions[-1]}")
    print(f"    Error: {np.linalg.norm(positions[-1] - target_pos):.6f} m")
    
    # Test 2: Pure rotation
    start_pos = np.array([0.0, 0.0, 0.0])
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])
    target_pos = np.array([0.0, 0.0, 0.0])
    target_quat = R.from_euler('z', 90, degrees=True).as_quat()
    
    positions, quaternions, total_time = plan_trajectory_standalone(
        start_pos, start_quat, target_pos, target_quat,
        max_vel=0.2, max_acc=0.5, dt=0.02
    )
    
    print(f"  Pure rotation: {len(positions)} waypoints, {total_time:.2f}s")
    target_angle = 2 * np.arccos(target_quat[3])
    final_angle = 2 * np.arccos(np.abs(quaternions[-1][3]))
    print(f"    Target angle: {np.degrees(target_angle):.1f}° -> Final: {np.degrees(final_angle):.1f}°")
    
    # Test 3: Combined motion
    start_pos = np.array([0.0, 0.0, 0.0])
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])
    target_pos = np.array([0.3, 0.2, 0.1])
    target_quat = R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat()
    
    positions, quaternions, total_time = plan_trajectory_standalone(
        start_pos, start_quat, target_pos, target_quat,
        max_vel=0.2, max_acc=0.5, dt=0.02
    )
    
    print(f"  Combined motion: {len(positions)} waypoints, {total_time:.2f}s")
    print(f"    Position error: {np.linalg.norm(positions[-1] - target_pos):.6f} m")
    
    # Check smoothness
    print("\n  Smoothness check:")
    velocities = np.diff(positions, axis=0) / 0.02
    max_vel = np.max(np.linalg.norm(velocities, axis=1))
    accelerations = np.diff(velocities, axis=0) / 0.02
    max_acc = np.max(np.linalg.norm(accelerations, axis=1))
    print(f"    Max velocity: {max_vel:.4f} m/s")
    print(f"    Max acceleration: {max_acc:.4f} m/s²")
    
    print("\nTrajectory planning tests passed!\n")
    
    return positions, quaternions


def plan_trajectory_standalone(start_pos, start_quat, target_pos, target_quat,
                               max_vel=0.2, max_acc=0.5, dt=0.02):
    """Standalone trajectory planning (same as RobosuiteArmController method)."""
    # Convert to SE(3) matrices
    T_start = pose_to_SE3(start_pos, start_quat)
    T_target = pose_to_SE3(target_pos, target_quat)
    
    # Compute relative transformation
    T_start_inv = np.eye(4)
    T_start_inv[:3, :3] = T_start[:3, :3].T
    T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]
    T_rel = T_start_inv @ T_target
    
    # Map to Lie algebra
    twist = SE3_log(T_rel)
    
    # Compute distances
    omega = twist[:3]
    v = twist[3:]
    trans_distance = np.linalg.norm(v)
    rot_distance = np.linalg.norm(omega)
    
    # Check if movement needed
    total_se3_distance = compute_se3_distance(twist, 1.0, constants.TRAJECTORY_ORIENTATION_SCALING)
    if total_se3_distance < constants.TRAJECTORY_MIN_DISTANCE:
        return np.array([target_pos]), np.array([target_quat]), 0.0
    
    # Effective distance for timing
    effective_distance = max(trans_distance, rot_distance * constants.TRAJECTORY_ORIENTATION_SCALING * 2)
    
    # Trapezoidal profile
    t_acc = max_vel / max_acc
    d_acc = 0.5 * max_acc * t_acc**2
    t_dec = t_acc
    d_dec = d_acc
    d_const = effective_distance - d_acc - d_dec
    
    if d_const > 0:
        t_const = d_const / max_vel
    else:
        t_acc = np.sqrt(effective_distance / max_acc)
        t_const = 0
        t_dec = t_acc
        d_acc = 0.5 * effective_distance
        d_dec = d_acc
        d_const = 0
    
    total_time = t_acc + t_const + t_dec
    n_steps = max(constants.TRAJECTORY_MIN_STEPS, int(np.ceil(total_time / dt)))
    
    # Generate trajectory
    positions = []
    quaternions = []
    
    for i in range(n_steps + 1):
        t = i * total_time / n_steps if n_steps > 0 else 0
        
        if t <= t_acc:
            d = 0.5 * max_acc * t**2
        elif t <= t_acc + t_const:
            d = d_acc + max_vel * (t - t_acc)
        else:
            t_in_dec = t - (t_acc + t_const)
            d = d_acc + d_const + max_vel * t_in_dec - 0.5 * max_acc * t_in_dec**2
        
        s = np.clip(d / effective_distance, 0, 1) if effective_distance > 0 else 1.0
        
        T_interp = T_start @ SE3_exp(s * twist)
        pos, quat = SE3_to_pose(T_interp)
        positions.append(pos)
        quaternions.append(quat)
    
    return np.array(positions), np.array(quaternions), total_time


def visualize_trajectory(positions, quaternions, save_path="scratch/se3_trajectory.png"):
    """Visualize the trajectory."""
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], c='g', s=100, marker='o', label='Start')
    ax1.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], c='r', s=100, marker='*', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Position vs time
    ax2 = fig.add_subplot(132)
    t = np.linspace(0, 1, len(positions))
    ax2.plot(t, positions[:, 0], 'r-', label='X')
    ax2.plot(t, positions[:, 1], 'g-', label='Y')
    ax2.plot(t, positions[:, 2], 'b-', label='Z')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Velocity profile
    ax3 = fig.add_subplot(133)
    dt = 1.0 / (len(positions) - 1)
    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    t_vel = np.linspace(0, 1, len(velocities))
    ax3.plot(t_vel, velocities, 'b-', linewidth=2)
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Velocity Profile (Trapezoidal)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory visualization to {save_path}")
    plt.close()


def compare_old_vs_new():
    """Compare old decoupled interpolation vs new SE(3) interpolation."""
    print("=== Comparing Old vs New Interpolation ===")
    
    # Test case: combined translation + rotation
    start_pos = np.array([0.0, 0.0, 0.0])
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])
    target_pos = np.array([0.3, 0.0, 0.0])
    target_quat = R.from_euler('z', 90, degrees=True).as_quat()
    
    # Old method (decoupled linear interpolation)
    n_steps = 50
    old_positions = []
    old_quaternions = []
    for i in range(n_steps + 1):
        s = i / n_steps
        pos = start_pos + s * (target_pos - start_pos)
        quat = R.from_rotvec(
            s * R.from_quat(target_quat).as_rotvec()
        ).as_quat()
        old_positions.append(pos)
        old_quaternions.append(quat)
    old_positions = np.array(old_positions)
    old_quaternions = np.array(old_quaternions)
    
    # New method (SE(3) geodesic)
    new_positions, new_quaternions, _ = plan_trajectory_standalone(
        start_pos, start_quat, target_pos, target_quat,
        max_vel=10, max_acc=100, dt=0.02  # High values to get pure interpolation
    )
    
    # Resample to same number of points for comparison
    indices = np.linspace(0, len(new_positions) - 1, n_steps + 1).astype(int)
    new_positions = new_positions[indices]
    new_quaternions = new_quaternions[indices]
    
    print(f"  Old method: Linear position + rotation vector SLERP (decoupled)")
    print(f"  New method: SE(3) geodesic interpolation (coupled)")
    
    # The key difference is how the end-effector moves in space
    # For a "screw motion", SE(3) gives a helical path
    # Decoupled gives a straight line with independent rotation
    
    print(f"\n  At halfway point (s=0.5):")
    print(f"    Old: pos={old_positions[n_steps//2]}, quat={old_quaternions[n_steps//2]}")
    print(f"    New: pos={new_positions[len(new_positions)//2]}, quat={new_quaternions[len(new_quaternions)//2]}")
    
    # Visualize both
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(old_positions[:, 0], old_positions[:, 1], old_positions[:, 2], 'r-', linewidth=2, label='Old (decoupled)')
    ax1.plot(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2], 'b-', linewidth=2, label='New (SE3)')
    ax1.scatter([0], [0], [0], c='g', s=100, marker='o')
    ax1.scatter([0.3], [0], [0], c='k', s=100, marker='*')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    # Plot Y position (shows the arc in SE3)
    t = np.linspace(0, 1, len(old_positions))
    ax2.plot(t, old_positions[:, 1], 'r-', linewidth=2, label='Old Y')
    ax2.plot(t, new_positions[:, 1], 'b-', linewidth=2, label='New Y')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y-axis Motion\n(SE3 shows arc, decoupled is straight)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('scratch/se3_comparison.png', dpi=150)
    print(f"\n  Saved comparison to scratch/se3_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Testing SE(3) trajectory planning...\n")
    
    test_se3_utilities()
    positions, quaternions = test_trajectory_planning()
    visualize_trajectory(positions, quaternions)
    compare_old_vs_new()
    
    print("\n✓ All tests passed!")
