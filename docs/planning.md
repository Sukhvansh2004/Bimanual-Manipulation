# Planning Architecture

The planning module (`robosuite_franka_manipulation.planning`) provides a robust framework for generating, validating, and executing complex dual-arm manipulation tasks. It integrates grasp synthesis, trajectory interpolation, inverse kinematics (IK) solvers, and physics-based collision checking.

## Overview

The planning pipeline follows these stages:
1.  **Grasp Generation**: Sample viable grasps on a target object using `GraspGenerator`.
2.  **Feasibility Filtering**: Filter grasps based on reachability (IK) and collision checks.
3.  **Trajectory Generation**: Generate SE(3) trajectories for picking, lifting, and placing.
4.  **Trajectory Validation**: Validate entire trajectories waypoint-by-waypoint using the MuJoCo physics engine.

## Key Components

### 1. Trajectory Generator (`trajectory_generator.py`)

This is the core module for creating and verifying robot motions.

-   **`generate_object_trajectory(start_pose, target_pose, ...)`**: Creates a list of `StampedPose` waypoints representing the object's path (e.g., lift, approach, insert). It uses SE(3) interpolation (SLERP for rotation, linear for position) to ensure smooth motion.
-   **`generate_end_effector_trajectory(object_trajectory, grasp, ...)`**: Computes the corresponding end-effector trajectory by applying the grasp transform to the object's path.
-   **`check_trajectory_feasibility(env, robot_id, eef_trajectory, ...)`**: The main validation function. It iterates through each waypoint in a trajectory, solving IK for the robot arm and checking for collisions in the MuJoCo simulation. It supports "warm-starting" the IK solver with the previous waypoint's solution for temporal consistency.

### 2. Collision Checker (`collision_checker.py`)

Handles collision queries against the environment.

-   **`check_single_grasp_feasibility(...)`**: Checks if a single static grasp pose is reachable and collision-free.
-   **`find_feasible_grasp_pairs(...)`**: For dual-arm tasks, this function finds pairs of grasps (one for each arm) that are simultaneously valid and do not cause the robots to collide with each other or the environment. It leverages `check_dual_arm_pose_feasibility`.
-   **`check_dual_arm_pose_feasibility(...)`**: Verified if a specific configuration of two arms is valid.

### 3. Grasp Generator (`grasp_generator.py`)

Wraps the GraspGen service to provide high-level grasp sampling APIs.

-   **`generate_grasps(object_name, gripper_types, ...)`**: Interacts with the `GraspService` to retrieve 6-DOF grasp candidates for specified grippers.

## Usage Example

### Generating and Validating a Pick-and-Place Trajectory

```python
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_object_trajectory, 
    generate_end_effector_trajectory,
    check_trajectory_feasibility
)
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager

# 1. Define Start and Target Poses
start_pose = StampedPose(position=[0.5, 0, 0.85], quaternion=[0, 0, 0, 1], frame_id="world")
target_pose = StampedPose(position=[0.5, 0, 1.0], quaternion=[0, 0, 0, 1], frame_id="world")

# 2. Generate Object Path
object_traj = generate_object_trajectory(start_pose, target_pose, tf_manager=tf_manager)

# 3. Calculate Robot Path for a Specific Grasp
robot_traj = generate_end_effector_trajectory(object_traj, my_grasp, tf_manager=tf_manager)

# 4. Validate
results, is_feasible = check_trajectory_feasibility(
    env=env,
    robot_id=0,
    eef_trajectory=robot_traj,
    tf_manager=tf_manager,
    object_trajectory=object_traj,
    object_name="plate_target",
    check_collisions=True
)

if is_feasible:
    print("Trajectory is valid! Executing...")
else:
    print(f"Trajectory failed at waypoint {results[0].waypoint_index}")
```

## Implementation Details

### Inverse Kinematics (IK)
The module uses a robust Scipy-based IK solver (`solve_ik_scipy`) wrapped by `enumerate_ik_solutions_for_arm`. This wrapper adds robustness by trying multiple random seed configurations (including a "warm start" from the previous solution) to avoid local minima and handle the redundancy of 7-DOF arms.

### Coordinate Frames
All planning functions rely on `StampedPose` and `TransformManager` to ensure coordinate systems are handled correctly. Trajectories are typically computed in the `world` frame, but the IK solver targets specific sites (e.g., `gripper0_right_ft_frame`). The `check_trajectory_feasibility` function automatically handles the necessary frame conversions, including applying a -90 degree Z-axis rotation correction to align standard grasp conventions with the Franka flange frame.

### Optimization
-   **Warm Starting**: The IK solver uses the solution from the previous waypoint as a seed for the next, significantly speeding up trajectory validation and ensuring joint-space continuity.
-   **Dynamic Programming**: When validating pairs of grasps for dual-arm tasks, results are cached to avoid re-computing feasibility for the same grasp used in different pairs.
