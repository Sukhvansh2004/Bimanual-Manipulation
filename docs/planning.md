# Planning Architecture

The planning module (`robosuite_franka_manipulation.planning`) provides a robust framework for generating, validating, and executing complex dual-arm manipulation tasks. It integrates collision-aware path planning on SE(3), grasp synthesis, trajectory interpolation, inverse kinematics (IK) solvers, and physics-based collision checking.

## Overview

The planning pipeline follows these stages:

1. **Grasp Generation**: Sample viable grasps on a target object using `GraspGenerator`.
2. **Feasibility Filtering**: Filter grasps based on reachability (IK) and collision checks.
3. **Object Trajectory Planning**: Plan a collision-free object path using RRT-Connect in SE(3).
4. **Trajectory Smoothing**: Refine the RRT path with shortcutting and cubic Bézier smoothing.
5. **Robot Trajectory Generation**: Compute end-effector trajectories from the object path + grasp transform.
6. **Trajectory Validation**: Validate entire trajectories waypoint-by-waypoint using IK + MuJoCo collision engine.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Grasp        │────▶│  Object Path     │────▶│  Robot Trajectory    │
│ Generation   │     │  Planning (RRT)  │     │  Generation + IK     │
└──────────────┘     └──────────────────┘     └──────────────────────┘
                           │                          │
                     Bézier Smoothing          Collision Validation
```

## Key Components

### 1. Object Trajectory Planner (`object_trajectory_planner.py`)

The centrepiece of the planning module. It plans a collision-free path for an object through the workspace, ignoring contacts with robot arms/grippers (since the robots will be carrying the object).

#### RRT-Connect in SE(3)

Uses bidirectional Rapidly-exploring Random Tree (RRT-Connect) on the SE(3) manifold:

- **State space**: SE(3) — position (R³) + orientation (quaternion SO(3))
- **Distance metric**: Weighted combination of Euclidean position distance and rotation angle
- **Collision checking**: MuJoCo contact engine — checks the object body against all static environment geometry (tables, shelves, walls), ignoring robot geoms
- **Steering**: SE(3) interpolation (`interpolate_pose`) with step-size limiting
- **Goal bias**: Configurable probability of sampling the goal pose directly

#### Path Smoothing (Hybrid Bézier)

After the raw RRT path is found, a two-phase smoothing pipeline runs:

**Phase 1 — Shortcutting**: Randomly picks two waypoints on the path and tries to connect them directly. If the direct connection is collision-free, intermediate waypoints are removed. Repeated for `RRT_SMOOTHING_ITERATIONS` attempts.

**Phase 2 — Per-segment Hybrid Bézier**: For each segment of the shortcut path:
1. Fit a cubic Bézier curve with Catmull-Rom tangents (using the full path for continuity)
2. Collision-check all dense samples on the Bézier segment
3. If collision-free → use the smooth Bézier points
4. If any collision → fall back to linear interpolation (lerp for positions, slerp for rotations)

This hybrid approach maximizes smoothness while guaranteeing collision-free paths.

**Position smoothing**: Standard cubic Bézier with Catmull-Rom tangent vectors (De Casteljau evaluation).

**Rotation smoothing**: Cubic Bézier on SO(3) using De Casteljau SLERP cascade. Control quaternions are constructed by applying so(3) tangent vectors via the exponential map.

#### Meshcat Visualization

The `RRTVisualizer` class provides real-time visualization of the planning process:
- **Cyan / Orange lines**: Tree A / Tree B growth edges
- **Red spheres**: Collision-rejected samples
- **Green line**: Final RRT path
- **Orange line**: Shortcut path (after Phase 1)
- **Grey lines + blue/red spheres**: Bézier control polygon and handles
- **Cyan / Red curve**: Bézier curve (cyan = collision-free, red = colliding segment)
- **White / Magenta spheres**: Start / goal poses

#### Configuration

Key constants in `constants.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `RRT_MAX_ITERATIONS` | 5000 | Maximum tree expansion attempts |
| `RRT_STEP_SIZE` | 0.05 | Step size for tree extension (SE(3) distance) |
| `RRT_GOAL_BIAS` | 0.15 | Probability of sampling the goal |
| `RRT_ROTATION_WEIGHT` | 0.1 | Weight for rotation in SE(3) distance |
| `RRT_COLLISION_RESOLUTION` | 0.02 | Edge collision checking resolution (m) |
| `RRT_CONNECT_MAX_STEPS` | 50 | Max connect steps toward a target |
| `RRT_SMOOTHING_ITERATIONS` | 200 | Number of shortcutting attempts |
| `RRT_ORIENTATION_NOISE` | 0.3 | Max random orientation perturbation (rad) |
| `RRT_BOUNDS_MARGIN` | [0.3, 0.3, 0.3] | Margin around start/goal for workspace bounds |
| `RRT_MIN_Z` | 0.78 | Minimum z height for planning |
| `INTERPOLATION_TYPE` | `'linear'` | `'linear'` (lerp+slerp) or `'SE3'` (geodesic) |

#### Usage

```python
from robosuite_franka_manipulation.planning.object_trajectory_planner import plan_object_trajectory

trajectory, success = plan_object_trajectory(
    env=env,
    object_name="flatPlate",
    start_pose=start_pose,      # StampedPose
    target_pose=target_pose,    # StampedPose
    tf_manager=tf_manager,
    num_steps=50,               # Output trajectory density
    rrt_vis=rrt_visualizer,     # Optional RRTVisualizer for Meshcat
)

if success:
    print(f"Planned trajectory with {len(trajectory)} waypoints")
else:
    print("RRT failed — fell back to direct interpolation")
```

### 2. Trajectory Generator (`trajectory_generator.py`)

Handles SE(3) trajectory construction and IK-based validation.

- **`generate_object_trajectory(start_pose, target_pose, tf_manager, num_steps, lift_height, approach_offset)`**: Creates a list of `StampedPose` waypoints representing the object's path (lift → approach → insert). Uses SE(3) interpolation for smooth motion.
- **`generate_end_effector_trajectory(object_trajectory, grasp, tf_manager)`**: Computes the corresponding end-effector trajectory by applying the grasp transform to the object's path.
- **`check_trajectory_feasibility(env, robot_id, eef_trajectory, tf_manager, ...)`**: The main validation function. Iterates through each waypoint, solving IK for the robot arm and checking for collisions in the MuJoCo engine. Returns per-waypoint `WaypointFeasibilityResult` objects.

### 3. Collision Checker (`collision_checker.py`)

Handles collision queries against the environment for grasp feasibility.

- **`check_single_grasp_feasibility(...)`**: Checks if a single static grasp pose is reachable and collision-free.
- **`find_feasible_grasp_pairs(...)`**: For dual-arm tasks, finds pairs of grasps (one for each arm) that are simultaneously valid and don't cause inter-robot collisions.
- **`check_dual_arm_pose_feasibility(...)`**: Verifies if a specific dual-arm configuration is valid.

### 4. Grasp Generator (`grasp_generator.py`)

Wraps the GraspGen service to provide high-level grasp sampling APIs.

- **`generate_grasps(object_name, gripper_types, ...)`**: Interacts with the `GraspService` to retrieve 6-DOF grasp candidates for specified grippers (finger and/or suction).

### 5. Planning Utilities (`planning/utils/`)

| Module | Purpose |
|--------|---------|
| `collision_utils.py` | MuJoCo geom ID lookups, robot geom masks, collision filtering |
| `ik_utils.py` | Mink QP-based IK solver with multi-seed enumeration |
| `mujoco_utils.py` | Native MuJoCo model/data access for direct C API calls |
| `trajectory_utils.py` | `WaypointFeasibilityResult` dataclass for per-waypoint results |
| `visualization_utils.py` | General Meshcat visualization helpers |
| `robot_mesh_visualizer.py` | Robot mesh rendering in Meshcat (ghost robots along trajectory) |
| `object_mesh_visualizer.py` | Object mesh rendering along planned trajectory |

## Usage Example

### Generating and Validating a Pick-and-Place Trajectory

```python
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_object_trajectory,
    generate_end_effector_trajectory,
    check_trajectory_feasibility,
)
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation import constants

# 1. Define Start and Target Poses
start_pose = StampedPose(position=[0.5, 0, 0.85], quaternion=[0, 0, 0, 1], frame_id="world")
target_pose = StampedPose(position=[0.5, 0, 1.0], quaternion=[0, 0, 0, 1], frame_id="world")

# 2. Generate Object Path (with configurable lift height and approach offset)
object_traj = generate_object_trajectory(
    start_pose,
    target_pose,
    tf_manager=tf_manager,
    num_steps=constants.OBJECT_TRAJ_NUM_STEPS,
    lift_height=constants.OBJECT_TRAJ_LIFT_HEIGHT,
    approach_offset=constants.OBJECT_TRAJ_APPROACH_OFFSET,
)

# 3. Calculate Robot Path for a Specific Grasp
robot_traj = generate_end_effector_trajectory(object_traj, my_grasp, tf_manager=tf_manager)

# 4. Validate with detailed per-waypoint results
results, is_feasible = check_trajectory_feasibility(
    env=env,
    robot_id=0,
    eef_trajectory=robot_traj,
    tf_manager=tf_manager,
    object_trajectory=object_traj,
    object_name="plate_target",
    check_collisions=True,
    arm="right",
)

if is_feasible:
    print("Trajectory is valid! Executing...")
else:
    for i, result in enumerate(results):
        if not result.is_feasible:
            print(f"Waypoint {i} failed: IK={result.ik_success}, Collision-free={result.collision_free}")
            break
```

## Implementation Details

### Inverse Kinematics (IK)

The module uses a mink QP-based differential IK solver (`solve_ik_mink`) wrapped by `enumerate_ik_solutions_for_arm`. This wrapper adds robustness by:
- Trying multiple random seed configurations (`IK_NUM_SEED_CONFIGS = 48`)
- Warm-starting from the previous waypoint's solution for trajectory continuity
- Elbow angle discretization (`IK_ELBOW_DISCRETIZATION = 8`) to handle 7-DOF redundancy
- Filtering duplicate solutions (`IK_UNIQUE_TOLERANCE = 1e-3`)
- All QP solver configuration is centralized in `constants.py` (`MINK_*` constants)

### Coordinate Frames

All planning functions rely on `StampedPose` and `TransformManager` to ensure correct frame handling. Trajectories are computed in the `world` frame, but the IK solver targets specific robot sites (e.g., `gripper0_right_ft_frame`). The `check_trajectory_feasibility` function automatically applies the -90° Z-axis rotation correction to align standard grasp conventions with the Franka flange frame.

### Optimization

- **Warm Starting**: IK solver uses the previous waypoint's solution as seed for the next, ensuring joint-space continuity and faster convergence.
- **Dynamic Programming**: When validating grasp pairs for dual-arm tasks, results are cached to avoid re-computing feasibility for the same grasp in different pair combinations.
- **Multiprocessing**: The trajectory generation scripts (`trajectory_generation.py`, `dual_trajectory_generation.py`) use Python `multiprocessing` with configurable worker pools to evaluate multiple grasp candidates in parallel.

### Collision Filtering

The experiment config system supports collision filter exclusion patterns via `collision_filter.exclude_patterns`. This allows planning to ignore expected contacts (e.g., the manipulated object touching the shelf during insertion). The function `get_collision_filter_fn()` in `experiment_config.py` constructs the appropriate MuJoCo geom filter.
