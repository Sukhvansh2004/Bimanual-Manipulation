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

### 1. Object Trajectory Planner (`trajectory/object_planner.py`)

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
from robosuite_franka_manipulation.planning.trajectory import plan_object_trajectory

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

### 6. Dual-Arm Parallel Planner (`dual_arm/`)

Enables **simultaneous** execution of two arm motions with **reactive, online collision avoidance** powered by mink's QP-based differential IK. This replaces the earlier "plan-then-stagger" approach where arms moved sequentially to avoid collisions.

#### Architecture — Asymmetric Leader–Follower

| Role | Arm | Strategy |
|------|-----|----------|
| **Leader (Arm A)** | Pre-planned | Follows a pre-computed joint trajectory open-loop (e.g. carrying an object to a handover point). |
| **Follower (Arm B)** | Reactive | Driven by a mink QP solve at every simulation tick, tracking a reference trajectory while satisfying collision-avoidance constraints against Arm A and any manipulated object. |

Arm A's motion is deterministic; Arm B continuously adapts its joint velocities to the latest Arm A configuration, guaranteeing collision avoidance at every step.

#### Lazy Initialisation (`_ensure_mink_init`)

On first use, the planner builds the following mink objects from the MuJoCo model:

| Component | Class | Purpose |
|-----------|-------|---------|
| **EEF tracker** | `mink.FrameTask` | Drives Arm B's end-effector (`robot1_right_hand`) toward a target SE3 pose. Cost = `MINK_POSITION_COST` / `MINK_ORIENTATION_COST`. |
| **Posture bias** | `mink.PostureTask` | Regularises Arm B toward the reference configuration. Cost = `MINK_POSTURE_COST`. |
| **Damping** | `mink.DampingTask` | Penalises large joint velocities for smooth motion. Cost = `MINK_DAMPING_COST`. |
| **DOF freezing** | `mink.DofFreezingTask` | Freezes every DOF except Arm B's 7 arm joints (pins Arm A joints, both grippers, and all other model DOFs). |
| **Velocity limits** | `mink.VelocityLimit` | Per-joint velocity bounds from the MuJoCo model. |
| **Configuration limits** | `mink.ConfigurationLimit` | Per-joint position bounds with `MINK_CONFIG_LIMIT_GAIN`. |
| **Collision avoidance** | `mink.CollisionAvoidanceLimit` | Inequality constraints preventing inter-arm and arm-object collisions (see below). |

Index maps (`_qpos_idx_a`, `_qpos_idx_b`) are pre-computed so the planner can inject/extract each arm's 7 joint values from the full 17-DOF configuration vector.

#### Collision Avoidance via `CollisionAvoidanceLimit`

The collision avoidance limit constructs two sets of **geom-pair groups**:

1. **Arm B ↔ Arm A**: Every collision geom on Arm B is paired against every collision geom on Arm A (arm links + gripper). The QP forbids joint velocities that would bring any pair closer than `MINK_COLLISION_MIN_DIST`.
2. **Arm B ↔ Object**: Same pairing against the manipulated object's geoms (e.g. the mug or plate being handed over). This prevents the receiving arm from colliding with the object while approaching.

Arm A ↔ Object collisions are **not** constrained because they represent expected grasp contact (the object is held by Arm A).

The `DofFreezingTask` ensures only Arm B's 7 joints appear as decision variables in the QP — Arm A's joints and both grippers are fixed parameters, so the solver cannot move them.

**Key constants** (in `constants.py`):

| Constant | Default | Description |
|----------|---------|-------------|
| `MINK_COLLISION_MIN_DIST` | 0.005 | Minimum allowed distance between geom pairs (m) |
| `MINK_COLLISION_GAIN` | 0.85 | Avoidance gain — higher = more aggressive avoidance |
| `MINK_COLLISION_DETECT_DIST` | 0.1 | Distance at which collision constraints activate (m) |

#### Per-Tick Execution Loop (`_execute_mink_loop`)

The core loop runs for up to `2 × num_steps` ticks:

```
for each tick:
    1. Advance waypoint index along both trajectories

    2. Sync mink Configuration with the simulation state
       — read full qpos from MuJoCo
       — overwrite Arm A's slots with the *planned* waypoint
         (so the QP sees where Arm A *will* be, not where it
          currently is if the controller lags slightly)

    3. Set FrameTask target for Arm B
       — FK the reference joints → read SE3 → set_target
       — restore the mink config so the QP sees *current* state

    4. Set PostureTask target = reference config
       (biases the QP solution toward the collision-free reference)

    5. Solve the QP (_solve_mink_step)
       — tasks:       [FrameTask, DampingTask, PostureTask]
       — limits:      [VelocityLimit, ConfigLimit, CollisionAvoidanceLimit]
       — constraints: [DofFreezingTask]
       — call mink.solve_ik → get joint velocities
       — integrate velocities → new q for Arm B
       — if QP fails → fall back to reference waypoint

    6. Build composite action
       — Arm A: direct joint-position target from traj_a[wp_idx]
       — Arm B: QP-solved joints (or fallback)
       — env.step([armA_joints, gripA, armB_joints, gripB])

    7. Drift-correcting sleep: sleep(max(0, target_time − elapsed))

    8. Convergence check (at last waypoint):
       — if max|q_actual − q_target| < threshold for BOTH arms → success
```

#### Public API

**`execute_parallel(traj_a, target_b, gripper_a, gripper_b, ...)`**

1. Normalises `traj_a` into a `(N, 7)` array.
2. Resolves `target_b` — if a `StampedPose`, attempts a **collision-free joint-space RRT** (see below); if the RRT fails or `traj_a` is unavailable, falls back to IK + quintic interpolation to produce a `(M, 7)` reference trajectory; if already a joint array, normalises it.
3. Resamples the shorter trajectory (cubic interpolation) so both have equal length.
4. Calls `_execute_mink_loop`.

**`execute_parallel_eef(eef_traj_a, target_b, ...)`**

Converts an EEF (Cartesian) trajectory for Arm A into joint space via sequential IK, then delegates to `execute_parallel`.

#### Usage

```python
from robosuite_franka_manipulation.planning.dual_arm import (
    DualArmParallelPlanner,
)

planner = DualArmParallelPlanner(
    env=system.env,
    env_lock=system.env_lock,
    arm_a=system.left_arm,
    arm_b=system.right_arm,
    tf_manager=system.tf_manager,
    collision_check=True,
    manipulated_object="mug_0",       # object held by arm A
)

result = planner.execute_parallel(
    traj_a=joint_segment,             # (N, 7) pre-planned joints for arm A
    target_b=handover_pre_pose,       # StampedPose for arm B
    gripper_a=1.0,                    # keep gripper A closed
    gripper_b=-1.0,                   # open gripper B
)

if result.success:
    print(f"Both arms converged in {result.steps_taken} steps")
    print(f"Collision-free: {result.collision_free}")
else:
    print(f"Errors: arm_a={result.final_pos_error_a:.4f}, "
          f"arm_b={result.final_pos_error_b:.4f}")
```

#### Key Helper Methods

| Method | Purpose |
|--------|---------|
| `_set_arm_b_target_from_joints(ref_q_b)` | FK reference joints → SE3 → set `FrameTask` target, then restore config so the QP sees the *current* state. |
| `_solve_mink_step()` | Assembles tasks/limits/constraints, calls `mink.solve_ik` (daqp solver), integrates velocity → returns Arm B's new 7-DOF config. Returns `None` on QP failure. |
| `_resolve_trajectory_b(target_b, ...)` | Converts a `StampedPose` into a collision-free joint trajectory via RRT (or quintic fallback), or normalises an existing array. |
| `_plan_collision_free_trajectory_b(...)` | Runs joint-space RRT-Connect against Arm A's final config, falls back to quintic interpolation on failure. |
| `_eef_to_joint_trajectory(arm, eef_traj)` | Converts an EEF trajectory to joint space via sequential IK (warm-started). |
| `_joint_error(arm, target_q)` | `max(\|q_actual − q_target\|)` — used for convergence checking. |

#### Joint-Space RRT for Arm B Reference Trajectory (`rrt/joint/`)

When `target_b` is a `StampedPose`, the planner uses a **joint-space RRT-Connect** algorithm to find a collision-free reference trajectory for Arm B, rather than blindly interpolating in joint space.

The joint-space planner is a thin wrapper around the **generic `rrt_connect_generic()`** core defined in `rrt_planner.py`.  The generic function implements the bi-directional RRT-Connect tree-building loop parameterised by domain-specific callbacks (distance, steer, sample, collision, segment-free, copy).  Both the SE(3) object planner (`rrt_connect_se3`) and the joint-space arm planner (`rrt_connect_joint_space`) delegate to this shared core, eliminating algorithm duplication.

##### Motivation

The previous approach — solve IK for the target pose and quintic-interpolate from the current to the target joints — could produce a reference trajectory that passes through self-collision, inter-arm collision, or environment collision states.  The reactive mink QP would then need large deviations from the reference, potentially destabilising tracking or failing entirely.

##### Algorithm

1.  **IK** — Solve IK for the target `StampedPose` to obtain `goal_q`.
2.  **Direct check** — If linear interpolation from `start_q` (Arm B's current joints) to `goal_q` is collision-free, return it immediately (fast path).
3.  **RRT-Connect** — Build two trees rooted at `start_q` and `goal_q`.  At each iteration:
    - Sample a random configuration uniformly within joint limits (with `ARM_RRT_GOAL_BIAS` probability of sampling the goal).
    - Extend tree A toward the sample; extend tree B toward the new node.
    - Collision checking sets Arm B to the candidate config and Arm A to its **final planned configuration** (`traj_a[-1]`), runs `mj_fwdPosition`, and checks for any contacts involving Arm B's geoms.
4.  **Shortcut smoothing** — Delegates to `shortcut_generic()` from `path_smoothing.py` — the same domain-agnostic shortcutting loop used by the SE(3) planner, with a joint-space collision callback.  Randomly picks two non-adjacent waypoints; if the direct segment is collision-free, removes intermediaries.  Repeated for `ARM_RRT_SMOOTHING_ITERATIONS` rounds.
5.  **Densification** — Delegates to `densify_generic()` from `path_smoothing.py` — the same arc-length-based resampling used by the SE(3) planner, with joint-space distance and linear interpolation callbacks.  Resamples the smoothed path to `num_steps` evenly-spaced waypoints.
6.  **Fallback** — If the RRT fails after `ARM_RRT_MAX_ITERATIONS`, the planner falls back to the quintic interpolation baseline.  The reactive mink QP will still handle collision avoidance at runtime.

##### Visualisation

The `RRTVisualizer` (shared with the SE(3) planner) can be passed to `rrt_connect_joint_space()` and `plan_arm_trajectory()` via the `rrt_vis` parameter.  Joint-space states are projected to 3-D end-effector positions through forward kinematics (`make_ee_pos_fn`), enabling:

- **Tree edges** shown at the EE positions corresponding to each joint config.
- **Collision samples** shown as red spheres at the rejected EE position.
- **Final / smoothed paths** drawn at the EE positions along the planned path.

The same `pos_fn` parameter on `draw_path()`, `draw_smoothed_path()`, and `draw_shortcut_path()` is used by both the SE(3) and joint-space planners, keeping the visualiser domain-agnostic.

##### Generic Path Post-Processing (`path_smoothing.py`)

Both the SE(3) and joint-space planners share three generic building blocks:

| Function | Description |
|----------|-------------|
| `shortcut_generic(path, *, segment_free_fn, max_iterations)` | Domain-agnostic random shortcutting.  Randomly selects two waypoints and removes intermediate nodes when `segment_free_fn` returns `True`. |
| `densify_generic(path, num_steps, *, distance_fn, interpolate_fn)` | Domain-agnostic arc-length resampling.  Computes cumulative arc-length, normalises to `[0, 1]`, and samples at uniform parameter values using `interpolate_fn`. |
| `bezier_smooth_generic(path, *, collision_fn, eval_segment_fn, linear_fallback_fn, …)` | Domain-agnostic per-segment Bézier smoothing with collision-based fallback to linear interpolation. |

##### SE(3) Path Smoothing (`rrt/se3/smoothing.py`)

The SE(3)-specific wrappers (`_shortcut_path`, `smooth_path`, `densify_path`) live in `rrt/se3/smoothing.py` and delegate to the generic utilities with SE(3) collision, distance, and interpolation callbacks.  Includes quaternion SLERP helpers and De Casteljau SO(3) Bézier evaluation.

The joint-space wrappers (`shortcut_joint_path`, `bezier_smooth_joint_path`, `densify_joint_path`) delegate to the same generics with joint-space callbacks.

##### Collision Checking

The collision checker (`check_arm_config_collision`) is designed for speed inside the RRT inner loop:

- Sets both arms' qpos in MuJoCo, calls `mj_fwdPosition` (cheaper than full `mj_forward`).
- Scans the contact buffer for contacts involving Arm B's geom IDs.
- Does **not** flag Arm A vs environment/object contacts (those are expected and irrelevant for Arm B's planning).
- Simulation state is backed up once before the RRT and restored in a `finally` block.

##### Constants (`constants.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `ARM_RRT_MAX_ITERATIONS` | 3000 | Maximum bi-directional RRT iterations |
| `ARM_RRT_STEP_SIZE` | 0.25 | Tree extension step (radians in joint space) |
| `ARM_RRT_GOAL_BIAS` | 0.20 | Probability of sampling the goal config |
| `ARM_RRT_COLLISION_RESOLUTION` | 0.05 | Edge checking resolution (radians between checks) |
| `ARM_RRT_CONNECT_MAX_STEPS` | 50 | Max greedy connect steps toward a new node |
| `ARM_RRT_SMOOTHING_ITERATIONS` | 150 | Number of random shortcut attempts |
| `ARM_RRT_JOINT_MARGIN` | 0.05 | Safety margin inside joint limits for sampling |

##### Why Arm A's Final Config?

The RRT checks collisions against Arm A at `traj_a[-1]` — the pose Arm A will occupy when both arms are at the handover point.  This is the most restrictive configuration (arms are closest together) and the most relevant for ensuring Arm B can safely arrive at its target without colliding with Arm A.  Transient collisions during the motion are still handled by the reactive mink QP at execution time.

### 7. Segment Assignment Planner (`assignment/`)

Solves the **optimal robot-to-segment assignment problem** for dual-arm manipulation. Given a discretised object trajectory and a set of pre-generated robot trajectories for each arm, the planner determines which arm should carry the object at every segment, minimising a combined cost of smoothness and manipulability while accounting for handover switching penalties.

#### Typed Trajectory Data (`models.py`)

All trajectory data flowing through the assignment pipeline is represented by **typed dataclasses** (not raw dictionaries), providing IDE autocompletion, static type checking, and runtime validation:

| Dataclass | Fields | Key Properties |
|-----------|--------|----------------|
| `ObjectTrajectory` | `positions`, `quaternions`, `frame_ids` | `n_waypoints` |
| `RobotTrajectory` | `eef_positions`, `eef_quaternions`, `eef_frame_ids`, `object_positions`, `object_quaternions`, `object_frame_ids`, `is_feasible`, `ik_success`, `collision_free`, `joint_configs` | `n_waypoints`, `feasibility_ratio` |
| `TrajectoryBundle` | `object_trajectory`, `robot0_trajectories`, `robot1_trajectories` | `n_segments`, `get_trajectory(robot_id, traj_idx)`, `get_robot_trajectories(robot_id)` |

All numeric fields are coerced to `np.ndarray` via `__post_init__`, so constructing from lists works seamlessly. `TrajectoryBundle` is the single entry point — returned by `load_all_trajectories()` and accepted by `compute_optimal_robot_assignment()`, `compute_cost_matrix()`, `compute_switching_cost()`, and `visualize_trajectory_feasibility()`.

```python
from robosuite_franka_manipulation.planning.assignment import (
    ObjectTrajectory, RobotTrajectory, TrajectoryBundle,
)

# Construct programmatically
bundle = TrajectoryBundle(
    object_trajectory=ObjectTrajectory(positions=pos, quaternions=quat, frame_ids=fids),
    robot0_trajectories=[RobotTrajectory(...)],
    robot1_trajectories=[RobotTrajectory(...)],
)
assert bundle.n_segments == len(pos) - 1
assert bundle.get_trajectory(0, 0).feasibility_ratio >= 0.0
```

#### Problem Formulation

An object trajectory with $N$ waypoints defines $N - 1$ **segments** $[p_i, p_{i+1}]$. Each segment can be executed by either Robot 0 or Robot 1 using one of several pre-computed grasp trajectories. The optimisation selects a sequence of *(robot, trajectory)* pairs that minimises total cost subject to:

1. **Intra-arm trajectory continuity** — an arm cannot switch to a different grasp trajectory mid-task; only a handover (arm switch) may change the active trajectory.
2. **Transfer feasibility constraints** — certain *(from_robot, from_traj, to_robot, to_traj)* transitions can be explicitly blocked.
3. **Initial robot constraint** — optionally force the first segment to a specific arm.

#### Cost Model

Each segment assignment $A_{i,j}$ (segment $i$, state $j$) incurs a **task cost**:

$$C(A_{i,j}) = \alpha \cdot C_1(A_{i,j}) + \beta \cdot C_2(A_{i,j})$$

| Component | Formula | Meaning |
|-----------|---------|---------|
| $C_1$ — **Smoothness** | $\|q_{i+1} - q_i\|^2$ | Squared joint displacement over the segment |
| $C_2$ — **Manipulability** | $\sum_k \frac{1}{w(q_k) + \varepsilon}$ | Reciprocal Yoshikawa manipulability (low → penalised) |

When a Jacobian function is available, $w(q)$ is the Yoshikawa index $\sqrt{\det(J J^\top)}$; otherwise a simple joint-distance-from-centre proxy is used.

Infeasible segments (IK failures or collisions) receive $\infty$ cost.

**Switching cost**: When the assignment switches arms between consecutive segments, a **handover primitive** is generated and its smoothness + manipulability cost is evaluated via IK. Additionally, an **inter-arm collision check** verifies that the two arms do not spatially interfere at the handover meeting point (the moment when the receiver arm reaches the grasp pose while the transfer arm holds the object). If inter-arm collision is detected, the handover is rejected ($\infty$ cost). If no IK solver is provided (e.g. `--no-env`), a fallback constant `ASSIGNMENT_FALLBACK_SWITCH_COST` is used and collision checking is skipped.

#### Viterbi Dynamic Programming Solver

The solver uses a **Viterbi-style forward pass** over the $N-1$ segments:

```
Initialisation:
    J*(0, s) = cost_matrix[0, s]   for each state s (filtered by initial_robot)

Forward pass (i = 1 … N-2):
    For each current state cs:
        For each previous state ps:
            - Skip if same arm but different trajectory (continuity constraint)
            - Compute switching_cost(ps → cs)
            - total = J*(i-1, ps) + switching_cost
            - Track best predecessor

        J*(i, cs) = best_total + cost_matrix[i, cs]

Termination:
    best_final = argmin_s J*(N-2, s)

Backtrack:
    Recover optimal sequence from predecessor pointers
```

The result is an `AssignmentResult` containing:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `List[int]` | Robot ID (0 or 1) for each segment |
| `trajectory_paths` | `List[str]` | NPZ filename for each segment (e.g. `"robot0_trajectory_6.npz"`) |
| `total_cost` | `float` | Globally optimal total cost |
| `task_costs` | `np.ndarray` | Per-segment task cost |
| `switching_costs` | `np.ndarray` | Per-segment switching cost |
| `handover_points` | `List[int]` | Segment indices where arm switches occur |
| `cost_matrix` | `np.ndarray` | `(N-1, 2)` — best cost per robot per segment |
| `trajectory_matrix` | `np.ndarray` | `(N-1, 2)` — best trajectory index per robot per segment |

#### Inter-Arm Collision Checking

During handover cost evaluation, the planner checks for **physical interference between the two robot arms**. This is critical because a handover that looks cost-optimal may be physically impossible if the arms collide at the meeting point.

The check works as follows:

1. IK is solved for both arms at their respective handover poses (transfer-hold and receiver-final-approach).
2. Both joint configurations are set simultaneously in MuJoCo.
3. A forward pass (`mj_forward`) updates the contact state.
4. `check_inter_arm_collision` scans the contact buffer for any contact between Robot 0 geoms (arm + gripper) and Robot 1 geoms (arm + gripper).
5. If any inter-arm contact is detected, the handover is rejected (cost = $\infty$), forcing the DP solver to find an alternative.
6. The simulation state is always restored after each check.

This prevents the planner from assigning handovers at workspace configurations where the arms would physically interfere, which would otherwise only be caught at execution time by the reactive `DualArmParallelPlanner`.

#### Factory Functions

Three factory functions create closures around the simulation environment for IK, Jacobian, and collision evaluation:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `make_ik_solver_fn(env, tf_manager, env_lock)` | `(StampedPose, int) → Optional[np.ndarray]` | Creates an IK solver that applies orientation correction and delegates to `enumerate_ik_solutions_for_arm` |
| `make_jacobian_fns(env, env_lock)` | `Dict[int, (np.ndarray) → np.ndarray]` | Creates per-robot Jacobian closures mapping joint configs to 6×7 Jacobians via `compute_arm_jacobian` |
| `make_collision_check_fn(env, env_lock, manipulated_object=None)` | `(np.ndarray, np.ndarray) → bool` | Creates a collision checker that sets both arms' joint configs, runs MuJoCo forward, and returns whether they are collision-free (inter-arm + arm-environment, ignoring the manipulated object) |

When `--no-env` is used, all three are `None` and the planner falls back to simpler heuristics (joint-distance manipulability proxy, constant switching cost, no collision checking).

#### Transfer Feasibility Constraints

The `transfer_feasibility` field in the experiment JSON allows explicit blocking of specific handover transitions. It should be either `null` (no constraints) or a dict with string keys of the form `"from_robot,from_traj,to_robot,to_traj"` mapping to boolean values:

```json
"transfer_feasibility": {
    "0,3,1,5": false,
    "1,2,0,0": false
}
```

Keys mapping to `false` block that specific handover. The parser decodes string keys to `(int, int, int, int)` tuples and checks both directions during switching-cost evaluation.

#### Configuration

Experiment JSON `assignment` block:

```json
"assignment": {
    "alpha": 10.0,
    "beta": 1.0,
    "epsilon": 1e-06,
    "initial_robot": null,
    "transfer_feasibility": null
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `alpha` | `10.0` | Smoothness weight ($\|Δq\|^2$) |
| `beta` | `1.0` | Manipulability weight ($Σ 1/(w + ε)$) |
| `epsilon` | `1e-6` | Numerical floor for manipulability reciprocal |
| `initial_robot` | `null` | Force first segment to this robot (0 or 1); `null` = auto |
| `transfer_feasibility` | `null` | Per-transition feasibility dict or `null` |

Constants in `constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `ASSIGNMENT_ALPHA` | `10.0` | Default smoothness weight |
| `ASSIGNMENT_BETA` | `1.0` | Default manipulability weight |
| `ASSIGNMENT_EPSILON` | `1e-6` | Default manipulability floor |
| `ASSIGNMENT_FALLBACK_SWITCH_COST` | `1e6` | Switching cost when no IK solver available |

#### I/O

Results are persisted as `.npz` files:

- **`save_assignment_result(result, filepath)`** — serialize all fields to NumPy arrays.
- **`load_assignment_result(filepath)`** — deserialize back to `AssignmentResult`.
- **`load_all_trajectories(data_dir)`** — load `object_trajectory.npz` and all `robot{0,1}_trajectory_*.npz` files into a `TrajectoryBundle` dataclass with typed `ObjectTrajectory` and `RobotTrajectory` fields.

#### CLI Usage

```bash
# Full run with environment (accurate IK + Jacobian costs):
python scripts/planning/segment_assignment.py \
    --experiment experiments/flatPlate_table_to_shelf.json

# Quick dry-run without environment:
python scripts/planning/segment_assignment.py \
    --experiment experiments/flatPlate_table_to_shelf.json --no-env

# Override cost weights:
python scripts/planning/segment_assignment.py \
    --experiment experiments/flatPlate_table_to_shelf.json \
    --alpha 15.0 --beta 2.0

# Skip matplotlib plots:
python scripts/planning/segment_assignment.py \
    --experiment experiments/flatPlate_table_to_shelf.json --no-plot
```

#### Structured Logging

All assignment submodules use Python's `logging` module with the standard `__name__`-based logger hierarchy:

| Level | Examples |
|-------|----------|
| `INFO` | Cost matrix dimensions, finite-entry counts, assignment total cost & handover count, trajectory loading summaries |
| `DEBUG` | Per-segment cost breakdowns (smoothness / manipulability), per-state feasibility ratios, IK failures, collision rejections, switching cost evaluations, Viterbi forward-pass terminal state |

Enable with `--debug` on the CLI or `logging.getLogger('robosuite_franka_manipulation.planning.assignment').setLevel(logging.DEBUG)` programmatically.

#### Submodule Summary

| Module | Purpose |
|--------|---------|
| `models.py` | `ObjectTrajectory`, `RobotTrajectory`, `TrajectoryBundle` dataclasses |
| `trajectory_io.py` | NPZ disk I/O — `load_all_trajectories()` → `TrajectoryBundle` |
| `cost.py` | Segment cost, cost-matrix construction, handover primitive cost, switching cost |
| `solver.py` | Viterbi DP solver — `compute_optimal_robot_assignment()` |
| `result.py` | `AssignmentResult` dataclass + NPZ save/load |
| `factories.py` | Environment-bound closures for IK, Jacobian, and collision checking |

#### Workflow Position

```
trajectory_generation.py  →  data/<experiment>/robot{0,1}_trajectory_*.npz
    └── segment_assignment.py  →  data/<experiment>/assignment_results.npz
        └── dual_arm_handover_traj.py  (reads assignment to execute handovers)
```

---

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
