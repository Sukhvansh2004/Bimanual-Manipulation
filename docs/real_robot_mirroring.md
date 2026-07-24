# Real-robot mirroring (digital twin → hardware)

Mirrors the robosuite simulation onto the physical dual-arm Franka setup: every simulation
step's per-arm joint targets + gripper command are forwarded to the `robosuite_bridge`
ROS 2 package, and the simulation is paced to the real hardware via joint-state feedback.

Everything is wired through the single `env_step()` choke point
(`bimanual_manipulation.utils.env_utils`), so all execution paths that already go through
`env_step` (system_manager execution, teleop, trajectory_executor) mirror automatically.

## Enabling it

Mirroring is gated **two independent ways**, so it only ever drives the robot for real
execution:

1. **Opt-in env var (master switch), off by default** — because `env_step` is also used by
   many *offline* planning/visualization pipelines that must never move the real robot:
   ```bash
   export BIMANUAL_REAL_ROBOT=1
   ```
   With it unset (or `0`), `env_step` behaves exactly as before (sim only).

2. **Per-env binding** — even with the env var on, mirroring only fires for the env
   explicitly designated as the execution env via `mark_real_robot_env(env)`. `SystemManager`
   marks `self.env` and deliberately does **not** mark `self.planning_env`, so the separate
   planning env used for feasibility rollouts (which also calls `env_step` in the same
   process) never reaches the hardware. If you drive a custom env directly, call
   `mark_real_robot_env(your_env)` yourself.

## Prerequisites

1. Source ROS 2 Humble in the same shell as the sim (so `rclpy` and the bridge topics are
   reachable from the conda env):
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/franka_ros2_ws/install/setup.bash
   ```
2. Bring up the real dual-arm bridge (real hardware, FCI unlocked, e-stop in reach):
   ```bash
   ros2 launch robosuite_bridge bimanual_robosuite.launch.py
   ```
3. Run the bimanual pipeline as usual, with `BIMANUAL_REAL_ROBOT=1` set.

## How it maps

| sim robot | gripper (sim)        | → real arm  | real gripper        |
|-----------|----------------------|-------------|---------------------|
| robot 0   | PandaGripper (jaw)   | `right_arm` | Franka Hand         |
| robot 1   | PandaSuctionGripper  | `left_arm`  | cobot vacuum pump   |

- Arm: the 7 sim joint targets (radians, absolute JOINT_POSITION mode) are published as-is
  (Panda↔FR3 share joint order/signs).
- Gripper: the sim gripper scalar is forwarded; `> 0` grips/closes (suction on), `<= 0`
  releases/opens (suction off) — matching the bridge.

## Safety behaviour

- **Velocity-scaled command time (no violent jumps):** every command carries a per-command
  `time_from_start` = `max(min_time, distance / desired_speed)`, where `distance` is the
  largest per-joint move from the real arm's *current measured* position to the target. Far
  targets (first command, post-`reset()` teleport) get proportionally more time and ease in
  at a bounded joint speed; small streaming deltas get `min_time`. This is sent as an
  optional 9th element on the action topic. **Belt-and-suspenders:** the bridge node
  *also* independently velocity-limits (`max_joint_speed`) using its own joint-state feed,
  so any publisher — even a manual `ros2 topic pub` — is protected.
- **Feedback gate (lockstep):** after each command the sim step *blocks* until the real
  arms reach the commanded config within a tolerance (or a timeout — which is stretched to
  at least the commanded move time so a slow eased-in move can finish). The real joint
  states are **not** written into the sim (that would corrupt sim physics / object
  contacts) — the sim just waits.

## Tuning (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `BIMANUAL_REAL_ROBOT` | `0` | Master switch. `1`/`true` enables mirroring. |
| `BIMANUAL_REAL_DESIRED_SPEED` | `0.6` | Target joint speed (rad/s) used to compute each command's `time_from_start`. |
| `BIMANUAL_REAL_MIN_TIME` | `0.1` | Floor on per-command time (s) for small streaming deltas. |
| `BIMANUAL_REAL_BLIND_TIME` | `4.0` | Conservative command time (s) used when no joint feedback is available yet. |
| `BIMANUAL_REAL_GATE` | `1` | Feedback gate on/off. `0` = monitor-only (never block). |
| `BIMANUAL_REAL_GATE_TOL` | `0.1` | Per-joint convergence tolerance (rad) for the gate. |
| `BIMANUAL_REAL_GATE_TIMEOUT` | `5.0` | Extra seconds to wait for settling (added on top of the commanded move time) before advancing anyway. |

The bridge node (`robosuite_bridge/action_follower_node`) independently velocity-limits
every goal via these parameters:
- `max_joint_speed` (default `0.6` rad/s) — soft comfort cap applied to all joints.
- `joint_velocity_limits` (default FR3 limits `[2.62,2.62,2.62,2.62,5.26,4.18,5.26]` rad/s)
  and `velocity_safety_factor` (default `0.5`) — the hardware ceiling.

Each joint's effective speed is `min(max_joint_speed, limit_i * safety_factor)`, and the
goal duration is stretched so no joint exceeds it — so a joint can never be commanded past
its hardware velocity limit even if `max_joint_speed` or a requested time is set aggressively.

## Notes / limitations

- Under `use_fake_hardware:=true`, MuJoCo's mock system doesn't integrate the effort
  commands into motion, so `franka/joint_states` never moves and the gate will time out
  every step. That's a fake-hardware artifact only; set `BIMANUAL_REAL_GATE=0` if you want
  to exercise the pipeline against fake hardware.
- The real robot lags the sim by one step by construction (sim steps, then commands real,
  then waits) — the gate keeps that lag bounded.
- The bridge now derives each goal's duration per-command (from the 9th action element
  and/or its own velocity limit), so the `trajectory_time_from_start` launch arg only acts
  as a *minimum* floor — you rarely need to tune it now; tune `BIMANUAL_REAL_DESIRED_SPEED`
  / the bridge's `max_joint_speed` instead to make motion faster or gentler.
