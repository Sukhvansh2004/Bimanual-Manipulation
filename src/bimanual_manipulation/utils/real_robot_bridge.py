"""Real-robot mirroring bridge (sim -> real hardware).

Publishes each simulation step's per-arm joint targets + gripper command to the
``robosuite_bridge`` ROS 2 action-follower topics so the physical dual-arm Franka setup
mirrors whatever happens in the robosuite simulation.

Design / safety notes
---------------------
* **Opt-in, off by default.** Mirroring only happens when the ``BIMANUAL_REAL_ROBOT``
  environment variable is truthy (``1``/``true``/``yes``/``on``). This matters because
  ``env_step`` is also called by many *offline* planning/visualization pipelines that must
  never move the real robot -- those simply don't set the env var.
* **Transport.** Publishes ``std_msgs/Float64MultiArray`` (7 joint radians + 1 gripper
  command) to ``/<arm>/robosuite_action`` -- exactly the contract
  ``robosuite_bridge/action_follower_node`` consumes. Uses ``rclpy`` directly (works from
  the conda env as long as ROS 2 Humble is sourced).
* **Arm mapping** (confirmed with the setup): sim robot 0 (PandaGripper, parallel-jaw)
  -> real ``right_arm``; sim robot 1 (PandaSuctionGripper) -> real ``left_arm``.
* **Gripper convention** already matches the bridge: ``> 0`` grips/closes (or turns suction
  on), ``<= 0`` releases/opens (suction off). The raw sim gripper scalar is forwarded and
  the bridge applies the threshold.
* **Panda(sim) -> FR3(real) joints** are forwarded directly (same 7-DOF order/signs/zero).
* **Startup / teleport safety (velocity-scaled time).** Every command carries a per-command
  ``time_from_start`` = ``max(min_time, distance / desired_speed)``, where distance is the
  largest per-joint move from the real arm's *current measured* position to the target. So a
  far-away target (first command, or a post-``env.reset()`` teleport) is given proportionally
  more time and eases in at a bounded joint speed instead of moving violently; small
  streaming deltas get ``min_time``. (The bridge node independently velocity-limits too, so
  this is defense in depth.)
* **Per-env binding.** Mirroring only fires for the env explicitly marked as the real
  execution env via ``mark_real_robot_env(env)``. This is essential because the pipeline
  runs a *separate* ``planning_env`` for feasibility rollouts in the same process -- that env
  is never marked, so its ``env_step`` calls never reach the hardware.
* **Feedback: monitor + gate (keep sim and real in lockstep).** After publishing each
  step's targets, the bridge reads the real arms' measured joint states and *blocks* the
  simulation step until they converge to within a tolerance of the commanded configuration
  (or a timeout elapses). This paces the sim to the real hardware -- preferring alignment /
  safety over throughput -- WITHOUT ever writing measured joints back into the sim (which
  would corrupt sim dynamics and object contacts). Configurable / disable-able via env vars.

Failures are swallowed and logged: a ROS hiccup must never crash the simulation loop.
"""

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# Sim robot index -> real arm namespace (see module docstring).
ARM_NAMESPACES = {0: 'right_arm', 1: 'left_arm'}
NUM_ARM_JOINTS = 7


def real_robot_enabled() -> bool:
    """True if real-robot mirroring is requested via the BIMANUAL_REAL_ROBOT env var."""
    return os.environ.get('BIMANUAL_REAL_ROBOT', '0').strip().lower() in (
        '1', 'true', 'yes', 'on')


class RealRobotBridge:
    """Publishes sim per-arm joint/gripper targets to the robosuite_bridge action topics."""

    def __init__(self):
        self._ok = False
        self._node = None
        self._pubs = {}
        self._joint_state = {}          # namespace -> {joint_name: position}
        self._last_target = {}          # namespace -> np.ndarray(7) last commanded joint target

        # Each command carries a per-command time_from_start = max(min_time, distance /
        # desired_speed), where distance is the largest per-joint move from the arm's CURRENT
        # measured position to the target. So a far-away target (e.g. the first command, or a
        # post-reset teleport) is given proportionally more time and eases in instead of
        # moving violently; small streaming deltas get `min_time`. The bridge node also
        # velocity-limits independently, so this is the primary guard plus a backup.
        self._desired_speed = float(os.environ.get('BIMANUAL_REAL_DESIRED_SPEED', '0.6'))  # rad/s
        self._min_time = float(os.environ.get('BIMANUAL_REAL_MIN_TIME', '0.1'))            # s
        # Used only when joint feedback isn't available yet (can't measure distance).
        self._blind_time = float(os.environ.get('BIMANUAL_REAL_BLIND_TIME', '4.0'))        # s

        # Feedback gate: after each command, wait until the real arms are within
        # `gate_tol` rad of the commanded config before letting the sim advance (safety /
        # lockstep priority). Set BIMANUAL_REAL_GATE=0 for monitor-only (never block).
        self._gate_enabled = os.environ.get('BIMANUAL_REAL_GATE', '1').strip().lower() in (
            '1', 'true', 'yes', 'on')
        self._gate_tol = float(os.environ.get('BIMANUAL_REAL_GATE_TOL', '0.1'))       # rad
        self._gate_timeout = float(os.environ.get('BIMANUAL_REAL_GATE_TIMEOUT', '5.0'))  # s
        self._gate_poll_dt = 0.02

        try:
            import rclpy
            from rclpy.node import Node  # noqa: F401  (imported for clarity)
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray

            self._rclpy = rclpy
            self._Float64MultiArray = Float64MultiArray

            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node('bimanual_real_robot_bridge')

            for _rid, ns in ARM_NAMESPACES.items():
                self._pubs[ns] = self._node.create_publisher(
                    Float64MultiArray, f'/{ns}/robosuite_action', 10)
                self._node.create_subscription(
                    JointState, f'/{ns}/franka/joint_states',
                    (lambda msg, ns=ns: self._on_joint_state(ns, msg)), 10)

            self._ok = True
            logger.warning(
                'RealRobotBridge ACTIVE: mirroring sim -> REAL arms %s. The physical robot '
                'will move.', ARM_NAMESPACES)
        except Exception as exc:  # noqa: BLE001 - never let ROS setup crash the sim
            logger.error('RealRobotBridge init failed (%s); real mirroring disabled.', exc)
            self._ok = False

    # -- ROS plumbing -------------------------------------------------------------------

    def _on_joint_state(self, namespace, msg):
        self._joint_state[namespace] = dict(zip(msg.name, msg.position))

    def _spin_once(self):
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:  # noqa: BLE001
            pass

    def _current_real_joints(self, namespace, side):
        names = [f'{side}_fr3_joint{i}' for i in range(1, NUM_ARM_JOINTS + 1)]
        js = self._joint_state.get(namespace)
        if not js:
            return None
        try:
            return np.array([js[n] for n in names], dtype=float)
        except KeyError:
            return None

    def _publish(self, namespace, joints, gripper, time_from_start):
        # 9-element message: 7 joints + gripper + per-command time_from_start (seconds).
        msg = self._Float64MultiArray()
        msg.data = [float(x) for x in joints] + [float(gripper), float(time_from_start)]
        self._pubs[namespace].publish(msg)

    def _command_time(self, namespace, side, target):
        """Velocity-scaled time for this move: max(min_time, distance / desired_speed)."""
        current = self._current_real_joints(namespace, side)
        if current is None:
            # No feedback yet -> assume it could be far; give a conservative slow time.
            return self._blind_time
        distance = float(np.max(np.abs(target - current)))
        if self._desired_speed <= 0.0:
            return self._min_time
        return max(self._min_time, distance / self._desired_speed)

    # -- public API ---------------------------------------------------------------------

    def reset(self):
        """Clear commanded-target state (call after env.reset()); next command re-eases in."""
        self._last_target.clear()

    def mirror(self, env, full_action):
        """Forward the resolved sim action to the real arms. Never raises."""
        if not self._ok:
            return
        try:
            self._mirror(env, full_action)
        except Exception as exc:  # noqa: BLE001 - sim loop must survive ROS errors
            logger.warning('RealRobotBridge.mirror failed: %s', exc)

    def _mirror(self, env, full_action):
        full_action = np.asarray(full_action, dtype=float).ravel()

        self._spin_once()  # refresh measured joint states before computing move times
        max_time = 0.0
        idx = 0
        for robot_id, robot in enumerate(env.robots):
            robot_dim = robot.action_dim
            gripper_dof = sum(
                robot.gripper[a].dof for a in robot.arms if a in robot.gripper)
            arm_dim = robot_dim - gripper_dof

            arm_block = full_action[idx:idx + arm_dim]
            gripper_val = full_action[idx + arm_dim] if gripper_dof > 0 else 0.0
            idx += robot_dim

            namespace = ARM_NAMESPACES.get(robot_id)
            if namespace is None or namespace not in self._pubs:
                continue
            if len(arm_block) < NUM_ARM_JOINTS:
                continue
            target = np.array(arm_block[:NUM_ARM_JOINTS], dtype=float)
            side = namespace.split('_')[0]

            t = self._command_time(namespace, side, target)
            self._publish(namespace, target, float(gripper_val), t)
            self._last_target[namespace] = target
            max_time = max(max_time, t)

        self._spin_once()

        # Feedback gate: pace the sim to the real hardware. Wait at least as long as the
        # longest move we just commanded (so a slow eased-in move completes), then up to
        # gate_timeout more for tracking to settle.
        self._await_convergence(min_wait=max_time)

    def _await_convergence(self, min_wait=0.0):
        """Block until the real arms reach the commanded config (within tol), or timeout."""
        if not self._gate_enabled or not self._last_target:
            return
        deadline = time.monotonic() + max(self._gate_timeout, min_wait + self._gate_timeout)
        while time.monotonic() < deadline:
            self._spin_once()
            worst = self._worst_joint_error()
            if worst is not None and worst <= self._gate_tol:
                return
            time.sleep(self._gate_poll_dt)
        worst = self._worst_joint_error()
        logger.warning(
            'RealRobotBridge: real arms did not converge to the sim config in time '
            '(worst joint error %s rad, tol %.3f); advancing the sim anyway.',
            f'{worst:.3f}' if worst is not None else 'unknown (no joint_states)',
            self._gate_tol)

    def _worst_joint_error(self):
        """Max per-joint |measured - commanded| across both arms, or None if unavailable."""
        worst = None
        for namespace, target in self._last_target.items():
            side = namespace.split('_')[0]
            current = self._current_real_joints(namespace, side)
            if current is None:
                return None
            err = float(np.max(np.abs(current - target)))
            worst = err if worst is None else max(worst, err)
        return worst


# Module-level singleton, created lazily only when mirroring is enabled.
_real_bridge = None

# ids of env objects explicitly designated as the REAL execution env. Only these mirror --
# the pipeline's separate planning_env is never marked, so its env_step calls stay sim-only.
_mirror_env_ids = set()


def mark_real_robot_env(env):
    """Designate `env` as the real-execution env whose env_step calls mirror to hardware.

    Call this once on the execution env (e.g. system.env) -- NOT on the planning env. Safe
    to call unconditionally; it only has an effect when BIMANUAL_REAL_ROBOT is enabled.
    """
    _mirror_env_ids.add(id(env))


def unmark_real_robot_env(env):
    """Stop mirroring `env` (e.g. before closing it)."""
    _mirror_env_ids.discard(id(env))


def is_real_robot_env(env) -> bool:
    return id(env) in _mirror_env_ids


def get_real_robot_bridge():
    """Return the shared RealRobotBridge, creating it on first use if enabled (else None)."""
    global _real_bridge
    if _real_bridge is None and real_robot_enabled():
        _real_bridge = RealRobotBridge()
    return _real_bridge


def reset_real_robot_bridge():
    """Reset per-arm command state (call after env.reset()); next command re-eases in."""
    if _real_bridge is not None:
        _real_bridge.reset()
