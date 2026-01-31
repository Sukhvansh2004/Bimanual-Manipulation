"""Feasibility checks for planning.

This module provides utilities to evaluate whether desired end-effector poses
are reachable (via IK) and collision-free (via MuJoCo contact checking).

Notes
- This repo uses robosuite, which exposes a MuJoCo simulation via ``env.sim``.
- In the default dual-arm handoff environment, arm joints are named
  ``robot0_joint1..7`` and ``robot1_joint1..7``.
- End-effector sites are named ``gripper{robot_id}_right_grip_site``.
"""

from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import numpy as np
import mujoco

from robosuite_franka_manipulation.constants import DEFAULT_ROBOT_ACTION_DIM
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager

from robosuite_franka_manipulation.planning.utils.common import ArrayLike
from robosuite_franka_manipulation.planning.utils import mujoco_utils
from robosuite_franka_manipulation.planning.utils import ik_utils
from robosuite_franka_manipulation.planning.utils import collision_utils


def check_dual_arm_pose_feasibility(
    env: DualArmEnvironment,
    eef_pose_robot0: Union[StampedPose, dict, Tuple[ArrayLike, ArrayLike], ArrayLike],
    eef_pose_robot1: Union[StampedPose, dict, Tuple[ArrayLike, ArrayLike], ArrayLike],
    *,
    input_frame: str = "world",
    tf_manager: Optional[TransformManager] = None,
    env_lock=None,
    arm: str = "right",
    # IK discretization / search params
    num_seed_configs: int = 24,
    elbow_discretization: int = 8,
    max_solutions_per_arm: int = 24,
    # IK solver params
    pos_tol: float = 1e-3,
    ori_tol: float = 5e-3,
    max_iters: int = 200,
    damping: float = 1e-2,
    step_size: float = 0.6,
    # Collision params
    contact_dist_threshold: float = 1e-6,
    # Pairing caps
    max_pairs: int = 256,
) -> List[List[Union[np.ndarray, bool]]]:
    f"""Check feasibility for *all* (discretized) dual-arm joint solutions at given EEF poses.

    This function:
    1) Runs IK for each arm from multiple seed configurations to obtain a set of
       joint solutions for robot0 and robot1.
    2) Forms joint-solution pairs (cartesian product) up to ``max_pairs``.
    3) For each pair, sets the MuJoCo qpos, runs forward kinematics, and checks
       for contacts involving either robot.

    Args:
        env: Robosuite environment (or wrapper) exposing ``env.sim.model`` and ``env.sim.data``.
        eef_pose_robot0: Desired EEF pose for robot0, in ``input_frame``.
        eef_pose_robot1: Desired EEF pose for robot1, in ``input_frame``.
        input_frame: Frame the input poses are expressed in (default: "world").
        tf_manager: Optional TransformManager. If not provided, a temporary one is created.
        env_lock: Optional threading.RLock used to guard sim state.

    Returns:
        A list of entries: ``[q_arm0 ({DEFAULT_ROBOT_ACTION_DIM},), q_arm1 ({DEFAULT_ROBOT_ACTION_DIM},), feasible_bool]``.

    Notes
    - If either arm has no IK solutions, returns an empty list.
    - Feasible means: IK converged for both arms AND no MuJoCo contact involving
      either robot is present at that configuration.
    """

    if tf_manager is None:
        tf_manager = TransformManager(env, env_lock=env_lock)

    q_solutions_0 = ik_utils.enumerate_ik_solutions_for_arm(
        env,
        robot_id=0,
        target_pose=eef_pose_robot0,
        target_frame=input_frame,
        arm=arm,
        tf_manager=tf_manager,
        env_lock=env_lock,
        num_seed_configs=num_seed_configs,
        elbow_discretization=elbow_discretization,
        max_solutions=max_solutions_per_arm,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        max_iters=max_iters,
        damping=damping,
        step_size=step_size,
    )

    q_solutions_1 = ik_utils.enumerate_ik_solutions_for_arm(
        env,
        robot_id=1,
        target_pose=eef_pose_robot1,
        target_frame=input_frame,
        arm=arm,
        tf_manager=tf_manager,
        env_lock=env_lock,
        num_seed_configs=num_seed_configs,
        elbow_discretization=elbow_discretization,
        max_solutions=max_solutions_per_arm,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        max_iters=max_iters,
        damping=damping,
        step_size=step_size,
    )

    if len(q_solutions_0) == 0 or len(q_solutions_1) == 0:
        return []

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        jids0 = mujoco_utils.get_arm_joint_ids(model, 0)
        jids1 = mujoco_utils.get_arm_joint_ids(model, 1)
        qpos_idx0 = mujoco_utils.get_joint_qpos_indices(model, jids0)
        qpos_idx1 = mujoco_utils.get_joint_qpos_indices(model, jids1)

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        results: List[List[Union[np.ndarray, bool]]] = []
        try:
            pair_count = 0
            for q0 in q_solutions_0:
                for q1 in q_solutions_1:
                    pair_count += 1
                    if pair_count > int(max_pairs):
                        break

                    data.qpos[:] = qpos_backup
                    data.qvel[:] = qvel_backup
                    mujoco_utils.set_qpos(data, qpos_idx0, np.asarray(q0, dtype=float))
                    mujoco_utils.set_qpos(data, qpos_idx1, np.asarray(q1, dtype=float))
                    
                    # Unwrap for native call
                    mujoco.mj_forward(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

                    feasible = not collision_utils.in_collision(env, contact_dist_threshold=contact_dist_threshold, env_lock=None)
                    results.append([np.asarray(q0, dtype=float), np.asarray(q1, dtype=float), bool(feasible)])

                if pair_count > int(max_pairs):
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            
            # Unwrap for native call
            mujoco.mj_forward(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

        return results
