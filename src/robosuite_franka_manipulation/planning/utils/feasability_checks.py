"""Feasibility checks for planning.

This module provides utilities to evaluate whether desired end-effector poses
are reachable (via IK) and collision-free (via MuJoCo contact checking).

Notes
- This repo uses robosuite, which exposes a MuJoCo simulation via ``env.sim``.
- In the default dual-arm handoff environment, arm joints are named
  ``robot0_joint1..7`` and ``robot1_joint1..7``.
- End-effector sites are named ``gripper{robot_id}_right_grip_site``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import mujoco

from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation.transforms import StampedPose, TransformManager


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class DualArmFeasibilityResult:
	joint_angles_arm0: np.ndarray
	joint_angles_arm1: np.ndarray
	feasible: bool


def _as_stamped_pose(pose: Union[StampedPose, dict, Tuple[ArrayLike, ArrayLike], ArrayLike], frame_id: str) -> StampedPose:
	"""Normalize pose input into a StampedPose.

	Supported inputs:
	- StampedPose
	- dict with keys {'pos'/'position', 'quat'/'quaternion'}
	- ((pos, quat)) tuple
	- flat length-7 (pos[3] + quat[4]) array-like
	"""
	if isinstance(pose, StampedPose):
		return pose

	if isinstance(pose, dict):
		pos = pose.get("pos", pose.get("position"))
		quat = pose.get("quat", pose.get("quaternion"))
		if pos is None or quat is None:
			raise ValueError("Pose dict must contain 'pos'/'position' and 'quat'/'quaternion'.")
		return StampedPose(position=np.asarray(pos, dtype=float), quaternion=np.asarray(quat, dtype=float), frame_id=frame_id)

	if isinstance(pose, tuple) and len(pose) == 2:
		pos, quat = pose
		return StampedPose(position=np.asarray(pos, dtype=float), quaternion=np.asarray(quat, dtype=float), frame_id=frame_id)

	arr = np.asarray(pose, dtype=float).reshape(-1)
	if arr.size != 7:
		raise ValueError("Pose must be a StampedPose, dict, (pos,quat) tuple, or a flat length-7 array [x,y,z,qx,qy,qz,qw].")
	return StampedPose(position=arr[:3], quaternion=arr[3:], frame_id=frame_id)


def _get_arm_joint_ids(model, robot_id: int) -> List[int]:
	prefix = f"robot{robot_id}_joint"
	joint_ids = [j for j in range(model.njnt) if (model.joint(j).name or "").startswith(prefix)]
	# Ensure ordering joint1..joint7
	joint_ids.sort(key=lambda j: int((model.joint(j).name or "").replace(prefix, "") or "0"))
	return joint_ids


def _get_joint_qpos_indices(model, joint_ids: Sequence[int]) -> np.ndarray:
	idx: List[int] = []
	for j in joint_ids:
		qposadr = model.joint(j).qposadr
		# robosuite binding_utils returns a length-1 ndarray for *_adr fields
		if isinstance(qposadr, np.ndarray):
			idx.append(int(qposadr.reshape(-1)[0]))
		else:
			idx.append(int(qposadr))
	return np.asarray(idx, dtype=int)


def _get_joint_dof_indices(model, joint_ids: Sequence[int]) -> np.ndarray:
	idx: List[int] = []
	for j in joint_ids:
		dofadr = model.joint(j).dofadr
		if isinstance(dofadr, np.ndarray):
			idx.append(int(dofadr.reshape(-1)[0]))
		else:
			idx.append(int(dofadr))
	return np.asarray(idx, dtype=int)


def _get_joint_limits(model, joint_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
	# model.jnt_range is (njnt, 2)
	jnt_range = np.asarray(model.jnt_range, dtype=float)
	lo = np.asarray([jnt_range[j, 0] for j in joint_ids], dtype=float)
	hi = np.asarray([jnt_range[j, 1] for j in joint_ids], dtype=float)
	return lo, hi


def _eef_site_name(robot_id: int, arm: str = "right") -> str:
	# In this repo's robosuite configs we use the single "right" arm.
	# Site naming convention from env inspection: gripper{robot_id}_right_grip_site
	return f"gripper{robot_id}_{arm}_grip_site"


def _raw_mujoco_model(model):
	"""Return raw ``mujoco.MjModel`` from robosuite wrapper, if needed."""
	return getattr(model, "_model", model)


def _raw_mujoco_data(data):
	"""Return raw ``mujoco.MjData`` from robosuite wrapper, if needed."""
	return getattr(data, "_data", data)


def _get_qpos(data, qpos_idx: np.ndarray) -> np.ndarray:
	qpos = np.asarray(data.qpos)
	if qpos.ndim == 2 and qpos.shape[1] == 1:
		return np.asarray(qpos[qpos_idx, 0], dtype=float)
	return np.asarray(qpos[qpos_idx], dtype=float)


def _set_qpos(data, qpos_idx: np.ndarray, q: np.ndarray) -> None:
	q = np.asarray(q, dtype=float).reshape(-1)
	qpos = data.qpos
	if np.asarray(qpos).ndim == 2:
		qpos[qpos_idx, 0] = q
	else:
		qpos[qpos_idx] = q


def _compute_site_error(model, data, site_id: int, target_pos: np.ndarray, target_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	cur_pos = np.asarray(data.site_xpos[site_id], dtype=float)
	cur_mat = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
	target_rot = R.from_quat(np.asarray(target_quat, dtype=float))
	cur_rot = R.from_matrix(cur_mat)
	# orientation error as rotation vector taking current -> target
	rot_err = (target_rot * cur_rot.inv()).as_rotvec()
	pos_err = np.asarray(target_pos, dtype=float) - cur_pos
	return pos_err, rot_err


def _solve_ik_dls(
	model,
	data,
	site_id: int,
	joint_qpos_idx: np.ndarray,
	joint_dof_idx: np.ndarray,
	joint_lo: np.ndarray,
	joint_hi: np.ndarray,
	target_pos: np.ndarray,
	target_quat: np.ndarray,
	q_init: np.ndarray,
	*,
	max_iters: int,
	pos_tol: float,
	ori_tol: float,
	damping: float,
	step_size: float,
) -> Optional[np.ndarray]:
	"""Iterative IK using damped least squares on a MuJoCo site."""
	raw_model = _raw_mujoco_model(model)
	raw_data = _raw_mujoco_data(data)

	q = np.asarray(q_init, dtype=float).copy()
	q = q.reshape(-1)
	q = np.clip(q, joint_lo, joint_hi)
	_set_qpos(data, joint_qpos_idx, q)
	mujoco.mj_forward(raw_model, raw_data)

	nv = int(model.nv)
	jacp = np.zeros((3, nv), dtype=float)
	jacr = np.zeros((3, nv), dtype=float)

	lam2 = float(damping) ** 2

	for _ in range(int(max_iters)):
		pos_err, rot_err = _compute_site_error(model, data, site_id, target_pos, target_quat)
		if np.linalg.norm(pos_err) <= pos_tol and np.linalg.norm(rot_err) <= ori_tol:
			return q

		mujoco.mj_jacSite(raw_model, raw_data, jacp, jacr, site_id)
		J = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])  # (6, 7)
		err = np.concatenate([pos_err, rot_err], axis=0)

		# dq = J^T (J J^T + lam^2 I)^-1 * err
		JJt = J @ J.T
		dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), err)
		dq = step_size * dq

		q = np.clip(q + dq, joint_lo, joint_hi)
		_set_qpos(data, joint_qpos_idx, q)
		mujoco.mj_forward(raw_model, raw_data)

	# Final check
	pos_err, rot_err = _compute_site_error(model, data, site_id, target_pos, target_quat)
	if np.linalg.norm(pos_err) <= pos_tol and np.linalg.norm(rot_err) <= ori_tol:
		return q
	return None


def _unique_solutions(solutions: Iterable[np.ndarray], atol: float = 1e-3) -> List[np.ndarray]:
	uniq: List[np.ndarray] = []
	for q in solutions:
		if not any(np.allclose(q, u, atol=atol, rtol=0.0) for u in uniq):
			uniq.append(q)
	return uniq


def enumerate_ik_solutions_for_arm(
	env,
	*,
	robot_id: int,
	target_pose: Union[StampedPose, dict, Tuple[ArrayLike, ArrayLike], ArrayLike],
	target_frame: str = "world",
	arm: str = "right",
	tf_manager: Optional[TransformManager] = None,
	env_lock=None,
	num_seed_configs: int = 24,
	elbow_discretization: int = 8,
	max_solutions: int = 32,
	pos_tol: float = 1e-3,
	ori_tol: float = 5e-3,
	max_iters: int = 200,
	damping: float = 1e-2,
	step_size: float = 0.6,
	unique_atol: float = 1e-3,
) -> List[np.ndarray]:
	"""Enumerate IK solutions for a single arm by running IK from multiple seeds.

	Discretization strategy (simple + robust):
	- Sample the last joint ("elbow/wrist" redundancy) across its joint range.
	- Add random seeds around the current configuration.
	- Run DLS IK from each seed; keep converged solutions; de-duplicate.

	Returns:
		List of joint angle arrays of shape (7,).
	"""

	ctx = env_lock if env_lock is not None else nullcontext()
	with ctx:
		model = env.sim.model
		data = env.sim.data
		raw_model = _raw_mujoco_model(model)
		raw_data = _raw_mujoco_data(data)

		joint_ids = _get_arm_joint_ids(model, robot_id)
		if len(joint_ids) != 7:
			raise RuntimeError(f"Expected 7 arm joints for robot{robot_id}, found {len(joint_ids)}.")

		joint_qpos_idx = _get_joint_qpos_indices(model, joint_ids)
		joint_dof_idx = _get_joint_dof_indices(model, joint_ids)
		joint_lo, joint_hi = _get_joint_limits(model, joint_ids)

		# MuJoCo site poses (site_xpos/site_xmat) are expressed in WORLD.
		# Therefore, run IK in world coordinates.
		if tf_manager is None:
			tf_manager = TransformManager(env, env_lock=env_lock)
		pose_sp = _as_stamped_pose(target_pose, target_frame)
		if pose_sp.frame_id != "world":
			pose_sp = tf_manager.transform_pose(pose_sp, "world")

		target_pos = np.asarray(pose_sp.position, dtype=float)
		target_quat = np.asarray(pose_sp.quaternion, dtype=float)

		site_name = _eef_site_name(robot_id, arm=arm)
		# robosuite wraps the MjModel; use its name->id helper
		site_id = int(model.site_name2id(site_name))

		q_current = np.asarray(data.qpos[joint_qpos_idx], dtype=float)
		q_current = _get_qpos(data, joint_qpos_idx)

		seeds: List[np.ndarray] = []

		# 1) Discretize last joint across its limits
		last_lo, last_hi = float(joint_lo[-1]), float(joint_hi[-1])
		if elbow_discretization > 1 and np.isfinite(last_lo) and np.isfinite(last_hi) and last_hi > last_lo:
			for v in np.linspace(last_lo, last_hi, elbow_discretization):
				q0 = q_current.copy()
				q0[-1] = float(v)
				seeds.append(q0)
		else:
			seeds.append(q_current.copy())

		# 2) Random perturbation seeds
		rng = np.random.default_rng(0)
		for _ in range(max(0, int(num_seed_configs))):
			q0 = q_current + rng.normal(scale=0.25, size=q_current.shape)
			q0 = np.clip(q0, joint_lo, joint_hi)
			seeds.append(q0)

		# 3) A few uniform random seeds in joint space
		for _ in range(max(0, int(num_seed_configs // 3))):
			u = rng.uniform(low=joint_lo, high=joint_hi)
			seeds.append(u)

		# Save and restore simulation state
		qpos_backup = np.array(data.qpos, copy=True)
		qvel_backup = np.array(data.qvel, copy=True)

		solutions: List[np.ndarray] = []
		try:
			for q0 in seeds:
				q_sol = _solve_ik_dls(
					model,
					data,
					site_id,
					joint_qpos_idx,
					joint_dof_idx,
					joint_lo,
					joint_hi,
					target_pos,
					target_quat,
					q0,
					max_iters=max_iters,
					pos_tol=pos_tol,
					ori_tol=ori_tol,
					damping=damping,
					step_size=step_size,
				)
				if q_sol is None:
					continue
				solutions.append(np.asarray(q_sol, dtype=float))
				solutions = _unique_solutions(solutions, atol=unique_atol)
				if len(solutions) >= int(max_solutions):
					break
		finally:
			data.qpos[:] = qpos_backup
			data.qvel[:] = qvel_backup
			mujoco.mj_forward(raw_model, raw_data)

		return solutions


def _geom_belongs_to_robot(geom_name: str) -> Optional[int]:
	# Return robot_id if geom is part of a robot/gripper, else None.
	# Observed prefixes: "robot0_...", "robot1_...", "gripper0_...", "gripper1_..."
	if geom_name.startswith("robot0_") or geom_name.startswith("gripper0_"):
		return 0
	if geom_name.startswith("robot1_") or geom_name.startswith("gripper1_"):
		return 1
	return None


def _in_collision(env, *, contact_dist_threshold: float = 1e-6, env_lock=None) -> bool:
	"""Return True if either robot has any contacts below the threshold."""

	ctx = env_lock if env_lock is not None else nullcontext()
	with ctx:
		model = env.sim.model
		data = env.sim.data

		ncon = int(data.ncon)
		if ncon <= 0:
			return False

		for i in range(ncon):
			c = data.contact[i]
			if float(c.dist) > float(contact_dist_threshold):
				continue
			g1 = model.geom(int(c.geom1)).name or ""
			g2 = model.geom(int(c.geom2)).name or ""
			r1 = _geom_belongs_to_robot(g1)
			r2 = _geom_belongs_to_robot(g2)
			# We only care about robot-involving contacts
			if r1 is None and r2 is None:
				continue
			return True
		return False


def check_dual_arm_pose_feasibility(
	env,
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
	"""Check feasibility for *all* (discretized) dual-arm joint solutions at given EEF poses.

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
		env_lock: Optional threading.Lock used to guard sim state.

	Returns:
		A list of entries: ``[q_arm0 (7,), q_arm1 (7,), feasible_bool]``.

	Notes
	- If either arm has no IK solutions, returns an empty list.
	- Feasible means: IK converged for both arms AND no MuJoCo contact involving
	  either robot is present at that configuration.
	"""

	if tf_manager is None:
		tf_manager = TransformManager(env, env_lock=env_lock)

	q_solutions_0 = enumerate_ik_solutions_for_arm(
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

	q_solutions_1 = enumerate_ik_solutions_for_arm(
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
		raw_model = _raw_mujoco_model(model)
		raw_data = _raw_mujoco_data(data)

		jids0 = _get_arm_joint_ids(model, 0)
		jids1 = _get_arm_joint_ids(model, 1)
		qpos_idx0 = _get_joint_qpos_indices(model, jids0)
		qpos_idx1 = _get_joint_qpos_indices(model, jids1)

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
					_set_qpos(data, qpos_idx0, np.asarray(q0, dtype=float))
					_set_qpos(data, qpos_idx1, np.asarray(q1, dtype=float))
					mujoco.mj_forward(raw_model, raw_data)

					feasible = not _in_collision(env, contact_dist_threshold=contact_dist_threshold, env_lock=None)
					results.append([np.asarray(q0, dtype=float), np.asarray(q1, dtype=float), bool(feasible)])

				if pair_count > int(max_pairs):
					break
		finally:
			data.qpos[:] = qpos_backup
			data.qvel[:] = qvel_backup
			mujoco.mj_forward(raw_model, raw_data)

		return results

