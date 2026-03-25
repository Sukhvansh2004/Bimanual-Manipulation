"""Mink QP IK stress test — profiles the QP-based differential-IK hot path.

Mirrors the structure of scipy_stress_test.py so the two backends can
be compared directly.  The mink hot-path per QP iteration is:

  1. mink.solve_ik(config, tasks, dt, …)        (proxQP solve)
  2. config.integrate_inplace(vel, dt)            (velocity integration)
  3. config.get_transform_frame_to_world(…)      (FK for convergence check)
  4. R_target * R.from_matrix(…).inv().as_rotvec()  (rotation error)
  5. Stall detection bookkeeping

Additionally, per seed the solver must:
  ● Construct mink.Configuration + 3–4 tasks + limits   (setup overhead)
  ● Post-solve: Jacobian + SVD for manipulability        (same as scipy)
"""

import time
import numpy as np
import mink
import mujoco
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.planning.ik.mink_solver import (
    solve_ik_mink,
    _build_target_se3,
    _generate_seeds,
    _compute_jacobian,
)
from bimanual_manipulation.planning.ik.manipulability import (
    compute_manipulability_from_jacobian,
)


# ── Segment-assignment scale constants ──
COST_EVAL_APPROACH_STEPS = 5
COST_EVAL_RETREAT_STEPS = 3
POSES_PER_TRANSFER = 1 + COST_EVAL_RETREAT_STEPS     # 4
POSES_PER_RECEIVER = 1 + COST_EVAL_APPROACH_STEPS     # 6
POSES_PER_HANDOVER = POSES_PER_TRANSFER + POSES_PER_RECEIVER  # 10
N_ALIVE = 20

# Mink parameters
NUM_SEEDS = constants.MINK_IK_NUM_SEEDS               # 48
MAX_ITERS = constants.MINK_IK_MAX_ITERS               # 300
POS_TOL = constants.MINK_IK_POS_TOLERANCE             # 5e-3
ROT_TOL = constants.MINK_IK_ROT_TOLERANCE             # 5e-3
DT = constants.MINK_IK_DT                             # 0.02
STALL_WINDOW = constants.MINK_IK_STALL_WINDOW         # 20
STALL_TOL = constants.MINK_IK_STALL_TOL               # 1e-6


def _setup_env():
    """Create env and extract all the raw MuJoCo handles needed for profiling."""
    env = make_dual_arm_env(
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    env.reset()

    model = env.sim.model
    data = env.sim.data
    robot_id = 0
    arm = "right"

    joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
    joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
    joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)
    joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)

    site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)
    model_native = mujoco_utils.get_native_mj_object(model)
    data_native = mujoco_utils.get_native_mj_object(data)
    site_id = mujoco.mj_name2id(
        model_native, mujoco.mjtObj.mjOBJ_SITE, site_name,
    )

    full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

    # Current EEF pose (home)
    target_pos = np.array(data_native.site_xpos[site_id], dtype=float).copy()
    target_mat = np.array(data_native.site_xmat[site_id], dtype=float).reshape(3, 3)
    target_quat = R.from_matrix(target_mat).as_quat()

    # Frozen DOFs (everything except the arm)
    frozen_dofs = mujoco_utils.get_frozen_dof_indices(model_native, joint_dof_idx)

    return dict(
        env=env,
        model_native=model_native,
        data_native=data_native,
        site_id=site_id,
        site_name=site_name,
        joint_qpos_idx=joint_qpos_idx,
        joint_dof_idx=joint_dof_idx,
        joint_lo=joint_lo,
        joint_hi=joint_hi,
        full_qpos=full_qpos,
        target_pos=target_pos,
        target_quat=target_quat,
        frozen_dofs=frozen_dofs,
        nv=model_native.nv,
    )


def _random_reachable_target(ctx, rng):
    """FK a random joint config to get a guaranteed-reachable target."""
    q = rng.uniform(ctx["joint_lo"] + 0.15, ctx["joint_hi"] - 0.15)
    q_full = ctx["full_qpos"].copy()
    q_full[ctx["joint_qpos_idx"]] = q
    ctx["data_native"].qpos[:] = q_full
    mujoco.mj_fwdPosition(ctx["model_native"], ctx["data_native"])
    pos = np.array(ctx["data_native"].site_xpos[ctx["site_id"]], dtype=float).copy()
    mat = np.array(ctx["data_native"].site_xmat[ctx["site_id"]], dtype=float).reshape(3, 3)
    quat = R.from_matrix(mat).as_quat()
    return pos, quat


def _perturbed_pose(pos, quat, rng, pos_noise=0.02, rot_noise=0.05):
    """Small perturbation — mimics consecutive waypoints."""
    new_pos = pos + rng.uniform(-pos_noise, pos_noise, 3)
    rotvec = R.from_quat(quat).as_rotvec() + rng.uniform(-rot_noise, rot_noise, 3)
    new_quat = R.from_rotvec(rotvec).as_quat()
    return new_pos, new_quat


# =====================================================================
#  Micro-benchmarks — measure each sub-operation in the mink hot path
# =====================================================================

def bench_configuration_creation(ctx, n=500):
    """mink.Configuration creation cost (done once per seed)."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        _ = mink.Configuration(m, q=fp.copy())
        times[i] = time.perf_counter() - t0
    return times


def bench_task_setup(ctx, n=500):
    """Cost of creating FrameTask + PostureTask + DampingTask + FreezingTask + limits."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    frozen = ctx["frozen_dofs"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]

    times = np.empty(n)
    for i in range(n):
        config = mink.Configuration(m, q=fp.copy())
        t0 = time.perf_counter()

        eef_task = mink.FrameTask(
            frame_name=sn, frame_type="site",
            position_cost=constants.MINK_POSITION_COST,
            orientation_cost=constants.MINK_ORIENTATION_COST,
            gain=constants.MINK_FRAME_GAIN,
            lm_damping=constants.MINK_LM_DAMPING,
        )
        target_se3 = _build_target_se3(tp, tq)
        eef_task.set_target(target_se3)

        posture_task = mink.PostureTask(model=m, cost=constants.MINK_POSTURE_COST)
        posture_task.set_target_from_configuration(config)
        damping_task = mink.DampingTask(model=m, cost=constants.MINK_DAMPING_COST)
        freeze_task = mink.DofFreezingTask(model=m, dof_indices=frozen) if frozen else None
        limits = [mink.ConfigurationLimit(m)]

        times[i] = time.perf_counter() - t0
    return times


def bench_qp_solve(ctx, n=500):
    """Cost of a single mink.solve_ik QP step."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    frozen = ctx["frozen_dofs"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]

    config = mink.Configuration(m, q=fp.copy())
    eef_task = mink.FrameTask(
        frame_name=sn, frame_type="site",
        position_cost=constants.MINK_POSITION_COST,
        orientation_cost=constants.MINK_ORIENTATION_COST,
        gain=constants.MINK_FRAME_GAIN,
        lm_damping=constants.MINK_LM_DAMPING,
    )
    eef_task.set_target(_build_target_se3(tp, tq))
    posture_task = mink.PostureTask(model=m, cost=constants.MINK_POSTURE_COST)
    posture_task.set_target_from_configuration(config)
    damping_task = mink.DampingTask(model=m, cost=constants.MINK_DAMPING_COST)
    tasks = [eef_task, posture_task, damping_task]
    if frozen:
        tasks.append(mink.DofFreezingTask(model=m, dof_indices=frozen))
    limits = [mink.ConfigurationLimit(m)]

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        vel = mink.solve_ik(config, tasks, DT, solver=constants.MINK_QP_SOLVER, limits=limits)
        times[i] = time.perf_counter() - t0
    return times


def bench_integrate(ctx, n=2000):
    """Cost of config.integrate_inplace(vel, dt)."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    config = mink.Configuration(m, q=fp.copy())

    # Create a dummy velocity
    vel = np.zeros(m.nv)
    vel[ctx["joint_dof_idx"]] = 0.01

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        config.integrate_inplace(vel, DT)
        times[i] = time.perf_counter() - t0
    return times


def bench_convergence_check(ctx, n=2000):
    """Cost of get_transform + rotation error check (done each iteration)."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]
    R_target = R.from_quat(tq)

    config = mink.Configuration(m, q=fp.copy())

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        T_current = config.get_transform_frame_to_world(sn, "site")
        achieved_pos = T_current.translation()
        achieved_rot = T_current.rotation()
        pos_err = float(np.linalg.norm(achieved_pos - tp))
        rot_err = float(np.linalg.norm(
            (R_target * R.from_matrix(achieved_rot.as_matrix()).inv()).as_rotvec()
        ))
        times[i] = time.perf_counter() - t0
    return times


def bench_build_target_se3(ctx, n=5000):
    """Cost of _build_target_se3."""
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        _build_target_se3(tp, tq)
        times[i] = time.perf_counter() - t0
    return times


def bench_manipulability(ctx, n=5000):
    """SVD-based manipulability (post-solve scoring) — same as scipy."""
    rng = np.random.default_rng(0)
    Js = [rng.standard_normal((6, 7)) for _ in range(200)]
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        compute_manipulability_from_jacobian(Js[i % len(Js)])
        times[i] = time.perf_counter() - t0
    return times


# =====================================================================
#  Single-seed benchmark — one full solve_ik_mink call
# =====================================================================

def bench_single_seed(ctx, n=50):
    """Single solve_ik_mink call — one seed, up to MAX_ITERS QP steps."""
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]
    rng = np.random.default_rng(42)

    times = np.empty(n)
    converged = np.empty(n, dtype=bool)

    for i in range(n):
        q0 = rng.uniform(lo, hi)
        t0 = time.perf_counter()
        result = solve_ik_mink(
            m, fp, sn, idx_q, idx_d, tp, tq, q0,
        )
        times[i] = time.perf_counter() - t0
        converged[i] = result is not None

    return times, converged


# =====================================================================
#  Macro benchmark — full multi-seed solve (48 seeds)
# =====================================================================

def bench_full_multiseed(ctx, n_targets=20):
    """Full 48-seed mink IK for n random reachable targets."""
    rng = np.random.default_rng(42)
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]

    targets = []
    for _ in range(n_targets):
        pos, quat = _random_reachable_target(ctx, rng)
        targets.append((pos, quat))

    times = np.empty(n_targets)
    n_solutions = np.empty(n_targets, dtype=int)
    q_home = np.array(constants.FRANKA_HOME_JOINT_POSITION, dtype=float)

    for i, (tp, tq) in enumerate(targets):
        seeds = _generate_seeds(q_home, lo, hi, NUM_SEEDS, q_current=None)
        t0 = time.perf_counter()
        converged = []
        for seed in seeds:
            result = solve_ik_mink(m, fp, sn, idx_q, idx_d, tp, tq, seed)
            if result is not None:
                converged.append(result)
        times[i] = time.perf_counter() - t0
        n_solutions[i] = len(converged)

    return times, n_solutions


# =====================================================================
#  Handover-level benchmark (production workload simulation)
# =====================================================================

def bench_handover_workload(ctx, n_handovers=50):
    """Simulate n handover evaluations — each with 10 IK solves."""
    rng = np.random.default_rng(42)
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]
    q_home = np.array(constants.FRANKA_HOME_JOINT_POSITION, dtype=float)

    handover_times = np.empty(n_handovers)
    ik_call_times = []
    total_solutions = 0
    total_ik_calls = 0

    for h in range(n_handovers):
        base_pos, base_quat = _random_reachable_target(ctx, rng)
        q_warm = q_home.copy()
        t_handover = time.perf_counter()

        for step in range(POSES_PER_HANDOVER):
            p, q = _perturbed_pose(base_pos, base_quat, rng)
            seeds = _generate_seeds(q_warm, lo, hi, NUM_SEEDS, q_current=None)
            t_ik = time.perf_counter()
            converged = []
            for seed in seeds:
                result = solve_ik_mink(m, fp, sn, idx_q, idx_d, p, q, seed)
                if result is not None:
                    converged.append(result)
            ik_call_times.append(time.perf_counter() - t_ik)
            total_ik_calls += 1
            total_solutions += len(converged)
            if converged:
                q_warm = converged[0]

        handover_times[h] = time.perf_counter() - t_handover

        if (h + 1) % 10 == 0:
            avg = np.mean(handover_times[:h+1]) * 1000
            print(f"  [{h+1:4d}/{n_handovers}] avg={avg:.0f} ms/handover")

    return handover_times, np.array(ik_call_times), total_solutions, total_ik_calls


# =====================================================================
#  Iteration-level profiling — instrument one solve to see iter costs
# =====================================================================

def bench_iteration_breakdown(ctx, n_seeds=5):
    """Instrument a few mink IK solves to report per-iteration timing breakdown.

    Manually replicates solve_ik_mink with per-phase timing.
    """
    m = ctx["model_native"]
    fp = ctx["full_qpos"]
    sn = ctx["site_name"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]
    frozen = ctx["frozen_dofs"]
    R_target = R.from_quat(tq)

    rng = np.random.default_rng(0)

    phase_qp = []
    phase_integrate = []
    phase_fk_check = []
    phase_rot_err = []
    phase_setup = []
    total_iters = 0

    for _ in range(n_seeds):
        q0 = rng.uniform(lo, hi)
        q_full = fp.copy()
        q_full[idx_q] = q0

        # --- Setup phase ---
        t_setup = time.perf_counter()
        config = mink.Configuration(m, q=q_full)
        eef_task = mink.FrameTask(
            frame_name=sn, frame_type="site",
            position_cost=constants.MINK_POSITION_COST,
            orientation_cost=constants.MINK_ORIENTATION_COST,
            gain=constants.MINK_FRAME_GAIN,
            lm_damping=constants.MINK_LM_DAMPING,
        )
        eef_task.set_target(_build_target_se3(tp, tq))
        posture_task = mink.PostureTask(model=m, cost=constants.MINK_POSTURE_COST)
        posture_task.set_target_from_configuration(config)
        damping_task = mink.DampingTask(model=m, cost=constants.MINK_DAMPING_COST)
        tasks = [eef_task, posture_task, damping_task]
        if frozen:
            tasks.append(mink.DofFreezingTask(model=m, dof_indices=frozen))
        limits = [mink.ConfigurationLimit(m)]
        phase_setup.append(time.perf_counter() - t_setup)

        best_err = float("inf")
        stall_count = 0

        for _ in range(MAX_ITERS):
            total_iters += 1

            # QP solve
            t0 = time.perf_counter()
            vel = mink.solve_ik(config, tasks, DT, solver=constants.MINK_QP_SOLVER, limits=limits)
            phase_qp.append(time.perf_counter() - t0)

            # Integrate
            t0 = time.perf_counter()
            config.integrate_inplace(vel, DT)
            phase_integrate.append(time.perf_counter() - t0)

            # FK / convergence check
            t0 = time.perf_counter()
            T_current = config.get_transform_frame_to_world(sn, "site")
            achieved_pos = T_current.translation()
            achieved_rot = T_current.rotation()
            phase_fk_check.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            pos_err = float(np.linalg.norm(achieved_pos - tp))
            rot_err = float(np.linalg.norm(
                (R_target * R.from_matrix(achieved_rot.as_matrix()).inv()).as_rotvec()
            ))
            phase_rot_err.append(time.perf_counter() - t0)

            if pos_err <= POS_TOL and rot_err <= ROT_TOL:
                break

            total_err = pos_err + rot_err
            if total_err < best_err - STALL_TOL:
                best_err = total_err
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= STALL_WINDOW:
                    break

    return {
        "setup": np.array(phase_setup),
        "qp_solve": np.array(phase_qp),
        "integrate": np.array(phase_integrate),
        "fk_check": np.array(phase_fk_check),
        "rot_err": np.array(phase_rot_err),
        "total_iters": total_iters,
        "n_seeds": n_seeds,
    }


# =====================================================================
#  Main
# =====================================================================

def _fmt_times(times_s, unit="us"):
    """Format timing array as mean/median/p95."""
    if unit == "us":
        t = times_s * 1e6
        u = "us"
    elif unit == "ms":
        t = times_s * 1e3
        u = "ms"
    else:
        t = times_s
        u = "s"
    return (f"mean={np.mean(t):.1f} {u}, "
            f"med={np.median(t):.1f} {u}, "
            f"p95={np.percentile(t, 95):.1f} {u}")


def main():
    print("=" * 70)
    print("Mink QP IK Stress Test")
    print(f"  Seeds per IK call    : {NUM_SEEDS}")
    print(f"  Max iters per seed   : {MAX_ITERS}")
    print(f"  DT per QP step       : {DT}")
    print(f"  Pos/Rot tolerance    : {POS_TOL} m / {ROT_TOL} rad")
    print(f"  Stall window/tol     : {STALL_WINDOW} / {STALL_TOL}")
    print(f"  QP solver            : {constants.MINK_QP_SOLVER}")
    print(f"  Poses per handover   : {POSES_PER_HANDOVER}")
    print("=" * 70)

    print("\nSetting up environment...")
    ctx = _setup_env()
    print("  Done.\n")

    # ── 1. Micro-benchmarks ──
    print("─" * 70)
    print("1. MICRO-BENCHMARKS (per-operation cost)")
    print("─" * 70)

    print("\n  a) mink.Configuration creation:")
    t = bench_configuration_creation(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  b) Task setup (Frame + Posture + Damping + Freeze + Limits):")
    t = bench_task_setup(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  c) Single mink.solve_ik (1 QP step):")
    t = bench_qp_solve(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  d) config.integrate_inplace:")
    t = bench_integrate(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  e) Convergence check (get_transform + rotation error):")
    t = bench_convergence_check(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  f) _build_target_se3:")
    t = bench_build_target_se3(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  g) Manipulability (SVD of 6x7):")
    t = bench_manipulability(ctx)
    print(f"     {_fmt_times(t)}")

    # ── 2. Iteration breakdown ──
    print("\n" + "─" * 70)
    print("2. PER-ITERATION BREAKDOWN (instrumented solve, 5 seeds)")
    print("─" * 70)

    breakdown = bench_iteration_breakdown(ctx, n_seeds=5)
    total_iters = breakdown["total_iters"]
    avg_iters = total_iters / breakdown["n_seeds"]
    print(f"\n  Total QP iterations across {breakdown['n_seeds']} seeds: "
          f"{total_iters} (avg {avg_iters:.0f}/seed)")

    qp_mean = np.mean(breakdown["qp_solve"]) * 1e6
    int_mean = np.mean(breakdown["integrate"]) * 1e6
    fk_mean = np.mean(breakdown["fk_check"]) * 1e6
    rot_mean = np.mean(breakdown["rot_err"]) * 1e6
    setup_mean = np.mean(breakdown["setup"]) * 1e6
    iter_total = qp_mean + int_mean + fk_mean + rot_mean

    print(f"\n  Per-seed setup                    : {setup_mean:.0f} us")
    print(f"  Per-iteration total               : {iter_total:.0f} us")
    print(f"    mink.solve_ik (QP)              : {qp_mean:.0f} us "
          f"({qp_mean/iter_total*100:.0f}%)")
    print(f"    integrate_inplace               : {int_mean:.0f} us "
          f"({int_mean/iter_total*100:.0f}%)")
    print(f"    get_transform (FK check)        : {fk_mean:.0f} us "
          f"({fk_mean/iter_total*100:.0f}%)")
    print(f"    rotation error (scipy.R)        : {rot_mean:.0f} us "
          f"({rot_mean/iter_total*100:.0f}%)")

    # ── 3. Single-seed full solve ──
    print("\n" + "─" * 70)
    print("3. SINGLE-SEED solve_ik_mink (1 seed, up to 300 iters)")
    print("─" * 70)

    times, converged = bench_single_seed(ctx, n=50)
    print(f"\n  Wall time : {_fmt_times(times, 'ms')}")
    print(f"  Converged : {converged.sum()}/{len(converged)} "
          f"({converged.mean()*100:.0f}%)")
    conv_times = times[converged]
    fail_times = times[~converged]
    if len(conv_times) > 0:
        print(f"  Converged wall time : {_fmt_times(conv_times, 'ms')}")
    if len(fail_times) > 0:
        print(f"  Failed wall time    : {_fmt_times(fail_times, 'ms')}")

    # ── 4. Full multi-seed (48 seeds) ──
    print("\n" + "─" * 70)
    print(f"4. FULL MULTI-SEED ({NUM_SEEDS} seeds, 20 targets)")
    print("─" * 70)

    solve_times, n_sols = bench_full_multiseed(ctx, n_targets=20)
    print(f"\n  Wall time : {_fmt_times(solve_times, 'ms')}")
    print(f"  Solutions : mean={n_sols.mean():.1f}, "
          f"med={np.median(n_sols):.0f}, max={n_sols.max()}")
    print(f"  Projected per-IK-call with {NUM_SEEDS} seeds: "
          f"{np.mean(solve_times)*1000:.0f} ms")

    # ── 5. Handover workload ──
    n_bench_handovers = 50
    print("\n" + "─" * 70)
    print(f"5. HANDOVER WORKLOAD ({n_bench_handovers} handovers, "
          f"{POSES_PER_HANDOVER} IK each)")
    print("─" * 70)

    ht, ik_times, total_sols, total_iks = bench_handover_workload(
        ctx, n_handovers=n_bench_handovers,
    )

    print(f"\n  Per handover: {_fmt_times(ht, 's')}")
    print(f"  Per IK call : {_fmt_times(ik_times, 'ms')}")
    print(f"  Solutions   : {total_sols}/{total_iks} IK calls found solutions "
          f"({total_sols/total_iks*100:.0f}%)")

    total_wall = ht.sum()
    mean_ik_ms = np.mean(ik_times) * 1e3
    mean_handover_s = np.mean(ht)
    print(f"\n  Total wall time: {total_wall:.1f}s for {n_bench_handovers} handovers")
    print(f"  Avg IK call    : {mean_ik_ms:.0f} ms "
          f"({NUM_SEEDS} seeds × {MAX_ITERS} max iters)")

    # ── 6. Projections ──
    print("\n" + "─" * 70)
    print("6. FULL-RUN PROJECTIONS")
    print("─" * 70)

    mean_segment_s = mean_ik_ms / 1000 * N_ALIVE
    segments = 79
    proj_total = mean_segment_s * segments
    print(f"\n  Avg single IK call       : {mean_ik_ms:.0f} ms")
    print(f"  Per segment ({N_ALIVE} alive)   : {mean_segment_s:.1f}s")
    print(f"  Full pipeline ({segments} segs) : {proj_total:.0f}s "
          f"({proj_total/60:.1f} min)")

    # Breakdown
    print(f"\n  ── Per-iteration cost breakdown ──")
    print(f"    QP solve     : {qp_mean:.0f} us ({qp_mean/iter_total*100:.0f}%)")
    print(f"    Integrate    : {int_mean:.0f} us ({int_mean/iter_total*100:.0f}%)")
    print(f"    FK check     : {fk_mean:.0f} us ({fk_mean/iter_total*100:.0f}%)")
    print(f"    Rot error    : {rot_mean:.0f} us ({rot_mean/iter_total*100:.0f}%)")
    print(f"    Setup/seed   : {setup_mean:.0f} us (amortized over {avg_iters:.0f} iters)")

    # Seed reduction projections
    print(f"\n  ── Potential optimisations ──")
    for n_seeds in [48, 32, 24, 16, 8]:
        proj_ms = mean_ik_ms * n_seeds / NUM_SEEDS
        proj_segment = proj_ms / 1000 * N_ALIVE
        proj_pipeline = proj_segment * segments
        print(f"    {n_seeds:2d} seeds → ~{proj_ms:.0f} ms/call, "
              f"~{proj_segment:.1f}s/segment, ~{proj_pipeline:.0f}s pipeline")

    # Stall early-exit savings
    conv_pct = converged.mean() * 100
    avg_conv_ms = np.mean(conv_times) * 1e3 if len(conv_times) > 0 else 0
    avg_fail_ms = np.mean(fail_times) * 1e3 if len(fail_times) > 0 else 0
    print(f"\n    Convergence rate: {conv_pct:.0f}%")
    print(f"    Converged avg: {avg_conv_ms:.0f} ms vs Failed avg: {avg_fail_ms:.0f} ms")
    if len(fail_times) > 0 and avg_fail_ms > avg_conv_ms:
        saved = (avg_fail_ms - avg_conv_ms) * (1 - converged.mean()) * NUM_SEEDS
        print(f"    Faster stall detection could save ~{saved:.0f} ms/IK call")

    ctx["env"].close()
    print("\nDone.")


if __name__ == "__main__":
    main()
