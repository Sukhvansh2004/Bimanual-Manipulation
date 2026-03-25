"""Scipy L-BFGS-B IK stress test — profiles the optimisation hot path.

Mimics the segment-assignment workload where Scipy IK is called for
each alive state at each segment boundary.  From the production logs:
  - Segment 17: 114s for 20 IK calls → ~5.7s per IK call
  - Segment 27: 183s for 38 IK calls → ~4.8s per IK call

Each IK call runs 48 seeds × up to 200 L-BFGS-B iterations.
Per iteration the hot path does:
  1. full_qpos copy + write                         (numpy)
  2. mujoco.mj_fwdPosition                          (MuJoCo FK)
  3. compute_site_error_fast → R.from_matrix +
     (R_target * cur_rot.inv()).as_rotvec()           (scipy Rotation)
  4. mujoco.mj_jacSite                              (MuJoCo Jacobian)
  5. J.T @ err                                      (numpy matmul)
  6. 2 × np.zeros((3, nv))                          (Jacobian buffers)

This script measures each sub-operation independently, then runs
realistic IK batches to project full-pipeline timing.
"""

import time
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.planning.ik.scipy_solver import (
    _fk_objective_and_grad,
    _generate_seeds,
    solve_ik_scipy,
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
N_HANDOVERS = 5700
N_ALIVE = 20                                          # typical alive states

# Scipy parameters
NUM_SEEDS = constants.SCIPY_IK_NUM_SEEDS               # 48
MAX_ITERS = constants.SCIPY_IK_MAX_ITERS               # 200
POS_TOL = constants.SCIPY_IK_POS_TOLERANCE
ROT_TOL = constants.SCIPY_IK_ROT_TOLERANCE


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

    return dict(
        env=env,
        model_native=model_native,
        data_native=data_native,
        site_id=site_id,
        joint_qpos_idx=joint_qpos_idx,
        joint_dof_idx=joint_dof_idx,
        joint_lo=joint_lo,
        joint_hi=joint_hi,
        full_qpos=full_qpos,
        target_pos=target_pos,
        target_quat=target_quat,
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
#  Micro-benchmarks — measure each sub-operation in the hot path
# =====================================================================

def bench_mj_fwdPosition(ctx, n=10_000):
    """Pure mj_fwdPosition cost (no data copy, just FK)."""
    m, d = ctx["model_native"], ctx["data_native"]
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        mujoco.mj_fwdPosition(m, d)
        times[i] = time.perf_counter() - t0
    return times


def bench_qpos_copy_and_fk(ctx, n=10_000):
    """full_qpos copy + qpos write + mj_fwdPosition (as done per iter)."""
    m, d = ctx["model_native"], ctx["data_native"]
    fp = ctx["full_qpos"]
    idx = ctx["joint_qpos_idx"]
    rng = np.random.default_rng(0)
    q_arms = [rng.uniform(ctx["joint_lo"], ctx["joint_hi"]) for _ in range(min(n, 1000))]

    times = np.empty(n)
    for i in range(n):
        q = q_arms[i % len(q_arms)]
        t0 = time.perf_counter()
        q_full = fp.copy()
        q_full[idx] = q
        d.qpos[:] = q_full
        mujoco.mj_fwdPosition(m, d)
        times[i] = time.perf_counter() - t0
    return times


def bench_rotation_error(ctx, n=10_000):
    """R.from_matrix + (R_target * cur.inv()).as_rotvec() — the scipy.Rotation cost."""
    d = ctx["data_native"]
    sid = ctx["site_id"]
    R_target = R.from_quat(ctx["target_quat"])
    target_pos = ctx["target_pos"]

    # Pre-run FK so site data is valid
    mujoco.mj_fwdPosition(ctx["model_native"], d)

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        cur_pos = np.asarray(d.site_xpos[sid], dtype=float)
        cur_mat = np.asarray(d.site_xmat[sid], dtype=float).reshape(3, 3)
        cur_rot = R.from_matrix(cur_mat)
        rot_err = (R_target * cur_rot.inv()).as_rotvec()
        pos_err = target_pos - cur_pos
        times[i] = time.perf_counter() - t0
    return times


def bench_jacobian(ctx, n=10_000):
    """mj_jacSite + column slicing."""
    m, d = ctx["model_native"], ctx["data_native"]
    sid = ctx["site_id"]
    dof_idx = ctx["joint_dof_idx"]
    nv = ctx["nv"]

    mujoco.mj_fwdPosition(m, d)

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        times[i] = time.perf_counter() - t0
    return times


def bench_gradient(ctx, n=10_000):
    """J.T @ err  (6→7 matmul)."""
    rng = np.random.default_rng(0)
    J = rng.standard_normal((6, 7))
    err = rng.standard_normal(6)

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        _ = -2.0 * J.T @ err
        times[i] = time.perf_counter() - t0
    return times


def bench_full_fk_obj_and_grad(ctx, n=2_000):
    """Full _fk_objective_and_grad call (as called by L-BFGS-B each iteration)."""
    m, d = ctx["model_native"], ctx["data_native"]
    sid = ctx["site_id"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    fp = ctx["full_qpos"]
    tp = ctx["target_pos"]
    R_t = R.from_quat(ctx["target_quat"])
    rng = np.random.default_rng(0)
    q_arms = [rng.uniform(ctx["joint_lo"], ctx["joint_hi"]) for _ in range(200)]

    times = np.empty(n)
    for i in range(n):
        q = q_arms[i % len(q_arms)]
        t0 = time.perf_counter()
        _fk_objective_and_grad(q, m, d, sid, idx_q, idx_d, fp, tp, R_t)
        times[i] = time.perf_counter() - t0
    return times


def bench_manipulability(ctx, n=5_000):
    """SVD-based manipulability (post-solve scoring)."""
    rng = np.random.default_rng(0)
    Js = [rng.standard_normal((6, 7)) for _ in range(200)]

    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        compute_manipulability_from_jacobian(Js[i % len(Js)])
        times[i] = time.perf_counter() - t0
    return times


def bench_minimize_single_seed(ctx, n=50):
    """Single L-BFGS-B minimize call (1 seed, up to MAX_ITERS iterations)."""
    m, d = ctx["model_native"], ctx["data_native"]
    sid = ctx["site_id"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    fp = ctx["full_qpos"]
    tp = ctx["target_pos"]
    tq = ctx["target_quat"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]
    bounds = list(zip(lo.tolist(), hi.tolist()))
    R_target = R.from_quat(tq)
    rng = np.random.default_rng(42)

    times = np.empty(n)
    n_iters = np.empty(n, dtype=int)
    n_fevals = np.empty(n, dtype=int)
    converged = np.empty(n, dtype=bool)

    for i in range(n):
        q0 = rng.uniform(lo, hi)
        t0 = time.perf_counter()
        result = minimize(
            _fk_objective_and_grad,
            q0,
            args=(m, d, sid, idx_q, idx_d, fp, tp, R_target),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": MAX_ITERS, "ftol": 1e-12, "gtol": 1e-8},
        )
        times[i] = time.perf_counter() - t0
        n_iters[i] = result.nit
        n_fevals[i] = result.nfev
        converged[i] = result.success

    return times, n_iters, n_fevals, converged


# =====================================================================
#  Macro benchmark — full solve_ik_scipy (48 seeds)
# =====================================================================

def bench_solve_ik_scipy(ctx, n_targets=20):
    """Full solve_ik_scipy for n random reachable targets."""
    rng = np.random.default_rng(42)
    m, d = ctx["model_native"], ctx["data_native"]
    sid = ctx["site_id"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    fp = ctx["full_qpos"]
    lo, hi = ctx["joint_lo"], ctx["joint_hi"]

    # Generate targets by FK of random configs
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
        sols = solve_ik_scipy(
            m, d, sid, idx_q, idx_d, fp,
            tp, tq, lo, hi,
            q_seeds=seeds,
        )
        times[i] = time.perf_counter() - t0
        n_solutions[i] = len(sols)

    return times, n_solutions


# =====================================================================
#  Handover-level benchmark (production workload simulation)
# =====================================================================

def bench_handover_workload(ctx, n_handovers=100):
    """Simulate n handover evaluations — each with 10 IK solves."""
    rng = np.random.default_rng(42)
    m, d = ctx["model_native"], ctx["data_native"]
    sid = ctx["site_id"]
    idx_q = ctx["joint_qpos_idx"]
    idx_d = ctx["joint_dof_idx"]
    fp = ctx["full_qpos"]
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
            sols = solve_ik_scipy(
                m, d, sid, idx_q, idx_d, fp,
                p, q, lo, hi,
                q_seeds=seeds,
            )
            ik_call_times.append(time.perf_counter() - t_ik)
            total_ik_calls += 1
            total_solutions += len(sols)
            if sols:
                q_warm = sols[0]

        handover_times[h] = time.perf_counter() - t_handover

        if (h + 1) % 20 == 0:
            avg = np.mean(handover_times[:h+1]) * 1000
            print(f"  [{h+1:4d}/{n_handovers}] avg={avg:.0f} ms/handover")

    return handover_times, np.array(ik_call_times), total_solutions, total_ik_calls


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
    print("Scipy L-BFGS-B IK Stress Test")
    print(f"  Seeds per IK call    : {NUM_SEEDS}")
    print(f"  Max iters per seed   : {MAX_ITERS}")
    print(f"  Pos/Rot tolerance    : {POS_TOL} m / {ROT_TOL} rad")
    print(f"  Poses per handover   : {POSES_PER_HANDOVER}")
    print("=" * 70)

    print("\nSetting up environment...")
    ctx = _setup_env()
    print("  Done.\n")

    # ── 1. Micro-benchmarks ──
    print("─" * 70)
    print("1. MICRO-BENCHMARKS (per-operation cost)")
    print("─" * 70)

    print("\n  a) mj_fwdPosition (pure FK, no data copy):")
    t = bench_mj_fwdPosition(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  b) qpos copy + write + mj_fwdPosition:")
    t = bench_qpos_copy_and_fk(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  c) Rotation error (R.from_matrix + rotvec):")
    t = bench_rotation_error(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  d) mj_jacSite + column slice + vstack:")
    t = bench_jacobian(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  e) Gradient: -2 * J.T @ err:")
    t = bench_gradient(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  f) Full _fk_objective_and_grad (a+b+c+d+e combined):")
    t = bench_full_fk_obj_and_grad(ctx)
    print(f"     {_fmt_times(t)}")

    print("\n  g) Manipulability (SVD of 6x7):")
    t = bench_manipulability(ctx)
    print(f"     {_fmt_times(t)}")

    # ── 2. Single-seed L-BFGS-B ──
    print("\n" + "─" * 70)
    print("2. SINGLE-SEED L-BFGS-B (1 minimize call)")
    print("─" * 70)

    times, n_iters, n_fevals, converged = bench_minimize_single_seed(ctx, n=50)
    print(f"\n  Wall time : {_fmt_times(times, 'ms')}")
    print(f"  Iterations: mean={n_iters.mean():.0f}, "
          f"med={np.median(n_iters):.0f}, max={n_iters.max()}")
    print(f"  Func evals: mean={n_fevals.mean():.0f}, "
          f"med={np.median(n_fevals):.0f}, max={n_fevals.max()}")
    print(f"  Converged : {converged.sum()}/{len(converged)} "
          f"({converged.mean()*100:.0f}%)")
    ms_per_iter = (times / n_iters) * 1e3
    print(f"  ms/iter   : mean={ms_per_iter.mean():.3f}, "
          f"med={np.median(ms_per_iter):.3f}")
    total_obj_time = n_fevals.mean() * np.mean(bench_full_fk_obj_and_grad(ctx, n=500))
    overhead_pct = (1 - total_obj_time / np.mean(times)) * 100
    print(f"  L-BFGS-B overhead (beyond obj+grad calls): {overhead_pct:.0f}%")

    # ── 3. Full solve_ik_scipy (48 seeds) ──
    print("\n" + "─" * 70)
    print(f"3. FULL solve_ik_scipy ({NUM_SEEDS} seeds, 20 targets)")
    print("─" * 70)

    solve_times, n_sols = bench_solve_ik_scipy(ctx, n_targets=20)
    print(f"\n  Wall time : {_fmt_times(solve_times, 'ms')}")
    print(f"  Solutions : mean={n_sols.mean():.1f}, "
          f"med={np.median(n_sols):.0f}, max={n_sols.max()}")
    print(f"  Projected per-IK-call with {NUM_SEEDS} seeds: "
          f"{np.mean(solve_times)*1000:.0f} ms")

    # ── 4. Handover workload ──
    n_bench_handovers = 50
    print("\n" + "─" * 70)
    print(f"4. HANDOVER WORKLOAD ({n_bench_handovers} handovers, "
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

    # ── 5. Projections ──
    print("\n" + "─" * 70)
    print("5. FULL-RUN PROJECTIONS")
    print("─" * 70)

    # Segment-assignment scale: each segment evaluates N_ALIVE IK calls
    mean_segment_s = mean_ik_ms / 1000 * N_ALIVE
    segments = 79
    proj_total = mean_segment_s * segments
    print(f"\n  Avg single IK call       : {mean_ik_ms:.0f} ms")
    print(f"  Per segment ({N_ALIVE} alive)   : {mean_segment_s:.1f}s")
    print(f"  Full pipeline ({segments} segs) : {proj_total:.0f}s "
          f"({proj_total/60:.1f} min)")
    proj_handover = mean_handover_s * N_HANDOVERS
    print(f"  Full handover eval       : {proj_handover:.0f}s "
          f"({proj_handover/60:.1f} min)")

    # Breakdown: what fraction is objective vs overhead
    obj_time_us = np.mean(bench_full_fk_obj_and_grad(ctx, n=500)) * 1e6
    print(f"\n  _fk_objective_and_grad   : {obj_time_us:.0f} us/call")
    print(f"  At {MAX_ITERS} iters × {NUM_SEEDS} seeds = "
          f"{MAX_ITERS * NUM_SEEDS * obj_time_us / 1e6:.1f}s (worst case per IK)")

    # Speedup estimates
    print("\n  ── Potential optimisations ──")
    # a) Pre-allocate Jacobian buffers (avoid 2× np.zeros per iter)
    t_jac = bench_jacobian(ctx)
    t_rot = bench_rotation_error(ctx)
    t_fk = bench_mj_fwdPosition(ctx)
    t_obj = bench_full_fk_obj_and_grad(ctx, n=500)
    obj_mean = np.mean(t_obj) * 1e6
    jac_mean = np.mean(t_jac) * 1e6
    rot_mean = np.mean(t_rot) * 1e6
    fk_mean = np.mean(t_fk) * 1e6
    print(f"    FK (mj_fwdPosition)       : {fk_mean:.1f} us ({fk_mean/obj_mean*100:.0f}%)")
    print(f"    Rotation error (scipy.R)  : {rot_mean:.1f} us ({rot_mean/obj_mean*100:.0f}%)")
    print(f"    Jacobian (mj_jacSite)     : {jac_mean:.1f} us ({jac_mean/obj_mean*100:.0f}%)")
    other = obj_mean - fk_mean - rot_mean - jac_mean
    print(f"    Other (copy+grad+vstack)  : {other:.1f} us ({other/obj_mean*100:.0f}%)")

    # b) Reducing seeds
    for n_seeds in [48, 32, 24, 16, 8]:
        proj_ms = mean_ik_ms * n_seeds / NUM_SEEDS
        proj_segment = proj_ms / 1000 * N_ALIVE
        proj_pipeline = proj_segment * segments
        print(f"    {n_seeds:2d} seeds → ~{proj_ms:.0f} ms/call, "
              f"~{proj_segment:.1f}s/segment, ~{proj_pipeline:.0f}s pipeline")

    ctx["env"].close()
    print("\nDone.")


if __name__ == "__main__":
    main()
