"""EAIK stress test — simulates the IK workload of segment assignment.

Mimics the handover-evaluation IK volume from a mug_inversion-scale run:
  - ~5,700 handover evaluations
  - 10 IK poses per handover (4 transfer + 6 receiver)
  - Each IK call sweeps 7 joints × 8 values = 56 analytical solves

Separates two bottlenecks:
  1. UrdfRobot construction (URDF parsing / C++ build) — dominates on cache miss
  2. bot.IK() analytical solve — the actual IK computation

The production code uses @lru_cache(maxsize=256) ≈ 4.6 warm-start configs
(256 / 56 keys-per-config). With warm-starting, q_reference changes every
call → cache thrashes → UrdfRobot rebuild dominates.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from eaik.IK_URDF import UrdfRobot
from bimanual_manipulation.constants import (
    PANDA_URDF_PATH,
    FRANKA_HOME_JOINT_POSITION,
    EAIK_NUM_SWEEP_VALUES,
    EAIK_WARM_START_WINDOW,
    EAIK_GRID_STEP,
)

# ── Panda joint limits (from URDF) ──
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

URDF = str(PANDA_URDF_PATH)

# ── Segment-assignment scale constants ──
COST_EVAL_APPROACH_STEPS = 5
COST_EVAL_RETREAT_STEPS = 3
POSES_PER_TRANSFER = 1 + COST_EVAL_RETREAT_STEPS   # hold + retreat = 4
POSES_PER_RECEIVER = 1 + COST_EVAL_APPROACH_STEPS   # pre + approach = 6
POSES_PER_HANDOVER = POSES_PER_TRANSFER + POSES_PER_RECEIVER  # 10

# Mug-inversion-scale run: ~5,700 switching-cost evaluations
N_HANDOVERS = 5700
TOTAL_IK_CALLS = N_HANDOVERS * POSES_PER_HANDOVER  # ~57,000


# ── Helpers ──

_bot_cache = {}
_build_time_accum = 0.0  # track UrdfRobot construction time
_ik_time_accum = 0.0     # track bot.IK() solve time
_build_count = 0
_cache_hits = 0

def _build_robot(locked_joint: int, locked_value: float):
    global _build_time_accum, _build_count, _cache_hits
    key = (locked_joint, round(locked_value, 6))
    if key in _bot_cache:
        _cache_hits += 1
        return _bot_cache[key]
    t0 = time.perf_counter()
    _bot_cache[key] = UrdfRobot(URDF, fixed_axes=[(locked_joint, locked_value)])
    _build_time_accum += time.perf_counter() - t0
    _build_count += 1
    return _bot_cache[key]

def _reset_counters():
    global _build_time_accum, _ik_time_accum, _build_count, _cache_hits
    _build_time_accum = 0.0
    _ik_time_accum = 0.0
    _build_count = 0
    _cache_hits = 0


def _sweep_values(joint_idx, q_ref=None, num_sweep=EAIK_NUM_SWEEP_VALUES):
    lo, hi = float(JOINT_LO[joint_idx]), float(JOINT_HI[joint_idx])
    if q_ref is not None:
        centre = float(np.clip(q_ref[joint_idx], lo, hi))
        half = EAIK_WARM_START_WINDOW
        w_lo, w_hi = max(lo, centre - half), min(hi, centre + half)
        vals = np.linspace(w_lo, w_hi, num_sweep)
        if not np.any(np.isclose(vals, centre, atol=1e-8)):
            vals = np.sort(np.append(vals, centre))
    else:
        vals = np.linspace(lo, hi, num_sweep)
    # Snap to fixed grid (matches production eaik_solver.py)
    vals = np.unique(np.clip(np.round(vals / EAIK_GRID_STEP) * EAIK_GRID_STEP, lo, hi))
    return vals


def solve_ik_eaik(target_pos, target_quat, q_reference=None):
    """Minimal replica of solve_ik_eaik — sweep all 7 joints."""
    global _ik_time_accum
    rot_mat = R.from_quat(target_quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = target_pos

    solutions = []
    for accept_ls in (False, True):
        for jidx in range(7):
            for val in _sweep_values(jidx, q_reference):
                try:
                    bot = _build_robot(jidx, float(val))
                except Exception:
                    continue
                try:
                    t0 = time.perf_counter()
                    result = bot.IK(T)
                    _ik_time_accum += time.perf_counter() - t0
                except Exception:
                    continue
                for q_sol, is_ls in zip(result.Q, result.is_LS):
                    if is_ls and not accept_ls:
                        continue
                    q = np.asarray(q_sol, dtype=float)
                    if q.shape[0] != 7:
                        continue
                    if np.any(q < JOINT_LO - 1e-6) or np.any(q > JOINT_HI + 1e-6):
                        continue
                    solutions.append(np.clip(q, JOINT_LO, JOINT_HI))
        if solutions:
            break
    return solutions


def _random_reachable_pose(rng):
    """Generate a random pose roughly in the Panda workspace."""
    pos = rng.uniform([0.2, -0.4, 0.1], [0.6, 0.4, 0.7])
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return pos, q


def _perturbed_pose(pos, quat, rng, pos_noise=0.02, rot_noise=0.05):
    """Small perturbation — mimics consecutive waypoints along a handover."""
    new_pos = pos + rng.uniform(-pos_noise, pos_noise, 3)
    rotvec = R.from_quat(quat).as_rotvec() + rng.uniform(-rot_noise, rot_noise, 3)
    new_quat = R.from_rotvec(rotvec).as_quat()
    return new_pos, new_quat


# ── Main stress test ──

def main():
    rng = np.random.default_rng(42)
    q_home = FRANKA_HOME_JOINT_POSITION

    print("=" * 70)
    print("EAIK Stress Test — Segment-Assignment Scale")
    print(f"  Handovers to evaluate : {N_HANDOVERS:,}")
    print(f"  IK poses per handover : {POSES_PER_HANDOVER}")
    print(f"  Total IK calls        : {TOTAL_IK_CALLS:,}")
    print(f"  Sweep per joint       : {EAIK_NUM_SWEEP_VALUES}")
    print(f"  Sub-solves per call   : 7 × {EAIK_NUM_SWEEP_VALUES} = {7 * EAIK_NUM_SWEEP_VALUES}")
    print(f"  LRU cache size        : 256 (holds ~{256 // (7 * EAIK_NUM_SWEEP_VALUES)} warm-start configs)")
    print("=" * 70)

    # ── 1. Cached single-call timing (best case — same q_reference) ──
    print("\n--- 1. Cached single-call (same q_ref=q_home, 1000 calls) ---")
    print("       All UrdfRobot objects pre-built → measures pure IK solve time")
    # Pre-build cache for q_home
    for jidx in range(7):
        for val in _sweep_values(jidx, q_home):
            _build_robot(jidx, float(val))

    _reset_counters()
    pos0, quat0 = _random_reachable_pose(rng)
    times_cached = []
    for _ in range(1000):
        p, q = _perturbed_pose(pos0, quat0, rng)
        t_start = time.perf_counter()
        sols = solve_ik_eaik(p, q, q_reference=q_home)
        times_cached.append(time.perf_counter() - t_start)
    tc = np.array(times_cached) * 1000
    print(f"  Mean   : {tc.mean():.3f} ms/call (≈ pure bot.IK() cost)")
    print(f"  Median : {np.median(tc):.3f} ms/call")
    print(f"  P95    : {np.percentile(tc, 95):.3f} ms")
    print(f"  Builds : {_build_count} (should be 0)")
    print(f"  Hits   : {_cache_hits}")

    # ── 2. Uncached single-call timing (worst case — fresh q_reference each time) ──
    print("\n--- 2. Uncached single-call (varying q_ref, 100 calls) ---")
    print("       UrdfRobot rebuilt every call → measures build + IK cost")
    _bot_cache.clear()
    _reset_counters()
    times_uncached = []
    q_prev = q_home.copy()
    for i in range(100):
        p, q = _perturbed_pose(pos0, quat0, rng)
        t_start = time.perf_counter()
        sols = solve_ik_eaik(p, q, q_reference=q_prev)
        times_uncached.append(time.perf_counter() - t_start)
        if sols:
            q_prev = sols[0]  # warm-start changes → new cache keys
    tu = np.array(times_uncached) * 1000
    print(f"  Mean   : {tu.mean():.1f} ms/call (build + IK)")
    print(f"  Median : {np.median(tu):.1f} ms/call")
    print(f"  P95    : {np.percentile(tu, 95):.1f} ms")
    print(f"  Builds : {_build_count} ({_build_time_accum*1000:.1f} ms total, "
          f"{_build_time_accum/_build_count*1000:.3f} ms/build)")
    print(f"  Hits   : {_cache_hits}")
    print(f"  IK     : {_ik_time_accum*1000:.1f} ms total "
          f"({_ik_time_accum/100*1000:.3f} ms/call)")
    pct_build = _build_time_accum / (_build_time_accum + _ik_time_accum) * 100
    print(f"  Split  : {pct_build:.0f}% build, {100-pct_build:.0f}% IK")

    # ── 3. Handover-level timing (realistic workload) ──
    n_handovers_bench = min(N_HANDOVERS, 500)
    print(f"\n--- 3. Handover-level timing ({n_handovers_bench} handovers, 10 IK each) ---")
    _bot_cache.clear()
    _reset_counters()
    handover_times = []
    ik_fail_count = 0
    for h in range(n_handovers_bench):
        base_pos, base_quat = _random_reachable_pose(rng)
        q_warm = q_home.copy()
        t_handover_start = time.perf_counter()

        # Transfer arm: 4 poses (hold + retreat)
        for i in range(POSES_PER_TRANSFER):
            p, q = _perturbed_pose(base_pos, base_quat, rng)
            sols = solve_ik_eaik(p, q, q_reference=q_warm)
            if sols:
                q_warm = sols[0]
            else:
                ik_fail_count += 1

        # Receiver arm: 6 poses (pre + approach)
        q_warm = q_home.copy()
        for i in range(POSES_PER_RECEIVER):
            p, q = _perturbed_pose(base_pos, base_quat, rng)
            sols = solve_ik_eaik(p, q, q_reference=q_warm)
            if sols:
                q_warm = sols[0]
            else:
                ik_fail_count += 1

        elapsed = time.perf_counter() - t_handover_start
        handover_times.append(elapsed)

        if (h + 1) % 100 == 0:
            avg_so_far = np.mean(handover_times) * 1000
            print(f"  [{h+1:4d}/{n_handovers_bench}] avg={avg_so_far:.1f} ms/handover")

    ht = np.array(handover_times) * 1000
    print(f"\n  Handover timing ({n_handovers_bench} handovers):")
    print(f"    Mean   : {ht.mean():.1f} ms/handover")
    print(f"    Median : {np.median(ht):.1f} ms/handover")
    print(f"    P95    : {np.percentile(ht, 95):.1f} ms")
    print(f"    IK failures: {ik_fail_count}/{n_handovers_bench * POSES_PER_HANDOVER}")

    total_wall = ht.sum() / 1000
    print(f"\n  Time breakdown (total {total_wall:.1f}s):")
    print(f"    UrdfRobot builds: {_build_time_accum:.1f}s "
          f"({_build_time_accum/total_wall*100:.0f}%) — "
          f"{_build_count} builds, {_build_time_accum/_build_count*1000:.2f} ms/build")
    print(f"    bot.IK() solves : {_ik_time_accum:.1f}s "
          f"({_ik_time_accum/total_wall*100:.0f}%)")
    print(f"    Other overhead  : {total_wall - _build_time_accum - _ik_time_accum:.1f}s")
    print(f"    Cache hits      : {_cache_hits}, misses (builds): {_build_count}")

    # ── Full-run projection ──
    mean_per_handover_s = ht.mean() / 1000
    projected_total = mean_per_handover_s * N_HANDOVERS
    ik_only_per_handover = _ik_time_accum / n_handovers_bench
    projected_ik_only = ik_only_per_handover * N_HANDOVERS
    print(f"\n--- Full-run projection ({N_HANDOVERS:,} handovers) ---")
    print(f"  With current caching : {projected_total:.1f}s ({projected_total/60:.1f} min)")
    print(f"  IK-only (no builds)  : {projected_ik_only:.1f}s ({projected_ik_only/60:.1f} min)")
    print(f"  Speedup if cached    : {projected_total/projected_ik_only:.1f}×")

    print("\nDone.")


if __name__ == "__main__":
    main()
