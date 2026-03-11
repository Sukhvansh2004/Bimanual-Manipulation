#!/usr/bin/env python3
"""
Find Feasible Arm Trajectory Pair.

Given a data directory containing pre-generated trajectories
(object_trajectory.npz, robot0_trajectory_*.npz, robot1_trajectory_*.npz),
this script finds pairs where BOTH robot trajectories are feasible at every
waypoint, using only the pre-computed feasibility data stored in the .npz files.

No environment or experiment config is needed — everything comes from the data.

The script:
1. Loads the per-waypoint `is_feasible` arrays from all robot trajectory files.
2. For each (robot0_i, robot1_j) pair, computes the logical AND of their
   feasibility arrays to find jointly feasible waypoints.
3. Reports all fully feasible pairs and the best partial pairs.

Usage:
    python find_feasible_pair.py <data_dir>

Examples:
    python find_feasible_pair.py data/shelf_to_box
    python find_feasible_pair.py data/ycb_mug_table_to_rack
    python find_feasible_pair.py data/ycb_mug_table_to_rack --skip-prefilter
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_index(filename: str, prefix: str) -> Optional[int]:
    """Extract the integer index from e.g. 'robot0_trajectory_7.npz'."""
    m = re.match(rf"{prefix}_(\d+)\.npz$", filename)
    return int(m.group(1)) if m else None


def load_feasibility_arrays(data_dir: str):
    """Load is_feasible arrays from all robot trajectory files.

    Returns:
        robot0: dict[int -> np.ndarray(bool)]  – per-waypoint feasibility
        robot1: dict[int -> np.ndarray(bool)]
    """
    robot0: Dict[int, np.ndarray] = {}
    robot1: Dict[int, np.ndarray] = {}

    for fpath in sorted(glob.glob(os.path.join(data_dir, "robot*_trajectory_*.npz"))):
        fname = os.path.basename(fpath)
        for prefix, store in [("robot0_trajectory", robot0), ("robot1_trajectory", robot1)]:
            idx = _parse_index(fname, prefix)
            if idx is not None:
                data = np.load(fpath)
                if "is_feasible" not in data:
                    print(f"  WARNING: {fname} has no 'is_feasible' field – skipping.")
                    break
                store[idx] = data["is_feasible"].astype(bool)
                break

    return robot0, robot1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find feasible dual-arm trajectory pairs from pre-generated data."
    )
    parser.add_argument("data_dir", type=str, help="Path to data directory containing trajectories")
    parser.add_argument("--skip-prefilter", action="store_true",
                        help="Include individually-infeasible trajectories in pair search")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: directory not found: {data_dir}")
        sys.exit(1)

    # ---- 1. Load feasibility arrays ----
    robot0, robot1 = load_feasibility_arrays(data_dir)
    print(f"Loaded {len(robot0)} robot0 and {len(robot1)} robot1 trajectories from {data_dir}")

    if not robot0 or not robot1:
        print("Error: need at least one trajectory for each robot.")
        sys.exit(1)

    # ---- 2. Pre-filter individually feasible ----
    if args.skip_prefilter:
        r0 = robot0
        r1 = robot1
    else:
        r0 = {k: v for k, v in robot0.items() if v.all()}
        r1 = {k: v for k, v in robot1.items() if v.all()}
        print(f"Pre-filter: robot0 {len(r0)}/{len(robot0)} fully feasible, "
              f"robot1 {len(r1)}/{len(robot1)} fully feasible")

    if not r0 or not r1:
        print("No individually feasible trajectories for one or both robots.")
        print("Try --skip-prefilter to check all pairs anyway.")
        sys.exit(1)

    # ---- 3. Evaluate all pairs ----
    total_pairs = len(r0) * len(r1)
    print(f"Evaluating {total_pairs} pairs ({len(r0)} x {len(r1)})...")

    results = []
    for i in sorted(r0):
        for j in sorted(r1):
            f0 = r0[i]
            f1 = r1[j]
            covered = f0 | f1  # waypoint is feasible if EITHER robot can reach it
            n_ok = int(covered.sum())
            n_total = len(covered)
            # Per-waypoint attribution
            both = f0 & f1
            only0 = f0 & ~f1
            only1 = f1 & ~f0

            # Handover validity: every transition from one robot's exclusive
            # region to the other's must pass through at least one BOTH waypoint.
            # Walk through waypoints tracking which robot is "active". A switch
            # without an intervening BOTH waypoint makes the pair invalid.
            has_valid_handovers = True
            if covered.all():
                last_exclusive = None  # 0 = robot0-only, 1 = robot1-only
                saw_both_since_last_exclusive = False
                for k in range(n_total):
                    if both[k]:
                        saw_both_since_last_exclusive = True
                    elif only0[k]:
                        if last_exclusive == 1 and not saw_both_since_last_exclusive:
                            has_valid_handovers = False
                            break
                        last_exclusive = 0
                        saw_both_since_last_exclusive = False
                    elif only1[k]:
                        if last_exclusive == 0 and not saw_both_since_last_exclusive:
                            has_valid_handovers = False
                            break
                        last_exclusive = 1
                        saw_both_since_last_exclusive = False
                    # else: neither feasible → already caught by covered.all()

            n_both = int(both.sum())
            results.append({
                "i": i,
                "j": j,
                "all_covered": covered.all(),
                "valid_handovers": has_valid_handovers,
                "all_feasible": covered.all() and has_valid_handovers,
                "joint_feasible_count": n_ok,
                "both_count": n_both,
                "total_waypoints": n_total,
                "both": both,
                "only0": only0,
                "only1": only1,
                "covered": covered,
            })

    # ---- 4. Report ----
    feasible_pairs = [r for r in results if r["all_feasible"]]
    partial_pairs = sorted(
        [r for r in results if not r["all_feasible"]],
        key=lambda r: r["joint_feasible_count"],
        reverse=True,
    )

    # Also collect pairs that are fully covered but fail handover check
    covered_no_handover = [r for r in results if r["all_covered"] and not r["valid_handovers"]]

    print("\n" + "=" * 70)
    print(f"  RESULTS  –  {len(feasible_pairs)} fully feasible pair(s) out of {total_pairs}")
    print(f"             ({len(covered_no_handover)} covered all waypoints but lack valid handover overlap)")
    print("=" * 70)

    if feasible_pairs:
        print("\n  FEASIBLE PAIRS (all waypoints covered + valid handovers):")
        print("  " + "-" * 50)
        for r in feasible_pairs:
            print(f"\n    robot0_trajectory_{r['i']}  +  robot1_trajectory_{r['j']}    "
                  f"({r['joint_feasible_count']}/{r['total_waypoints']} covered, "
                  f"{r['both_count']} BOTH)")
            print(f"    {'Waypoint':<10} {'Feasible by':<15}")
            print(f"    {'--------':<10} {'-----------':<15}")
            for k in range(r["total_waypoints"]):
                if r["both"][k]:
                    tag = "BOTH  <-- handover possible"
                elif r["only0"][k]:
                    tag = "robot0"
                elif r["only1"][k]:
                    tag = "robot1"
                else:
                    tag = "NONE"
                print(f"    {k:<10} {tag}")
    else:
        print("\n  No fully feasible pairs found.")

    if partial_pairs:
        top_n = min(10, len(partial_pairs))
        print(f"\n  TOP {top_n} PARTIAL PAIRS (ranked by jointly feasible waypoints):")
        print("  " + "-" * 50)
        for r in partial_pairs[:top_n]:
            pct = 100.0 * r["joint_feasible_count"] / r["total_waypoints"]
            print(f"    robot0_trajectory_{r['i']}  +  robot1_trajectory_{r['j']}    "
                  f"{r['joint_feasible_count']}/{r['total_waypoints']} ({pct:.0f}%)")

    print()

    if feasible_pairs:
        best = feasible_pairs[0]
        bi, bj = best["i"], best["j"]
        print(f"  BEST PAIR:  robot0_trajectory_{bi}  +  robot1_trajectory_{bj}")
        print(f"    traj0: {os.path.join(data_dir, f'robot0_trajectory_{bi}.npz')}")
        print(f"    traj1: {os.path.join(data_dir, f'robot1_trajectory_{bj}.npz')}")
    print()


if __name__ == "__main__":
    main()
