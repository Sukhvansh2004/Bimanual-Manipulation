#!/usr/bin/env python3
"""Sequential evaluation campaign for the failure taxonomy and learned filter.

For each (task, seed) instance:
  1. Generate grasp-conditioned robot trajectories into a per-seed data dir
     (kept, unlike ``pipeline_iterative.py`` which deletes them).
  2. Run the requested planner configs with ``--handover-log`` so every
     exact handover evaluation is recorded.
  3. Execute each successful assignment headlessly.

Runs one subprocess at a time to stay within limited RAM.  Completed steps
are skipped on re-run, so the campaign can be resumed after a crash.

Usage::

    python scripts/learned_filter/campaign.py --tasks mug_inversion \
        --seeds 1001 1002 --configs exact beam

    python scripts/learned_filter/campaign.py --tasks mug_inversion \
        --seeds 1004 --configs exact_learned beam_learned \
        --model models/handover_filter.pt --threshold 0.1 --cost-slack 0.8 --tag _s08
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, Tuple, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable
MAX_OOM_RETRIES = 2
TRAJGEN_LOCK = _PROJECT_ROOT / "results" / "campaign" / ".trajgen.lock"
OOM_RETRY_DELAY_S = 15

CONFIGS = {
    # Exact DP (paper "Proposed"): unlimited beam, handovers at every segment
    "exact": ["--clearance-radius", "0.03"],
    # Beam search (paper "Beam=10, Stride=3")
    "beam": ["--clearance-radius", "0.03", "--beam-width", "10", "--handover-stride", "3"],
    # Regrasp-graph-style baseline (Wan et al. 2020; Colombo et al. 2024) adapted to a
    # prescribed trajectory: feasibility-only edges, unit handover cost, i.e. the
    # minimum-handover path with no smoothness / manipulability terms.
    "regrasp_graph": ["--clearance-radius", "0.03", "--alpha", "0", "--beta", "0",
                      "--switch-cost-override", "1"],
}
CONFIGS["exact_learned"] = CONFIGS["exact"]
CONFIGS["beam_learned"] = CONFIGS["beam"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CAMPAIGN] %(message)s")
logger = logging.getLogger(__name__)


def run(
    cmd: Sequence[object], log_file: Path, timeout: int,
) -> Tuple[Union[int, str], float]:
    """Run a subprocess from the project root, logging its output to a file.

    Args:
        cmd: Command and arguments (converted to ``str``).
        log_file: File receiving stdout and stderr.
        timeout: Wall-clock limit in seconds.

    Returns:
        ``(return_code, wall_time_s)``; the return code is ``"timeout"``
        when the limit is hit.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s", " ".join(map(str, cmd)))
    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            rc = subprocess.run(
                list(map(str, cmd)), stdout=lf, stderr=lf, cwd=_PROJECT_ROOT,
                timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl"},
            ).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    dt = time.time() - t0
    logger.info("  -> rc=%s in %.0fs", rc, dt)
    return rc, dt


def kill_orphaned_workers() -> int:
    """Kill orphaned multiprocessing workers left behind by a killed parent.

    A SIGKILLed parent (e.g. by the OOM killer) leaves its ``spawn`` workers
    re-parented to init, still holding their memory.

    Returns:
        Number of processes signalled.
    """
    n = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            ppid = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == 1 and PY in cmdline and "multiprocessing.spawn" in cmdline:
            try:
                os.kill(int(proc.name), signal.SIGKILL)
                n += 1
            except OSError:
                pass
    return n


def run_with_retry(
    cmd: Sequence[object], log_file: Path, timeout: int, retries: int = MAX_OOM_RETRIES,
) -> Tuple[Union[int, str], float]:
    """Like :func:`run`, but retry after a SIGKILL (typically the OOM killer).

    Args:
        cmd: Command and arguments (converted to ``str``).
        log_file: File receiving stdout and stderr.
        timeout: Wall-clock limit in seconds per attempt.
        retries: Maximum number of retries after a SIGKILL.

    Returns:
        ``(return_code, wall_time_s)`` of the last attempt.
    """
    rc, dt = run(cmd, log_file, timeout)
    for attempt in range(retries):
        if rc != -signal.SIGKILL:
            break
        n = kill_orphaned_workers()
        logger.warning("  killed (likely OOM); reaped %d orphaned workers, retry %d/%d",
                       n, attempt + 1, retries)
        time.sleep(OOM_RETRY_DELAY_S)
        rc, dt = run(cmd, log_file, timeout)
    return rc, dt


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive inter-process file lock for the duration of the block.

    Trajectory generation starts an in-process GraspGen ZMQ service on a fixed
    port, so concurrent campaign processes must not run it at the same time.

    Args:
        path: Lock file path (created if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def planning_failure_reason(log_file: Path) -> str:
    """Classify a failed assignment run from its log.

    Args:
        log_file: Assignment log written by :func:`run`.

    Returns:
        ``"no_feasible_start"``, ``"dp_dead_end@<seg>/<total>"``,
        ``"no_terminal_state"`` or ``"crash"``.
    """
    text = log_file.read_text(errors="ignore") if log_file.exists() else ""
    if "all trajectories infeasible" in text:
        return "no_feasible_start"
    for line in text.splitlines():
        if "All states died at segment" in line:
            return "dp_dead_end@" + line.split("segment")[-1].strip().rstrip(".")
    if "No feasible assignment found" in text:
        return "no_terminal_state"
    return "crash"


def main() -> None:
    """Parse arguments and run all requested (task, seed, config) steps."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--configs", nargs="+", default=["exact", "beam"], choices=list(CONFIGS))
    ap.add_argument("--model", default=None, help="Learned filter checkpoint (for *_learned configs)")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--cost-slack", type=float, default=0.0)
    ap.add_argument("--filter-order", action="store_true",
                    help="Best-predicted-first candidate ordering (for *_learned configs)")
    ap.add_argument("--tag", default="", help="Suffix for *_learned result names (e.g. τ value)")
    ap.add_argument("--results", default="results/campaign")
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--plan-timeout", type=int, default=5400)
    args = ap.parse_args()

    for task in args.tasks:
        exp_json = f"experiments/{task}.json"
        for seed in args.seeds:
            data_dir = Path("data/campaign") / task / f"s{seed}"
            res_dir = Path(args.results) / task / f"s{seed}"
            res_dir.mkdir(parents=True, exist_ok=True)

            # 1. Trajectory generation
            if not (_PROJECT_ROOT / data_dir / "object_trajectory.npz").exists() or \
                    not list((_PROJECT_ROOT / data_dir).glob("robot1_trajectory_*.npz")):
                with exclusive_lock(TRAJGEN_LOCK):
                    rc, dt = run_with_retry(
                        [PY, "scripts/pipeline/trajectory_generation.py",
                         "--experiment", exp_json, "--save-traj", "--no-meshcat",
                         "--workers", "1", "--seed", seed, "--output-dir", data_dir],
                        res_dir / "trajgen.log", timeout=3600)
                if rc != 0:
                    logger.error("Trajectory generation failed for %s s%d", task, seed)
                    continue

            for cfg in args.configs:
                name = cfg
                if cfg.endswith("_learned"):
                    name = f"{cfg}{args.tag}"
                out_json = res_dir / f"{name}.json"
                if out_json.exists():
                    logger.info("skip %s/%s/%s (done)", task, seed, name)
                    continue

                # 2. Assignment
                cmd = [PY, "scripts/pipeline/segment_assignment.py",
                       "--experiment", exp_json, "--data-dir", data_dir, "--no-plot",
                       "--seed", seed,
                       "--output-assignment", res_dir / f"assignment_{name}.npz",
                       "--output-metrics", res_dir / f"{name}_planning.json",
                       "--handover-log", res_dir / f"handover_log_{name}.npz",
                       *CONFIGS[cfg]]
                if cfg.endswith("_learned"):
                    if args.model is None:
                        ap.error("--model is required for *_learned configs")
                    cmd += ["--learned-filter", args.model, "--filter-threshold", args.threshold,
                            "--cost-slack", args.cost_slack]
                    if args.filter_order:
                        cmd.append("--filter-order")
                rc, dt = run_with_retry(cmd, res_dir / f"{name}_assignment.log", args.plan_timeout)

                record = {"task": task, "seed": seed, "config": name,
                          "wall_time_s": dt, "planning_success": rc == 0}
                pm = res_dir / f"{name}_planning.json"
                if rc == 0 and pm.exists():
                    record["planning"] = json.loads(pm.read_text())
                else:
                    record["failure_reason"] = (
                        "timeout" if rc == "timeout"
                        else planning_failure_reason(res_dir / f"{name}_assignment.log"))

                # 3. Execution
                if rc == 0 and not args.no_execute:
                    em = res_dir / f"{name}_exec.json"
                    erc, edt = run_with_retry([PY, "scripts/pipeline/dual_arm_handover_traj.py",
                                    "--experiment", exp_json,
                                    "--assignment", res_dir / f"assignment_{name}.npz",
                                    "--data-dir", data_dir,
                                    "--output-metrics", em,
                                    "--headless", "--no-meshcat", "--smoothen", "--seed", seed],
                                   res_dir / f"{name}_exec.log", timeout=900)
                    record["execution"] = json.loads(em.read_text()) if em.exists() else None
                    record["execution_rc"] = erc

                out_json.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
