"""JSON-based results persistence.

Results are stored per-experiment under ``results/<experiment>/<config_id>.json``
with planning and execution metrics.  A companion ``results/summary.json``
aggregates all results for quick loading by the analysis module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.evaluation.metrics import ExperimentResult, PlanningMetrics, ExecutionMetrics

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy types to JSON-safe Python types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_result(result: ExperimentResult, results_dir: Optional[Path] = None) -> Path:
    """Save an ExperimentResult to JSON.

    File path: ``<results_dir>/<experiment>/<config_id>.json``

    Returns:
        Path to the saved JSON file.
    """
    root = results_dir or RESULTS_ROOT
    exp_dir = root / result.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    out_path = exp_dir / f"{result.config_id}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, cls=_NumpyEncoder)

    logger.info("Saved result: %s", out_path)
    return out_path


def load_result(path: Path) -> ExperimentResult:
    """Load an ExperimentResult from JSON."""
    with open(path) as f:
        d = json.load(f)

    planning = None
    if "planning" in d and d["planning"] is not None:
        planning = PlanningMetrics(**d["planning"])

    execution = None
    if "execution" in d and d["execution"] is not None:
        execution = ExecutionMetrics(**d["execution"])

    return ExperimentResult(
        experiment=d["experiment"],
        config_label=d["config_label"],
        config_id=d["config_id"],
        planning=planning,
        execution=execution,
    )


def load_all_results(results_dir: Optional[Path] = None) -> List[ExperimentResult]:
    """Load all results from the results directory recursively."""
    root = results_dir or RESULTS_ROOT
    results: List[ExperimentResult] = []

    if not root.exists():
        return results

    # Search recursively for all .json files
    for json_file in sorted(root.rglob("*.json")):
        if json_file.name == "summary.json" or json_file.name == "aggregate_summary.json":
            continue
        # Skip standalone exec metric files (raw output from dual_arm_handover_traj.py);
        # their data is already embedded in the main <config_id>.json results.
        if json_file.stem.endswith("_exec"):
            continue
        try:
            results.append(load_result(json_file))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)

    return results


def build_summary(results_dir: Optional[Path] = None) -> Dict:
    """Build and save a summary.json combining all results.

    Returns the summary dict.
    """
    root = results_dir or RESULTS_ROOT
    results = load_all_results(root)

    summary = {
        "n_results": len(results),
        "experiments": sorted({r.experiment for r in results}),
        "configs": sorted({r.config_label for r in results}),
        "results": [r.to_dict() for r in results],
    }

    summary_path = root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    logger.info("Summary saved: %s (%d results)", summary_path, len(results))
    return summary
