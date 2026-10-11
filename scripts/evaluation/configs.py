"""Ablation and baseline configuration definitions.

Each configuration is a dict with:
  - label:              Human-readable name for tables/plots
  - alpha:              Smoothness weight  (None → use experiment default)
  - beta:               Manipulability weight  (None → use experiment default)
  - beam_width:         DP beam width (None → unlimited)
  - handover_stride:    Evaluate handovers every K segments (default 1)
  - clearance_radius:   Min object–obstacle distance for handover (None → off)
  - use_ik:             Whether to create env for IK/Jacobian/collision
  - disable_feasibility_filter: Skip grasp-pair feasibility filtering
  - switch_cost_override: Override switching cost (0 → free switches)
  - use_dpp:            Use DPP entropy maximization for grasp selection (False → top-N by score)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Configuration definition
# ═══════════════════════════════════════════════════════════════════════════

AblationConfig = Dict[str, Any]


def _cfg(
    label: str,
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    beam_width: Optional[int] = None,
    handover_stride: int = 1,
    clearance_radius: Optional[float] = None,
    use_ik: bool = True,
    disable_feasibility_filter: bool = False,
    switch_cost_override: Optional[float] = None,
    use_dpp: bool = True,
) -> AblationConfig:
    """Build an ablation config dict.

    Args:
        label: Human-readable name; also the source of ``config_id``.
        alpha: Smoothness weight, or ``None`` for the experiment default.
        beta: Manipulability weight, or ``None`` for the experiment default.
        beam_width: DP beam width, or ``None`` for unlimited.
        handover_stride: Evaluate handovers every K segments.
        clearance_radius: Min object–obstacle handover clearance in metres,
            or ``None`` to disable.
        use_ik: Whether to create an env for IK/Jacobian/collision checks.
        disable_feasibility_filter: Skip grasp-pair feasibility filtering.
        switch_cost_override: Switching cost override (0 → free switches,
            ``np.inf`` → no switches), or ``None`` to keep the default.
        use_dpp: Use DPP entropy maximization for grasp selection.

    Returns:
        Config dict with one key per argument.
    """
    return {
        "label": label,
        "alpha": alpha,
        "beta": beta,
        "beam_width": beam_width,
        "handover_stride": handover_stride,
        "clearance_radius": clearance_radius,
        "use_ik": use_ik,
        "disable_feasibility_filter": disable_feasibility_filter,
        "switch_cost_override": switch_cost_override,
        "use_dpp": use_dpp,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Baseline  (default experiment JSON values — α=40, β=0.5)
# ═══════════════════════════════════════════════════════════════════════════

BASELINE = _cfg("Baseline (α=40, β=0.5)", clearance_radius=0.03)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Ablation: Without Switching Penalty  (C_switch = 0)
# ═══════════════════════════════════════════════════════════════════════════

NO_SWITCH_PENALTY = _cfg(
    "No Switching Penalty (Cswitch=0)",
    switch_cost_override=0.0,
    clearance_radius=0.03,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Ablation: Without Manipulability Cost  (β = 0)
# ═══════════════════════════════════════════════════════════════════════════

NO_MANIPULABILITY = _cfg(
    "No Manipulability (β=0)",
    beta=0.0,
    clearance_radius=0.03,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Ablation: Without Grasp Pair Feasibility Filtering
# ═══════════════════════════════════════════════════════════════════════════

NO_FEASIBILITY_FILTER = _cfg(
    "No Feasibility Filter",
    disable_feasibility_filter=True,
    clearance_radius=0.03,
)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Ablation: Handover Sampling Rate  (stride = 1, 3, 5)
# ═══════════════════════════════════════════════════════════════════════════

STRIDE_1 = _cfg("Stride=1", handover_stride=1, beam_width=10)
STRIDE_3 = _cfg("Stride=3", handover_stride=3, beam_width=10)
STRIDE_5 = _cfg("Stride=5", handover_stride=5, beam_width=10)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Ablation: Handover Clearance Radius
# ═══════════════════════════════════════════════════════════════════════════

CLEARANCE_NONE = _cfg("Clearance=off", clearance_radius=None, beam_width=10)
CLEARANCE_003 = _cfg("Clearance=0.03m", clearance_radius=0.03, beam_width=10)
CLEARANCE_010 = _cfg("Clearance=0.10m", clearance_radius=0.10, beam_width=10)
CLEARANCE_015 = _cfg("Clearance=0.15m", clearance_radius=0.15, beam_width=10)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Ablation: Beam Pruning  (unlimited, 1, 5, 10, 20)
# ═══════════════════════════════════════════════════════════════════════════

BEAM_UNLIMITED = _cfg("Beam=unlimited", beam_width=None)
BEAM_1 = _cfg("Beam=1 (greedy)", beam_width=1)
BEAM_5 = _cfg("Beam=5", beam_width=5)
BEAM_10 = _cfg("Beam=10", beam_width=10)
BEAM_20 = _cfg("Beam=20", beam_width=20)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Misc.
# ═══════════════════════════════════════════════════════════════════════════

BEAM_STRIDE = _cfg("Beam=10, Stride=3", beam_width=10, handover_stride=3, clearance_radius=0.03)

# ═══════════════════════════════════════════════════════════════════════════
# 9. Method Comparisons (for Paper Table 1)
# ═══════════════════════════════════════════════════════════════════════════
# These correspond to the four methods compared in the paper:
#   Greedy         — beam_width=1 (locally optimal, no look-ahead)
#   Single Robot   — force all segments onto one robot (infinite switch cost)
#   No-Diversity   — disable DPP entropy maximization (use top-N grasps by score)
#   Proposed       — full method (= BASELINE)

GREEDY = _cfg("Greedy (beam=1)", beam_width=1, clearance_radius=0.03)

SINGLE_ROBOT = _cfg(
    "Single Robot",
    switch_cost_override=np.inf,   # effectively prohibit switches
    clearance_radius=0.03,
)

NO_DIVERSITY = _cfg(
    "No Diversity",
    use_dpp=False,
    clearance_radius=0.03,
)

PROPOSED = _cfg("Proposed", beam_width=None, clearance_radius=0.03)  # full method — no beam pruning


# ═══════════════════════════════════════════════════════════════════════════
# Grouped collections for the batch runner
# ═══════════════════════════════════════════════════════════════════════════

ALL_ABLATIONS: Dict[str, List[AblationConfig]] = {
    "baseline": [BASELINE],
    "switch_penalty": [BASELINE, NO_SWITCH_PENALTY],
    "manipulability": [BASELINE, NO_MANIPULABILITY],
    "feasibility_filter": [BASELINE, NO_FEASIBILITY_FILTER],
    "handover_stride": [STRIDE_1, STRIDE_3, STRIDE_5],
    "clearance_radius": [CLEARANCE_NONE, CLEARANCE_003, CLEARANCE_010, CLEARANCE_015],
    "beam_width": [BEAM_UNLIMITED, BEAM_1, BEAM_5, BEAM_10, BEAM_20],
    "method_comparison": [GREEDY, SINGLE_ROBOT, NO_DIVERSITY, PROPOSED],
    # Minimal set for the paper tables
    # Table 1 (Method Comparison): Proposed, Greedy, Single Robot, No Diversity
    # Table 2 (Ablation Study):    Proposed, No Switching Penalty, No Manipulability
    # NOTE: NO_DIVERSITY has use_dpp=False → run_experiment() auto-switches to data/<task>_no_dpp/
    "paper": [PROPOSED, GREEDY, SINGLE_ROBOT, NO_DIVERSITY, NO_SWITCH_PENALTY, NO_MANIPULABILITY, BEAM_STRIDE, STRIDE_1, STRIDE_3, STRIDE_5, BEAM_5, BEAM_10, BEAM_20],
    "custom": [BEAM_STRIDE]
}

# ═══════════════════════════════════════════════════════════════════════════
# Paper Experiments
# ═══════════════════════════════════════════════════════════════════════════

PAPER_EXPERIMENTS = [
    "flatPlate_table_to_shelf",
    "mug_inversion",
    "flatPlate_box_to_box",
    "ycb_plate_shelf_to_table",
]

# Flat list of all unique configs for running everything
ALL_CONFIGS: List[AblationConfig] = [
    BASELINE,
    NO_SWITCH_PENALTY,
    NO_MANIPULABILITY,
    NO_FEASIBILITY_FILTER,
    STRIDE_1,  # Same as BASELINE for stride
    STRIDE_3,
    STRIDE_5,
    CLEARANCE_NONE,
    CLEARANCE_003,
    CLEARANCE_010,
    CLEARANCE_015,
    BEAM_UNLIMITED,
    BEAM_1,
    BEAM_5,
    BEAM_10,
    BEAM_20,
    GREEDY,
    SINGLE_ROBOT,
    NO_DIVERSITY,
    PROPOSED,
]

# Remove duplicates by label
_seen = set()
_unique: List[AblationConfig] = []
for c in ALL_CONFIGS:
    if c["label"] not in _seen:
        _seen.add(c["label"])
        _unique.append(c)
ALL_CONFIGS = _unique


def config_id(cfg: AblationConfig) -> str:
    """Return a filesystem-friendly identifier for a config.

    Derived from ``cfg["label"]``: spaces become ``_``, ``(``, ``)``, ``,``
    and ``=`` are removed, and ``.`` becomes ``p``. Other characters (e.g.
    ``α``) are kept.

    Args:
        cfg: Ablation config dict.

    Returns:
        Identifier string, e.g. ``Beam10_Stride3`` for ``"Beam=10, Stride=3"``.
    """
    return cfg["label"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "").replace(".", "p")
