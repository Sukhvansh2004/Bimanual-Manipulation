"""Ablation and baseline configuration definitions.

Each configuration is a dict with:
  - label:              Human-readable name for tables/plots
  - alpha:              Smoothness weight  (None → use experiment default)
  - beta:               Manipulability weight
  - beam_width:         DP beam width (None → unlimited)
  - handover_stride:    Evaluate handovers every K segments
  - clearance_radius:   Min object–obstacle distance for handover (None → off)
  - use_ik:             Whether to create env for IK/Jacobian/collision
  - disable_feasibility_filter: Skip grasp-pair feasibility filtering
  - switch_cost_override: Override switching cost (0 → free switches)
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
) -> AblationConfig:
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Baseline  (default experiment JSON values — α=10, β=1, beam=10)
# ═══════════════════════════════════════════════════════════════════════════

BASELINE = _cfg("Baseline (α=10, β=1)", beam_width=10)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Ablation: Without Switching Penalty  (C_switch = 0)
# ═══════════════════════════════════════════════════════════════════════════

NO_SWITCH_PENALTY = _cfg(
    "No Switching Penalty (Cswitch=0)",
    switch_cost_override=0.0,
    beam_width=10,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Ablation: Without Manipulability Cost  (β = 0)
# ═══════════════════════════════════════════════════════════════════════════

NO_MANIPULABILITY = _cfg(
    "No Manipulability (β=0)",
    beta=0.0,
    beam_width=10,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Ablation: Without Grasp Pair Feasibility Filtering
# ═══════════════════════════════════════════════════════════════════════════

NO_FEASIBILITY_FILTER = _cfg(
    "No Feasibility Filter",
    disable_feasibility_filter=True,
    beam_width=10,
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
CLEARANCE_005 = _cfg("Clearance=0.05m", clearance_radius=0.05, beam_width=10)
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
# 8. Method Comparisons (for Paper Table 1)
# ═══════════════════════════════════════════════════════════════════════════
# These correspond to the four methods compared in the paper:
#   Greedy         — beam_width=1 (locally optimal, no look-ahead)
#   Single Robot   — force all segments onto one robot (infinite switch cost)
#   No-Diversity   — disable grasp-pair feasibility filtering
#   Proposed       — full method (= BASELINE)

GREEDY = _cfg("Greedy (beam=1)", beam_width=1)

SINGLE_ROBOT = _cfg(
    "Single Robot",
    switch_cost_override=np.inf,   # effectively prohibit switches
    beam_width=10,
)

NO_DIVERSITY = _cfg(
    "No Diversity",
    disable_feasibility_filter=True,
    beam_width=10,
)

PROPOSED = _cfg("Proposed", beam_width=None)  # full method — no beam pruning


# ═══════════════════════════════════════════════════════════════════════════
# Grouped collections for the batch runner
# ═══════════════════════════════════════════════════════════════════════════

ALL_ABLATIONS: Dict[str, List[AblationConfig]] = {
    "baseline": [BASELINE],
    "switch_penalty": [BASELINE, NO_SWITCH_PENALTY],
    "manipulability": [BASELINE, NO_MANIPULABILITY],
    "feasibility_filter": [BASELINE, NO_FEASIBILITY_FILTER],
    "handover_stride": [STRIDE_1, STRIDE_3, STRIDE_5],
    "clearance_radius": [CLEARANCE_NONE, CLEARANCE_005, CLEARANCE_010, CLEARANCE_015],
    "beam_width": [BEAM_UNLIMITED, BEAM_1, BEAM_5, BEAM_10, BEAM_20],
    "method_comparison": [GREEDY, SINGLE_ROBOT, NO_DIVERSITY, PROPOSED],
    # Minimal set for the paper tables (6 unique configs, no duplicates)
    # Table 1: Greedy, Single Robot, No Diversity, Proposed
    # Table 2: Proposed (=Baseline), No Switching Penalty, No Manipulability, No Diversity (=No Feas. Filter)
    "paper": [PROPOSED, GREEDY, SINGLE_ROBOT, NO_SWITCH_PENALTY, NO_MANIPULABILITY],
}

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
    CLEARANCE_005,
    CLEARANCE_010,
    CLEARANCE_015,
    BEAM_UNLIMITED,
    BEAM_1,
    BEAM_5,
    BEAM_10,  # Same as BASELINE for beam
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
    """Return a filesystem-safe identifier for a config."""
    return cfg["label"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "").replace(".", "p")
