#!/bin/bash
# ============================================================================
# Step 1: Generate EEF trajectories for all 4 paper tasks.
#
# For each task, generates trajectories WITH DPP (default) and WITHOUT DPP
# (for the "No Diversity" ablation).  Existing robot trajectories are cleaned
# before each run; object trajectories are kept and reused.
#
# DPP trajectories → data/<task>/               (used by Proposed, Greedy,
#                                                 SingleRobot, NoSwitch, NoManip)
# No-DPP trajs     → data/<task>_no_dpp/        (used by NoDiversity ablation)
#
# Usage:
#   ./scripts/paper_1_generate_trajectories.sh              # all tasks
#   ./scripts/paper_1_generate_trajectories.sh --workers 2  # override workers
#   ./scripts/paper_1_generate_trajectories.sh --skip-dpp   # only generate no-DPP
#   ./scripts/paper_1_generate_trajectories.sh --skip-no-dpp  # only generate DPP
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
WORKERS=4
SKIP_DPP=false
SKIP_NO_DPP=false

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="$2"; shift 2 ;;
        --skip-dpp)
            SKIP_DPP=true; shift ;;
        --skip-no-dpp)
            SKIP_NO_DPP=true; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Conda ─────────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_arm_manipulation

TASKS=(
    "ycb_plate_table_to_shelf"
    "ycb_plate_shelf_to_table"
    "mug_inversion"
    "flatPlate_box_to_box"
)

TRAJ_GEN="scripts/pipeline/trajectory_generation.py"

echo "============================================================"
echo "  Paper Step 1: Trajectory Generation"
echo "  Tasks:   ${TASKS[*]}"
echo "  Workers: $WORKERS"
echo "  Time:    $(date)"
echo "============================================================"

FAILED=0

# ── Helper: clean robot trajectories (keep object_trajectory.npz) ─────────
clean_robot_trajs() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        echo "  Cleaning robot trajectories in $dir ..."
        find "$dir" -name 'robot*_trajectory_*.npz' -delete
        # Also remove old assignment configs
        rm -rf "$dir/configs"
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# A) Generate trajectories WITH DPP (default — use_dpp=true in PlanningConfig)
# ══════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DPP" == false ]]; then
    echo ""
    echo "================================================================"
    echo "  A) Generating trajectories WITH DPP (grasp diversity)"
    echo "================================================================"

    for task in "${TASKS[@]}"; do
        echo ""
        echo "── $task (DPP) ──────────────────────────────────────────"
        DATA_DIR="data/$task"
        clean_robot_trajs "$DATA_DIR"

        python "$TRAJ_GEN" \
            --experiment "experiments/${task}.json" \
            --save-traj \
            --reuse-object-traj \
            --no-meshcat \
        && echo "  ✓ $task (DPP) done" \
        || { echo "  ✗ $task (DPP) FAILED"; FAILED=$((FAILED+1)); }
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# B) Generate trajectories WITHOUT DPP (for "No Diversity" ablation)
#    We create a patched experiment JSON with use_dpp=false and
#    output_dir pointing to data/<task>_no_dpp/.
# ══════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_NO_DPP" == false ]]; then
    echo ""
    echo "================================================================"
    echo "  B) Generating trajectories WITHOUT DPP (No Diversity ablation)"
    echo "================================================================"

    TMPDIR_PATCH=$(mktemp -d)
    trap "rm -rf $TMPDIR_PATCH" EXIT

    for task in "${TASKS[@]}"; do
        echo ""
        echo "── $task (no-DPP) ───────────────────────────────────────"

        NO_DPP_DIR="data/${task}_no_dpp"
        mkdir -p "$NO_DPP_DIR"

        # Copy object trajectory from DPP dir so both share the same object path
        OBJ_TRAJ="data/${task}/object_trajectory.npz"
        if [[ -f "$OBJ_TRAJ" ]]; then
            cp "$OBJ_TRAJ" "$NO_DPP_DIR/object_trajectory.npz"
        else
            echo "  WARNING: No object trajectory at $OBJ_TRAJ — will be generated"
        fi

        clean_robot_trajs "$NO_DPP_DIR"

        # Build patched JSON: set use_dpp=false, output_dir → no_dpp dir
        PATCHED_JSON="$TMPDIR_PATCH/${task}_no_dpp.json"
        python3 -c "
import json, sys
with open('experiments/${task}.json') as f:
    cfg = json.load(f)
cfg.setdefault('planning', {})['use_dpp'] = False
cfg.setdefault('output', {})['output_dir'] = '$NO_DPP_DIR'
cfg.setdefault('output', {})['save_trajectories'] = True
cfg.setdefault('visualization', {})['enable_meshcat'] = False
with open('$PATCHED_JSON', 'w') as f:
    json.dump(cfg, f, indent=2)
"

        python "$TRAJ_GEN" \
            --experiment "$PATCHED_JSON" \
            --save-traj \
            --reuse-object-traj \
            --no-meshcat \
        && echo "  ✓ $task (no-DPP) done" \
        || { echo "  ✗ $task (no-DPP) FAILED"; FAILED=$((FAILED+1)); }
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Trajectory Generation Complete — $(date)"
echo "============================================================"
for task in "${TASKS[@]}"; do
    DPP_COUNT=$(ls "data/$task"/robot*_trajectory_*.npz 2>/dev/null | wc -l)
    NODPP_COUNT=$(ls "data/${task}_no_dpp"/robot*_trajectory_*.npz 2>/dev/null | wc -l)
    echo "  $task:  DPP=$DPP_COUNT trajs   no-DPP=$NODPP_COUNT trajs"
done

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "  WARNING: $FAILED task(s) failed!"
    exit 1
else
    echo "  All tasks succeeded."
fi
