#!/usr/bin/env bash
###############################################################################
# run_overnight.sh — Comprehensive Overnight Pipeline
#
# This script runs the FULL pipeline unattended:
#   Phase 0 — Back up existing trajectory data
#   Phase 1 — Regenerate trajectories for 3 failing experiments
#             (mug_inversion, flatPlate_box_to_box, ycb_plate_table_to_shelf)
#   Phase 2 — Run planning for all 4 paper experiments × 5 ablation configs
#   Phase 3 — Run execution for configs with finite planning cost
#   Phase 4 — Collect & print paper metrics
#
# Everything runs headlessly (no DISPLAY, no meshcat, no rendering).
# Logs are saved per-phase in results/overnight_<TIMESTAMP>/.
#
# Usage:
#   chmod +x scripts/run_overnight.sh
#   nohup bash scripts/run_overnight.sh &> /tmp/overnight_runner.log &
#
#   # Or in a tmux/screen session:
#   bash scripts/run_overnight.sh
###############################################################################
set -uo pipefail   # don't use -e: we want to continue on partial failures

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/sukhvansh/Dual-Arm-Franka-Control-MTP"
CONDA_ENV="dual_arm_manipulation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/results/overnight_${TIMESTAMP}"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"
DATA_DIR="${PROJECT_ROOT}/data"

# Experiments that need trajectory regeneration (data quality issues)
REGEN_EXPERIMENTS=("mug_inversion" "flatPlate_box_to_box" "ycb_plate_table_to_shelf")

# ALL 4 paper experiments for planning + execution
PAPER_EXPERIMENTS=("ycb_plate_table_to_shelf" "flatPlate_shelf_to_box" "mug_inversion" "flatPlate_box_to_box")

# Grasp generation parameters for regeneration
# Increase raw candidates for better entropy pool; keep entropy count moderate
# to avoid blowing up planning time (handover evals scale O(n_r0 × n_r1))
REGEN_NUM_GRASPS=400     # up from 200 — larger candidate pool
REGEN_NUM_ENTROPY=25     # up from 20  — slightly more trajectory diversity

# Execution timeout per config (seconds)
EXEC_TIMEOUT=600

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────

log_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    printf "║  %-66s  ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

log_step() {
    echo "[$(date '+%H:%M:%S')] $1"
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    local diff=$(( now - start ))
    local mins=$(( diff / 60 ))
    local secs=$(( diff % 60 ))
    echo "${mins}m ${secs}s"
}

# ─── SETUP ───────────────────────────────────────────────────────────────────

log_banner "OVERNIGHT PIPELINE — ${TIMESTAMP}"
log_step "Project root: ${PROJECT_ROOT}"
log_step "Log directory: ${LOG_DIR}"
mkdir -p "${LOG_DIR}"

# Activate conda
log_step "Activating conda environment: ${CONDA_ENV}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}" || { echo "FATAL: cannot activate conda env"; exit 1; }
log_step "Python: $(which python) ($(python --version 2>&1))"

# Set headless environment
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

PIPELINE_START=$(date +%s)

###############################################################################
# PHASE 0 — Back up existing trajectory data
###############################################################################
log_banner "PHASE 0: BACKING UP TRAJECTORY DATA"

for exp in "${REGEN_EXPERIMENTS[@]}"; do
    src="${DATA_DIR}/${exp}"
    if [[ -d "${src}" ]]; then
        dst="${DATA_DIR}/${exp}_backup_${TIMESTAMP}"
        log_step "Backing up ${exp} → $(basename "${dst}")"
        mv "${src}" "${dst}"
    else
        log_step "No existing data for ${exp} — nothing to back up"
    fi
done

###############################################################################
# PHASE 1 — Regenerate trajectories (3 failing experiments)
###############################################################################
log_banner "PHASE 1: TRAJECTORY REGENERATION"

# Create patched experiment configs with increased grasp parameters
TMPDIR_CONFIGS=$(mktemp -d -t overnight_configs_XXXX)
log_step "Creating patched configs in ${TMPDIR_CONFIGS}"

python3 << PYEOF
import json, os

exps = ['mug_inversion', 'flatPlate_box_to_box', 'ycb_plate_table_to_shelf']
exp_dir = '${EXPERIMENTS_DIR}'
tmp_dir = '${TMPDIR_CONFIGS}'
num_grasps = ${REGEN_NUM_GRASPS}
num_entropy = ${REGEN_NUM_ENTROPY}

for name in exps:
    src = os.path.join(exp_dir, name + '.json')
    with open(src) as f:
        cfg = json.load(f)

    # Increase grasp diversity
    cfg['planning']['num_grasps'] = num_grasps
    cfg['planning']['num_entropy_grasps'] = num_entropy

    # Disable all visualization
    vis = cfg.setdefault('visualization', {})
    vis['visualize_mesh'] = False
    vis['enable_meshcat'] = False

    # Force-enable trajectory saving with explicit output dir
    out = cfg.setdefault('output', {})
    out['save_trajectories'] = True
    out['output_dir'] = os.path.join('${DATA_DIR}', name)

    dst = os.path.join(tmp_dir, name + '.json')
    with open(dst, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('  Patched: {} (grasps={}, entropy={})'.format(name, num_grasps, num_entropy))
PYEOF

PHASE1_START=$(date +%s)
REGEN_SUCCESS=0
REGEN_FAIL=0

for exp in "${REGEN_EXPERIMENTS[@]}"; do
    log_step "━━━ Regenerating: ${exp} ━━━"
    EXP_START=$(date +%s)
    LOGFILE="${LOG_DIR}/phase1_trajgen_${exp}.log"

    patched_cfg="${TMPDIR_CONFIGS}/${exp}.json"
    if [[ ! -f "${patched_cfg}" ]]; then
        log_step "ERROR: patched config not found: ${patched_cfg}"
        REGEN_FAIL=$((REGEN_FAIL + 1))
        continue
    fi

    python scripts/planning/trajectory_generation.py \
        --experiment "${patched_cfg}" \
        --save-traj \
        --no-meshcat \
        2>&1 | tee "${LOGFILE}"

    rc=${PIPESTATUS[0]}
    elapsed=$(elapsed_since ${EXP_START})

    if [[ ${rc} -eq 0 ]]; then
        # Verify data was actually created
        n_r0=$(find "${DATA_DIR}/${exp}" -name 'robot0_trajectory_*.npz' 2>/dev/null | wc -l)
        n_r1=$(find "${DATA_DIR}/${exp}" -name 'robot1_trajectory_*.npz' 2>/dev/null | wc -l)
        log_step "OK: ${exp} — ${n_r0} R0 + ${n_r1} R1 trajectories (${elapsed})"
        REGEN_SUCCESS=$((REGEN_SUCCESS + 1))
    else
        log_step "FAIL: ${exp} — exit code ${rc} (${elapsed})"
        REGEN_FAIL=$((REGEN_FAIL + 1))
    fi
done

# Clean up temp configs
rm -rf "${TMPDIR_CONFIGS}"

PHASE1_ELAPSED=$(elapsed_since ${PHASE1_START})
log_step "Phase 1 complete: ${REGEN_SUCCESS} OK, ${REGEN_FAIL} FAIL (${PHASE1_ELAPSED})"

# Check if any data exists for ALL paper experiments before proceeding
MISSING_DATA=()
for exp in "${PAPER_EXPERIMENTS[@]}"; do
    if [[ ! -d "${DATA_DIR}/${exp}" ]]; then
        MISSING_DATA+=("${exp}")
    fi
done

if [[ ${#MISSING_DATA[@]} -gt 0 ]]; then
    log_step "WARNING: Missing data for: ${MISSING_DATA[*]}"
    log_step "Planning will skip these experiments"
fi

###############################################################################
# PHASE 2 — Planning (all 4 experiments × 5 paper configs)
###############################################################################
log_banner "PHASE 2: ASSIGNMENT PLANNING"

PHASE2_START=$(date +%s)
LOGFILE="${LOG_DIR}/phase2_planning.log"

log_step "Running planning for ${#PAPER_EXPERIMENTS[@]} experiments × 5 configs..."
log_step "Log: ${LOGFILE}"

python scripts/run_paper_evaluation.py \
    --plan-only \
    --debug \
    2>&1 | tee "${LOGFILE}"

PHASE2_RC=${PIPESTATUS[0]}
PHASE2_ELAPSED=$(elapsed_since ${PHASE2_START})
log_step "Phase 2 complete (exit code ${PHASE2_RC}, ${PHASE2_ELAPSED})"

###############################################################################
# PHASE 3 — Execution (headless simulation for finite-cost configs)
###############################################################################
log_banner "PHASE 3: HEADLESS EXECUTION"

PHASE3_START=$(date +%s)
LOGFILE="${LOG_DIR}/phase3_execution.log"

log_step "Running execution with timeout=${EXEC_TIMEOUT}s per config..."
log_step "Log: ${LOGFILE}"

python scripts/run_paper_evaluation.py \
    --exec-only \
    --exec-timeout ${EXEC_TIMEOUT} \
    2>&1 | tee "${LOGFILE}"

PHASE3_RC=${PIPESTATUS[0]}
PHASE3_ELAPSED=$(elapsed_since ${PHASE3_START})
log_step "Phase 3 complete (exit code ${PHASE3_RC}, ${PHASE3_ELAPSED})"

###############################################################################
# PHASE 4 — Metrics summary
###############################################################################
log_banner "PHASE 4: PAPER METRICS"

LOGFILE="${LOG_DIR}/phase4_metrics.log"

python scripts/run_paper_evaluation.py \
    --metrics-only \
    2>&1 | tee "${LOGFILE}"

###############################################################################
# FINAL SUMMARY
###############################################################################
PIPELINE_ELAPSED=$(elapsed_since ${PIPELINE_START})

log_banner "PIPELINE COMPLETE — Total: ${PIPELINE_ELAPSED}"

cat << EOF

═══════════════════════════════════════════════════════════════════════════════
  OVERNIGHT PIPELINE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

  Start time:      ${TIMESTAMP}
  Total duration:  ${PIPELINE_ELAPSED}
  Log directory:   ${LOG_DIR}

  Phase 1 (Traj Gen):  ${REGEN_SUCCESS}/${#REGEN_EXPERIMENTS[@]} OK  (${PHASE1_ELAPSED})
  Phase 2 (Planning):  exit=${PHASE2_RC}  (${PHASE2_ELAPSED})
  Phase 3 (Execution): exit=${PHASE3_RC}  (${PHASE3_ELAPSED})

  Log files:
    - phase1_trajgen_*.log   — per-experiment trajectory generation
    - phase2_planning.log    — assignment planning
    - phase3_execution.log   — headless execution
    - phase4_metrics.log     — paper metrics tables
    - paper_evaluation.log   — detailed planning/execution log (in results/)

  Backed-up data:
$(for exp in "${REGEN_EXPERIMENTS[@]}"; do
    bk="${DATA_DIR}/${exp}_backup_${TIMESTAMP}"
    if [[ -d "${bk}" ]]; then
        echo "    - ${bk}"
    fi
done)

  New trajectory data:
$(for exp in "${PAPER_EXPERIMENTS[@]}"; do
    d="${DATA_DIR}/${exp}"
    if [[ -d "${d}" ]]; then
        n0=$(find "${d}" -name 'robot0_trajectory_*.npz' 2>/dev/null | wc -l)
        n1=$(find "${d}" -name 'robot1_trajectory_*.npz' 2>/dev/null | wc -l)
        echo "    - ${exp}: ${n0} R0 + ${n1} R1 trajectories"
    else
        echo "    - ${exp}: NO DATA"
    fi
done)

  Results: ${PROJECT_ROOT}/results/
═══════════════════════════════════════════════════════════════════════════════

EOF

echo "Done. Check the log files above for detailed results."
