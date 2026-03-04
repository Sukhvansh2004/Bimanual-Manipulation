#!/bin/bash
# ============================================================================
# Paper Evaluation Shell Wrapper
# ============================================================================
# This script activates the conda environment and runs the evaluation.
#
# Usage:
#   ./scripts/run_paper_eval.sh                     # Full evaluation (plan + execute)
#   ./scripts/run_paper_eval.sh --plan-only         # Planning only
#   ./scripts/run_paper_eval.sh --exec-only         # Execution only (needs existing assignments)
#   ./scripts/run_paper_eval.sh --metrics-only      # Print metrics from existing results
#   ./scripts/run_paper_eval.sh --debug             # Enable debug logging
#   ./scripts/run_paper_eval.sh --only flatPlate_shelf_to_box  # Single experiment
#
# Logs are saved to:
#   results/paper_evaluation.log       — Python logging output
#   results/paper_eval_stdout.log      — Full stdout+stderr
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_arm_manipulation

echo "============================================================"
echo "  Paper Evaluation Runner"
echo "  Project: $PROJECT_ROOT"
echo "  Python:  $(which python)"
echo "  Args:    $@"
echo "  Time:    $(date)"
echo "============================================================"

# Create results directory if needed
mkdir -p results

# Run the evaluation, tee output to a log file
python scripts/evaluation/run_paper_evaluation.py "$@" 2>&1 | tee results/paper_eval_stdout.log

echo ""
echo "============================================================"
echo "  Evaluation complete at $(date)"
echo "  Logs: results/paper_evaluation.log"
echo "         results/paper_eval_stdout.log"
echo "============================================================"
