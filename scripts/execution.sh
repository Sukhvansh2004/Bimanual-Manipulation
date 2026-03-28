#!/bin/bash
# ============================================================================
# Run execution for experiments.
#
# Usage:
#   ./scripts/execution.sh              # all experiments
#   ./scripts/execution.sh --paper      # only paper experiments/configs
#   ./scripts/execution.sh --only exp1  # pinpoint experiments
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=("--execute-only") # Default to execute only
while [[ $# -gt 0 ]]; do
    case "$1" in
        --paper) ARGS+=("--paper"); shift ;;
        --only) ARGS+=("--only"); shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do ARGS+=("$1"); shift; done ;;
        --workers) ARGS+=("--workers" "$2"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

python scripts/evaluation/run_all.py "${ARGS[@]}"
