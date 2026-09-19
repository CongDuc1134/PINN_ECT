#!/usr/bin/env bash
# ==============================================================================
# Linux Bash Runner for Domain Adaptation on Real ECT Data (5kHz)
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${PYTHON_EXEC:-}" ]]; then
    if [[ -x "/home/dat.lt19010205/.conda/envs/CongDuc/bin/python" ]]; then
        PYTHON="/home/dat.lt19010205/.conda/envs/CongDuc/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON="$(command -v python3)"
    else
        PYTHON="python"
    fi
else
    PYTHON="$PYTHON_EXEC"
fi

echo "================================================================================"
echo "[START] Domain Adaptation Benchmark with 10-Fold LODO on 5kHz Real ECT Data"
echo "Python Executable: $PYTHON"
echo "================================================================================"

cd "$PROJECT_ROOT"
"$PYTHON" domain_adaptation/run_all.py "$@"

echo "================================================================================"
echo "[DONE] Domain adaptation benchmark finished!"
echo "================================================================================"
