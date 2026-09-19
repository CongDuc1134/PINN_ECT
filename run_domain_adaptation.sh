#!/usr/bin/env bash
# ==============================================================================
# Linux / Git-Bash Automation Script: run_domain_adaptation.sh
# Runs the full Domain Adaptation benchmark on 5kHz Real ECT Data
# Features:
#   - 10-Fold Leave-One-Defect-Out (LODO) Cross-Validation
#   - Global Pooled Out-of-Fold (OOF) Metrics Aggregation
#   - Evaluates: Zero-Shot, Few-Shot PEFT, MMD Alignment, Physics-TTA
#   - Automatic checkpoint & real data discovery (no hardcoded paths required)
#   - Real-time timestamped colored logging to console & log file
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# 1. PYTHON EXECUTABLE RESOLUTION
# ------------------------------------------------------------------------------
if [[ -z "${PYTHON_EXEC:-}" ]]; then
    if command -v conda &>/dev/null && conda info --envs 2>/dev/null | grep -q "konabi"; then
        PYTHON="conda run --no-capture-output -n konabi python"
    elif [[ -x "/home/dat.lt19010205/.conda/envs/CongDuc/bin/python" ]]; then
        PYTHON="/home/dat.lt19010205/.conda/envs/CongDuc/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON="$(command -v python3)"
    else
        PYTHON="python"
    fi
else
    PYTHON="$PYTHON_EXEC"
fi

# ------------------------------------------------------------------------------
# 2. LOGGING & DISPLAY SETUP
# ------------------------------------------------------------------------------
LOG_FILE="$SCRIPT_DIR/domain_adaptation/results/run_domain_adaptation.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

log() {
    local level="$1"
    shift
    local timestamp
    timestamp="$(date +'%Y-%m-%d %H:%M:%S')"
    local msg="[$timestamp] [$level] $*"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_header() { log "${CYAN}START${NC}" "${CYAN}$*${NC}"; }
log_step()   { log "${YELLOW}STEP${NC}"  "${YELLOW}$*${NC}"; }
log_ok()     { log "${GREEN}OK${NC}"    "${GREEN}$*${NC}"; }
log_err()    { log "${RED}ERROR${NC}"   "${RED}$*${NC}"; }

SCRIPT_START_TIME=$(date +%s)

echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"
log_header "================================================================================"
log_header "STARTING AUTOMATED DOMAIN ADAPTATION PIPELINE (10-FOLD LODO OOF)"
log_header "================================================================================"
log "INFO" "Script Directory : $SCRIPT_DIR"
log "INFO" "Python Executable: $PYTHON"
log "INFO" "Log File         : $LOG_FILE"
log "INFO" "Arguments Passed : $*"
echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. RUN DOMAIN ADAPTATION MASTER RUNNER
# ------------------------------------------------------------------------------
log_step "Launching domain_adaptation/run_all.py..."

cd "$SCRIPT_DIR"
$PYTHON domain_adaptation/run_all.py "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# ------------------------------------------------------------------------------
# 4. SUMMARY & ELAPSED TIME
# ------------------------------------------------------------------------------
SCRIPT_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo "" >> "$LOG_FILE"
log_header "================================================================================"
if [[ "$EXIT_CODE" -eq 0 ]]; then
    log_ok "DOMAIN ADAPTATION COMPLETED SUCCESSFULLY in ${MINUTES}m ${SECONDS}s!"
    log "INFO" "Results Summary Table: $SCRIPT_DIR/domain_adaptation/results/lodo_oof_master_summary.csv"
    log "INFO" "Detailed Predictions : $SCRIPT_DIR/domain_adaptation/results/lodo_oof_master_predictions.csv"
else
    log_err "DOMAIN ADAPTATION FAILED (Exit Code: $EXIT_CODE). Check log above: $LOG_FILE"
fi
log_header "================================================================================"

exit "$EXIT_CODE"
