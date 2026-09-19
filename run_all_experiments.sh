#!/usr/bin/env bash
# ==============================================================================
# Linux Bash Script: run_all_experiments.sh
# Run all experimental evaluation scripts sequentially folder-by-folder on Linux
# Features:
#   - Real-time timestamped logging to both stdout and experiments_run.log
#   - Automatic skip logic for already completed models/experiments
#   - Resumable training for partial runs
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

export TMPDIR="${TMPDIR:-/work/dat.lt19010205/Cong_Duc/check/tmp}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
export PARALLEL_RUNS="${PARALLEL_RUNS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"

mkdir -p "$TMPDIR" 2>/dev/null || true

LOG_FILE="$SCRIPT_DIR/experiments_run.log"

# Color Codes
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# Timestamped logging helper
log() {
    local level="$1"
    shift
    local timestamp
    timestamp="$(date +'%Y-%m-%d %H:%M:%S')"
    local msg="[$timestamp] [$level] $*"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_header()  { log "${CYAN}START${NC}" "${CYAN}$*${NC}"; }
log_step()    { log "${YELLOW}STEP${NC}" "${YELLOW}$*${NC}"; }
log_ok()      { log "${GREEN}OK${NC}" "${GREEN}$*${NC}"; }
log_skip()    { log "${BLUE}SKIP${NC}" "${BLUE}$*${NC}"; }
log_err()     { log "${RED}ERROR${NC}" "${RED}$*${NC}"; }

SCRIPT_START_TIME=$(date +%s)

echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"
log_header "STARTING EXPERIMENT EVALUATION SWEEP (new/)"
log "INFO" "Script Directory : $SCRIPT_DIR"
log "INFO" "Python Executable: $PYTHON"
log "INFO" "TMPDIR           : $TMPDIR"
log "INFO" "PARALLEL_RUNS    : $PARALLEL_RUNS"
log "INFO" "OMP/MKL THREADS  : $OMP_NUM_THREADS"
log "INFO" "SKIP_IF_EXISTS   : $SKIP_IF_EXISTS"
log "INFO" "AUTO_RESUME      : $AUTO_RESUME"
log "INFO" "FORCE_RETRAIN    : $FORCE_RETRAIN"
log "INFO" "Log File         : $LOG_FILE"
echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"

# Helper to check if folder/file outputs exist
is_step1_done() {
    [[ "$FORCE_RETRAIN" != "1" && "$SKIP_IF_EXISTS" == "1" ]] && \
    [[ -f "$SCRIPT_DIR/classical/Outputs_classical_baseline/classical_baseline_all_raw.csv" ]] && \
    [[ -f "$SCRIPT_DIR/classical/Outputs_classical_baseline/classical_baseline_stability_5percent.csv" ]]
}

# ==============================================================================
# PARALLEL EXECUTION: Launch all 5 jobs simultaneously as background jobs
# Each job writes to its own log file to avoid output interleaving
# ==============================================================================
log_header "LAUNCHING ALL 5 JOBS IN PARALLEL (MULTI-THREADED & CONCURRENT)"

LOG_CLASSICAL="$SCRIPT_DIR/log_classical.log"
LOG_XIONG="$SCRIPT_DIR/log_mlp_xiong.log"
LOG_MLP="$SCRIPT_DIR/log_mlp_all.log"
LOG_CNN="$SCRIPT_DIR/log_cnn.log"
LOG_SINGLE_TASK="$SCRIPT_DIR/log_cnn_single_task.log"

# --- JOB 1: Classical baseline ---
if is_step1_done; then
    log_skip "[JOB 1] classical_baseline.py already completed. Skipping!"
    PID_CLASSICAL=""
else
    log "RUN" "[JOB 1] Launching classical/classical_baseline.py in background..."
    (cd "$SCRIPT_DIR/classical" && "$PYTHON" classical_baseline.py > "$LOG_CLASSICAL" 2>&1) &
    PID_CLASSICAL=$!
fi

# --- JOB 2: MLP Xiong 2023 baseline ---
log "RUN" "[JOB 2] Launching mlp/run_eval_xiong.py in background..."
(cd "$SCRIPT_DIR/mlp" && "$PYTHON" run_eval_xiong.py > "$LOG_XIONG" 2>&1) &
PID_XIONG=$!

# --- JOB 3: MLP-PINN evaluation ---
log "RUN" "[JOB 3] Launching mlp/run_eval_all.py in background..."
(cd "$SCRIPT_DIR/mlp" && "$PYTHON" run_eval_all.py > "$LOG_MLP" 2>&1) &
PID_MLP=$!

# --- JOB 4: CNN-PINN multi-task evaluation ---
log "RUN" "[JOB 4] Launching cnn/run_eval_all.py in background..."
(cd "$SCRIPT_DIR/cnn" && "$PYTHON" run_eval_all.py > "$LOG_CNN" 2>&1) &
PID_CNN=$!

# --- JOB 5: CNN Single-Task ablation (4 models run concurrently in parallel) ---
log "RUN" "[JOB 5] Launching 4 CNN Single-Task scripts simultaneously in parallel..."
(
    cd "$SCRIPT_DIR/cnn"
    LOG_ST_CLF_NOPINN="$SCRIPT_DIR/log_st_clf_nopinn.log"
    LOG_ST_CLF_PINN="$SCRIPT_DIR/log_st_clf_pinn.log"
    LOG_ST_REG_NOPINN="$SCRIPT_DIR/log_st_reg_nopinn.log"
    LOG_ST_REG_PINN="$SCRIPT_DIR/log_st_reg_pinn.log"

    "$PYTHON" cnn_single_task_classification_nopinn.py > "$LOG_ST_CLF_NOPINN" 2>&1 &
    PID_ST1=$!
    "$PYTHON" cnn_single_task_classification_pinn.py   > "$LOG_ST_CLF_PINN"   2>&1 &
    PID_ST2=$!
    "$PYTHON" cnn_single_task_regression_nopinn.py     > "$LOG_ST_REG_NOPINN" 2>&1 &
    PID_ST3=$!
    "$PYTHON" cnn_single_task_regression_pinn.py       > "$LOG_ST_REG_PINN"   2>&1 &
    PID_ST4=$!

    wait "$PID_ST1"
    wait "$PID_ST2"
    wait "$PID_ST3"
    wait "$PID_ST4"

    cat "$LOG_ST_CLF_NOPINN" "$LOG_ST_CLF_PINN" "$LOG_ST_REG_NOPINN" "$LOG_ST_REG_PINN" > "$LOG_SINGLE_TASK" 2>/dev/null || true
) &
PID_SINGLE_TASK=$!

log_header "All 5 jobs launched. Waiting for completion..."
log "INFO" "JOB 1 PID (classical)        : ${PID_CLASSICAL:-SKIPPED}"
log "INFO" "JOB 2 PID (mlp xiong)        : $PID_XIONG"
log "INFO" "JOB 3 PID (mlp all)          : $PID_MLP"
log "INFO" "JOB 4 PID (cnn multi-task)   : $PID_CNN"
log "INFO" "JOB 5 PID (cnn single-task)  : $PID_SINGLE_TASK"

# ==============================================================================
# WAIT FOR ALL JOBS & COLLECT EXIT CODES
# ==============================================================================
EXIT_CLASSICAL=0
EXIT_XIONG=0
EXIT_MLP=0
EXIT_CNN=0
EXIT_SINGLE_TASK=0

if [[ -n "$PID_CLASSICAL" ]]; then
    wait "$PID_CLASSICAL"; EXIT_CLASSICAL=$?
fi
wait "$PID_XIONG";       EXIT_XIONG=$?
wait "$PID_MLP";         EXIT_MLP=$?
wait "$PID_CNN";         EXIT_CNN=$?
wait "$PID_SINGLE_TASK"; EXIT_SINGLE_TASK=$?

# Append individual logs to the main log file
echo "" >> "$LOG_FILE"
log "INFO" "=== LOG: classical_baseline.py ===" && cat "$LOG_CLASSICAL" >> "$LOG_FILE" 2>/dev/null || true
log "INFO" "=== LOG: mlp/run_eval_xiong.py ===" && cat "$LOG_XIONG" >> "$LOG_FILE" 2>/dev/null || true
log "INFO" "=== LOG: mlp/run_eval_all.py ===" && cat "$LOG_MLP" >> "$LOG_FILE" 2>/dev/null || true
log "INFO" "=== LOG: cnn/run_eval_all.py ===" && cat "$LOG_CNN" >> "$LOG_FILE" 2>/dev/null || true
log "INFO" "=== LOG: cnn/single_task_ablation ===" && cat "$LOG_SINGLE_TASK" >> "$LOG_FILE" 2>/dev/null || true

# ==============================================================================
# REPORT RESULTS PER JOB
# ==============================================================================
log_header "========================================================"
log_header "PARALLEL EXECUTION COMPLETE — RESULTS PER JOB"
log_header "========================================================"

report_job() {
    local name="$1" code="$2" logf="$3"
    if [[ "$code" -eq 0 ]]; then
        log_ok  "  [$name] SUCCESS (exit 0) — log: $logf"
    else
        log_err "  [$name] FAILED  (exit $code) — see: $logf"
    fi
}

report_job "JOB 1 classical_baseline.py" "$EXIT_CLASSICAL"   "$LOG_CLASSICAL"
report_job "JOB 2 mlp/run_eval_xiong.py" "$EXIT_XIONG"       "$LOG_XIONG"
report_job "JOB 3 mlp/run_eval_all.py"   "$EXIT_MLP"         "$LOG_MLP"
report_job "JOB 4 cnn/run_eval_all.py"   "$EXIT_CNN"         "$LOG_CNN"
report_job "JOB 5 cnn/single_task_all"   "$EXIT_SINGLE_TASK" "$LOG_SINGLE_TASK"

# Overall exit status
OVERALL=$((EXIT_CLASSICAL + EXIT_XIONG + EXIT_MLP + EXIT_CNN + EXIT_SINGLE_TASK))

# ------------------------------------------------------------------------------
# SUMMARY & ELAPSED TIME
# ------------------------------------------------------------------------------
SCRIPT_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

log_header "========================================================"
if [[ "$OVERALL" -eq 0 ]]; then
    log_header "SUCCESS: ALL 5 JOBS COMPLETED SUCCESSFULLY!"
else
    log_header "WARNING: ONE OR MORE JOBS FAILED. Check logs above."
fi
log_header "Total Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_header "========================================================"

exit "$OVERALL"
