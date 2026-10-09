#!/usr/bin/env bash
# ==============================================================================
# Bash Script: run_all_experiments.sh
# Môi trường: conda activate konabi
# Mục tiêu: Chạy đối chứng 2 mô hình cốt lõi: CNN-PINN và CNN-NoPINN
# Chạy tuần tự (sequential) với 1 bộ thông số cố định, tối ưu cho máy cá nhân
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# 1. KÍCH HOẠT MÔI TRƯỜNG CONDA 'konabi'
# ------------------------------------------------------------------------------
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate konabi 2>/dev/null || true
fi

# Tự động tìm đường dẫn Python của môi trường konabi trên Windows / Linux
if [[ -z "${PYTHON_EXEC:-}" ]]; then
    if [[ -x "C:/ProgramData/miniconda3/envs/konabi/python.exe" ]]; then
        PYTHON="C:/ProgramData/miniconda3/envs/konabi/python.exe"
    elif [[ -x "/c/ProgramData/miniconda3/envs/konabi/python.exe" ]]; then
        PYTHON="/c/ProgramData/miniconda3/envs/konabi/python.exe"
    elif [[ -x "$CONDA_PREFIX/python.exe" ]]; then
        PYTHON="$CONDA_PREFIX/python.exe"
    elif [[ -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    elif command -v python &>/dev/null; then
        PYTHON="$(command -v python)"
    elif command -v python3 &>/dev/null; then
        PYTHON="$(command -v python3)"
    else
        PYTHON="python"
    fi
else
    PYTHON="$PYTHON_EXEC"
fi

# ------------------------------------------------------------------------------
# 2. THIẾT LẬP THƯ MỤC TẠM VÀ LOG
# ------------------------------------------------------------------------------
export TMPDIR="${TMPDIR:-$SCRIPT_DIR/tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || true

LOG_FILE="$SCRIPT_DIR/experiment_pinn_vs_nopinn.log"

CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
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
log_ok()     { log "${GREEN}OK${NC}"    "${GREEN}$*${NC}"; }
log_err()    { log "${RED}ERROR${NC}"  "${RED}$*${NC}"; }

# ------------------------------------------------------------------------------
# 3. THÔNG SỐ CỐ ĐỊNH CHO 1 TRƯỜNG HỢP ĐỐI CHỨNG (PINN vs No-PINN)
# ------------------------------------------------------------------------------
export TRAIN_PERCENT="${TRAIN_PERCENT:-10}"            # Tỷ lệ dữ liệu train: 10%
export BATCH_SIZE="${BATCH_SIZE:-8}"                   # Kích thước batch: 8
export EPOCHS="${EPOCHS:-300}"                         # Tổng số epoch: 300
export ALPHA_INIT="${ALPHA_INIT:-1.0}"                 # Trọng số khởi tạo PINN loss
export PINN_ACTIVATION_EPOCH="${PINN_ACTIVATION_EPOCH:-100}" # Kích hoạt PINN từ epoch 100
export SEED="${SEED:-42}"                              # Random seed: 42
export AUTO_RESUME="${AUTO_RESUME:-1}"                 # Tự động resume nếu có checkpoint
export FORCE_RETRAIN="${FORCE_RETRAIN:-0}"             # Bỏ qua nếu đã hoàn thành

# Cấu hình luồng CPU tối ưu cho máy tính
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"

# Chế độ chạy: all (cả 2), pinn (chỉ PINN), nopinn (chỉ No-PINN)
MODE="${1:-all}"

echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"
log_header "BẮT ĐẦU CHẠY ĐỐI CHỨNG: CNN-PINN & CNN-NoPINN"
log "INFO" "Thư mục dự án    : $SCRIPT_DIR"
log "INFO" "Python Executable: $PYTHON"
log "INFO" "Môi trường Conda : konabi"
log "INFO" "Chế độ chạy      : $MODE"
log "INFO" "Tập dữ liệu      : TRAIN_PERCENT=${TRAIN_PERCENT}%"
log "INFO" "Batch size       : ${BATCH_SIZE}"
log "INFO" "Số epochs        : ${EPOCHS}"
log "INFO" "Alpha PINN       : ${ALPHA_INIT}"
log "INFO" "Warmup Epoch     : ${PINN_ACTIVATION_EPOCH}"
log "INFO" "Random Seed      : ${SEED}"
log "INFO" "File Log         : $LOG_FILE"
echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"

SCRIPT_START_TIME=$(date +%s)
cd "$SCRIPT_DIR/cnn"

EXIT_PINN=0
EXIT_NOPINN=0

# ------------------------------------------------------------------------------
# 4. CHẠY 1: CNN-PINN (MÔ HÌNH ĐỀ XUẤT CÓ VẬT LÝ DẪN ĐƯỜNG)
# ------------------------------------------------------------------------------
if [[ "$MODE" == "all" || "$MODE" == "pinn" ]]; then
    log_header "[1/2] Đang chạy mô hình CNN-PINN (main_percent_new.py)..."
    "$PYTHON" main_percent_new.py 2>&1 | tee -a "$LOG_FILE"
    EXIT_PINN=${PIPESTATUS[0]}
    if [[ "$EXIT_PINN" -eq 0 ]]; then
        log_ok "[1/2] Hoàn thành huấn luyện CNN-PINN!"
    else
        log_err "[1/2] Lỗi khi chạy CNN-PINN (exit code: $EXIT_PINN)"
    fi
fi

# ------------------------------------------------------------------------------
# 5. CHẠY 2: CNN-NoPINN (MÔ HÌNH ĐỐI CHỨNG THUẦN DỮ LIỆU)
# ------------------------------------------------------------------------------
if [[ "$MODE" == "all" || "$MODE" == "nopinn" ]]; then
    log_header "[2/2] Đang chạy mô hình CNN-NoPINN (main_percent_no_pinn.py)..."
    "$PYTHON" main_percent_no_pinn.py 2>&1 | tee -a "$LOG_FILE"
    EXIT_NOPINN=${PIPESTATUS[0]}
    if [[ "$EXIT_NOPINN" -eq 0 ]]; then
        log_ok "[2/2] Hoàn thành huấn luyện CNN-NoPINN!"
    else
        log_err "[2/2] Lỗi khi chạy CNN-NoPINN (exit code: $EXIT_NOPINN)"
    fi
fi

# ------------------------------------------------------------------------------
# 6. BÁO CÁO KẾT THÚC
# ------------------------------------------------------------------------------
SCRIPT_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

OVERALL=$((EXIT_PINN + EXIT_NOPINN))

log_header "========================================================"
if [[ "$OVERALL" -eq 0 ]]; then
    log_ok "THÀNH CÔNG: ĐÃ HOÀN TẤT ĐỐI CHỨNG CẢ 2 MÔ HÌNH (PINN & No-PINN)!"
else
    log_err "CẢNH BÁO: Có lỗi xảy ra trong quá trình chạy. Vui lòng xem $LOG_FILE"
fi
log_header "Tổng thời gian thực thi: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_header "========================================================"

exit "$OVERALL"
