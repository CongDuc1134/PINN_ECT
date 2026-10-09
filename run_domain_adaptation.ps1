# ==============================================================================
# run_domain_adaptation.ps1
# POWERSHELL MASTER RUNNER: DOMAIN ADAPTATION & LEARNING RATE STUDY
#
# Cách sử dụng:
#   .\run_domain_adaptation.ps1                (Menu tương tác chọn tất cả chế độ)
#   .\run_domain_adaptation.ps1 lr             (Khảo sát 3 mức LR: 1e-3, 5e-4, 1e-4 so sánh PINN vs NoPINN)
#   .\run_domain_adaptation.ps1 lr -Epochs 500 -Lrs "1e-3,5e-4,1e-4"
#   .\run_domain_adaptation.ps1 p1             (Chạy Protocol 1: Scan 1 vs Scan 2)
#   .\run_domain_adaptation.ps1 p2             (Chạy Protocol 2: 10-Fold Defect Cross Validation)
#   .\run_domain_adaptation.ps1 all            (Chạy cả Protocol 1 & 2 cho toàn bộ mô hình)
#   .\run_domain_adaptation.ps1 cnn            (Chạy riêng cho mô hình CNN Proposed vs NoPINN)
#   .\run_domain_adaptation.ps1 test           (Quick smoke test 1 epoch)
#   .\run_domain_adaptation.ps1 pretrain       (Tiền huấn luyện CNN Synthetic PINN & NoPINN)
# ==============================================================================

param(
    [string]$Mode = "",
    [string]$Lrs = "1e-3,5e-4,1e-4",
    [int]$Epochs = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

# Thiết lập bảng mã UTF-8 cho PowerShell
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Xác định Python executable trong môi trường conda 'konabi'
$PythonExe = "C:\ProgramData\miniconda3\envs\konabi\python.exe"
if (-not (Test-Path $PythonExe)) {
    try {
        $condaBase = (conda info --base 2>$null).Trim()
        if ($condaBase -and (Test-Path "$condaBase\envs\konabi\python.exe")) {
            $PythonExe = "$condaBase\envs\konabi\python.exe"
        }
    } catch {}
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$IsInteractive = [string]::IsNullOrWhiteSpace($Mode)

function Show-Header {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "       ECT DOMAIN ADAPTATION & PINN BENCHMARK: MASTER RUNNER" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Python Executable : $PythonExe"
    Write-Host "Workspace Root    : $ScriptDir"
    Write-Host "Backbone Mode     : KHONG DONG BANG (Finetune toan bo mo hinh end-to-end)" -ForegroundColor Green
    Write-Host ""
    try {
        & $PythonExe -c "import torch; print(f'Hardware Status   : GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.cuda.is_available()}')" 2>$null
    } catch {}
    Write-Host "================================================================================" -ForegroundColor Cyan
}

if ($IsInteractive) {
    Show-Header
    Write-Host ""
    Write-Host "Hay chon che do chay benchmark (Mac dinh: Finetune toan bo mo hinh, khong freeze):" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  [1] Chay CA 2 GIAO THUC (Protocol 1 & 2 - Khong freeze backbone) [Khuyen nghi]"
    Write-Host "  [2] Chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test - Khong freeze backbone)"
    Write-Host "  [3] Chay GIAO THUC 2 (10-Fold Defect LODO - Khong freeze backbone)"
    Write-Host "  [4] Chay rieng cho mo hinh CNN (CNN_Proposed PINN vs CNN_NoPINN)"
    Write-Host "  [5] Chay Quick Smoke Test (1 epoch de kiem tra nhanh)"
    Write-Host "  [6] Khao sat 3 muc Learning Rate (1e-3, 5e-4, 1e-4) -> So sanh PINN vs NoPINN"
    Write-Host "  [7] Tien huan luyen CNN tren du lieu mo phong Synthetic (PINN & NoPINN)"
    Write-Host "  [0] Thoat"
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    
    $Choice = Read-Host "Nhap lua chon cua ban [0-7, mac dinh 1]"
    if ([string]::IsNullOrWhiteSpace($Choice)) {
        $Choice = "1"
    }

    switch ($Choice) {
        "1" { $Mode = "all" }
        "2" { $Mode = "p1" }
        "3" { $Mode = "p2" }
        "4" { $Mode = "cnn" }
        "5" { $Mode = "test" }
        "6" { $Mode = "lr" }
        "7" { $Mode = "pretrain" }
        "0" { 
            Write-Host "Thoat chuong trinh." -ForegroundColor Gray
            exit 0 
        }
        default {
            Write-Host "Lua chon khong hop le: '$Choice'. Mac dinh chuyen sang chay tat ca (all)." -ForegroundColor Yellow
            $Mode = "all"
        }
    }
}

Write-Host ""
Write-Host "Dang khoi chay che do: $Mode" -ForegroundColor Green

if ($Mode.ToLower() -in @("pretrain", "7")) {
    $PretrainScript = Join-Path $ScriptDir "tools\run_single_experiment.ps1"
    if (Test-Path $PretrainScript) {
        & $PretrainScript
        exit $LASTEXITCODE
    } else {
        Write-Host "[LOI] Khong tim thay script: $PretrainScript" -ForegroundColor Red
        exit 1
    }
}

$BenchmarkScript = "domain_adaptation\benchmark_evaluation_protocols.py"
$CmdArgs = @($BenchmarkScript, "--no_freeze_backbone")

switch ($Mode.ToLower()) {
    { $_ -in @("all", "1") } {
        Write-Host "[START] Dang khoi chay CA 2 GIAO THUC BENCHMARK..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "all")
        if ($Epochs -gt 0) { $CmdArgs += @("--epochs", $Epochs.ToString()) }
    }
    { $_ -in @("p1", "2") } {
        Write-Host "[START] Dang khoi chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "1")
        if ($Epochs -gt 0) { $CmdArgs += @("--epochs", $Epochs.ToString()) }
    }
    { $_ -in @("p2", "3") } {
        Write-Host "[START] Dang khoi chay GIAO THUC 2 (10-Fold Leave-One-Defect-Out)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "2")
        if ($Epochs -gt 0) { $CmdArgs += @("--epochs", $Epochs.ToString()) }
    }
    { $_ -in @("cnn", "4") } {
        Write-Host "[START] Dang khoi chay Benchmark rieng cho mo hinh CNN (Proposed vs NoPINN)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "all", "--models", "cnn_proposed,cnn_nopinn")
        if ($Epochs -gt 0) { $CmdArgs += @("--epochs", $Epochs.ToString()) }
    }
    { $_ -in @("test", "5") } {
        Write-Host "[START] Dang chay Quick Smoke Test (1 epoch)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "1", "--epochs", "1")
    }
    { $_ -in @("lr", "6") } {
        $targetEpochs = if ($Epochs -gt 0) { $Epochs } else { 500 }
        Write-Host "[START] Dang khoi chay Khao sat 3 muc Learning Rate ($Lrs, $targetEpochs epochs) so sanh PINN vs NoPINN..." -ForegroundColor Cyan
        $BenchmarkScript = "domain_adaptation\run_lr_experiments.py"
        $CmdArgs = @($BenchmarkScript, "--lrs", $Lrs, "--epochs", $targetEpochs.ToString(), "--protocol", "1", "--models", "cnn_proposed,cnn_nopinn", "--no_freeze_backbone")
    }
    default {
        Write-Host "[LOI] Che do '$Mode' khong hop le. Cac che do ho tro: all, p1, p2, cnn, test, lr, pretrain" -ForegroundColor Red
        exit 1
    }
}

if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $CmdArgs += $ExtraArgs
}

Write-Host "Command: $PythonExe $($CmdArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $PythonExe @CmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "[LOI] Tien trinh gap loi voi Exit Code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
    if ($IsInteractive) {
        Read-Host "Nhan Enter de thoat..."
    }
    exit $LASTEXITCODE
} else {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host "[HOAN TAT] Tien trinh da ket thuc thanh cong!" -ForegroundColor Green
    if ($Mode.ToLower() -in @("lr", "6")) {
        Write-Host "Ket qua luu tai: domain_adaptation\experiments_lr_study\" -ForegroundColor Cyan
    } else {
        Write-Host "Ket qua luu tai: domain_adaptation\finetune_results\runs\" -ForegroundColor Cyan
    }
    Write-Host "================================================================================" -ForegroundColor Green
    if ($IsInteractive) {
        Read-Host "Nhan Enter de tiep tuc..."
    }
}
