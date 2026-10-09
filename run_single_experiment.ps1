# PowerShell Script: run_single_experiment.ps1
# Chay doi chung CNN-PINN va CNN-NoPINN trong moi truong konabi

param(
    [ValidateSet("all", "pinn", "nopinn")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Thiet lap bien moi truong
$env:TRAIN_PERCENT = "10"
$env:BATCH_SIZE = "8"
$env:EPOCHS = "500"
$env:ALPHA_INIT = "1.0"
$env:PINN_ACTIVATION_EPOCH = "100"
$env:SEED = "42"
$env:AUTO_RESUME = "1"
$env:FORCE_RETRAIN = "0"

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TORCH_NUM_THREADS = "4"

# 2. Xac dinh Python trong moi truong konabi
$PythonExe = "python"
if (Test-Path "C:\ProgramData\miniconda3\envs\konabi\python.exe") {
    $PythonExe = "C:\ProgramData\miniconda3\envs\konabi\python.exe"
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "BAT DAU CHAY DOI CHUNG: CNN-PINN & CNN-NoPINN" -ForegroundColor Cyan
Write-Host "Python Executable : $PythonExe"
Write-Host "Che do chay       : $Mode"
Write-Host "Cau hinh          : TRAIN_PERCENT=$($env:TRAIN_PERCENT)%, EPOCHS=$($env:EPOCHS), ALPHA=$($env:ALPHA_INIT), WARMUP=$($env:PINN_ACTIVATION_EPOCH)"
Write-Host "================================================================================"

Set-Location "$ScriptDir\cnn"

if ($Mode -eq "all" -or $Mode -eq "pinn") {
    Write-Host "`n[1/2] Dang chay CNN-PINN (main_percent_new.py)..." -ForegroundColor Yellow
    & $PythonExe main_percent_new.py
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[1/2] CNN-PINN gap loi voi exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[1/2] Hoan thanh huan luyen CNN-PINN!" -ForegroundColor Green
}

if ($Mode -eq "all" -or $Mode -eq "nopinn") {
    Write-Host "`n[2/2] Dang chay CNN-NoPINN (main_percent_no_pinn.py)..." -ForegroundColor Yellow
    & $PythonExe main_percent_no_pinn.py
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[2/2] CNN-NoPINN gap loi voi exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[2/2] Hoan thanh huan luyen CNN-NoPINN!" -ForegroundColor Green
}

Set-Location $ScriptDir
Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "[HOAN TAT] Da thuc hien xong tat ca thu nghiem!" -ForegroundColor Green
Write-Host "================================================================================"
