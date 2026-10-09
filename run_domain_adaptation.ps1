# ==============================================================================
# run_domain_adaptation.ps1
# POWERSHELL MASTER RUNNER FOR DOMAIN ADAPTATION BENCHMARKS
#
# Cách sử dụng:
#   .\run_domain_adaptation.ps1                (Menu tương tác chọn chế độ)
#   .\run_domain_adaptation.ps1 all            (Chạy cả Protocol 1 & 2 cho toàn bộ mô hình)
#   .\run_domain_adaptation.ps1 p1             (Chạy Protocol 1: Scan 1 vs Scan 2)
#   .\run_domain_adaptation.ps1 p2             (Chạy Protocol 2: 10-Fold Defect Cross Validation)
#   .\run_domain_adaptation.ps1 cnn            (Chạy riêng cho mô hình CNN)
#   .\run_domain_adaptation.ps1 test           (Chạy 1-epoch quick smoke test)
#
# Có thể truyền thêm các tham số tùy chọn bổ sung:
#   .\run_domain_adaptation.ps1 p1 --epochs 100 --lr 1e-6
# ==============================================================================

param(
    [string]$Mode = "",
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
    Write-Host "           ECT DOMAIN ADAPTATION: MASTER BENCHMARK RUNNER" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Python Executable : $PythonExe"
    Write-Host "Workspace Root    : $ScriptDir"
    Write-Host ""
    try {
        & $PythonExe -c "import torch; print(f'Hardware Status   : GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.cuda.is_available()}')" 2>$null
    } catch {}
    Write-Host "================================================================================" -ForegroundColor Cyan
}

if ($IsInteractive) {
    Show-Header
    Write-Host ""
    Write-Host "Hay chon che do chay benchmark:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  [1] Chay CA 2 GIAO THUC (Protocol 1: Scan Split + Protocol 2: 10-Fold Defect) [Khuyen nghi]"
    Write-Host "  [2] Chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test - Sensor Drift & Repeatability)"
    Write-Host "  [3] Chay GIAO THUC 2 (10-Fold Leave-One-Defect-Out Cross-Validation)"
    Write-Host "  [4] Chay rieng cho mo hinh CNN (Protocol 1 & 2)"
    Write-Host "  [5] Chay Quick Smoke Test (1 epoch de kiem tra nhanh)"
    Write-Host "  [0] Thoat"
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    
    $Choice = Read-Host "Nhap lua chon cua ban [0-5, mac dinh 1]"
    if ([string]::IsNullOrWhiteSpace($Choice)) {
        $Choice = "1"
    }

    switch ($Choice) {
        "1" { $Mode = "all" }
        "2" { $Mode = "p1" }
        "3" { $Mode = "p2" }
        "4" { $Mode = "cnn" }
        "5" { $Mode = "test" }
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
$BenchmarkScript = "domain_adaptation\benchmark_evaluation_protocols.py"

$CmdArgs = @($BenchmarkScript)

switch ($Mode.ToLower()) {
    { $_ -in @("all", "1") } {
        Write-Host "[START] Dang khoi chay CA 2 GIAO THUC BENCHMARK..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "all")
    }
    { $_ -in @("p1", "2") } {
        Write-Host "[START] Dang khoi chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "1")
    }
    { $_ -in @("p2", "3") } {
        Write-Host "[START] Dang khoi chay GIAO THUC 2 (10-Fold Leave-One-Defect-Out)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "2")
    }
    { $_ -in @("cnn", "4") } {
        Write-Host "[START] Dang khoi chay Benchmark rieng cho mo hinh CNN..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "all", "--models", "cnn_proposed,cnn_nopinn")
    }
    { $_ -in @("test", "5") } {
        Write-Host "[START] Dang chay Quick Smoke Test (1 epoch)..." -ForegroundColor Cyan
        $CmdArgs += @("--protocol", "1", "--epochs", "1")
    }
    default {
        Write-Host "[LOI] Che do '$Mode' khong hop le. Cac che do ho tro: all, p1, p2, cnn, test" -ForegroundColor Red
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
    Write-Host "Ket qua luu tai: domain_adaptation\finetune_results\runs\" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Green
    if ($IsInteractive) {
        Read-Host "Nhan Enter de tiep tuc..."
    }
}
