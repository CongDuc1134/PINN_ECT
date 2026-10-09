@echo off
REM ==============================================================================
REM run_domain_adaptation.bat
REM SINGLE UNIFIED MASTER RUNNER FOR DOMAIN ADAPTATION BENCHMARKS
REM
REM Usage:
REM   Double-click in Explorer : Opens interactive menu
REM   run_domain_adaptation.bat               (Interactive menu)
REM   run_domain_adaptation.bat all           (Runs Protocol 1 & 2 on all models)
REM   run_domain_adaptation.bat p1            (Runs Protocol 1: Scan 1 vs Scan 2)
REM   run_domain_adaptation.bat p2            (Runs Protocol 2: 10-Fold Cross Validation)
REM   run_domain_adaptation.bat cnn           (Runs CNN models only)
REM   run_domain_adaptation.bat test          (Runs 1-epoch quick smoke test)
REM ==============================================================================

chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

REM 1. Resolve Python executable in conda 'konabi' environment
set "PYTHON_EXE=C:\ProgramData\miniconda3\envs\konabi\python.exe"
if not exist "!PYTHON_EXE!" (
    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do (
        if exist "%%i\envs\konabi\python.exe" set "PYTHON_EXE=%%i\envs\konabi\python.exe"
    )
)
if not exist "!PYTHON_EXE!" (
    set "PYTHON_EXE=python"
)

REM 2. Check if called with arguments
if not "%~1"=="" (
    set "INTERACTIVE=0"
    goto :HANDLE_ARGS
)
set "INTERACTIVE=1"

:MENU
cls
echo ================================================================================
echo           ECT DOMAIN ADAPTATION: MASTER BENCHMARK RUNNER
echo ================================================================================
echo Python Executable : !PYTHON_EXE!
echo Workspace Root    : %~dp0
echo.
"!PYTHON_EXE!" -c "import torch; print(f'Hardware Status   : GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.cuda.is_available()}')" 2>nul
echo ================================================================================
echo.
echo Hay chon che do chay benchmark:
echo.
echo   [1] Chay CA 2 GIAO THUC (Protocol 1: Scan Split + Protocol 2: 10-Fold Defect) [Khuyen nghi]
echo   [2] Chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test - Sensor Drift & Repeatability)
echo   [3] Chay GIAO THUC 2 (10-Fold Leave-One-Defect-Out Cross-Validation)
echo   [4] Chay rieng cho mo hinh CNN (Protocol 1 & 2)
echo   [5] Chay Quick Smoke Test (1 epoch de kiem tra nhanh)
echo   [0] Thoat
echo.
echo ================================================================================
set /p "CHOICE=Nhap lua chon cua ban [0-5, mac dinh 1]: "
if "!CHOICE!"=="" set "CHOICE=1"

if "!CHOICE!"=="1" goto :RUN_ALL
if "!CHOICE!"=="2" goto :RUN_P1
if "!CHOICE!"=="3" goto :RUN_P2
if "!CHOICE!"=="4" goto :RUN_CNN
if "!CHOICE!"=="5" goto :RUN_TEST
if "!CHOICE!"=="0" goto :EXIT
echo Lua chon khong hop le!
pause
goto :MENU

:HANDLE_ARGS
set "ARG=%~1"
if /i "!ARG!"=="all" goto :RUN_ALL
if /i "!ARG!"=="p1" goto :RUN_P1
if /i "!ARG!"=="p2" goto :RUN_P2
if /i "!ARG!"=="cnn" goto :RUN_CNN
if /i "!ARG!"=="test" goto :RUN_TEST
echo Tham so '!ARG!' khong hop le. Cac tham so ho tro: all, p1, p2, cnn, test
goto :EXIT

:RUN_ALL
echo.
echo [START] Dang khoi chay CA 2 GIAO THUC BENCHMARK...
"!PYTHON_EXE!" domain_adaptation/benchmark_evaluation_protocols.py --protocol all %2 %3 %4
goto :AFTER_RUN

:RUN_P1
echo.
echo [START] Dang khoi chay GIAO THUC 1 (Scan 1 Train / Scan 2 Test)...
"!PYTHON_EXE!" domain_adaptation/benchmark_evaluation_protocols.py --protocol 1 %2 %3 %4
goto :AFTER_RUN

:RUN_P2
echo.
echo [START] Dang khoi chay GIAO THUC 2 (10-Fold Leave-One-Defect-Out)...
"!PYTHON_EXE!" domain_adaptation/benchmark_evaluation_protocols.py --protocol 2 %2 %3 %4
goto :AFTER_RUN

:RUN_CNN
echo.
echo [START] Dang khoi chay Benchmark cho CNN...
"!PYTHON_EXE!" domain_adaptation/benchmark_evaluation_protocols.py --protocol all --models cnn_proposed,cnn_nopinn %2 %3 %4
goto :AFTER_RUN

:RUN_TEST
echo.
echo [START] Dang chay Quick Smoke Test (1 epoch)...
"!PYTHON_EXE!" domain_adaptation/benchmark_evaluation_protocols.py --protocol 1 --epochs 1 %2 %3 %4
goto :AFTER_RUN

:AFTER_RUN
echo.
echo ================================================================================
echo [HOAN TAT] Tien trinh da ket thuc thanh cong!
echo Ket qua luu tai: domain_adaptation\finetune_results\protocols\
echo ================================================================================
if "!INTERACTIVE!"=="1" (
    pause
)

:EXIT
endlocal
