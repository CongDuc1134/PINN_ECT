@echo off
REM ==============================================================================
REM run_domain_adaptation.bat
REM SINGLE UNIFIED MASTER RUNNER FOR ALL DOMAIN ADAPTATION BENCHMARKS
REM
REM Usage:
REM   Double-click in Explorer : Opens interactive menu
REM   run_domain_adaptation.bat            (Interactive menu)
REM   run_domain_adaptation.bat all        (Runs all 43 models)
REM   run_domain_adaptation.bat all_mlp    (Runs all 24 MLP models)
REM   run_domain_adaptation.bat mlp        (Runs 12 Multitask MLP models)
REM   run_domain_adaptation.bat xiong      (Runs 12 Xiong PINN models)
REM   run_domain_adaptation.bat cnn        (Runs 19 CNN models)
REM   run_domain_adaptation.bat single     (Runs 1 default model)
REM   run_domain_adaptation.bat status     (Displays master summary)
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
echo   [1] Chay TAT CA 43 models (CNN + Multitask MLP + Xiong PINN) [Khuyen nghi]
echo   [2] Chay TOAN BO 24 model MLP (12 Multitask MLP + 12 Xiong PINN)
echo   [3] Chay 12 model Multitask MLP PINN
echo   [4] Chay 12 model Xiong et al. PINN
echo   [5] Chay 19 model CNN (PINN + Baselines)
echo   [6] Chay 1 model PINN mac dinh (5%% data, a1, seed 123)
echo   [7] Xem bang tong hop ket qua hien tai (master summary)
echo   [0] Thoat
echo.
echo ================================================================================
set /p "CHOICE=Nhap lua chon cua ban [0-7, mac dinh 1]: "
if "!CHOICE!"=="" set "CHOICE=1"

if "!CHOICE!"=="1" goto :RUN_ALL
if "!CHOICE!"=="2" goto :RUN_ALL_MLP
if "!CHOICE!"=="3" goto :RUN_MLP
if "!CHOICE!"=="4" goto :RUN_XIONG
if "!CHOICE!"=="5" goto :RUN_CNN
if "!CHOICE!"=="6" goto :RUN_SINGLE
if "!CHOICE!"=="7" goto :SHOW_STATUS
if "!CHOICE!"=="0" goto :EXIT
echo Lua chon khong hop le!
pause
goto :MENU

:HANDLE_ARGS
set "ARG=%~1"
if /i "!ARG!"=="all" goto :RUN_ALL
if /i "!ARG!"=="all_mlp" goto :RUN_ALL_MLP
if /i "!ARG!"=="mlp" goto :RUN_MLP
if /i "!ARG!"=="xiong" goto :RUN_XIONG
if /i "!ARG!"=="cnn" goto :RUN_CNN
if /i "!ARG!"=="single" goto :RUN_SINGLE
if /i "!ARG!"=="status" goto :SHOW_STATUS
echo Tham so '!ARG!' khong hop le. Cac tham so ho tro: all, all_mlp, mlp, xiong, cnn, single, status
goto :EXIT

:RUN_ALL
echo.
echo [START] Dang khoi chay TOAN BO 43 MODELS (Tu dong bo qua cac model da chay)...
"!PYTHON_EXE!" domain_adaptation/run_batch_all.py --suite all --skip-completed %2 %3 %4
goto :AFTER_RUN

:RUN_ALL_MLP
echo.
echo [START] Dang khoi chay TOAN BO 24 MODEL MLP (Multitask MLP + Xiong PINN)...
"!PYTHON_EXE!" domain_adaptation/run_batch_all.py --suite all_mlp --skip-completed %2 %3 %4
goto :AFTER_RUN

:RUN_MLP
echo.
echo [START] Dang khoi chay 12 MODEL MULTITASK MLP PINN...
"!PYTHON_EXE!" domain_adaptation/run_batch_all.py --suite mlp --skip-completed %2 %3 %4
goto :AFTER_RUN

:RUN_XIONG
echo.
echo [START] Dang khoi chay 12 MODEL XIONG ET AL. PINN...
"!PYTHON_EXE!" domain_adaptation/run_batch_all.py --suite xiong --skip-completed %2 %3 %4
goto :AFTER_RUN

:RUN_CNN
echo.
echo [START] Dang khoi chay 19 MODEL CNN...
"!PYTHON_EXE!" domain_adaptation/run_batch_all.py --suite cnn --skip-completed %2 %3 %4
goto :AFTER_RUN

:RUN_SINGLE
echo.
echo [START] Dang chay 1 MODEL MAC DINH (5%% PINN base a1 seed 123)...
"!PYTHON_EXE!" domain_adaptation/run_all.py %2 %3 %4
goto :AFTER_RUN

:SHOW_STATUS
echo.
echo ================================================================================
echo                  BANG TONG HOP KET QUA HIEN TAI
echo ================================================================================
"!PYTHON_EXE!" -c "import os, pandas as pd; f='domain_adaptation/results/master_all_models_summary.csv'; print(f'Duong dan: {f}'); df=pd.read_csv(f) if os.path.exists(f) else None; print(f'Tong so model da danh gia: {df[\"Evaluated_Model\"].nunique() if df is not None else 0}'); cols=[c for c in ['Architecture','Evaluated_Model','Method','Overall_MAE (mm)','Clf_Accuracy (%%)'] if df is not None and c in df.columns]; print(df[cols].to_string(index=False) if df is not None else 'Chua co du lieu.')"
echo ================================================================================
if "!INTERACTIVE!"=="1" (
    pause
    goto :MENU
)
goto :EXIT

:AFTER_RUN
echo.
echo ================================================================================
echo [HOAN TAT] Tien trinh da ket thuc thanh cong!
echo File tong hop: domain_adaptation\results\master_all_models_summary.csv
echo Thu muc ket qua:
echo   - CNN: domain_adaptation\results\cnn\
echo   - MLP: domain_adaptation\results\mlp\
echo ================================================================================
if "!INTERACTIVE!"=="1" (
    pause
)

:EXIT
endlocal
