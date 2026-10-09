@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM 1. Tim Python trong moi truong konabi
set "PYTHON_EXE="
if exist "C:\ProgramData\miniconda3\envs\konabi\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\miniconda3\envs\konabi\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 set "PYTHON_EXE=python"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Khong tim thay Python. Vui long chay: conda activate konabi
    pause
    exit /b 1
)

REM 2. Thu muc tam va log
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TMPDIR=%~dp0tmp"
set "LOG_FILE=%~dp0experiment_pinn_vs_nopinn.log"

REM 3. Cau hinh thong so
set "TRAIN_PERCENT=10"
set "BATCH_SIZE=8"
set "EPOCHS=500"
set "ALPHA_INIT=1.0"
set "PINN_ACTIVATION_EPOCH=100"
set "SEED=42"
set "AUTO_RESUME=1"
set "FORCE_RETRAIN=0"

set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "TORCH_NUM_THREADS=4"

echo ================================================================================
echo BAT DAU CHAY DOI CHUNG: CNN-PINN VA CNN-NoPINN
echo Workspace Root    : %~dp0
echo Python Executable : %PYTHON_EXE%
echo Cau hinh          : TRAIN_PERCENT=%TRAIN_PERCENT%%%, EPOCHS=%EPOCHS%, ALPHA=%ALPHA_INIT%
echo ================================================================================
echo.
echo Chon che do chay:
echo   [1] Chay ca 2: PINN roi den No-PINN (Mac dinh)
echo   [2] Chi chay CNN-PINN (main_percent_new.py)
echo   [3] Chi chay CNN-NoPINN (main_percent_no_pinn.py)
echo   [0] Thoat
echo ================================================================================
set "CHOICE=1"
set /p "CHOICE=Nhap lua chon cua ban [1-3, mac dinh 1]: "

if "%CHOICE%"=="0" goto :EXIT

cd /d "%~dp0cnn"

if "%CHOICE%"=="2" goto :RUN_PINN_ONLY
if "%CHOICE%"=="3" goto :RUN_NOPINN_ONLY

:RUN_BOTH
echo.
echo [1/2] Dang chay CNN-PINN (main_percent_new.py)...
"%PYTHON_EXE%" main_percent_new.py
if errorlevel 1 goto :FAIL
echo.
echo [2/2] Dang chay CNN-NoPINN (main_percent_no_pinn.py)...
"%PYTHON_EXE%" main_percent_no_pinn.py
if errorlevel 1 goto :FAIL
goto :DONE

:RUN_PINN_ONLY
echo.
echo Dang chay CNN-PINN (main_percent_new.py)...
"%PYTHON_EXE%" main_percent_new.py
if errorlevel 1 goto :FAIL
goto :DONE

:RUN_NOPINN_ONLY
echo.
echo Dang chay CNN-NoPINN (main_percent_no_pinn.py)...
"%PYTHON_EXE%" main_percent_no_pinn.py
if errorlevel 1 goto :FAIL
goto :DONE

:FAIL
echo.
echo [ERROR] Qua trinh thuc thi gap loi!
pause
exit /b 1

:DONE
echo.
echo ================================================================================
echo [HOAN TAT] Qua trinh thuc thi da ket thuc thanh cong!
echo ================================================================================
pause

:EXIT
endlocal
