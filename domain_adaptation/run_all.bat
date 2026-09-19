@echo off
REM ==============================================================================
REM Windows 1-Click Runner for Domain Adaptation on Real ECT Data (5kHz)
REM ==============================================================================

echo [START] Running Domain Adaptation Benchmark with 10-Fold LODO on 5kHz Real Data...
conda run -n konabi python domain_adaptation/run_all.py %*
echo [DONE] Domain adaptation benchmark finished!
pause
