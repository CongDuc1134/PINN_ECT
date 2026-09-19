@echo off
REM ==============================================================================
REM Windows Automation Script: run_domain_adaptation.bat
REM 1-Click Automated Runner for Domain Adaptation on Real 5kHz ECT Data
REM ==============================================================================

echo ================================================================================
echo [START] Launching Automated Domain Adaptation (10-Fold LODO Pooled OOF)
echo ================================================================================

conda run --no-capture-output -n konabi python domain_adaptation/run_all.py %*

echo ================================================================================
echo [DONE] Domain Adaptation Pipeline Finished!
echo Results are saved in: domain_adaptation\results\
echo ================================================================================
pause
