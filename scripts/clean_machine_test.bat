@echo off
REM Clean-Machine Test Runner for Windows (Roadmap Phase 14)
REM Creates an isolated temporary virtual environment, installs dependencies,
REM executes pytest, and verifies all Streamlit pages.

setlocal
cd /d "%~dp0\.."

echo =======================================================
echo   Launching Clean-Machine Verification Test (Windows)
echo =======================================================

python scripts\clean_machine_test.py
if errorlevel 1 (
    echo [ERROR] Clean-machine test FAILED!
    exit /b 1
)

echo [SUCCESS] Clean-machine test completed with zero errors!
endlocal
