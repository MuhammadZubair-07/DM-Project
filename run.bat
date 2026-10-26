@echo off
REM Patient Record Deduplication Tool - Windows One-Command Launcher
REM Creates venv if missing, installs pinned requirements, and launches Streamlit.

setlocal

echo =======================================================
echo   Patient Record Deduplication Tool (DSC306 Project 11)
echo =======================================================

echo [1/3] Checking Python virtual environment...
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.11+ is installed and on PATH.
        exit /b 1
    )
)

echo [2/3] Installing / verifying pinned requirements...
call .venv\Scripts\activate.bat
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    exit /b 1
)

echo [3/3] Launching Streamlit application...
echo NOTE: Disclaimer: Student project, not fit for clinical, financial or legal use.
python -m streamlit run app.py

endlocal
