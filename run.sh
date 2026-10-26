#!/usr/bin/env bash
# Patient Record Deduplication Tool - Linux/macOS One-Command Launcher
# Creates venv if missing, installs pinned requirements, and launches Streamlit.

set -e

echo "======================================================="
echo "  Patient Record Deduplication Tool (DSC306 Project 11)"
echo "======================================================="

echo "[1/3] Checking Python virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    if command -v python3 &>/dev/null; then
        python3 -m venv .venv
    else
        python -m venv .venv
    fi
fi

echo "[2/3] Installing / verifying pinned requirements..."
source .venv/bin/activate || source .venv/Scripts/activate
pip install --quiet -r requirements.txt

echo "[3/3] Launching Streamlit application..."
echo "NOTE: Disclaimer: Student project, not fit for clinical, financial or legal use."
python -m streamlit run app.py
