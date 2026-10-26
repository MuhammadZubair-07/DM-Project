#!/usr/bin/env bash
# Clean-Machine Test Runner for Linux/macOS (Roadmap Phase 14)
# Creates an isolated temporary virtual environment, installs dependencies,
# executes pytest, and verifies all Streamlit pages.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "======================================================="
echo "  Launching Clean-Machine Verification Test (Linux/macOS)"
echo "======================================================="

if command -v python3 &>/dev/null; then
    python3 scripts/clean_machine_test.py
else
    python scripts/clean_machine_test.py
fi

echo "[SUCCESS] Clean-machine test completed with zero errors!"
