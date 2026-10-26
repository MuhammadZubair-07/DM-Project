"""Clean-Machine Automated Verification Script (Roadmap Phase 14).

Creates an isolated temporary virtual environment, installs dependencies
from requirements.txt, runs the test suite, performs a headless application
smoke import across all six pages, and cleans up the temporary directory.

Usage:
    python scripts/clean_machine_test.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Executes a shell command and streams stdout/stderr."""
    print(f">> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    req_file = project_root / "requirements.txt"

    print("=======================================================")
    print("  Clean-Machine Packaging & Portability Test")
    print("=======================================================")
    print(f"Project Root: {project_root}")
    print(f"Requirements: {req_file}")

    if not req_file.exists():
        print(f"[FAIL] requirements.txt not found at {req_file}")
        return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="npmi_clean_test_"))
    print(f"Created temporary sandbox: {temp_dir}")

    try:
        venv_dir = temp_dir / "venv"
        print("\n[1/4] Creating clean virtual environment...")
        ret = run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
        if ret != 0:
            print("[FAIL] Virtual environment creation failed.")
            return 1

        # Determine python/pip executable in temp venv
        if sys.platform == "win32":
            py_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
            pytest_exe = venv_dir / "Scripts" / "pytest.exe"
        else:
            py_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
            pytest_exe = venv_dir / "bin" / "pytest"

        print(f"Temporary Python interpreter: {py_exe}")

        print("\n[2/4] Installing dependencies from requirements.txt...")
        ret = run_cmd([str(pip_exe), "install", "--no-warn-script-location", "--quiet", "-r", str(req_file)])
        if ret != 0:
            print("[FAIL] Failed to install requirements in clean environment.")
            return 1
        print("[PASS] Dependencies installed successfully.")

        # Set PYTHONPATH so temporary environment can resolve src and pages
        clean_env = os.environ.copy()
        clean_env["PYTHONPATH"] = str(project_root)

        print("\n[3/4] Running automated test suite in clean environment...")
        ret = run_cmd(
            [str(pytest_exe), "-q", "tests/"],
            cwd=project_root,
            env=clean_env,
        )
        if ret != 0:
            print("[FAIL] Test suite failed in clean environment.")
            return 1
        print("[PASS] All tests passed in clean environment.")

        print("\n[4/4] Executing headless Streamlit smoke imports across all pages...")
        smoke_script = (
            "import py_compile, sys, os\n"
            "pages = ['app.py', 'pages/1_Data_and_Quality.py', 'pages/2_Model_Performance.py', "
            "'pages/3_Single_Prediction.py', 'pages/4_Batch_Scoring.py', "
            "'pages/5_Settings_and_Cost_Matrix.py', 'pages/6_Merge_Review.py']\n"
            "for p in pages:\n"
            "    py_compile.compile(p, doraise=True)\n"
            "    print(f'  - Verified: {p}')\n"
            "print('All pages verified successfully!')\n"
        )

        ret = run_cmd([str(py_exe), "-c", smoke_script], cwd=project_root, env=clean_env)
        if ret != 0:
            print("[FAIL] Headless smoke check failed.")
            return 1
        print("[PASS] Headless smoke checks passed across all pages.")

        print("\n=======================================================")
        print("  CLEAN-MACHINE TEST PASSED SUCCESSFULLY!")
        print("=======================================================")
        return 0

    finally:
        print(f"\nCleaning up temporary sandbox: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
