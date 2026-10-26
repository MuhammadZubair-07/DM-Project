"""Dependency and Import Verification Script (Roadmap Phase 14).

Scans all Python source files in the repository using AST analysis,
extracts all top-level module imports, filters out standard library
and local modules, and verifies that every third-party package is
explicitly declared in requirements.txt.

Fails with non-zero exit code if an import is missing from requirements.txt.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Mapping of Python import module names to PyPI package distribution names
IMPORT_TO_PKG_MAP: dict[str, str] = {
    "sklearn": "scikit-learn",
    "rapidfuzz": "rapidfuzz",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
}

# Standard library module names in Python 3.10 - 3.12
STDLIB_MODULES: set[str] = {
    "__future__", "_thread", "abc", "aifc", "argparse", "array", "ast", "asynchat",
    "asyncio", "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath",
    "cmd", "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email", "encodings",
    "ensurepip", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "msilib", "msvcrt", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re",
    "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3",
    "ssl", "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symbol", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
}

# Local application packages to ignore
LOCAL_PACKAGES: set[str] = {"src", "pages", "tests", "scripts"}


def get_all_python_files(root: Path) -> list[Path]:
    """Finds all non-ignored Python source files in the project."""
    ignored_dirs = {".venv", ".git", ".pytest_cache", "__pycache__", "build", "dist"}
    py_files: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in ignored_dirs for part in p.parts):
            continue
        py_files.append(p)
    return sorted(py_files)


def extract_imported_modules(py_file: Path) -> set[str]:
    """Parses a Python file and returns all top-level imported module names."""
    modules: set[str] = set()
    try:
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))
    except Exception as e:
        print(f"[WARN] Failed to parse {py_file}: {e}")
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                modules.add(top_pkg)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top_pkg = node.module.split(".")[0]
                modules.add(top_pkg)

    return modules


def parse_requirements_txt(req_path: Path) -> set[str]:
    """Reads requirements.txt and returns normalized package names."""
    if not req_path.exists():
        raise FileNotFoundError(f"requirements.txt not found at {req_path}")

    packages: set[str] = set()
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip version specifiers (==, >=, <=, ~=, <, >)
        match = re.match(r"^([A-Za-z0-9_\-]+)", line)
        if match:
            pkg_name = match.group(1).lower().replace("_", "-")
            packages.add(pkg_name)

    return packages


def main() -> int:
    """Main verification routine."""
    root_dir = Path(__file__).resolve().parent.parent
    req_file = root_dir / "requirements.txt"

    print("=======================================================")
    print("  Import & Dependency Verification (check_imports.py)")
    print("=======================================================")
    print(f"Project root: {root_dir}")
    print(f"Reading requirements: {req_file}")

    declared_pkgs = parse_requirements_txt(req_file)
    print(f"Declared packages in requirements.txt ({len(declared_pkgs)}): {sorted(declared_pkgs)}")

    py_files = get_all_python_files(root_dir)
    print(f"Scanning {len(py_files)} Python source files...\n")

    third_party_needed: set[str] = set()
    file_imports_map: dict[str, list[Path]] = {}

    for f in py_files:
        mods = extract_imported_modules(f)
        for m in mods:
            if m in STDLIB_MODULES or m in LOCAL_PACKAGES:
                continue
            # Map module name to package name
            pkg = IMPORT_TO_PKG_MAP.get(m, m).lower().replace("_", "-")
            third_party_needed.add(pkg)
            file_imports_map.setdefault(pkg, []).append(f)

    # Parquet read requirement: pandas needs pyarrow for parquet reading
    third_party_needed.add("pyarrow")

    print(f"Discovered {len(third_party_needed)} third-party package dependencies:")
    for pkg in sorted(third_party_needed):
        status = "[OK] DECLARED" if pkg in declared_pkgs else "[FAIL] MISSING"
        files_cnt = len(file_imports_map.get(pkg, []))
        print(f"  - {pkg:<20} : {status} (used in {files_cnt} files)")

    missing = third_party_needed - declared_pkgs
    print("\n-------------------------------------------------------")
    if missing:
        print(f"[FAIL] Missing dependencies in requirements.txt: {sorted(missing)}")
        print("Please add these packages to requirements.txt.")
        return 1
    else:
        print("[PASS] All imported dependencies are explicitly declared in requirements.txt!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
