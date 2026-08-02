"""The README's runnable example has to keep running.

A broken quickstart is worse than no quickstart, so CI executes it and checks the
line that matters rather than just the exit code.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "detect_regime_change.py")


def test_regime_example_finds_lehman_on_the_day():
    proc = subprocess.run(
        [sys.executable, EXAMPLE],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"example failed:\n{proc.stdout}\n{proc.stderr}"
    assert "2008-09-15" in proc.stdout
    assert "Lehman Brothers bankruptcy" in proc.stdout
    assert "0 trading days" in proc.stdout
