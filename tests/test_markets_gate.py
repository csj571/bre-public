"""Launch-blocking markets gate over the vendored real VIX slice.

Runs validation/markets/validate_regimes.py as a subprocess from the repo root
(its import resolution is CWD-dependent) and asserts the gate passes: both
pre-registered structural breaks (Lehman 2008, COVID 2020) detected within
tolerance, and the documented-adjusted false-alarm rate under threshold. See
the SECONDARY_EVENTS honesty note in that script — the raw FAR against the
2-break label set is printed alongside and is expected to be high.

If this gate fails, that is a finding about the detector or the data — do not
tune constants or add events to make it pass.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "validation", "markets", "validate_regimes.py")
DATA = os.path.join(REPO_ROOT, "validation", "markets", "data", "vix.csv")


@pytest.mark.skipif(not os.path.exists(DATA), reason="vendored VIX slice missing")
def test_markets_regime_gate():
    proc = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"markets gate failed:\n{proc.stdout}\n{proc.stderr}"
    assert "detected 2/2 breaks" in proc.stdout
    assert "GATE: PASS" in proc.stdout


if __name__ == "__main__":
    test_markets_regime_gate()
    print("markets gate passes")
