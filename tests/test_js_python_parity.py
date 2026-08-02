"""The browser showcase must agree with the validated Python engine, exactly.

The showcase page runs the JavaScript BOCPD (sim/signal.js, via
showcase/replay.js) live in the browser; the published latency numbers come
from engine/changepoint.py. Those are two independent implementations, so this
test runs both over every vendored series and asserts identical flag indices —
not "close", identical.

The one configured difference is the run-length truncation: the simulator
defaults to 200, the Python engine to 300, and showcase/replay.js pins the JS
side to the Python default. That pinning is what this test protects; drop it and
the flag sets diverge (2018-02-05 instead of 2018-02-02 on the VIX slice).

Skips cleanly when Node is unavailable.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "validation", "markets"))

from engine.changepoint import detect_changepoints       # noqa: E402
from validate_regimes import load                        # noqa: E402

DATA = os.path.join(REPO_ROOT, "validation", "markets", "data")
HARNESS = os.path.join(REPO_ROOT, "tools", "bocpd_flags.mjs")

SERIES = [
    "vix.csv",
    "sp500_logret_gfc.csv",
    "sp500_logret_covid.csv",
    "sp500_rv21_gfc.csv",
    "sp500_rv21_covid.csv",
]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node not installed")


def _python_flags(path):
    rows = load(path)
    values = [v for _, v in rows]
    mean = statistics.mean(values)
    sd = statistics.pstdev(values) or 1.0
    return detect_changepoints([(v - mean) / sd for v in values])


def _js_flags(path):
    proc = subprocess.run(
        ["node", HARNESS, path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"node harness failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)["flags"]


@pytest.mark.parametrize("filename", SERIES)
def test_js_and_python_bocpd_flag_the_same_indices(filename):
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not vendored")
    assert _js_flags(path) == _python_flags(path)
