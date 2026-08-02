"""The vendored S&P series must be exactly what fetch_data.py derives.

The VIX slice is a straight copy of a published dataset (provenance in
validation/markets/data/README.md). The four S&P CSVs are *derived* — log
returns and 21-day realized vol computed from real closes — so a reader has to
take them on trust unless the derivation is re-runnable. It is: this test
regenerates all four into a temp directory from the same integrity-gated source
and byte-compares them against the committed files.

`fetch_data.main()` re-runs its own provenance gate on the way through (four
closes checked against independently documented historical values), so a
mismatch in the upstream dataset aborts before anything is compared.

Needs the source dataset: pip install ".[data]". Skips cleanly without it.
"""
import filecmp
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "validation", "markets"))

DATA = os.path.join(REPO_ROOT, "validation", "markets", "data")
DERIVED = [
    "sp500_logret_gfc.csv",
    "sp500_logret_covid.csv",
    "sp500_rv21_gfc.csv",
    "sp500_rv21_covid.csv",
]

pytest.importorskip("skfolio", reason="source dataset not installed (pip install '.[data]')")

import fetch_data  # noqa: E402


@pytest.mark.parametrize("filename", DERIVED)
def test_vendored_series_match_a_fresh_derivation(filename, tmp_path, monkeypatch):
    committed = os.path.join(DATA, filename)
    if not os.path.exists(committed):
        pytest.skip(f"{filename} not vendored")

    monkeypatch.setattr(fetch_data, "DATA_DIR", str(tmp_path))
    fetch_data.main()

    regenerated = os.path.join(str(tmp_path), filename)
    assert os.path.exists(regenerated), "fetch_data.py did not produce this file"
    assert filecmp.cmp(committed, regenerated, shallow=False), (
        f"{filename} differs from a fresh derivation — the committed data is stale "
        f"or the derivation changed"
    )
