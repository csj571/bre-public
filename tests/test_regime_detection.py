"""The headline result, pinned as a test: 2008 and COVID, on real market data.

Every number quoted in the README and on the showcase page is asserted here, so
the claim cannot drift away from the code. Detection latency is measured in
TRADING DAYS from the pre-registered break onset to the first BOCPD flag.

Pre-registered onsets (fixed before any of this data was scored):
    Lehman Brothers bankruptcy   2008-09-15
    COVID-19 crash               2020-02-20

This measures DETECTION, not prediction — the flag lands on or after the onset,
never before it.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "validation", "markets"))

from engine.changepoint import detect_changepoints            # noqa: E402
from validate_regimes import load, nearest_index, zscore      # noqa: E402

DATA = os.path.join(REPO_ROOT, "validation", "markets", "data")

LEHMAN = "2008-09-15"
COVID = "2020-02-20"


def _flag_dates(filename):
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not vendored")
    rows = load(path)
    dates = [d for d, _ in rows]
    flags = detect_changepoints(zscore([v for _, v in rows]))
    return dates, flags


def _latency(dates, flags, onset, tol=10):
    """Trading days from the onset to the first flag inside [onset, onset+tol]."""
    idx = nearest_index(dates, onset)
    assert dates[idx] == onset, f"{onset} is not a trading day in this series"
    hit = next((f for f in flags if idx <= f <= idx + tol), None)
    return None if hit is None else hit - idx


def test_sp500_log_returns_gfc_lehman_detected_same_day():
    dates, flags = _flag_dates("sp500_logret_gfc.csv")
    assert _latency(dates, flags, LEHMAN) == 0


def test_sp500_log_returns_covid_detected_in_two_days():
    dates, flags = _flag_dates("sp500_logret_covid.csv")
    assert _latency(dates, flags, COVID) == 2


def test_vix_slice_detects_both_breaks_at_zero_and_two_days():
    dates, flags = _flag_dates("vix.csv")
    assert _latency(dates, flags, LEHMAN) == 0
    assert _latency(dates, flags, COVID) == 2
    assert len(flags) == 14, "flag count on the vendored slice changed"


def test_smoothed_realized_vol_is_the_documented_negative_control():
    """Vanilla Adams-MacKay BOCPD assumes i.i.d.-within-regime observations. Fed
    a strongly autocorrelated 21-day rolling-vol series it misses COVID entirely
    — the failure mode the BOCPD literature predicts, and the reason the headline
    runs on log returns. See validation/markets/results/regime_validation.md.
    """
    dates, flags = _flag_dates("sp500_rv21_covid.csv")
    assert _latency(dates, flags, COVID) is None
