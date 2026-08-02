"""Tests for the Beta-Binomial truth-tracker. numpy + stdlib;
runnable with pytest or directly (`python tests/test_beta_binomial.py`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.beta_binomial import (
    betainc, beta_ppf, BetaBinomial, BetaBinomialCalibration,
)


def test_incomplete_beta_known_values():
    assert abs(betainc(0.3, 1.0, 1.0) - 0.3) < 1e-9      # uniform CDF
    assert abs(betainc(0.5, 2.0, 2.0) - 0.5) < 1e-9      # symmetric Beta(2,2)
    assert abs(beta_ppf(0.5, 3.0, 3.0) - 0.5) < 1e-6     # symmetric -> median 0.5
    # ppf/cdf roundtrip
    x = beta_ppf(0.4, 3.0, 5.0)
    assert abs(betainc(x, 3.0, 5.0) - 0.4) < 1e-6


def test_posterior_converges():
    rng = np.random.default_rng(0)
    bb = BetaBinomial().observe(rng.random(2000) < 0.7)
    assert bb.n == 2000
    assert abs(bb.mean - 0.7) < 0.03                     # converged to the true rate


def test_credible_interval_sane():
    rng = np.random.default_rng(1)
    small = BetaBinomial().observe(rng.random(20) < 0.7)
    big = BetaBinomial().observe(rng.random(2000) < 0.7)
    lo_s, hi_s = small.credible_interval(0.95)
    lo_b, hi_b = big.credible_interval(0.95)
    assert lo_s < small.mean < hi_s and lo_b < big.mean < hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s)                 # interval shrinks with data
    assert lo_b < 0.7 < hi_b                             # 95% CI brackets the truth


def test_update_counts():
    bb = BetaBinomial().update_counts(7, 10)             # uniform prior + 7/10
    assert bb.alpha == 8.0 and bb.beta == 4.0
    assert abs(bb.mean - 8.0 / 12.0) < 1e-12


def test_bin_calibration_truth_rate():
    # well-calibrated: P(true)=p at every confidence p
    rng = np.random.default_rng(2)
    probs = rng.random(5000)
    outc = rng.random(5000) < probs
    cal = BetaBinomialCalibration().observe(probs, outc)
    for row in cal.curve():
        if row["n"] >= 50:
            assert row["ci_low"] <= row["bin_center"] <= row["ci_high"]  # truth ~ confidence

    # overconfident: top bin claims ~0.95 but is right ~0.6
    hi = BetaBinomialCalibration()
    hi.observe([0.95] * 200, (rng.random(200) < 0.6))
    top = [r for r in hi.curve() if r["bin_center"] > 0.9][0]
    assert top["truth_rate"] < 0.9 and top["ci_high"] < 0.9   # detects the overconfidence


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  " + name)
            except Exception as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
