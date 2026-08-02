"""Tests for the B5 decision stack (BOCPD, adaptive Kalman, policy gate).

NOTE: the regime series here are SYNTHETIC — these are algorithm smoke tests
("does BOCPD recover a known switch"), not external validation. Real markets
validation scores BOCPD against dated breaks on a vendored VIX/S&P slice.
Runnable with pytest or directly (`python tests/test_decision_stack.py`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.changepoint import BOCPD, detect_changepoints, changepoint_scores
from engine.kalman import AdaptiveKalman, kalman_filter
from engine.policy_gate import Action, decide


def test_bocpd_mean_shift():
    rng = np.random.default_rng(0)
    series = np.concatenate([rng.normal(0.0, 0.5, 120), rng.normal(4.0, 0.5, 120)])
    flags = detect_changepoints(series)
    sc = changepoint_scores(flags, [120], tol=20)
    assert sc["n_detected"] == 1                 # caught the switch
    assert sc["false_alarms"] <= 3               # few spurious flags in calm data


def test_bocpd_variance_shift():
    rng = np.random.default_rng(1)
    series = np.concatenate([rng.normal(0.0, 0.4, 150), rng.normal(0.0, 2.0, 150)])
    flags = detect_changepoints(series)
    sc = changepoint_scores(flags, [150], tol=25)
    assert sc["n_detected"] == 1


def test_kalman_tracks_step():
    rng = np.random.default_rng(2)
    series = np.concatenate([np.zeros(40), np.full(40, 5.0)]) + rng.normal(0, 0.2, 80)
    filt = kalman_filter(series)
    assert abs(filt[39] - 0.0) < 1.0             # tracking the low level
    assert abs(filt[-1] - 5.0) < 1.0             # converged after the step


def test_kalman_noise_adapts_upward():
    kf = AdaptiveKalman()
    for z in [0.0] * 20:
        kf.update(z)
    r_calm = kf.r
    for z in [3.0, -3.0] * 10:                   # sudden high-variance stream
        kf.update(z)
    assert kf.r > r_calm                         # observation-noise estimate widened


def test_policy_gate_routes():
    assert decide(0.05, 0.1, 0.95) == Action.ACT
    assert decide(0.05, 0.3, 0.50) == Action.DEFER          # low epistemic + low conf -> irreducible
    assert decide(0.50, 0.1, 0.50, eig=1.0) == Action.QUERY_MORE
    assert decide(0.50, 0.1, 0.99, eig=0.0, eig_min=0.1) == Action.DEFER  # not worth querying
    assert decide(0.05, 0.1, 0.99, sensitive=True) == Action.DEFER


def test_changepoint_scores():
    sc = changepoint_scores([121], [120], tol=5)
    assert sc["n_detected"] == 1 and sc["latencies"] == [1] and sc["false_alarms"] == 0
    sc2 = changepoint_scores([121, 10], [120], tol=5)
    assert sc2["false_alarms"] == 1 and abs(sc2["false_alarm_rate"] - 0.5) < 1e-9
    sc3 = changepoint_scores([], [120], tol=5)
    assert sc3["n_detected"] == 0 and sc3["false_alarm_rate"] == 0.0


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
