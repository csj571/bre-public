"""Tests for the calibration core. numpy-only; runnable with
pytest or directly (`python tests/test_calibration.py`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.calibration import (
    brier_score, brier_decomposition, ece, mce, reliability_curve,
    PlattScaler, IsotonicCalibrator, cohens_kappa, passes_ece_gate,
)


def test_brier_known_values():
    assert brier_score([0, 1, 1, 0], [0, 1, 1, 0]) == 0.0          # perfect
    probs = [0.9] * 10
    outc = [1] * 5 + [0] * 5
    assert abs(brier_score(probs, outc) - 0.41) < 1e-9             # hand-computed
    assert abs(brier_score([0.5] * 8, [1, 0] * 4) - 0.25) < 1e-9   # chance baseline


def test_ece_known_and_calibrated():
    # 10 preds at conf 0.9, only half correct -> single bin, |0.9-0.5| = 0.4
    assert abs(ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9
    # perfectly calibrated discrete forecasts -> ECE ~ 0
    probs = np.array([0.1] * 100 + [0.5] * 100 + [0.9] * 100)
    y = np.array([1] * 10 + [0] * 90 + [1] * 50 + [0] * 50 + [1] * 90 + [0] * 10, float)
    assert ece(probs, y) < 1e-9
    assert mce(probs, y) >= ece(probs, y)                          # MCE >= ECE always


def test_brier_decomposition_identity():
    # discrete forecasts (one value per bin) -> decomposition is exact, reliability 0
    probs = np.array([0.1] * 100 + [0.5] * 100 + [0.9] * 100)
    y = np.array([1] * 10 + [0] * 90 + [1] * 50 + [0] * 50 + [1] * 90 + [0] * 10, float)
    d = brier_decomposition(probs, y)
    assert np.isclose(d.reliability, 0.0, atol=1e-9)               # well-calibrated
    assert np.isclose(d.brier, brier_score(probs, y), atol=1e-9)   # rel - res + unc == BS
    assert d.resolution > 0 and d.uncertainty > 0


def test_reliability_curve():
    probs = np.array([0.1] * 100 + [0.9] * 100)
    y = np.array([1] * 10 + [0] * 90 + [1] * 90 + [0] * 10, float)
    rc = reliability_curve(probs, y)
    assert int(rc.bin_count.sum()) == 200
    assert np.allclose(rc.bin_accuracy, rc.bin_confidence, atol=1e-9)  # calibrated -> diagonal


def _overconfident_set(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    true_p = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_p).astype(float)
    model_p = 1.0 / (1.0 + np.exp(-2.5 * latent))   # sharpened -> overconfident
    return model_p, y


def test_platt_reduces_ece():
    p, y = _overconfident_set()
    raw = ece(p, y)
    cal = ece(PlattScaler().fit(p, y).transform(p), y)
    assert raw > 0.05                       # the raw model really is miscalibrated
    assert cal < raw                        # Platt helps
    assert cal < 0.05                        # and gets us under the gate


def test_platt_does_not_diverge_on_saturating_data():
    """Regression: undamped Newton/IRLS used to run away on data that saturates
    the sigmoid — W -> 0, so H -> ridge*I and the step exploded. The fit landed
    at |a| ~ 1e8 with a hard 0/1 transform, which made ECE *worse* than the raw
    confidences while only emitting a numpy overflow warning. The backtracking
    line search (Platt 1999) fixes it; coefficients must stay finite and small,
    and recalibration must not hurt."""
    rng = np.random.default_rng(0)
    n = 300
    # Bimodal near-0/near-1 confidences whose outcomes are deliberately shrunk
    # toward the middle: the MLE wants a huge slope, so Newton saturates.
    p = np.clip(rng.beta(0.4, 0.4, n), 0.01, 0.99)
    y = (rng.random(n) < np.clip(p * 0.7 + 0.1, 0.0, 1.0)).astype(float)

    scaler = PlattScaler().fit(p, y)
    assert np.isfinite(scaler.a) and np.isfinite(scaler.b)
    assert abs(scaler.a) < 100.0 and abs(scaler.b) < 100.0

    out = scaler.transform(p)
    assert np.all(np.isfinite(out))
    assert 0.0 < out.min() and out.max() < 1.0      # not hard-saturated
    assert ece(out, y) <= ece(p, y)                 # recalibration never hurts


def test_isotonic_reduces_ece():
    p, y = _overconfident_set(seed=1)
    raw = ece(p, y)
    cal = ece(IsotonicCalibrator().fit(p, y).transform(p), y)
    assert cal < raw


def test_cohens_kappa():
    a = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    assert cohens_kappa(a, a) == 1.0
    rng = np.random.default_rng(2)
    x, z = rng.integers(0, 2, 1000), rng.integers(0, 2, 1000)
    assert abs(cohens_kappa(x, z)) < 0.1     # independent raters -> ~0


def test_ece_gate():
    assert passes_ece_gate(0.05) is True
    assert passes_ece_gate(0.20) is False


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
