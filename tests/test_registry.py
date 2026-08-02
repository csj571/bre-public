"""Tests for the Brier-gated prior registry. numpy-only; runnable with pytest
or directly.

The research repo additionally runs this gate over a vendored TruthfulQA
prediction slice (real model, external labels). That eval lives outside this
markets-focused public tree, so the corresponding integration test is not
carried here — see PROVENANCE.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.registry import (
    DEFAULT_MAX_BRIER,
    PriorRegistry,
    PromotionBlocked,
    diff_snapshots,
)

CLOCK = lambda: 1234.5  # deterministic timestamps  # noqa: E731


def _reg(**kw):
    kw.setdefault("clock", CLOCK)
    reg = PriorRegistry(**kw)
    reg.seed({"ell": 1.0, "eta": 0.5, "policy": {"actEntropyMax": 0.4}})
    return reg


def _feed(reg, n, prob, outcome):
    for _ in range(n):
        reg.record_outcome(prob, outcome)


# ---------------- versioning mirrors the JS registry ----------------

def test_seed_current_list_reset():
    reg = _reg()
    assert reg.current()["id"] == "v1"
    assert reg.current()["diff"] is None
    assert len(reg.list()) == 1
    _feed(reg, 30, 0.9, 1)
    reg.promote({"ell": 2.0}, "test")
    assert reg.current()["id"] == "v2"
    reg.reset({"ell": 1.0})
    assert reg.current()["id"] == "v1"
    assert reg.n_resolved == 0            # outcomes cleared with the session


def test_version_stamp_carries_gate_evidence():
    reg = _reg()
    _feed(reg, 25, 0.9, 1)
    v = reg.promote({"ell": 2.0, "eta": 0.5, "policy": {"actEntropyMax": 0.4}}, "learned")
    assert v["id"] == "v2" and v["author"] == "BRE-1"
    assert v["gate"]["n_resolved"] == 25
    assert v["gate"]["rolling_brier"] == pytest.approx(0.01)   # (0.9-1)^2
    assert v["gate"]["max_brier"] == DEFAULT_MAX_BRIER
    assert {"key": "ell", "from": "1.000", "to": "2.000"} in v["diff"]


# ---------------- the B6 gate ----------------

def test_blocked_below_min_resolved():
    reg = _reg()
    _feed(reg, 10, 0.9, 1)               # good but too few
    ok, reason = reg.can_promote()
    assert not ok and "10 resolved" in reason
    with pytest.raises(PromotionBlocked, match="resolved outcomes"):
        reg.promote({"ell": 2.0}, "premature")
    assert reg.current()["id"] == "v1"   # nothing appended


def test_blocked_on_bad_rolling_brier():
    reg = _reg()
    _feed(reg, 30, 0.9, 0)               # confidently wrong: Brier 0.81
    ok, reason = reg.can_promote()
    assert not ok and "exceeds gate" in reason
    with pytest.raises(PromotionBlocked, match="rolling Brier"):
        reg.promote({"ell": 2.0}, "overconfident")


def test_chance_forecaster_sits_on_the_boundary():
    reg = _reg()
    _feed(reg, 20, 0.5, 1)               # Brier exactly 0.25 -> not above gate
    ok, _ = reg.can_promote()
    assert ok                             # <= threshold passes; must BEAT it via config
    tight = _reg(max_brier=0.1)
    _feed(tight, 20, 0.5, 1)
    assert not tight.can_promote()[0]


def test_rolling_window_ages_out_old_garbage():
    reg = _reg(window=30)
    _feed(reg, 30, 0.9, 0)               # garbage era
    assert not reg.can_promote()[0]
    _feed(reg, 30, 0.9, 1)               # good era fills the whole window
    ok, reason = reg.can_promote()
    assert ok, reason
    assert reg.rolling_brier() == pytest.approx(0.01)


def test_record_outcome_validates():
    reg = _reg()
    with pytest.raises(ValueError):
        reg.record_outcome(1.5, 1)
    with pytest.raises(ValueError):
        reg.record_outcome(0.5, 0.3)


# ---------------- diff ----------------

def test_diff_snapshots_nested_and_added():
    d = diff_snapshots(
        {"ell": 1.0, "policy": {"a": 1, "b": 2}, "gone": True},
        {"ell": 2.0, "policy": {"a": 1, "b": 3}, "new": "x"},
    )
    keys = {e["key"] for e in d}
    assert keys == {"ell", "policy", "gone", "new"}
    sub = next(e for e in d if e["key"] == "policy")["subdiff"]
    assert sub == [{"key": "b", "from": "2", "to": "3"}]
    assert diff_snapshots(None, {"a": 1}) is None


# ---------------- persistence ----------------

def test_save_load_round_trip(tmp_path):
    reg = _reg()
    _feed(reg, 25, 0.8, 1)
    reg.promote({"ell": 3.0}, "learned", author="tester")
    path = str(tmp_path / "registry.json")
    reg.save(path)
    back = PriorRegistry.load(path, clock=CLOCK)
    assert back.counter == 2
    assert back.current()["snapshot"] == {"ell": 3.0}
    assert back.n_resolved == 25
    assert back.rolling_brier() == pytest.approx(reg.rolling_brier())
    _feed(back, 1, 0.8, 1)               # window still functional after load
    assert back.can_promote()[0]


if __name__ == "__main__":
    import inspect
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                continue
            fn(**kwargs)
    print("registry tests pass (tmp_path case needs pytest)")
