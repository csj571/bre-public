"""Brier-gated prior registry: a posterior is promoted to prior only with evidence.

Python counterpart of the JS ``registry.js`` in sim/: the same versioned prior
store with the same entry shape
``{id: 'vN', timestamp, author, justification, snapshot, diff}`` — plus the two
things that version lacks:

1. **The calibration gate.** ``promote()`` succeeds only when the rolling Brier
   score over the last N *resolved* outcomes beats a threshold — a posterior
   earns promotion to prior with evidence, not a justification string. The
   evidence (rolling Brier, N, threshold) is stamped into the version entry.
2. **Persistence.** ``save()`` / ``load()`` JSON round-trip.

Defaults: at least ``DEFAULT_MIN_RESOLVED = 20`` resolved outcomes and rolling
Brier ≤ ``DEFAULT_MAX_BRIER = 0.25`` — the always-predict-0.5 chance baseline,
so a candidate must at least beat an uninformative forecaster. Both are
configurable per registry; tighten ``max_brier`` for consumers whose forecasts
should be sharp.

numpy/stdlib only, torch-free. The JS registries remain visualization-layer
artifacts and are intentionally ungated.
"""
from __future__ import annotations

import json
import time
from collections import deque

from engine.calibration import brier_score

DEFAULT_MIN_RESOLVED = 20
DEFAULT_MAX_BRIER = 0.25      # chance baseline: Brier of always predicting 0.5
DEFAULT_WINDOW = 50


class PromotionBlocked(RuntimeError):
    """Raised when promote() is called while the Brier gate is closed."""


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def diff_snapshots(a, b):
    """Recursive snapshot diff, mirroring registry.js diffSnapshots.

    Returns None when there is no previous snapshot; otherwise a list of
    ``{key, from, to}`` records, with nested dicts under ``{key, subdiff}``.
    """
    if not a:
        return None
    out = []
    for k in sorted(set(a or {}) | set(b or {})):
        av = (a or {}).get(k)
        bv = (b or {}).get(k)
        if isinstance(av, dict):
            sub = diff_snapshots(av, bv or {})
            if sub:
                out.append({"key": k, "subdiff": sub})
        elif json.dumps(av, sort_keys=True) != json.dumps(bv, sort_keys=True):
            out.append({"key": k, "from": _fmt(av), "to": _fmt(bv)})
    return out


class PriorRegistry:
    """Versioned prior store whose promotions are gated on rolling Brier."""

    def __init__(self, *, min_resolved: int = DEFAULT_MIN_RESOLVED,
                 max_brier: float = DEFAULT_MAX_BRIER,
                 window: int = DEFAULT_WINDOW, clock=time.time):
        self.min_resolved = int(min_resolved)
        self.max_brier = float(max_brier)
        self.window = int(window)
        self._clock = clock
        self.versions: list[dict] = []
        self.counter = 0
        self._outcomes: deque[tuple[float, float]] = deque(maxlen=self.window)

    # ---- versioning (mirrors the JS API) ----

    def seed(self, initial: dict) -> dict:
        v = {
            "id": "v1",
            "timestamp": self._clock(),
            "author": "human seed",
            "justification": "Initial seed prior.",
            "snapshot": json.loads(json.dumps(initial)),
            "diff": None,
        }
        self.versions = [v]
        self.counter = 1
        return v

    def current(self) -> dict | None:
        return self.versions[-1] if self.versions else None

    def list(self) -> list[dict]:
        return list(self.versions)

    def reset(self, initial: dict) -> dict:
        self.versions = []
        self.counter = 0
        self._outcomes.clear()
        return self.seed(initial)

    # ---- the B6 gate ----

    def record_outcome(self, prob: float, outcome) -> None:
        """Record one resolved forecast: predicted P(outcome=1) and the {0,1} result."""
        p = float(prob)
        y = float(outcome)
        if not 0.0 <= p <= 1.0 or y not in (0.0, 1.0):
            raise ValueError(f"bad resolved outcome: prob={prob}, outcome={outcome}")
        self._outcomes.append((p, y))

    @property
    def n_resolved(self) -> int:
        return len(self._outcomes)

    def rolling_brier(self) -> float | None:
        """Brier score over the rolling outcome window; None with no outcomes."""
        if not self._outcomes:
            return None
        probs = [p for p, _ in self._outcomes]
        ys = [y for _, y in self._outcomes]
        return brier_score(probs, ys)

    def can_promote(self) -> tuple[bool, str]:
        n = self.n_resolved
        if n < self.min_resolved:
            return False, (f"only {n} resolved outcomes; gate needs >= {self.min_resolved}")
        rb = self.rolling_brier()
        if rb > self.max_brier:
            return False, (f"rolling Brier {rb:.4f} over last {n} outcomes exceeds "
                           f"gate {self.max_brier}")
        return True, (f"rolling Brier {rb:.4f} over last {n} outcomes within gate "
                      f"{self.max_brier}")

    def promote(self, snapshot: dict, justification: str, author: str = "BRE-1") -> dict:
        """Promote posterior→prior — only through the Brier gate.

        Raises PromotionBlocked (with the gate's reason) when the evidence is
        insufficient. On success the version entry carries the gate evidence.
        """
        ok, reason = self.can_promote()
        if not ok:
            raise PromotionBlocked(reason)
        self.counter += 1
        prev = self.current()
        v = {
            "id": f"v{self.counter}",
            "timestamp": self._clock(),
            "author": author,
            "justification": justification,
            "snapshot": json.loads(json.dumps(snapshot)),
            "diff": diff_snapshots(prev["snapshot"] if prev else None, snapshot),
            "gate": {
                "rolling_brier": self.rolling_brier(),
                "n_resolved": self.n_resolved,
                "max_brier": self.max_brier,
                "min_resolved": self.min_resolved,
            },
        }
        self.versions.append(v)
        return v

    # ---- persistence ----

    def save(self, path: str) -> None:
        payload = {
            "config": {"min_resolved": self.min_resolved, "max_brier": self.max_brier,
                       "window": self.window},
            "counter": self.counter,
            "versions": self.versions,
            "outcomes": list(self._outcomes),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path: str, clock=time.time) -> "PriorRegistry":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        reg = cls(min_resolved=cfg["min_resolved"], max_brier=cfg["max_brier"],
                  window=cfg["window"], clock=clock)
        reg.counter = payload["counter"]
        reg.versions = payload["versions"]
        for p, y in payload["outcomes"]:
            reg._outcomes.append((float(p), float(y)))
        return reg
