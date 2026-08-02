"""Beta-Binomial conjugate truth-tracker (BUILD_PLAN B2; spec component 2's
missing half).

The honest belief-tracker for binary / true-false claims: start from a Beta prior
and update with each resolved outcome to a Beta posterior — closed form, never
overconfident, with a built-in credible interval. No sampling.

Credible intervals need Beta quantiles; with no scipy allowed we hand-roll the
regularized incomplete beta `I_x(a,b)` (Lentz continued fraction, Numerical
Recipes) and invert it by bisection. stdlib `math` only.

`BetaBinomialCalibration` applies one tracker per confidence bin to answer the
calibration question — "when confidence was stated at c, how often was the claim
true?" — giving a reliability curve with credible bands that feeds B0 / the B6
registry.
"""
import math
from typing import List, Sequence, Tuple

_FPMIN = 1e-300
_EPS = 3e-14


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) = CDF of Beta(a, b) at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-10) -> float:
    """Inverse CDF (quantile) of Beta(a, b) via bisection on betainc."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


class BetaBinomial:
    """Conjugate Beta(alpha, beta) tracker for a binary process. Default prior is
    uniform Beta(1, 1); pass alpha0=beta0=0.5 for Jeffreys."""

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.alpha, self.beta = float(alpha0), float(beta0)

    def update(self, outcome) -> "BetaBinomial":
        if outcome:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        return self

    def update_counts(self, k: int, n: int) -> "BetaBinomial":
        """Fold in k successes out of n trials."""
        self.alpha += float(k)
        self.beta += float(n - k)
        return self

    def observe(self, outcomes: Sequence) -> "BetaBinomial":
        for o in outcomes:
            self.update(o)
        return self

    @property
    def n(self) -> float:
        """Number of observations folded in (excludes the prior)."""
        return (self.alpha - self.alpha0) + (self.beta - self.beta0)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return a * b / (s * s * (s + 1.0))

    def std(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        tail = (1.0 - level) / 2.0
        return beta_ppf(tail, self.alpha, self.beta), beta_ppf(1.0 - tail, self.alpha, self.beta)

    def reset(self) -> None:
        self.alpha, self.beta = self.alpha0, self.beta0


class BetaBinomialCalibration:
    """One Beta-Binomial per confidence bin: tracks the observed truth rate (with a
    credible band) at each stated-confidence level. The reliability diagram, but
    with honest uncertainty where data is thin."""

    def __init__(self, n_bins: int = 10, alpha0: float = 1.0, beta0: float = 1.0):
        self.n_bins = n_bins
        self.bins = [BetaBinomial(alpha0, beta0) for _ in range(n_bins)]

    def _idx(self, p: float) -> int:
        return min(int(p * self.n_bins), self.n_bins - 1)

    def observe(self, probs: Sequence[float], outcomes: Sequence) -> "BetaBinomialCalibration":
        for p, y in zip(probs, outcomes):
            self.bins[self._idx(float(p))].update(bool(y))
        return self

    def curve(self, level: float = 0.95) -> List[dict]:
        """Per non-empty bin: nominal center, count, posterior truth rate, credible band."""
        out = []
        for b in range(self.n_bins):
            bb = self.bins[b]
            if bb.n == 0:
                continue
            lo, hi = bb.credible_interval(level)
            out.append({
                "bin_center": (b + 0.5) / self.n_bins,
                "n": int(bb.n),
                "truth_rate": bb.mean,
                "ci_low": lo,
                "ci_high": hi,
            })
        return out
