"""Bayesian Online Changepoint Detection (Adams & MacKay 2007) — the Python
counterpart of the JS `BOCPD` in sim/signal.js.

Normal-Inverse-Gamma conjugate, Student-t posterior predictive, constant hazard
H = 1/lambda. Maintains the run-length posterior online and flags regime
transitions. Pure Python (stdlib `math` for lgamma) — no scipy.

Also exposes `changepoint_scores()` (detection latency + false-alarm rate) for
validating flags against KNOWN breakpoints — the external-truth metric. Note the
discrete trigger is a heuristic layer (named constants) over the principled
run-length posterior, which is `R` / `p0`; consume the posterior, not the flags.
"""
import math
from typing import List, Sequence


def _student_t_logpdf(x, mu, kappa, alpha, beta):
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    scale = math.sqrt(scale2)
    z = (x - mu) / scale
    return (math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi) - math.log(scale)
            - ((df + 1.0) / 2.0) * math.log(1.0 + z * z / df))


class BOCPD:
    def __init__(self, hazard_lambda: float = 50.0,
                 mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0,
                 max_run: int = 300,
                 change_prob_threshold: float = 0.5, peak_run_min: int = 6,
                 collapse_run_max: int = 2, refractory: int = 8):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.max_run = max_run
        # heuristic discrete-trigger tunables (the run-length posterior itself is exact)
        self.change_prob_threshold = change_prob_threshold
        self.peak_run_min = peak_run_min
        self.collapse_run_max = collapse_run_max
        self.refractory = refractory
        self.reset()

    def reset(self) -> None:
        self.mu = [self.mu0]
        self.kappa = [self.kappa0]
        self.alpha = [self.alpha0]
        self.beta = [self.beta0]
        self.R = [1.0]                 # run-length posterior
        self.history = []              # (mode_run, p0) per step
        self._refractory_left = 0

    def update(self, x: float) -> dict:
        x = float(x)
        T = len(self.R)
        pred = [math.exp(_student_t_logpdf(x, self.mu[r], self.kappa[r], self.alpha[r], self.beta[r]))
                for r in range(T)]
        # message passing: growth vs changepoint
        newR = [0.0] * (T + 1)
        change_p = 0.0
        for r in range(T):
            newR[r + 1] = self.R[r] * pred[r] * (1.0 - self.hazard)
            change_p += self.R[r] * pred[r] * self.hazard
        newR[0] = change_p
        s = sum(newR)
        if s > 0:
            newR = [v / s for v in newR]
        # NIG sufficient-statistic updates (prepend the prior for run-length 0)
        newMu, newKappa, newAlpha, newBeta = [self.mu0], [self.kappa0], [self.alpha0], [self.beta0]
        for r in range(T):
            k, m, a, b = self.kappa[r], self.mu[r], self.alpha[r], self.beta[r]
            newKappa.append(k + 1.0)
            newMu.append((k * m + x) / (k + 1.0))
            newAlpha.append(a + 0.5)
            newBeta.append(b + (k * (x - m) * (x - m)) / (2.0 * (k + 1.0)))
        self.mu, self.kappa, self.alpha, self.beta, self.R = newMu, newKappa, newAlpha, newBeta, newR
        # truncate run-length support
        if len(self.R) > self.max_run:
            self.R = self.R[:self.max_run]
            self.mu = self.mu[:self.max_run]
            self.kappa = self.kappa[:self.max_run]
            self.alpha = self.alpha[:self.max_run]
            self.beta = self.beta[:self.max_run]
            ss = sum(self.R)
            if ss > 0:
                self.R = [v / ss for v in self.R]
        # mode run length
        mode_run, mode_p = 0, 0.0
        for r, pr in enumerate(self.R):
            if pr > mode_p:
                mode_p, mode_run = pr, r
        self.history.append((mode_run, self.R[0]))
        if len(self.history) > 200:
            self.history.pop(0)
        # discrete trigger (heuristic): high changepoint mass, or a run-length-mode collapse
        fired = self.R[0] > self.change_prob_threshold
        if not fired and len(self.history) > 6:
            recent = self.history[-8:]
            peak = max(h[0] for h in recent)
            if peak >= self.peak_run_min and mode_run <= self.collapse_run_max:
                fired = True
        if self._refractory_left > 0:
            self._refractory_left -= 1
            fired = False
        if fired:
            self._refractory_left = self.refractory
        return {"p0": self.R[0], "mode_run": mode_run, "changepoint": fired}


def detect_changepoints(series: Sequence[float], **kwargs) -> List[int]:
    """Run BOCPD over a 1-D series; return the indices where a changepoint fired."""
    bocpd = BOCPD(**kwargs)
    return [i for i, x in enumerate(series) if bocpd.update(x)["changepoint"]]


def changepoint_scores(flagged: Sequence[int], true_points: Sequence[int], tol: int) -> dict:
    """Validate detected changepoints against KNOWN breakpoints (markets anchor).

    detection latency = steps from each true break to the first flag in [t, t+tol]
    (None if missed); false-alarm rate = fraction of flags not attributable to any
    true break within tol. This is the external-truth metric — only meaningful when
    `true_points` are real dated breaks, not self-generated.
    """
    flagged = sorted(flagged)
    true_points = sorted(true_points)
    latencies = []
    for t in true_points:
        hit = next((f for f in flagged if t <= f <= t + tol), None)
        latencies.append(None if hit is None else hit - t)
    n_false = sum(1 for f in flagged if not any(t <= f <= t + tol for t in true_points))
    return {
        "latencies": latencies,
        "n_detected": sum(1 for l in latencies if l is not None),
        "n_true": len(true_points),
        "false_alarms": n_false,
        "false_alarm_rate": (n_false / len(flagged)) if flagged else 0.0,
    }
