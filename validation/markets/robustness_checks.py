"""E1 robustness checks — the committed script behind results/robustness_checks.md.

Reproduces every table in that document from the same integrity-gated S&P 500
series `fetch_data.py` uses (no CSVs needed on disk; the series is loaded
directly). Four sections, matching the doc:

  A. Expanded pre-registered 9-break truth set over the full 1990-2022 series
     (globally z-scored daily log returns), hazard lambda 50 and 250. Strict
     scoring credits flags in [t, t+10] trading days; the disclosed "lead"
     window additionally credits [t-5, t+10].
  B. Calm-period negative controls (2004-06, 2013-14), z-scored per window.
  C. Hazard-rate sensitivity on the two crisis windows (lambda 25/50/100/250),
     z-scored per window — the same windowing as validate_regimes.py.
  D. Shuffled-returns placebo: the full series shuffled with
     random.Random(seed) for seeds 0..4 (lambda=50) and 0..2 (lambda=250),
     z-scored after shuffling.

Run from the repo root (needs skfolio installed for the data, nothing else):

    python validation/markets/robustness_checks.py
"""
import math
import os
import random
import statistics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine.changepoint import detect_changepoints  # noqa: E402
from fetch_data import load_sp500                   # noqa: E402

# Nine sharply-dateable onsets from standard market chronology, registered
# before the full-series run (see robustness_checks.md §A).
BREAKS = {
    "Asian crisis": "1997-10-27",
    "Russia/LTCM": "1998-08-17",
    "9/11 reopen": "2001-09-17",
    "Lehman": "2008-09-15",
    "US downgrade": "2011-08-08",
    "Yuan flash crash": "2015-08-24",
    "Volmageddon": "2018-02-05",
    "COVID": "2020-02-20",
    "Ukraine": "2022-02-24",
}

CALM_WINDOWS = (("2004-01-01", "2006-12-31"), ("2013-01-01", "2014-12-31"))
CRISIS_WINDOWS = (("Lehman", ("2007-01-01", "2009-12-31"), "2008-09-15"),
                  ("COVID", ("2019-01-01", "2020-12-31"), "2020-02-20"))
STRICT_TOL = 10          # trading days, [t, t+10]
LEAD = 5                 # disclosed lead window, [t-5, t+10]
PLACEBO_SEEDS = {50: range(5), 250: range(3)}


def zscore(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) or 1.0
    return [(x - m) / s for x in xs]


def first_index_at_or_after(dates, target):
    for i, d in enumerate(dates):
        if d >= target:
            return i
    return None


def main():
    dates, values = load_sp500()
    ret_dates = dates[1:]
    logret = [math.log(values[i + 1] / values[i]) for i in range(len(values) - 1)]
    years = len(logret) / 252.0

    print("=== A. Full-series 9-break truth set (globally z-scored log returns) ===")
    full_z = zscore(logret)
    for lam in (50, 250):
        flags = sorted(detect_changepoints(full_z, hazard_lambda=lam))
        attributable = set()
        strict_hits = lead_hits = 0
        print(f"-- lambda={lam}: total flags {len(flags)} ({len(flags) / years:.1f}/yr)")
        for name, onset in BREAKS.items():
            t = first_index_at_or_after(ret_dates, onset)
            strict = next((f for f in flags if t <= f <= t + STRICT_TOL), None)
            lead = next((f for f in flags if t - LEAD <= f <= t + STRICT_TOL), None)
            strict_hits += strict is not None
            lead_hits += lead is not None
            attributable.update(f for f in flags if t - LEAD <= f <= t + STRICT_TOL)
            s = f"+{strict - t}" if strict is not None else "miss"
            l = f"{lead - t:+d}" if lead is not None else "miss"
            print(f"   {name:18s} {onset}  strict={s:5s} lead={l}")
        print(f"   strict {strict_hits}/9, lead {lead_hits}/9, "
              f"flags attributable to named breaks: {len(attributable)}")

    print("=== B. Calm-period negative controls (per-window z-score) ===")
    for lo, hi in CALM_WINDOWS:
        idx = [i for i, d in enumerate(ret_dates) if lo <= d <= hi]
        window = zscore([logret[i] for i in idx])
        wyears = len(idx) / 252.0
        for lam in (50, 250):
            n = len(detect_changepoints(window, hazard_lambda=lam))
            print(f"   {lo} -> {hi}  lambda={lam}: {n} flags ({n / wyears:.1f}/yr)")

    print("=== C. Hazard sweep on crisis windows (per-window z-score) ===")
    for name, (lo, hi), onset in CRISIS_WINDOWS:
        idx = [i for i, d in enumerate(ret_dates) if lo <= d <= hi]
        wdates = [ret_dates[i] for i in idx]
        window = zscore([logret[i] for i in idx])
        t = first_index_at_or_after(wdates, onset)
        for lam in (25, 50, 100, 250):
            flags = sorted(detect_changepoints(window, hazard_lambda=lam))
            hit = next((f for f in flags if t <= f <= t + STRICT_TOL), None)
            lat = hit - t if hit is not None else "miss"
            print(f"   {name} lambda={lam}: latency={lat}, window flags={len(flags)}")

    print("=== D. Shuffled-returns placebo (z-scored after shuffle) ===")
    for lam, seeds in PLACEBO_SEEDS.items():
        counts = []
        for seed in seeds:
            shuffled = logret[:]
            random.Random(seed).shuffle(shuffled)
            counts.append(len(detect_changepoints(zscore(shuffled), hazard_lambda=lam)))
        real = len(detect_changepoints(full_z, hazard_lambda=lam))
        print(f"   lambda={lam}: real {real} flags | shuffled (seeds {list(seeds)}): {counts}")


if __name__ == "__main__":
    main()
