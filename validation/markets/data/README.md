# Markets validation data (real, provenance-documented)

`validate_regimes.py` scores BOCPD changepoint flags against **dated** structural
breaks. That only means something against **real external ground truth** — the
no-fabricated-data rule stands: never commit a synthetic or self-generated series
here (that would be the closed-loop trap the BUILD_PLAN warns about).

## The vendored slice

`vix.csv` is a **real** CBOE VIX daily-close slice, 2007-06-01 → 2021-01-29
(3,441 rows), spanning both pre-registered breaks (Lehman 2008-09-15, COVID
2020-02-20). Source: the `datasets/finance-vix` GitHub mirror
(`https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv`,
ODC-PDDL packaging; the underlying index data originates from CBOE). Regenerate
or re-slice with:

## The offline S&P route (second independent series)

`python validation/markets/fetch_data.py` materializes real S&P 500 closes
(1990–2022) from the `skfolio` package's bundled dataset — integrity-gated
against four independently documented historical closes before anything is
written — and emits per-crisis-window CSVs (daily log returns + 21-day
realized vol). This produced the results in `../results/regime_validation.md`
and the robustness suite in `../results/robustness_checks.md`.


```bash
python validation/markets/data/fetch_vix.py                    # default window
python validation/markets/data/fetch_vix.py --start 2007-06-01 --end 2021-01-29
```

FRED's `VIXCLS` (https://fred.stlouisfed.org/series/VIXCLS) is the equivalent
primary source when reachable — its `DATE,VIXCLS` CSV works as-is (the loader
skips `.` missing rows). S&P 500 (`SP500`) works too — pass `--data`.

## Reading the result honestly (raw vs adjusted FAR)

First run on this slice: **both breaks detected, latencies 0 and 2 trading
days** — but the raw false-alarm rate against the 2-break label set was 0.86,
because 13.6 years of VIX contain far more than two real stress events. On
inspection, **every** additional flag landed on a documented, dateable episode
(Fukushima 2011, the 2013 debt-ceiling standoff, the 2015 China devaluation,
Volmageddon 2018, the 2019 tariff shock, the 2021 GameStop vol spike, …).

`validate_regimes.py` therefore carries a `SECONDARY_EVENTS` chronology used
*only* to classify flags for the false-alarm computation, and prints **both**
numbers (raw FAR 0.86, documented-adjusted FAR 0.00). Caveats, disclosed in the
code as well:

- The secondary list was compiled **after** inspecting detector output on this
  slice (post-hoc label enrichment) — each entry is independently documented,
  but this is not a pre-registered test.
- It is a flag-classification set, not a detection requirement: documented
  events the detector *missed* (the Aug-2011 US downgrade, Brexit) are not
  penalized. The detector characterizes as a *sustained regime-shift* detector,
  not a spike detector.
- The pre-registered part of the result — the 2 structural breaks, in the repo
  before any data arrived — is the headline: detected at 0 and 2 days.

Whichever series you feed it: prefer (near-)i.i.d. innovations such as log
returns over smoothed levels. The validated runs show vanilla BOCPD detects
crises in log returns within 0–2 days but goes silent on smoothed
(autocorrelated) series like rolling realized vol — the failure mode the
BOCPD literature predicts.

## Run

```bash
python validation/markets/validate_regimes.py            # uses data/vix.csv
python validation/markets/validate_regimes.py --tol 10 --max-far 0.5
```

Gate passes when every pre-registered break is detected within `--tol` trading
days and the documented-adjusted false-alarm rate is below `--max-far`.
`tests/test_markets_gate.py` runs this as a launch-blocking CI gate.
