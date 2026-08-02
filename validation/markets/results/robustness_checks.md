# E1 robustness checks — expanded truth set, negative controls, sensitivity, placebo

*Run 2026-07-04, same data pipeline as `regime_validation.md` (real S&P 500
closes 1990–2022, integrity-gated; daily log returns, z-scored). These checks
were run to stress the headline E1 result; several of them **temper** it, and
they are reported in full.*

## A. Expanded pre-registered truth set (9 breaks, full 33-year series)

Nine sharply-dateable onsets chosen from standard market chronology *before*
this run (the only overlap with previously-observed flags is Lehman + COVID
from the window runs). Strict scoring credits flags in `[t, t+10]` trading
days; a disclosed "lead" window `[t−5, t+10]` additionally credits flags where
the detector fired on the selloff days immediately *preceding* the famous
date.

| break | onset | λ=50 strict | λ=50 lead | λ=250 strict | λ=250 lead |
|---|---|---|---|---|---|
| Asian crisis crash | 1997-10-27 | **+0** | +0 | **+0** | +0 |
| Russia default / LTCM | 1998-08-17 | miss | miss | miss | miss |
| 9/11 reopening | 2001-09-17 | miss | miss | miss | miss |
| GFC / Lehman | 2008-09-15 | **+0** | +0 | miss | miss |
| US downgrade Black Monday | 2011-08-08 | miss | miss | miss | miss |
| Yuan-deval flash crash | 2015-08-24 | miss | **−2** | miss | −1 |
| Volmageddon | 2018-02-05 | miss | **−5** | **+0** | +0 |
| COVID-19 crash | 2020-02-20 | **+2** | +2 | **+2** | +2 |
| Russia invades Ukraine | 2022-02-24 | miss | miss | miss | miss |

Totals: λ=50 → 131 flags over 33 years (4.0/yr), 3/9 strict detections (5/9
with lead window), and only 5 of 131 flags attributable to the nine named
breaks. λ=250 → 38 flags (1.2/yr), 3/9 strict — and **loses Lehman** in the
full-series context (a flag pattern that window runs at the same λ do not
show; long-history posterior adaptation and global z-scoring change behavior).

**Interpretation (post-hoc, labeled as such).** The misses are not random:
they cluster where the market was *already inside* a stressed regime —
Russia/LTCM during the Asian-crisis aftermath, 9/11 during the dot-com bear,
Aug-2011 after the July selloff, Feb-2022 inside the 2022 inflation-bear vol
regime. A run-length detector flags **calm→turbulent transitions**; it is
structurally unable to flag a famous event that does not change the local
statistical regime. The detected set (1997, 2008, 2015, 2018, 2020) is
consistent with that reading. The lead-window hits are also informative: for
2015 and 2018 the detector fired on the actual first selloff days
(2015-08-20, 2018-01-29), which precede the canonical "crash day."

## B. Calm-period negative controls

| window | λ=50 flags | λ=250 flags |
|---|---|---|
| 2004-01-01 → 2006-12-31 | **18** (6.0/yr) | 5 (1.7/yr) |
| 2013-01-01 → 2014-12-31 | **10** (5.0/yr) | 3 (1.5/yr) |

At default hazard the detector is **chatty in calm markets** — it fires more
often per year in flat 2004–06 than inside the GFC window. (Mechanism: within
a calm window, z-scoring amplifies small vol wiggles, and the NIG posterior
tightens enough that modest surprises carry changepoint mass.) Raising the
hazard prior to λ=250 quiets it by ~4× while preserving the crisis-window
latencies (§C).

## C. Hazard-rate sensitivity (crisis windows)

| break (window run) | λ=25 | λ=50 | λ=100 | λ=250 |
|---|---|---|---|---|
| Lehman latency | miss* | **0** | **0** | **0** |
| Lehman window total flags | 26 | 5 | 3 | 2 |
| COVID latency | 2 | **2** | **2** | **2** |
| COVID window total flags | 10 | 6 | 5 | 4 |

The headline latencies are **robust across λ = 50–250**; λ=25 floods the
refractory period and breaks detection. Fewer, cleaner flags at higher λ come
at no latency cost *within crisis windows* — but see §A for the full-series
λ=250 Lehman miss: hazard tuning does not transfer blindly across contexts.

*\*λ=25 fires constantly; the refractory heuristic suppresses the Lehman-day
flag.*

## D. Shuffled-returns placebo

Shuffling the full return series destroys all regime structure (vol
clustering) while keeping the fat-tailed marginal distribution identical.

| series | λ=50 flags | λ=250 flags |
|---|---|---|
| real returns | 131 | 38 |
| shuffled (5 seeds, λ=50; 3 seeds, λ=250) | 188–213 | 80–89 |

The placebo does **not** go silent — shuffled i.i.d. fat-tailed data triggers
*more* flags than real data. Two conclusions: (i) a substantial fraction of
flags on any daily-return series are tail-outlier reactions, not regime
evidence — **an individual BOCPD flag is weak evidence**; (ii) real returns
producing ~35% fewer flags than their shuffled counterpart is itself evidence
of regime structure (volatility clustering makes tails *predictable* inside a
regime, suppressing false surprise).

## What survives, what changes

**Survives:** the E1 headline — 0/2-trading-day detection latency at Lehman
and COVID in crisis windows — is robust to hazard choice (λ 50–250) and is
now supported against a 9-break pre-registered set for the calm→turbulent
transitions it can structurally detect (Asian crisis also at 0 days).

**Changes:** the detector at defaults is not a precision instrument on long
series — ~4 flags/yr, chatty in calm regimes, outlier-sensitive (placebo),
and structurally blind to breaks inside ongoing turbulence. The claim it
supports is *"fast detection of calm→turbulent regime transitions at
documented crisis onsets,"* **not** *"low-false-alarm event detection."* The
right consumption pattern is the run-length posterior as a feature (as
`policy_gate.py` consumes uncertainty) rather than the discrete flag stream.

Reproduce: `python validation/markets/robustness_checks.py` (the committed
script behind every table above; loads the same integrity-gated series as
`fetch_data.py`, no CSVs needed; placebo shuffle seeds 0–4 at λ=50 and 0–2 at
λ=250 are fixed in the script). Crisis-window headline numbers come from
`validate_regimes.py` as in `regime_validation.md`.
