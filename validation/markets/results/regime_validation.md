# Regime validation on real S&P 500 data — first external-truth result

*Run: 2026-07-03, from the repo root, with the **unmodified** harness
(`validate_regimes.py`, BOCPD defaults: hazard λ=50, NIG prior (0,1,1,1),
trigger threshold 0.5, tol=10 trading days, FAR gate < 0.5).*

## Data provenance

Real S&P 500 daily closes, 1990-01-02 → 2022-12-28 (8,313 rows), from the
`skfolio` package's bundled dataset (shipped inside the PyPI wheel — no network
access needed). Integrity-gated before use: four closes verified exactly against
independently documented values (Lehman day 1192.70; GFC low 752.44 on
2008-11-20; pre-COVID peak 3386.15 on 2020-02-19; COVID low 2237.40 on
2020-03-23). Reproduce with `python validation/markets/fetch_data.py`.

Two input transforms, scored per crisis window:

- **daily log returns** — approximately i.i.d. within a regime, matching the
  BOCPD model assumption; a volatility regime shift is a variance changepoint.
- **21-day realized volatility (annualized)** — a VIX-like smoothed series that
  deliberately **violates** the i.i.d.-within-regime assumption (strong
  autocorrelation). Included as a negative control.

## Results

| window | series | breaks detected | latency (trading days) | flags | nominal FAR | gate |
|---|---|---|---|---|---|---|
| GFC 2007–2009 | log returns | 1/1 | **0** | 5 | 0.80 | FAIL (FAR) |
| COVID 2019–2020 | log returns | 1/1 | **2** | 6 | 0.83 | FAIL (FAR) |
| GFC 2007–2009 | realized vol | 1/1 | 3 | 1 | 0.00 | **PASS** |
| COVID 2019–2020 | realized vol | 0/1 | — | 0 | 0.00 | FAIL (missed) |

**Detection headline:** on near-i.i.d. input (log returns) the engine's vanilla
Adams-MacKay BOCPD detected both documented structural breaks essentially
immediately — Lehman flagged **on the day** (2008-09-15), the COVID crash
flagged 2 trading days after the 2020-02-20 onset.

## The "false alarms" are not noise (post-hoc audit)

The FAR gate fails because the pre-registered truth set contains only two
crises, while each multi-year window contains other genuine regime events.
Every extra flag coincides with a documented, dateable market event
(annotation done after the run — this is a post-hoc audit, not a re-scoring):

| flag date | documented event |
|---|---|
| 2007-02-27 | Feb-2007 global selloff (Shanghai −8.8%, Dow −416, record one-day VIX jump) |
| 2008-03-11 | Bear Stearns collapse week (Fed TSLF announced 3/11; rescue 3/14–16) |
| 2008-09-15 | **Lehman Brothers bankruptcy — scored break, latency 0** |
| 2009-03-10 | bear-market bottom (3/9 low; +6.4% rally on 3/10) |
| 2009-10-01 | October-2009 correction onset (−2.6% day) |
| 2019-05-13 | US–China tariff retaliation selloff (−2.4%) |
| 2019-08-05 | yuan breaks 7, currency-manipulator designation (−3.0%) |
| 2020-01-27 | first COVID-19 selloff (−1.6%, worst day in months) |
| 2020-02-24 | **COVID-19 crash — scored break onset 2/20, latency 2 days** |
| 2020-06-11 | second-wave scare (−5.9%, worst day since March 2020) |
| 2020-09-03 | September-2020 tech unwind (−3.5%) |

11 of 11 flags land on documented events. The correct reading: the detector is
not noisy; the two-event truth set under-specifies "true regime changes" over a
multi-year window, so the nominal FAR overstates the false-alarm behavior. The
gate verdict is reported as-is rather than re-scored against an expanded,
flag-informed truth set (that would be circular).

## The negative control behaved as the literature predicts

On the smoothed realized-vol series the run-length posterior never spikes
during COVID (max changepoint mass p0 ≈ 0.02 in Feb–Apr 2020): the strong
autocorrelation of a rolling-window series violates the i.i.d.-within-regime
assumption, and vanilla BOCPD absorbs the slow rise instead of flagging it.
This is exactly the documented weakness of unmodified Adams-MacKay on
autocorrelated financial series (see e.g. score-driven / autoregressive BOCPD
extensions: *Quantitative Finance* 2024, doi:10.1080/14697688.2024.2337300;
arXiv:2407.16376). Feed the detector (near-)i.i.d. innovations, not smoothed
levels. The lone GFC realized-vol pass (single flag, 2008-09-18) shows the
failure is COVID's fast-then-smooth profile interacting with autocorrelation,
not a broken implementation.

## Caveats

- Two scored breaks is a minimal truth set; the field standard is
  multi-annotator benchmarks (Turing Change Point Dataset, arXiv:2003.06222).
- Defaults only — no tuning of hazard rate, prior, or trigger heuristics was
  done for this run (tuning to pass the gate post-hoc would overfit it).
- The S&P series proxies VIX; the harness's documented breaks are onset dates,
  and "latency" is measured against those onsets.
