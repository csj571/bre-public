# The showcase — 2008 and COVID, replayed

A zero-build page that streams real market data through the engine's changepoint
detector one trading day at a time, and scores the result against crisis dates
that were registered before the data was ever loaded.

```bash
python3 -m http.server        # FROM THE REPOSITORY ROOT
# then open http://localhost:8000/showcase/
```

Serve the **repository root**, not this directory: the page fetches the CSVs in
`validation/markets/data/`, and ES-module imports need HTTP anyway (`file://`
will not work).

## What it shows

| Series | Break | Detection latency |
|---|---|---|
| S&P 500 daily log returns, 2007–2009 | Lehman, 2008-09-15 | **0 trading days** |
| S&P 500 daily log returns, 2019–2020 | COVID, 2020-02-20 | **2 trading days** |
| CBOE VIX daily close, 2007–2021 | both, in one unbroken run | **0 and 2 trading days** |
| 21-day realized vol, 2019–2020 *(control)* | COVID | **missed** |

Three stacked panels move together: the observed series, the **run-length
posterior mode** (how long the current regime has lasted), and **P(change | data)**.
The story is the middle panel — a long ramp that collapses when the world changes.

## Files

| File | Contents |
|---|---|
| `replay.js` | the computation: CSV parsing, z-scoring, the BOCPD sweep, latency scoring. No DOM. |
| `cases.js` | which series get replayed and which onsets they are scored against |
| `showcase.js` | rendering, playback, transport, readout |
| `showcase.css` | self-contained styling (borrows the simulator's palette, depends on none of its CSS) |

## Why you can trust the numbers on the page

The page runs the JavaScript BOCPD from `../sim/signal.js`; the published,
validated numbers come from `../engine/changepoint.py`. Those are two
independent implementations, so `tests/test_js_python_parity.py` runs both over
every vendored series and asserts they flag **identical indices** — and
`tools/bocpd_flags.mjs` gives you the same comparison by hand:

```bash
node tools/bocpd_flags.mjs validation/markets/data/vix.csv
python validation/markets/validate_regimes.py
```

One configured difference: `replay.js` pins the JS run-length truncation to 300
(`PY_MAX_RUN`), the Python engine's default. The simulator ships 200 because it
never sees a series this long, and the truncation changes the posterior tail.

## What this is not

It is **detection, not prediction** — every flag lands on or after the onset.
It is not a trading signal and not a backtest. And the flag stream is a heuristic
layer over the run-length posterior: if you build on this, consume the posterior.
The scope limits are spelled out in
[`../validation/markets/results/regime_validation.md`](../validation/markets/results/regime_validation.md)
and [`../validation/markets/results/robustness_checks.md`](../validation/markets/results/robustness_checks.md).
