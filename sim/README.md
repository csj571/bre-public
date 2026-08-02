# The reference simulator

A **zero-build** browser simulator for the whole Bayesian decision stack — plain
ES modules, no bundler, no tracking. The only external request it makes is a
webfont; block it and the page falls back to system fonts and works offline.
(The [showcase](../showcase/) makes no external requests at all.)

```bash
python3 -m http.server        # then open http://localhost:8000/sim/
node test.mjs                 # 67 golden-value tests
```

ES-module imports need HTTP, so serve it rather than opening `index.html` over
`file://`. The tests need Node ≥ 18; the simulator itself needs only a browser.

Looking for the 2008 / COVID result on real data? That is the
[showcase](../showcase/), not this page. This one runs on generated scenarios.

## What to look at

Drive a regime shift and watch three things move together:

1. **The run-length posterior** spikes as BOCPD loses confidence in the current
   regime (`signal.js`).
2. **The epistemic/aleatoric split** separates — reducible ignorance vs
   irreducible noise (`entropy.js`).
3. **The policy gate** moves ACT → QUERY MORE → DEFER, abstaining rather than
   bluffing (`modes.js`).

That third transition is the argument in one animation: the system says "I don't
know" *before* it is wrong, because the uncertainty decomposition told it to.

## Modules

| File | Contents |
|---|---|
| `gp.js` | Gaussian process math — Cholesky with recursive jitter, kernels, the LMC multi-output GP, log-space Metropolis–Hastings MCMC over hyperparameters + Gelman–Rubin |
| `signal.js` | Adaptive Kalman filter + BOCPD changepoint detection |
| `entropy.js` | Predictive entropy and the BALD epistemic/aleatoric decomposition |
| `acquisition.js` | EI, UCB, BALD, and MES (Max-value Entropy Search, Wang & Jegelka 2017) |
| `registry.js` | Prior registry with versioning and semantic seeding |
| `coupling.js` | Cross-signal coupling for the Continuity Layer |
| `diagnostics.js` | Importance-weighted hyperparameter-ensemble diagnostics |
| `modes.js` | The four scenario modes: axis labels, slider semantics, decision verbs, kernel defaults, scenario generators (deterministic via Mulberry32) |
| `main.js` | UI controller and orchestration |

The numerics match the Python `../engine/` — recursive-jitter Cholesky, prior
std = `eta`. The one deliberate difference: this copy has **no cosine kernels**,
which the Python GP does.

`signal.js`'s BOCPD is the detector the [showcase](../showcase/) drives, and
`tests/test_js_python_parity.py` holds it byte-for-flag identical to
`engine/changepoint.py` on every series in this repo.

## Read this before believing a mode

The four modes are **vocabulary skins over one unchanged engine**. Swapping modes
retargets axis labels, slider semantics, decision verbs, and the scenario stream —
the GP, Kalman, BOCPD, entropy decomposition, and promotion gate underneath are
identical. That is the point: one engine, several dialects.

It is **not** a claim that all four domains are validated. They are not:

| Mode | Status |
|---|---|
| **Trading & Sharpe** | The domain with external validation behind it — but the *validated* result is the Python harness in [`../validation/markets/`](../validation/markets/) scored against dated structural breaks, **not** anything computed in this browser. Nothing here is a trading signal or a backtest. |
| **Glidepath** | Illustrative. Synthetic multi-decade wealth scenario; no external validation. |
| **Health & Longevity** | **Synthetic scenario skin for demonstration only.** The physiological→state mapping this mode dramatizes is permanently descoped from validation claims — the 2024–25 literature shows it is non-stationary within a person, and HRV indexes coarse autonomic arousal rather than discrete states, so no cheap ground truth exists. Do not read this mode as state detection. |
| **Continuity Layer** | Illustrative. Demonstrates the coupling design; no external validation. |

**Everything in this simulator runs on generated scenarios.** Data you drive
through it is synthetic by construction, so no run in this browser is evidence
about anything. The external-truth result is in
[`../validation/markets/`](../validation/markets/), and the showcase replays it.
