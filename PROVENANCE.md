# Provenance

## Where this came from

This repository is an extraction from a private research monorepo. Three things
came across: the Bayesian engine, the reference simulator, and the markets
regime-detection validation that scores the engine against dated structural
breaks. Nothing here references the private tree, and nothing here depends on it.

| Here | Upstream | Relationship |
|---|---|---|
| `engine/*.py` | `engine/` | verbatim copies, except one renamed module (below) |
| `sim/*.js`, `sim/index.html`, `sim/style.css`, `sim/test.mjs` | `bre1-simulator/` | verbatim copies |
| `validation/markets/*.py` | `products/markets/` | verbatim except the paths in docstrings |
| `validation/markets/data/`, `validation/markets/results/` | same | verbatim |
| `tests/` | `tests/` | subset; the torch and TruthfulQA cases are dropped (below) |
| `showcase/`, `tools/bocpd_flags.mjs` | — | written for this repository |
| `index.html`, `README.md`, `PROVENANCE.md`, `.github/`, `pyproject.toml` | — | written for this repository |
| `tests/test_regime_detection.py`, `test_js_python_parity.py`, `test_data_integrity.py` | — | written for this repository |

## The one renamed module

The whole engine is here — all nine primitives. One of them changed name on the
way across:

| Upstream | Here |
|---|---|
| `engine/somatic_bayesian.py` | `engine/preference.py` |
| `SomaticBayesianEngine` | `BayesianPreferenceLearner` |
| `tests/test_somatic_bayesian.py` | `tests/test_preference.py` |

"Somatic" is a leftover from an earlier framing of the project (somatic-marker
theory, the physiological limb that is permanently descoped — see the non-claims
in the README). The class never had anything to do with physiology: it is an
online Bayesian preference learner over a linear reward model, Bradley-Terry
likelihood, Laplace posterior. The public name says that.

The rename is the only edit — the mathematics, the API surface, the buffer
layout, and the Cholesky-with-jitter fallback are unchanged, and no compatibility
alias is provided because the old name was never public. `engine/__init__.py`
exports it lazily under the new name, so torch is still an optional extra rather
than a dependency: `pip install "bre-engine[torch]"`.

## What was deliberately left behind

- **The RLHF training study** (`products/alignment/`) — trainers, the synthetic
  sequence environment, run logs, and the study dashboard. The preference learner
  those trainers consume *is* here; the study that exercises it is not, so
  nothing in this repo claims a validated result for it.
- **The focus-tracker service** (`products/health/`) — a FastAPI app consuming
  the GP.
- **The TruthfulQA calibration eval** — the vendored prediction slice and its
  harness, along with the one test in `tests/test_registry.py` that scored the
  Brier gate against those labels. The calibration module itself
  (`engine/calibration.py`) is here and fully tested.
- **The spec PDFs and internal planning docs** — design documents and cross-repo
  roadmaps, none of which are needed to run or audit anything in this tree.

## Dangling roadmap references

The engine, harness and test docstrings cite the research repo's internal
roadmap IDs — "BUILD_PLAN B5", "FIX_PLAN P2.4" and similar. Those planning
documents are not shipped here, and the references are left in place on purpose:
these files are byte-identical copies, and the parity/drift guards below are
only meaningful while that stays true. Read them as provenance markers, not as
pointers to something missing.

## Data

Both series are real and externally sourced. Neither was generated here — the
whole point of the validation is that the ground truth is not self-produced.

- **`vix.csv`** — CBOE VIX daily closes, 2007-06-01 → 2021-01-29, from the
  `datasets/finance-vix` published dataset (ODC-PDDL packaging; the underlying
  index data originates from CBOE). Copied as published.
- **`sp500_logret_*.csv`, `sp500_rv21_*.csv`** — *derived* from real S&P 500
  daily closes (1990–2022) shipped inside the `skfolio` package. `fetch_data.py`
  re-derives them, checking four closes against independently documented
  historical values before it writes anything, and
  `tests/test_data_integrity.py` byte-compares the committed files against a
  fresh derivation.

Full provenance and the honest reading of the false-alarm numbers:
[`validation/markets/data/README.md`](validation/markets/data/README.md).

## Keeping copies honest

The engine and simulator files are copies, and copies drift. Two guards run in CI:

- `tests/test_js_python_parity.py` — the JavaScript BOCPD and the Python BOCPD
  must flag identical indices on every vendored series. This catches drift
  between `sim/signal.js` and `engine/changepoint.py` in either direction.
- `tests/test_regime_detection.py` — the published latencies (0 days, 2 days)
  and the negative control's miss are asserted directly, so a change to the
  engine that would alter the headline fails the build instead of quietly
  restating it.

Edits to `sim/` should be made upstream and re-copied; edits made here will drift
from the canonical simulator line.

## None of the mathematics is novel

Kriging/GP regression dates to the 1950s, the Kalman filter to 1960, the Brier
score to 1950, BOCPD to Adams & MacKay 2007, BALD to Houlsby et al. 2011. The
contribution is a faithful, inspectable, CPU-scale synthesis with per-domain
external validation and explicitly written-down non-claims.
