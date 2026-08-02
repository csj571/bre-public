// main.js — BRE-1 Reference Simulator orchestration
import { Kernels, KERNEL_NAMES, KERNEL_LABELS, fitGP, predictGP, logEvidence } from './gp.js';
import { AdaptiveKalman, BOCPD } from './signal.js';
import { decompose } from './entropy.js';
import { expectedImprovement, upperConfidenceBound, bald, maxValueEntropySearch, argmax } from './acquisition.js';
import { PriorRegistry } from './registry.js';
import { hyperEnsemble, optimizeHyperparameters } from './diagnostics.js';
import { MODES, DEFAULT_MODE_ID, getModeById } from './modes.js';
import { LAYER_LABELS } from './coupling.js';

// ─────────── State ───────────
const state = {
  mode: null,                  // active mode object (set by applyMode)
  scenarioGen: null,           // active generator instance
  sessions: [],                // session log across mode swaps
  sessionIndex: 0,             // current session id
  running: false,
  tickRateHz: 2.0,
  obsX: [], obsY: [], obsSigma: [], obsRegime: [], obsT: [], obsSession: [],
  t: 0,
  xDomain: [0, 30],
  plotConfig: { xLabel: 'x', yLabel: 'y', xRange: [0, 30], yRange: [-2, 4], xUnits: '', yUnits: '' },
  testGrid: null,
  hp: { ell: 1.2, eta: 1.0, sigma_n: 0.3, period: 4 },
  kernelName: 'matern52',
  posterior: null,    // {mu, sigmaF, sigmaN_x}
  entropy: null,      // {Htotal, Halea, Hepi}
  acquisition: { fn: 'bald', surface: null, xstar: null, value: null },
  // Decision policy thresholds (entropy-based). See classifyAndLog for how they
  // gate act / defer / query.
  policy: {
    actEntropyMax: 0.4,    // act only if total entropy H_total ≤ this
    deferNoiseRatio: 0.55, // defer if aleatoric share H_alea/H_total ≥ this
    queryEigMin: 0.4       // else query if epistemic entropy H_epi ≥ this
  },
  decisions: [],
  decisionCounts: { act: 0, defer: 0, query: 0 },
  changePoints: [],
  modelEvidence: {},
  // I2 hyperparameter-ensemble diagnostics (real; see diagnostics.js).
  diagnostics: { ess: null, entropyNorm: null, count: 16 },
  diagSeed: 1234,             // bumped by "Rerun diagnostics" to redraw the ensemble
  showTruth: false,
  hover: { x: null, idx: null },
  selectedNode: null,
  essHistory: [],             // ring buffer for the ESS sparkline
  // Dynamic epistemic-variance inflation on a detected regime change. The peak is
  // scaled by the change probability p0 (not a fixed constant) and decays over a
  // short window so the uncertainty band visibly "re-opens" then settles.
  epistemicInflation: 1.0,
  epistemicInflationPeak: 1.0,
  epistemicInflationStart: 0,
  epistemicInflationUntil: 0,
  // for D4 promotion gate — entropy-reduction basis (NOT log-MLL).
  // log_BF = max(H_epi_reference − H_epi_current, 0).
  // Reference resets on every promotion AND on every change-point detection so
  // the signal is always non-negative while the model is learning. Cumulative
  // log-MLL was the wrong basis: under fixed hyperparameters it isn't monotone
  // in N, so promotions fired on direction of drift rather than learning.
  logBayesFactor: 0,          // current log_BF (in nats)
  obsSinceLastPromo: 0,
  epistemicEntropyRef: null,  // H_epi at last reset; null until first step
  bfThreshold: Math.log(10),  // log scale (default BF ≈ 10)
  // hierarchical partial-pooling strength τ (0 = siloed, 1 = full pooling)
  poolingStrength: 0.5,
  // trace history
  traceHistory: [],
  // Continuity mode only: latest per-layer coupled-system state (null otherwise).
  layers: null,
};

// How strongly a detected change inflates epistemic variance: peak = 1 + GAIN·p0,
// where p0 = BOCPD P(change). A confident shift (p0≈0.8) ⇒ ~1.4×; a marginal one
// inflates less. Clamped so the band never balloons unboundedly.
const INFLATION_GAIN = 0.5;
const INFLATION_MAX = 1.6;
const INFLATION_WINDOW_MS = 800;

const registry = new PriorRegistry();
const kalman = new AdaptiveKalman();
// Tight prior so the r=0 predictive isn't artificially wide; BOCPD then detects real shifts.
const bocpd = new BOCPD({ lambda: 30, mu0: 0, kappa0: 5, alpha0: 2, beta0: 0.3 });

// ─────────── Mode-driven grid + observation σ estimate ───────────
// We no longer carry hand-rolled scenarios. The scenario generator (from
// modes.js) emits (x, y, sigma_obs, regime) per tick. For the GP grid we
// pull domain from state.plotConfig and estimate per-grid noise from the
// observed sigma stream (default to the mode's prior sigma_n).
function makeTestGrid() {
  const [a, b] = state.plotConfig.xRange;
  state.xDomain = [a, b];
  const N = 200;
  const grid = new Float64Array(N);
  for (let i = 0; i < N; i++) grid[i] = a + (b - a) * i / (N - 1);
  state.testGrid = grid;
}

// Estimate σ_obs(x) on the grid by kernel-smoothing the observed sigmas.
function estimateSigmaN(grid) {
  const N = grid.length;
  const out = new Float64Array(N);
  if (state.obsSigma.length === 0) {
    out.fill(state.hp.sigma_n);
    return out;
  }
  // 1-D Nadaraya-Watson smoother in x.
  const bw = Math.max(1e-3, (state.xDomain[1] - state.xDomain[0]) * 0.08);
  for (let i = 0; i < N; i++) {
    const xi = grid[i];
    let num = 0, den = 0;
    for (let j = 0; j < state.obsX.length; j++) {
      const d = (state.obsX[j] - xi) / bw;
      const w = Math.exp(-0.5 * d * d);
      num += w * state.obsSigma[j]; den += w;
    }
    out[i] = den > 1e-9 ? num / den : state.hp.sigma_n;
  }
  return out;
}

// ─────────── DOM refs ───────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const gpCanvas = $('#gp-canvas');
const entropyCanvas = $('#entropy-canvas');
const acqCanvas = $('#acq-canvas');
const stripCanvas = $('#strip-canvas');
const bocpdCanvas = $('#bocpd-canvas');
const traceCanvas = $('#trace-canvas');

// ─────────── Canvas DPR helpers ───────────
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.max(1, rect.width * dpr);
  c.height = Math.max(1, rect.height * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ─────────── GP Engine ───────────
function recomputePosterior() {
  const kfun = Kernels[state.kernelName];
  const grid = state.testGrid;
  const sigmaN = estimateSigmaN(grid);
  if (state.obsX.length === 0) {
    const N = grid.length;
    const muRange = state.plotConfig.yRange;
    const muMid = 0.5 * (muRange[0] + muRange[1]);
    const mu = new Float64Array(N).fill(muMid);
    const sigmaF = new Float64Array(N).fill(state.hp.eta);
    state.posterior = { mu, sigmaF, sigmaN, model: null };
    state.entropy = decompose(sigmaF, sigmaN);
    return;
  }
  const X = state.obsX;
  const y = state.obsY;
  const model = fitGP(X, y, kfun, state.hp);
  const { mu, sigmaF } = predictGP(model, grid);
  if (state.epistemicInflation > 1.0) {
    for (let i = 0; i < sigmaF.length; i++) sigmaF[i] *= state.epistemicInflation;
  }
  state.posterior = { mu, sigmaF, sigmaN, model };
  state.entropy = decompose(sigmaF, sigmaN);
}

function recomputeAcquisition() {
  if (!state.posterior) return;
  const { mu, sigmaF, sigmaN } = state.posterior;
  const sigmaY = new Float64Array(mu.length);
  for (let i = 0; i < mu.length; i++) sigmaY[i] = Math.sqrt(sigmaF[i] * sigmaF[i] + sigmaN[i] * sigmaN[i]);
  let surface;
  const fn = state.acquisition.fn;
  if (fn === 'ei') {
    const fBest = state.obsY.length ? Math.max(...state.obsY) : 0;
    surface = expectedImprovement(mu, sigmaY, fBest);
  } else if (fn === 'ucb') {
    surface = upperConfidenceBound(mu, sigmaY, 2.0);
  } else if (fn === 'mes') {
    surface = maxValueEntropySearch(mu, sigmaF);
  } else {
    surface = bald(state.entropy.Hepi);
  }
  const am = argmax(surface);
  state.acquisition.surface = surface;
  state.acquisition.xstar = state.testGrid[am.index];
  state.acquisition.value = am.value;
}

function recomputeModelEvidence() {
  if (state.obsX.length < 2) {
    state.modelEvidence = {};
    return;
  }
  const out = {};
  for (const name of KERNEL_NAMES) {
    out[name] = logEvidence(state.obsX, state.obsY, Kernels[name], state.hp);
  }
  state.modelEvidence = out;
}

// I2 diagnostics: importance-weighted hyperparameter ensemble (see diagnostics.js).
// Throttled like model evidence — it refits the GP many times, so we don't run it
// every tick. Pushes the real ESS into the sparkline ring buffer.
function recomputeDiagnostics() {
  if (state.obsX.length < 2) {
    state.diagnostics = { ess: null, entropyNorm: null, count: state.diagnostics.count };
    return;
  }
  const d = hyperEnsemble(state.obsX, state.obsY, state.kernelName, {
    count: state.diagnostics.count, seed: state.diagSeed
  });
  state.diagnostics = d;
  if (d.ess != null) {
    state.essHistory.push(d.ess);
    if (state.essHistory.length > 50) state.essHistory.shift();
  }
}

// ─────────── Tick ───────────
function tick() {
  if (!state.scenarioGen) return;
  const sample = state.scenarioGen.next();
  const { x, y, sigma_obs, regime } = sample;
  state.t += 1;
  state.obsX.push(x);
  state.obsY.push(y);
  state.obsSigma.push(sigma_obs);
  state.obsT.push(state.t);
  state.obsRegime.push(String(regime));
  state.obsSession.push(state.sessionIndex);

  // Continuity mode rides per-layer state on an additive `layers` field; the 1-D
  // engine above never touches it. Captured here so the coupling panel can render it.
  if (state.mode?.id === 'continuity' && sample.layers) state.layers = sample.layers;

  // S2: Kalman + BOCPD
  const k = kalman.update(y);
  const cp = bocpd.update(y);
  if (cp.changeFired) {
    state.changePoints.push({ t: state.t, x, p: cp.p0 });
    // Epistemic-variance inflation, scaled by how confident the detector is that a
    // change just happened (cp.p0 = P(change)); a stronger shift re-opens the band
    // further. Decays back to 1.0 over INFLATION_WINDOW_MS in loop().
    const peak = Math.min(INFLATION_MAX, 1 + INFLATION_GAIN * cp.p0);
    state.epistemicInflation = peak;
    state.epistemicInflationPeak = peak;
    state.epistemicInflationStart = performance.now();
    state.epistemicInflationUntil = state.epistemicInflationStart + INFLATION_WINDOW_MS;
    // broaden lengthscale prior (visible effect: slightly larger ell over time)
    // (the prior re-weights toward broader scales — we let the user move it, but we nudge)
    // No automatic change to ell, but flag in registry justification.
    flashRegimeFlag();
    // Reset the entropy-reduction reference: a new regime invalidates the
    // old baseline. Without this, log_BF would be measured against a
    // pre-shift posterior and the gate would fire spuriously.
    onChangePointDetected();
  }

  state.obsSinceLastPromo += 1;
  registry.tickDormancy(1);   // age dormant seeds; the current prior stays active
  recomputePosterior();
  recomputeAcquisition();

  // Recompute model evidence + ensemble diagnostics at low rate (both refit the GP).
  if (state.t % 5 === 0) { recomputeModelEvidence(); recomputeDiagnostics(); }

  // Decision: classify the most-uncertain or recommended point
  classifyAndLog();

  // D4 promotion check
  checkPromotion();

  updateInspectorReadouts();
  updateRegistry();
}

function classifyAndLog() {
  if (!state.entropy || !state.posterior) return;
  const { Htotal, Halea, Hepi } = state.entropy;
  // Pick the point with highest acquisition value
  let am = argmax(state.acquisition.surface);
  const i = am.index;
  const x = state.testGrid[i];
  const htot = Htotal[i], hale = Halea[i], hepi = Hepi[i];
  const noiseShare = htot > 0 ? hale / htot : 0;
  let cls;
  if (htot <= state.policy.actEntropyMax) cls = 'act';
  else if (noiseShare >= state.policy.deferNoiseRatio) cls = 'defer';
  else if (hepi >= state.policy.queryEigMin) cls = 'query';
  else cls = 'defer';
  state.decisionCounts[cls] += 1;
  const decision = {
    t: state.t,
    class: cls,
    x: round(x, 2),
    mu: round(state.posterior.mu[i], 3),
    sigmaF: round(state.posterior.sigmaF[i], 3),
    Htot: round(htot, 3),
    Hepi: round(hepi, 3),
    Hale: round(hale, 3),
    priorVersion: registry.current()?.id || 'v0'
  };
  state.decisions.push(decision);
  if (state.decisions.length > 300) state.decisions.shift();
  appendDecisionLogRow(decision);
}

function round(v, k) { const p = Math.pow(10, k); return Math.round(v * p) / p; }

function meanEpistemic() {
  if (!state.entropy || !state.entropy.Hepi) return null;
  const arr = state.entropy.Hepi;
  let s = 0, n = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) { s += v; n += 1; }
  }
  return n > 0 ? s / n : null;
}

function checkPromotion() {
  if (!state.posterior || state.obsX.length < 4) return;
  const Hepi = meanEpistemic();
  if (Hepi === null) return;

  // Initialise / hold reference until enough data has accumulated for a stable baseline.
  if (state.epistemicEntropyRef === null) {
    state.epistemicEntropyRef = Hepi;
    state.logBayesFactor = 0;
    $('#d4-evidence').textContent = '0.00';
    $('#d4-since').textContent = String(state.obsSinceLastPromo);
    return;
  }

  // log Bayes factor (entropy-reduction basis, nats). Clipped at 0 — we only
  // count learning, never forgetting.
  const logBF = Math.max(state.epistemicEntropyRef - Hepi, 0);
  state.logBayesFactor = logBF;

  // Minimum observations between promotions to avoid jitter.
  const enoughObs = state.obsSinceLastPromo >= 10;
  const overThreshold = logBF >= state.bfThreshold;
  // Regime change forces a promotion candidacy: a new regime is by definition
  // a new prior context. We still require enoughObs to avoid back-to-back promotions.
  const regimeForcing = state.changePoints.length > (registry.versions.length - 1);
  const ready = enoughObs && (overThreshold || regimeForcing);

  $('#d4-evidence').textContent = logBF.toFixed(2);
  $('#d4-since').textContent = String(state.obsSinceLastPromo);

  if (ready) {
    const currentRegime = state.obsRegime[state.obsRegime.length - 1] || null;
    // Semantic seeding: if a dormant seed already encodes this regime, germinate
    // it (retrieval-in-context) rather than minting a brand-new version.
    const dormant = registry.matchRegime(currentRegime);
    if (dormant) {
      registry.retrieve(dormant.id, state.t);
      state.epistemicEntropyRef = Hepi;
      state.obsSinceLastPromo = 0;
      state.logBayesFactor = 0;
      flashGate('retrieved');
      updateRegistry({ animateLast: true });
      return;
    }
    const snap = {
      kernel: state.kernelName,
      ell: state.hp.ell, eta: state.hp.eta, sigma_n: state.hp.sigma_n,
      priors: { ell: 'Gamma(2,1)', eta: 'HalfN(0,1)', sigma_n: 'HalfN(0,0.5)' }
    };
    const bfRatio = Math.exp(logBF);
    const regimeLabel = currentRegime ? currentRegime.charAt(0).toUpperCase() + currentRegime.slice(1) : 'Context';
    const name = `${regimeLabel}-regime prior`;
    const just = `Germination event after ${state.obsSinceLastPromo} observations; log BF ${logBF.toFixed(2)} nats (BF ≈ ${bfRatio.toFixed(1)}, epistemic-reduction basis).${regimeForcing ? ' Regime-shift forced.' : ''} Seeded for the “${currentRegime || 'context'}” regime.`;
    registry.promote(snap, just, 'BRE-1', {
      name,
      provenance: { source: 'D4 gate', regime: currentRegime, tick: state.t, parentId: registry.current()?.id || null },
      germinated: true
    });
    // Reset reference to current H_epi so the next promotion measures further learning.
    state.epistemicEntropyRef = Hepi;
    state.obsSinceLastPromo = 0;
    state.logBayesFactor = 0;
    flashGate('germinated');
    updateRegistry({ animateLast: true });
  }
}

// Reset the entropy reference whenever a change point fires — the new regime
// invalidates the old baseline. Called from the BOCPD detection branch.
function onChangePointDetected() {
  const Hepi = meanEpistemic();
  if (Hepi !== null) state.epistemicEntropyRef = Hepi;
  state.logBayesFactor = 0;
}

function flashRegimeFlag() {
  const el = $('#s2-flag');
  el.textContent = 'regime shift';
  el.classList.remove('fired'); void el.offsetWidth;
  el.classList.add('fired');
  setTimeout(() => { el.textContent = 'stationary'; el.classList.remove('fired'); }, 2000);
}
function flashGate(label = 'promoted') {
  const g = $('#d4-gate'); g.textContent = 'open · ' + label;
  g.classList.add('open');
  setTimeout(() => { g.textContent = 'closed'; g.classList.remove('open'); }, 1500);
}

// ─────────── Rendering ───────────
function plotAxes(ctx, w, h, padL, padR, padT, padB, xDom, yDom, opts = {}) {
  ctx.fillStyle = 'transparent';
  ctx.clearRect(0, 0, w, h);
  // Grid
  ctx.strokeStyle = getCssVar('--grid');
  ctx.lineWidth = 1;
  // X gridlines
  const numX = opts.xTicks || 6;
  for (let i = 0; i <= numX; i++) {
    const px = padL + (w - padL - padR) * i / numX;
    ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, h - padB); ctx.stroke();
  }
  const numY = opts.yTicks || 4;
  for (let i = 0; i <= numY; i++) {
    const py = padT + (h - padT - padB) * i / numY;
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
  }
  // Tick labels
  if (opts.labels !== false) {
    ctx.fillStyle = getCssVar('--text-faint');
    ctx.font = '10px "Geist Mono", monospace';
    ctx.textBaseline = 'top'; ctx.textAlign = 'center';
    for (let i = 0; i <= numX; i++) {
      const px = padL + (w - padL - padR) * i / numX;
      const xv = xDom[0] + (xDom[1] - xDom[0]) * i / numX;
      ctx.fillText(xv.toFixed(1), px, h - padB + 4);
    }
    ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    for (let i = 0; i <= numY; i++) {
      const py = padT + (h - padT - padB) * i / numY;
      const yv = yDom[1] - (yDom[1] - yDom[0]) * i / numY;
      ctx.fillText(yv.toFixed(1), padL - 6, py);
    }
  }
}

function toPx(v, lo, hi, lo_px, hi_px) {
  return lo_px + (hi_px - lo_px) * (v - lo) / (hi - lo);
}

const PADS = { L: 36, R: 14, T: 10, B: 24 };

function getYDomain() {
  // Compute global y-domain based on data and posterior, with mode-defined initial range.
  let ymin = state.plotConfig.yRange[0], ymax = state.plotConfig.yRange[1];
  if (state.obsY.length) {
    let lo = Infinity, hi = -Infinity;
    for (const y of state.obsY) { if (y < lo) lo = y; if (y > hi) hi = y; }
    if (state.posterior) {
      const { mu, sigmaF, sigmaN } = state.posterior;
      for (let i = 0; i < mu.length; i++) {
        const top = mu[i] + 1.96 * Math.sqrt(sigmaF[i]*sigmaF[i] + sigmaN[i]*sigmaN[i]);
        const bot = mu[i] - 1.96 * Math.sqrt(sigmaF[i]*sigmaF[i] + sigmaN[i]*sigmaN[i]);
        if (top > hi) hi = top;
        if (bot < lo) lo = bot;
      }
    }
    const pad = Math.max(0.3, (hi - lo) * 0.1);
    ymin = lo - pad; ymax = hi + pad;
  }
  return [ymin, ymax];
}

function renderGP() {
  const { ctx, w, h } = setupCanvas(gpCanvas);
  const xDom = state.xDomain;
  const yDom = getYDomain();
  plotAxes(ctx, w, h, PADS.L, PADS.R, PADS.T, PADS.B, xDom, yDom);

  if (!state.posterior) return;
  const { mu, sigmaF, sigmaN } = state.posterior;
  const grid = state.testGrid;
  const xpx = (x) => toPx(x, xDom[0], xDom[1], PADS.L, w - PADS.R);
  const ypx = (y) => toPx(y, yDom[1], yDom[0], PADS.T, h - PADS.B);

  // Aleatoric band (outermost, lightest)
  ctx.fillStyle = withAlpha(getCssVar('--posterior'), 0.10);
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) {
    const total = Math.sqrt(sigmaF[i]*sigmaF[i] + sigmaN[i]*sigmaN[i]);
    const top = mu[i] + 1.96 * total;
    ctx.lineTo(xpx(grid[i]), ypx(top));
  }
  for (let i = grid.length - 1; i >= 0; i--) {
    const total = Math.sqrt(sigmaF[i]*sigmaF[i] + sigmaN[i]*sigmaN[i]);
    const bot = mu[i] - 1.96 * total;
    ctx.lineTo(xpx(grid[i]), ypx(bot));
  }
  ctx.closePath(); ctx.fill();

  // Epistemic band (inner, darker)
  ctx.fillStyle = withAlpha(getCssVar('--posterior'), 0.25);
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) {
    const top = mu[i] + 1.96 * sigmaF[i];
    ctx.lineTo(xpx(grid[i]), ypx(top));
  }
  for (let i = grid.length - 1; i >= 0; i--) {
    const bot = mu[i] - 1.96 * sigmaF[i];
    ctx.lineTo(xpx(grid[i]), ypx(bot));
  }
  ctx.closePath(); ctx.fill();

  // True function overlay (optional). When mode-defined regimes have hard
  // boundaries (e.g. Glidepath regime A/B/C edges at 10/15 yr), draw faint
  // vertical guides; otherwise just trace the observation cloud trend.
  if (state.showTruth) {
    ctx.strokeStyle = withAlpha(getCssVar('--text'), 0.35);
    ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const modeId = state.mode?.id;
    if (modeId === 'glidepath') {
      for (const xb of [10, 15]) {
        const px = xpx(xb);
        ctx.beginPath(); ctx.moveTo(px, PADS.T); ctx.lineTo(px, h - PADS.B); ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  // Posterior mean
  ctx.strokeStyle = getCssVar('--posterior');
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) {
    const px = xpx(grid[i]), py = ypx(mu[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Change-point markers
  ctx.strokeStyle = getCssVar('--paradigm');
  ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 4]);
  for (const cp of state.changePoints) {
    const px = xpx(cp.x);
    ctx.beginPath(); ctx.moveTo(px, PADS.T); ctx.lineTo(px, h - PADS.B); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Observations
  for (let i = 0; i < state.obsX.length; i++) {
    const px = xpx(state.obsX[i]);
    const py = ypx(state.obsY[i]);
    const err = state.obsSigma[i];
    // error bar
    ctx.strokeStyle = withAlpha(getCssVar('--observation'), 0.5);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, ypx(state.obsY[i] - err)); ctx.lineTo(px, ypx(state.obsY[i] + err)); ctx.stroke();
    // point
    ctx.fillStyle = getCssVar('--observation');
    ctx.beginPath(); ctx.arc(px, py, 2.6, 0, Math.PI * 2); ctx.fill();
  }

  // Acquisition argmax crosshair
  if (state.acquisition.xstar !== null) {
    const px = xpx(state.acquisition.xstar);
    ctx.strokeStyle = withAlpha(getCssVar('--posterior'), 0.5);
    ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(px, PADS.T); ctx.lineTo(px, h - PADS.B); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Hover crosshair
  if (state.hover.idx !== null) {
    const i = state.hover.idx;
    const px = xpx(state.testGrid[i]);
    ctx.strokeStyle = withAlpha(getCssVar('--text'), 0.2);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, PADS.T); ctx.lineTo(px, h - PADS.B); ctx.stroke();
    // dot at mu
    ctx.fillStyle = getCssVar('--posterior');
    ctx.beginPath(); ctx.arc(px, ypx(mu[i]), 3.5, 0, Math.PI * 2); ctx.fill();
  }
}

function withAlpha(hex, alpha) {
  // hex can be #rrggbb or rgb()
  if (hex.startsWith('#')) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  if (hex.startsWith('rgb(')) {
    return hex.replace('rgb(', 'rgba(').replace(')', `,${alpha})`);
  }
  return hex;
}

function renderEntropy() {
  const { ctx, w, h } = setupCanvas(entropyCanvas);
  const xDom = state.xDomain;
  if (!state.entropy) { ctx.clearRect(0, 0, w, h); return; }
  const { Htotal, Halea, Hepi } = state.entropy;
  let ymax = 0;
  for (let i = 0; i < Htotal.length; i++) if (Htotal[i] > ymax) ymax = Htotal[i];
  // Allow negative entropy (Gaussian can be negative)
  let ymin = 0;
  for (let i = 0; i < Halea.length; i++) if (Halea[i] < ymin) ymin = Halea[i];
  const yDom = [ymin - 0.1, ymax + 0.2];
  plotAxes(ctx, w, h, PADS.L, PADS.R, PADS.T, PADS.B, xDom, yDom, { yTicks: 3 });

  const grid = state.testGrid;
  const xpx = (x) => toPx(x, xDom[0], xDom[1], PADS.L, w - PADS.R);
  const ypx = (y) => toPx(y, yDom[1], yDom[0], PADS.T, h - PADS.B);
  const zero = ypx(0);

  // Aleatoric area (from 0 baseline)
  ctx.fillStyle = withAlpha(getCssVar('--posterior'), 0.25);
  ctx.beginPath();
  ctx.moveTo(xpx(grid[0]), zero);
  for (let i = 0; i < grid.length; i++) ctx.lineTo(xpx(grid[i]), ypx(Halea[i]));
  ctx.lineTo(xpx(grid[grid.length-1]), zero);
  ctx.closePath(); ctx.fill();

  // Epistemic area stacked on top of aleatoric
  ctx.fillStyle = withAlpha(getCssVar('--posterior'), 0.55);
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) ctx.lineTo(xpx(grid[i]), ypx(Halea[i]));
  for (let i = grid.length - 1; i >= 0; i--) ctx.lineTo(xpx(grid[i]), ypx(Halea[i] + Hepi[i]));
  ctx.closePath(); ctx.fill();

  // Total line
  ctx.strokeStyle = getCssVar('--posterior');
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) {
    const px = xpx(grid[i]), py = ypx(Htotal[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function renderAcq() {
  const { ctx, w, h } = setupCanvas(acqCanvas);
  const xDom = state.xDomain;
  if (!state.acquisition.surface) { ctx.clearRect(0, 0, w, h); return; }
  const surf = state.acquisition.surface;
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < surf.length; i++) { if (surf[i] < ymin) ymin = surf[i]; if (surf[i] > ymax) ymax = surf[i]; }
  if (ymax - ymin < 1e-9) { ymax = ymin + 1; }
  const yDom = [ymin - 0.05 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)];
  plotAxes(ctx, w, h, PADS.L, PADS.R, 6, 18, xDom, yDom, { yTicks: 2, labels: false });

  // Re-draw bottom labels manually
  ctx.fillStyle = getCssVar('--text-faint');
  ctx.font = '10px "Geist Mono", monospace';
  ctx.textBaseline = 'top'; ctx.textAlign = 'center';
  const grid = state.testGrid;
  const xpx = (x) => toPx(x, xDom[0], xDom[1], PADS.L, w - PADS.R);
  const ypx = (y) => toPx(y, yDom[1], yDom[0], 6, h - 18);

  // Filled curve
  ctx.fillStyle = withAlpha(getCssVar('--prior'), 0.3);
  ctx.beginPath();
  ctx.moveTo(xpx(grid[0]), ypx(yDom[0]));
  for (let i = 0; i < grid.length; i++) ctx.lineTo(xpx(grid[i]), ypx(surf[i]));
  ctx.lineTo(xpx(grid[grid.length-1]), ypx(yDom[0]));
  ctx.closePath(); ctx.fill();

  ctx.strokeStyle = getCssVar('--prior');
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < grid.length; i++) {
    const px = xpx(grid[i]), py = ypx(surf[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // argmax
  if (state.acquisition.xstar !== null) {
    const px = xpx(state.acquisition.xstar);
    ctx.strokeStyle = getCssVar('--paradigm');
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 6); ctx.lineTo(px, h - 18); ctx.stroke();
  }
}

function renderStrip() {
  const c = stripCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  if (!state.entropy) return;
  const { Htotal, Halea, Hepi } = state.entropy;
  const xDom = state.xDomain;
  const grid = state.testGrid;
  const xpx = (x) => toPx(x, xDom[0], xDom[1], PADS.L, w - PADS.R);

  // For each pixel column in plot area, find nearest grid index and classify
  const plotL = PADS.L, plotR = w - PADS.R;
  const plotW = plotR - plotL;
  for (let p = 0; p < plotW; p++) {
    const xVal = xDom[0] + (xDom[1] - xDom[0]) * p / plotW;
    // bin to grid
    let idx = Math.floor((xVal - xDom[0]) / (xDom[1] - xDom[0]) * (grid.length - 1));
    if (idx < 0) idx = 0; if (idx > grid.length - 1) idx = grid.length - 1;
    const htot = Htotal[idx], hale = Halea[idx], hepi = Hepi[idx];
    const noiseShare = htot > 0 ? hale / htot : 0;
    let cls;
    let margin = 0;
    if (htot <= state.policy.actEntropyMax) { cls = 'act'; margin = (state.policy.actEntropyMax - htot) / Math.max(0.01, state.policy.actEntropyMax); }
    else if (noiseShare >= state.policy.deferNoiseRatio) { cls = 'defer'; margin = (noiseShare - state.policy.deferNoiseRatio) / Math.max(0.01, 1 - state.policy.deferNoiseRatio); }
    else if (hepi >= state.policy.queryEigMin) { cls = 'query'; margin = (hepi - state.policy.queryEigMin) / Math.max(0.01, hepi); }
    else { cls = 'defer'; margin = 0.2; }
    const alpha = 0.45 + 0.55 * Math.min(1, Math.max(0, margin));
    const color = cls === 'act' ? getCssVar('--act') : cls === 'query' ? getCssVar('--query') : getCssVar('--defer');
    ctx.fillStyle = withAlpha(color, alpha);
    ctx.fillRect(plotL + p, 0, 1, h);
  }
}

function renderBocpd() {
  const c = bocpdCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const hist = bocpd.history;
  if (hist.length === 0) return;
  // max run length seen
  let maxR = 1;
  for (const e of hist) if (e.modeR > maxR) maxR = e.modeR;
  const xpx = (i) => (i / Math.max(1, hist.length - 1)) * w;
  const ypx = (r) => h - (r / Math.max(1, maxR)) * h;
  // fill area
  ctx.fillStyle = withAlpha(getCssVar('--posterior'), 0.25);
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < hist.length; i++) ctx.lineTo(xpx(i), ypx(hist[i].modeR));
  ctx.lineTo(w, h); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = getCssVar('--posterior');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    if (i === 0) ctx.moveTo(xpx(i), ypx(hist[i].modeR));
    else ctx.lineTo(xpx(i), ypx(hist[i].modeR));
  }
  ctx.stroke();
  // P(r=0) overlay as red dots when high
  for (let i = 0; i < hist.length; i++) {
    if (hist[i].p0 > 0.3) {
      ctx.fillStyle = withAlpha(getCssVar('--paradigm'), Math.min(1, hist[i].p0));
      ctx.beginPath(); ctx.arc(xpx(i), 4, 2.5, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function renderTrace() {
  const c = traceCanvas;
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const hist = state.traceHistory;
  if (hist.length < 2) {
    // draw seed line
    ctx.strokeStyle = withAlpha(getCssVar('--prior'), 0.4);
    ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
    return;
  }
  let mn = Infinity, mx = -Infinity;
  for (const v of hist) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mx - mn < 1e-6) { mx = mn + 1; }
  ctx.strokeStyle = getCssVar('--prior');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const px = (i / (hist.length - 1)) * w;
    const py = h - ((hist[i] - mn) / (mx - mn)) * h;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

// ESS trajectory sparkline — reads state.essHistory. The y-axis is fixed to the
// full possible range [1, K] (K = ensemble size) so height reads as absolute
// posterior concentration: near K ⇒ many hyperparameters fit equally well (diffuse);
// near 1 ⇒ one hyperparameter set dominates (data pin θ down). Reference line at K.
function renderEssSparkline() {
  const c = document.getElementById('ess-canvas');
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  if (rect.width === 0) return;
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const hist = state.essHistory;
  const K = state.diagnostics?.count || 16;
  if (!hist || hist.length < 2) {
    ctx.strokeStyle = withAlpha(getCssVar('--text-faint') || '#4e5868', 0.4);
    ctx.beginPath(); ctx.moveTo(0, h - 1); ctx.lineTo(w, h - 1); ctx.stroke();
    return;
  }
  const mn = 1, mx = K;
  const yOf = (v) => h - ((v - mn) / (mx - mn)) * h;

  // Reference line at the ceiling K (= fully diffuse).
  ctx.strokeStyle = withAlpha(getCssVar('--text-faint') || '#4e5868', 0.5);
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, yOf(K)); ctx.lineTo(w, yOf(K));
  ctx.stroke();
  ctx.setLineDash([]);

  // Trajectory
  ctx.strokeStyle = getCssVar('--posterior') || '#4fd1c7';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const px = (i / (hist.length - 1)) * w;
    const py = yOf(hist[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Range readout
  const rangeEl = document.getElementById('i2-ess-range');
  if (rangeEl) rangeEl.textContent = `ESS ${hist[hist.length - 1].toFixed(1)} / ${K}`;
}

// Live I3 entropy formula: substitute the σ values at grid point i so the
// reader sees the decomposition with real numbers, not just an abstract formula.
function renderI3Formula(i) {
  const el = $('#i3-formula');
  if (!el || !state.posterior || !state.entropy) return;
  const sf = state.posterior.sigmaF[i], sn = state.posterior.sigmaN[i];
  const h = state.entropy.Htotal[i];
  el.innerHTML = `H = ½·log(2πe(σ<sub>f</sub>² + σ<sub>n</sub>²)) = ½·log(2πe(${sf.toFixed(2)}² + ${sn.toFixed(2)}²)) = <b>${h.toFixed(3)}</b> nats`;
}

// ─────────── Inspector readouts ───────────
function updateInspectorReadouts() {
  const n = state.obsX.length;
  $('#s1-n').textContent = String(n);
  if (n) {
    $('#s1-x').textContent = state.obsX[n-1].toFixed(2);
    $('#s1-y').textContent = state.obsY[n-1].toFixed(2);
    $('#s1-regime').textContent = state.obsRegime[n-1];
  }
  $('#s2-p0').textContent = (bocpd.R[0] || 0).toFixed(2);
  // mode r
  let modeR = 0, modeP = 0;
  for (let r = 0; r < bocpd.R.length; r++) if (bocpd.R[r] > modeP) { modeP = bocpd.R[r]; modeR = r; }
  $('#s2-mode').textContent = String(modeR);
  $('#s2-rv').textContent = kalman.residVar.toFixed(3);
  $('#s2-cp').textContent = String(state.changePoints.length);

  // S3
  if (n) {
    $('#s3-mu').textContent = `[${kalman.x.toFixed(2)}]`;
    $('#s3-cov').textContent = kalman.residVar.toFixed(3);
  }

  // P2 groups
  const na = state.obsRegime.filter(r => r === 'A').length;
  const nb = state.obsRegime.filter(r => r === 'B').length;
  $('#p2-na').textContent = na; $('#p2-nb').textContent = nb;

  // P1 current
  const cur = registry.current();
  if (cur) {
    $('#p1-ver').textContent = cur.id;
    $('#p1-author').textContent = cur.author;
    $('#p1-just').textContent = cur.justification;
  }

  // I1
  if (state.posterior) {
    const sf = state.posterior.sigmaF;
    let mean = 0, peakMu = -Infinity, peakX = 0;
    for (let i = 0; i < sf.length; i++) {
      mean += sf[i];
      if (state.posterior.mu[i] > peakMu) { peakMu = state.posterior.mu[i]; peakX = state.testGrid[i]; }
    }
    mean /= sf.length;
    $('#i1-sigf').textContent = mean.toFixed(3);
    $('#i1-peakmu').textContent = `${peakMu.toFixed(2)} @ x=${peakX.toFixed(2)}`;
  }

  // I2: real importance-weighted hyperparameter-ensemble diagnostics (diagnostics.js).
  const diag = state.diagnostics;
  if (diag && diag.ess != null) {
    $('#i2-ess').textContent = diag.ess.toFixed(1);
    $('#i2-ent').textContent = diag.entropyNorm.toFixed(2);
    $('#i2-k').textContent = String(diag.count);
  } else {
    $('#i2-ess').textContent = '—';
    $('#i2-ent').textContent = '—';
    $('#i2-k').textContent = String(diag ? diag.count : 16);
  }
  // log mu trace
  if (state.posterior) {
    state.traceHistory.push(state.posterior.mu[100]);
    if (state.traceHistory.length > 80) state.traceHistory.shift();
  }
  renderTrace();

  // I3 at cursor / center
  if (state.entropy) {
    const i = state.hover.idx ?? 100;
    $('#i3-htot').textContent = state.entropy.Htotal[i].toFixed(3);
    $('#i3-hale').textContent = state.entropy.Halea[i].toFixed(3);
    $('#i3-hepi').textContent = state.entropy.Hepi[i].toFixed(3);
    renderI3Formula(i);
  }

  // I4 evidence
  if (state.posterior?.model) {
    $('#i4-current').textContent = state.posterior.model.logEvidence.toFixed(2);
  }
  renderBMA();

  // D1
  $('#d1-fn').textContent = state.acquisition.fn.toUpperCase();
  if (state.acquisition.xstar !== null) {
    $('#d1-xstar').textContent = state.acquisition.xstar.toFixed(2);
    $('#d1-avalue').textContent = state.acquisition.value.toFixed(3);
  }

  // D3
  $('#d3-act').textContent = state.decisionCounts.act;
  $('#d3-defer').textContent = state.decisionCounts.defer;
  $('#d3-query').textContent = state.decisionCounts.query;

  // ── Mode-aware derived readouts ──
  updateDerivedReadouts();
}

function updateDerivedReadouts() {
  if (!state.mode || !state.posterior) return;
  const mid = state.mode.id;
  if (mid === 'trading') {
    // Sharpe ≈ mean(μ_post) / sqrt(mean(σ_f²) + mean(σ_n²)) * √252
    const mu = state.posterior.mu, sf = state.posterior.sigmaF;
    if (mu.length && sf.length) {
      let mMu = 0, mSf2 = 0;
      for (let i = 0; i < mu.length; i++) { mMu += mu[i]; mSf2 += sf[i] * sf[i]; }
      mMu /= mu.length; mSf2 /= sf.length;
      const sigN = parseFloat($('#slider-sigman').value) || 0.05;
      const denom = Math.sqrt(mSf2 + sigN * sigN);
      const sharpe = denom > 1e-6 ? (mMu / denom) * Math.sqrt(252) : 0;
      $('#i3-derived').textContent = sharpe.toFixed(2);
    }
    // Max drawdown across observed y series
    const ys = state.obsY;
    if (ys.length > 1) {
      let peak = ys[0], maxDD = 0;
      for (let i = 1; i < ys.length; i++) {
        if (ys[i] > peak) peak = ys[i];
        const dd = (peak - ys[i]);
        if (dd > maxDD) maxDD = dd;
      }
      $('#d3-derived').textContent = maxDD.toFixed(3);
    } else {
      $('#d3-derived').textContent = '0.000';
    }
  } else if (mid === 'health') {
    // Adaptation index = mean μ_post − 65 (baseline ms)
    const mu = state.posterior.mu;
    if (mu.length) {
      let m = 0;
      for (let i = 0; i < mu.length; i++) m += mu[i];
      m /= mu.length;
      const adapt = m - 65;
      $('#i3-derived').textContent = adapt.toFixed(2);
    }
  }
}

function renderBMA() {
  const container = $('#i4-bma');
  if (!Object.keys(state.modelEvidence).length) return;
  // Softmax
  const vals = Object.values(state.modelEvidence);
  const maxV = Math.max(...vals);
  const exps = vals.map(v => Math.exp(v - maxV));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sum);
  let html = '';
  KERNEL_NAMES.forEach((k, i) => {
    const p = probs[i] || 0;
    const sel = k === state.kernelName ? ' selected' : '';
    html += `<div class="bma-row${sel}"><span class="bma-label">${KERNEL_LABELS[k]}</span><div class="bma-bar"><div class="bma-bar-fill" style="width:${(p*100).toFixed(1)}%"></div></div><span class="bma-num">${(p*100).toFixed(1)}%</span></div>`;
  });
  container.innerHTML = html;
}

// ─────────── Registry rendering ───────────
function updateRegistry(opts = {}) {
  const scroll = $('#registry-scroll');
  const versions = registry.list();
  scroll.innerHTML = '';
  versions.forEach((v, idx) => {
    const card = document.createElement('div');
    card.className = 'reg-card';
    if (opts.animateLast && idx === versions.length - 1) card.classList.add('new');
    const time = new Date(v.timestamp);
    const hh = String(time.getHours()).padStart(2, '0');
    const mm = String(time.getMinutes()).padStart(2, '0');
    const ss = String(time.getSeconds()).padStart(2, '0');
    let diffHtml = '';
    if (v.diff && v.diff.length) {
      diffHtml = '<div class="reg-diff">' + v.diff.map(d => {
        if (d.subdiff) return `${d.key}: …`;
        return `${d.key}: <span class="from">${d.from}</span> <span class="arrow">→</span> <span class="to">${d.to}</span>`;
      }).join('<br/>') + '</div>';
    }
    const tag = v.germinated
      ? '<span class="reg-tag germinated">germinated</span>'
      : '<span class="reg-tag seed">seed</span>';
    const dormancyBadge = v.dormancy > 0
      ? `<span class="reg-dormancy${v.dormancy > 30 ? ' warm' : ''}">dormant ${v.dormancy}t</span>`
      : '<span class="reg-dormancy active">active</span>';
    const retrievedBadge = v.retrievals > 0
      ? `<span class="reg-retrieved">retrieved ×${v.retrievals}</span>` : '';
    card.innerHTML = `
      <div class="reg-head">
        <span class="reg-ver">${v.id}</span>
        ${tag}
        <span class="reg-time">${hh}:${mm}:${ss}</span>
      </div>
      ${v.name ? `<div class="reg-seed-name">${v.name}</div>` : ''}
      <div class="reg-author">${v.author}</div>
      <div class="reg-just">${v.justification}</div>
      ${diffHtml}
      <div class="reg-meta mono small">${dormancyBadge}${retrievedBadge}</div>
    `;
    scroll.appendChild(card);
  });
  // Scroll to right
  scroll.scrollLeft = scroll.scrollWidth;
}

// ─────────── Decision log row ───────────
function appendDecisionLogRow(d) {
  const tbody = $('#log-tbody');
  const tr = document.createElement('tr');
  tr.className = 'cls-' + d.class;
  tr.innerHTML = `
    <td>${d.t}</td>
    <td>${d.class}</td>
    <td>${d.x.toFixed(2)}</td>
    <td>${d.mu.toFixed(2)}</td>
    <td>${d.sigmaF.toFixed(2)}</td>
    <td>${d.Htot.toFixed(2)}</td>
    <td>${d.Hepi.toFixed(2)}</td>
    <td>${d.priorVersion}</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);
  while (tbody.childElementCount > 50) tbody.removeChild(tbody.lastChild);
}

// ─────────── Architecture SVG (8-node graph) ───────────
function buildArch() {
  const svg = $('#arch-svg');
  // Node positions in 360 x 320 viewBox
  const nodes = [
    { id: 'S1', label: 'S1', sub: 'Stream',  x: 30,  y: 30 },
    { id: 'S2', label: 'S2', sub: 'BOCPD',   x: 30,  y: 95 },
    { id: 'S3', label: 'S3', sub: 'Features',x: 30,  y: 160 },
    { id: 'P1', label: 'P1', sub: 'Registry',x: 150, y: 250 },
    { id: 'P2', label: 'P2', sub: 'Hier.',   x: 150, y: 188 },
    { id: 'P3', label: 'P3', sub: 'Kernel',  x: 150, y: 125 },
    { id: 'I1', label: 'I1', sub: 'GP',      x: 270, y: 30 },
    { id: 'I2', label: 'I2', sub: 'Ensemble',x: 270, y: 95 },
    { id: 'I3', label: 'I3', sub: 'Entropy', x: 270, y: 160 },
    { id: 'I4', label: 'I4', sub: 'Evidence',x: 270, y: 225 },
    { id: 'D1', label: 'D1', sub: 'Acquis.', x: 150, y: 30 },
    { id: 'D2', label: 'D2', sub: 'Policy',  x: 150, y: 65 },
    // Use compact pos for D1-D4 in middle column lower
    { id: 'D3', label: 'D3', sub: 'Action',  x: 30,  y: 250 },
    { id: 'D4', label: 'D4', sub: 'Promote', x: 270, y: 290 }
  ];
  // simpler: redo positions in a clean 3-column grid
  const W = 360, H = 320;
  const cols = { S: 30, M: 165, R: 300 };
  const rows = { r0: 30, r1: 78, r2: 126, r3: 174, r4: 222, r5: 270 };
  const positions = {
    S1: [cols.S, rows.r0], S2: [cols.S, rows.r1], S3: [cols.S, rows.r2],
    P3: [cols.M, rows.r0], P2: [cols.M, rows.r1], P1: [cols.M, rows.r5],
    I1: [cols.R, rows.r0], I2: [cols.R, rows.r1], I3: [cols.R, rows.r2], I4: [cols.R, rows.r3],
    D1: [cols.M, rows.r2], D2: [cols.M, rows.r3], D3: [cols.M, rows.r4], D4: [cols.R, rows.r5]
  };
  const nW = 60, nH = 36;

  // Edges (data flow)
  const edges = [
    ['S1','S2'], ['S2','S3'], ['S3','P3'], ['S3','D1'],
    ['P3','I1'], ['P2','P3'], ['P1','P3'],
    ['I1','I2'], ['I1','I3'], ['I1','I4'],
    ['I3','D1'], ['I3','D2'], ['I4','D4'],
    ['D1','D2'], ['D2','D3'], ['D4','P1'],
    ['S2','P3']
  ];

  // Clear
  svg.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';

  // Draw edges
  for (const [from, to] of edges) {
    const [x1, y1] = positions[from];
    const [x2, y2] = positions[to];
    // anchor to right of from, left of to
    const sx = x1 + nW/2, sy = y1 + nH/2;
    const tx = x2 + nW/2, ty = y2 + nH/2;
    // Bezier control points
    const mx = (sx + tx) / 2;
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', `M ${sx} ${sy} C ${mx} ${sy}, ${mx} ${ty}, ${tx} ${ty}`);
    path.setAttribute('class', 'edge flow');
    path.dataset.from = from; path.dataset.to = to;
    svg.appendChild(path);
  }
  // Draw nodes (after edges so they sit on top)
  for (const id of Object.keys(positions)) {
    const [x, y] = positions[id];
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('transform', `translate(${x}, ${y})`);
    g.dataset.node = id;
    g.style.cursor = 'pointer';
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', nW); rect.setAttribute('height', nH);
    rect.setAttribute('rx', 6); rect.setAttribute('ry', 6);
    rect.setAttribute('class', 'node-rect');
    g.appendChild(rect);
    const lbl = document.createElementNS(ns, 'text');
    lbl.setAttribute('x', nW/2); lbl.setAttribute('y', 16);
    lbl.setAttribute('text-anchor', 'middle');
    lbl.setAttribute('class', 'node-label');
    lbl.textContent = id;
    g.appendChild(lbl);
    const sub = document.createElementNS(ns, 'text');
    sub.setAttribute('x', nW/2); sub.setAttribute('y', 28);
    sub.setAttribute('text-anchor', 'middle');
    sub.setAttribute('class', 'node-sub');
    sub.textContent = labelSub(id);
    g.appendChild(sub);
    g.addEventListener('click', () => focusNode(id));
    svg.appendChild(g);
  }
}

function labelSub(id) {
  // Mode-aware: prefer the mode's per-node short label if present, else fall back.
  const ov = state.mode?.nodeOverrides?.[id];
  if (ov && ov.short) return ov.short;
  return ({ S1:'Stream', S2:'BOCPD', S3:'Features', P1:'Registry', P2:'Hier.', P3:'Kernel',
    I1:'GP', I2:'Ensemble', I3:'Entropy', I4:'Evidence', D1:'Acquis.', D2:'Policy', D3:'Action', D4:'Promote' })[id] || id;
}

// Plain-language, one-line "what this node does / what its numbers mean". Shown
// under the architecture graph when a node is focused. Engine-level (mode-agnostic)
// so it always matches the math; modes retarget the card titles, not these.
const NODE_DOCS = {
  S1: 'Emits one observation (x, y, σ_obs) per tick from the active scenario stream.',
  S2: 'Kalman filter tracks the level; BOCPD watches for a regime change and reports P(change).',
  S3: 'Turns the conditioned signal into the features (mean, spread) the GP consumes.',
  P1: 'Versioned log of promoted priors — every belief update is timestamped and justified.',
  P2: 'Partial-pooling strength τ: 0 keeps regimes/cohorts siloed, 1 fully shares information.',
  P3: 'The GP kernel and its hyperparameters ℓ (lengthscale), η (amplitude), σ_n (noise).',
  I1: 'The posterior over f(x): mean μ plus epistemic (model) and aleatoric (noise) bands.',
  I2: 'Importance-weighted hyperparameter ensemble. ESS = effective draws; entropy = how diffuse θ is.',
  I3: 'Splits predictive entropy into reducible (epistemic) and irreducible (aleatoric) parts.',
  I4: 'Log marginal likelihood per kernel — how well each kernel explains the data (BMA weights).',
  D1: 'Acquisition α(x): where sampling next is most useful (EI / UCB / MES / BALD). Argmax = x*.',
  D2: 'Turns entropy into an action: act when confident, defer under noise, query when info is cheap.',
  D3: 'Audit log of every decision with the full pre-action belief state.',
  D4: 'Promotes the posterior to a new prior when epistemic entropy has dropped past the BF gate.'
};

function focusNode(id) {
  state.selectedNode = id;
  // Update SVG active class
  document.querySelectorAll('.arch-svg .node-rect').forEach(r => r.classList.remove('active'));
  const g = document.querySelector(`.arch-svg [data-node="${id}"] .node-rect`);
  if (g) g.classList.add('active');
  // Plain-language description under the graph
  const doc = document.getElementById('node-doc');
  if (doc) doc.textContent = NODE_DOCS[id] || '';
  // Card highlight + scroll
  document.querySelectorAll('.node-card').forEach(c => c.classList.remove('active'));
  const card = document.getElementById('card-' + id);
  if (card) {
    card.classList.add('active');
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// ─────────── Continuity Layer panel (coupled-system graph) ───────────
// Mirrors the buildArch() SVG pattern: built once when Continuity mode is applied,
// then renderCoupling() (gated in the loop) only mutates attributes/classes.
function buildCouplingGraph() {
  const svg = $('#coupling-svg');
  if (!svg || !state.scenarioGen?.coupling) return;
  const C = state.scenarioGen.coupling;
  const ids = state.scenarioGen.layerIds;
  const L = ids.length;
  const ns = 'http://www.w3.org/2000/svg';
  svg.innerHTML = '';
  // ring layout
  const cx = 180, cy = 145, R = 108, r = 27;
  const pos = [];
  for (let i = 0; i < L; i++) {
    const a = -Math.PI / 2 + i * 2 * Math.PI / L;
    pos.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
  }
  // edges: one per coupled unordered pair (structural topology). Lit by live corr at render.
  const EPS = 0.05;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      if (Math.max(Math.abs(C[i][j]), Math.abs(C[j][i])) < EPS) continue;
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', pos[i][0]); line.setAttribute('y1', pos[i][1]);
      line.setAttribute('x2', pos[j][0]); line.setAttribute('y2', pos[j][1]);
      line.setAttribute('class', 'coupling-edge');
      line.dataset.i = i; line.dataset.j = j;
      svg.appendChild(line);
    }
  }
  // nodes
  for (let i = 0; i < L; i++) {
    const [x, y] = pos[i];
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('transform', `translate(${x}, ${y})`);
    g.setAttribute('class', 'layer-node calm');
    g.dataset.layer = i;
    g.style.cursor = 'pointer';
    const circ = document.createElementNS(ns, 'circle');
    circ.setAttribute('r', r);
    circ.setAttribute('class', 'layer-circle');
    g.appendChild(circ);
    const lbl = document.createElementNS(ns, 'text');
    lbl.setAttribute('text-anchor', 'middle');
    lbl.setAttribute('y', 4);
    lbl.setAttribute('class', 'layer-label');
    lbl.textContent = ids[i];
    g.appendChild(lbl);
    g.addEventListener('click', () => {
      const lay = state.layers;
      const reg = lay ? lay.regimes[i] : '—';
      const lvl = lay ? lay.levels[i].toFixed(2) : '—';
      $('#coupling-doc').textContent = `${ids[i]} · ${LAYER_LABELS[ids[i]] || ids[i]} — ${reg}, level ${lvl}. Edge brightness = live correlation with the rest of the system.`;
      document.querySelectorAll('#coupling-svg .layer-node').forEach(n => n.classList.remove('selected'));
      g.classList.add('selected');
    });
    svg.appendChild(g);
  }
}

function renderCoupling() {
  const lay = state.layers;
  const idxEl = $('#coupling-index'), meter = $('#coupling-meter');
  if (!lay) {
    if (idxEl) idxEl.textContent = '—';
    if (meter) meter.style.width = '0%';
    return;
  }
  const idx = lay.index;
  if (idxEl) idxEl.textContent = Math.round(idx * 100) + '%';
  if (meter) {
    meter.style.width = `${Math.round(idx * 100)}%`;
    // cool (high coherence) → warm (fragmenting)
    meter.style.background = idx > 0.85 ? 'var(--mode-accent, #6aa6e0)'
      : idx > 0.7 ? '#e0b06a' : '#e07a6a';
  }
  const coh = $('#coupling-coh'); if (coh) coh.textContent = (lay.meanCorr ?? 0).toFixed(2);
  const fie = $('#coupling-fiedler'); if (fie) fie.textContent = (lay.fiedler ?? 0).toFixed(2);
  const fragCount = lay.regimes.filter(r => r === 'fragmenting').length;
  const fragEl = $('#coupling-frag'); if (fragEl) fragEl.textContent = `${fragCount}/${lay.regimes.length}`;

  // node regimes
  document.querySelectorAll('#coupling-svg .layer-node').forEach(g => {
    const i = +g.dataset.layer;
    const reg = lay.regimes[i] || 'calm';
    g.classList.remove('calm', 'stress', 'fragmenting');
    g.classList.add(reg);
  });
  // edges lit by live |corr|; cascade class when both endpoints fragmenting
  document.querySelectorAll('#coupling-svg .coupling-edge').forEach(line => {
    const i = +line.dataset.i, j = +line.dataset.j;
    const c = lay.corr ? Math.abs(lay.corr[i][j]) : 0;
    line.style.opacity = (0.08 + 0.85 * c).toFixed(3);
    const cascade = lay.regimes[i] === 'fragmenting' && lay.regimes[j] === 'fragmenting';
    line.classList.toggle('cascade', cascade);
  });
}

// Reflect the active coherence metric in the toggle buttons.
function syncMetricToggle() {
  const m = state.scenarioGen?.getMetric ? state.scenarioGen.getMetric() : 'corr';
  const bc = $('#metric-corr'), bf = $('#metric-fiedler');
  if (bc) bc.classList.toggle('active', m === 'corr');
  if (bf) bf.classList.toggle('active', m === 'fiedler');
}

// ─────────── Loop ───────────
let lastTickMs = 0;
let rafId = null;
function loop(now) {
  // Variance-inflation decay: ramp from the detected peak back down to 1.0.
  if (now < state.epistemicInflationUntil) {
    const t = (now - state.epistemicInflationStart) / INFLATION_WINDOW_MS;
    state.epistemicInflation = 1.0 + (state.epistemicInflationPeak - 1.0) * (1 - t);
  } else if (state.epistemicInflation !== 1.0) {
    state.epistemicInflation = 1.0;
    recomputePosterior(); recomputeAcquisition();
  }
  // Tick at requested rate
  const interval = 1000 / state.tickRateHz;
  if (state.running && now - lastTickMs >= interval) {
    lastTickMs = now;
    tick();
  }
  // Always render
  renderGP(); renderEntropy(); renderAcq(); renderStrip(); renderBocpd(); renderEssSparkline();
  if (state.mode?.id === 'continuity') renderCoupling();
  rafId = requestAnimationFrame(loop);
}

// ─────────── Events ───────────
function bind() {
  // Mode chip re-opens the picker
  $('#chip-mode').addEventListener('click', openModePicker);
  $('#link-skip-mode').addEventListener('click', (e) => {
    e.preventDefault();
    applyMode(getModeById(DEFAULT_MODE_ID));
    closeModePicker();
  });
  $('#btn-run').addEventListener('click', () => { state.running = true; });
  $('#btn-pause').addEventListener('click', () => { state.running = false; });
  $('#btn-step').addEventListener('click', () => { state.running = false; tick(); });
  $('#slider-tick-rate').addEventListener('input', (e) => {
    state.tickRateHz = parseFloat(e.target.value);
    $('#readout-tick').textContent = state.tickRateHz.toFixed(1) + ' Hz';
  });
  $('#btn-reset').addEventListener('click', () => resetSession(true));
  $('#btn-theme').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  });
  // Continuity Layer metric toggle: switch which coherence metric drives the index.
  document.querySelectorAll('.metric-toggle .mt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (state.scenarioGen?.setMetric) state.scenarioGen.setMetric(btn.dataset.metric);
      syncMetricToggle();
    });
  });
  $('#btn-help').addEventListener('click', () => $('#help-modal').hidden = false);
  $('#btn-help-close').addEventListener('click', () => $('#help-modal').hidden = true);
  $('#help-modal').addEventListener('click', (e) => { if (e.target.id === 'help-modal') $('#help-modal').hidden = true; });
  $('#btn-export').addEventListener('click', exportJSON);

  $('#toggle-truth').addEventListener('change', (e) => { state.showTruth = e.target.checked; });

  // P3 kernel + hp
  $('#kernel-select').addEventListener('change', (e) => {
    state.kernelName = e.target.value;
    recomputePosterior(); recomputeAcquisition(); recomputeModelEvidence(); recomputeDiagnostics(); updateInspectorReadouts();
  });
  // Optimize hyperparameters by type-II MLE (maximise the log marginal likelihood).
  $('#btn-optimize')?.addEventListener('click', () => {
    if (state.obsX.length < 2) return;
    const { hp } = optimizeHyperparameters(state.obsX, state.obsY, state.kernelName, state.hp);
    setSliderValue('#slider-ell', hp.ell, 2);
    setSliderValue('#slider-eta', hp.eta, 2);
    setSliderValue('#slider-sigman', hp.sigma_n, 3);
    recomputePosterior(); recomputeAcquisition(); recomputeModelEvidence(); recomputeDiagnostics(); updateInspectorReadouts();
  });
  bindSlider('#slider-ell',    '#readout-ell',    (v) => { state.hp.ell = v; throttledRecompute(); }, 2);
  bindSlider('#slider-eta',    '#readout-eta',    (v) => { state.hp.eta = v; throttledRecompute(); }, 2);
  bindSlider('#slider-sigman', '#readout-sigman', (v) => { state.hp.sigma_n = v; throttledRecompute(); }, 2);

  bindSlider('#slider-tau',    '#readout-tau',    (v) => { state.poolingStrength = v; }, 2);
  bindSlider('#slider-tact',   '#readout-tact',   (v) => { state.policy.actEntropyMax = v; }, 2);
  bindSlider('#slider-tdef',   '#readout-tdef',   (v) => { state.policy.deferNoiseRatio = v; }, 2);
  bindSlider('#slider-eig',    '#readout-eig',    (v) => { state.policy.queryEigMin = v; }, 2);
  // BF slider is presented to the user as the desired Bayes factor (≥ 1),
  // converted to its log for internal comparison against the entropy-reduction
  // signal (which is itself in nats).
  bindSlider('#slider-bf',     '#readout-bf',     (v) => { state.bfThreshold = Math.log(Math.max(v, 1.0001)); }, 1);

  $('#acq-fn').addEventListener('change', (e) => {
    state.acquisition.fn = e.target.value;
    recomputeAcquisition(); updateInspectorReadouts();
  });

  $('#btn-rerun').addEventListener('click', () => {
    // Redraw the hyperparameter ensemble with a fresh seed, then refresh readouts.
    state.diagSeed = (Math.imul(state.diagSeed, 1103515245) + 12345) >>> 0;
    recomputeDiagnostics();
    updateInspectorReadouts();
  });

  $('#log-toggle').addEventListener('click', () => {
    const body = $('#log-body');
    const log = $('#decision-log');
    const open = body.hidden;
    body.hidden = !open;
    log.classList.toggle('open', open);
  });

  $('#btn-copy-log').addEventListener('click', () => {
    const blob = JSON.stringify(state.decisions.slice(-20), null, 2);
    navigator.clipboard?.writeText(blob);
  });

  // Hover on GP plot
  gpCanvas.addEventListener('mousemove', (e) => {
    const rect = gpCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const xDom = state.xDomain;
    const w = rect.width;
    const xVal = xDom[0] + (xDom[1] - xDom[0]) * (px - PADS.L) / (w - PADS.L - PADS.R);
    if (xVal < xDom[0] || xVal > xDom[1]) {
      state.hover = { x: null, idx: null };
      updateHoverReadout(); return;
    }
    const idx = Math.round((xVal - xDom[0]) / (xDom[1] - xDom[0]) * (state.testGrid.length - 1));
    state.hover = { x: xVal, idx };
    updateHoverReadout();
  });
  gpCanvas.addEventListener('mouseleave', () => { state.hover = { x: null, idx: null }; updateHoverReadout(); });

  // Resize
  window.addEventListener('resize', () => { /* will re-render naturally */ });
}

function updateHoverReadout() {
  const i = state.hover.idx;
  if (i === null || !state.posterior) {
    ['hov-x','hov-mu','hov-sf','hov-sn','hov-h'].forEach(id => $('#' + id).textContent = '—');
    return;
  }
  const x = state.testGrid[i];
  $('#hov-x').textContent = x.toFixed(2);
  $('#hov-mu').textContent = state.posterior.mu[i].toFixed(3);
  $('#hov-sf').textContent = state.posterior.sigmaF[i].toFixed(3);
  $('#hov-sn').textContent = state.posterior.sigmaN[i].toFixed(3);
  $('#hov-h').textContent = state.entropy.Htotal[i].toFixed(3);
  // also push to I3 card
  $('#i3-htot').textContent = state.entropy.Htotal[i].toFixed(3);
  $('#i3-hale').textContent = state.entropy.Halea[i].toFixed(3);
  $('#i3-hepi').textContent = state.entropy.Hepi[i].toFixed(3);
  renderI3Formula(i);
}

function bindSlider(sliderSel, readoutSel, cb, decimals = 2) {
  const el = $(sliderSel), out = readoutSel ? $(readoutSel) : null;
  el.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (out) out.textContent = v.toFixed(decimals);
    cb(v);
  });
}

let recomputeTimer = null;
function throttledRecompute() {
  if (recomputeTimer) return;
  recomputeTimer = setTimeout(() => {
    recomputeTimer = null;
    recomputePosterior(); recomputeAcquisition(); updateInspectorReadouts();
  }, 80);
}

function resetSession(updateGrid) {
  state.obsX = []; state.obsY = []; state.obsSigma = []; state.obsRegime = []; state.obsT = []; state.obsSession = [];
  state.t = 0;
  state.decisions = [];
  state.decisionCounts = { act: 0, defer: 0, query: 0 };
  state.changePoints = [];
  state.modelEvidence = {};
  state.epistemicEntropyRef = null;
  state.logBayesFactor = 0;
  state.obsSinceLastPromo = 0;
  state.traceHistory = [];
  state.essHistory = [];
  state.diagnostics = { ess: null, entropyNorm: null, count: state.diagnostics.count || 16 };
  state.epistemicInflation = 1.0;
  state.epistemicInflationPeak = 1.0;
  state.epistemicInflationStart = 0;
  state.epistemicInflationUntil = 0;
  kalman.x = 0; kalman.P = 1; kalman.initialized = false; kalman.residVar = 0.5; kalman.r = 0.5;
  bocpd.reset();
  $('#log-tbody').innerHTML = '';
  if (updateGrid) makeTestGrid();
  if (state.scenarioGen) state.scenarioGen.reset();
  registry.reset({
    kernel: state.kernelName,
    ell: state.hp.ell, eta: state.hp.eta, sigma_n: state.hp.sigma_n,
    priors: { ell: 'Gamma(2,1)', eta: 'HalfN(0,1)', sigma_n: 'HalfN(0,0.5)' }
  }, {
    name: 'Seed prior',
    provenance: { source: 'human seed', regime: null, tick: 0, parentId: null }
  });
  recomputePosterior(); recomputeAcquisition();
  updateRegistry();
  updateInspectorReadouts();
  // If we're inside an active mode session, also re-apply mode again so the
  // current session entry's startTick is consistent.
  if (state.mode) startSession(state.mode);
}

// ─────────── Mode system ───────────
function posteriorSummary() {
  if (!state.posterior) return null;
  const { mu, sigmaF, sigmaN } = state.posterior;
  let sm = 0, ss = 0, sn = 0;
  for (let i = 0; i < mu.length; i++) { sm += mu[i]; ss += sigmaF[i]; sn += sigmaN[i]; }
  return {
    mean_mu: sm / mu.length,
    mean_sigma_f: ss / mu.length,
    mean_sigma_n: sn / mu.length
  };
}

function closeCurrentSession() {
  // attach final readouts to the existing session record
  if (!state.sessions.length) return;
  const sess = state.sessions[state.sessions.length - 1];
  sess.endTick = state.t;
  sess.finalRegistry = registry.list().map(v => ({ id: v.id, name: v.name, author: v.author, justification: v.justification, germinated: v.germinated, dormancy: v.dormancy, retrievals: v.retrievals }));
  sess.finalDecisions = {
    counts: { ...state.decisionCounts },
    last: state.decisions.slice(-20)
  };
  sess.finalPosteriorSummary = posteriorSummary();
}

function startSession(mode) {
  closeCurrentSession();
  state.sessionIndex = state.sessions.length;
  state.sessions.push({
    index: state.sessionIndex,
    modeId: mode.id,
    modeName: mode.name,
    startTick: state.t,
    endTick: null,
    finalRegistry: null,
    finalDecisions: null,
    finalPosteriorSummary: null,
    exportSchema: mode.exportSchema
  });
}

function setSliderValue(sel, value, decimals = 2) {
  const el = $(sel);
  if (!el) return;
  // clamp to min/max
  const mn = parseFloat(el.min), mx = parseFloat(el.max);
  let v = value;
  if (!Number.isFinite(v)) return;
  if (v < mn) v = mn; if (v > mx) v = mx;
  el.value = String(v);
  // dispatch input so existing handlers fire
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function applyMode(mode) {
  const prev = state.mode;
  state.mode = mode;
  state.layers = null;   // cleared on every switch; repopulated by tick() in Continuity mode
  // 1. plot config
  state.plotConfig = {
    xLabel: mode.domain.xLabel,
    yLabel: mode.domain.yLabel,
    xUnits: mode.domain.xUnits,
    yUnits: mode.domain.yUnits,
    xRange: mode.domain.xRange.slice(),
    yRange: mode.domain.yRange.slice()
  };
  // 2. scenario generator
  state.scenarioGen = mode.scenario(42);
  // 3. rewrite eyebrows + titles
  for (const [nodeId, ov] of Object.entries(mode.nodeOverrides || {})) {
    document.querySelectorAll(`[data-mode-eyebrow="${nodeId}"]`).forEach(el => { el.textContent = ov.eyebrow; });
    document.querySelectorAll(`[data-mode-title="${nodeId}"]`).forEach(el => { el.textContent = ov.title; });
  }
  // 4. rewrite slider labels + tooltips
  for (const [sliderId, ov] of Object.entries(mode.controlOverrides || {})) {
    document.querySelectorAll(`[data-mode-label="${sliderId}"]`).forEach(el => {
      el.textContent = ov.label + (ov.unit ? ` (${ov.unit})` : '');
    });
    document.querySelectorAll(`[data-mode-tooltip="${sliderId}"]`).forEach(el => {
      el.title = ov.tooltip;
    });
  }
  // 5. kernel + defaults
  state.kernelName = mode.defaults.kernel;
  const ksel = $('#kernel-select');
  if (ksel) ksel.value = state.kernelName;
  setSliderValue('#slider-ell',    mode.defaults.ell, 2);
  setSliderValue('#slider-eta',    mode.defaults.eta, 2);
  setSliderValue('#slider-sigman', mode.defaults.sigma_n, 3);
  setSliderValue('#slider-tau',    mode.defaults.poolingStrength, 2);
  setSliderValue('#slider-tact',   mode.defaults.actEntropyMax, 2);
  setSliderValue('#slider-tdef',   mode.defaults.deferNoiseRatio, 2);
  setSliderValue('#slider-eig',    mode.defaults.queryEigMin, 2);
  setSliderValue('#slider-bf',     mode.defaults.bf, 1);
  // 5b. Per-mode BOCPD hazard (1/lambda). Trading runs aggressive so vol
  // regime shifts fire within demo windows; Glidepath stays patient so a
  // 30-yr horizon isn't choked with false positives.
  if (Number.isFinite(mode.defaults.bocpdLambda) && mode.defaults.bocpdLambda > 1) {
    bocpd.hazard = 1 / mode.defaults.bocpdLambda;
  }
  // Per-mode change-point prior (Normal-Inverse-Gamma). Modes that don't specify
  // one fall back to the original scale so switching away from a small-signal mode
  // (e.g. Continuity, whose index is in [0,1]) restores the default detector.
  const bp = mode.defaults.bocpdPrior || { mu0: 0, kappa0: 5, alpha0: 2, beta0: 0.3 };
  bocpd.mu0 = bp.mu0; bocpd.kappa0 = bp.kappa0; bocpd.alpha0 = bp.alpha0; bocpd.beta0 = bp.beta0;
  // 5c. Continuity Layer panel — only this mode shows the coupled-system graph.
  const couplingPanel = $('#coupling-panel');
  if (couplingPanel) {
    if (mode.id === 'continuity') {
      couplingPanel.hidden = false;
      buildCouplingGraph();
      syncMetricToggle();
    } else {
      couplingPanel.hidden = true;
    }
  }
  // 6. chip
  applyChip(mode);
  // 7. kernel formula label
  applyKernelFormula(mode);
  // 8. I3 derived readouts + D3 label
  applyDerivedReadouts(mode);
  // 9. update grid + reset trace state
  makeTestGrid();
  resetSession(false);
  // Start a new session entry. resetSession also calls startSession when state.mode is set,
  // but at this point we've already replaced state.mode; the previous call already created
  // the entry for the new mode. Re-do to ensure modeId is the *new* mode.
  // (Idempotent — closeCurrentSession only writes once.)
  if (state.sessions.length === 0 || state.sessions[state.sessions.length - 1].modeId !== mode.id) {
    startSession(mode);
  } else {
    // reset the most recent session entry's startTick
    const s = state.sessions[state.sessions.length - 1];
    s.startTick = state.t;
  }
}

function applyChip(mode) {
  $('#chip-name').textContent = mode.name;
  $('#chip-icon').innerHTML = mode.icon;
  const chip = $('#chip-mode');
  chip.style.setProperty('--mode-accent', mode.accent);
}

function applyKernelFormula(mode) {
  const el = $('#kernel-formula');
  if (!el) return;
  const k = mode.defaults.kernel;
  const f = {
    rbf:        'k(r) = η² exp(−r² / 2ℓ²)',
    matern32:   'k(r) = η²(1 + √3·r/ℓ) exp(−√3·r/ℓ)',
    matern52:   'k(r) = η²(1 + √5·r/ℓ + 5r²/(3ℓ²)) exp(−√5·r/ℓ)',
    periodic:   'k(r) = η² exp(− 2 sin²(π r / p) / ℓ²)',
    spectralmix:'k(r) = Σ_q w_q exp(−2π² r² v_q) cos(2π r μ_q)',
    linear_matern52: 'k(r) = σ_lin² x x′ + k_Matérn52(r)'
  }[k] || '';
  el.textContent = f;
}

function applyDerivedReadouts(mode) {
  // I3 row
  const i3Row = $('#i3-derived-row');
  const i3Label = $('#i3-derived-label');
  const aleLbl = $('#i3-hale-label');
  const epiLbl = $('#i3-hepi-label');
  if (mode.id === 'trading') {
    i3Row.hidden = false;
    i3Label.innerHTML = 'Sharpe · √252';
    aleLbl.innerHTML = 'H<sub>aleatoric</sub> · noise floor';
    epiLbl.innerHTML = 'H<sub>epistemic</sub> · model';
  } else if (mode.id === 'health') {
    i3Row.hidden = false;
    i3Label.innerHTML = 'Adaptation index (ms)';
    aleLbl.innerHTML = 'Day-to-day variability';
    epiLbl.innerHTML = 'Adaptation uncertainty';
  } else {
    i3Row.hidden = true;
    aleLbl.innerHTML = 'H<sub>aleatoric</sub>';
    epiLbl.innerHTML = 'H<sub>epistemic</sub>';
  }
  // D3 labels
  $('#d3-act-label').textContent = mode.decisions.act.label.toLowerCase();
  $('#d3-defer-label').textContent = mode.decisions.defer.label.toLowerCase();
  $('#d3-query-label').textContent = mode.decisions.query.label.toLowerCase();
  const d3Row = $('#d3-derived-row');
  const d3Label = $('#d3-derived-label');
  if (mode.id === 'trading') {
    d3Row.hidden = false;
    d3Label.textContent = 'max drawdown';
  } else {
    d3Row.hidden = true;
  }
}

// ─────────── Mode picker UI ───────────
function renderModePicker() {
  const grid = $('#mode-picker-grid');
  if (!grid) return;
  grid.innerHTML = '';
  for (const mode of MODES) {
    const card = document.createElement('article');
    card.className = 'mode-card';
    card.style.setProperty('--mode-accent', mode.accent);
    card.innerHTML = `
      <div class="mode-card-icon">${mode.icon}</div>
      <h3>${mode.name}</h3>
      <p class="mode-card-tagline">${mode.tagline}</p>
      <canvas class="mode-card-sample" data-mode-sample="${mode.id}" width="400" height="160"></canvas>
      <p class="mode-card-desc">${mode.description}</p>
      <ul class="mode-card-lines">
        <li><span class="verb" style="color:${mode.accent}">${mode.decisions.act.label}</span> when ${mode.actLine}</li>
        <li><span class="verb" style="color:${mode.accent}">${mode.decisions.defer.label}</span> when ${mode.deferLine}</li>
        <li><span class="verb" style="color:${mode.accent}">${mode.decisions.query.label}</span> when ${mode.queryLine}</li>
      </ul>
      <button class="mode-card-launch" data-testid="button-launch-${mode.id}" data-mode-launch="${mode.id}">Launch in ${mode.name}</button>
    `;
    grid.appendChild(card);
  }
  // bind launches
  grid.querySelectorAll('[data-mode-launch]').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = getModeById(btn.dataset.modeLaunch);
      // Suppress no-op relaunch: if the same mode is already active, just
      // close the picker. Avoids the duplicate-session entry that appears
      // when the user picks the auto-applied default mode from the modal.
      if (state.mode && state.mode.id === mode.id) {
        closeModePicker();
        return;
      }
      applyMode(mode);
      closeModePicker();
    });
  });
  // render sample plots
  requestAnimationFrame(renderModeSamples);
}

const _sampleCache = new Map();
function renderModeSamples() {
  for (const mode of MODES) {
    const cv = document.querySelector(`canvas[data-mode-sample="${mode.id}"]`);
    if (!cv) continue;
    if (_sampleCache.has(mode.id)) {
      const img = _sampleCache.get(mode.id);
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(img, 0, 0, cv.width, cv.height);
      continue;
    }
    // generate 60 points
    const gen = mode.scenario(7);
    const pts = [];
    for (let i = 0; i < 60; i++) pts.push(gen.next());
    const [xa, xb] = mode.domain.xRange;
    let ya = Infinity, yb = -Infinity;
    for (const p of pts) { if (p.y < ya) ya = p.y; if (p.y > yb) yb = p.y; }
    const pad = Math.max(0.1, (yb - ya) * 0.15);
    ya -= pad; yb += pad;
    const w = cv.width, h = cv.height;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    // soft fill
    ctx.fillStyle = mode.accent + '22';
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < pts.length; i++) {
      const px = (pts[i].x - xa) / (xb - xa) * w;
      const py = h - (pts[i].y - ya) / (yb - ya) * h;
      ctx.lineTo(px, py);
    }
    ctx.lineTo(w, h); ctx.closePath(); ctx.fill();
    // line
    ctx.strokeStyle = mode.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const px = (pts[i].x - xa) / (xb - xa) * w;
      const py = h - (pts[i].y - ya) / (yb - ya) * h;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // cache as image
    const img = new Image();
    img.src = cv.toDataURL();
    _sampleCache.set(mode.id, img);
  }
}

function openModePicker() {
  const dlg = $('#mode-picker');
  if (!dlg) return;
  if (!dlg.open) dlg.showModal();
  requestAnimationFrame(renderModeSamples);
}

function closeModePicker() {
  const dlg = $('#mode-picker');
  if (dlg?.open) dlg.close();
}

function exportJSON() {
  const mode = state.mode || MODES[0];
  // Ensure current session has up-to-date final fields
  if (state.sessions.length) {
    const s = state.sessions[state.sessions.length - 1];
    s.endTick = state.t;
    s.finalRegistry = registry.list().map(v => ({ id: v.id, name: v.name, author: v.author, justification: v.justification, germinated: v.germinated, dormancy: v.dormancy, retrievals: v.retrievals }));
    s.finalDecisions = { counts: { ...state.decisionCounts }, last: state.decisions.slice(-20) };
    s.finalPosteriorSummary = posteriorSummary();
  }
  const decisionsField = mode.exportShape?.decisions_field || 'decisions';
  const out = {
    schema: mode.exportSchema,
    exportedAt: new Date().toISOString(),
    mode: { id: mode.id, name: mode.name },
    plotConfig: state.plotConfig,
    kernel: state.kernelName,
    hyperparameters: { ell: state.hp.ell, eta: state.hp.eta, sigma_n: state.hp.sigma_n },
    priorRegistry: registry.list().map(v => ({
      version: v.id, variable: 'gp', family: 'kernel-hyperprior',
      name: v.name, params: v.snapshot, author: v.author, justification: v.justification,
      provenance: v.provenance, germinated: v.germinated,
      dormancy: v.dormancy, retrievals: v.retrievals, lastRetrievedTick: v.lastRetrievedTick,
      timestamp: new Date(v.timestamp).toISOString(), diff: v.diff
    })),
    observations: state.obsX.map((x, i) => ({
      t: state.obsT[i], session: state.obsSession[i], x, y: state.obsY[i],
      sigma_obs: state.obsSigma[i], regime: state.obsRegime[i]
    })),
    changePoints: state.changePoints.map(c => ({ t: c.t, x: c.x, posterior_mass: c.p })),
    [decisionsField]: state.decisions,
    sessions: state.sessions,
    currentPosterior: state.posterior ? {
      x_grid: Array.from(state.testGrid),
      mu: Array.from(state.posterior.mu),
      sigma_f: Array.from(state.posterior.sigmaF),
      sigma_n_x: Array.from(state.posterior.sigmaN),
      H_total: Array.from(state.entropy.Htotal),
      H_epi: Array.from(state.entropy.Hepi),
      H_alea: Array.from(state.entropy.Halea)
    } : null,
    modelEvidence: state.modelEvidence,
    ...(mode.id === 'continuity' && state.scenarioGen?.coupling ? {
      continuity: {
        layerIds: state.scenarioGen.layerIds,
        couplingMatrix: state.scenarioGen.coupling,
        metric: state.scenarioGen.getMetric ? state.scenarioGen.getMetric() : 'corr',
        latestLayers: state.layers
      }
    } : {}),
    ...(mode.exportShape || {})
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `bre1-${mode.id}-${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

// ─────────── Init ───────────
function init() {
  bind();
  buildArch();
  renderModePicker();
  // Apply default mode first so engine state is fully populated even if the
  // user dismisses the picker without clicking a card.
  applyMode(getModeById(DEFAULT_MODE_ID));
  state.running = true;
  rafId = requestAnimationFrame(loop);
  // Auto-show the picker on first paint per spec.
  requestAnimationFrame(() => openModePicker());
}

init();
