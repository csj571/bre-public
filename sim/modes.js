// modes.js — BRE-1 mode contracts and scenario generators
// A mode retargets vocabulary, priors, scenario stream, kernel default,
// decision semantics, and export schema while leaving the engine intact.

import { createCoupledSystem, createCoherenceTracker, dominantRegime } from './coupling.js';

// ─────────── Seeded RNG (Mulberry32) ───────────
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFrom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ─────────── GLIDEPATH scenario ───────────
// Regime-switching real log-wealth, 30-yr horizon, dt=0.25 yr.
// Three seeded paradigm shifts in [5,25] yr — inheritance, crash, business sale.
export function createGlidepathScenario(seed = 42) {
  const dt = 0.25;
  let rng = mulberry32(seed);
  // pre-draw paradigm positions and magnitudes deterministically
  function drawShifts(r) {
    return [
      { x: 5 + r() * 20, jump: 0.4  },
      { x: 5 + r() * 20, jump: -0.6 },
      { x: 5 + r() * 20, jump: 0.3  }
    ].sort((a, b) => a.x - b.x);
  }
  let shifts = drawShifts(rng);
  let firedShifts = 0;
  let x = 0;
  let logW = 0;            // accumulated log-wealth
  function regimeFor(xv) {
    if (xv < 10) return 'A';
    if (xv < 15) return 'B';
    return 'C';
  }
  function driftVol(reg, xv) {
    if (reg === 'A') return { mu: 0.06, sigma: 0.18 + 0.02 * (xv / 10) };
    if (reg === 'B') return { mu: 0.04, sigma: 0.22 };
    return { mu: 0.02, sigma: 0.14 };
  }
  return {
    seed,
    next() {
      x += dt;
      const reg = regimeFor(x);
      const { mu, sigma } = driftVol(reg, x);
      // dlogW ≈ (μ - 0.5 σ²) dt + σ √dt · Z
      const z = gaussianFrom(rng);
      logW += (mu - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * z;
      // fire paradigm shift if x has crossed its threshold
      while (firedShifts < shifts.length && x >= shifts[firedShifts].x) {
        logW += shifts[firedShifts].jump;
        firedShifts += 1;
      }
      const sigma_obs = 0.12 + 0.04 * Math.sin(0.6 * x);
      const noise = gaussianFrom(rng) * sigma_obs;
      return { x, y: logW + noise, sigma_obs, regime: reg };
    },
    reset() {
      rng = mulberry32(seed);
      shifts = drawShifts(rng);
      firedShifts = 0;
      x = 0;
      logW = 0;
    }
  };
}

// ─────────── TRADING & SHARPE scenario ───────────
// OU drift + volatility-clustering regime switching every ~80 ticks.
export function createTradingScenario(seed = 42) {
  let rng = mulberry32(seed);
  let x = 0;
  let alpha = 0;           // slow mean-reverting drift, on log-return scale
  let cumLogRet = 0;
  let peak = 0;
  let vol = 0.008;
  let regimeName = 'low';
  let ticksToSwitch = 80 + Math.floor((rng() - 0.5) * 20);
  const kappa = 0.05;      // OU mean reversion strength
  const theta = 0;
  const sigmaAlpha = 0.005;
  const beta = 0.3;        // drawdown penalty coefficient on latent f
  return {
    seed,
    next() {
      x += 1;
      // OU step for alpha
      alpha += kappa * (theta - alpha) + sigmaAlpha * gaussianFrom(rng);
      // volatility regime switch?
      ticksToSwitch -= 1;
      if (ticksToSwitch <= 0) {
        regimeName = regimeName === 'low' ? 'high' : 'low';
        vol = regimeName === 'low' ? 0.008 : 0.028;
        ticksToSwitch = 80 + Math.floor((rng() - 0.5) * 20);
      }
      const dret = alpha + vol * gaussianFrom(rng);
      cumLogRet += dret;
      if (cumLogRet > peak) peak = cumLogRet;
      const drawdown = peak - cumLogRet;
      // The "true latent" we want the GP to recover: drift minus drawdown penalty.
      const latent = cumLogRet - beta * drawdown;
      // Observation noise = vol itself — heteroscedastic per regime.
      const sigma_obs = vol * 1.2;
      const obsNoise = gaussianFrom(rng) * sigma_obs;
      return { x, y: latent + obsNoise, sigma_obs, regime: regimeName };
    },
    reset() {
      rng = mulberry32(seed);
      x = 0; alpha = 0; cumLogRet = 0; peak = 0;
      vol = 0.008; regimeName = 'low';
      ticksToSwitch = 80 + Math.floor((rng() - 0.5) * 20);
    }
  };
}

// ─────────── HEALTH & LONGEVITY scenario ───────────
// HRV proxy: baseline + logistic adaptation - fatigue exponential + weekly cycle.
// Illness drop at day 60, training program change at day 120.
export function createHealthScenario(seed = 42) {
  let rng = mulberry32(seed);
  let x = 0;
  const baseline = 65;
  let adaptation = 0;
  let fatigue = 0;
  let illnessActive = false;
  let illnessStart = 60;
  let programShifted = false;
  return {
    seed,
    next() {
      x += 1;
      // logistic adaptation up to +12 ms over 180 days
      const targetAdapt = 12 / (1 + Math.exp(-(x - 80) / 25));
      adaptation += (targetAdapt - adaptation) * 0.08;
      // training-program change at day 120 — shift drift up by 4 ms.
      let progShift = 0;
      if (x >= 120) {
        if (!programShifted) programShifted = true;
        progShift = 4 * (1 - Math.exp(-(x - 120) / 10));
      }
      // fatigue with random spikes on "hard training days"
      if (rng() < 0.18) fatigue += 3 + rng() * 4;
      fatigue *= 0.85;
      // illness event around day 60: HRV drops 20 % over 14 days
      let illness = 0;
      if (x >= illnessStart && x < illnessStart + 14) {
        const t = x - illnessStart;
        illness = -0.20 * baseline * Math.exp(-t / 6) * (1 - Math.exp(-t / 1.5));
      }
      // weekly periodic ripple
      const periodic = 1.5 * Math.sin(2 * Math.PI * x / 7);
      const latent = baseline + adaptation + progShift - fatigue + illness + periodic;
      const sigma_obs = Math.max(2, 8 - 3 * (x / 180));
      const noise = gaussianFrom(rng) * sigma_obs;
      let regime = 'baseline';
      if (x >= illnessStart && x < illnessStart + 14) regime = 'illness';
      else if (x >= 120) regime = 'program-B';
      else if (x >= illnessStart + 14) regime = 'recovery';
      return { x, y: latent + noise, sigma_obs, regime };
    },
    reset() {
      rng = mulberry32(seed);
      x = 0; adaptation = 0; fatigue = 0;
      illnessActive = false; programShifted = false;
    }
  };
}

// ─────────── CONTINUITY LAYER scenario ───────────
// An eight-layer coupled system (Intelligence, Energy, Economic, Information,
// Governance, Social, Supply-chain, Security). The coupled dynamics + coherence
// metrics live in coupling.js; here we wire them into the mode contract: the
// engine's scalar `y` is the cross-layer continuity index in [0,1], while the
// full per-layer state rides along on an additive `layers` field that only the
// Continuity panel reads. Fragmentation (low coherence) is observed more noisily.
export function createContinuityScenario(seed = 42) {
  const L = 8;
  const sys = createCoupledSystem(seed, { L });
  const coh = createCoherenceTracker({ L, window: 20, corrThreshold: 0.45, metric: 'corr' });
  // The engine reasons over a 0–100 continuity SCORE rather than the raw [0,1]
  // index: a sub-unit-variance signal has negative differential entropy, which
  // degenerates the entropy-threshold decision policy (everything reads as
  // "confident enough to act"). Scoring to 0–100 puts entropies in the same
  // positive regime as the other modes so Stabilise / Hold / Probe all engage.
  // The coupling panel still shows the normalised [0,1] coherence on its gauge.
  const SCALE = 100;
  let x = 0;
  return {
    seed,
    coupling: sys.C,          // the (static) coupling matrix, for the panel topology
    layerIds: sys.layerIds,
    setMetric(m) { coh.setMetric(m); },
    getMetric() { return coh.getMetric(); },
    next() {
      x += 1;
      const s = sys.step();
      const r = coh.push(s.levels);
      const idx = r.index;
      // less coherent ⇒ telemetry is harder to trust ⇒ wider observation noise
      const sigma_obs = (0.015 + 0.05 * (1 - idx)) * SCALE;
      return {
        x, y: idx * SCALE, sigma_obs,
        regime: dominantRegime(s.regimes),
        // additive, Continuity-only — ignored by the 1-D engine, read by the panel
        layers: {
          levels: s.levels, deltas: s.deltas, regimes: s.regimes,
          corr: r.corr, meanCorr: r.meanCorr, corrIndex: r.corrIndex,
          fiedler: r.fiedler, fiedlerIndex: r.fiedlerIndex,
          index: idx, ready: r.ready,
          shockFired: s.shockFired, shockLayer: s.shockLayer
        }
      };
    },
    reset() { sys.reset(); coh.reset(); x = 0; }
  };
}

// ─────────── Inline SVG icons (24×24, currentColor) ───────────
const ICON_GLIDEPATH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18 C 6 15, 9 8, 12 9 S 18 14, 21 5"/><circle cx="3" cy="18" r="1.3" fill="currentColor"/><circle cx="12" cy="9" r="1.3" fill="currentColor"/><circle cx="21" cy="5" r="1.3" fill="currentColor"/></svg>`;
const ICON_TRADING = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 14 L 6 12 L 8 16 L 11 7 L 14 13 L 17 9 L 21 11"/><path d="M3 20 L 21 20" opacity="0.4"/></svg>`;
const ICON_HEALTH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12 L 7 12 L 9 7 L 12 17 L 14 10 L 16 12 L 21 12"/></svg>`;
const ICON_CONTINUITY = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 6 L12 12 L19 6 M5 18 L12 12 L19 18 M12 12 L12 12"/><circle cx="5" cy="6" r="1.5" fill="currentColor"/><circle cx="19" cy="6" r="1.5" fill="currentColor"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/><circle cx="5" cy="18" r="1.5" fill="currentColor"/><circle cx="19" cy="18" r="1.5" fill="currentColor"/></svg>`;

// ─────────── MODES ───────────
export const MODES = [
  {
    id: 'glidepath',
    name: 'Glidepath',
    accent: '#4fd1c7',
    tagline: 'Multi-decade wealth trajectory under regime-switching markets.',
    description: 'A fiduciary planner ingests a client\'s realised real-wealth path across accumulation, sequence-of-returns risk, and decumulation regimes. The engine decides when to rebalance, when to hold through noise, and when to request fresh cash-flow data.',
    icon: ICON_GLIDEPATH,
    actLine: 'epistemic uncertainty has collapsed enough to justify a rebalance.',
    deferLine: 'aleatoric market noise dominates — trading would burn alpha.',
    queryLine: 'a fresh cash-flow update would meaningfully tighten the posterior.',
    domain: {
      xLabel: 'Years from now',
      yLabel: 'Real log-wealth',
      xUnits: 'yr',
      yUnits: 'log(USD)',
      xRange: [0, 30],
      yRange: [-2, 4]
    },
    noiseSemantics: {
      label: 'Market volatility (annualised σ)',
      aleatoric: 'Irreducible market noise — short-term price fluctuations no amount of data smooths out.',
      epistemic: 'Reducible planner uncertainty — narrows as realised returns + client behaviour accumulate.'
    },
    decisions: {
      act:   { label: 'Rebalance',     verb: 'rebalances to target glidepath' },
      defer: { label: 'Hold',          verb: 'holds — noise dominates signal' },
      query: { label: 'Request data',  verb: 'requests updated cash flows' }
    },
    defaults: {
      kernel: 'matern32',
      ell: 4.0,
      eta: 1.2,
      sigma_n: 0.15,
      poolingStrength: 0.5,
      actEntropyMax: 0.4,
      deferNoiseRatio: 0.7,
      queryEigMin: 0.15,
      bf: 10,
      bocpdLambda: 40
    },
    controlOverrides: {
      'slider-ell':    { label: 'Mean-reversion horizon',  tooltip: 'Years over which log-wealth reverts to its long-run drift. ~4 yr matches typical macro cycles.', unit: 'yr' },
      'slider-eta':    { label: 'Drift amplitude',         tooltip: 'Magnitude of the latent log-wealth process. Larger = more pronounced wealth swings.',           unit: 'log-USD' },
      'slider-sigman': { label: 'Market noise σ',          tooltip: 'Annualised volatility of observed log-returns. 0.15 ≈ 15% — long-run equity-like.',              unit: 'vol' },
      'slider-tact':   { label: 'Rebalance threshold',     tooltip: 'Maximum total entropy before planner stops trading and waits for clearer signal.',              unit: 'nats' },
      'slider-tdef':   { label: 'Hold threshold',          tooltip: 'If aleatoric / total entropy ratio exceeds this, defer — more data won\'t sharpen the call.',   unit: 'ratio' },
      'slider-eig':    { label: 'Min EIG to query',        tooltip: 'Expected information gain a new data point must clear before requesting it.',                  unit: 'nats' },
      'slider-bf':     { label: 'Promotion gate (BF)',     tooltip: 'Bayes factor of epistemic entropy reduction needed before D4 writes a new prior version.',      unit: 'BF' },
      'slider-tau':    { label: 'Cohort pooling τ',        tooltip: 'Partial-pooling across the client cohort. 0 = fiduciary firewall, 1 = full pooling.',           unit: '' }
    },
    nodeOverrides: {
      S1: { eyebrow: 'S1 · Client stream',    title: 'Realised wealth path' },
      S2: { eyebrow: 'S2 · Regime detect',    title: 'Paradigm-shift filter' },
      S3: { eyebrow: 'S3 · Features',         title: 'Log-return + drift' },
      P1: { eyebrow: 'P1 · Prior book',       title: 'Planner belief log' },
      P2: { eyebrow: 'P2 · Cohort pooling',   title: 'Cross-client hierarchy' },
      P3: { eyebrow: 'P3 · Kernel',           title: 'Wealth-process kernel' },
      I1: { eyebrow: 'I1 · GP surrogate',     title: 'Wealth posterior' },
      I2: { eyebrow: 'I2 · Ensemble',         title: 'Hyperparameter ensemble' },
      I3: { eyebrow: 'I3 · Uncertainty',      title: 'Market vs planner risk' },
      I4: { eyebrow: 'I4 · Evidence',         title: 'Kernel comparison' },
      D1: { eyebrow: 'D1 · Acquisition',      title: 'Next-data utility' },
      D2: { eyebrow: 'D2 · Policy',           title: 'Rebalance · Hold · Query' },
      D3: { eyebrow: 'D3 · Action log',       title: 'Glidepath rebalances' },
      D4: { eyebrow: 'D4 · Promotion',        title: 'Belief promotion gate' }
    },
    scenario: createGlidepathScenario,
    exportSchema: 'bre1-glidepath/v1',
    exportShape: { horizon_years: 30, decisions_field: 'rebalances' }
  },

  {
    id: 'trading',
    name: 'Trading & Sharpe',
    accent: '#f5b94a',
    tagline: 'Mean-reverting log-returns with volatility clustering — Sharpe-aware.',
    description: 'A discretionary or systematic trader observes a noisy log-return stream with hand-rolled GARCH-style vol clustering. The engine sizes positions when posterior Sharpe clears threshold, sits flat through noise floors, and requests more market data when epistemic dispersion is high.',
    icon: ICON_TRADING,
    actLine: 'posterior Sharpe ratio exceeds your risk budget — size up.',
    deferLine: 'aleatoric market noise dominates — no edge to trade.',
    queryLine: 'epistemic uncertainty is high — wait for confirming bars.',
    domain: {
      xLabel: 'Trading day',
      yLabel: 'Cumulative log-return',
      xUnits: 'd',
      yUnits: 'log',
      xRange: [0, 250],
      yRange: [-0.4, 0.4]
    },
    noiseSemantics: {
      label: 'Tick volatility',
      aleatoric: 'Market noise floor — you can\'t trade your way through it.',
      epistemic: 'Model misspecification — shrinks as you observe more bars.'
    },
    decisions: {
      act:   { label: 'Enter / size up', verb: 'enters or sizes the position up' },
      defer: { label: 'Stay flat',       verb: 'stays flat — noise dominates' },
      query: { label: 'Wait for data',   verb: 'waits for more market data' }
    },
    defaults: {
      kernel: 'spectralmix',
      ell: 8.0,
      eta: 0.15,
      sigma_n: 0.02,
      poolingStrength: 0.2,
      actEntropyMax: 0.2,
      deferNoiseRatio: 0.6,
      queryEigMin: 0.2,
      bf: 8,
      bocpdLambda: 15
    },
    controlOverrides: {
      'slider-ell':    { label: 'Bar lookback',           tooltip: 'Trading-day horizon over which the kernel correlates returns. ~8 d ≈ swing-trading band.',           unit: 'd' },
      'slider-eta':    { label: 'Signal amplitude',       tooltip: 'Magnitude of the latent return process. Smaller values reflect typical equity log-return scale.',   unit: 'log' },
      'slider-sigman': { label: 'Tick noise σ',           tooltip: 'Per-bar observation noise on log-returns. 0.02 ≈ daily equity tick noise floor.',                    unit: 'log' },
      'slider-tact':   { label: 'Entry threshold',        tooltip: 'Maximum total entropy before the engine green-lights an entry.',                                     unit: 'nats' },
      'slider-tdef':   { label: 'Noise-floor threshold',  tooltip: 'If aleatoric / total entropy exceeds this, stay flat — no edge to trade.',                          unit: 'ratio' },
      'slider-eig':    { label: 'Min EIG to wait',        tooltip: 'Expected information gain another bar must clear before the engine asks for more data.',            unit: 'nats' },
      'slider-bf':     { label: 'Regime promotion (BF)',  tooltip: 'Bayes factor of epistemic reduction needed before a new vol regime gets logged.',                    unit: 'BF' },
      'slider-tau':    { label: 'Cross-asset pooling τ',  tooltip: 'Partial-pooling across correlated tickers. 0 = ticker-siloed, 1 = full pooling.',                    unit: '' }
    },
    nodeOverrides: {
      S1: { eyebrow: 'S1 · Tape',             title: 'Bar ingest' },
      S2: { eyebrow: 'S2 · Vol regime',       title: 'Cluster detector' },
      S3: { eyebrow: 'S3 · Features',         title: 'Return + drawdown' },
      P1: { eyebrow: 'P1 · Strategy book',    title: 'Versioned thesis log' },
      P2: { eyebrow: 'P2 · Cross-asset',      title: 'Correlation hierarchy' },
      P3: { eyebrow: 'P3 · Kernel',           title: 'Spectral-mixture kernel' },
      I1: { eyebrow: 'I1 · GP surrogate',     title: 'Return posterior' },
      I2: { eyebrow: 'I2 · Ensemble',         title: 'Ensemble + Sharpe' },
      I3: { eyebrow: 'I3 · Uncertainty',      title: 'Market vs model risk' },
      I4: { eyebrow: 'I4 · Evidence',         title: 'Kernel comparison' },
      D1: { eyebrow: 'D1 · Acquisition',      title: 'Next-bar utility' },
      D2: { eyebrow: 'D2 · Policy',           title: 'Enter · Flat · Wait' },
      D3: { eyebrow: 'D3 · Execution',        title: 'Order log + max DD' },
      D4: { eyebrow: 'D4 · Promotion',        title: 'Regime promotion gate' }
    },
    scenario: createTradingScenario,
    exportSchema: 'bre1-trading/v1',
    exportShape: { trading_days: 250, decisions_field: 'orders' }
  },

  {
    id: 'health',
    name: 'Health & Longevity',
    accent: '#8b7dd8',
    tagline: 'Daily wearables data, training response, adaptation vs fatigue.',
    description: 'A coach observes a noisy daily HRV signal under varying training loads. The engine recommends increasing load when adaptation is detectable above baseline, holding through high-noise weeks, and pulling additional biomarkers when the answer matters.',
    icon: ICON_HEALTH,
    actLine: 'adaptation index has cleared baseline — progressive overload is justified.',
    deferLine: 'day-to-day variability dominates — your signal is too noisy to act on.',
    queryLine: 'a sleep score or cortisol reading would tighten the posterior cheaply.',
    domain: {
      xLabel: 'Day of training block',
      yLabel: 'HRV (ms)',
      xUnits: 'd',
      yUnits: 'ms',
      xRange: [0, 180],
      yRange: [40, 100]
    },
    noiseSemantics: {
      label: 'Daily HRV noise floor',
      aleatoric: 'Day-to-day variability — the floor of biological noise.',
      epistemic: 'Adaptation uncertainty — am I actually getting fitter?'
    },
    decisions: {
      act:   { label: 'Increase load',         verb: 'green-lights a load increase' },
      defer: { label: 'Maintain',              verb: 'holds load — signal is noisy' },
      query: { label: 'Pull biomarker',        verb: 'requests an extra biomarker (sleep, cortisol)' }
    },
    defaults: {
      kernel: 'periodic',
      ell: 7.0,
      eta: 6.0,
      sigma_n: 5.0,
      poolingStrength: 0.4,
      actEntropyMax: 1.0,
      deferNoiseRatio: 0.65,
      queryEigMin: 0.3,
      bf: 12,
      bocpdLambda: 30
    },
    controlOverrides: {
      'slider-ell':    { label: 'Adaptation horizon',     tooltip: 'Days over which physiological response correlates. ~7 d aligns with weekly training cycles.', unit: 'd' },
      'slider-eta':    { label: 'Response amplitude',     tooltip: 'Magnitude of latent HRV process in ms.',                                                       unit: 'ms' },
      'slider-sigman': { label: 'Daily HRV noise',        tooltip: 'Per-day measurement noise in ms. Drops as the client adheres to their measurement protocol.', unit: 'ms' },
      'slider-tact':   { label: 'Load-increase threshold',tooltip: 'Maximum entropy before coach green-lights progressive overload.',                              unit: 'nats' },
      'slider-tdef':   { label: 'Maintain threshold',     tooltip: 'If day-to-day variability dominates total entropy, hold the current load.',                   unit: 'ratio' },
      'slider-eig':    { label: 'Min EIG to pull marker', tooltip: 'Expected information gain a biomarker pull must justify.',                                    unit: 'nats' },
      'slider-bf':     { label: 'Program-change gate (BF)',tooltip: 'Bayes factor of epistemic reduction needed before logging a new training-program version.', unit: 'BF' },
      'slider-tau':    { label: 'Athlete pooling τ',      tooltip: 'Partial-pooling across athletes in cohort. 0 = N=1, 1 = full pooling.',                       unit: '' }
    },
    nodeOverrides: {
      S1: { eyebrow: 'S1 · Wearable feed',    title: 'Daily HRV ingest' },
      S2: { eyebrow: 'S2 · Illness detect',   title: 'Sudden-drop filter' },
      S3: { eyebrow: 'S3 · Features',         title: 'HRV + load + sleep' },
      P1: { eyebrow: 'P1 · Coaching log',     title: 'Program version book' },
      P2: { eyebrow: 'P2 · Athlete cohort',   title: 'Cross-athlete hierarchy' },
      P3: { eyebrow: 'P3 · Kernel',           title: 'Matérn-3/2 + Periodic-7d' },
      I1: { eyebrow: 'I1 · GP surrogate',     title: 'Adaptation posterior' },
      I2: { eyebrow: 'I2 · Ensemble',         title: 'Hyperparameter ensemble' },
      I3: { eyebrow: 'I3 · Uncertainty',      title: 'Variability vs adaptation' },
      I4: { eyebrow: 'I4 · Evidence',         title: 'Kernel comparison' },
      D1: { eyebrow: 'D1 · Acquisition',      title: 'Next-marker utility' },
      D2: { eyebrow: 'D2 · Policy',           title: 'Load · Maintain · Pull' },
      D3: { eyebrow: 'D3 · Coaching log',     title: 'Load decisions' },
      D4: { eyebrow: 'D4 · Promotion',        title: 'Program promotion gate' }
    },
    scenario: createHealthScenario,
    exportSchema: 'bre1-health/v1',
    exportShape: { block_days: 180, decisions_field: 'coaching_log' }
  },

  {
    id: 'continuity',
    name: 'Continuity Layer',
    accent: '#6aa6e0',
    tagline: 'Eight coupled layers — coherence under cascade and fragmentation.',
    description: 'An operator watches eight interdependent layers — Intelligence, Energy, Economic, Information, Governance, Social, Supply-chain, Security — that move together until a shock cascades through their coupling. The engine reasons over a single cross-layer continuity index: it stabilises when coherence is collapsing, holds while the system is self-maintaining, and probes a layer when fresh telemetry would tighten the call.',
    icon: ICON_CONTINUITY,
    actLine: 'epistemic uncertainty has collapsed enough to commit a stabilising intervention.',
    deferLine: 'cross-layer churn dominates — hold; the system is self-maintaining.',
    queryLine: 'epistemic uncertainty is high — fresh telemetry from a layer would tighten the posterior.',
    domain: {
      xLabel: 'Tick',
      yLabel: 'Continuity score',
      xUnits: 't',
      yUnits: '/100',
      xRange: [0, 300],
      yRange: [40, 100]
    },
    noiseSemantics: {
      label: 'Layer telemetry noise',
      aleatoric: 'Irreducible cross-layer churn — the system\'s baseline jitter no amount of data removes.',
      epistemic: 'Reducible operator uncertainty — narrows as the coherence history accumulates.'
    },
    decisions: {
      act:   { label: 'Stabilise', verb: 'commits a stabilising intervention' },
      defer: { label: 'Hold',      verb: 'holds — coherence is self-maintaining' },
      query: { label: 'Probe',     verb: 'probes a layer for fresh telemetry' }
    },
    defaults: {
      kernel: 'matern52',
      ell: 14.0,
      eta: 15.0,
      sigma_n: 3.0,
      poolingStrength: 0.4,
      actEntropyMax: 3.0,
      deferNoiseRatio: 0.55,
      queryEigMin: 0.3,
      bf: 4,
      bocpdLambda: 15,
      // Change-point prior matched to the 0–100 continuity-score scale (mean ≈ 85).
      bocpdPrior: { mu0: 90, kappa0: 1, alpha0: 2, beta0: 20 }
    },
    controlOverrides: {
      'slider-ell':    { label: 'Coherence horizon',     tooltip: 'Ticks over which the continuity index stays correlated. Sets how far the engine smooths.',          unit: 't' },
      'slider-eta':    { label: 'Continuity amplitude',  tooltip: 'Magnitude of the latent coherence process on the 0–100 continuity-score scale.',                      unit: '/100' },
      'slider-sigman': { label: 'Telemetry noise σ',     tooltip: 'Observation noise on the continuity score. Widens automatically as the system fragments.',            unit: '/100' },
      'slider-tact':   { label: 'Stabilise threshold',   tooltip: 'Maximum total entropy before the engine green-lights a stabilising intervention.',                    unit: 'nats' },
      'slider-tdef':   { label: 'Hold threshold',        tooltip: 'If aleatoric / total entropy exceeds this, hold — the system is self-maintaining.',                  unit: 'ratio' },
      'slider-eig':    { label: 'Min EIG to probe',      tooltip: 'Expected information gain a layer probe must clear before the engine requests it.',                  unit: 'nats' },
      'slider-bf':     { label: 'Continuity prior gate (BF)', tooltip: 'Bayes factor of epistemic reduction needed before D4 germinates a new continuity prior.',        unit: 'BF' },
      'slider-tau':    { label: 'Cross-layer pooling τ', tooltip: 'Partial-pooling across the eight layers. 0 = layers reasoned in isolation, 1 = full pooling.',        unit: '' }
    },
    nodeOverrides: {
      S1: { eyebrow: 'S1 · See reality',          title: 'Eight-layer telemetry' },
      S2: { eyebrow: 'S2 · Cascade detect',       title: 'Fragmentation filter' },
      S3: { eyebrow: 'S3 · Features',             title: 'Coherence features' },
      P1: { eyebrow: 'P1 · Continuity priors',    title: 'Seeded prior book' },
      P2: { eyebrow: 'P2 · Relationships',        title: 'Cross-layer coupling' },
      P3: { eyebrow: 'P3 · Kernel',               title: 'Coherence kernel' },
      I1: { eyebrow: 'I1 · GP surrogate',         title: 'Continuity posterior' },
      I2: { eyebrow: 'I2 · Ensemble',             title: 'Hyperparameter ensemble' },
      I3: { eyebrow: 'I3 · Uncertainty',          title: 'Systemic vs model risk' },
      I4: { eyebrow: 'I4 · Evidence',             title: 'Kernel comparison' },
      D1: { eyebrow: 'D1 · Acquisition',          title: 'Next-probe utility' },
      D2: { eyebrow: 'D2 · Policy',               title: 'Stabilise · Hold · Probe' },
      D3: { eyebrow: 'D3 · Action log',           title: 'Stabilisation log' },
      D4: { eyebrow: 'D4 · Germination',          title: 'Continuity prior gate' }
    },
    scenario: createContinuityScenario,
    exportSchema: 'bre1-continuity/v1',
    exportShape: { layers: 8, decisions_field: 'stabilizations' }
  }
];

export const DEFAULT_MODE_ID = 'glidepath';

export function getModeById(id) {
  return MODES.find(m => m.id === id) || MODES[0];
}
