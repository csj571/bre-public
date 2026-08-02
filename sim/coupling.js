// coupling.js — coupled multi-layer dynamics + cross-layer coherence metrics.
//
// A genuinely multi-dimensional companion to the 1-D engine: it simulates L
// coupled "layers" (a Continuity-Layer system) and collapses their cross-layer
// relationship structure into a single scalar **continuity index** in [0,1].
// That index is what the existing 1-D engine (GP / BOCPD / entropy / decision)
// consumes as its `y` — so none of the engine modules need to change. The full
// per-layer state is exposed only for visualization.
//
// Pure, deterministic (seeded Mulberry32), zero-dependency. Self-contained copy
// of the RNG helpers to match the house style in modes.js.
//
// ── Symbol glossary ───────────────────────────────────────────────────────────
//   levels[i]     current level of layer i (deviation tracked against baseline 0)
//   C[i][j]       coupling: how much layer j's deviation pushes into layer i
//   selfPersist   AR(1) decay of each layer's own deviation per step (<1 ⇒ stable)
//   shock         occasional impulse injected into one layer; propagates via C (a cascade)
//   corr[i][j]    windowed Pearson correlation between layers i and j
//   meanCorr      mean off-diagonal correlation ∈ [-1,1] — "how together layers move"
//   fiedler       algebraic connectivity (2nd-smallest Laplacian eigenvalue) of the
//                 thresholded |corr| graph — collapses sharply when the system fragments
//   index         continuity index ∈ [0,1]; driven by meanCorr (default) or fiedler
// ───────────────────────────────────────────────────────────────────────────────

// ---- deterministic RNG (own copy, matches modes.js) ----
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussianFrom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ---- layer identity (the Continuity-Layer 8) ----
export const LAYER_IDS = ['INT', 'ENE', 'ECO', 'INF', 'GOV', 'SOC', 'SUP', 'SEC'];
export const LAYER_LABELS = {
  INT: 'Intelligence', ENE: 'Energy', ECO: 'Economic', INF: 'Information',
  GOV: 'Governance', SOC: 'Social', SUP: 'Supply-chain', SEC: 'Security'
};

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

// ---- coupling matrix ----
// C[i][j] (i≠j) = directed influence of layer j on layer i. Diagonal = 0 (self
// persistence is applied separately). Each row's off-diagonal weights are scaled
// to a coupling budget so the vector AR(1) stays stable.
export function makeCouplingMatrix(seed, opts = {}) {
  const L = opts.L || 8;
  const density = opts.density ?? 0.55;
  const maxWeight = opts.maxWeight ?? 0.5;
  const budget = opts.budget ?? 0.6;       // max total incoming off-diagonal weight per row
  const asymmetry = opts.asymmetry ?? 0.35;
  const rng = mulberry32((seed >>> 0) ^ 0x9e3779b9);
  const C = Array.from({ length: L }, () => new Array(L).fill(0));
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      if (rng() < density) {
        const base = rng() * maxWeight;
        // directional asymmetry so i→j and j→i differ
        const dir = 1 + asymmetry * (rng() - 0.5) * 2;
        C[i][j] = base * dir;
      }
    }
    // scale row to the coupling budget
    let rowSum = 0;
    for (let j = 0; j < L; j++) rowSum += Math.abs(C[i][j]);
    if (rowSum > budget) {
      const s = budget / rowSum;
      for (let j = 0; j < L; j++) C[i][j] *= s;
    }
  }
  return C;
}

// ---- coupled-system simulator (mirrors the modes.js scenario shape) ----
// Each layer tracks a shared common factor plus an idiosyncratic deviation:
//     level_i = loading_i · commonFactor + dev_i
// In calm periods the common factor dominates, so the layers move together and
// are highly cross-correlated → high coherence. A shock injects a large dev into
// one layer and CASCADES through the coupling matrix C into its neighbours; those
// layers' motion is then driven by their own shock rather than the common factor,
// so they DECOUPLE from the rest → cross-correlation (and the continuity index)
// drops. Deviations then decay and coherence recovers.
//
// Stability: dev is an AR process with decay devDecay plus coupling injection
// gain·budget; devDecay + gain·budget < 1 keeps the deviation field contracting.
export function createCoupledSystem(seed = 42, opts = {}) {
  const L = opts.L || 8;
  const phiC = opts.phiC ?? 0.9;          // common-factor persistence (OU)
  const sigmaC = opts.sigmaC ?? 0.12;     // common-factor innovation scale
  const devDecay = opts.devDecay ?? 0.85; // idiosyncratic deviation decay (slow ⇒ longer dips)
  const couplingGain = opts.couplingGain ?? 0.22; // how strongly dev cascades via C
  const sigmaDev = opts.sigmaDev ?? 0.06;         // idiosyncratic noise floor
  const layerIds = LAYER_IDS.slice(0, L);

  let C = makeCouplingMatrix(seed, { ...opts, L });
  const outInfluence = new Array(L).fill(0);
  for (let j = 0; j < L; j++) for (let i = 0; i < L; i++) outInfluence[j] += Math.abs(C[i][j]);

  let rng, common, dev, levels, t, nextShockAt;
  // per-layer loadings on the common factor (deterministic, all positive ⇒ coherent baseline)
  const loadings = new Array(L);
  { const lr = mulberry32((seed >>> 0) ^ 0x51ed270b);
    for (let i = 0; i < L; i++) loadings[i] = 0.65 + 0.35 * lr(); }

  function pickShockLayer() {
    // weighted by out-influence so central layers cause bigger cascades
    let total = 0;
    for (let j = 0; j < L; j++) total += outInfluence[j] + 0.05;
    let r = rng() * total;
    for (let j = 0; j < L; j++) { r -= outInfluence[j] + 0.05; if (r <= 0) return j; }
    return L - 1;
  }
  function scheduleShock() { nextShockAt = t + 45 + Math.floor(rng() * 45); }

  function init() {
    rng = mulberry32(seed >>> 0);
    common = 0;
    dev = new Array(L).fill(0);
    levels = new Array(L).fill(0);
    t = 0;
    scheduleShock();
  }
  init();

  function step() {
    t += 1;
    const prevLevels = levels.slice();
    const prevDev = dev.slice();
    let shockFired = false, shockLayer = null;
    const shock = new Array(L).fill(0);
    if (t >= nextShockAt) {
      shockLayer = pickShockLayer();
      shock[shockLayer] = (1.8 + rng() * 1.6) * (rng() < 0.5 ? -1 : 1);
      shockFired = true;
      scheduleShock();
    }
    // shared common factor (OU)
    common = phiC * common + sigmaC * gaussianFrom(rng);
    // idiosyncratic deviations: decay + cascade injection + shock + floor noise
    const nextDev = new Array(L), next = new Array(L);
    for (let i = 0; i < L; i++) {
      let coupled = 0;
      for (let j = 0; j < L; j++) coupled += C[i][j] * prevDev[j];
      nextDev[i] = clamp(devDecay * prevDev[i] + couplingGain * coupled + shock[i] + sigmaDev * gaussianFrom(rng), -12, 12);
      next[i] = clamp(loadings[i] * common + nextDev[i], -14, 14);
    }
    dev = nextDev;
    levels = next;
    const deltas = new Array(L);
    const regimes = new Array(L);
    for (let i = 0; i < L; i++) {
      deltas[i] = next[i] - prevLevels[i];
      // a layer fragments when its idiosyncratic deviation dwarfs the noise floor
      const d = Math.abs(nextDev[i]);
      regimes[i] = d < 3 * sigmaDev ? 'calm' : d < 9 * sigmaDev ? 'stress' : 'fragmenting';
    }
    return { t, levels: levels.slice(), deltas, regimes, shockFired, shockLayer };
  }

  return { step, reset: init, get C() { return C; }, L, layerIds };
}

// Collapse per-layer regimes into one scalar label for the 1-D engine's `regime`.
export function dominantRegime(regimes) {
  let frag = 0, stress = 0;
  for (const r of regimes) { if (r === 'fragmenting') frag++; else if (r === 'stress') stress++; }
  if (frag >= 1) return 'fragmenting';
  if (stress >= 2) return 'stress';
  return 'calm';
}

// ---- pure coherence helpers (unit-tested) ----

// Windowed Pearson correlation matrix. `rows` is W vectors of length L.
export function correlationMatrix(rows) {
  const W = rows.length;
  const L = W ? rows[0].length : 0;
  const mean = new Array(L).fill(0);
  for (let w = 0; w < W; w++) for (let i = 0; i < L; i++) mean[i] += rows[w][i];
  for (let i = 0; i < L; i++) mean[i] /= W;
  const std = new Array(L).fill(0);
  for (let w = 0; w < W; w++) for (let i = 0; i < L; i++) { const d = rows[w][i] - mean[i]; std[i] += d * d; }
  for (let i = 0; i < L; i++) std[i] = Math.sqrt(std[i] / W);
  const M = Array.from({ length: L }, () => new Array(L).fill(0));
  for (let i = 0; i < L; i++) {
    M[i][i] = 1;
    for (let j = i + 1; j < L; j++) {
      if (std[i] < 1e-9 || std[j] < 1e-9) { M[i][j] = M[j][i] = 0; continue; }
      let cov = 0;
      for (let w = 0; w < W; w++) cov += (rows[w][i] - mean[i]) * (rows[w][j] - mean[j]);
      cov /= W;
      const r = clamp(cov / (std[i] * std[j]), -1, 1);
      M[i][j] = M[j][i] = r;
    }
  }
  return M;
}

export function meanOffDiagonal(M) {
  const L = M.length;
  if (L < 2) return 0;
  let s = 0, n = 0;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (i !== j) { s += M[i][j]; n++; }
  return n ? s / n : 0;
}

// Algebraic connectivity (Fiedler value): 2nd-smallest eigenvalue of the
// combinatorial Laplacian of the graph whose edges are |corr| ≥ threshold.
// 0 when the graph is disconnected; grows with how tightly the layers are bound.
// Uses a capped cyclic Jacobi eigensolve (the matrix is tiny, L×L). Finite-guarded.
export function algebraicConnectivity(absCorr, threshold = 0.5) {
  const L = absCorr.length;
  if (L < 2) return 0;
  const A = Array.from({ length: L }, () => new Array(L).fill(0));
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    if (i !== j && absCorr[i][j] >= threshold) A[i][j] = absCorr[i][j];
  }
  const Lap = Array.from({ length: L }, () => new Array(L).fill(0));
  for (let i = 0; i < L; i++) {
    let deg = 0;
    for (let j = 0; j < L; j++) if (i !== j) { deg += A[i][j]; Lap[i][j] = -A[i][j]; }
    Lap[i][i] = deg;
  }
  const eig = jacobiEigenvalues(Lap);
  eig.sort((a, b) => a - b);
  const lambda2 = eig[1];
  return Number.isFinite(lambda2) ? Math.max(0, lambda2) : 0;
}

// Cyclic Jacobi eigenvalue iteration for a small symmetric matrix.
function jacobiEigenvalues(Min) {
  const n = Min.length;
  const A = Min.map(r => r.slice());
  for (let sweep = 0; sweep < 40; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-14) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const tsign = theta >= 0 ? 1 : -1;
        const tval = tsign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(tval * tval + 1);
        const s = tval * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  const eig = new Array(n);
  for (let i = 0; i < n; i++) eig[i] = A[i][i];
  return eig;
}

// ---- coherence tracker (sliding window over level vectors) ----
// metric: 'corr' (default) drives the index from mean off-diagonal correlation;
// 'fiedler' drives it from algebraic connectivity. Both are always computed so the
// panel can show one as a secondary readout. Switchable live via setMetric().
export function createCoherenceTracker(opts = {}) {
  const L = opts.L || 8;
  const window = opts.window || 40;
  const corrThreshold = opts.corrThreshold ?? 0.5;
  const minSamples = opts.minSamples || 8;
  let metric = opts.metric || 'corr';
  let rows = [];

  function readout() {
    if (rows.length < minSamples) {
      return { index: 0.5, meanCorr: 0, corrIndex: 0.5, fiedler: 0, fiedlerIndex: 0,
        corr: Array.from({ length: L }, (_, i) => Array.from({ length: L }, (_, j) => (i === j ? 1 : 0))),
        ready: false };
    }
    const corr = correlationMatrix(rows);
    const meanCorr = meanOffDiagonal(corr);
    const corrIndex = clamp((1 + meanCorr) / 2, 0, 1);
    const absCorr = corr.map(r => r.map(Math.abs));
    const fiedler = algebraicConnectivity(absCorr, corrThreshold);
    const fiedlerIndex = clamp(fiedler / L, 0, 1);   // normalise λ2 into [0,1]
    const index = metric === 'fiedler' ? fiedlerIndex : corrIndex;
    return { index, meanCorr, corrIndex, fiedler, fiedlerIndex, corr, ready: true };
  }

  return {
    push(levels) {
      rows.push(levels.slice());
      if (rows.length > window) rows.shift();
      return readout();
    },
    reset() { rows = []; },
    setMetric(m) { metric = (m === 'fiedler') ? 'fiedler' : 'corr'; },
    getMetric() { return metric; }
  };
}
