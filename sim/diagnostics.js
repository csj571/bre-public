// diagnostics.js — honest posterior diagnostics + hyperparameter optimisation.
//
// The GP posterior is exact given hyperparameters θ = (ℓ, η, σ_n); there is no
// Markov chain, so chain diagnostics like r̂ are undefined here. What IS well
// defined and informative is how concentrated the *marginal* over θ is — i.e.
// how strongly the data pin down the hyperparameters.
//
// We measure it with an importance-weighted ensemble (node I2): draw θ_k from the
// priors, weight each by its marginal likelihood p(y | X, θ_k), and report the
// Kish effective sample size and the (normalised) weight entropy. This is a real,
// reproducible computation — not an animation.
//
//   weight    w_k ∝ p(y | X, θ_k)             (prior as the importance proposal)
//   ESS       (Σ w_k)² / Σ w_k²   ∈ [1, K]    effective number of contributing draws
//   ensemble  H_w = −Σ p_k log p_k, normalised by log K ∈ [0,1]
//   entropy   ~0 ⇒ one θ dominates (data pin θ down); ~1 ⇒ many θ fit equally well.
//
// Priors (matching the P3 panel): ℓ ~ Gamma(2,1), η ~ HalfN(0,1), σ_n ~ HalfN(0,0.5).

import { Kernels, logEvidence } from './gp.js';

// Small deterministic RNG so a given (data, seed) always yields the same ensemble.
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
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Draw one hyperparameter set from the priors.
function drawFromPriors(rng) {
  // Gamma(shape=2, scale=1) = sum of two Exp(1) draws.
  const ell = -Math.log(rng() + 1e-12) - Math.log(rng() + 1e-12);
  const eta = Math.abs(gaussian(rng) * 1.0);     // HalfN(0,1)
  const sigma_n = Math.abs(gaussian(rng) * 0.5); // HalfN(0,0.5)
  return {
    ell: Math.min(20, Math.max(0.1, ell)),
    eta: Math.min(15, Math.max(0.05, eta)),
    sigma_n: Math.min(15, Math.max(0.005, sigma_n)),
    period: 4
  };
}

// Importance-weighted hyperparameter ensemble. Returns { ess, entropyNorm, count,
// best } where `best` is the highest-evidence draw (a warm start for optimisation).
export function hyperEnsemble(X, y, kernelName, opts = {}) {
  const count = opts.count || 16;
  const kfun = Kernels[kernelName] || Kernels.matern52;
  if (!X || X.length < 2) return { ess: 1, entropyNorm: 0, count, best: null };
  const rng = mulberry32(opts.seed || 1234);
  const draws = [];
  let maxLE = -Infinity;
  for (let k = 0; k < count; k++) {
    const hp = drawFromPriors(rng);
    const le = logEvidence(X, y, kfun, hp);
    if (Number.isFinite(le) && le > maxLE) maxLE = le;
    draws.push({ hp, le });
  }
  if (!Number.isFinite(maxLE)) return { ess: 1, entropyNorm: 0, count, best: null };
  let sw = 0;
  for (const d of draws) { d.w = Number.isFinite(d.le) ? Math.exp(d.le - maxLE) : 0; sw += d.w; }
  if (sw <= 0) return { ess: 1, entropyNorm: 0, count, best: null };
  let sumP2 = 0, ent = 0, best = draws[0];
  for (const d of draws) {
    const p = d.w / sw;
    sumP2 += p * p;
    if (p > 0) ent -= p * Math.log(p);
    if (d.le > best.le) best = d;
  }
  return {
    ess: 1 / sumP2,                                   // = (Σw)²/Σw²
    entropyNorm: count > 1 ? ent / Math.log(count) : 0,
    count,
    best: { hp: best.hp, logEvidence: best.le }
  };
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

// Type-II maximum likelihood: coordinate ascent on the log marginal likelihood
// over (ℓ, η, σ_n). Cheap and bounded — a few multiplicative line searches per
// parameter, warm-started from the current hyperparameters. Reuses logEvidence.
export function optimizeHyperparameters(X, y, kernelName, hp0, opts = {}) {
  const kfun = Kernels[kernelName] || Kernels.matern52;
  if (!X || X.length < 2) return { hp: { ...hp0 }, logEvidence: -Infinity };
  const bounds = { ell: [0.1, 20], eta: [0.05, 15], sigma_n: [0.005, 15] };
  const factors = [0.5, 0.7, 0.85, 1.18, 1.4, 2.0];
  const passes = opts.passes || 5;
  let best = { ...hp0 };
  let bestLE = logEvidence(X, y, kfun, best);
  for (let pass = 0; pass < passes; pass++) {
    let improved = false;
    for (const p of ['ell', 'eta', 'sigma_n']) {
      for (const f of factors) {
        const trial = { ...best, [p]: clamp(best[p] * f, bounds[p][0], bounds[p][1]) };
        const le = logEvidence(X, y, kfun, trial);
        if (le > bestLE + 1e-9) { bestLE = le; best = trial; improved = true; }
      }
    }
    if (!improved) break;
  }
  return { hp: best, logEvidence: bestLE };
}
