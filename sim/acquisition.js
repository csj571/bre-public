// acquisition.js — acquisition functions that score "where to sample next".
//
// All acquisitions assume MAXIMISATION of the latent function f(x). Each takes
// the GP posterior over a grid and returns a Float64Array of utilities; `argmax`
// then picks the next query point.
//
// ── Symbol glossary ───────────────────────────────────────────────────────────
//   mu[i]      posterior mean of f at grid point i
//   sigmaF[i]  posterior epistemic std-dev of f at i (uncertainty about f itself)
//   sigmaY[i]  predictive std-dev incl. observation noise, √(σ_f² + σ_n²)
//   fBest      best objective value observed so far (the incumbent)
//   beta       UCB exploration weight (std-devs above the mean)
//   Hepi[i]    epistemic entropy at i (mutual information I(y; f)) — see entropy.js
//   y*         a sampled value of the global maximum max_x f(x) (used by MES)
//   Φ, φ       standard-normal CDF and PDF
// ───────────────────────────────────────────────────────────────────────────────

const SQRT2 = Math.sqrt(2);

function erf(x) {
  // Abramowitz-Stegun 7.1.26 rational approximation (|error| < 1.5e-7).
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

const stdNormCDF = (x) => 0.5 * (1 + erf(x / SQRT2));
const stdNormPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

// EI: Expected Improvement (maximization).
// EI(x) = (μ−f*)Φ(z) + σ·φ(z),  z = (μ−f*)/σ.
export function expectedImprovement(mu, sigmaY, fBest) {
  const N = mu.length;
  const a = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const s = sigmaY[i];
    if (s < 1e-9) { a[i] = 0; continue; }
    const z = (mu[i] - fBest) / s;
    a[i] = (mu[i] - fBest) * stdNormCDF(z) + s * stdNormPDF(z);
  }
  return a;
}

// UCB: Upper Confidence Bound. UCB(x) = μ + β·σ.
export function upperConfidenceBound(mu, sigmaY, beta = 2.0) {
  const N = mu.length;
  const a = new Float64Array(N);
  for (let i = 0; i < N; i++) a[i] = mu[i] + beta * sigmaY[i];
  return a;
}

// BALD: query where the model is most epistemically uncertain.
// Equals the epistemic entropy (mutual information between y and f).
export function bald(Hepi) {
  return Float64Array.from(Hepi);
}

// ── Max-value Entropy Search (Wang & Jegelka, ICML 2017) ───────────────────────
// A real information-theoretic acquisition: pick x that, in expectation, most
// reduces the entropy of the distribution over the *maximum value* y* = max_x f(x).
//
//   α_MES(x) = (1/M) Σ_{y*} [ γφ(γ)/(2Φ(γ)) − logΦ(γ) ],   γ = (y* − μ(x))/σ_f(x)
//
// The max-values y* are sampled from the GP's implied max-distribution via the
// Gumbel approximation: the CDF of the max is F*(y) = Πᵢ Φ((y−μᵢ)/σᵢ); we read
// off its quartiles by bisection, fit a Gumbel, and draw quasi-random quantiles
// (deterministic — given the same posterior, the acquisition is reproducible).
export function maxValueEntropySearch(mu, sigmaF, opts = {}) {
  const N = mu.length;
  const a = new Float64Array(N);
  if (N === 0) return a;
  const M = opts.samples || 12;
  const ys = sampleMaxValues(mu, sigmaF, M);
  if (!ys.length) return a;
  for (let i = 0; i < N; i++) {
    const s = sigmaF[i];
    if (s < 1e-9) { a[i] = 0; continue; }
    let acc = 0;
    for (let m = 0; m < ys.length; m++) {
      const gamma = (ys[m] - mu[i]) / s;
      const cdf = Math.max(stdNormCDF(gamma), 1e-10);
      const pdf = stdNormPDF(gamma);
      acc += (gamma * pdf) / (2 * cdf) - Math.log(cdf);
    }
    a[i] = acc / ys.length;
  }
  return a;
}

// Sample M plausible values of the global maximum via the Gumbel approximation.
function sampleMaxValues(mu, sigmaF, M) {
  const N = mu.length;
  let maxMean = -Infinity, lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N; i++) {
    const s = Math.max(sigmaF[i], 1e-6);
    if (mu[i] > maxMean) maxMean = mu[i];
    if (mu[i] - 1 < lo) lo = mu[i] - 1;
    if (mu[i] + 6 * s + 1 > hi) hi = mu[i] + 6 * s + 1;
  }
  // log CDF of the max at y: Σᵢ log Φ((y−μᵢ)/σᵢ)
  const logMaxCDF = (y) => {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const cdf = stdNormCDF((y - mu[i]) / Math.max(sigmaF[i], 1e-6));
      s += Math.log(Math.max(cdf, 1e-300));
    }
    return s;
  };
  // Bisect for the y where the max-CDF equals probability p.
  const quantile = (p) => {
    const target = Math.log(p);
    let a = lo, b = hi;
    for (let it = 0; it < 40; it++) {
      const mid = 0.5 * (a + b);
      if (logMaxCDF(mid) < target) a = mid; else b = mid;
    }
    return 0.5 * (a + b);
  };
  const y25 = quantile(0.25), y50 = quantile(0.50), y75 = quantile(0.75);
  let b = (y75 - y25) / 1.57253;            // Gumbel scale from the IQR
  if (!(b > 1e-6)) b = Math.max(1e-3, Math.abs(y50) * 0.05 + 1e-3);
  const aGum = y50 + 0.3665 * b;            // Gumbel location
  const ys = [];
  for (let m = 0; m < M; m++) {
    const r = (m + 0.5) / M;                // deterministic stratified quantiles
    let yStar = aGum - b * Math.log(-Math.log(r));
    if (yStar < maxMean + 1e-3) yStar = maxMean + 1e-3;  // a max must exceed the mean surface
    ys.push(yStar);
  }
  return ys;
}

export function argmax(arr) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bv) { bv = arr[i]; bi = i; }
  }
  return { index: bi, value: bv };
}
