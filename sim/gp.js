// gp.js — Cholesky-based Gaussian Process regression. Pure functions, ES module.
//
// A GP places a prior over functions; conditioning on observations (X, y) gives a
// closed-form Gaussian posterior over f at any test point. Everything here is
// exact given the hyperparameters — there is no sampling.
//
// ── Hyperparameter glossary (the `hp` object) ──────────────────────────────────
//   ell      lengthscale ℓ — distance over which f stays correlated. Larger ℓ ⇒
//            smoother, more slowly-varying fits.
//   eta      amplitude η — prior std-dev of f. η² is the kernel's value at zero
//            distance, i.e. the marginal prior variance of f(x).
//   sigma_n  observation-noise std-dev σ_n — added in quadrature on the kernel
//            diagonal; sets how far the fit is allowed to miss each data point.
//   period   p — repeat length, used only by the periodic kernel.
//
// ── Other symbols ──────────────────────────────────────────────────────────────
//   r            lag |x1 − x2| between two inputs (kernels are stationary in r,
//                except `linear_matern52`).
//   L            lower-triangular Cholesky factor of K(X,X)+σ_n²I, so K = L Lᵀ.
//   alpha        K⁻¹y, the dual weights used for the posterior mean.
//   logEvidence  log marginal likelihood log p(y | X, hp) — the model's own score
//                for how well these hyperparameters explain the data.
//   JITTER       tiny diagonal nugget (1e-6) for numerical positive-definiteness.
// ───────────────────────────────────────────────────────────────────────────────

const JITTER = 1e-6;

// ---------- Kernels ----------
// Each kernel takes (x1, x2, hp) where hp = { ell, eta, ... } and returns scalar covariance.

export const Kernels = {
  rbf(x1, x2, hp) {
    const d = x1 - x2;
    return hp.eta * hp.eta * Math.exp(-0.5 * d * d / (hp.ell * hp.ell));
  },
  matern32(x1, x2, hp) {
    const d = Math.abs(x1 - x2);
    const s = Math.sqrt(3) * d / hp.ell;
    return hp.eta * hp.eta * (1 + s) * Math.exp(-s);
  },
  matern52(x1, x2, hp) {
    const d = Math.abs(x1 - x2);
    const s = Math.sqrt(5) * d / hp.ell;
    return hp.eta * hp.eta * (1 + s + s * s / 3) * Math.exp(-s);
  },
  periodic(x1, x2, hp) {
    const p = hp.period || 4.0;
    const d = Math.PI * Math.abs(x1 - x2) / p;
    const sd = Math.sin(d);
    return hp.eta * hp.eta * Math.exp(-2 * sd * sd / (hp.ell * hp.ell));
  },
  spectralmix(x1, x2, hp) {
    // sum of two RBFs at different scales
    const d = x1 - x2;
    const k1 = hp.eta * hp.eta * 0.6 * Math.exp(-0.5 * d * d / (hp.ell * hp.ell));
    const k2 = hp.eta * hp.eta * 0.4 * Math.exp(-0.5 * d * d / ((hp.ell * 0.3) * (hp.ell * 0.3)));
    return k1 + k2;
  },
  linear_matern52(x1, x2, hp) {
    const linear = 0.05 * x1 * x2;
    return linear + Kernels.matern52(x1, x2, hp);
  },
  // matern52_linear: Matérn-5/2 + small linear-trend kernel.
  // Like linear_matern52 but with a tunable linear coefficient (hp.alpha_lin,
  // default 0.05) — the base kernel the LMC block below shares across output
  // channels. Composite kernel — chosen over a linear mean function because
  // fitGP assumes zero-mean and supports composite kernels trivially.
  matern52_linear(x1, x2, hp) {
    const alpha = (hp && Number.isFinite(hp.alpha_lin)) ? hp.alpha_lin : 0.05;
    return alpha * x1 * x2 + Kernels.matern52(x1, x2, hp);
  }
};

export const KERNEL_NAMES = ['rbf', 'matern32', 'matern52', 'periodic', 'spectralmix', 'linear_matern52', 'matern52_linear'];
export const KERNEL_LABELS = {
  rbf: 'RBF',
  matern32: 'Matérn-3/2',
  matern52: 'Matérn-5/2',
  periodic: 'Periodic',
  spectralmix: 'RBF mixture (2-scale)',  // honest label: a 2-scale RBF sum, not a Wilson-Adams spectral-mixture kernel
  linear_matern52: 'Linear + Matérn-5/2',
  matern52_linear: 'Matérn-5/2 + Linear',
  lmc_matern52_linear: 'LMC · Matérn-5/2 + Linear (bivariate)'
};

// ---------- LMC (Linear Model of Coregionalization) ----------
// Bivariate-output GP. Replaces the composite hack (pre-averaging two distinct
// observation channels into one scalar before the likelihood sees them).
//
// Joint kernel:  K_joint(x,x') = B ⊗ k(x,x',θ)
// where k is matern52_linear (Matérn-5/2 + α·x·x', α = 0.05) and B is a
// 2×2 coregionalization matrix  B = W·Wᵀ + diag(κ).
//
// v1: W and κ are FIXED at initialization (no hyperparameter learning).
// Learning W and κ (e.g. by marginal-likelihood ascent or HMC over the
// coregionalization params) is a v1.1 task — DO NOT add it here.
//
// Engine-level only for now: no mode in this build consumes the LMC yet
// (the retired AI-Project-Portfolio mode was its consumer); the math is
// kept canonical + tested in test.mjs.
export const LMC_INIT = {
  W: [1.0, 0.7],        // column vector W (two output channels)
  kappa: [0.10, 0.10],  // diagonal nugget per channel
  alpha_lin: 0.05       // linear-trend coefficient inside the base kernel k
};

// Build B = W·Wᵀ + diag(κ) as a 2×2 array.
export function lmcMatrixB(W = LMC_INIT.W, kappa = LMC_INIT.kappa) {
  return [
    [W[0] * W[0] + kappa[0], W[0] * W[1]],
    [W[1] * W[0], W[1] * W[1] + kappa[1]]
  ];
}

// ---------- Linear algebra ----------

// Pure Cholesky: returns lower-triangular L with A = L Lᵀ, or null if A is not
// positive-definite. Jitter is the caller's job (see choleskyJitter) — mirroring
// engine/gp.py, which adds escalating jitter to the matrix rather than fabricating
// a value for a non-positive pivot.
export function cholesky(A) {
  const N = A.length;
  const L = [];
  for (let i = 0; i < N; i++) L.push(new Float64Array(N));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 0) return null;        // not PD — signal failure, don't fake it
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

// Recursive-jitter Cholesky, matching engine/gp.py: try A + jitter·I with jitter
// escalating 1e-6 → 1e-1 until positive-definite. Throws if even max jitter fails.
export function choleskyJitter(A, base = JITTER, max = 1e-1) {
  let jitter = base;
  while (jitter < max) {
    const Aj = A.map((row, i) => {
      const r = Float64Array.from(row);
      r[i] += jitter;
      return r;
    });
    const L = cholesky(Aj);
    if (L) return L;
    jitter *= 10;
  }
  throw new Error('Cholesky failed even at max jitter (matrix not positive-definite)');
}

// Forward substitution: solves L x = b
export function forwardSolve(L, b) {
  const N = L.length;
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= L[i][k] * x[k];
    x[i] = sum / L[i][i];
  }
  return x;
}

// Backward substitution: solves L^T x = b
export function backwardSolve(L, b) {
  const N = L.length;
  const x = new Float64Array(N);
  for (let i = N - 1; i >= 0; i--) {
    let sum = b[i];
    for (let k = i + 1; k < N; k++) sum -= L[k][i] * x[k];
    x[i] = sum / L[i][i];
  }
  return x;
}

// Solve A x = b given Cholesky L of A.
export function choleskySolve(L, b) {
  const y = forwardSolve(L, b);
  return backwardSolve(L, y);
}

// ---------- GP fit + predict ----------

// Build K(X1, X2) given kernel function k(x1, x2, hp).
export function buildK(X1, X2, kfun, hp) {
  const N = X1.length, M = X2.length;
  const K = [];
  for (let i = 0; i < N; i++) {
    const row = new Float64Array(M);
    for (let j = 0; j < M; j++) row[j] = kfun(X1[i], X2[j], hp);
    K.push(row);
  }
  return K;
}

// Fit GP: returns {L, alpha, logEvidence}. K_obs = K(X,X) + sigma_n^2 * I.
export function fitGP(X, y, kfun, hp) {
  const N = X.length;
  if (N === 0) return null;
  const K = [];
  for (let i = 0; i < N; i++) {
    const row = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      row[j] = kfun(X[i], X[j], hp);
      if (i === j) row[j] += hp.sigma_n * hp.sigma_n;
    }
    K.push(row);
  }
  const L = choleskyJitter(K);
  const alpha = choleskySolve(L, y);
  // log marginal likelihood
  let logDet = 0;
  for (let i = 0; i < N; i++) logDet += Math.log(L[i][i]);
  let yAlpha = 0;
  for (let i = 0; i < N; i++) yAlpha += y[i] * alpha[i];
  const logEvidence = -0.5 * yAlpha - logDet - 0.5 * N * Math.log(2 * Math.PI);
  return { L, alpha, K, logEvidence, kfun, hp, X, y };
}

// Predict at test points. Returns {mu, sigmaF (epistemic stddev), sigmaY (incl. noise)}.
export function predictGP(model, Xstar) {
  const M = Xstar.length;
  const mu = new Float64Array(M);
  const sigmaF = new Float64Array(M);
  if (!model || !model.L) {
    // No data: prior mean 0, prior std = eta (matches engine/gp.py; was hardcoded 1.0)
    const eta = (model && model.hp && Number.isFinite(model.hp.eta)) ? model.hp.eta : 1.0;
    for (let i = 0; i < M; i++) sigmaF[i] = eta;
    return { mu, sigmaF };
  }
  const { L, alpha, kfun, hp, X } = model;
  const N = X.length;
  for (let i = 0; i < M; i++) {
    const kstar = new Float64Array(N);
    for (let j = 0; j < N; j++) kstar[j] = kfun(Xstar[i], X[j], hp);
    let m = 0;
    for (let j = 0; j < N; j++) m += kstar[j] * alpha[j];
    mu[i] = m;
    // v = L^{-1} k*
    const v = forwardSolve(L, kstar);
    let kss = kfun(Xstar[i], Xstar[i], hp);
    let dot = 0;
    for (let j = 0; j < N; j++) dot += v[j] * v[j];
    let varF = kss - dot;
    if (varF < 0) varF = 0;
    sigmaF[i] = Math.sqrt(varF);
  }
  return { mu, sigmaF };
}

// Log evidence for a candidate kernel given X,y and hp (for BMA bar chart)
export function logEvidence(X, y, kfun, hp) {
  if (X.length === 0) return -Infinity;
  try {
    const m = fitGP(X, y, kfun, hp);
    return m.logEvidence;
  } catch (e) {
    return -Infinity;
  }
}

// ---------- LMC bivariate fit + predict ----------
//
// Observations: X (length N), Yd = first channel, Yc = second channel.
// We stack the two channels into a single 2N joint observation vector
//   y_joint = [Yd_0..Yd_{N-1}, Yc_0..Yc_{N-1}]
// and build the 2N×2N joint covariance
//   K_joint[(p,i),(q,j)] = B[p][q] * k(x_i, x_j)
// plus per-channel observation noise sigma_n^2 on the diagonal.
//
// This is the honest multi-output GP: both channels are observed jointly
// under a shared latent process coupled by B, instead of being averaged
// into a scalar before the likelihood.
export function fitLMC(X, Yd, Yc, kfun, hp, B) {
  const N = X.length;
  if (N === 0) return null;
  const M = 2 * N;
  // base kernel matrix k(x_i, x_j)
  const Kb = [];
  for (let i = 0; i < N; i++) {
    const row = new Float64Array(N);
    for (let j = 0; j < N; j++) row[j] = kfun(X[i], X[j], hp);
    Kb.push(row);
  }
  const sn2 = hp.sigma_n * hp.sigma_n;
  const Kj = [];
  for (let a = 0; a < M; a++) Kj.push(new Float64Array(M));
  for (let p = 0; p < 2; p++) {
    for (let q = 0; q < 2; q++) {
      const bpq = B[p][q];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          Kj[p * N + i][q * N + j] = bpq * Kb[i][j];
        }
      }
    }
  }
  // per-channel observation noise on the diagonal
  for (let a = 0; a < M; a++) Kj[a][a] += sn2;
  const yj = new Float64Array(M);
  for (let i = 0; i < N; i++) { yj[i] = Yd[i]; yj[N + i] = Yc[i]; }
  const L = choleskyJitter(Kj);
  const alpha = choleskySolve(L, yj);
  return { L, alpha, Kb, B, kfun, hp, X, N, yj };
}

// Predict bivariate posterior at each test point in Xstar.
// Returns per-channel mean arrays muD, muC and, at each x, the full 2×2
// posterior covariance Sigma (Sigma11=first-channel var, Sigma22=second-channel
// var, Sigma12=cross-cov) plus the cross-channel correlation rho(x).
export function predictLMC(model, Xstar) {
  const Mg = Xstar.length;
  const muD = new Float64Array(Mg);
  const muC = new Float64Array(Mg);
  const s11 = new Float64Array(Mg);
  const s22 = new Float64Array(Mg);
  const s12 = new Float64Array(Mg);
  const rho = new Float64Array(Mg);
  if (!model) {
    for (let i = 0; i < Mg; i++) { s11[i] = 1; s22[i] = 1; }
    return { muD, muC, s11, s22, s12, rho };
  }
  const { L, alpha, kfun, hp, X, N, B } = model;
  const Mj = 2 * N;
  for (let g = 0; g < Mg; g++) {
    const xs = Xstar[g];
    // base cross-covariance vector kb*(x) over training x
    const kb = new Float64Array(N);
    for (let j = 0; j < N; j++) kb[j] = kfun(xs, X[j], hp);
    const kss = kfun(xs, xs, hp);
    // For each output channel p in {0,1}: kstar_p is a 2N vector:
    //   kstar_p[(q,j)] = B[p][q] * kb[j]
    // posterior mean mu_p = kstar_p^T alpha
    // posterior cov  Sigma_pq = B[p][q]*kss - kstar_p^T K^{-1} kstar_q
    const kstar = [new Float64Array(Mj), new Float64Array(Mj)];
    for (let p = 0; p < 2; p++) {
      for (let q = 0; q < 2; q++) {
        for (let j = 0; j < N; j++) kstar[p][q * N + j] = B[p][q] * kb[j];
      }
    }
    let mD = 0, mC = 0;
    for (let a = 0; a < Mj; a++) { mD += kstar[0][a] * alpha[a]; mC += kstar[1][a] * alpha[a]; }
    muD[g] = mD; muC[g] = mC;
    const v0 = forwardSolve(L, kstar[0]);
    const v1 = forwardSolve(L, kstar[1]);
    let dot00 = 0, dot11 = 0, dot01 = 0;
    for (let a = 0; a < Mj; a++) { dot00 += v0[a] * v0[a]; dot11 += v1[a] * v1[a]; dot01 += v0[a] * v1[a]; }
    let c11 = B[0][0] * kss - dot00;
    let c22 = B[1][1] * kss - dot11;
    let c12 = B[0][1] * kss - dot01;
    if (c11 < 0) c11 = 0;
    if (c22 < 0) c22 = 0;
    s11[g] = c11; s22[g] = c22; s12[g] = c12;
    const denom = Math.sqrt(c11 * c22);
    rho[g] = denom > 1e-9 ? Math.max(-1, Math.min(1, c12 / denom)) : 0;
  }
  return { muD, muC, s11, s22, s12, rho };
}

// ---------- Posterior trajectory sampling ----------
//
// Draws n_samples joint function paths from the GP posterior over x_grid.
// Builds the M×M posterior covariance on the grid, factorizes it once with
// Cholesky, and returns n_samples paths via mu + L_post · z (z ~ N(0,I)).
// rng is an optional () => U(0,1) for reproducibility; defaults to Math.random.
export function sampleTrajectories(model, xGrid, nSamples, rng) {
  const M = xGrid.length;
  const draw = rng || Math.random;
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = draw();
    while (v === 0) v = draw();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // posterior mean + covariance on the grid
  const { mu, cov } = posteriorJoint(model, xGrid);
  const Lp = choleskyJitter(cov);
  const paths = [];
  for (let s = 0; s < nSamples; s++) {
    const z = new Float64Array(M);
    for (let i = 0; i < M; i++) z[i] = gauss();
    const path = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      let acc = mu[i];
      for (let k = 0; k <= i; k++) acc += Lp[i][k] * z[k];
      path[i] = acc;
    }
    paths.push(path);
  }
  return paths;
}

// Joint posterior mean vector and full MxM covariance on xGrid (latent f,
// epistemic only — noise excluded; add sigma_n^2 if predictive draws of y
// are wanted). Returns { mu, cov }.
export function posteriorJoint(model, xGrid) {
  const M = xGrid.length;
  const mu = new Float64Array(M);
  const cov = [];
  for (let i = 0; i < M; i++) cov.push(new Float64Array(M));
  if (!model) {
    for (let i = 0; i < M; i++) cov[i][i] = 1;
    return { mu, cov };
  }
  const { L, alpha, kfun, hp, X } = model;
  const N = X.length;
  // Kstar: M x N cross-covariance; V = L^{-1} Kstar^T (N x M)
  const Kstar = [];
  for (let i = 0; i < M; i++) {
    const row = new Float64Array(N);
    for (let j = 0; j < N; j++) row[j] = kfun(xGrid[i], X[j], hp);
    Kstar.push(row);
    let m = 0;
    for (let j = 0; j < N; j++) m += row[j] * alpha[j];
    mu[i] = m;
  }
  const V = [];
  for (let i = 0; i < M; i++) V.push(forwardSolve(L, Kstar[i]));
  for (let i = 0; i < M; i++) {
    for (let j = i; j < M; j++) {
      let kij = kfun(xGrid[i], xGrid[j], hp);
      let dot = 0;
      for (let k = 0; k < N; k++) dot += V[i][k] * V[j][k];
      let c = kij - dot;
      cov[i][j] = c;
      cov[j][i] = c;
    }
  }
  return { mu, cov };
}

// Sample n_samples paths of the FIRST channel only from the LMC posterior
// (the channel whose sign/threshold events downstream estimators integrate).
// We build the first-channel posterior covariance on xGrid by the same
// joint-kernel machinery restricted to output p=0.
export function sampleTrajectoriesLMC(model, xGrid, nSamples, rng) {
  const Mg = xGrid.length;
  const draw = rng || Math.random;
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = draw();
    while (v === 0) v = draw();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  if (!model) {
    const paths = [];
    for (let s = 0; s < nSamples; s++) {
      const p = new Float64Array(Mg);
      for (let i = 0; i < Mg; i++) p[i] = gauss();
      paths.push(p);
    }
    return paths;
  }
  const { L, alpha, kfun, hp, X, N, B } = model;
  const Mj = 2 * N;
  // first-channel mean + cross-cov vectors
  const mu = new Float64Array(Mg);
  const kstarD = []; // Mg vectors of length Mj for output p=0
  for (let g = 0; g < Mg; g++) {
    const xs = xGrid[g];
    const kb = new Float64Array(N);
    for (let j = 0; j < N; j++) kb[j] = kfun(xs, X[j], hp);
    const ks = new Float64Array(Mj);
    for (let q = 0; q < 2; q++) for (let j = 0; j < N; j++) ks[q * N + j] = B[0][q] * kb[j];
    kstarD.push(ks);
    let m = 0;
    for (let a = 0; a < Mj; a++) m += ks[a] * alpha[a];
    mu[g] = m;
  }
  // first-channel posterior covariance on grid
  const Vd = [];
  for (let g = 0; g < Mg; g++) Vd.push(forwardSolve(L, kstarD[g]));
  const cov = [];
  for (let i = 0; i < Mg; i++) cov.push(new Float64Array(Mg));
  for (let i = 0; i < Mg; i++) {
    for (let j = i; j < Mg; j++) {
      const kij = B[0][0] * kfun(xGrid[i], xGrid[j], hp);
      let dot = 0;
      for (let a = 0; a < Mj; a++) dot += Vd[i][a] * Vd[j][a];
      let c = kij - dot;
      cov[i][j] = c; cov[j][i] = c;
    }
  }
  const Lp = choleskyJitter(cov);
  const paths = [];
  for (let s = 0; s < nSamples; s++) {
    const z = new Float64Array(Mg);
    for (let i = 0; i < Mg; i++) z[i] = gauss();
    const path = new Float64Array(Mg);
    for (let i = 0; i < Mg; i++) {
      let acc = mu[i];
      for (let k = 0; k <= i; k++) acc += Lp[i][k] * z[k];
      path[i] = acc;
    }
    paths.push(path);
  }
  return paths;
}

// ---------- MCMC over GP hyperparameters (spec component 5's realization) ----------
//
// Metropolis-Hastings walker over (ell, eta, sigma_n) in log-space, target =
// marginal log-likelihood + log-prior. Priors: independent log-normal on each
// hyperparameter with mean log(default) and sigma = 1 (weakly informative).
// Pure function: caller supplies X, y, kfun, the default hp (for prior means),
// proposal scale, chain length, and an rng. Returns the retained samples as
// { ell:[], eta:[], sigma_n:[] }.
//
// Engine-level only in this build: the on-screen I2 panel deliberately shows
// the closed-form importance-weighted ensemble (diagnostics.js) instead of
// chain diagnostics; mcmcChain/gelmanRubin are kept canonical + tested for
// consumers that want full posterior sampling over hyperparameters.
function logNormalLogPrior(value, meanLog, sd) {
  if (value <= 0) return -Infinity;
  const lv = Math.log(value);
  const z = (lv - meanLog) / sd;
  return -0.5 * z * z - Math.log(value) - Math.log(sd) - 0.5 * Math.log(2 * Math.PI);
}

export function mcmcChain(X, y, kfun, hpDefault, opts = {}) {
  const burn = opts.burn ?? 500;
  const keep = opts.keep ?? 1500;
  const step = opts.step ?? 0.08;       // Gaussian proposal sd in log-space
  const rng = opts.rng || Math.random;
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const meanLog = {
    ell: Math.log(Math.max(1e-3, hpDefault.ell)),
    eta: Math.log(Math.max(1e-3, hpDefault.eta)),
    sigma_n: Math.log(Math.max(1e-3, hpDefault.sigma_n))
  };
  function target(hp) {
    let le;
    try {
      const m = fitGP(X, y, kfun, hp);
      le = m ? m.logEvidence : -Infinity;
    } catch (e) { le = -Infinity; }
    if (!Number.isFinite(le)) return -Infinity;
    const lp = logNormalLogPrior(hp.ell, meanLog.ell, 1)
             + logNormalLogPrior(hp.eta, meanLog.eta, 1)
             + logNormalLogPrior(hp.sigma_n, meanLog.sigma_n, 1);
    return le + lp;
  }
  // initialize at a jittered draw around the prior mean
  let cur = {
    ell: Math.exp(meanLog.ell + 0.5 * gauss()),
    eta: Math.exp(meanLog.eta + 0.5 * gauss()),
    sigma_n: Math.exp(meanLog.sigma_n + 0.5 * gauss()),
    period: hpDefault.period, alpha_lin: hpDefault.alpha_lin
  };
  let curLP = target(cur);
  const out = { ell: [], eta: [], sigma_n: [] };
  const total = burn + keep;
  for (let it = 0; it < total; it++) {
    const prop = {
      ell: cur.ell * Math.exp(step * gauss()),
      eta: cur.eta * Math.exp(step * gauss()),
      sigma_n: cur.sigma_n * Math.exp(step * gauss()),
      period: cur.period, alpha_lin: cur.alpha_lin
    };
    const propLP = target(prop);
    // log-space proposal is symmetric in log; include the log-Jacobian of the
    // multiplicative move (proportional to ratio of params) for detailed balance.
    const logJac = (Math.log(prop.ell) - Math.log(cur.ell))
                 + (Math.log(prop.eta) - Math.log(cur.eta))
                 + (Math.log(prop.sigma_n) - Math.log(cur.sigma_n));
    if (Math.log(rng()) < (propLP - curLP) + logJac) { cur = prop; curLP = propLP; }
    if (it >= burn) { out.ell.push(cur.ell); out.eta.push(cur.eta); out.sigma_n.push(cur.sigma_n); }
  }
  return out;
}

// Gelman-Rubin r̂ for a single scalar parameter across chains (array of arrays).
// W = mean within-chain variance, B = between-chain variance × n,
// V_hat = ((n-1)/n) W + B/n, r̂ = sqrt(V_hat / W).
export function gelmanRubin(chains) {
  const m = chains.length;
  if (m < 2) return 1;
  const n = chains[0].length;
  if (n < 2) return 1;
  const means = chains.map(c => c.reduce((a, b) => a + b, 0) / n);
  const grand = means.reduce((a, b) => a + b, 0) / m;
  // within-chain variances
  const wvars = chains.map((c, k) => {
    const mk = means[k];
    let s = 0;
    for (let i = 0; i < n; i++) { const d = c[i] - mk; s += d * d; }
    return s / (n - 1);
  });
  const W = wvars.reduce((a, b) => a + b, 0) / m;
  let bsum = 0;
  for (let k = 0; k < m; k++) { const d = means[k] - grand; bsum += d * d; }
  const B = (n / (m - 1)) * bsum;
  if (W <= 1e-12) return 1;
  const Vhat = ((n - 1) / n) * W + B / n;
  return Math.sqrt(Vhat / W);
}
