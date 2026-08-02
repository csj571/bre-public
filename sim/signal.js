// signal.js — Kalman 1-D filter + BOCPD change-point detector.
//
// Two online estimators that condition the raw stream before the GP sees it:
//   • AdaptiveKalman — tracks the running level of the signal and adapts its own
//     noise estimate, giving a cheap residual-variance readout (node S2).
//   • BOCPD — Bayesian Online Change-point Detection: maintains a distribution
//     over "how long since the last regime change" and fires when it collapses.
//
// ── Kalman symbols ─────────────────────────────────────────────────────────────
//   x         state estimate (the filtered level)
//   P         state-error variance
//   q         process-noise variance — how much the level may drift per step
//   r         observation-noise variance — adapted online from residuals
//   residVar  EWMA of the squared innovation (residual variance)
//   alpha     EWMA rate for residVar (0.1 ⇒ 10% new, 90% history)
//
// ── BOCPD symbols ──────────────────────────────────────────────────────────────
//   lambda    expected run length; hazard H = 1/lambda is the per-step prior
//             probability of a change.
//   mu0,kappa0,alpha0,beta0   Normal-Inverse-Gamma conjugate prior on each run's
//             (mean, variance): mu0 prior mean, kappa0 its pseudo-count, and
//             (alpha0, beta0) the inverse-gamma shape/scale on the variance.
//   R[r]      run-length posterior: P(run length = r | data so far).
//   p0        R[0] — posterior mass on "a change just happened".
//   _refractory  cooldown (ticks) after a detection, to suppress jitter.
// ───────────────────────────────────────────────────────────────────────────────

// ---------- Adaptive 1-D Kalman ----------
// State: scalar mean. Process noise q adapts to residual variance.
export class AdaptiveKalman {
  constructor() {
    this.x = 0;
    this.P = 1.0;
    this.q = 0.01;       // process noise
    this.r = 0.5;        // observation noise (will adapt)
    this.residVar = 0.5; // EWMA residual variance
    this.alpha = 0.1;    // EWMA rate
    this.initialized = false;
  }
  update(z) {
    if (!this.initialized) {
      this.x = z;
      this.initialized = true;
      return { filtered: this.x, residual: 0 };
    }
    // Predict
    const xpred = this.x;
    const Ppred = this.P + this.q;
    // Innovation
    const y = z - xpred;
    const S = Ppred + this.r;
    const K = Ppred / S;
    this.x = xpred + K * y;
    this.P = (1 - K) * Ppred;
    // Adapt residual variance (EWMA)
    this.residVar = (1 - this.alpha) * this.residVar + this.alpha * y * y;
    // Adapt observation noise toward the residual variance. The 0.7/0.3 blend is
    // a deliberately gentle EWMA: r tracks changes in noise within a few steps
    // without chasing single-sample spikes (which would destabilise the gain K).
    this.r = 0.7 * this.r + 0.3 * this.residVar;
    return { filtered: this.x, residual: y };
  }
}

// ---------- BOCPD (Adams & MacKay) ----------
// Gaussian observation model with conjugate Normal-Inverse-Gamma prior
// Hazard H(r) = 1/lambda
export class BOCPD {
  constructor({ lambda = 50, mu0 = 0, kappa0 = 1, alpha0 = 1, beta0 = 1 } = {}) {
    this._refractory = 0;
    this.hazard = 1 / lambda;
    this.mu0 = mu0;
    this.kappa0 = kappa0;
    this.alpha0 = alpha0;
    this.beta0 = beta0;
    // Posterior parameters indexed by run length r
    this.mu = [mu0];
    this.kappa = [kappa0];
    this.alpha = [alpha0];
    this.beta = [beta0];
    // Run-length posterior
    this.R = [1.0];
    this.history = []; // list of R distributions over time (for inset plot)
    this.maxRun = 200;
  }

  // Student-t predictive PDF for next observation
  studentT(x, mu, kappa, alpha, beta) {
    const df = 2 * alpha;
    const scale2 = beta * (kappa + 1) / (alpha * kappa);
    const scale = Math.sqrt(scale2);
    const z = (x - mu) / scale;
    // log-pdf of Student-t
    const lpdf = lgamma((df + 1) / 2) - lgamma(df / 2)
      - 0.5 * Math.log(df * Math.PI) - Math.log(scale)
      - ((df + 1) / 2) * Math.log(1 + z * z / df);
    return Math.exp(lpdf);
  }

  update(x) {
    const T = this.R.length;
    // Predictive probabilities
    const pred = new Array(T);
    for (let r = 0; r < T; r++) {
      pred[r] = this.studentT(x, this.mu[r], this.kappa[r], this.alpha[r], this.beta[r]);
    }
    // Growth probabilities: R_new[r+1] = R[r] * pred[r] * (1 - H)
    // Change probabilities: R_new[0] = sum_r R[r] * pred[r] * H
    const newR = new Array(T + 1).fill(0);
    let changeP = 0;
    for (let r = 0; r < T; r++) {
      newR[r + 1] = this.R[r] * pred[r] * (1 - this.hazard);
      changeP += this.R[r] * pred[r] * this.hazard;
    }
    newR[0] = changeP;

    // Normalize
    let sum = 0;
    for (let r = 0; r < newR.length; r++) sum += newR[r];
    if (sum > 0) for (let r = 0; r < newR.length; r++) newR[r] /= sum;

    // Update sufficient statistics
    const newMu = [this.mu0];
    const newKappa = [this.kappa0];
    const newAlpha = [this.alpha0];
    const newBeta = [this.beta0];
    for (let r = 0; r < T; r++) {
      const k = this.kappa[r];
      const m = this.mu[r];
      const a = this.alpha[r];
      const b = this.beta[r];
      newKappa.push(k + 1);
      newMu.push((k * m + x) / (k + 1));
      newAlpha.push(a + 0.5);
      newBeta.push(b + (k * (x - m) * (x - m)) / (2 * (k + 1)));
    }
    this.mu = newMu;
    this.kappa = newKappa;
    this.alpha = newAlpha;
    this.beta = newBeta;
    this.R = newR;

    // Truncate to max run length
    if (this.R.length > this.maxRun) {
      this.R = this.R.slice(0, this.maxRun);
      this.mu = this.mu.slice(0, this.maxRun);
      this.kappa = this.kappa.slice(0, this.maxRun);
      this.alpha = this.alpha.slice(0, this.maxRun);
      this.beta = this.beta.slice(0, this.maxRun);
      // renormalize
      let s = 0;
      for (let i = 0; i < this.R.length; i++) s += this.R[i];
      if (s > 0) for (let i = 0; i < this.R.length; i++) this.R[i] /= s;
    }

    // Track history of mode run length for inset
    let modeR = 0, modeP = 0;
    for (let r = 0; r < this.R.length; r++) {
      if (this.R[r] > modeP) { modeP = this.R[r]; modeR = r; }
    }
    this.history.push({ modeR, p0: this.R[0] });
    if (this.history.length > 200) this.history.shift();

    // Detect change-points by run-length mode collapse: when modeR was high then
    // suddenly drops to near 0, the posterior is re-anchoring to a fresh segment.
    let changeFired = this.R[0] > 0.5;
    if (!changeFired && this.history.length > 6) {
      const recent = this.history.slice(-8);
      const peakRecent = Math.max(...recent.map(e => e.modeR));
      if (peakRecent >= 6 && modeR <= 2) changeFired = true;
    }
    if (this._refractory > 0) { this._refractory -= 1; changeFired = false; }
    if (changeFired) this._refractory = 8;
    return { p0: this.R[0], modeR, changeFired, R: this.R.slice() };
  }

  reset() {
    this.mu = [this.mu0];
    this.kappa = [this.kappa0];
    this.alpha = [this.alpha0];
    this.beta = [this.beta0];
    this.R = [1.0];
    this.history = [];
  }
}

// Lanczos approximation to lgamma
export function lgamma(x) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
