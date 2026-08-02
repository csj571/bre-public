// test.mjs — engine sanity checks for the BRE-1 math modules.
// Dev-only; the shipped app needs no build and no Node. Run with:
//   node bre1-simulator/test.mjs     (from the repo root)
//   node test.mjs                    (from inside bre1-simulator/)
//
// These are golden-value + property tests that protect the engine against
// regressions when refining the math. They do not touch the DOM.

import {
  fitGP, predictGP, logEvidence, Kernels, cholesky, choleskyJitter,
  lmcMatrixB, fitLMC, predictLMC, sampleTrajectoriesLMC, LMC_INIT,
  mcmcChain, gelmanRubin
} from './gp.js';
import { decompose, gaussianEntropy } from './entropy.js';
import { BOCPD } from './signal.js';
import { maxValueEntropySearch, expectedImprovement, argmax } from './acquisition.js';
import { hyperEnsemble, optimizeHyperparameters } from './diagnostics.js';
import {
  makeCouplingMatrix, createCoupledSystem, createCoherenceTracker,
  correlationMatrix, meanOffDiagonal, algebraicConnectivity, dominantRegime
} from './coupling.js';
import { PriorRegistry } from './registry.js';

let passed = 0, failed = 0;
function ok(name, cond, detail = '') {
  if (cond) { passed++; console.log(`  ✓ ${name}`); }
  else { failed++; console.log(`  ✗ ${name}${detail ? '  — ' + detail : ''}`); }
}
function approx(a, b, tol) { return Math.abs(a - b) <= tol; }

// Deterministic Gaussian stream for reproducible tests.
function rng(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

console.log('gp.js');
{
  const X = [0, 0.5, 1, 1.5, 2, 2.5, 3];
  const y = X.map(Math.sin);
  const hp = { ell: 0.8, eta: 1.0, sigma_n: 0.05, period: 4 };
  const model = fitGP(X, y, Kernels.rbf, hp);
  ok('fitGP returns a model with finite log-evidence', model && Number.isFinite(model.logEvidence), String(model && model.logEvidence));
  const { mu, sigmaF } = predictGP(model, [1.0, 10.0]);
  ok('posterior mean recovers a training point', approx(mu[0], Math.sin(1.0), 0.12), `mu=${mu[0].toFixed(3)} vs ${Math.sin(1).toFixed(3)}`);
  ok('epistemic σ grows away from the data', sigmaF[1] > sigmaF[0], `near=${sigmaF[0].toFixed(3)} far=${sigmaF[1].toFixed(3)}`);
  ok('logEvidence is finite', Number.isFinite(logEvidence(X, y, Kernels.rbf, hp)));
}

console.log('gp.js — jitter + prior std (P0.3 reconciliation)');
{
  // cholesky signals non-PD instead of fabricating a pivot.
  ok('cholesky returns null on a non-PD matrix', cholesky([[1, 2], [2, 1]]) === null);
  // choleskyJitter escalates until PD and the factorization is real: L·Lᵀ ≈ A + jitter·I.
  const A = [[1, 1], [1, 1]];                   // rank-1, singular
  const L = choleskyJitter(A);
  let finite = true;
  for (const row of L) for (const v of row) if (!Number.isFinite(v)) finite = false;
  ok('choleskyJitter factorizes an ill-conditioned matrix with finite entries', finite);
  const recon = L[1][0] * L[0][0];              // (L·Lᵀ)[1][0]
  ok('jittered factorization reproduces the off-diagonal', approx(recon, 1, 1e-9), `got ${recon}`);
  // Prior std with no data is eta, not hardcoded 1.0.
  const prior = predictGP(null, [0, 1]);
  ok('no-model prior std defaults to 1.0 when eta unknown', prior.sigmaF[0] === 1.0);
  const priorEta = predictGP({ hp: { eta: 0.4 } }, [0, 1]);
  ok('prior std = eta when hp is known', approx(priorEta.sigmaF[0], 0.4, 1e-12), `got ${priorEta.sigmaF[0]}`);
}

console.log('gp.js — LMC (multi-output GP)');
{
  // Golden value: B = W·Wᵀ + diag(κ) for the fixed init.
  const B = lmcMatrixB([1.0, 0.7], [0.10, 0.10]);
  ok('lmcMatrixB golden value', approx(B[0][0], 1.10, 1e-12) && approx(B[0][1], 0.70, 1e-12)
     && approx(B[1][0], 0.70, 1e-12) && approx(B[1][1], 0.59, 1e-12),
     JSON.stringify(B));
  ok('B is symmetric', B[0][1] === B[1][0]);

  const X = [0, 0.5, 1, 1.5, 2, 2.5, 3];
  const Yd = X.map(x => Math.sin(x));
  const Yc = X.map(x => 0.7 * Math.sin(x) + 0.05);   // correlated second channel
  const hp = { ell: 0.8, eta: 1.0, sigma_n: 0.05, alpha_lin: LMC_INIT.alpha_lin };
  const model = fitLMC(X, Yd, Yc, Kernels.matern52_linear, hp, B);
  ok('fitLMC returns a model', !!model && model.N === X.length);
  const p = predictLMC(model, [1.0, 2.0, 8.0]);
  ok('LMC first-channel mean tracks its data', approx(p.muD[0], Math.sin(1.0), 0.15), `mu=${p.muD[0].toFixed(3)}`);
  ok('LMC second-channel mean tracks its data', approx(p.muC[0], 0.7 * Math.sin(1.0) + 0.05, 0.15), `mu=${p.muC[0].toFixed(3)}`);
  let varsOk = true, rhoOk = true;
  for (let i = 0; i < 3; i++) {
    if (p.s11[i] < 0 || p.s22[i] < 0) varsOk = false;
    if (p.rho[i] < -1 - 1e-9 || p.rho[i] > 1 + 1e-9) rhoOk = false;
  }
  ok('LMC per-channel variances are non-negative', varsOk);
  ok('LMC cross-channel ρ ∈ [-1, 1]', rhoOk);
  ok('correlated channels give positive ρ near data', p.rho[0] > 0, `rho=${p.rho[0].toFixed(3)}`);
  ok('variance grows away from the data', p.s11[2] > p.s11[0], `near=${p.s11[0].toFixed(3)} far=${p.s11[2].toFixed(3)}`);
  const paths = sampleTrajectoriesLMC(model, [0, 1, 2, 3], 5, rng(42));
  ok('LMC trajectory sampler returns the requested paths', paths.length === 5 && paths[0].length === 4
     && paths.every(pt => Array.from(pt).every(Number.isFinite)));
}

console.log('gp.js — MCMC over hyperparameters');
{
  const r = rng(2718);
  const X = [], y = [];
  for (let i = 0; i < 12; i++) { const x = i * 0.3; X.push(x); y.push(Math.sin(x) + 0.05 * gauss(r)); }
  const hp = { ell: 0.8, eta: 1.0, sigma_n: 0.1 };
  const chain = mcmcChain(X, y, Kernels.rbf, hp, { burn: 50, keep: 120, step: 0.15, rng: rng(7) });
  ok('mcmcChain keeps the requested number of samples',
     chain.ell.length === 120 && chain.eta.length === 120 && chain.sigma_n.length === 120);
  ok('all retained samples are positive/finite',
     ['ell', 'eta', 'sigma_n'].every(k => chain[k].every(v => Number.isFinite(v) && v > 0)));
  ok('the chain moves (not stuck at the init)', new Set(chain.ell).size > 1);
  // Gelman-Rubin: identical chains ⇒ r̂ ≈ 1; well-separated chains ⇒ r̂ > 1.1.
  const c = chain.ell;
  // identical chains: B = 0 so r̂ = sqrt((n-1)/n), just under 1
  ok('r̂ ≈ 1 for identical chains', approx(gelmanRubin([c, c.slice()]), 1, 0.01),
     `rhat=${gelmanRubin([c, c.slice()]).toFixed(4)}`);
  const shifted = c.map(v => v + 100);
  ok('r̂ > 1.1 for divergent chains', gelmanRubin([c, shifted]) > 1.1,
     `rhat=${gelmanRubin([c, shifted]).toFixed(2)}`);
  ok('r̂ = 1 for a single chain (undefined case)', gelmanRubin([c]) === 1);
}

console.log('entropy.js');
{
  const sigmaF = Float64Array.from([0.1, 0.5, 1.0]);
  const sigmaN = Float64Array.from([0.2, 0.2, 0.2]);
  const { Htotal, Halea, Hepi } = decompose(sigmaF, sigmaN);
  let identity = true, nonneg = true;
  for (let i = 0; i < 3; i++) {
    if (!approx(Hepi[i], Htotal[i] - Halea[i], 1e-9)) identity = false;
    if (Hepi[i] < -1e-9) nonneg = false;
  }
  ok('H_epi = H_total − H_alea (decomposition identity)', identity);
  ok('H_epi ≥ 0', nonneg);
  ok('gaussianEntropy matches ½log(2πe σ²)', approx(gaussianEntropy(1), 0.5 * Math.log(2 * Math.PI * Math.E), 1e-12));
}

console.log('signal.js — BOCPD');
{
  const bocpd = new BOCPD({ lambda: 30, mu0: 0, kappa0: 5, alpha0: 2, beta0: 0.3 });
  const r = rng(7);
  let firedAfterStep = false;
  for (let i = 0; i < 80; i++) {
    const mean = i < 40 ? 0 : 6;            // a clear level shift at i = 40
    const res = bocpd.update(mean + 0.3 * gauss(r));
    if (i >= 40 && i <= 60 && res.changeFired) firedAfterStep = true;
  }
  ok('BOCPD fires on a clear level shift', firedAfterStep);
}

console.log('acquisition.js — MES');
{
  const N = 11;
  const mu = new Float64Array(N);          // flat mean
  const sigmaF = new Float64Array(N).fill(0.1);
  sigmaF[5] = 1.0;                          // one clearly-uncertain region
  const a = maxValueEntropySearch(mu, sigmaF, { samples: 12 });
  let nonneg = true;
  for (let i = 0; i < N; i++) if (a[i] < -1e-9) nonneg = false;
  ok('MES is non-negative', nonneg);
  ok('MES peaks at the uncertain region', argmax(a).index === 5, `argmax=${argmax(a).index}`);
  // EI sanity: improvement only above the incumbent.
  const ei = expectedImprovement(Float64Array.from([0, 1, 2]), Float64Array.from([0.5, 0.5, 0.5]), 1);
  ok('EI is largest where mean is highest', argmax(ei).index === 2);
}

console.log('diagnostics.js — ensemble + optimisation');
{
  const r = rng(11);
  const X = [], y = [];
  for (let i = 0; i < 20; i++) { const x = i * 0.2; X.push(x); y.push(Math.sin(x) + 0.05 * gauss(r)); }
  const d = hyperEnsemble(X, y, 'rbf', { count: 16, seed: 1234 });
  ok('ESS within [1, K]', d.ess >= 1 - 1e-9 && d.ess <= 16 + 1e-9, `ess=${d.ess.toFixed(2)}`);
  ok('ensemble entropy within [0, 1]', d.entropyNorm >= -1e-9 && d.entropyNorm <= 1 + 1e-9, `H=${d.entropyNorm.toFixed(3)}`);
  ok('ensemble is reproducible for a fixed seed', hyperEnsemble(X, y, 'rbf', { count: 16, seed: 1234 }).ess === d.ess);
  const hp0 = { ell: 5.0, eta: 0.2, sigma_n: 2.0, period: 4 };
  const le0 = logEvidence(X, y, Kernels.rbf, hp0);
  const opt = optimizeHyperparameters(X, y, 'rbf', hp0, { passes: 5 });
  ok('optimizer does not decrease the log-evidence', opt.logEvidence >= le0 - 1e-9, `${le0.toFixed(2)} → ${opt.logEvidence.toFixed(2)}`);
}

console.log('coupling.js — dynamics + coherence');
{
  // Determinism: same seed ⇒ identical matrices and trajectories.
  ok('coupling matrix is reproducible', JSON.stringify(makeCouplingMatrix(7)) === JSON.stringify(makeCouplingMatrix(7)));
  const sysA = createCoupledSystem(123), sysB = createCoupledSystem(123);
  let identical = true, finite = true;
  for (let i = 0; i < 60; i++) {
    const a = sysA.step(), b = sysB.step();
    for (let k = 0; k < a.levels.length; k++) {
      if (a.levels[k] !== b.levels[k]) identical = false;
      if (!Number.isFinite(a.levels[k])) finite = false;
    }
  }
  ok('coupled system is deterministic for a fixed seed', identical);
  ok('coupled system stays finite/bounded over time', finite);
  ok('a shock fires within a reasonable horizon', (() => {
    const s = createCoupledSystem(5); let fired = false;
    for (let i = 0; i < 120; i++) if (s.step().shockFired) fired = true;
    return fired;
  })());

  // correlationMatrix: identical columns ⇒ corr ≈ 1; independent ⇒ small.
  const r2 = rng(3);
  const same = [], indep = [];
  for (let w = 0; w < 40; w++) { const v = gauss(r2); same.push([v, v]); indep.push([gauss(r2), gauss(r2)]); }
  ok('correlation of identical series ≈ 1', approx(correlationMatrix(same)[0][1], 1, 1e-6));
  ok('correlation of independent series is small', Math.abs(correlationMatrix(indep)[0][1]) < 0.4);
  ok('correlation diagonal is 1', correlationMatrix(indep)[0][0] === 1);

  // meanOffDiagonal of a known matrix.
  ok('meanOffDiagonal excludes the diagonal', approx(meanOffDiagonal([[1, 0.5], [0.5, 1]]), 0.5, 1e-9));

  // Fiedler: disconnected ⇒ ~0; fully connected ⇒ > 0; always finite.
  const disconnected = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const connected = [[1, 0.9, 0.8], [0.9, 1, 0.85], [0.8, 0.85, 1]];
  ok('Fiedler ≈ 0 for a disconnected graph', algebraicConnectivity(disconnected, 0.5) < 1e-6);
  ok('Fiedler > 0 for a connected graph', algebraicConnectivity(connected, 0.5) > 0.1);
  ok('Fiedler is finite', Number.isFinite(algebraicConnectivity(connected, 0.5)));

  // Coherence tracker: bounds, readiness, and the synchronized vs independent extremes.
  const tA = createCoherenceTracker({ L: 4, minSamples: 8 });
  ok('tracker not ready before minSamples', tA.push([0, 0, 0, 0]).ready === false);
  const tSync = createCoherenceTracker({ L: 3, minSamples: 8 });
  const r3 = rng(9); let last = null;
  for (let w = 0; w < 30; w++) { const v = gauss(r3); last = tSync.push([v, v + 1e-6, v - 1e-6]); }
  ok('index ∈ [0,1]', last.index >= -1e-9 && last.index <= 1 + 1e-9, `index=${last.index.toFixed(3)}`);
  ok('synchronized layers ⇒ index near 1', last.index > 0.9, `index=${last.index.toFixed(3)}`);
  const tInd = createCoherenceTracker({ L: 3, minSamples: 8 });
  const r4 = rng(21); let li = null;
  for (let w = 0; w < 40; w++) li = tInd.push([gauss(r4), gauss(r4), gauss(r4)]);
  ok('independent layers ⇒ index near 0.5', Math.abs(li.index - 0.5) < 0.25, `index=${li.index.toFixed(3)}`);
  tInd.setMetric('fiedler');
  const rf = tInd.push([gauss(r4), gauss(r4), gauss(r4)]);
  ok('metric toggle drives index from Fiedler', rf.index === rf.fiedlerIndex && tInd.getMetric() === 'fiedler');

  // End-to-end behaviour: a coupled system should be coherent in calm and lose
  // coherence around a shock cascade (the continuity index dips).
  {
    const sys = createCoupledSystem(2024, { L: 8 });
    const tracker = createCoherenceTracker({ L: 8, window: 40, corrThreshold: 0.45 });
    let calmSum = 0, calmN = 0, minAroundShock = 1, sawShock = false;
    for (let i = 0; i < 400; i++) {
      const s = sys.step();
      const c = tracker.push(s.levels);
      if (!c.ready) continue;
      if (s.shockFired) sawShock = true;
      if (sawShock && i < 400) minAroundShock = Math.min(minAroundShock, c.index);
      if (!s.shockFired && s.regimes.every(r => r === 'calm')) { calmSum += c.index; calmN++; }
    }
    const calmMean = calmN ? calmSum / calmN : 0;
    ok('coherence is high during calm periods', calmMean > 0.7, `calmMean=${calmMean.toFixed(3)}`);
    ok('coherence dips below calm after a cascade', minAroundShock < calmMean, `min=${minAroundShock.toFixed(3)} calm=${calmMean.toFixed(3)}`);
  }

  // dominantRegime collapse.
  ok('dominantRegime flags fragmentation', dominantRegime(['calm', 'fragmenting', 'calm']) === 'fragmenting');
  ok('dominantRegime flags stress on ≥2', dominantRegime(['stress', 'stress', 'calm']) === 'stress');
  ok('dominantRegime calm otherwise', dominantRegime(['calm', 'stress', 'calm']) === 'calm');
}

console.log('registry.js — semantic seeding');
{
  const reg = new PriorRegistry();
  const base = { kernel: 'rbf', ell: 1, eta: 1 };
  const s = reg.seed(base, { name: 'Seed prior', provenance: { source: 'human seed', regime: null, tick: 0 } });
  ok('seed starts active (dormancy 0)', s.dormancy === 0 && s.retrievals === 0);
  ok('seed is not a germination', s.germinated === false && s.name === 'Seed prior');

  const v2 = reg.promote({ ...base, ell: 2 }, 'Germination event', 'BRE-1',
    { name: 'Calm-regime prior', provenance: { source: 'D4 gate', regime: 'calm', tick: 12 }, germinated: true });
  ok('promote is a germination with a name', v2.germinated === true && v2.name === 'Calm-regime prior');
  ok('promote records a diff', Array.isArray(v2.diff) && v2.diff.length > 0);

  reg.tickDormancy(5);
  ok('tickDormancy ages dormant seeds, not the current one',
     reg.versions[0].dormancy === 5 && reg.current().dormancy === 0);

  // a dormant seed encoding the current regime is the germination candidate
  const v3 = reg.promote({ ...base, ell: 3 }, 'g', 'BRE-1',
    { provenance: { regime: 'stress', tick: 20 }, germinated: true });
  ok('matchRegime finds the dormant seed for a recurring regime', reg.matchRegime('calm')?.id === 'v2');
  ok('matchRegime skips the current prior', reg.matchRegime('stress') === null);
  ok('matchRegime returns null for unseen regimes', reg.matchRegime('fragmenting') === null);

  const before = reg.versions[1].retrievals;
  reg.retrieve('v2', 99);
  ok('retrieve resets dormancy and counts the retrieval',
     reg.versions[1].dormancy === 0 && reg.versions[1].retrievals === before + 1 && reg.versions[1].lastRetrievedTick === 99);

  // backward-compatibility: original keys still present
  ok('versions keep the original keys', ['id', 'timestamp', 'author', 'justification', 'snapshot', 'diff'].every(k => k in v2));
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
