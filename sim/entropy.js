// entropy.js — predictive entropy and its aleatoric / epistemic decomposition.
//
// For a Gaussian predictive y ~ N(μ, σ_f² + σ_n²), the differential entropy is
// H = ½·log(2πe·variance). We split the total uncertainty into two parts:
//
//   H_total      = ½·log(2πe(σ_f² + σ_n²))   all of it
//   H_aleatoric  = ½·log(2πe·σ_n²)           irreducible observation noise (the floor)
//   H_epistemic  = H_total − H_aleatoric      reducible model uncertainty = I(y; f)
//
// H_epistemic is exactly the BALD score (Houlsby et al. 2011): the mutual
// information between the next observation y and the latent function f. It shrinks
// as data accumulates; H_aleatoric does not.
//
// ── Symbol glossary ───────────────────────────────────────────────────────────
//   sigmaF[i]  epistemic std-dev of f from the GP (uncertainty about f)
//   sigmaN[i]  per-point aleatoric std-dev (heteroscedastic observation noise)
//   Htotal / Halea / Hepi  the three entropy curves above, per grid point
// ───────────────────────────────────────────────────────────────────────────────

// Gaussian differential entropy: 0.5 log(2 pi e sigma^2)
const LOG_2_PI_E = Math.log(2 * Math.PI * Math.E);

export function gaussianEntropy(sigma2) {
  if (sigma2 <= 0) sigma2 = 1e-12;
  return 0.5 * Math.log(2 * Math.PI * Math.E * sigma2);
}

// Decompose predictive entropy into aleatoric + epistemic
// sigmaF: epistemic stddev from GP
// sigmaN: known per-x aleatoric stddev (heteroscedastic noise function)
export function decompose(sigmaF, sigmaN) {
  const M = sigmaF.length;
  const Htotal = new Float64Array(M);
  const Halea = new Float64Array(M);
  const Hepi = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    const sf2 = sigmaF[i] * sigmaF[i];
    const sn2 = sigmaN[i] * sigmaN[i];
    Htotal[i] = gaussianEntropy(sf2 + sn2);
    Halea[i] = gaussianEntropy(sn2);
    // BALD = H[y] - E[H[y|f]] = total - aleatoric (= epistemic info)
    Hepi[i] = Htotal[i] - Halea[i];
  }
  return { Htotal, Halea, Hepi };
}
