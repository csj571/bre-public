// replay.js — the showcase's computation layer. No DOM, no rendering.
//
// This module runs the SAME detector the validated Python result runs:
// `BOCPD` from ../sim/signal.js is a line-for-line counterpart of
// `engine/changepoint.py`, and `tests/test_js_python_parity.py` asserts the two
// produce byte-identical flag indices on every vendored series in this repo.
//
// The one thing that must be set explicitly is `maxRun`: the simulator ships a
// 200-step run-length truncation (it never sees a series this long), while the
// Python engine defaults to 300. Truncation changes the posterior tail, so the
// showcase pins it to the Python default — that is the whole of the difference.

import { BOCPD } from '../sim/signal.js';

export const PY_MAX_RUN = 300;

/** Parse a `date,value` CSV exactly the way validate_regimes.py's loader does. */
export function parseCsv(text) {
  const out = [];
  for (const line of text.trim().split('\n').slice(1)) {
    const parts = line.split(',');
    if (parts.length < 2) continue;
    const date = parts[0].trim();
    const raw = parts[1].trim();
    if (raw === '' || raw === '.') continue;   // FRED encodes missing as "."
    out.push({ date, value: parseFloat(raw) });
  }
  return out;
}

/** Population z-score over the whole series — the harness's preprocessing. */
export function zscore(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const varp = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  const sd = Math.sqrt(varp) || 1.0;
  return values.map(v => (v - mean) / sd);
}

/**
 * Stream a series through BOCPD, keeping the per-step posterior readouts the
 * showcase animates: p0 (mass on "a change just happened"), the run-length
 * mode, and the discrete flags.
 */
export function runBocpd(values, { lambda = 50, maxRun = PY_MAX_RUN } = {}) {
  const bocpd = new BOCPD({ lambda });
  bocpd.maxRun = maxRun;
  const p0 = [];
  const modeRun = [];
  const flags = [];
  values.forEach((x, i) => {
    const step = bocpd.update(x);
    p0.push(step.p0);
    modeRun.push(step.modeR);
    if (step.changeFired) flags.push(i);
  });
  return { p0, modeRun, flags };
}

/** Index of the row whose date is closest to `target` (harness rule). */
export function nearestIndex(dates, target) {
  const t = Date.parse(target + 'T00:00:00Z');
  let best = 0;
  let bestGap = Infinity;
  dates.forEach((d, i) => {
    const gap = Math.abs(Date.parse(d + 'T00:00:00Z') - t);
    if (gap < bestGap) { bestGap = gap; best = i; }
  });
  return best;
}

/**
 * Trading days from a pre-registered onset to the first flag in [onset, onset+tol].
 * Returns null when the break is missed. Never negative: this is detection, and
 * a flag before the onset would be counted as a separate (unmatched) flag.
 */
export function latency(flags, onsetIdx, tol = 10) {
  const hit = flags.find(f => f >= onsetIdx && f <= onsetIdx + tol);
  return hit === undefined ? null : hit - onsetIdx;
}

/** Load one CSV and score it against its pre-registered onsets. */
export async function loadCase(spec, fetchImpl = fetch) {
  const res = await fetchImpl(spec.path);
  if (!res.ok) throw new Error(`${spec.path}: ${res.status}`);
  const rows = parseCsv(await res.text());
  const dates = rows.map(r => r.date);
  const raw = rows.map(r => r.value);
  const series = zscore(raw);
  const { p0, modeRun, flags } = runBocpd(series);
  const breaks = spec.breaks
    .filter(b => dates.length && dates[0] <= b.date && b.date <= dates[dates.length - 1])
    .map(b => {
      const index = nearestIndex(dates, b.date);
      const lat = latency(flags, index);
      return { ...b, index, latency: lat, flagDate: lat === null ? null : dates[index + lat] };
    });
  return { ...spec, dates, raw, series, p0, modeRun, flags, breaks };
}
