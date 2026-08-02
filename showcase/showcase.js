// showcase.js — rendering and playback for the regime-detection showcase.
//
// All of the arithmetic lives in replay.js (and, under that, in the simulator's
// signal.js). This file only draws it: canvases, tabs, transport, readout.

import { CASES } from './cases.js';
import { loadCase } from './replay.js';

const $ = id => document.getElementById(id);

const el = {
  panel: $('panel'),
  error: $('error'),
  tabs: $('tabs'),
  resultsBody: $('results-body'),
  title: $('case-title'),
  sub: $('case-sub'),
  note: $('case-note'),
  xStart: $('x-start'),
  xEnd: $('x-end'),
  series: $('chart-series'),
  runlength: $('chart-runlength'),
  p0: $('chart-p0'),
  play: $('btn-play'),
  restart: $('btn-restart'),
  scrub: $('scrub'),
  speed: $('speed'),
  theme: $('btn-theme'),
  statDate: $('stat-date'),
  statObs: $('stat-obs'),
  statObsLabel: $('stat-obs-label'),
  statRun: $('stat-run'),
  statP0: $('stat-p0'),
  statStatus: $('stat-status'),
};

const state = {
  cases: [],
  active: null,
  cursor: 0,          // last revealed index
  playing: false,
  raf: null,
  lastFrame: 0,
};

// ---------------------------------------------------------------- utilities

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function fmtValue(kase, v) {
  const scaled = v * kase.scale;
  const digits = kase.scale === 100 ? 2 : 2;
  return `${scaled.toFixed(digits)}${kase.unitLabel}`;
}

function fmtLatency(days) {
  if (days === null) return 'missed';
  return days === 0 ? 'same day' : `${days} trading day${days === 1 ? '' : 's'}`;
}

/** Resize a canvas for the device pixel ratio; returns its 2D context in CSS px. */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = parseInt(canvas.getAttribute('height'), 10);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

// ------------------------------------------------------------------ drawing

// `top` leaves a strip above the plot area for each panel's caption, so the
// caption never collides with the topmost y-axis tick.
const PAD = { left: 46, right: 12, top: 22, bottom: 8 };

function xOf(i, n, w) {
  const span = w - PAD.left - PAD.right;
  return PAD.left + (n <= 1 ? 0 : (i / (n - 1)) * span);
}

function drawFrame(ctx, w, h, label) {
  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, h - PAD.bottom + 0.5);
  ctx.lineTo(w - PAD.right, h - PAD.bottom + 0.5);
  ctx.stroke();
  if (label) {
    ctx.fillStyle = css('--text-faint');
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, 0, 11);
  }
}

/** Vertical markers shared by all three panels: break onsets and fired flags. */
function drawMarkers(ctx, w, h, kase, upTo) {
  const n = kase.series.length;
  ctx.save();
  ctx.lineWidth = 1;
  for (const f of kase.flags) {
    if (f > upTo) continue;
    const x = xOf(f, n, w);
    ctx.strokeStyle = css('--observation');
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, PAD.top);
    ctx.lineTo(x + 0.5, h - PAD.bottom);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const b of kase.breaks) {
    if (b.index > upTo) continue;
    const x = xOf(b.index, n, w);
    ctx.strokeStyle = css('--break');
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x + 0.5, PAD.top);
    ctx.lineTo(x + 0.5, h - PAD.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawPlayhead(ctx, w, h, kase, upTo) {
  if (upTo >= kase.series.length - 1) return;
  const x = xOf(upTo, kase.series.length, w);
  ctx.strokeStyle = css('--text-faint');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + 0.5, PAD.top);
  ctx.lineTo(x + 0.5, h - PAD.bottom);
  ctx.stroke();
}

function drawSeries(kase, upTo) {
  const { ctx, w, h } = fitCanvas(el.series);
  const n = kase.raw.length;
  const lo = Math.min(...kase.raw);
  const hi = Math.max(...kase.raw);
  const pad = (hi - lo) * 0.08 || 1;
  // don't pad a non-negative series (VIX, realized vol) down past zero
  const y0 = lo >= 0 ? Math.max(0, lo - pad) : lo - pad;
  const y1 = hi + pad;
  const yOf = v => h - PAD.bottom - ((v - y0) / (y1 - y0)) * (h - PAD.top - PAD.bottom);

  drawFrame(ctx, w, h, kase.unit);

  // axis ticks — anchor the middle gridline on zero when the series straddles it
  // (log returns), otherwise on the midpoint (VIX, realized vol)
  const mid = lo < 0 && hi > 0 ? 0 : (y0 + y1) / 2;
  ctx.fillStyle = css('--text-faint');
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  for (const v of [y1, mid, y0]) {
    const y = yOf(v);
    ctx.fillText((v * kase.scale).toFixed(kase.scale === 100 ? 1 : 0), PAD.left - 6, y + 3);
    ctx.strokeStyle = css('--grid');
    ctx.beginPath();
    ctx.moveTo(PAD.left, y + 0.5);
    ctx.lineTo(w - PAD.right, y + 0.5);
    ctx.stroke();
  }

  drawMarkers(ctx, w, h, kase, upTo);

  ctx.strokeStyle = css('--posterior');
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  for (let i = 0; i <= upTo; i++) {
    const x = xOf(i, n, w);
    const y = yOf(kase.raw[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // label each detected break in place
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'left';
  for (const b of kase.breaks) {
    if (b.index > upTo) continue;
    ctx.fillStyle = css('--break');
    const x = Math.min(xOf(b.index, n, w) + 5, w - PAD.right - 118);
    ctx.fillText(b.date, x, PAD.top + 14);
    if (b.latency !== null && b.index + b.latency <= upTo) {
      ctx.fillStyle = css('--observation');
      ctx.fillText(`flagged +${b.latency}d`, x, PAD.top + 26);
    }
  }

  drawPlayhead(ctx, w, h, kase, upTo);
}

function drawRunLength(kase, upTo) {
  const { ctx, w, h } = fitCanvas(el.runlength);
  const n = kase.modeRun.length;
  const hi = Math.max(...kase.modeRun, 1);
  const yOf = v => h - PAD.bottom - (v / hi) * (h - PAD.top - PAD.bottom);

  drawFrame(ctx, w, h, 'run length since last change (posterior mode)');
  ctx.fillStyle = css('--text-faint');
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(String(hi), PAD.left - 6, yOf(hi) + 8);
  ctx.fillText('0', PAD.left - 6, h - PAD.bottom);

  drawMarkers(ctx, w, h, kase, upTo);

  ctx.beginPath();
  ctx.moveTo(xOf(0, n, w), h - PAD.bottom);
  for (let i = 0; i <= upTo; i++) ctx.lineTo(xOf(i, n, w), yOf(kase.modeRun[i]));
  ctx.lineTo(xOf(upTo, n, w), h - PAD.bottom);
  ctx.closePath();
  ctx.fillStyle = css('--query');
  ctx.globalAlpha = 0.22;
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = css('--query');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= upTo; i++) {
    const x = xOf(i, n, w);
    const y = yOf(kase.modeRun[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  drawPlayhead(ctx, w, h, kase, upTo);
}

function drawP0(kase, upTo) {
  const { ctx, w, h } = fitCanvas(el.p0);
  const n = kase.p0.length;
  // Min-max autoscaled, with BOTH ends of the axis printed. p0 sits on a floor
  // near the hazard prior (1/lambda = 0.02) and rarely approaches 1 — flags come
  // mostly from the run-length mode collapsing, not from p0 crossing 0.5 — so a
  // 0..1 axis is an empty strip and a 0..max axis is a solid block. What carries
  // information is the excursion above the floor; the printed range is there so
  // nobody reads that excursion as bigger than it is.
  const lo = Math.min(...kase.p0);
  const hi = Math.max(...kase.p0);

  // Degenerate case, and it is the interesting one: on the smoothed control
  // series the predictive is so flat across run lengths that p0 never leaves the
  // hazard prior at all. Autoscaling a range that small is meaningless, so say so.
  if (hi - lo < 1e-6) {
    drawFrame(ctx, w, h, `P(change | data) · flat at the hazard prior ${hi.toPrecision(3)} — no evidence of a change, ever`);
    const y = (PAD.top + h - PAD.bottom) / 2;
    ctx.strokeStyle = css('--observation');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y + 0.5);
    ctx.lineTo(xOf(upTo, n, w), y + 0.5);
    ctx.stroke();
    drawPlayhead(ctx, w, h, kase, upTo);
    return;
  }

  const span = hi - lo;
  const yOf = v => h - PAD.bottom - ((v - lo) / span) * (h - PAD.top - PAD.bottom);

  // Print enough significant digits to actually separate the two ends: on some
  // windows the whole excursion lives in the fifth decimal, and a fixed 3 digits
  // renders the axis as "0.0200 – 0.0200".
  let digits = 3;
  while (digits < 6 && lo.toPrecision(digits) === hi.toPrecision(digits)) digits += 1;
  const sig = v => v.toPrecision(digits);
  drawFrame(ctx, w, h, `P(change | data) · axis ${sig(lo)} – ${sig(hi)}`);
  ctx.fillStyle = css('--text-faint');
  ctx.font = '9px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(sig(hi), PAD.left - 6, yOf(hi) + 8);
  ctx.fillText(sig(lo), PAD.left - 6, h - PAD.bottom);

  // A line, not bars or a filled area: these series run to thousands of points,
  // and either of those degenerates into a solid block at screen width.
  ctx.strokeStyle = css('--observation');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= upTo; i++) {
    const x = xOf(i, n, w);
    const y = yOf(kase.p0[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  drawPlayhead(ctx, w, h, kase, upTo);
}

// ------------------------------------------------------------------ readout

function renderReadout(kase, i) {
  el.statDate.textContent = kase.dates[i];
  el.statObs.textContent = fmtValue(kase, kase.raw[i]);
  el.statObsLabel.textContent = kase.unit;
  el.statRun.textContent = kase.modeRun[i];
  el.statP0.textContent = kase.p0[i].toFixed(3);

  const recentFlag = kase.flags.some(f => f <= i && i - f < 5);
  el.statStatus.textContent = recentFlag ? 'regime change flagged' : 'stable regime';
  el.statStatus.classList.toggle('fired', recentFlag);
}

function render() {
  const kase = state.active;
  if (!kase) return;
  drawSeries(kase, state.cursor);
  drawRunLength(kase, state.cursor);
  drawP0(kase, state.cursor);
  renderReadout(kase, state.cursor);
  el.scrub.value = String(state.cursor);
}

// ----------------------------------------------------------------- playback

function stop() {
  state.playing = false;
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = null;
  el.play.textContent = '▶ Replay';
}

function tick(now) {
  if (!state.playing) return;
  const dt = Math.min(now - state.lastFrame, 100);
  state.lastFrame = now;
  // ~60 observations/second at 1x, scaled by the speed selector
  const step = Math.max(1, Math.round((dt / 1000) * 60 * Number(el.speed.value)));
  state.cursor = Math.min(state.cursor + step, state.active.series.length - 1);
  render();
  if (state.cursor >= state.active.series.length - 1) { stop(); return; }
  state.raf = requestAnimationFrame(tick);
}

function play() {
  if (!state.active) return;
  if (state.cursor >= state.active.series.length - 1) state.cursor = 0;
  state.playing = true;
  el.play.textContent = '❚❚ Pause';
  state.lastFrame = performance.now();
  state.raf = requestAnimationFrame(tick);
}

// -------------------------------------------------------------------- setup

function selectCase(id, { autoplay = true } = {}) {
  stop();
  const kase = state.cases.find(c => c.id === id);
  if (!kase) return;
  state.active = kase;
  state.cursor = 0;

  for (const tab of el.tabs.children) {
    tab.setAttribute('aria-selected', String(tab.dataset.id === id));
  }
  el.title.textContent = kase.title;
  el.sub.textContent = kase.subtitle;
  el.note.textContent = kase.note;
  el.xStart.textContent = kase.dates[0];
  el.xEnd.textContent = kase.dates[kase.dates.length - 1];
  el.scrub.max = String(kase.series.length - 1);

  render();
  if (autoplay) play();
}

function renderTabs() {
  el.tabs.innerHTML = '';
  for (const kase of state.cases) {
    const btn = document.createElement('button');
    btn.className = 'tab';
    btn.role = 'tab';
    btn.dataset.id = kase.id;
    btn.textContent = kase.title;
    btn.setAttribute('aria-selected', 'false');
    btn.addEventListener('click', () => selectCase(kase.id));
    el.tabs.appendChild(btn);
  }
}

function renderResults() {
  const rows = [];
  for (const kase of state.cases) {
    for (const b of kase.breaks) {
      const missed = b.latency === null;
      rows.push(`
        <tr class="${kase.control ? 'control' : ''}">
          <td>${kase.title}${kase.control ? ' <span class="tag">control</span>' : ''}<br>
              <span class="num" style="color: var(--text-faint)">${kase.subtitle}</span></td>
          <td>${b.name}</td>
          <td class="num">${b.date}</td>
          <td class="num">${missed ? '—' : b.flagDate}</td>
          <td class="num"><span class="latency ${missed ? 'miss' : ''}">${fmtLatency(b.latency)}</span></td>
          <td class="num">${kase.flags.length}</td>
        </tr>`);
    }
  }
  el.resultsBody.innerHTML = rows.join('');
}

function bindControls() {
  el.play.addEventListener('click', () => (state.playing ? stop() : play()));
  el.restart.addEventListener('click', () => { state.cursor = 0; render(); play(); });
  el.scrub.addEventListener('input', () => {
    stop();
    state.cursor = Number(el.scrub.value);
    render();
  });
  el.theme.addEventListener('click', () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    render();
  });
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 80);
  });
}

function showError(err) {
  el.error.hidden = false;
  el.error.innerHTML = `
    <p><strong>Could not load the market data.</strong> ${String(err)}</p>
    <p>The page reads the CSVs in <code>validation/markets/data/</code> over HTTP, so it has to be
    served from the <em>repository root</em> — not opened as a <code>file://</code> URL, and not
    served from inside <code>showcase/</code>:</p>
    <p><code>python3 -m http.server</code> &nbsp;→&nbsp; <code>http://localhost:8000/showcase/</code></p>`;
  el.resultsBody.innerHTML = '<tr><td colspan="6">—</td></tr>';
}

async function main() {
  try {
    state.cases = await Promise.all(CASES.map(spec => loadCase(spec)));
  } catch (err) {
    showError(err);
    return;
  }
  el.panel.hidden = false;
  renderResults();
  renderTabs();
  bindControls();
  selectCase(state.cases[0].id, { autoplay: false });
}

main();
