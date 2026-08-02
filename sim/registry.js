// registry.js — Prior registry with version tracking (in-memory).
//
// Priors are treated as named **seeds** with provenance and a lifecycle: a seed
// lies dormant until its context recurs, at which point it is *retrieved in
// context* ("germinated") rather than re-minted. Promotions are themselves
// germination events. All of this is additive — the original
// {id,timestamp,author,justification,snapshot,diff} keys are preserved, so older
// consumers keep working; the new fields default sensibly.

function clone(x) { return JSON.parse(JSON.stringify(x)); }

export class PriorRegistry {
  constructor() {
    this.versions = [];
    this.counter = 0;
  }

  seed(initial, meta = {}) {
    const v = {
      id: 'v1',
      timestamp: Date.now(),
      author: meta.author || 'human seed',
      justification: meta.justification || 'Initial seed prior for BRE-1 session.',
      snapshot: clone(initial),
      diff: null,
      // ── semantic-seeding fields ──
      name: meta.name || 'Seed prior',
      provenance: meta.provenance || { source: 'human seed', regime: null, tick: 0, parentId: null },
      dormancy: 0,           // ticks since seeded without a retrieval
      retrievals: 0,         // times germinated-in-context
      lastRetrievedTick: null,
      germinated: false      // a hand seed, not a germination event
    };
    this.versions = [v];
    this.counter = 1;
    return v;
  }

  promote(snapshot, justification, author = 'BRE-1', meta = {}) {
    this.counter += 1;
    const prev = this.versions[this.versions.length - 1];
    const v = {
      id: 'v' + this.counter,
      timestamp: Date.now(),
      author,
      justification,
      snapshot: clone(snapshot),
      diff: diffSnapshots(prev?.snapshot, snapshot),
      // ── semantic-seeding fields ──
      name: meta.name || ('Prior v' + this.counter),
      provenance: meta.provenance || { source: author, regime: null, tick: null, parentId: prev?.id || null },
      dormancy: 0,
      retrievals: 0,
      lastRetrievedTick: null,
      germinated: meta.germinated ?? true   // promotions are germination events
    };
    this.versions.push(v);
    return v;
  }

  // Age every non-current seed by deltaTicks; the current prior stays "active".
  tickDormancy(deltaTicks = 1) {
    const cur = this.current();
    for (const v of this.versions) { if (v !== cur) v.dormancy += deltaTicks; }
  }

  // Mark a seed as retrieved-in-context: its dormancy resets and its retrieval
  // count climbs. Returns the seed, or null if the id is unknown.
  retrieve(id, tick) {
    const v = this.versions.find(x => x.id === id);
    if (!v) return null;
    v.dormancy = 0;
    v.retrievals += 1;
    v.lastRetrievedTick = tick ?? null;
    return v;
  }

  // Find a dormant (non-current) seed whose provenance regime matches the given
  // key — the candidate to germinate when that context recurs. Most-recent first.
  matchRegime(regimeKey) {
    if (regimeKey == null) return null;
    const cur = this.current();
    for (let i = this.versions.length - 1; i >= 0; i--) {
      const v = this.versions[i];
      if (v === cur) continue;
      if (v.provenance && v.provenance.regime === regimeKey) return v;
    }
    return null;
  }

  current() {
    return this.versions[this.versions.length - 1];
  }

  list() {
    return this.versions.slice();
  }

  reset(initial, meta = {}) {
    this.versions = [];
    this.counter = 0;
    return this.seed(initial, meta);
  }
}

function fmt(v) {
  if (typeof v === 'number') return v.toFixed(3);
  if (Array.isArray(v)) return '[' + v.map(fmt).join(',') + ']';
  if (v && typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export function diffSnapshots(a, b) {
  if (!a) return null;
  const out = [];
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  for (const k of keys) {
    const av = a?.[k], bv = b?.[k];
    if (av && typeof av === 'object' && !Array.isArray(av)) {
      const sub = diffSnapshots(av, bv || {});
      if (sub && sub.length) out.push({ key: k, subdiff: sub });
    } else if (JSON.stringify(av) !== JSON.stringify(bv)) {
      out.push({ key: k, from: fmt(av), to: fmt(bv) });
    }
  }
  return out;
}
