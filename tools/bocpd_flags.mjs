// bocpd_flags.mjs — run the browser detector over a CSV from the command line.
//
//   node tools/bocpd_flags.mjs validation/markets/data/vix.csv
//
// Prints JSON: {"n": <rows>, "flags": [<indices>], "flagDates": [...]}.
// This deliberately goes through showcase/replay.js, i.e. exactly the code path
// the showcase page runs in the browser, so tests/test_js_python_parity.py is
// comparing the shipped page against the Python engine rather than a lookalike.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseCsv, runBocpd, zscore } from '../showcase/replay.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const target = process.argv[2];
if (!target) {
  console.error('usage: node tools/bocpd_flags.mjs <csv path>');
  process.exit(2);
}

const abs = path.isAbsolute(target) ? target : path.resolve(process.cwd(), target);
if (!fs.existsSync(abs)) {
  console.error(`no such file: ${abs} (paths are relative to the repo root, ${path.resolve(here, '..')})`);
  process.exit(2);
}

const rows = parseCsv(fs.readFileSync(abs, 'utf8'));
const { flags } = runBocpd(zscore(rows.map(r => r.value)));
console.log(JSON.stringify({
  n: rows.length,
  flags,
  flagDates: flags.map(i => rows[i].date),
}));
