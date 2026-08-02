"""Fetch a real VIX slice for the markets validation.

Downloads daily CBOE VIX closes from the `datasets/finance-vix` GitHub mirror
(ODC-PDDL packaging; the underlying index data originates from CBOE) and writes
the two-column `date,value` CSV that `validate_regimes.py` consumes. FRED's
`VIXCLS` (https://fred.stlouisfed.org/series/VIXCLS) is the equivalent primary
source when reachable — its CSV works as-is per data/README.md.

The default window 2007-06-01 .. 2021-01-29 is the README's suggested slice: it
spans both documented structural breaks (Lehman 2008-09-15, COVID 2020-02-20)
while keeping the BOCPD run small.

Usage (from the repo root):
    python validation/markets/data/fetch_vix.py
    python validation/markets/data/fetch_vix.py --start 2007-06-01 --end 2021-01-29 --out validation/markets/data/vix.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import urllib.request

SOURCE_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
DEFAULT_START = "2007-06-01"
DEFAULT_END = "2021-01-29"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vix.csv")


def slice_rows(csv_text: str, start: str, end: str) -> list[tuple[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    out = []
    for row in reader:
        d = row["DATE"].strip()
        v = row["CLOSE"].strip()
        if start <= d <= end and v not in ("", "."):
            out.append((d, v))
    out.sort(key=lambda r: r[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    rows = slice_rows(text, args.start, args.end)
    if not rows:
        raise SystemExit(f"no rows in [{args.start}, {args.end}] — check the source window")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "value"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({rows[0][0]} .. {rows[-1][0]}) to {args.out}")


if __name__ == "__main__":
    main()
