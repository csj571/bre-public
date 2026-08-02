"""Materialize REAL market data for the regime validation (see data/README.md).

The validation needs an external, dated series — intentionally not vendored in
this repo. This script materializes one locally from the `skfolio` package's
bundled S&P 500 index dataset (real daily closes, 1990-01-02 .. 2022-12-28,
shipped inside the wheel — installable offline from PyPI: `pip install skfolio`).

Provenance gate: before writing anything, four closes are checked EXACTLY against
independently documented historical values (Lehman day, the 2020-02-19 pre-COVID
peak, the 2020-03-23 COVID low, the 2008-11-20 GFC low). Any mismatch aborts.

Outputs (CSV, `date,value`) under validation/markets/data/:
  sp500_logret_gfc.csv    daily log returns, 2007-01-01 .. 2009-12-31
  sp500_logret_covid.csv  daily log returns, 2019-01-01 .. 2020-12-31
  sp500_rv21_gfc.csv      21-day realized vol (annualized %), same GFC window
  sp500_rv21_covid.csv    21-day realized vol (annualized %), same COVID window

Run from the repo root, then score each window with the (unmodified) harness:

    python validation/markets/fetch_data.py
    python validation/markets/validate_regimes.py --data validation/markets/data/sp500_logret_gfc.csv
    python validation/markets/validate_regimes.py --data validation/markets/data/sp500_logret_covid.csv

Why log returns: the engine's BOCPD (Normal-Inverse-Gamma / Student-t) assumes
observations are i.i.d. within a regime. Daily log returns are approximately so,
and a volatility regime shift is a variance changepoint the NIG model detects
natively. The smoothed realized-vol series deliberately violates the i.i.d.
assumption (strong autocorrelation) and is included as a documented negative
control — the failure mode the BOCPD literature predicts for vanilla Adams-MacKay.
"""
import csv
import math
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Independently documented S&P 500 closes used as an integrity gate.
KNOWN_CLOSES = {
    "2008-09-15": 1192.70,   # Lehman bankruptcy day
    "2008-11-20": 752.44,    # GFC closing low
    "2020-02-19": 3386.15,   # pre-COVID all-time high
    "2020-03-23": 2237.40,   # COVID closing low
}

WINDOWS = {
    "gfc": ("2007-01-01", "2009-12-31"),
    "covid": ("2019-01-01", "2020-12-31"),
}

RV_WINDOW = 21  # trading days


def _load_sp500_from_wheel():
    """Read the wheel's bundled CSV directly, without importing the package.

    skfolio 0.7.0 crashes at import time against cvxpy >= 1.7 (a TypeError in
    skfolio.typing). The dataset itself is just a csv.gz inside the wheel, so
    when the package import fails we locate the file via importlib (which does
    not execute skfolio/__init__.py) and parse it with stdlib gzip+csv.
    """
    import csv as _csv
    import gzip
    import importlib.util
    spec = importlib.util.find_spec("skfolio")
    if spec is None or not spec.submodule_search_locations:
        print("Missing dependency: pip install skfolio")
        sys.exit(2)
    path = os.path.join(list(spec.submodule_search_locations)[0],
                        "datasets", "data", "sp500_index.csv.gz")
    if not os.path.exists(path):
        print(f"skfolio is installed but its bundled dataset is missing at {path}.")
        sys.exit(2)
    dates, values = [], []
    with gzip.open(path, "rt") as f:
        reader = _csv.reader(f)
        next(reader)                          # header: Date,SP500
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                dates.append(row[0].strip())
                values.append(float(row[1]))
    return dates, values


def load_sp500():
    try:
        from skfolio.datasets import load_sp500_index
    except ImportError:
        print("Missing dependency: pip install skfolio")
        sys.exit(2)
    except Exception as exc:
        # skfolio installed but its import chain is broken (e.g. skfolio 0.7.0
        # vs cvxpy >= 1.7). The data doesn't need the optimizer stack: fall
        # back to reading the bundled CSV directly. Integrity gate still runs.
        print(f"skfolio package import failed ({type(exc).__name__}: {exc}); "
              f"reading the wheel's bundled dataset directly.")
        return _load_sp500_from_wheel()
    frame = load_sp500_index()
    series = frame[frame.columns[0]]
    dates = [ts.date().isoformat() for ts in series.index]
    values = [float(v) for v in series.values]
    return dates, values


def verify(dates, values):
    lookup = dict(zip(dates, values))
    for date, expected in KNOWN_CLOSES.items():
        got = lookup.get(date)
        if got is None or abs(got - expected) > 0.02:
            print(f"INTEGRITY FAILURE at {date}: got {got}, expected {expected}. "
                  f"Refusing to write data.")
            sys.exit(1)
        print(f"  verified close {date} = {got}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "value"])
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    dates, values = load_sp500()
    print(f"Loaded S&P 500 closes: {dates[0]} .. {dates[-1]} ({len(dates)} rows)")
    verify(dates, values)

    logret = [(dates[i + 1], math.log(values[i + 1] / values[i]))
              for i in range(len(values) - 1)]

    rv = []
    for i in range(RV_WINDOW, len(logret) + 1):
        window = [r for _, r in logret[i - RV_WINDOW:i]]
        mean = sum(window) / RV_WINDOW
        var = sum((x - mean) ** 2 for x in window) / RV_WINDOW
        rv.append((logret[i - 1][0], math.sqrt(var * 252) * 100.0))

    os.makedirs(DATA_DIR, exist_ok=True)
    for name, (lo, hi) in WINDOWS.items():
        write_csv(os.path.join(DATA_DIR, f"sp500_logret_{name}.csv"),
                  [(d, f"{v:.10f}") for d, v in logret if lo <= d <= hi])
        write_csv(os.path.join(DATA_DIR, f"sp500_rv21_{name}.csv"),
                  [(d, f"{v:.6f}") for d, v in rv if lo <= d <= hi])
    print("Done. Score each window with validate_regimes.py --data <file>.")


if __name__ == "__main__":
    main()
