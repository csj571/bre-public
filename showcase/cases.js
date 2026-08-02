// cases.js — what gets replayed, and what counts as a hit.
//
// The `breaks` dates are PRE-REGISTERED: they are the onsets committed in
// validation/markets/validate_regimes.py before any of this data was scored,
// and they are the only dates the latency numbers are measured against.
//
// Paths are relative to the repo root, so serve the repo root over HTTP and
// open /showcase/ (see showcase/README.md).

export const LEHMAN = { name: 'Lehman Brothers bankruptcy', date: '2008-09-15' };
export const COVID = { name: 'COVID-19 crash', date: '2020-02-20' };

export const CASES = [
  {
    id: 'gfc',
    title: 'Global Financial Crisis',
    subtitle: 'S&P 500 daily log returns · 2007–2009',
    path: '../validation/markets/data/sp500_logret_gfc.csv',
    unit: 'daily log return',
    scale: 100,               // display as %
    unitLabel: '%',
    breaks: [LEHMAN],
    note: 'Near-i.i.d. within a regime — the input the BOCPD model actually assumes.',
  },
  {
    id: 'covid',
    title: 'COVID-19 crash',
    subtitle: 'S&P 500 daily log returns · 2019–2020',
    path: '../validation/markets/data/sp500_logret_covid.csv',
    unit: 'daily log return',
    scale: 100,
    unitLabel: '%',
    breaks: [COVID],
    note: 'Same detector, same defaults, different decade.',
  },
  {
    id: 'vix',
    title: 'Both crises, one unbroken run',
    subtitle: 'CBOE VIX daily close · Jun 2007 – Jan 2021',
    path: '../validation/markets/data/vix.csv',
    unit: 'VIX',
    scale: 1,
    unitLabel: '',
    breaks: [LEHMAN, COVID],
    note: '13.6 years, no resets between crises, no per-window tuning.',
  },
  {
    id: 'control',
    title: 'Negative control',
    subtitle: '21-day realized volatility · 2019–2020',
    path: '../validation/markets/data/sp500_rv21_covid.csv',
    unit: 'annualized vol',
    scale: 1,
    unitLabel: '%',
    breaks: [COVID],
    control: true,
    note: 'A smoothed, strongly autocorrelated series violates the i.i.d.-within-regime '
      + 'assumption. Vanilla BOCPD absorbs the rise instead of flagging it — the failure '
      + 'the literature predicts, and the reason the headline runs on log returns.',
  },
];
