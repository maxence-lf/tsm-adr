# TSMC ADR Premium

Measuring and modelling the premium of TSMC's NYSE-listed ADR (TSM) over its
Taiwan-listed shares (2330.TW), and asking why arbitrage does not close it.

**Status:** work in progress — data pipeline and premium construction done;
statistical analysis under way. Target completion: February 2027.

## Research questions
1. What drives the premium?
2. Does it mean-revert, and how fast?
3. Does a strategy trading it survive transaction costs?
4. What limits arbitrage (conversion constraints, foreign ownership rules, short-selling)?

## Key methodological issue: non-synchronous prices
Taipei closes at 05:30 UTC, New York at 20:00/21:00 UTC, so no pair of closing
prices is observed at the same time. The premium is built under two alignments
(same calendar date vs. previous NY close) and the measurement noise induced by
asynchrony is quantified before any modelling.

Diagnostics with the FRED exchange rate, 2010-01-04 to 2026-10-23:

| Definition | Observations | Mean premium | Premium std | Daily change std | Daily change autocorrelation (lag 1) |
|---|---:|---:|---:|---:|---:|
| `same_day` | 3,975 | 6.65% | 6.44% | 2.24% | -0.436 |
| `prev_ny` | 4,107 | 6.54% | 6.30% | 1.97% | -0.452 |

Daily changes have an autocorrelation of about -0.44 under both definitions, a
sign of measurement noise: a gap caused by asynchrony reverses the next day. It is
unchanged by the FX source (-0.441 and -0.453 with cleaned Yahoo data), so this
noise comes from the non-synchronous closes and not from the exchange rate.

## Data
- Prices: daily unadjusted closes from Yahoo Finance via `yfinance` (unofficial
  API) for TSM (USD) and 2330.TW (TWD).
- Exchange rate: FRED series DEXTAUS (Federal Reserve H.10, TWD per USD), recorded
  at noon New York time.
- Yahoo TWD=X is kept for comparison only (`scripts/compare_fx.py`).

Raw data is not redistributed.

## Data cleaning
Yahoo's TWD=X series contains two gross errors, 1.8015 on 2011-10-25 and 3.67 on
2014-12-31, against 30.11 and 31.60 in FRED. They create premiums of about -94%
and -88%.

![Premium computed from raw Yahoo FX, with the two errors](figures/premium_raw_with_fx_errors.png)

Comparing the two sources reveals a less visible problem. Between 2011 and 2016,
Yahoo often departs from FRED: the gap exceeds 1% on 12% of dates in 2011, 52% in
2013, 41% in 2014, 24% in 2015 and 20% in 2016, against about 1% in most years
since 2017. From 2013 to 2016 this includes nine persistent shifts of 10 to 48
consecutive dates at about -2.5%, such as 2013-09-09 to 2013-11-18. A rolling median
cannot detect them, since the neighbouring values are shifted too: only 87 of the
406 dates with a gap above 1% are flagged by the 2% rule on Yahoo's own series.
FRED shows the same move on just 1 of the 124 flagged dates, and the rule flags
none of FRED's own values.

![Relative gap between Yahoo TWD=X and FRED DEXTAUS](figures/fx_yahoo_vs_fred.png)

FRED is therefore used for the premium. The alignment rule takes the rate on the
NY date of the ADR used, so it is never later than the ADR price:

- `same_day`: FRED rate of day d.
- `prev_ny`: FRED rate of `adr_date`, the previous NY session.

The rate is recorded 4 hours before the ADR close. Missing values are forward-filled
for at most 3 business days, and never past the last published value.
`scripts/build_premium.py --fx-source yahoo` reproduces the earlier Yahoo version,
and `--no-clean` rebuilds the raw figure above.

## Reproduce
Requires Python 3.12.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q
    python scripts/build_premium.py

## Roadmap
- [x] Data pipeline and premium construction (two alignments, unit tests)
- [x] Asynchrony diagnostics and choice of reference definition
- [ ] Stationarity, mean reversion (ADF, AR(1)/OU half-life), structural breaks
- [ ] Explanatory factors (flows, FX, relative US/Taiwan returns)
- [ ] Walk-forward strategy with transaction costs, bootstrapped Sharpe
- [ ] Extension to other Taiwanese ADRs (UMC, ASE, Chunghwa Telecom)

## Tools
Python, pandas, matplotlib, pytest. Development assisted by Claude Code.