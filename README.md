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

## Data
Daily unadjusted closes from Yahoo Finance via `yfinance` (unofficial API):
TSM (USD), 2330.TW (TWD), TWD=X (USD/TWD). Raw data is not redistributed.

## Reproduce
Requires Python 3.12.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q
    python scripts/build_premium.py

## Roadmap
- [x] Data pipeline and premium construction (two alignments, unit tests)
- [ ] Asynchrony diagnostics and choice of reference definition
- [ ] Stationarity, mean reversion (ADF, AR(1)/OU half-life), structural breaks
- [ ] Explanatory factors (flows, FX, relative US/Taiwan returns)
- [ ] Walk-forward strategy with transaction costs, bootstrapped Sharpe
- [ ] Extension to other Taiwanese ADRs (UMC, ASE, Chunghwa Telecom)

## Tools
Python, pandas, matplotlib, pytest. Development assisted by Claude Code.