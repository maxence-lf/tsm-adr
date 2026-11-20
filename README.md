# TSMC ADR Premium

TSMC's ADR on the NYSE (TSM) trades at a persistent premium over the same shares in Taipei
(2330.TW), in apparent breach of the law of one price. This project measures the premium
from 2010 to 2026 and asks why arbitrage does not close it, and who pays for it.

## Key results

- **Measurement.** The two markets never trade at the same time: half of each daily
  change in the premium reverses the next day (lag-1 autocorrelation -0.45), and Yahoo's
  exchange rate contains errors, so the official FRED rate is used.
- **Slow reversion.** Corrected for this noise with a signal-plus-noise model, the
  premium's half-life is roughly 30 to 70 trading days, not the few days of a naive AR(1).
- **US demand.** The ADR carries more US market risk than the local share (S&P 500 beta
  1.06 against 0.85), so a long local / short ADR pair has a beta of -0.20.
- **No profitable strategy.** A strategy with rules fixed before any backtest earns 0.360
  before costs and 0.088 after (per 1 USD per leg); its net Sharpe ratio of 0.05 has an
  interval of [-0.19, 0.29], costs of 82 bp per round trip absorb most of a 109 bp gross
  edge, and the transaction tax is 39% of the costs. It loses on three other Taiwanese
  ADRs.
- **Limits to arbitrage.** Regulation caps the creation of new ADRs, and costs and slow
  reversion imply a no-arbitrage band of about 1 to 2 points of premium, a lower bound.

![TSM ADR premium over 2330.TW](figures/premium_history.png)

## What distinguishes this project

- **Asynchrony corrected** before any modelling: two definitions of the premium, a
  simulation of the bias it creates, and a model that separates the true premium from
  the noise.
- **P&L on executable prices**: each leg trades at its first close after the decision,
  because about 2.4 points per trade of the apparent convergence in the premium cannot be
  captured.
- **Pre-registered out-of-sample test** on UMC, ASE and Chunghwa Telecom: the plan was
  committed and tagged (`prereg-step6`) before any of their data was downloaded, with the
  same methods and no new parameter. Asynchrony, the US beta of the ADR and the failure
  of the strategy hold on 3 of 3 securities.

## Reports

- [Short report](docs/report.md): the question, the main results and what is
  demonstrated, derived or exploratory.
- [Technical report](docs/technical_report.md): every step with its naive approach,
  problem, detection, correction and impact; what did not work; what is not verified;
  all tables.
- [Pre-registration of the out-of-sample test](docs/preregistration_step6.md).

## Reproduce

Requires Python 3.12. Raw data is not redistributed: the scripts download and cache it.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q
    python scripts/build_premium.py
    python scripts/build_premium.py --fx-source yahoo
    python scripts/build_premium.py --no-clean
    python scripts/export_premium_table.py
    python scripts/compare_fx.py
    python scripts/mean_reversion.py
    python scripts/simulate_level_shift.py
    python scripts/make_report_figures.py
    python scripts/factors.py
    python scripts/backtest.py
    python scripts/backtest_robustness.py
    python scripts/step6_out_of_sample.py
    python scripts/arbitrage_band.py
    python scripts/make_report_tables.py
    python scripts/check_report_numbers.py

## Repository structure

    src/adr_premium/   data loading, FX cleaning, premium, statistics, factors,
                       backtest engine, robustness tools, no-arbitrage band
    scripts/           one script per analysis, plus report tables and checks
    tests/             unit tests, look-ahead tests and the TSMC non-regression test
    docs/              reports, pre-registration and generated tables
    figures/           figures used in the reports
    data/              local cache (not redistributed)

## Tools

Python, pandas, NumPy, statsmodels, SciPy, matplotlib, yfinance, pytest. Development
assisted by Claude Code.
