# Pre-registration: step 6, out-of-sample test on other Taiwanese ADRs

Written on 2026-11-20, before any price, volume or dividend data for these securities
was downloaded or looked at. It is committed to the repository before any such
download; the commit hash and date are the record of what was fixed in advance.

The TSMC results (steps 1 to 5) were obtained on TSM / 2330.TW. This step applies
exactly the same methods to other Taiwanese ADRs, so that the TSMC findings can be
tested on data that played no part in designing them.

## 1. Securities, periods and ADS ratios

| Company | ADR (NYSE) | Local share (TWSE) | Common shares per ADS | Sample |
|---|---|---|---:|---|
| United Microelectronics (UMC) | UMC | 2303.TW | 5 | 2010-01-01 to 2026-10-30 |
| ASE Technology Holding | ASX | 3711.TW | 2 | 2018-04-30 to 2026-10-30 |
| Chunghwa Telecom | CHT | 2412.TW | 10 | 2011-03-01 to 2026-10-30 |

The end date is the last date of the TSMC sample. The start date is the later of the
date above and the first date on which both lines have a close.

Ratios were checked in the companies' annual reports on Form 20-F (SEC EDGAR). No
filing checked mentions a change of the ADS ratio over the sample.

- **UMC.** "each representing five common shares" (or "5 Common Shares") in the 20-F
  filed on 2010-04-30, 2018-04-26 and 2026-04-30. The ADSs have been listed on the
  NYSE since 2000-09-19 (20-F 2010); the deposit agreement is dated 2009-10-21. The
  filings checked mention capital reductions only for the 2007 one (before the sample)
  and for investees.
- **ASE Technology Holding.** Established on 2018-04-30 by a share exchange with
  Advanced Semiconductor Engineering, Inc. (ASE) and the acquisition of Siliconware
  Precision Industries (SPIL); its shares trade on the TWSE as 3711 and its ADSs on the
  NYSE under the ticker previously used by ASE, "ASX". Its ADSs represent two common
  shares (20-F filed on 2019-04-26 and 2026-04-01; deposit agreement dated 2018-04-30;
  157,164,076 ADSs representing 314,328,152 shares as of 2026-02-28). Before the
  exchange, ASE's ADSs represented five ASE common shares (20-F filed on 2018-03-28):
  a different security with a different ratio. **No data before 2018-04-30 is used**,
  even if a data provider joins the two histories under the ticker ASX.
- **Chunghwa Telecom.** "each representing 10 Common Shares" (or "ten") in the 20-F
  filed in 2010, 2011, 2012, 2013 and on 2026-04-15. ADSs listed on the NYSE as CHT since
  2003-07-17; shares on the TWSE as 2412 since 2000-10-27 (20-F 2010). The company
  carried out capital reductions from 2007 to 2010; the last one cut capital stock by
  20%, with the cash paid on 2011-01-25 (20-F 2012). The ratio stayed at 10, but the
  reduction changes the number of shares, and the dates on which each line's
  unadjusted price reflects it were not verified. **The sample therefore starts on
  2011-03-01**, after the reduction.
- **TSMC (check of an earlier assumption).** "Each ADS represents the right to receive
  five common shares" (20-F filed on 2010-04-15) and "each ADS represents five (5)
  common shares" (20-F filed on 2026-04-16); the 20-F filed on 2018-04-19 gives the same
  NYSE listing date, 1997-10-08. The ratio of 5 used in steps 1 to 5 is confirmed.

Filings were checked at three or more dates per company, not every year; a change
that was later reversed would not have been seen, which is considered unlikely.

## 2. Data

Same sources and same code as for TSMC: unadjusted daily closes from Yahoo Finance
(`yfinance`), ex-date dividends from Yahoo Finance, FRED DEXTAUS for USD/TWD, S&P 500
closes from Yahoo Finance. Dividends are checked as for TSMC: each ADR dividend
against the ratio times the local dividend in USD, gaps above 5% reported.

Data errors:
- A price or exchange-rate value is corrected only if an independent source confirms
  the error; every correction is documented (date, series, old value, new value,
  source).
- Daily returns above 25% in absolute value, on either line, are flagged for
  verification. A flagged value is kept unless the previous rule applies.

## 3. Methods (identical to TSMC, no new parameter)

All parameters are those of the TSMC code at commit `c3dd0a6`.

1. **FX.** FRED DEXTAUS restricted to the sample, values outside [25, 40] removed,
   values more than 2% from a centred 5-day rolling median flagged for review and
   kept, gaps forward-filled for at most 3 business days.
2. **Premium.** premium = ADR / (ratio x local / FX) - 1, with the ratios above.
   `prev_ny` (last NY close strictly before the Taipei date, FX on that NY date) is the
   reference; `same_day` is the robustness check. Asynchrony diagnostics: level, std
   and lag-1 autocorrelation of daily changes under both definitions.
3. **Mean reversion.** q = ln(1 + premium). ADF and KPSS on the full sample; AR(1) and
   ARMA(1,1) half-lives, daily and weekly (last observation of each week, 5 trading days
   per week). Rolling windows with a monthly step: weekly ARMA(1,1) and weekly AR(1) on
   the same 156-week windows, daily ARMA(1,1) on 504-day windows; ARMA windows that do
   not converge or have phi > 0.999 are excluded. The four calendar sub-periods of
   TSMC (2010-2019, 2020-2023, 2024 to April 2026, May 2026 to the end) are reported
   descriptively where they contain data.
4. **Factors.** Weekly change of q regressed on the S&P 500 log return (read on the
   ADR's NY date) and the change in log FX (TWD per USD); Newey-West standard errors
   with floor(4 (n / 100)^(2/9)) lags, at least 1. ADR and local (in USD) S&P 500 betas
   from separate weekly regressions, as for TSMC.
5. **Strategy.** Exactly the TSMC rules and engine: weekly z-score of the `prev_ny` q
   against the 26 previous weeks, entry z > 1.5 (long local, short ADR, 1 USD per leg),
   exit z < 0 or after 26 weeks, short premium only; ADR executed at the first NY close
   on or after the Taipei decision date, local share at the next Taipei close, the local
   leg converted at the FRED rate of its own execution dates; P&L on executable prices.
   Robustness exactly as in step 5b: block bootstrap of the weekly Sharpe ratio
   (13-week blocks, 5,000 draws, seed 20261120), random-entry Monte-Carlo (2,000
   replications, same seed), one-day execution delay, S&P 500 beta hedge (beta on the
   156 weeks before each entry, at least 52, 1 bp per hedge trade).

## 4. Costs

- **Default:** the TSMC costs. ADR commission 5 bp, local commission 10 bp, Taiwan
  securities transaction tax 30 bp on sales of the local share, FX conversion 5 bp,
  ADR borrow fee 0.5% a year, withholding 21% on local dividends, the short pays the
  gross ADR dividend. All assumptions, not verified.
- **Higher costs, for these less liquid lines (fixed now):** commissions and FX doubled
  to stand in for wider spreads (ADR 10 bp, local 20 bp, FX 10 bp), borrow fee doubled
  to 1.0% a year; tax (30 bp) and withholding (21%) unchanged.
- Financing as for TSMC: the short-sale proceeds fund the local purchase; no interest
  income or financing cost apart from the borrow fee.

## 5. Hypotheses and confirmation criteria

Each hypothesis is evaluated separately for each security and reported for each one.
There is no pooling and no correction for multiple testing.

**Aggregation rule.** A hypothesis is "generalised" if it is confirmed for at least 2
of the 3 securities, and "not generalised" otherwise. This rule is the only one used
to draw conclusions.

| | Hypothesis (TSMC finding) | Confirmed if |
|---|---|---|
| H1 | Asynchrony noise | lag-1 autocorrelation of daily changes of the `prev_ny` premium, full sample, < -0.2 |
| H2 | A naive AR(1) understates the half-life | median weekly ARMA(1,1) half-life over the rolling 156-week windows > 2 x the median weekly AR(1) half-life over the same windows |
| H3 | The ADR has a higher US market beta than the local share | beta(ADR) - beta(local, USD) > 0 with a 95% Newey-West interval above 0; estimated as the slope of the weekly change of q on the S&P 500 return, full sample, which equals that difference exactly since dq = r(ADR) - r(local, USD) |
| H4 | A simple strategy does not survive costs | 95% block-bootstrap interval of the net weekly Sharpe ratio, default costs, contains 0 or lies below 0 |
| H5 | (idea from the TSMC results, tested only here) a signal at the NY close does better | the difference in total net P&L (default costs), variant below minus `prev_ny` strategy, is positive |

**H4, reporting.** The estimated net Sharpe ratio and the width of its 95% interval
are reported. If the width exceeds 0.8, a confirmation is qualified as "weak (low
power)"; it still counts as confirmed for the aggregation rule.

**H5, reporting.** The difference in net P&L (variant minus `prev_ny`) is reported
with a 95% interval for that total from a block bootstrap of the paired weekly net P&L
differences (each draw sums the resampled weekly differences),
with the parameters of step 5b (13-week blocks, 5,000 draws, seed 20261120). H5 is
"confirmed" if the difference is positive and "strongly confirmed" only if the interval
excludes 0. For the aggregation rule, both count as confirmed.

**H5 variant.** Same rules (26-week z-score, entry 1.5, exit 0 or 26 weeks, short
premium only), with q computed from the `same_day` premium (NY close and Taipei close
of the same date d, FX on date d), sampled on the last date of each week on which both
markets traded. The decision is taken at the NY close of d; the ADR is executed at the
next NY close (first NY date > d) and the local share at the next Taipei close (first
Taipei date > d), the local leg converted at the FRED rate of its execution dates. The
engine extension will be written and tested (no decision uses later data, every leg
trades after its decision) before it is run on these securities. H5 is not evaluated
on TSMC, whose results suggested it.

## 6. Commitments

- Every result listed above will be reported for every security, including results
  that contradict the TSMC findings.
- No rule, parameter, sample date or cost will be changed after the data has been
  seen. If a data problem makes an analysis impossible, the problem and the analysis
  dropped are reported.
- Any analysis not listed in this document will be labelled "exploratory".
