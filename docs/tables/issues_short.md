| Issue | Impact (before -> after) |
|:---|:---|
| a) Errors in the Yahoo exchange rate | `prev_ny` daily change std over 2011-2016: 2.09% -> 1.74%; minimum premium -93.7% -> -6.0% |
| b) Measurement noise from non-synchronous closes | Autocorrelation `prev_ny`: -0.453 (Yahoo) -> -0.452 (FRED): the noise is not from the exchange rate |
| c) AR(1) half-life biased downwards | Weekly half-life by sub-period: 7.8-17.0 (AR(1)) -> 63.1-69.1 days (ARMA) |
| d) Sub-periods chosen after looking at the data | Weekly ARMA median 69.6 days, above 20 days in 92% of windows; its variation across windows (25.4 to 157.5 days) is rank-correlated with level shifts inside the windows (0.86, against 0.35 and 0.48 in a no-regime simulation); windows with a shift below 2 points: median 32.3 days, likely a lower bound |
| e) Chow test at chosen break dates | Break at the candidate dates: claimed -> not conclusive (7 of 7 placebo dates reject); monthly mean 25.7% in December 2025, 20.2% in January 2026 |
| f) US factors read on the wrong date (step 4) | Main-model S&P 500 coefficient on correctly aligned data: 0.24 |
| g) Two collinear US indices (step 4) | Univariate S&P 500 0.203 and Nasdaq-100 0.196 -> jointly -0.063 and 0.278 |
| h) P&L computed on premium changes (step 5) | About 2.4 points per trade of apparent convergence are not capturable |
| i) Exchange rate of the local leg (step 5) | Gross P&L +0.0452 in total, +13.7 bp per trade (t = 1.8) |
| j) Daily marks of two asynchronous legs (step 5) | Gross annual volatility 19.3% (daily) -> 10.5% (weekly); gross Sharpe 0.11 -> 0.21 |
| k) Dependence on the execution day (step 5) | Gross P&L 0.360 -> 0.080; net 0.088 -> -0.191; correlation of the exit loss with the asynchrony gap 0.56 -> 0.08 without July 2020 |
| l) A hedge that seems to work (step 5) | Net Sharpe 0.05 -> 0.10, but weekly volatility 10.5% -> 10.3%: the gain comes from the index rising |
| m) Rules designed on TSMC (step 6) | Net P&L: TSMC +0.088 (in sample) against UMC -0.340, ASE -0.240, Chunghwa -0.049 |
| n) Aggregation rule of H5 (step 6) | Intervals: UMC [-0.193, +0.997]; ASE [-0.190, +0.932]; Chunghwa [-0.293, +0.113] |
| o) Plan text against plan code (step 6) | Same H2 verdict for all three securities |
| p) Band computed on noisy deviations (step 7) | Weeks above D* (default, T = h): 4% to 23%; asynchrony per TSMC trade larger than D* |
