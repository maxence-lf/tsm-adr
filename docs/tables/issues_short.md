| Issue | Before -> after |
|:---|:---|
| a) Errors in the Yahoo exchange rate | Minimum premium -93.7% with Yahoo -> -6.0% with the official FRED rate; volatility of daily changes over 2011-2016 2.09% -> 1.74% |
| b) Noise from non-synchronous closes | Daily changes read as real moves -> treated as noise: lag-1 autocorrelation -0.45, unchanged with the FRED rate |
| c) Naive AR(1) half-life biased downwards | Weekly half-life within sub-periods 7.8-17.0 days (AR(1)) -> 63.1-69.1 days (signal-plus-noise ARMA(1,1)) |
| d) Sub-periods chosen after looking at the data | Point estimates by sub-period -> a range from rolling windows: median 69.6 days over all windows, 32.3 days in windows without a level shift |
| e) Chow test at chosen break dates | Break in April 2026 claimed -> not conclusive (7 of 7 placebo dates also reject); the decline is dated from monthly means (25.7% in December 2025, 20.2% in January 2026) |
| h) P&L computed on premium changes | Convergence counted on the observed premium -> about 2.4 points per trade cannot be captured (1.47 already gone at entry, 0.91 at exit); P&L computed on execution prices |
| k) Dependence on the execution day | Net P&L 0.088 when each leg trades at its first close after the decision -> -0.191 when both legs trade one day later |
| m) Rules designed on TSMC | Strategy judged on TSMC (net P&L +0.088) -> loses on all three out-of-sample ADRs (UMC -0.340, ASE -0.240, Chunghwa -0.049) |
