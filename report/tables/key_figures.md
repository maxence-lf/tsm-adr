| Quantity quoted in the text | Value | Source |
|---|---|---|
| Last FRED value | 2026-10-23 | fx_fred.csv |
| FRED values forward-filled since 2010 | 180 | fx_fred.csv |
| Forward-filled values used by prev_ny / same_day | 43 / 36 | premium CSVs |
| Empty weeks in the weekly prev_ny series | 12 | premium_prev_ny.csv |
| Last Yahoo price date | 2026-10-30 | raw_closes.csv |
| Manual check prev_ny premium | 18.81% | computed from raw_closes.csv, fx_fred.csv |
| Manual check same_day premium | 10.87% | computed |
| Manual check TSM change on d | -6.7% | raw_closes.csv |
| Manual check Yahoo vs FRED gap | 0.23% | raw_closes.csv, fx_fred.csv |
| Gross error 2011-10-25 prev_ny premium | -93.7% | recomputed |
| Gross error 2014-12-31 prev_ny premium | -88.3% | recomputed |
| Dates with |gap| > 1% | 406 | recomputed |
| of which flagged | 87 | recomputed |
| Flagged dates | 124 | recomputed |
| Share |gap|>1% in 2011 | 12% | recomputed |
| Share |gap|>1% in 2013 | 52% | recomputed |
| Share |gap|>1% in 2014 | 41% | recomputed |
| Share |gap|>1% in 2015 | 24% | recomputed |
| Share |gap|>1% in 2016 | 20% | recomputed |
| Share of flagged dates in 2011-2016 | 122 | recomputed |
| Verdict yahoo_point_off | 86 | recomputed |
| Verdict yahoo_neighbours_off | 28 | recomputed |
| Verdict no_fred_value | 9 | recomputed |
| Verdict fred_confirms | 1 | recomputed |
| Flagged dates comparable | 115 | recomputed |
| Mean |FRED deviation| on flagged dates | 0.10% | recomputed |
| Mean |Yahoo deviation| on flagged dates | 2.70% | recomputed |
| FRED values flagged by the same rule | 0 | recomputed |
| Persistent shifts | 9 | recomputed |
| Longest persistent shift (dates) | 48 | recomputed |
| Shortest persistent shift (dates) | 10 | recomputed |
| prev_ny 2011-2016 noise change | -17% | recomputed |
| prev_ny autocorr Yahoo / FRED | -0.453 / -0.452 | recomputed / premium CSVs |
| same_day 2011-2016 noise change | -17% | recomputed |
| same_day autocorr Yahoo / FRED | -0.441 / -0.436 | recomputed / premium CSVs |
| prev_ny min premium FRED | -6.0% | premium CSVs |
| same_day min premium FRED | -5.0% | premium CSVs |
| same_day daily change std (FRED) | 2.24% | premium CSVs |
| same_day autocorr (FRED) | -0.436 | premium CSVs |
| prev_ny daily change std (FRED) | 1.97% | premium CSVs |
| prev_ny autocorr (FRED) | -0.452 | premium CSVs |
| Premium sample | 2010-01-04 to 2026-10-23 | premium CSVs |
| Jumps above 10 points | 14 | premium_table.csv |
| Example 2025-04-08 same_day / prev_ny | 14.38% / 18.30% | premium_table.csv |
| Example 2025-04-09 same_day / prev_ny | 32.86% / 18.89% | premium_table.csv |
| Example 2025-04-10 same_day / prev_ny | 15.09% / 20.86% | premium_table.csv |
| Example TSM change 2025-04-09 | 12.3% | premium_table.csv |
| Example 2330.TW change 2025-04-10 | 9.9% | premium_table.csv |
| ADF p prev_ny daily | 0.23 | mean_reversion_results.csv |
| ADF p prev_ny weekly | 0.18 | mean_reversion_results.csv |
| ADF p same_day daily | 0.23 | mean_reversion_results.csv |
| ADF p same_day weekly | 0.39 | mean_reversion_results.csv |
| Weekly mean premium 2010-2019 | 3.1% | mean_reversion_results.csv |
| Weekly mean premium 2020-2023 | 7.3% | mean_reversion_results.csv |
| Weekly mean premium 2024-Apr 2026 | 19.2% | mean_reversion_results.csv |
| Weekly mean premium May 2026-end | 14.1% | mean_reversion_results.csv |
| AR(1) weekly/daily ratio by sub-period | 3.5, 4.1, 3.1 | mean_reversion_results.csv |
| Simulation AR(1), daily mean | 4.2 | recomputed simulation |
| Simulation AR(1), weekly mean | 14.0 | recomputed simulation |
| Simulation ARMA(1,1), daily mean | 35.0 | recomputed simulation |
| Simulation ARMA(1,1), weekly mean | 35.0 | recomputed simulation |
| Simulation true half-life | 34.3 | ln(0.5)/ln(0.98) |
| Simulation daily change autocorr | -0.44 | recomputed simulation |
| Simulation theta daily / weekly | -0.61 / -0.34 | recomputed simulation |
| Simulation replications | 30 | recomputed simulation |
| ARMA 2010-2019 daily / weekly | 61.9 / 63.1 | mean_reversion_results.csv |
| ARMA 2020-2023 daily / weekly | 54.6 / 63.9 | mean_reversion_results.csv |
| ARMA 2024-Apr 2026 daily / weekly | 53.4 / 69.1 | mean_reversion_results.csv |
| ARMA theta range daily | -0.76 to -0.63 | mean_reversion_results.csv |
| ARMA theta range weekly_friday | -0.59 to -0.44 | mean_reversion_results.csv |
| same_day ARMA daily / weekly by sub-period | 58.1 / 63.8; 53.1 / 71.2; 33.7 / 46.1 | mean_reversion_results.csv |
| Full-sample ARMA daily / weekly | 127.4 / 381.0 | mean_reversion_results.csv |
| Rolling ARMA(1,1) weekly, 3-year windows | median 69.6, IQR 34.6 to 122.3, >20 days 92%, lower bound >20 68%, windows 167, excluded 7 | mean_reversion_results.csv |
| Rolling ARMA(1,1) daily, 2-year windows | median 37.8, IQR 25.4 to 55.1, >20 days 81%, lower bound >20 41%, windows 178, excluded 13 | mean_reversion_results.csv |
| Rolling AR(1) weekly, 3-year windows | median 9.0, IQR 4.8 to 20.0, >20 days 25%, lower bound >20 13%, windows 167, excluded 0 | mean_reversion_results.csv |
| Rolling weekly ARMA median, windows ending 2013-2016 | 50.4 | mean_reversion_results.csv |
| Rolling weekly ARMA median, windows ending 2017-2019 | 25.4 | mean_reversion_results.csv |
| Rolling weekly ARMA median, windows ending 2020-2023 | 101.1 | mean_reversion_results.csv |
| Rolling weekly ARMA median, windows ending 2024-2026 | 157.5 | mean_reversion_results.csv |
| Weekly ARMA windows below 20 days | 13, ending 2018-03-30 to 2019-10-31 | mean_reversion_results.csv |
| Median weekly/daily ratio, rolling | 1.9 | mean_reversion_results.csv |
| Placebo Chow p-value range | 1.5e-14 to 2.7e-04 | mean_reversion_results.csv |
| Candidate Chow p-value range | 2.8e-07 to 1.9e-06 | mean_reversion_results.csv |
| Candidate weeks after | 26 to 30 | mean_reversion_results.csv |
| Since-2024 Chow p at 2026-01-02 | 0.001 | mean_reversion_results.csv |
| Since-2024 Chow p in 2025 | 0.232, 0.246 | mean_reversion_results.csv |
