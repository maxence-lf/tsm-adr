| Item | Value | Source |
|---|---:|---|
| Taipei date d | 2026-06-23 | `premium_table.csv` |
| NY date used by `prev_ny` (`adr_date`) | 2026-06-22 | previous NY session |
| TSM close on `adr_date` (USD) | 467.67 | `raw_closes.csv` |
| 2330.TW close on d (TWD) | 2,490.0 | `raw_closes.csv` |
| FRED USD/TWD on `adr_date` | 31.63 | `fx_fred.csv` |
| Local value of one ADR: 5 x 2,490.0 / 31.63 (USD) | 393.61 | computed |
| `prev_ny` premium: 467.67 / 393.61 - 1 | 18.81% | computed |
| `prev_ny` premium in `premium_table.csv` | 18.81% | pipeline |
| TSM close on d (USD) | 436.39 | `raw_closes.csv` |
| FRED USD/TWD on d | 31.63 | `fx_fred.csv` |
| `same_day` premium: 436.39 / 393.61 - 1 | 10.87% | computed |
| Yahoo TWD=X on `adr_date`, for comparison | 31.7020 | `raw_closes.csv` |
