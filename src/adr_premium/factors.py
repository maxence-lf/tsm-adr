"""Factor regressions for the ADR premium.

Convention: q = ln(1 + premium). With FX in TWD per USD, 1 + premium =
P_ADR * FX / (5 * P_local), so q = ln P_ADR + ln FX - ln P_local - ln 5 and
dq = r_ADR - r_local + d ln FX exactly.

Alignment: every US factor is read on the NY date of the ADR price used by the
premium (``adr_date``), never on the Taipei date. For prev_ny this is the previous
NY session; for same_day it is the same calendar date. The date actually read is
stored next to each factor so that ``check_alignment`` can verify it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

US_LEVELS = {"sp500": "spx", "ndx": "ndx"}  # panel name -> column of the market data
FACTORS = ["r_sp500", "r_ndx", "dln_fx", "dln_value_ratio"]


def _read_at_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Value of ``series`` on the last date at or before each of ``dates``, and that date."""
    s = series.dropna()
    pos = s.index.searchsorted(dates, side="right") - 1
    valid = (pos >= 0) & ~pd.isna(dates)
    values = np.full(len(dates), np.nan)
    used = pd.DatetimeIndex([pd.NaT] * len(dates))
    values[valid] = s.to_numpy()[pos[valid]]
    used = used.where(~valid, s.index[np.where(valid, pos, 0)])
    return values, used


def build_panel(premium: pd.DataFrame, market: pd.DataFrame, adr_close: pd.Series,
                freq: str = "W-FRI") -> pd.DataFrame:
    """One row per period, sampled on the last Taipei date of the period.

    ``premium`` is the output of premium_prev_ny or premium_same_day; ``market`` is
    download_market(); ``adr_close`` gives TSM closes on NY dates, for traded values.
    Empty periods are kept as NaN rows, so no change is computed across them.
    Traded values are summed between consecutive sampled rows: ADR value in USD over
    NY dates up to ``adr_date``, local value in USD over Taipei dates up to the Taipei
    date, converted with the premium's own rate.
    """
    d = premium.copy()
    if "adr_date" not in d:
        d["adr_date"] = d.index  # same_day: the ADR date is the Taipei date
    d["adr_date"] = pd.DatetimeIndex(d["adr_date"])

    adr_value = (market["adr_volume"] * adr_close.reindex(market.index)).fillna(0).cumsum()
    local_value = (market["local_volume"].reindex(d.index) * d["local"] / d["fx"]).fillna(0).cumsum()

    periods = d.index.to_period(freq)
    rows = d.assign(date=d.index, local_value_cum=local_value).groupby(periods).tail(1)
    rows.index = rows.index.to_period(freq)
    grid = pd.period_range(rows.index.min(), rows.index.max(), freq=freq)
    rows = rows.reindex(grid)

    panel = pd.DataFrame(index=grid.to_timestamp(how="end").normalize())
    panel.index.name = "period_end"
    panel["date"] = rows["date"].to_numpy()
    panel["adr_date"] = pd.DatetimeIndex(rows["adr_date"])
    adr_dates = pd.DatetimeIndex(panel["adr_date"])
    for name, column in US_LEVELS.items():
        values, used = _read_at_or_before(market[column], adr_dates)
        panel[f"ln_{name}"] = np.log(values)
        panel[f"{name}_date"] = used
    values, used = _read_at_or_before(adr_value, adr_dates)
    panel["adr_value_cum"], panel["adr_value_date"] = values, used
    panel["local_value_cum"] = rows["local_value_cum"].to_numpy()
    for col in ("adr", "local", "fx"):
        panel[f"ln_{col}"] = np.log(rows[col].to_numpy())
    panel["q"] = np.log1p(rows["premium"].to_numpy())

    panel["dq"] = panel["q"].diff()
    panel["r_adr"] = panel["ln_adr"].diff()
    panel["r_local"] = panel["ln_local"].diff()
    panel["dln_fx"] = panel["ln_fx"].diff()
    panel["r_local_usd"] = panel["r_local"] - panel["dln_fx"]
    panel["r_sp500"] = panel["ln_sp500"].diff()
    panel["r_ndx"] = panel["ln_ndx"].diff()
    ratio = panel["adr_value_cum"].diff() / panel["local_value_cum"].diff()
    panel["ln_value_ratio"] = np.log(ratio.where(ratio > 0))
    panel["dln_value_ratio"] = panel["ln_value_ratio"].diff()
    return panel


def identity_error(panel: pd.DataFrame) -> float:
    """Largest absolute gap between dq and r_ADR - r_local + d ln FX."""
    return float((panel["dq"] - (panel["r_adr"] - panel["r_local"] + panel["dln_fx"])).abs().max())


def check_alignment(panel: pd.DataFrame, market: pd.DataFrame) -> None:
    """Raise if a US factor is read after the ADR date, or not on the date recorded."""
    problems = []
    adr_date = pd.DatetimeIndex(panel["adr_date"])
    for name, column in US_LEVELS.items():
        used = pd.DatetimeIndex(panel[f"{name}_date"])
        later = (used > adr_date) & ~pd.isna(used)
        if later.any():
            problems.append(f"{name}: {int(later.sum())} rows read after the ADR date")
        known = ~pd.isna(used)
        expected = np.log(market[column].reindex(used[known]).to_numpy())
        actual = panel.loc[known, f"ln_{name}"].to_numpy()
        if not np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True):
            problems.append(f"{name}: values do not match the recorded dates")
    later = (pd.DatetimeIndex(panel["adr_value_date"]) > adr_date) & ~pd.isna(panel["adr_value_date"])
    if later.any():
        problems.append(f"adr traded value: {int(later.sum())} rows read after the ADR date")
    if problems:
        raise ValueError("; ".join(problems))


def newey_west_lags(n: int) -> int:
    """Newey and West (1994) rule of thumb, floor(4 (n / 100) ^ (2 / 9)).

    It gives 6 lags for the about 870 weekly observations of the full sample, 4 for a
    3-year rolling window and 4 for about 200 monthly observations. At least one lag is
    always used: asynchrony noise makes premium changes autocorrelated at lag 1.
    """
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def ols_newey_west(y: pd.Series, X: pd.DataFrame, lags: int | None = None) -> dict:
    """OLS of y on X with a constant and Newey-West (Bartlett) standard errors.

    Rows with a missing value are dropped. Returns a coefficient table (coef, se,
    ci_low, ci_high, pvalue, and the classical se for comparison), R2, the number of
    observations and the number of lags. Inference uses the normal distribution.
    """
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    n = len(data)
    lags = newey_west_lags(n) if lags is None else lags
    model = sm.OLS(data["y"], sm.add_constant(data[list(X.columns)]))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = fit.conf_int(0.05)
    table = pd.DataFrame({
        "coef": fit.params, "se": fit.bse, "ci_low": ci[0], "ci_high": ci[1],
        "pvalue": fit.pvalues, "se_classic": model.fit().bse,
    })
    return {"table": table, "r2": fit.rsquared, "n": n, "lags": lags}


def coefficient_test(result: dict, term: str, value: float) -> dict:
    """Two-sided test of coefficient == value, with the Newey-West standard error."""
    row = result["table"].loc[term]
    z = (row["coef"] - value) / row["se"]
    return {"term": term, "hypothesis": f"{term} = {value:g}", "stat": z, "pvalue": 2 * norm.sf(abs(z))}
