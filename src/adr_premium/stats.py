"""Stationarity and mean-reversion statistics for the premium.

Every function takes a series indexed by date. The unit of a half-life is the
spacing of the series: trading days for daily data, weeks for weekly data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

AR_PARAMS = 2  # constant and slope of the AR(1) regression


def adf_kpss(series: pd.Series) -> dict:
    """ADF (lags chosen by AIC, H0: unit root) and KPSS (H0: level stationary)."""
    s = series.dropna()
    adf_stat, adf_p, adf_lags, *_ = adfuller(s, regression="c", autolag="AIC", result_object=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="c", nlags="auto")
    return {
        "n": len(s),
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_lags": int(adf_lags),
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "kpss_lags": int(kpss_lags),
        # KPSS p-values come from a table limited to [0.01, 0.10]: beyond it they are bounds.
        "kpss_pvalue_is_bound": any(issubclass(w.category, InterpolationWarning) for w in caught),
    }


def _ar1_pairs(series: pd.Series) -> pd.DataFrame:
    """(dp_t, p_{t-1}) pairs. A NaN, such as an empty week, breaks the pair around it."""
    lag = series.shift(1)
    return pd.concat({"dp": series - lag, "lag": lag}, axis=1).dropna()


def _ols(pairs: pd.DataFrame):
    return sm.OLS(pairs["dp"], sm.add_constant(pairs["lag"])).fit()


def _half_life(phi: float) -> float:
    if phi >= 1:
        return np.inf
    if phi <= 0:
        return 0.0  # the gap is gone (or overshoots) within one period
    return np.log(0.5) / np.log(phi)


def ar1_half_life(series: pd.Series) -> dict:
    """Fit dp_t = a + b p_{t-1} by OLS; phi = 1 + b. The 95% CI comes from the CI of b."""
    pairs = _ar1_pairs(series)
    fit = _ols(pairs)
    b = fit.params["lag"]
    b_low, b_high = fit.conf_int(0.05).loc["lag"]
    return {
        "n": int(fit.nobs),
        "phi": 1 + b,
        "half_life": _half_life(1 + b),
        "half_life_ci_low": _half_life(1 + b_low),
        "half_life_ci_high": _half_life(1 + b_high),
        "r2": fit.rsquared,
    }


def to_weekly(series: pd.Series) -> pd.Series:
    """Last observation of each week, dated on its Friday. Empty weeks stay NaN."""
    return series.resample("W-FRI").last()


def chow_test(series: pd.Series, break_date) -> dict:
    """Chow F test of a change in the AR(1) parameters at ``break_date``.

    Pairs dated before ``break_date`` form the first regime, the others the second.
    The test assumes homoskedastic, serially uncorrelated residuals.
    """
    pairs = _ar1_pairs(series)
    before = pairs[pairs.index < pd.Timestamp(break_date)]
    after = pairs[pairs.index >= pd.Timestamp(break_date)]
    out = {"n_before": len(before), "n_after": len(after), "chow_f": np.nan, "chow_pvalue": np.nan}
    if min(len(before), len(after)) <= AR_PARAMS:
        return out
    rss_split = _ols(before).ssr + _ols(after).ssr
    dof = len(pairs) - 2 * AR_PARAMS
    f = ((_ols(pairs).ssr - rss_split) / AR_PARAMS) / (rss_split / dof)
    out["chow_f"] = f
    out["chow_pvalue"] = sps.f.sf(f, AR_PARAMS, dof)
    return out
