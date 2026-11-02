"""Construction of the ADR / local share premium.

Convention: premium = ADR price (USD) / (ratio * local price (TWD) / USDTWD) - 1

Central issue: Taipei closes at 13:30 Taipei time (05:30 UTC), New York at 16:00
New York time (20:00 or 21:00 UTC depending on daylight saving time).
The two closes of the same calendar date are about 15 hours apart: the New York
close includes information that Taipei has not seen yet.

Two definitions are therefore computed:
- ``same_day``   : NY close of day d vs Taipei close of day d (gap ~15 h,
                   NY "ahead" of Taipei)
- ``prev_ny``    : NY close of the last NY trading day strictly before d
                   vs Taipei close of day d (gap ~8.5 h, Taipei "ahead")

Neither is synchronous. Comparing their properties (volatility of daily changes,
autocorrelation) measures the noise caused by asynchrony.

FX alignment: the rate is taken on the NY date of the ADR used, so it is never
later than the ADR price. same_day uses the FRED rate of day d, prev_ny the FRED
rate of ``adr_date`` (the previous NY session). FRED (DEXTAUS) is recorded at noon
New York time, 4 hours before the ADR close.
"""
from __future__ import annotations

import pandas as pd

ADR_RATIO = 5  # 1 TSM ADR = 5 ordinary 2330.TW shares


def _as_series(s: pd.Series, name: str) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = name
    return s


def premium_same_day(adr_usd: pd.Series, local_twd: pd.Series, usdtwd: pd.Series,
                     ratio: int = ADR_RATIO) -> pd.DataFrame:
    """Premium from the closes of the same calendar date (common days only)."""
    df = pd.concat(
        [_as_series(adr_usd, "adr"), _as_series(local_twd, "local"), _as_series(usdtwd, "fx")],
        axis=1, join="inner",
    )
    df["local_in_usd"] = ratio * df["local"] / df["fx"]
    df["premium"] = df["adr"] / df["local_in_usd"] - 1
    return df


def premium_prev_ny(adr_usd: pd.Series, local_twd: pd.Series, usdtwd: pd.Series,
                    ratio: int = ADR_RATIO, fx_on_adr_date: bool = True) -> pd.DataFrame:
    """Premium with, for each Taipei date d, the last NY close strictly before d.

    The ADR used is indexed by the Taipei date. The NY date actually used is kept
    in the ``adr_date`` column for audit.

    The FX rate is read on ``adr_date``. ``fx_on_adr_date=False`` reads it on the
    Taipei date instead, which reproduces the earlier Yahoo-based version.
    """
    local = _as_series(local_twd, "local").to_frame()
    fx = _as_series(usdtwd, "fx")
    adr = _as_series(adr_usd, "adr").to_frame()
    adr["adr_date"] = adr.index

    # Shifting by one day guarantees "strictly before" in the merge_asof.
    adr_shifted = adr.copy()
    adr_shifted.index = adr_shifted.index + pd.Timedelta(days=1)

    df = pd.merge_asof(
        local.reset_index(names="date"),
        adr_shifted.reset_index(names="date"),
        on="date", direction="backward",
    ).set_index("date")
    fx_dates = pd.DatetimeIndex(df["adr_date"]) if fx_on_adr_date else df.index
    df["fx"] = fx.reindex(fx_dates).to_numpy()
    df = df.dropna(subset=["adr", "local", "fx"])

    # Maximum gap allowed between the Taipei date and the NY date used (weekends, holidays).
    gap = (df.index - pd.DatetimeIndex(df["adr_date"])).days
    df["ny_gap_days"] = gap
    df = df[df["ny_gap_days"] <= 5]

    df["local_in_usd"] = ratio * df["local"] / df["fx"]
    df["premium"] = df["adr"] / df["local_in_usd"] - 1
    return df


def asynchrony_diagnostics(same_day: pd.DataFrame, prev_ny: pd.DataFrame) -> pd.DataFrame:
    """Compare the two definitions: level, noise of the changes, autocorrelation."""
    rows = {}
    for name, df in {"same_day": same_day, "prev_ny": prev_ny}.items():
        p = df["premium"]
        dp = p.diff().dropna()
        rows[name] = {
            "n_obs": len(p),
            "premium_mean": p.mean(),
            "premium_std": p.std(),
            "daily_change_std": dp.std(),
            # Negative autocorrelation of the changes is a sign of measurement noise
            # (a gap caused by asynchrony reverses the next day).
            "daily_change_autocorr_lag1": dp.autocorr(lag=1),
        }
    return pd.DataFrame(rows).T
