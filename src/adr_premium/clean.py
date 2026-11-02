"""Cleaning of the USD/TWD series, applied when the data is read.

The raw cache (data/raw_closes.csv) is never modified. Two rules are used:
- removal: values outside [FX_MIN, FX_MAX] are impossible levels and are dropped;
- flagging only: values far from a centred rolling median are listed for manual
  review and kept. A rule on the daily change is avoided on purpose: a single bad
  point would also trigger on the following day.
"""
from __future__ import annotations

import pandas as pd

FX_MIN = 25.0
FX_MAX = 40.0
FLAG_WINDOW = 5  # trading days, centred
FLAG_THRESHOLD = 0.02  # relative deviation from the rolling median


def clean_fx(fx: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Return (cleaned series, removed points with columns date, value, reason).

    Missing values are not errors (markets have different holidays): they are
    left as they are and never reported.
    """
    # Explicit comparisons rather than between(): NaN must not count as out of range.
    out_of_range = (fx < FX_MIN) | (fx > FX_MAX)
    removed = pd.DataFrame({
        "date": fx.index[out_of_range],
        "value": fx[out_of_range].to_numpy(),
        "reason": f"outside [{FX_MIN:g}, {FX_MAX:g}]",
    })
    return fx[~out_of_range], removed


def flag_fx(fx: pd.Series) -> pd.DataFrame:
    """List the points far from the rolling median (date, value, median, deviation)."""
    s = fx.dropna()
    median = s.rolling(FLAG_WINDOW, center=True, min_periods=1).median()
    deviation = s / median - 1
    flagged = deviation.abs() > FLAG_THRESHOLD
    return pd.DataFrame({
        "date": s.index[flagged],
        "value": s[flagged].to_numpy(),
        "median": median[flagged].to_numpy(),
        "deviation": deviation[flagged].to_numpy(),
    })


def print_fx_report(removed: pd.DataFrame, flagged: pd.DataFrame) -> None:
    """Print the removed points and the points flagged for manual review."""
    print(f"\nFX points removed: {len(removed)}")
    if len(removed):
        print(removed.to_string(index=False))
    print(f"\nFX points flagged for manual review: {len(flagged)}")
    if len(flagged):
        print(flagged.to_string(index=False))
