"""Is the rank correlation between level shift and half-life mechanical?

On rolling 3-year windows, the weekly ARMA(1,1) half-life of prev_ny is strongly
rank-correlated with the level shift inside the window (difference between the mean
of the last and of the first year). Two explanations are possible:
(a) genuine level shifts in a window inflate phi;
(b) a mechanical effect: a slowly reverting process wanders away from its mean for
    longer, so first-year and last-year means differ more even without any regime
    change. Selecting "stable" windows then selects fast reversions.

This script simulates an AR(1) signal plus white noise WITHOUT regime change and runs
exactly the same rolling analysis as scripts/mean_reversion.py. Calibration on the
real prev_ny series: the real Taipei trading dates (same length and empty weeks), a
weekly half-life of 32 or 70 trading days, noise set analytically so that daily
changes have the real lag-1 autocorrelation, and a scale set so that daily changes
have the real standard deviation (both measured on prev_ny, which reflect noise and
innovations rather than regimes).

Usage: python scripts/simulate_level_shift.py
Output: data/level_shift_simulation.csv (one row per replication)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adr_premium.stats import arma11_half_life, level_shift, to_weekly

HALF_LIVES = [32, 70]  # trading days: the stable-window and overall rolling medians
REPLICATIONS = 50
WINDOW_WEEKS = 156  # as in mean_reversion.py
PHI_MAX = 0.999  # as in mean_reversion.py
SMALL_SHIFT = 2.0  # percentage points, as in the report
DAYS_PER_WEEK = 5
OUT = ROOT / "data" / "level_shift_simulation.csv"


def real_premium() -> pd.Series:
    """prev_ny premium in percentage points."""
    return pd.read_csv(ROOT / "data" / "premium_prev_ny.csv", index_col=0, parse_dates=True)["premium"] * 100


def calibration(half_life: float, change_std: float, change_autocorr: float) -> dict:
    """Daily phi, innovation sd and noise sd for an AR(1) plus white noise.

    For observed = s + e, with s an AR(1) (innovation variance su2) and e white
    (variance se2), the lag-1 autocorrelation of daily changes is
    rho = -(su2 (1 - phi) / (1 + phi) + se2) / (2 su2 / (1 + phi) + 2 se2),
    which gives se2 / su2 = (-rho (2 / (1 + phi)) - (1 - phi) / (1 + phi)) / (1 + 2 rho).
    The scale is then set from the variance of daily changes, 2 su2 / (1 + phi) + 2 se2.
    """
    phi = 0.5 ** (1 / half_life)
    rho = change_autocorr
    ratio = (-rho * 2 / (1 + phi) - (1 - phi) / (1 + phi)) / (1 + 2 * rho)
    su2 = change_std**2 / (2 / (1 + phi) + 2 * ratio)
    return {"phi": phi, "innovation_sd": np.sqrt(su2), "noise_sd": np.sqrt(ratio * su2)}


def simulate(index: pd.DatetimeIndex, cal: dict, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    n, phi = len(index), cal["phi"]
    u = rng.normal(0, cal["innovation_sd"], n)
    s = np.empty(n)
    s[0] = u[0] / np.sqrt(1 - phi**2)  # stationary start: no regime, no trend
    for t in range(1, n):
        s[t] = phi * s[t - 1] + u[t]
    return pd.Series(s + rng.normal(0, cal["noise_sd"], n), index=index)


def rolling_analysis(series: pd.Series) -> dict:
    """Same rolling analysis as mean_reversion.py and the report, on one series."""
    weekly = to_weekly(series)
    month_ends = series.groupby(series.index.to_period("M")).tail(1).index
    rows = []
    for end in month_ends:
        x = weekly.loc[:end].iloc[-WINDOW_WEEKS:]
        if len(x) < WINDOW_WEEKS:
            continue
        out = arma11_half_life(x)
        if not out["converged"] or out["phi"] > PHI_MAX:
            continue
        rows.append((out["half_life"] * DAYS_PER_WEEK, level_shift(weekly, x.first_valid_index(), end)))
    hl, shift = np.array(rows).T
    stable = np.abs(shift) < SMALL_SHIFT
    return {
        "windows_kept": len(hl),
        "median_half_life": np.median(hl),
        "rank_correlation": spearmanr(np.abs(shift), hl)[0],
        "stable_windows": int(stable.sum()),
        "median_half_life_stable": np.median(hl[stable]) if stable.any() else np.nan,
        "median_abs_shift": np.median(np.abs(shift)),
    }


def one_replication(args) -> dict:
    half_life, seed, index, cal = args
    series = simulate(index, cal, seed)
    return {"true_half_life": half_life, "seed": seed,
            "daily_change_autocorr": series.diff().autocorr(), "daily_change_std": series.diff().std(),
            **rolling_analysis(series)}


def main() -> None:
    real = real_premium()
    changes = real.diff().dropna()
    tasks = []
    for half_life in HALF_LIVES:
        cal = calibration(half_life, changes.std(), changes.autocorr())
        print(f"half-life {half_life}: phi {cal['phi']:.5f}, innovation sd {cal['innovation_sd']:.3f}, "
              f"noise sd {cal['noise_sd']:.3f} (points)")
        tasks += [(half_life, 1000 * half_life + rep, real.index, cal) for rep in range(REPLICATIONS)]
    with Pool() as pool:
        results = pd.DataFrame(pool.map(one_replication, tasks))
    results.to_csv(OUT, index=False)
    summary = results.groupby("true_half_life")[["daily_change_autocorr", "median_half_life", "rank_correlation",
                                                  "stable_windows", "median_half_life_stable"]].median()
    print(summary.round(3).to_string())
    print(f"Results: {OUT} ({len(results)} replications)")


if __name__ == "__main__":
    main()
