"""Compare the USD/TWD closes from Yahoo (TWD=X) with the official FRED series (DEXTAUS).

The premium calculation is not affected: this script only reports differences.

Usage: python scripts/compare_fx.py [--refresh]
Output: figures/fx_yahoo_vs_fred.png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.clean import FLAG_THRESHOLD, FLAG_WINDOW, clean_fx, flag_fx
from adr_premium.data import download, download_fred_fx

GAP_THRESHOLD = 0.01  # Yahoo/FRED gap above which the two sources really disagree
CONFIRM_THRESHOLD = FLAG_THRESHOLD / 2  # FRED move, same direction, confirming a flagged point
MIN_RUN = 10  # consecutive common dates for a gap to count as a persistent level shift


def relative_gap(yahoo: pd.Series, fred: pd.Series) -> pd.Series:
    """Yahoo / FRED - 1 on the dates where both sources have a value."""
    both = pd.concat({"yahoo": yahoo, "fred": fred}, axis=1, join="inner").dropna()
    return both["yahoo"] / both["fred"] - 1


def gap_stats(gap: pd.Series, by: pd.Index) -> pd.DataFrame:
    """Gap statistics in percent for each group."""
    pct = gap * 100
    grouped = pct.groupby(by, sort=False)
    return pd.DataFrame({
        "n": grouped.size(),
        "mean": grouped.mean(),
        "median": grouped.median(),
        "std": grouped.std(),
        "mean_abs": pct.abs().groupby(by, sort=False).mean(),
        "share_abs_gt_1pct": (pct.abs() > GAP_THRESHOLD * 100).groupby(by, sort=False).mean() * 100,
    })


def date_alignment(yahoo: pd.Series, fred: pd.Series) -> pd.DataFrame:
    """Gap dispersion with Yahoo moved by one observation, to detect a date offset."""
    rows = {}
    for lag in (-1, 0, 1):
        gap = relative_gap(yahoo.dropna().shift(lag), fred)
        rows[lag] = {"n": len(gap), "std_pct": gap.std() * 100, "mean_abs_pct": gap.abs().mean() * 100}
    out = pd.DataFrame(rows).T
    out.index.name = "yahoo_lag"
    return out


def persistent_gaps(gap: pd.Series) -> pd.DataFrame:
    """Runs of at least MIN_RUN dates with a gap above GAP_THRESHOLD and the same sign."""
    sign = (gap > GAP_THRESHOLD).astype(int) - (gap < -GAP_THRESHOLD).astype(int)
    runs = (sign != sign.shift()).cumsum()
    rows = [
        {"start": g.index[0].date(), "end": g.index[-1].date(), "n": len(g), "mean_gap_pct": g.mean() * 100}
        for _, g in gap.groupby(runs)
        if len(g) >= MIN_RUN and sign[g.index[0]] != 0
    ]
    return pd.DataFrame(rows, columns=["start", "end", "n", "mean_gap_pct"])


def compare_flagged(flagged: pd.DataFrame, gap: pd.Series, fred: pd.Series) -> pd.DataFrame:
    """Add to each flagged point the Yahoo/FRED gap and FRED's own rolling deviation."""
    s = fred.dropna()
    median = s.rolling(FLAG_WINDOW, center=True, min_periods=1).median()
    out = flagged.rename(columns={"value": "yahoo", "median": "yahoo_median",
                                  "deviation": "yahoo_dev"}).set_index("date")
    out["fred"] = s.reindex(out.index)
    out["gap"] = gap.reindex(out.index)
    out["fred_dev"] = (s / median - 1).reindex(out.index)
    out["median_gap"] = out["yahoo_median"] / out["fred"] - 1

    confirmed = ((out["fred_dev"] * out["yahoo_dev"]) > 0) & (out["fred_dev"].abs() > CONFIRM_THRESHOLD)
    disagree = out["gap"].abs() > GAP_THRESHOLD
    neighbours_off = ~disagree & (out["median_gap"].abs() > GAP_THRESHOLD)
    out["verdict"] = "unclear"
    out.loc[disagree & ~confirmed, "verdict"] = "yahoo_point_off"
    out.loc[neighbours_off & ~confirmed, "verdict"] = "yahoo_neighbours_off"
    out.loc[confirmed, "verdict"] = "fred_confirms"
    out.loc[out["fred"].isna(), "verdict"] = "no_fred_value"
    return out


def plot_gap(gap: pd.Series, flagged: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(gap.index, gap * 100, lw=0.7, color="#1f77b4", label="All common dates")
    points = flagged.dropna(subset=["gap"])
    ax.scatter(points.index, points["gap"] * 100, s=14, color="#d95f02", edgecolor="white",
               linewidth=0.5, zorder=3, label="Dates flagged in Yahoo")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Yahoo / FRED - 1 (%)")
    ax.set_title("USD/TWD: Yahoo TWD=X vs FRED DEXTAUS")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download FRED again")
    args = ap.parse_args()

    pd.set_option("display.width", 160)
    raw = download()
    fred = download_fred_fx(refresh=args.refresh)
    print(f"FRED: {fred.index[0].date()} to {fred.index[-1].date()}, "
          f"{len(fred)} rows, {fred.isna().sum()} missing values")

    yahoo, removed = clean_fx(raw["fx"])
    flagged = flag_fx(yahoo)
    gap = relative_gap(yahoo, fred)
    print(f"Common dates: {len(gap)} ({gap.index[0].date()} to {gap.index[-1].date()})")

    # Lag -1 compares FRED on d with the next Yahoo observation.
    print("\nDate alignment, excluding the noisy 2011-2016 period:")
    print(date_alignment(yahoo.loc["2017":], fred.loc["2017":]).round(3).to_string())

    print("\nGap Yahoo/FRED by year (%):")
    print(gap_stats(gap, gap.index.year).sort_index().round(3).to_string())

    weekdays = gap.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print("\nGap Yahoo/FRED by weekday (%):")
    print(gap_stats(gap, weekdays).reindex(order).round(3).to_string())

    print("\nPoints removed by clean_fx, against FRED:")
    print(removed.assign(fred=fred.reindex(removed["date"]).to_numpy()).to_string(index=False))

    print(f"\nPersistent gaps (at least {MIN_RUN} dates beyond {GAP_THRESHOLD:.0%}):")
    print(persistent_gaps(gap).round(2).to_string(index=False))

    table = compare_flagged(flagged, gap, fred)
    large = gap[gap.abs() > GAP_THRESHOLD]
    print(f"\nDates with |gap| > {GAP_THRESHOLD:.0%}: {len(large)}, "
          f"of which flagged: {large.index.isin(table.index).sum()}")
    print(f"\nThe {len(table)} flagged dates (deviations and gap in %):")
    shown = table.copy()
    shown[["yahoo_dev", "gap", "fred_dev"]] *= 100
    print(shown[["yahoo", "yahoo_median", "yahoo_dev", "fred", "gap", "fred_dev", "verdict"]].round(3).to_string())

    print("\nVerdicts:")
    print(table["verdict"].value_counts().to_string())
    print("\nVerdicts by year:")
    print(pd.crosstab(table.index.year, table["verdict"]).to_string())

    other = gap[~gap.index.isin(table.index)]
    compared = table.dropna(subset=["gap"])
    print(f"\nMean |gap|: flagged {compared['gap'].abs().mean():.2%}, other dates {other.abs().mean():.2%}")
    print(f"Mean |FRED deviation| on flagged dates: {compared['fred_dev'].abs().mean():.2%}")
    print(f"Mean |Yahoo deviation| on flagged dates: {compared['yahoo_dev'].abs().mean():.2%}")
    print(f"Points FRED would flag itself over the same period: {len(flag_fx(fred.loc[yahoo.index[0]:]))}")

    out = ROOT / "figures" / "fx_yahoo_vs_fred.png"
    plot_gap(gap, table, out)
    print(f"\nChart: {out}")


if __name__ == "__main__":
    main()
