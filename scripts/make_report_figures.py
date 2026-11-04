"""Zoom figures for the interim report.

- report_fx_error_oct2011.png: the 2011-10-25 Yahoo error, a few weeks around it.
- report_asynchrony_example.png: a jump of the same-day premium that reverses the
  next day (April 2025), with the prices behind it.

Usage: python scripts/make_report_figures.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from adr_premium.data import download, download_fred_fx
from adr_premium.premium import ADR_RATIO, premium_prev_ny

FIGURES = ROOT / "figures"
BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#7f7f7f"
FX_ERROR_WINDOW = ("2011-10-03", "2011-11-15")
FX_ERROR_DATE = "2011-10-25"
EXAMPLE_WINDOW = ("2025-03-31", "2025-04-18")
EXAMPLE_JUMP = "2025-04-09"


def _style(ax, weekdays=(mdates.MO, mdates.TH)) -> None:
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=weekdays))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def fx_error_zoom() -> Path:
    raw = download()
    fred = download_fred_fx()
    start, end = FX_ERROR_WINDOW
    # "Before": raw Yahoo rate on the Taipei date, as in the first version of the pipeline.
    before = premium_prev_ny(raw["adr"], raw["local"], raw["fx"], fx_on_adr_date=False)["premium"]
    after = pd.read_csv(ROOT / "data" / "premium_prev_ny.csv", index_col=0, parse_dates=True)["premium"]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    top.plot(before.loc[start:end].index, before.loc[start:end] * 100, color=GREY, lw=1.2, marker="o", ms=3,
             label="Raw Yahoo TWD=X")
    top.plot(after.loc[start:end].index, after.loc[start:end] * 100, color=BLUE, lw=1.2, marker="o", ms=3,
             label="FRED DEXTAUS")
    top.set_ylabel("prev_ny premium (%)")
    top.set_title("The 2011-10-25 exchange-rate error in Yahoo TWD=X")
    _style(top, weekdays=mdates.MO)

    yahoo = raw["fx"].loc[start:end].dropna()
    bottom.plot(yahoo.index, yahoo, color=GREY, lw=1.2, marker="o", ms=3, label="Yahoo TWD=X")
    f = fred.loc[start:end].dropna()
    bottom.plot(f.index, f, color=BLUE, lw=1.2, marker="o", ms=3, label="FRED DEXTAUS")
    bottom.set_ylabel("TWD per USD")
    bottom.annotate(f"{raw.loc[FX_ERROR_DATE, 'fx']:.4f}", (pd.Timestamp(FX_ERROR_DATE), raw.loc[FX_ERROR_DATE, "fx"]),
                    textcoords="offset points", xytext=(8, 0), fontsize=8, va="center")
    _style(bottom, weekdays=mdates.MO)
    fig.tight_layout()
    out = FIGURES / "report_fx_error_oct2011.png"
    fig.savefig(out, dpi=150)
    return out


def asynchrony_zoom() -> Path:
    t = pd.read_csv(ROOT / "data" / "premium_table.csv", index_col=0, parse_dates=True)
    start, end = EXAMPLE_WINDOW
    w = t.loc[start:end]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    top.plot(w.index, w["premium_same_day_pct"], color=BLUE, lw=1.4, marker="o", ms=4,
             label="same_day: NY close d vs Taipei close d")
    top.plot(w.index, w["premium_prev_ny_pct"], color=ORANGE, lw=1.4, marker="o", ms=4,
             label="prev_ny: NY close d-1 vs Taipei close d")
    top.axvline(pd.Timestamp(EXAMPLE_JUMP), color="black", lw=0.6, ls=":")
    top.set_ylabel("Premium (%)")
    top.set_title("A same-day jump that reverses the next day")
    _style(top)

    local_usd = ADR_RATIO * w["local"] / w["fx_same_day"]
    bottom.plot(w.index, w["adr_same_day"], color=BLUE, lw=1.4, marker="o", ms=4, label="TSM close on d (USD)")
    bottom.plot(w.index, local_usd, color=GREY, lw=1.4, marker="s", ms=4,
                label=f"{ADR_RATIO} x 2330.TW close on d / FRED rate on d (USD)")
    bottom.axvline(pd.Timestamp(EXAMPLE_JUMP), color="black", lw=0.6, ls=":")
    bottom.set_ylabel("USD per ADR")
    _style(bottom)
    fig.tight_layout()
    out = FIGURES / "report_asynchrony_example.png"
    fig.savefig(out, dpi=150)
    return out


def main() -> None:
    for out in (fx_error_zoom(), asynchrony_zoom()):
        print(f"Chart: {out}")


if __name__ == "__main__":
    main()
