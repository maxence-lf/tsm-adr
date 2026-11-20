"""No-arbitrage band of the premium for each security (roadmap step 7).

A derived calculation, not a test (see adr_premium.arbitrage). For each security, the
half-life h is the median weekly ARMA(1,1) half-life over rolling 3-year windows, read
from data/step6_results.csv (run scripts/step6_out_of_sample.py first). D* is computed
for T = h and T = 2h with the default and institutional costs of step 5, and compared
with the weekly deviations of q from its mean over the 26 previous weeks.

The deviations come from the prev_ny premium, which carries asynchrony noise (step 5:
about 1.5 points of the signalled deviation is already gone when the ADR trades), so
the shares of weeks above D* overstate the exploitable deviations.

Usage: python scripts/arbitrage_band.py
Output: data/arbitrage_band.csv, figures/arbitrage_band.png
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.arbitrage import breakeven_deviation, deviations, round_trip_cost, share_above
from adr_premium.backtest import Costs
from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import premium_prev_ny
from adr_premium.securities import STEP6, TSMC

COSTS = {
    "default": Costs(),
    "institutional": Costs(adr_commission=2e-4, local_commission=3e-4, fx_conversion=2e-4,
                           adr_borrow_fee=0.0025),
}


def half_lives() -> dict[str, float]:
    path = ROOT / "data" / "step6_results.csv"
    if not path.exists():
        raise SystemExit("data/step6_results.csv is missing: run scripts/step6_out_of_sample.py first")
    r = pd.read_csv(path)
    sel = r[(r["section"] == "mean_reversion") & (r["item"] == "rolling weekly arma11")
            & (r["metric"] == "median_half_life_days")]
    return {row.security: float(row.value) for row in sel.itertuples()}


def premium(sec) -> pd.Series:
    raw = sec.clip(download(security=sec))
    with contextlib.redirect_stdout(io.StringIO()):
        fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    return premium_prev_ny(raw["adr"], raw["local"], fx, ratio=sec.ratio)["premium"]


def main() -> None:
    hl = half_lives()
    rows, devs = [], {}
    for sec in [TSMC] + STEP6:
        dev = deviations(premium(sec))
        devs[sec.key] = dev
        h = hl[sec.key]
        for cname, c in COSTS.items():
            cost = round_trip_cost(c)
            for label, T in (("T = h", h), ("T = 2h", 2 * h)):
                d_star = breakeven_deviation(cost, c.adr_borrow_fee, h, T)
                rows.append({"security": sec.key, "half_life_days": h, "costs": cname,
                             "round_trip_cost": cost, "borrow_fee": c.adr_borrow_fee, "holding": label,
                             "holding_days": T, "d_star": d_star, **share_above(dev, d_star),
                             "weeks": len(dev), "dev_p90": float(dev.quantile(0.9)),
                             "dev_p99": float(dev.quantile(0.99)), "dev_std": float(dev.std())})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "data" / "arbitrage_band.csv", index=False)
    pd.set_option("display.width", 200)
    print("Derived calculation: D* = (C + b T / 252) / (1 - 0.5^(T / h)); deviations of weekly q from "
          "its 26-week trailing mean (prev_ny, includes asynchrony noise)")
    print(table[["security", "half_life_days", "costs", "holding", "round_trip_cost", "d_star", "share_above",
                 "share_abs_above", "dev_std", "dev_p90"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    names = {s.key: f"{s.name} ({s.adr})" for s in [TSMC] + STEP6}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, dev) in zip(axes.ravel(), devs.items()):
        ax.hist(dev * 100, bins=60, color="grey", alpha=0.7)
        t = table[table["security"] == key]
        for (cname, holding), style in {("default", "T = h"): ("C3", "-"), ("default", "T = 2h"): ("C3", "--"),
                                        ("institutional", "T = h"): ("C0", "-"),
                                        ("institutional", "T = 2h"): ("C0", "--")}.items():
            d = t[(t["costs"] == cname) & (t["holding"] == holding)]["d_star"].iloc[0]
            ax.axvline(d * 100, color=style[0], ls=style[1], lw=1.2, label=f"D*, {cname}, {holding}")
        ax.set_title(f"{names[key]}, h = {hl[key]:.0f} trading days")
        ax.set_xlabel("Weekly q minus 26-week trailing mean (points)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("No-arbitrage band (derived): deviations of the premium against the break-even D*")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "arbitrage_band.png", dpi=150)
    plt.close(fig)
    print("\nOutputs: data/arbitrage_band.csv, figures/arbitrage_band.png")


if __name__ == "__main__":
    main()
