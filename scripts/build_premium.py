"""Step 1: download the data, build the premium (2 definitions), save and plot.

Usage: python scripts/build_premium.py [--refresh] [--start 2010-01-01] [--no-clean]
                                        [--fx-source {fred,yahoo}]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.clean import clean_fx, flag_fx, prepare_fred_fx, print_fill_report, print_fx_report
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import asynchrony_diagnostics, premium_prev_ny, premium_same_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--no-clean", action="store_true",
                    help="chart the raw Yahoo FX series, to reproduce the 'before' figure "
                         "(implies --fx-source yahoo)")
    ap.add_argument("--fx-source", choices=["fred", "yahoo"], default="fred",
                    help="yahoo reproduces the earlier version, with the rate on the Taipei date")
    args = ap.parse_args()

    use_fred = args.fx_source == "fred" and not args.no_clean
    raw = download(start=args.start, refresh=args.refresh)
    if use_fred:
        fx, filled = prepare_fred_fx(download_fred_fx(refresh=args.refresh), raw.index[0])
    elif args.no_clean:
        fx = raw["fx"]
        print("FX cleaning disabled")
    else:
        fx, removed = clean_fx(raw["fx"])
        print_fx_report(removed, flag_fx(fx))
    same = premium_same_day(raw["adr"], raw["local"], fx)
    prev = premium_prev_ny(raw["adr"], raw["local"], fx, fx_on_adr_date=use_fred)
    if use_fred:
        print_fill_report(filled, same, prev)

    # The Yahoo runs only reproduce earlier results: the premium CSVs always hold the FRED version.
    if use_fred:
        same.to_csv(ROOT / "data" / "premium_same_day.csv")
        prev.to_csv(ROOT / "data" / "premium_prev_ny.csv")

    diag = asynchrony_diagnostics(same, prev)
    print("\nAsynchrony diagnostics\n", diag.round(4).to_string())
    print(f"\nLatest premium (prev_ny): {prev['premium'].iloc[-1]:.2%} on {prev.index[-1].date()}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(same.index, same["premium"] * 100, lw=0.8, label="Same-date closes (NY d vs TW d)")
    ax.plot(prev.index, prev["premium"] * 100, lw=0.8, alpha=0.8, label="NY d-1 vs TW d")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ADR premium (%)")
    ax.set_title("TSM ADR premium over 2330.TW (1 ADR = 5 shares)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if use_fred:
        name = "premium_history.png"
    else:
        name = "premium_raw_with_fx_errors.png" if args.no_clean else "premium_history_yahoo.png"
    out = ROOT / "figures" / name
    fig.savefig(out, dpi=150)
    print(f"Chart: {out}")


if __name__ == "__main__":
    main()
