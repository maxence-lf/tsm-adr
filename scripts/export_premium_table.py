"""Export one table: one row per Taipei trading day, premium under both alignments.

Usage: python scripts/export_premium_table.py
Output: data/premium_table.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adr_premium.clean import prepare_fred_fx, print_fill_report
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import premium_prev_ny, premium_same_day

OUTLIER_JUMP = 10.0  # percentage points of daily change flagged as suspicious


def main() -> None:
    raw = download()
    fx, filled = prepare_fred_fx(download_fred_fx(), raw.index[0])
    same = premium_same_day(raw["adr"], raw["local"], fx)
    prev = premium_prev_ny(raw["adr"], raw["local"], fx)
    print_fill_report(filled, same, prev)

    table = pd.DataFrame({
        "premium_same_day_pct": same["premium"] * 100,   # blue line
        "premium_prev_ny_pct": prev["premium"] * 100,    # orange line
        "adr_date_used_prev_ny": pd.to_datetime(prev["adr_date"]).dt.date,
        "adr_same_day": same["adr"],
        "adr_prev_ny": prev["adr"],
        "local": prev["local"],
        "fx_same_day": same["fx"],
        "fx_prev_ny": prev["fx"],
    })
    table.index.name = "date"
    table["gap_pct_points"] = table["premium_same_day_pct"] - table["premium_prev_ny_pct"]

    out = ROOT / "data" / "premium_table.csv"
    table.round(4).to_csv(out)
    print(f"Table: {out} ({len(table)} rows)")

    # Suspicious points: daily jump above threshold in either definition
    jumps = table[["premium_same_day_pct", "premium_prev_ny_pct"]].diff().abs()
    suspicious = table[(jumps > OUTLIER_JUMP).any(axis=1)]
    print(f"\nDays with a daily jump > {OUTLIER_JUMP} pts:")
    print(suspicious.round(2).to_string() if len(suspicious) else "none")


if __name__ == "__main__":
    main()