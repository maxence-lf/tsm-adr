"""Étape 1 : télécharge les données, construit la prime (2 définitions), sauvegarde et trace.

Usage : python scripts/build_premium.py [--refresh] [--start 2010-01-01]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.data import download
from adr_premium.premium import asynchrony_diagnostics, premium_prev_ny, premium_same_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    raw = download(start=args.start, refresh=args.refresh)
    same = premium_same_day(raw["adr"], raw["local"], raw["fx"])
    prev = premium_prev_ny(raw["adr"], raw["local"], raw["fx"])

    same.to_csv(ROOT / "data" / "premium_same_day.csv")
    prev.to_csv(ROOT / "data" / "premium_prev_ny.csv")

    diag = asynchrony_diagnostics(same, prev)
    print("\nDiagnostic d'asynchronie\n", diag.round(4).to_string())
    print(f"\nDernière prime (prev_ny) : {prev['premium'].iloc[-1]:.2%} au {prev.index[-1].date()}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(same.index, same["premium"] * 100, lw=0.8, label="Clôtures même date (NY d vs TW d)")
    ax.plot(prev.index, prev["premium"] * 100, lw=0.8, alpha=0.8, label="NY d-1 vs TW d")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Prime ADR (%)")
    ax.set_title("Prime de l'ADR TSM sur 2330.TW (1 ADR = 5 actions)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figures" / "premium_history.png"
    fig.savefig(out, dpi=150)
    print(f"Graphique : {out}")


if __name__ == "__main__":
    main()
