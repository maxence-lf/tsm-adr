"""Explanatory factors of the premium (roadmap step 4).

Reference: prev_ny premium with the FRED rate, weekly data sampled on the last Taipei
date of each week (Friday). The dependent variable is dq, the weekly change of
q = ln(1 + premium). Factors: S&P 500 and Nasdaq-100 log returns read on the ADR's NY
date, the change in log FX (TWD per USD), and the change in the log ratio of ADR to
local traded value. All standard errors are Newey-West.

The contemporaneous regressions describe co-movement within the same week; they are
not forecasts and do not imply a trading opportunity. The predictive regressions,
with factors lagged by one period, are reported separately.

Usage: python scripts/factors.py
Output: data/factor_results.csv, figures/rolling_factor_betas.png
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

from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_fred_fx, download_market
from adr_premium.factors import (FACTORS, build_panel, check_alignment, coefficient_test, identity_error,
                                 newey_west_lags, ols_newey_west)
from adr_premium.premium import premium_prev_ny, premium_same_day

SUBPERIODS = [
    ("full", None, None),
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020-2023", "2020-01-01", "2023-12-31"),
    ("2024-Apr 2026", "2024-01-01", "2026-04-30"),
    ("May 2026-end", "2026-05-01", None),
]
ROLLING_WEEKS = 156  # 3 years, as in mean_reversion.py
ROLLING_TERMS = ["r_sp500", "dln_fx"]
MIN_OBS = 52  # below one year of weekly data, a sub-period result is flagged
LABELS = {"r_sp500": "S&P 500 return", "r_ndx": "Nasdaq-100 return", "dln_fx": "d ln FX (TWD per USD)",
          "dln_value_ratio": "d ln traded value ADR / local"}


def panels() -> dict[str, pd.DataFrame]:
    raw = download()
    market = download_market()
    with contextlib.redirect_stdout(io.StringIO()):
        fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    prev = premium_prev_ny(raw["adr"], raw["local"], fx)
    same = premium_same_day(raw["adr"], raw["local"], fx)
    out = {
        ("prev_ny", "weekly"): build_panel(prev, market, raw["adr"], "W-FRI"),
        ("same_day", "weekly"): build_panel(same, market, raw["adr"], "W-FRI"),
        ("prev_ny", "monthly"): build_panel(prev, market, raw["adr"], "M"),
    }
    for (definition, frequency), panel in out.items():
        error = identity_error(panel)
        if error > 1e-10:
            raise ValueError(f"dq decomposition fails for {definition} {frequency}: {error:.2e}")
        check_alignment(panel, market)
        print(f"{definition} {frequency}: {panel['dq'].notna().sum()} changes, decomposition error {error:.1e}, "
              f"alignment checked")
    return out


def rows_from(result: dict, **labels) -> list[dict]:
    return [{**labels, "term": term, **row.to_dict(), "r2": result["r2"], "n": result["n"],
             "nw_lags": result["lags"]} for term, row in result["table"].iterrows()]


def regressions(panel: pd.DataFrame, definition: str, frequency: str, timing: str) -> list[dict]:
    """Univariate and multivariate regressions of dq, on the full sample and sub-periods."""
    shift = 1 if timing == "predictive" else 0
    rows = []
    for sample, start, end in SUBPERIODS:
        p = panel.loc[start:end]
        X = p[FACTORS].shift(shift)
        models = [(f"univariate {f}", [f]) for f in FACTORS] + [("multivariate", FACTORS)]
        for model, terms in models:
            result = ols_newey_west(p["dq"], X[terms])
            labels = dict(analysis="regression", definition=definition, frequency=frequency, timing=timing,
                          sample=sample, model=model)
            rows += rows_from(result, **labels)
            if "dln_fx" in terms:
                for value in (1, 0):
                    test = coefficient_test(result, "dln_fx", value)
                    rows.append({**labels, "analysis": "fx_test", "term": "dln_fx", "hypothesis": test["hypothesis"],
                                 "stat": test["stat"], "pvalue": test["pvalue"], "n": result["n"],
                                 "nw_lags": result["lags"]})
    return rows


def decomposition(panel: pd.DataFrame) -> list[dict]:
    """beta(dq) = beta(r_ADR) - beta(r_local in USD) on the S&P 500, same observations."""
    rows = []
    for sample, start, end in SUBPERIODS:
        p = panel.loc[start:end, ["dq", "r_adr", "r_local_usd", "r_sp500"]].dropna()
        betas = {y: ols_newey_west(p[y], p[["r_sp500"]])["table"].loc["r_sp500"] for y in ("dq", "r_adr", "r_local_usd")}
        for y, row in betas.items():
            rows.append({"analysis": "decomposition", "definition": "prev_ny", "frequency": "weekly",
                         "timing": "contemporaneous", "sample": sample, "model": f"{y} on r_sp500", "term": "r_sp500",
                         **row.to_dict(), "n": len(p)})
        gap = betas["r_adr"]["coef"] - betas["r_local_usd"]["coef"] - betas["dq"]["coef"]
        rows.append({"analysis": "decomposition_check", "definition": "prev_ny", "frequency": "weekly",
                     "sample": sample, "model": "beta_adr - beta_local_usd - beta_dq", "term": "r_sp500",
                     "coef": gap, "n": len(p)})
    return rows


def rolling(panel: pd.DataFrame) -> pd.DataFrame:
    """Joint regression of dq on the S&P 500 return and d ln FX, 3-year windows, monthly step."""
    rows = []
    month_ends = panel.groupby(panel.index.to_period("M")).tail(1).index
    for end in month_ends:
        window = panel.loc[:end].iloc[-ROLLING_WEEKS:]
        if len(window) < ROLLING_WEEKS:
            continue
        result = ols_newey_west(window["dq"], window[ROLLING_TERMS])
        for term in ROLLING_TERMS:
            rows.append({"analysis": "rolling", "definition": "prev_ny", "frequency": "weekly",
                         "timing": "contemporaneous", "sample": str(end.date()), "model": "dq on r_sp500 + dln_fx",
                         "term": term, **result["table"].loc[term].to_dict(), "r2": result["r2"],
                         "n": result["n"], "nw_lags": result["lags"]})
    return pd.DataFrame(rows)


def plot_rolling(roll: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, term, color in zip(axes, ROLLING_TERMS, ("#1f77b4", "#d95f02")):
        g = roll[roll["term"] == term]
        dates = pd.to_datetime(g["sample"])
        ax.plot(dates, g["coef"], color=color, lw=1.4, label="Coefficient")
        ax.fill_between(dates, g["ci_low"], g["ci_high"], color=color, alpha=0.2, lw=0, label="95% CI (Newey-West)")
        ax.axhline(0, color="black", lw=0.6)
        if term == "dln_fx":
            ax.axhline(1, color="black", lw=0.6, ls="--", label="1: no price adjustment")
        ax.set_ylabel(f"Coefficient on {LABELS[term]}")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Weekly prev_ny: d ln(1 + premium) on S&P 500 return and d ln FX, rolling 3-year windows")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def summary(table: pd.DataFrame, corr: float) -> None:
    def show(sel, title):
        print(f"\n{title}")
        cols = ["sample", "model", "term", "coef", "ci_low", "ci_high", "pvalue", "r2", "n"]
        print(sel[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    reg = table[table["analysis"] == "regression"]
    base = reg[(reg["definition"] == "prev_ny") & (reg["frequency"] == "weekly") & (reg["term"] != "const")]
    contemp, pred = base[base["timing"] == "contemporaneous"], base[base["timing"] == "predictive"]
    print(f"\nCorrelation of S&P 500 and Nasdaq-100 weekly returns: {corr:.2f} (multivariate coefficients on the two "
          f"are collinear)")
    show(contemp[contemp["sample"] == "full"], "CONTEMPORANEOUS, prev_ny weekly, full sample")
    show(contemp[(contemp["sample"] != "full") & (contemp["model"] == "multivariate")],
         "CONTEMPORANEOUS, prev_ny weekly, multivariate by sub-period")
    small = contemp[(contemp["sample"] != "full") & (contemp["n"] < MIN_OBS)]["sample"].unique()
    if len(small):
        print(f"  Fewer than {MIN_OBS} observations, not conclusive: {', '.join(small)}")

    tests = table[(table["analysis"] == "fx_test") & (table["definition"] == "prev_ny") & (table["frequency"] == "weekly")
                  & (table["timing"] == "contemporaneous")]
    print("\nFX coefficient tests (Newey-West), prev_ny weekly contemporaneous")
    print(tests[["sample", "model", "hypothesis", "stat", "pvalue", "n"]].to_string(index=False,
                                                                                   float_format=lambda v: f"{v:.4f}"))

    dec = table[table["analysis"].isin(["decomposition", "decomposition_check"])]
    print("\nDecomposition on the S&P 500: beta(dq) = beta(r_ADR) - beta(r_local in USD)")
    print(dec[["sample", "model", "coef", "ci_low", "ci_high", "n"]].to_string(index=False,
                                                                              float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 100)
    print("PREDICTIVE (factors lagged one week), reported separately from the contemporaneous results")
    show(pred[pred["sample"] == "full"], "PREDICTIVE, prev_ny weekly, full sample")
    show(pred[(pred["sample"] != "full") & (pred["model"] == "multivariate")],
         "PREDICTIVE, prev_ny weekly, multivariate by sub-period")
    print("=" * 100)

    rob = reg[(reg["timing"] == "contemporaneous") & (reg["sample"] == "full") & (reg["term"] != "const")
              & ~((reg["definition"] == "prev_ny") & (reg["frequency"] == "weekly"))]
    print("\nROBUSTNESS, contemporaneous, full sample")
    print(rob[["definition", "frequency", "model", "term", "coef", "ci_low", "ci_high", "pvalue", "r2", "n"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    pd.set_option("display.width", 200)
    data = panels()
    weekly = data[("prev_ny", "weekly")]
    print(f"Newey-West lags: {newey_west_lags(int(weekly['dq'].notna().sum()))} on the full weekly sample")

    rows = []
    for (definition, frequency), panel in data.items():
        rows += regressions(panel, definition, frequency, "contemporaneous")
    rows += regressions(weekly, "prev_ny", "weekly", "predictive")
    rows += decomposition(weekly)
    roll = rolling(weekly)
    table = pd.concat([pd.DataFrame(rows), roll], ignore_index=True)
    out_csv = ROOT / "data" / "factor_results.csv"
    table.to_csv(out_csv, index=False)

    summary(table, weekly[["r_sp500", "r_ndx"]].corr().iloc[0, 1])
    out_fig = ROOT / "figures" / "rolling_factor_betas.png"
    plot_rolling(roll, out_fig)
    for term in ROLLING_TERMS:
        g = roll[roll["term"] == term]
        print(f"\nRolling {term}: {len(g)} windows, coefficient from {g['coef'].min():.3f} to {g['coef'].max():.3f}, "
              f"median {g['coef'].median():.3f}; CI excludes 0 in {(g['ci_low'] > 0).mean() + (g['ci_high'] < 0).mean():.0%}"
              + (f", excludes 1 in {((g['ci_low'] > 1) | (g['ci_high'] < 1)).mean():.0%}" if term == "dln_fx" else ""))
    print(f"\nResults: {out_csv} ({len(table)} rows)\nChart: {out_fig}")


if __name__ == "__main__":
    main()
