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
# The main multivariate model uses one US index, the S&P 500, chosen a priori as the
# broad market index, not on fit: picking the index with the higher R2 would select the
# model on the same data it is tested on. The S&P 500 and the Nasdaq-100 have a weekly
# correlation of about 0.9, so entering both makes their separate coefficients
# meaningless. The Nasdaq-100 version, with the same structure, is a robustness check;
# the model with both indices is kept only as an appendix, flagged as collinear.
# The traded-value ratio is endogenous (trading volumes react to the premium and to
# the same news), so it could bias the other coefficients: it is left out of the main
# model and added only in a separate robustness model.
MAIN = ["r_sp500", "dln_fx"]
NDX_VARIANT = ["r_ndx", "dln_fx"]
VALUE_VARIANT = ["r_sp500", "dln_fx", "dln_value_ratio"]
MODELS = [(f"univariate {f}", [f]) for f in FACTORS] + [
    ("multivariate", MAIN),
    ("robustness: Nasdaq-100 instead of S&P 500", NDX_VARIANT),
    ("robustness: with traded-value ratio", VALUE_VARIANT),
    ("appendix: both indices (collinear)", ["r_sp500", "r_ndx", "dln_fx"]),
]
FX_TEST_MODELS = ["univariate dln_fx", "multivariate", "robustness: Nasdaq-100 instead of S&P 500",
                  "robustness: with traded-value ratio"]
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
        for model, terms in MODELS:
            result = ols_newey_west(p["dq"], X[terms])
            labels = dict(analysis="regression", definition=definition, frequency=frequency, timing=timing,
                          sample=sample, model=model)
            rows += rows_from(result, **labels)
            if model in FX_TEST_MODELS:
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


def pair_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Weekly return of long the local share (in USD) and short the ADR, equal dollar weights.

    With simple returns the pair earns R_local_usd - R_adr per dollar of each leg; in
    log returns the same position earns -dq exactly, so its beta is -beta(dq).
    """
    out = pd.DataFrame(index=panel.index)
    out["pair_simple"] = np.expm1(panel["r_local_usd"]) - np.expm1(panel["r_adr"])
    out["pair_log"] = panel["r_local_usd"] - panel["r_adr"]
    out["r_sp500"] = panel["r_sp500"]
    return out


def pair_beta(panel: pd.DataFrame) -> list[dict]:
    """S&P 500 beta of the pair, full sample and rolling 3-year windows (input for step 5)."""
    pair = pair_returns(panel)
    labels = dict(analysis="pair_beta", definition="prev_ny", frequency="weekly", timing="contemporaneous",
                  term="r_sp500")
    rows = []
    for y in ("pair_simple", "pair_log"):
        result = ols_newey_west(pair[y], pair[["r_sp500"]])
        rows.append({**labels, "sample": "full", "model": f"{y} on r_sp500",
                     **result["table"].loc["r_sp500"].to_dict(), "r2": result["r2"], "n": result["n"],
                     "nw_lags": result["lags"]})
    month_ends = pair.groupby(pair.index.to_period("M")).tail(1).index
    for end in month_ends:
        window = pair.loc[:end].iloc[-ROLLING_WEEKS:]
        if len(window) < ROLLING_WEEKS:
            continue
        result = ols_newey_west(window["pair_simple"], window[["r_sp500"]])
        rows.append({**labels, "analysis": "pair_beta_rolling", "sample": str(end.date()),
                     "model": "pair_simple on r_sp500", **result["table"].loc["r_sp500"].to_dict(),
                     "r2": result["r2"], "n": result["n"], "nw_lags": result["lags"]})
    return rows


def conclusions(table: pd.DataFrame) -> pd.DataFrame:
    """Robust, partial and not conclusive findings, each with the numbers behind it."""
    def pick(**k):
        sel = table
        for col, val in k.items():
            sel = sel[sel[col] == val]
        return sel

    weekly_full = dict(definition="prev_ny", frequency="weekly", sample="full")
    dec = pick(analysis="decomposition", sample="full").set_index("model")
    pair_full = pick(analysis="pair_beta", sample="full", model="pair_simple on r_sp500").iloc[0]
    pair_roll = pick(analysis="pair_beta_rolling")
    uni = pick(analysis="regression", timing="contemporaneous", **weekly_full)
    sp, ndx = (uni[(uni["model"] == f"univariate {f}") & (uni["term"] == f)].iloc[0] for f in ("r_sp500", "r_ndx"))
    fx_tests = pick(analysis="fx_test", timing="contemporaneous", **weekly_full).set_index(["model", "hypothesis"])
    fx_main = uni[(uni["model"] == "multivariate") & (uni["term"] == "dln_fx")].iloc[0]
    fx_uni = uni[(uni["model"] == "univariate dln_fx") & (uni["term"] == "dln_fx")].iloc[0]
    monthly_fx = pick(analysis="regression", timing="contemporaneous", definition="prev_ny", frequency="monthly",
                      sample="full", model="univariate dln_fx", term="dln_fx").iloc[0]
    roll_fx = pick(analysis="rolling", term="dln_fx")
    vol = uni[(uni["model"] == "univariate dln_value_ratio") & (uni["term"] == "dln_value_ratio")].iloc[0]
    with_vol = uni[uni["model"] == "robustness: with traded-value ratio"].set_index("term")
    main = uni[uni["model"] == "multivariate"].set_index("term")
    pred = pick(analysis="regression", timing="predictive", **weekly_full)
    pred = pred[pred["term"] != "const"]
    reg = table[(table["analysis"] == "regression") & (table["term"] != "const")]

    rows = [
        ("robust", "The ADR has a higher beta to the US market than the local share in USD",
         f"S&P 500 beta {dec.loc['r_adr on r_sp500', 'coef']:.2f} for the ADR against "
         f"{dec.loc['r_local_usd on r_sp500', 'coef']:.2f} for the local share in USD; long local / short ADR "
         f"beta {pair_full['coef']:.2f} [{pair_full['ci_low']:.2f}, {pair_full['ci_high']:.2f}]; negative in "
         f"{(pair_roll['coef'] < 0).mean():.0%} of {len(pair_roll)} rolling windows, CI excluding 0 in "
         f"{(pair_roll['ci_high'] < 0).mean():.0%}"),
        ("partial", "FX changes are not passed to the premium one for one; the size of the pass-through is not established",
         f"FX coefficient = 1 rejected (p {fx_tests.loc[('univariate dln_fx', 'dln_fx = 1'), 'pvalue']:.1e} "
         f"univariate, {fx_tests.loc[('multivariate', 'dln_fx = 1'), 'pvalue']:.1e} multivariate); coefficient "
         f"{fx_uni['coef']:.2f} univariate, {fx_main['coef']:.2f} multivariate, {monthly_fx['coef']:.2f} monthly; "
         f"rolling CI excludes 1 in {((roll_fx['ci_low'] > 1) | (roll_fx['ci_high'] < 1)).mean():.0%} of windows "
         f"but 0 in only {((roll_fx['ci_low'] > 0) | (roll_fx['ci_high'] < 0)).mean():.0%}"),
        ("not conclusive", "Traded-value ratio (separate robustness model, endogenous)",
         f"univariate coefficient {vol['coef']:.4f}, p {vol['pvalue']:.2f}; added to the main model, coefficient "
         f"{with_vol.loc['dln_value_ratio', 'coef']:.4f}, p {with_vol.loc['dln_value_ratio', 'pvalue']:.2f}, with the "
         f"S&P 500 coefficient moving from {main.loc['r_sp500', 'coef']:.2f} to {with_vol.loc['r_sp500', 'coef']:.2f} "
         f"and the FX coefficient from {main.loc['dln_fx', 'coef']:.2f} to {with_vol.loc['dln_fx', 'coef']:.2f}; "
         f"Yahoo volumes not verified"),
        ("not conclusive", "Predictive version (factors lagged one week)",
         f"R2 at most {pred['r2'].max():.1%} on the full sample; smallest p {pred['pvalue'].min():.3f} among "
         f"{len(reg)} coefficient tests in the whole analysis, {int((reg['pvalue'] < 0.05).sum())} below 0.05"),
        ("not conclusive", "Foreign investor flows",
         "not available: no documented official historical source (TWSE OpenAPI has no history; the BFI82U "
         "web endpoint is undocumented and needs one request per day)"),
        ("untested observation", "The premium tracks US tech slightly better than the broad market",
         f"univariate R2 {ndx['r2']:.1%} for the Nasdaq-100 against {sp['r2']:.1%} for the S&P 500, with the same "
         f"beta ({ndx['coef']:.2f} and {sp['coef']:.2f}), so the conclusion does not depend on the index chosen"),
    ]
    return pd.DataFrame([{"analysis": "conclusion", "category": c, "statement": st, "evidence": ev}
                         for c, st, ev in rows])


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
        print(sel[cols].astype({"n": "Int64"}).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    reg = table[table["analysis"] == "regression"]
    base = reg[(reg["definition"] == "prev_ny") & (reg["frequency"] == "weekly") & (reg["term"] != "const")]
    contemp, pred = base[base["timing"] == "contemporaneous"], base[base["timing"] == "predictive"]
    print(f"\nCorrelation of S&P 500 and Nasdaq-100 weekly returns: {corr:.2f} (multivariate coefficients on the two "
          f"are collinear)")
    main = contemp["model"].str.startswith("univariate") | (contemp["model"] == "multivariate")
    show(contemp[(contemp["sample"] == "full") & main], "CONTEMPORANEOUS, prev_ny weekly, full sample")
    show(contemp[(contemp["sample"] != "full") & (contemp["model"] == "multivariate")],
         "CONTEMPORANEOUS, prev_ny weekly, multivariate by sub-period")
    show(contemp[(contemp["sample"] == "full") & contemp["model"].str.startswith("robustness")],
         "ROBUSTNESS: Nasdaq-100 instead of S&P 500, prev_ny weekly, full sample")
    small = contemp[(contemp["sample"] != "full") & (contemp["n"] < MIN_OBS)]["sample"].unique()
    if len(small):
        print(f"  Fewer than {MIN_OBS} observations, not conclusive: {', '.join(small)}")

    tests = table[(table["analysis"] == "fx_test") & (table["definition"] == "prev_ny") & (table["frequency"] == "weekly")
                  & (table["timing"] == "contemporaneous")]
    print("\nFX coefficient tests (Newey-West), prev_ny weekly contemporaneous")
    print(tests[["sample", "model", "hypothesis", "stat", "pvalue", "n"]].astype({"n": "Int64"}).to_string(index=False,
                                                                                   float_format=lambda v: f"{v:.4f}"))

    dec = table[table["analysis"].isin(["decomposition", "decomposition_check"])]
    print("\nDecomposition on the S&P 500: beta(dq) = beta(r_ADR) - beta(r_local in USD)")
    print(dec[["sample", "model", "coef", "ci_low", "ci_high", "n"]].astype({"n": "Int64"}).to_string(index=False,
                                                                              float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 100)
    print("PREDICTIVE (factors lagged one week), reported separately from the contemporaneous results")
    main_pred = pred["model"].str.startswith("univariate") | (pred["model"] == "multivariate")
    show(pred[(pred["sample"] == "full") & main_pred], "PREDICTIVE, prev_ny weekly, full sample")
    show(pred[(pred["sample"] != "full") & (pred["model"] == "multivariate")],
         "PREDICTIVE, prev_ny weekly, multivariate by sub-period")
    print("=" * 100)

    rob = reg[(reg["timing"] == "contemporaneous") & (reg["sample"] == "full") & (reg["term"] != "const")
              & ~((reg["definition"] == "prev_ny") & (reg["frequency"] == "weekly"))
              & (reg["model"].str.startswith("univariate") | (reg["model"] == "multivariate"))]
    print("\nROBUSTNESS, contemporaneous, full sample")
    print(rob[["definition", "frequency", "model", "term", "coef", "ci_low", "ci_high", "pvalue", "r2", "n"]]
          .astype({"n": "Int64"}).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    app = contemp[(contemp["sample"] == "full") & contemp["model"].str.startswith("appendix")]
    show(app, f"APPENDIX: both indices together, collinear (weekly correlation {corr:.2f}); separate "
              f"coefficients are not interpretable")

    pair = table[table["analysis"] == "pair_beta"]
    roll = table[table["analysis"] == "pair_beta_rolling"]
    print("\nPAIR TRADE EXPOSURE: long local share (USD) / short ADR, equal dollar weights, beta to the S&P 500")
    print(pair[["model", "coef", "ci_low", "ci_high", "pvalue", "n"]].astype({"n": "Int64"}).to_string(index=False,
                                                                               float_format=lambda v: f"{v:.4f}"))
    print(f"  Rolling 3-year: {len(roll)} windows, beta from {roll['coef'].min():.3f} to {roll['coef'].max():.3f}, "
          f"median {roll['coef'].median():.3f}")

    print("\nCONCLUSIONS")
    for r in table[table["analysis"] == "conclusion"].itertuples():
        print(f"  [{r.category}] {r.statement}: {r.evidence}")


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
    rows += pair_beta(weekly)
    roll = rolling(weekly)
    table = pd.concat([pd.DataFrame(rows), roll], ignore_index=True)
    table = pd.concat([table, conclusions(table)], ignore_index=True)
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
