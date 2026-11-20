"""Tables of the final report for roadmap steps 4 to 7, built from the CSV files in data/.

Imported by make_report_tables.py, which writes every table to docs/tables/ and fills
the <!-- table:name --> markers of the reports. No number in these tables is typed by
hand: each one is read from a result file or recomputed from the cached data.

Inputs: data/factor_results.csv, data/backtest_results.csv, data/backtest_trades.csv,
data/backtest_robustness.csv, data/step6_hypotheses.csv, data/step6_results.csv,
data/step6_data_checks.csv, data/arbitrage_band.csv, and the cached prices.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adr_premium.backtest import Costs, Market
from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import premium_prev_ny
from adr_premium.securities import STEP6, TSMC
from adr_premium.stats import to_weekly

DATA = ROOT / "data"
SUBS = ["full", "2010-2019", "2020-2023", "2024-Apr 2026", "May 2026-end"]
NAMES = {"tsmc": "TSMC", "umc": "UMC", "ase": "ASE", "cht": "Chunghwa"}
JULY_2020 = "without July 2020"


def pct(x, d=1):
    x = float(x)
    return "n/a" if pd.isna(x) else f"{x * 100:.{d}f}%"


def f(x, d=2):
    x = float(x)
    if pd.isna(x):
        return "n/a"
    if np.isinf(x):
        return "inf"
    return f"{x:,.{d}f}"


def ci(c, lo, hi, d=2):
    return f"{f(c, d)} [{f(lo, d)}, {f(hi, d)}]"


def pv(p):
    return "n/a" if pd.isna(p) else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def md(df, align=None):
    cols = list(df.columns)
    align = align or ["l"] + ["r"] * (len(cols) - 1)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if a == "r" else ":---" for a in align) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def load_late() -> dict:
    read = lambda n: pd.read_csv(DATA / n)
    return {"fr": read("factor_results.csv"), "bt": read("backtest_results.csv"),
            "tr": read("backtest_trades.csv"), "rb": read("backtest_robustness.csv"),
            "h6": read("step6_hypotheses.csv"), "r6": read("step6_results.csv"),
            "c6": read("step6_data_checks.csv"), "band": read("arbitrage_band.csv")}


# ---------------------------------------------------------------- step 4

def _reg(fr, model, sample="full", definition="prev_ny", frequency="weekly", timing="contemporaneous"):
    return fr[(fr["analysis"] == "regression") & (fr["model"] == model) & (fr["sample"] == sample)
              & (fr["definition"] == definition) & (fr["frequency"] == frequency) & (fr["timing"] == timing)
              ].set_index("term")


def t_factor_main(L):
    rows = []
    for s in SUBS:
        r = _reg(L["fr"], "multivariate", s)
        rows.append([s, int(r["n"].iloc[0]), ci(r.loc["r_sp500", "coef"], r.loc["r_sp500", "ci_low"], r.loc["r_sp500", "ci_high"]),
                     ci(r.loc["dln_fx", "coef"], r.loc["dln_fx", "ci_low"], r.loc["dln_fx", "ci_high"]),
                     pct(r["r2"].iloc[0])])
    return md(pd.DataFrame(rows, columns=["Sample", "Weeks", "S&P 500 coefficient [95% CI]",
                                          "d ln FX coefficient [95% CI]", "R2"]))


def t_factor_models(L):
    fr = L["fr"]
    models = ["univariate r_sp500", "univariate r_ndx", "univariate dln_fx", "univariate dln_value_ratio",
              "multivariate", "robustness: Nasdaq-100 instead of S&P 500", "robustness: with traded-value ratio",
              "appendix: both indices (collinear)"]
    terms = ["r_sp500", "r_ndx", "dln_fx", "dln_value_ratio"]
    rows = []
    for m in models:
        r = _reg(fr, m)
        cells = [f"{f(r.loc[t, 'coef'], 3)} (p {pv(r.loc[t, 'pvalue'])})" if t in r.index else "" for t in terms]
        rows.append([m] + cells + [pct(r["r2"].iloc[0]), int(r["n"].iloc[0])])
    return md(pd.DataFrame(rows, columns=["Model (weekly d q, `prev_ny`, full sample)", "S&P 500", "Nasdaq-100",
                                          "d ln FX", "d ln traded-value ratio", "R2", "Weeks"]))


def t_factor_betas(L):
    fr = L["fr"]
    rows = []
    for s in SUBS:
        d = fr[(fr["analysis"] == "decomposition") & (fr["sample"] == s)].set_index("model")
        if d.empty:
            continue
        rows.append([s, int(d["n"].iloc[0]),
                     ci(d.loc["r_adr on r_sp500", "coef"], d.loc["r_adr on r_sp500", "ci_low"], d.loc["r_adr on r_sp500", "ci_high"]),
                     ci(d.loc["r_local_usd on r_sp500", "coef"], d.loc["r_local_usd on r_sp500", "ci_low"],
                        d.loc["r_local_usd on r_sp500", "ci_high"]),
                     ci(d.loc["dq on r_sp500", "coef"], d.loc["dq on r_sp500", "ci_low"], d.loc["dq on r_sp500", "ci_high"])])
    pair = fr[(fr["analysis"] == "pair_beta") & (fr["model"] == "pair_simple on r_sp500")].iloc[0]
    out = md(pd.DataFrame(rows, columns=["Sample", "Weeks", "ADR beta [95% CI]", "Local share beta, USD [95% CI]",
                                         "Difference = beta of d q [95% CI]"]))
    return out + (f"\nPair trade (long local in USD, short ADR, equal dollar legs, simple returns), full "
                  f"sample: S&P 500 beta {ci(pair['coef'], pair['ci_low'], pair['ci_high'])}.\n")


def t_factor_fx_tests(L):
    fr = L["fr"]
    rows = []
    for m in ["univariate dln_fx", "multivariate", "robustness: Nasdaq-100 instead of S&P 500",
              "robustness: with traded-value ratio"]:
        t = fr[(fr["analysis"] == "fx_test") & (fr["model"] == m) & (fr["sample"] == "full")
               & (fr["definition"] == "prev_ny") & (fr["frequency"] == "weekly") & (fr["timing"] == "contemporaneous")]
        coef = _reg(fr, m).loc["dln_fx", "coef"]
        p1 = t[t["hypothesis"].str.contains("= 1")]["pvalue"].iloc[0]
        p0 = t[t["hypothesis"].str.contains("= 0")]["pvalue"].iloc[0]
        rows.append([m, f(coef, 2), pv(p1), pv(p0)])
    for m in ("univariate dln_fx", "multivariate"):
        monthly = _reg(fr, m, frequency="monthly").loc["dln_fx"]
        rows.append([f"{m}, monthly data", f(monthly["coef"], 2), "", pv(monthly["pvalue"])])
    return md(pd.DataFrame(rows, columns=["Model (full sample)", "d ln FX coefficient", "p-value, coefficient = 1",
                                          "p-value, coefficient = 0"]))


def t_factor_rolling(L):
    fr = L["fr"]
    rows = []
    for term in ["r_sp500", "dln_fx"]:
        g = fr[(fr["analysis"] == "rolling") & (fr["term"] == term)]
        excl0 = ((g["ci_low"] > 0) | (g["ci_high"] < 0)).mean()
        excl1 = pct(((g["ci_low"] > 1) | (g["ci_high"] < 1)).mean(), 0) if term == "dln_fx" else ""
        rows.append([f"{term} (joint model)", len(g), f(g["coef"].median()), f"{f(g['coef'].min())} to {f(g['coef'].max())}",
                     pct((g["coef"] > 0).mean(), 0), pct(excl0, 0), excl1])
    g = fr[fr["analysis"] == "pair_beta_rolling"]
    rows.append(["pair trade beta", len(g), f(g["coef"].median()), f"{f(g['coef'].min())} to {f(g['coef'].max())}",
                 pct((g["coef"] > 0).mean(), 0), pct(((g["ci_low"] > 0) | (g["ci_high"] < 0)).mean(), 0), ""])
    return md(pd.DataFrame(rows, columns=["Coefficient, 3-year windows, monthly step", "Windows", "Median", "Range",
                                          "Share positive", "Share with 95% CI excluding 0",
                                          "Share with 95% CI excluding 1"]))


def t_factor_predictive(L):
    fr = L["fr"]
    p = fr[(fr["analysis"] == "regression") & (fr["timing"] == "predictive") & (fr["sample"] == "full")
           & (fr["term"] != "const")]
    rows = [[m, f"{f(g['coef'].iloc[0], 3)} (p {pv(g['pvalue'].iloc[0])})" if len(g) == 1 else
             "; ".join(f"{t} {f(c, 3)} (p {pv(pp)})" for t, c, pp in zip(g["term"], g["coef"], g["pvalue"])),
             pct(g["r2"].iloc[0])] for m, g in p.groupby("model", sort=False)]
    return md(pd.DataFrame(rows, columns=["Predictive model (factors lagged one week, full sample)",
                                          "Coefficients", "R2"]), ["l", "l", "r"])


def t_factor_same_day(L):
    r = _reg(L["fr"], "multivariate", definition="same_day")
    return md(pd.DataFrame([["`same_day`, full sample", int(r["n"].iloc[0]),
                             ci(r.loc["r_sp500", "coef"], r.loc["r_sp500", "ci_low"], r.loc["r_sp500", "ci_high"]),
                             ci(r.loc["dln_fx", "coef"], r.loc["dln_fx", "ci_low"], r.loc["dln_fx", "ci_high"]),
                             pct(r["r2"].iloc[0])]],
                           columns=["Definition", "Weeks", "S&P 500 [95% CI]", "d ln FX [95% CI]", "R2"]))


# ---------------------------------------------------------------- step 5

COST_SOURCES = {
    "adr_commission": ("ADR commission, per transaction", "assumption, not verified"),
    "local_commission": ("Local commission, per transaction",
                         "assumption; regulatory ceiling 0.1425% (Taipei Times, 2000), discounts common"),
    "local_sale_tax": ("Taiwan securities transaction tax, on sales of the local share",
                       "0.3% of the sale proceeds, stated in TSMC's 20-F (2026) and secondary sources"),
    "fx_conversion": ("FX conversion, per conversion", "assumption, not verified"),
    "adr_borrow_fee": ("ADR borrow fee, per year, pro rata", "assumption, not verified"),
    "dividend_withholding": ("Withholding on local dividends", "21% since 2018, 20% before (EY; Investment Taiwan)"),
}


def t_costs(_):
    c = Costs()
    inst = Costs(adr_commission=2e-4, local_commission=3e-4, fx_conversion=2e-4, adr_borrow_fee=0.0025)
    rows = []
    for k, (label, src) in COST_SOURCES.items():
        v, w = getattr(c, k), getattr(inst, k)
        fmt = (lambda x: f"{x * 1e4:.0f} bp") if k not in ("adr_borrow_fee", "dividend_withholding") else (lambda x: pct(x, 2))
        rows.append([label, fmt(v), fmt(w), src])
    return md(pd.DataFrame(rows, columns=["Cost", "Default", "Institutional scenario", "Source"]), ["l", "r", "r", "l"])


def _bt(L, scenario, period="full"):
    b = L["bt"]
    return b[(b["scenario"] == scenario) & (b["period"] == period)].set_index("pnl")


def t_backtest_headline(L):
    b = _bt(L, "a_priori")
    rb = L["rb"]
    boot = rb[rb["section"] == "bootstrap"].pivot(index="scenario", columns="metric", values="value")
    rows = []
    for col in ("gross", "net"):
        r = b.loc[col]
        rows.append([col.capitalize(), f(r["total_pnl"], 3), pct(r["ann_return"], 2), pct(r["ann_vol"]),
                     pct(r["ann_vol_weekly"]), f(r["sharpe"]),
                     ci(boot.loc[col, "sharpe"], boot.loc[col, "ci_low"], boot.loc[col, "ci_high"]),
                     f(r["max_drawdown"]), pct(r["win_rate"], 0)])
    r = b.loc["net"]
    out = md(pd.DataFrame(rows, columns=["P&L", "Total", "Annual return", "Annual vol, daily marks",
                                         "Annual vol, weekly", "Sharpe, daily", "Sharpe, weekly [95% bootstrap CI]",
                                         "Max drawdown", "Winning trades"]))
    return out + (f"\n{int(r['n_trades'])} trades, {f(r['avg_holding_weeks'], 1)} weeks on average, in position "
                  f"{pct(r['exposure'], 0)} of the time, over {f(r['years'], 1)} years.\n")


def t_backtest_costs(L):
    b = _bt(L, "a_priori").loc["net"]
    comps = [("gross", "Gross P&L (price, FX, dividends)"), ("sum_local_tax", "Taiwan transaction tax"),
             ("sum_local_commission", "Local commission"), ("sum_adr_commission", "ADR commission"),
             ("sum_fx_conversion", "FX conversion"), ("sum_borrow_fee", "ADR borrow fee"),
             ("sum_withholding_tax", "Withholding on dividends"), ("sum_net", "Net P&L")]
    rows = [[lab, f"{b['sum_gross' if k == 'gross' else k]:+.3f}"] for k, lab in comps]
    return md(pd.DataFrame(rows, columns=["Component (total, USD per 1 USD per leg)", "Value"]))


def t_backtest_subperiods(L):
    b = L["bt"]
    rows = []
    for s in SUBS[1:]:
        g = _bt(L, "a_priori", s)
        rows.append([s, int(g.loc["net", "n_trades"]), f(g.loc["gross", "total_pnl"], 3), f(g.loc["net", "total_pnl"], 3),
                     f(g.loc["net", "sharpe_weekly"]), pct(g.loc["net", "exposure"], 0)])
    return md(pd.DataFrame(rows, columns=["Sub-period", "Trades (by entry)", "Gross P&L", "Net P&L", "Net Sharpe, weekly",
                                          "Exposure"]))


def t_backtest_variants(L):
    rows = []
    for sc, label in [("a_priori", "Rules fixed a priori, default costs"),
                      ("short_pays_net_dividend", "Short pays the ADR dividend net of withholding"),
                      ("execution_lag_1d", "Execution one trading day later on each leg")]:
        g = _bt(L, sc)
        rows.append([label, f(g.loc["gross", "total_pnl"], 3), f(g.loc["net", "total_pnl"], 3),
                     f(g.loc["net", "sharpe_weekly"]), f(g.loc["net", "max_drawdown"])])
    return md(pd.DataFrame(rows, columns=["Variant", "Gross P&L", "Net P&L", "Net Sharpe, weekly", "Net max drawdown"]))


def t_backtest_grid(L):
    b = L["bt"]
    g = b[(b["period"] == "full") & b["scenario"].str.startswith("grid_")]
    out = []
    for col, label in (("sharpe_weekly", "Net Sharpe, weekly"), ("ann_return", "Net annual return"),
                       ("n_trades", "Trades")):
        net = g[g["pnl"] == "net"].pivot(index="entry_z", columns="window", values=col)
        fmt = (lambda x: f(x)) if col == "sharpe_weekly" else (lambda x: pct(x, 2)) if col == "ann_return" else (lambda x: str(int(x)))
        rows = [[f"{label}, entry z > {e:g}"] + [fmt(net.loc[e, w]) for w in net.columns] for e in net.index]
        out += rows
    return md(pd.DataFrame(out, columns=["Quantity"] + [f"Window {int(w)} weeks" for w in sorted(g["window"].unique())]))


def t_backtest_borrow(L):
    b = L["bt"]
    g = b[(b["period"] == "full") & (b["pnl"] == "net") & b["scenario"].str.startswith("borrow_")]
    rows = [[pct(r.borrow_fee, 2), f(r.total_pnl, 3), pct(r.ann_return, 2), f(r.sharpe_weekly), f(r.sum_borrow_fee, 3)]
            for r in g.itertuples()]
    return md(pd.DataFrame(rows, columns=["ADR borrow fee", "Net P&L", "Net annual return", "Net Sharpe, weekly",
                                          "Total borrow cost"]))


def t_backtest_trades(L):
    t = L["tr"][L["tr"]["scenario"] == "a_priori"]
    rows = [[r.entry_decision, r.exit_decision, r.exit_reason, int(r.holding_weeks), f(r.entry_z), f(r.exit_z),
             f"{r.gross_pnl:+.4f}", f"{r.net_pnl:+.4f}"] for r in t.itertuples()]
    return md(pd.DataFrame(rows, columns=["Entry decision", "Exit decision", "Exit reason", "Weeks", "z at entry",
                                          "z at exit", "Gross P&L", "Net P&L"]), ["l", "l", "l", "r", "r", "r", "r", "r"])


def _rb(L, section):
    r = L["rb"][L["rb"]["section"] == section]
    return r.pivot(index="scenario", columns="metric", values="value")


def t_robustness_mc(L):
    m = _rb(L, "monte_carlo").astype(float)
    rows = [[c.capitalize(), f(m.loc[c, "actual"], 3), f(m.loc[c, "sim_median"], 3),
             f"{f(m.loc[c, 'sim_p05'], 3)} to {f(m.loc[c, 'sim_p95'], 3)}", f(m.loc[c, "p_value"], 3)]
            for c in ("gross", "net")]
    return md(pd.DataFrame(rows, columns=["P&L", "Actual", "Random entries, median", "Random entries, 5% to 95%",
                                          "One-sided p-value"]))


def t_robustness_execution(L):
    e = _rb(L, "execution").astype(float)
    rows = []
    for sc in e.index:
        sample, variant = sc.split("|")
        r = e.loc[sc]
        rows.append([sample, variant.replace("_", " "), f(r["total_pnl"], 3), pct(r["mean_pnl"], 2),
                     pct(r["mean_loss_vs_normal"], 2), "" if pd.isna(r["t_loss"]) else f(r["t_loss"], 1)])
    order = {"normal": 0, "entry delayed": 1, "exit delayed": 2, "both delayed": 3}
    df = pd.DataFrame(rows, columns=["Trades", "Execution", "Total gross P&L", "Mean per trade",
                                     "Mean loss vs normal", "t of the loss"])
    df["o"] = df["Execution"].map(order)
    df = df.sort_values(["Trades", "o"]).drop(columns="o")
    return md(df, ["l", "l", "r", "r", "r", "r"])


def t_robustness_asynchrony(L):
    g = _rb(L, "asynchrony_gap").astype(float)
    rows = [[s, int(g.loc[s, "n"]), f(g.loc[s, "mean"] * 100), f(g.loc[s, "median"] * 100), f(g.loc[s, "t"], 1)]
            for s in ("entry", "exit")]
    out = md(pd.DataFrame(rows, columns=["Decision", "Trades", "Mean gap (points)", "Median gap (points)", "t"]))
    a = _rb(L, "asynchrony_test").astype(float)
    rows = [[s.split("|")[0], s.split("|")[1], int(a.loc[s, "n"]), f(a.loc[s, "pearson"]), pv(a.loc[s, "pearson_p"]),
             f(a.loc[s, "spearman"]), pv(a.loc[s, "spearman_p"])] for s in a.index]
    return out + "\n" + md(pd.DataFrame(rows, columns=["Delay at", "Trades", "n", "Pearson", "p", "Spearman", "p"]),
                           ["l", "l", "r", "r", "r", "r", "r"])


def t_robustness_costs(L):
    c = _rb(L, "costs").astype(float)
    labels = {"default": "Default", "institutional": "Institutional", "tax_only": "Tax only"}
    rows = [[labels[s], f(c.loc[s, "cost_per_trade_bp"], 0) + " bp", f(c.loc[s, "net"], 3),
             f(c.loc[s, "sharpe_weekly_net"]), pct(c.loc[s, "tax_share_of_costs"], 0)] for s in labels]
    out = md(pd.DataFrame(rows, columns=["Cost scenario", "All-in cost per round trip", "Net P&L", "Net Sharpe, weekly",
                                         "Tax share of costs"]))
    return out + (f"\nBreak-even all-in cost per round trip (gross P&L / trades): "
                  f"{f(c.loc['default', 'breakeven_cost_per_trade_bp'], 0)} bp per 1 USD per leg.\n")


def t_robustness_hedge(L):
    h = _rb(L, "hedge").astype(float)
    rows = [[s.split("|")[0], s.split("|")[1], f(h.loc[s, "total_pnl"], 3), pct(h.loc[s, "ann_vol_weekly"]),
             f(h.loc[s, "sharpe_weekly"]), f(h.loc[s, "max_drawdown"]), f(h.loc[s, "beta_to_spx_weekly"], 3)]
            for s in ["unhedged|gross", "hedged|gross", "unhedged|net", "hedged|net"]]
    gain = h.loc["hedged|gross", "total_pnl"] - h.loc["unhedged|gross", "total_pnl"]
    out = md(pd.DataFrame(rows, columns=["Version", "P&L", "Total", "Annual vol, weekly", "Sharpe, weekly",
                                         "Max drawdown", "Realised beta to the S&P 500"]), ["l", "l", "r", "r", "r", "r", "r"])
    return out + f"\nPrice P&L of the index hedge over all trades: {gain:+.3f}.\n"


def t_robustness_stress(L):
    s = L["rb"][L["rb"]["section"] == "stress"]
    piv = s.pivot(index="scenario", columns="metric", values="value")
    rows = []
    for sc in piv.index:
        r = piv.loc[sc]
        rows.append([sc.split("|")[0], sc.split("|")[1], pct(float(r["in_position_share"]), 0),
                     f(float(r["net_pnl"]), 4), f(float(r["worst_week_net"]), 4)])
    order = ["March 2020", "August 2024", "April 2025", "January-May 2026"]
    df = pd.DataFrame(rows, columns=["Period", "Version", "Days in position", "Net P&L", "Worst week, net"])
    df["o"] = df["Period"].map({p: i for i, p in enumerate(order)})
    return md(df.sort_values(["o", "Version"], ascending=[True, False]).drop(columns="o"), ["l", "l", "r", "r", "r"])


def t_fx_date_correction(L):
    """Effect of converting the local leg at the FRED rate of its own execution dates
    instead of the ADR's (correction of step 5a), on the gross P&L of the trades."""
    adr, local = download()["adr"], download()["local"]
    with contextlib.redirect_stdout(io.StringIO()):
        fx, _ = prepare_fred_fx(download_fred_fx(), download().index[0])
    mkt = Market.build(adr, local, fx)
    t = L["tr"][L["tr"]["scenario"] == "a_priori"]
    g = t["local_exit_price"] / t["local_entry_price"]
    fxa = lambda c: np.array([mkt.fx_at(pd.Timestamp(d)) for d in t[c]])
    old = g * fxa("adr_entry_date") / fxa("adr_exit_date")
    new = g * fxa("local_entry_date") / fxa("local_exit_date")
    d = (new - old).to_numpy()
    tstat = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
    return md(pd.DataFrame([[len(d), f"{d.sum():+.4f}", f"{d.mean() * 1e4:+.1f} bp", f(tstat, 1)]],
                           columns=["Trades", "Change in total gross P&L", "Change per trade", "t"]))


# ---------------------------------------------------------------- step 6

def t_step6_securities(_):
    src = {"tsmc": "20-F filed 2010-04-15, 2018-04-19, 2026-04-16",
           "umc": "20-F filed 2010-04-30, 2018-04-26, 2026-04-30",
           "ase": "20-F filed 2019-04-26, 2026-04-01 (ASE Inc., 5 shares, before 2018-04-30)",
           "cht": "20-F filed 2010, 2011, 2012, 2013 and 2026-04-15"}
    rows = [[NAMES[s.key], f"{s.adr} / {s.local}", s.ratio, s.start, s.end or "2026-10-30", src[s.key]]
            for s in [TSMC] + STEP6]
    return md(pd.DataFrame(rows, columns=["Security", "ADR / local", "Shares per ADS", "Sample start", "Sample end",
                                          "Ratio checked in"]), ["l", "l", "r", "l", "l", "l"])


def t_step6_hypotheses(L):
    h = L["h6"].copy()
    h["cell"] = h.apply(lambda r: r["value"] + ("" if r["security"] == "tsmc" else
                                                (" yes" if r["confirmed"] else " no"))
                        + (f", {r['qualifier']}" if isinstance(r["qualifier"], str) and r["qualifier"] else ""), axis=1)
    piv = h.pivot(index="hypothesis", columns="security", values="cell")
    crit = h.drop_duplicates("hypothesis").set_index("hypothesis")["criterion"]
    agg = L["r6"][L["r6"]["section"] == "aggregation"].pivot(index="item", columns="metric", values="value")
    rows = []
    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        tsmc = piv.loc[hyp, "tsmc"] if isinstance(piv.loc[hyp].get("tsmc"), str) else "not evaluated"
        rows.append([hyp, crit[hyp], tsmc] + [piv.loc[hyp, k] for k in ("umc", "ase", "cht")]
                    + [f"{agg.loc[hyp, 'confirmed_count']}/3, {agg.loc[hyp, 'verdict']}"])
    return md(pd.DataFrame(rows, columns=["", "Criterion", "TSMC (in sample)", "UMC", "ASE", "Chunghwa", "Aggregation"]),
              ["l"] * 7)


def _r6(L, section, item=None):
    r = L["r6"][L["r6"]["section"] == section]
    if item is not None:
        r = r[r["item"] == item]
    return r


def _v(L, sec, section, item, metric):
    r = L["r6"]
    x = r[(r["security"] == sec) & (r["section"] == section) & (r["item"] == item) & (r["metric"] == metric)]["value"]
    return float(x.iloc[0]) if len(x) else np.nan


def t_step6_strategy(L):
    keys = ["tsmc", "umc", "ase", "cht"]
    spec = [("Trades", "strategy", "default net", "n_trades", lambda x: str(int(x))),
            ("Gross P&L", "strategy", "default gross", "total_pnl", lambda x: f(x, 3)),
            ("Net P&L", "strategy", "default net", "total_pnl", lambda x: f(x, 3)),
            ("Net Sharpe, weekly", "strategy_bootstrap", "net", "sharpe", lambda x: f(x)),
            ("95% CI, low", "strategy_bootstrap", "net", "ci_low", lambda x: f(x)),
            ("95% CI, high", "strategy_bootstrap", "net", "ci_high", lambda x: f(x)),
            ("Random-entry p-value, gross", "strategy_monte_carlo", "gross", "p_value", lambda x: pv(x)),
            ("Break-even cost per trade (bp)", "strategy", "costs per trade", "breakeven_bp", lambda x: f(x, 0)),
            ("All-in cost per trade, default (bp)", "strategy", "costs per trade", "default_cost_bp", lambda x: f(x, 0)),
            ("All-in cost per trade, higher (bp)", "strategy", "costs per trade", "high_cost_bp", lambda x: f(x, 0)),
            ("Tax share of default costs", "strategy", "costs per trade", "tax_share_default", lambda x: pct(x, 0)),
            ("Net P&L, higher costs", "strategy", "high_costs net", "total_pnl", lambda x: f(x, 3)),
            ("Net P&L, execution one day later", "strategy", "execution_lag_1d", "net", lambda x: f(x, 3)),
            ("Exposure", "strategy", "default net", "exposure", lambda x: pct(x, 0)),
            ("Years", "strategy", "default net", "years", lambda x: f(x, 1))]
    rows = [[label] + [fmt(_v(L, k, sec, item, met)) for k in keys] for label, sec, item, met, fmt in spec]
    return md(pd.DataFrame(rows, columns=["Strategy, a priori rules", "TSMC", "UMC", "ASE", "Chunghwa"]))


def t_step6_mean_reversion(L):
    keys = ["tsmc", "umc", "ase", "cht"]
    rows = []
    for label, item, metric, d in [("Rolling weekly ARMA(1,1), median half-life", "rolling weekly arma11", "median_half_life_days", 1),
                                   ("Rolling weekly AR(1), median half-life", "rolling weekly ar1", "median_half_life_days", 1),
                                   ("Rolling daily ARMA(1,1), median half-life", "rolling daily arma11", "median_half_life_days", 1),
                                   ("Weekly windows", "rolling weekly arma11", "windows", 0),
                                   ("Weekly ARMA windows excluded", "rolling weekly arma11", "excluded", 0)]:
        rows.append([label] + [f(_v(L, k, "mean_reversion", item, metric), d) for k in keys])
    for label, metric in [("Mean `prev_ny` premium", "premium_mean"), ("Premium std", "premium_std"),
                          ("Lag-1 autocorrelation of daily changes", "daily_change_autocorr_lag1")]:
        fmt = (lambda x: pct(x)) if metric != "daily_change_autocorr_lag1" else (lambda x: f(x, 3))
        rows.append([label] + [fmt(_v(L, k, "asynchrony", "prev_ny", metric)) for k in keys])
    for label, metric in [("Exploratory: H2 on q, ARMA median", "arma"), ("Exploratory: H2 on q, AR(1) median", "ar1")]:
        rows.append([label, ""] + [f(_v(L, k, "exploratory", "H2 on q = ln(1 + premium)", metric), 1)
                                   for k in ("umc", "ase", "cht")])
    return md(pd.DataFrame(rows, columns=["Quantity (half-lives in trading days)", "TSMC", "UMC", "ASE", "Chunghwa"]))


def t_step6_factors(L):
    rows = []
    for k in ["tsmc", "umc", "ase", "cht"]:
        b = lambda item, m: _v(L, k, "factors_beta", item, m)
        s = lambda term, m: _v(L, k, "factors_main", term, m)
        rows.append([NAMES[k], f(b("r_adr", "coef")), f(b("r_local_usd", "coef")),
                     ci(b("dq", "coef"), b("dq", "ci_low"), b("dq", "ci_high"), 3),
                     ci(s("r_sp500", "coef"), s("r_sp500", "ci_low"), s("r_sp500", "ci_high")),
                     ci(s("dln_fx", "coef"), s("dln_fx", "ci_low"), s("dln_fx", "ci_high")), int(s("r_sp500", "n"))])
    return md(pd.DataFrame(rows, columns=["Security", "ADR beta", "Local beta (USD)", "Difference [95% CI]",
                                          "Main model: S&P 500 [95% CI]", "Main model: d ln FX [95% CI]", "Weeks"]))


def t_step6_h5(L):
    rows = []
    for k in ["umc", "ase", "cht"]:
        g = lambda m: _v(L, k, "h5", "same_day minus prev_ny", m)
        rows.append([NAMES[k], int(g("trades_prev_ny")), f(g("net_prev_ny"), 3), int(g("trades_same_day")),
                     f(g("net_same_day"), 3), ci(g("diff"), g("ci_low"), g("ci_high"), 3)])
    return md(pd.DataFrame(rows, columns=["Security", "Trades, `prev_ny`", "Net P&L, `prev_ny`", "Trades, `same_day`",
                                          "Net P&L, `same_day`", "Difference [95% block-bootstrap CI]"]))


def t_step6_hedge(L):
    rows = []
    for k in ["tsmc", "umc", "ase", "cht"]:
        g = lambda item, m: _v(L, k, "strategy_hedge", item, m)
        rows.append([NAMES[k], f(g("unhedged", "mean_beta")), int(g("unhedged", "unhedged_trades")),
                     f"{pct(g('unhedged', 'ann_vol_weekly'))} / {pct(g('hedged', 'ann_vol_weekly'))}",
                     f"{f(g('unhedged', 'sharpe_weekly'))} / {f(g('hedged', 'sharpe_weekly'))}",
                     f"{f(g('unhedged', 'max_drawdown'))} / {f(g('hedged', 'max_drawdown'))}"])
    return md(pd.DataFrame(rows, columns=["Security", "Mean pair beta at entry", "Trades without hedge",
                                          "Net vol, weekly (unhedged / hedged)", "Net Sharpe (unhedged / hedged)",
                                          "Net max drawdown (unhedged / hedged)"]))


def t_step6_data_checks(L):
    c = L["c6"]
    c = c[c["security"] != "tsmc"]
    rows = [[NAMES[r.security], r.check, "" if pd.isna(r.date) else r.date, r.value, r.action] for r in c.itertuples()]
    return md(pd.DataFrame(rows, columns=["Security", "Check", "Date", "Value", "Action"]), ["l", "l", "l", "l", "l"])


# ---------------------------------------------------------------- step 7

def t_arbitrage_band(L):
    b = L["band"]
    rows = []
    for k in ["tsmc", "umc", "ase", "cht"]:
        g = b[b["security"] == k].set_index(["costs", "holding"])
        rows.append([NAMES[k], f(g["half_life_days"].iloc[0], 1),
                     f(g.loc[("default", "T = h"), "d_star"] * 100), f(g.loc[("default", "T = 2h"), "d_star"] * 100),
                     f(g.loc[("institutional", "T = h"), "d_star"] * 100), f(g.loc[("institutional", "T = 2h"), "d_star"] * 100),
                     pct(g.loc[("default", "T = h"), "share_above"], 0), pct(g.loc[("default", "T = h"), "share_abs_above"], 0),
                     f(g["dev_std"].iloc[0] * 100)])
    c = b.drop_duplicates("costs").set_index("costs")
    out = md(pd.DataFrame(rows, columns=["Security", "h (trading days)", "D*, default, T = h", "D*, default, T = 2h",
                                         "D*, institutional, T = h", "D*, institutional, T = 2h",
                                         "Weeks with deviation > D* (default, T = h)", "Weeks with abs deviation > D*",
                                         "Deviation std (points)"]))
    return out + (f"\nD* in points of q. Round-trip cost C: {c.loc['default', 'round_trip_cost'] * 1e4:.0f} bp (default), "
                  f"{c.loc['institutional', 'round_trip_cost'] * 1e4:.0f} bp (institutional); borrow fee "
                  f"{pct(c.loc['default', 'borrow_fee'], 2)} and {pct(c.loc['institutional', 'borrow_fee'], 2)} a year.\n")


def t_premium_sign(_):
    rows = []
    for sec in [TSMC] + STEP6:
        raw = sec.clip(download(security=sec))
        with contextlib.redirect_stdout(io.StringIO()):
            fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
        w = to_weekly(premium_prev_ny(raw["adr"], raw["local"], fx, ratio=sec.ratio)["premium"]).dropna()
        rows.append([NAMES[sec.key], len(w), pct((w < 0).mean(), 0), pct((w < -0.02).mean(), 0), pct(w.min()),
                     pct(w.quantile(0.05)), pct(w.median()), pct(w.quantile(0.95)), pct(w.max())])
    return md(pd.DataFrame(rows, columns=["Security", "Weeks", "Weeks at a discount", "Weeks below -2%", "Min", "5%",
                                          "Median", "95%", "Max"]))


def issue_rows(L) -> list[list[str]]:
    """Issue rows for steps 4 to 7, appended to the table of methodological issues."""
    fr = L["fr"]
    both = _reg(fr, "appendix: both indices (collinear)")
    uni_ndx = _reg(fr, "univariate r_ndx").loc["r_ndx", "coef"]
    uni_spx = _reg(fr, "univariate r_sp500").loc["r_sp500", "coef"]
    b = _bt(L, "a_priori")
    lag = _bt(L, "execution_lag_1d")
    g = _rb(L, "asynchrony_gap").astype(float)
    hedge = _rb(L, "hedge").astype(float)
    a = _rb(L, "asynchrony_test").astype(float)
    nets = {k: _v(L, k, "strategy", "default net", "total_pnl") for k in ["tsmc", "umc", "ase", "cht"]}
    h5 = {k: (_v(L, k, "h5", "same_day minus prev_ny", "ci_low"), _v(L, k, "h5", "same_day minus prev_ny", "ci_high"))
          for k in ["umc", "ase", "cht"]}
    fxc = t_fx_date_correction(L).splitlines()[2].split("|")
    band = L["band"]
    bt = band[(band["costs"] == "default") & (band["holding"] == "T = h")]
    return [
        ["f) US factors read on the wrong date (step 4)",
         "With `prev_ny`, the Taipei date d pairs with the NY close of d-1; reading the S&P 500 on d would reintroduce "
         "asynchrony. A check compares the date of every factor value with the ADR's NY date",
         "Factors read on the ADR's NY date; the check runs on every panel and was confirmed by reintroducing the bug",
         "Main-model S&P 500 coefficient on correctly aligned data: "
         f"{f(_reg(fr, 'multivariate').loc['r_sp500', 'coef'])}"],
        ["g) Two collinear US indices (step 4)",
         "With both indices in one model, their coefficients change sign",
         "One index (S&P 500), chosen a priori; Nasdaq-100 as a robustness check",
         f"Univariate S&P 500 {f(uni_spx, 3)} and Nasdaq-100 {f(uni_ndx, 3)} -> jointly "
         f"{f(both.loc['r_sp500', 'coef'], 3)} and {f(both.loc['r_ndx', 'coef'], 3)}"],
        ["h) P&L computed on premium changes (step 5)",
         "The `prev_ny` premium pairs a Taipei close with an ADR close already past at decision time; at the entry "
         f"decision the ADR's own close has already moved the premium by {f(g.loc['entry', 'mean'] * 100)} points "
         f"(t = {f(g.loc['entry', 't'], 1)}), at the exit by {f(g.loc['exit', 'mean'] * 100, 2)} (t = {f(g.loc['exit', 't'], 1)})",
         "Each leg executed at its first close after the decision; P&L on execution prices and each leg's own closes",
         f"About {f((g.loc['exit', 'mean'] - g.loc['entry', 'mean']) * 100, 1)} points per trade of apparent "
         "convergence are not capturable"],
        ["i) Exchange rate of the local leg (step 5)",
         "The local sale was first converted at the FRED rate of the ADR's exit date, before the local sale itself",
         "FRED rate of the local leg's own execution dates, at entry and exit; a test fails with the old rule",
         f"Gross P&L {fxc[2].strip()} in total, {fxc[3].strip()} per trade (t = {fxc[4].strip()})"],
        ["j) Daily marks of two asynchronous legs (step 5)",
         "Daily P&L mixes a Taipei close and a NY close and reverses the next day",
         "Volatility and Sharpe ratio computed on weekly P&L",
         f"Gross annual volatility {pct(b.loc['gross', 'ann_vol'])} (daily) -> {pct(b.loc['gross', 'ann_vol_weekly'])} "
         f"(weekly); gross Sharpe {f(b.loc['gross', 'sharpe'])} -> {f(b.loc['gross', 'sharpe_weekly'])}"],
        ["k) Dependence on the execution day (step 5)",
         "Delaying both legs by one trading day",
         "Reported as a robustness result; the exit hypothesis based on asynchrony was tested and rejected",
         f"Gross P&L {f(b.loc['gross', 'total_pnl'], 3)} -> {f(lag.loc['gross', 'total_pnl'], 3)}; net "
         f"{f(b.loc['net', 'total_pnl'], 3)} -> {f(lag.loc['net', 'total_pnl'], 3)}; correlation of the exit loss with "
         f"the asynchrony gap {f(a.loc['exit|all trades', 'pearson'])} -> {f(a.loc['exit|without July 2020', 'pearson'])} "
         "without July 2020"],
        ["l) A hedge that seems to work (step 5)",
         "Hedging the pair's S&P 500 beta doubles the net Sharpe ratio",
         "Volatility and the hedge's own P&L reported next to the Sharpe ratio",
         f"Net Sharpe {f(hedge.loc['unhedged|net', 'sharpe_weekly'])} -> {f(hedge.loc['hedged|net', 'sharpe_weekly'])}, "
         f"but weekly volatility {pct(hedge.loc['unhedged|net', 'ann_vol_weekly'])} -> "
         f"{pct(hedge.loc['hedged|net', 'ann_vol_weekly'])}: the gain comes from the index rising"],
        ["m) Rules designed on TSMC (step 6)",
         "The rules were fixed before any backtest but chosen knowing TSMC's premium",
         "Pre-registered out-of-sample test on three other ADRs, same rules",
         f"Net P&L: TSMC {nets['tsmc']:+.3f} (in sample) against UMC {nets['umc']:+.3f}, ASE {nets['ase']:+.3f}, "
         f"Chunghwa {nets['cht']:+.3f}"],
        ["n) Aggregation rule of H5 (step 6)",
         "H5 is 'generalised' (2 of 3) while every interval contains 0",
         "No conclusion drawn on H5; the limit of the rule is reported",
         "Intervals: " + "; ".join(f"{NAMES[k]} [{h5[k][0]:+.3f}, {h5[k][1]:+.3f}]" for k in h5)],
        ["o) Plan text against plan code (step 6)",
         "The pre-registration text says mean reversion on q, the TSMC code it requires uses the premium",
         "TSMC code followed; q version reported as exploratory",
         "Same H2 verdict for all three securities"],
        ["p) Band computed on noisy deviations (step 7)",
         "Observed deviations exceed D* in a large share of weeks, while the strategy fails after costs",
         "D* presented as a derived lower bound; asynchrony and risk discussed",
         f"Weeks above D* (default, T = h): {pct(bt['share_above'].min(), 0)} to {pct(bt['share_above'].max(), 0)}; "
         "asynchrony per TSMC trade larger than D*"],
    ]


def short_issue_rows(L) -> list[list[str]]:
    """Rows h, k and m of the short issues table, with a before -> after readable on its own."""
    g = _rb(L, "asynchrony_gap").astype(float)
    b, lag = _bt(L, "a_priori"), _bt(L, "execution_lag_1d")
    nets = {k: _v(L, k, "strategy", "default net", "total_pnl") for k in ["tsmc", "umc", "ase", "cht"]}
    return [
        ["h) P&L computed on premium changes",
         f"Convergence counted on the observed premium -> about {f((g.loc['exit', 'mean'] - g.loc['entry', 'mean']) * 100, 1)} "
         f"points per trade cannot be captured ({f(abs(g.loc['entry', 'mean']) * 100)} already gone at entry, "
         f"{f(g.loc['exit', 'mean'] * 100)} at exit); P&L computed on execution prices"],
        ["k) Dependence on the execution day",
         f"Net P&L {f(b.loc['net', 'total_pnl'], 3)} when each leg trades at its first close after the decision -> "
         f"{f(lag.loc['net', 'total_pnl'], 3)} when both legs trade one day later"],
        ["m) Rules designed on TSMC",
         f"Strategy judged on TSMC (net P&L {nets['tsmc']:+.3f}) -> loses on all three out-of-sample ADRs "
         f"(UMC {nets['umc']:+.3f}, ASE {nets['ase']:+.3f}, Chunghwa {nets['cht']:+.3f})"],
    ]


def t_umc_2010(_):
    """Monthly mean prev_ny premium of UMC from January 2010 to June 2011."""
    sec = [s for s in STEP6 if s.key == "umc"][0]
    raw = sec.clip(download(security=sec))
    with contextlib.redirect_stdout(io.StringIO()):
        fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    p = premium_prev_ny(raw["adr"], raw["local"], fx, ratio=sec.ratio)["premium"].loc["2010-01":"2011-06"]
    g = p.groupby(p.index.to_period("M"))
    rows = [[str(m), pct(v.mean()), pct(v.min()), pct(v.max()), len(v)] for m, v in g]
    return md(pd.DataFrame(rows, columns=["Month", "Mean UMC `prev_ny` premium", "Min", "Max", "Trading days"]))


TABLES_LATE = [
    ("factor_main", t_factor_main), ("factor_models", t_factor_models), ("factor_betas", t_factor_betas),
    ("factor_fx_tests", t_factor_fx_tests), ("factor_rolling", t_factor_rolling),
    ("factor_predictive", t_factor_predictive), ("factor_same_day", t_factor_same_day),
    ("costs", t_costs), ("backtest_headline", t_backtest_headline), ("backtest_costs", t_backtest_costs),
    ("backtest_subperiods", t_backtest_subperiods), ("backtest_variants", t_backtest_variants),
    ("backtest_grid", t_backtest_grid), ("backtest_borrow", t_backtest_borrow), ("backtest_trades", t_backtest_trades),
    ("robustness_mc", t_robustness_mc), ("robustness_execution", t_robustness_execution),
    ("robustness_asynchrony", t_robustness_asynchrony), ("robustness_costs", t_robustness_costs),
    ("robustness_hedge", t_robustness_hedge), ("robustness_stress", t_robustness_stress),
    ("fx_date_correction", t_fx_date_correction),
    ("step6_securities", t_step6_securities), ("step6_hypotheses", t_step6_hypotheses),
    ("step6_strategy", t_step6_strategy), ("step6_mean_reversion", t_step6_mean_reversion),
    ("step6_factors", t_step6_factors), ("step6_h5", t_step6_h5), ("step6_hedge", t_step6_hedge),
    ("step6_data_checks", t_step6_data_checks),
    ("arbitrage_band", t_arbitrage_band), ("premium_sign", t_premium_sign), ("umc_2010", t_umc_2010),
]
