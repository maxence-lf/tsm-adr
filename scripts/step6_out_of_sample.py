"""Step 6: out-of-sample test on UMC, ASE and Chunghwa, as pre-registered.

Plan: docs/preregistration_step6.md (tag prereg-step6). Every method and parameter is
that of the TSMC code; the only additions are those the plan lists (higher-cost
scenario, H5 variant, aggregation rule). TSMC is run through the same pipeline as an
in-sample reference (H1 to H4 only; H5 is not evaluated on TSMC). Anything not in the
plan is labelled "exploratory".

Usage: python scripts/step6_out_of_sample.py
Output: data/step6_hypotheses.csv, data/step6_results.csv, data/step6_trades.csv,
        data/step6_data_checks.csv, figures/step6_premiums.png, figures/step6_cumulative_pnl.png
"""
import contextlib
import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.backtest import (Costs, Market, Rules, check_dividends, cost_breakdown, execution_dates,
                                  metrics, run_backtest, trade_final, PNL_COMPONENTS)
from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_dividends, download_fred_fx, download_market
from adr_premium.factors import build_panel, check_alignment, identity_error, ols_newey_west
from adr_premium.premium import asynchrony_diagnostics, premium_prev_ny, premium_same_day
from adr_premium.robustness import (block_bootstrap_sharpe, hedged_daily, monte_carlo_random_entries,
                                    p_value_above, pair_beta_at, sharpe, weekly_pnl)
from adr_premium.securities import STEP6, TSMC, Security
from adr_premium.stats import ar1_half_life, arma11_half_life, to_weekly
from mean_reversion import DAYS_PER_WEEK, MIN_WEEKS, full_sample, rolling_arma

SEED = 20261120  # as in step 5b
BLOCK_WEEKS, N_BOOTSTRAP, N_MONTE_CARLO = 13, 5000, 2000
HEDGE_WINDOW, HEDGE_MIN_WEEKS, HEDGE_COST = 156, 52, 1e-4
RETURN_FLAG = 0.25  # daily returns flagged for verification (pre-registered)
WEAK_CI_WIDTH = 0.8  # H4 qualifier (pre-registered)
HIGH_COSTS = replace(Costs(), adr_commission=10e-4, local_commission=20e-4, fx_conversion=10e-4,
                     adr_borrow_fee=0.01)  # pre-registered higher-cost scenario
SUBPERIODS = [("2010-2019", "2010-01-01", "2019-12-31"), ("2020-2023", "2020-01-01", "2023-12-31"),
              ("2024-Apr 2026", "2024-01-01", "2026-04-30"), ("May 2026-end", "2026-05-01", None)]

RESULTS: list[dict] = []


def rec(sec: str, section: str, item: str, **values) -> None:
    for k, v in values.items():
        RESULTS.append({"security": sec, "section": section, "item": item, "metric": k, "value": v})


def load(sec: Security):
    raw = sec.clip(download(security=sec))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fx, filled = prepare_fred_fx(download_fred_fx(), raw.index[0])
    div = sec.clip(download_dividends(security=sec))
    market = download_market(security=sec)
    return raw, fx, div, market, buf.getvalue()


# ---------------------------------------------------------------------------
def data_checks(sec: Security, raw, fx, div, fx_report: str) -> list[dict]:
    rows = []
    for col in ("adr", "local"):
        r = raw[col].dropna().pct_change().dropna()
        for date, v in r[r.abs() > RETURN_FLAG].items():
            rows.append({"security": sec.key, "check": f"{col} daily return > 25%", "date": date.date(),
                         "value": v, "action": "flagged, kept"})
        rows.append({"security": sec.key, "check": f"{col} closes", "date": raw[col].first_valid_index().date(),
                     "value": int(raw[col].notna().sum()), "action": "first date and count"})
    removed = [l for l in fx_report.splitlines() if l.startswith("FX points")]
    rows.append({"security": sec.key, "check": "FX report", "date": None, "value": "; ".join(removed),
                 "action": "as for TSMC"})
    chk = check_dividends(div, fx, ratio=sec.ratio)
    rows.append({"security": sec.key, "check": "dividends checked", "date": None, "value": len(chk),
                 "action": f"max |gap| {chk['rel_diff'].abs().max():.3f}"})
    for r in chk[chk["flag"]].itertuples():
        rows.append({"security": sec.key, "check": "dividend flagged (gap > 5% or no local match)",
                     "date": r.adr_ex_date.date(), "value": r.adr_dividend, "action": "reported, kept"})
    return rows


# ---------------------------------------------------------------------------
def mean_reversion(sec: Security, series: dict) -> dict:
    out = {}
    for row in full_sample(series):
        rec(sec.key, "mean_reversion", f"full {row['definition']} {row['frequency']}",
            adf_pvalue=row["adf_pvalue"], kpss_pvalue=row["kpss_pvalue"], ar1_phi=row["phi"],
            ar1_half_life_days=row["half_life_days"])
    s = series["prev_ny"]
    for freq, x, scale in (("daily", s, 1), ("weekly", to_weekly(s), DAYS_PER_WEEK)):
        a = arma11_half_life(x)
        rec(sec.key, "mean_reversion", f"full prev_ny {freq} ARMA(1,1)", phi=a["phi"], theta=a["theta"],
            half_life_days=a["half_life"] * scale, converged=a["converged"])
    roll = pd.DataFrame(rolling_arma(series))
    for (freq, model), g in roll.groupby(["frequency", "model"], sort=False):
        kept = g[~g["excluded"]]
        rec(sec.key, "mean_reversion", f"rolling {freq} {model}", windows=len(g), excluded=int(g["excluded"].sum()),
            median_half_life_days=kept["half_life_days"].median())
    wk = roll[roll["frequency"] == "weekly"]
    arma = wk[(wk["model"] == "arma11") & ~wk["excluded"]]
    ar1_all = wk[wk["model"] == "ar1"]
    ar1_same = ar1_all[ar1_all["end"].isin(arma["end"])]
    out["h2_arma"] = arma["half_life_days"].median()
    out["h2_ar1"] = ar1_all["half_life_days"].median()  # TSMC rule: all AR(1) windows
    out["h2_ar1_matched"] = ar1_same["half_life_days"].median()  # check: ARMA-kept windows only
    out["h2_windows"] = len(arma)
    weekly = to_weekly(s)
    for label, start, end in SUBPERIODS:
        x = weekly.loc[start:end]
        n = int(x.notna().sum())
        if n < MIN_WEEKS:
            rec(sec.key, "mean_reversion_subperiod", label, weeks=n, note="fewer than 52 weeks, not estimated")
            continue
        ar, am = ar1_half_life(x), arma11_half_life(x)
        rec(sec.key, "mean_reversion_subperiod", label, weeks=n, mean_premium=x.mean(),
            ar1_half_life_days=ar["half_life"] * DAYS_PER_WEEK, arma_half_life_days=am["half_life"] * DAYS_PER_WEEK)
    return out


# ---------------------------------------------------------------------------
def factors(sec: Security, prev: pd.DataFrame, market: pd.DataFrame, adr: pd.Series):
    panel = build_panel(prev, market, adr.dropna(), "W-FRI")
    err = identity_error(panel)
    if err > 1e-10:
        raise ValueError(f"{sec.key}: dq decomposition fails ({err:.1e})")
    check_alignment(panel, market)
    main = ols_newey_west(panel["dq"], panel[["r_sp500", "dln_fx"]])
    for term, row in main["table"].iterrows():
        rec(sec.key, "factors_main", term, coef=row["coef"], ci_low=row["ci_low"], ci_high=row["ci_high"],
            pvalue=row["pvalue"], r2=main["r2"], n=main["n"], nw_lags=main["lags"])
    p = panel[["dq", "r_adr", "r_local_usd", "r_sp500"]].dropna()
    betas = {y: ols_newey_west(p[y], p[["r_sp500"]])["table"].loc["r_sp500"] for y in ("dq", "r_adr", "r_local_usd")}
    for y, row in betas.items():
        rec(sec.key, "factors_beta", y, coef=row["coef"], ci_low=row["ci_low"], ci_high=row["ci_high"], n=len(p))
    return panel, betas


# ---------------------------------------------------------------------------
def strategy(sec: Security, raw, fx, div, panel, market) -> dict:
    adr, local = raw["adr"], raw["local"]
    res = run_backtest(adr, local, fx, div, rules=Rules(), costs=Costs(), ratio=sec.ratio)
    out = {"res": res}
    t = res.trades
    for col in ("gross", "net"):
        m = metrics(res.daily, t, col)
        rec(sec.key, "strategy", f"default {col}", **{k: m[k] for k in (
            "ann_return", "ann_vol_weekly", "sharpe_weekly", "max_drawdown", "total_pnl", "n_trades",
            "avg_holding_weeks", "win_rate", "exposure", "years")})
    rec(sec.key, "strategy", "cost breakdown", **cost_breakdown(res.daily).to_dict())
    for col in ("gross", "net"):
        w = weekly_pnl(res.daily[col])
        draws = block_bootstrap_sharpe(w, BLOCK_WEEKS, N_BOOTSTRAP, SEED)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        out[f"sharpe_{col}"] = (sharpe(w.to_numpy()), lo, hi)
        rec(sec.key, "strategy_bootstrap", col, sharpe=out[f"sharpe_{col}"][0], ci_low=lo, ci_high=hi,
            share_le_zero=float(np.mean(draws <= 0)))
    # Higher-cost scenario (pre-registered).
    hc = run_backtest(adr, local, fx, div, rules=Rules(), costs=HIGH_COSTS, ratio=sec.ratio)
    m = metrics(hc.daily, hc.trades, "net")
    w = weekly_pnl(hc.daily["net"])
    lo, hi = np.nanpercentile(block_bootstrap_sharpe(w, BLOCK_WEEKS, N_BOOTSTRAP, SEED), [2.5, 97.5])
    rec(sec.key, "strategy", "high_costs net", ann_return=m["ann_return"], sharpe_weekly=m["sharpe_weekly"],
        total_pnl=m["total_pnl"], sharpe_ci_low=lo, sharpe_ci_high=hi)
    if len(t):
        n = len(t)
        rec(sec.key, "strategy", "costs per trade", breakeven_bp=t["gross_pnl"].sum() / n * 1e4,
            default_cost_bp=(t["gross_pnl"].sum() - t["net_pnl"].sum()) / n * 1e4,
            high_cost_bp=(hc.trades["gross_pnl"].sum() - hc.trades["net_pnl"].sum()) / n * 1e4,
            tax_share_default=-t["local_tax"].sum() / (t["gross_pnl"].sum() - t["net_pnl"].sum()))
        # Random entries.
        mkt = Market.build(adr, local, fx, div, sec.ratio)
        sig = res.signal
        decisions = sig[sig["z"].notna() & sig["adr_exec"].notna() & sig["local_exec"].notna()]
        sim = monte_carlo_random_entries(decisions, t["holding_weeks"].to_numpy(), mkt, Costs(), N_MONTE_CARLO, SEED)
        for col, actual in (("gross", t["gross_pnl"].sum()), ("net", t["net_pnl"].sum())):
            rec(sec.key, "strategy_monte_carlo", col, actual=actual, sim_median=sim[col].median(),
                p_value=p_value_above(sim[col], actual))
        # One-day execution delay on both legs.
        lag = run_backtest(adr, local, fx, div, rules=Rules(extra_lag=1), costs=Costs(), ratio=sec.ratio)
        rec(sec.key, "strategy", "execution_lag_1d", gross=lag.trades["gross_pnl"].sum() if len(lag.trades) else 0.0,
            net=lag.trades["net_pnl"].sum() if len(lag.trades) else 0.0)
        # Beta hedge.
        betas = pd.Series([pair_beta_at(panel, d, HEDGE_WINDOW, HEDGE_MIN_WEEKS) for d in t["entry_decision"]],
                          index=t.index)
        hedged = hedged_daily(res.daily, t, betas, market["spx"], HEDGE_COST)
        for name, d in (("unhedged", res.daily), ("hedged", hedged)):
            mm = metrics(d, t.assign(net_pnl=np.nan, gross_pnl=np.nan), "net")
            rec(sec.key, "strategy_hedge", name, ann_vol_weekly=mm["ann_vol_weekly"],
                sharpe_weekly=mm["sharpe_weekly"], max_drawdown=mm["max_drawdown"], total_pnl=mm["total_pnl"],
                mean_beta=betas.mean(), unhedged_trades=int(betas.isna().sum()))
    for label, start, end in SUBPERIODS:
        d = res.daily.loc[start:end]
        if len(d):
            rec(sec.key, "strategy_subperiod", label, net=d["net"].sum(), gross=d["gross"].sum(),
                trades=int(((t["entry_decision"] >= pd.Timestamp(start)) &
                            (t["entry_decision"] <= pd.Timestamp(end or "2100-01-01"))).sum()) if len(t) else 0)
    return out


def h5(sec: Security, raw, fx, div, base) -> dict:
    alt = run_backtest(raw["adr"], raw["local"], fx, div, rules=Rules(signal="same_day"), costs=Costs(),
                       ratio=sec.ratio)
    w_base = weekly_pnl(base.daily["net"])
    w_alt = weekly_pnl(alt.daily["net"])
    idx = w_base.index.union(w_alt.index)
    diff = w_alt.reindex(idx, fill_value=0.0) - w_base.reindex(idx, fill_value=0.0)
    x = diff.to_numpy()
    rng = np.random.default_rng(SEED)
    n = len(x)
    n_blocks = -(-n // BLOCK_WEEKS)
    starts = rng.integers(0, n, size=(N_BOOTSTRAP, n_blocks))
    ix = (starts[:, :, None] + np.arange(BLOCK_WEEKS)[None, None, :]).reshape(N_BOOTSTRAP, -1)[:, :n] % n
    totals = x[ix].sum(axis=1)
    lo, hi = np.percentile(totals, [2.5, 97.5])
    total_alt = alt.trades["net_pnl"].sum() if len(alt.trades) else 0.0
    total_base = base.trades["net_pnl"].sum() if len(base.trades) else 0.0
    rec(sec.key, "h5", "same_day minus prev_ny", diff=float(x.sum()), ci_low=lo, ci_high=hi,
        net_same_day=total_alt, net_prev_ny=total_base, trades_same_day=len(alt.trades),
        trades_prev_ny=len(base.trades))
    return {"diff": float(x.sum()), "lo": lo, "hi": hi, "alt": alt}


# ---------------------------------------------------------------------------
def analyse(sec: Security, with_h5: bool):
    raw, fx, div, market, fx_report = load(sec)
    checks = data_checks(sec, raw, fx, div, fx_report)
    r = sec.ratio
    prev = premium_prev_ny(raw["adr"], raw["local"], fx, ratio=r)
    same = premium_same_day(raw["adr"], raw["local"], fx, ratio=r)
    diag = asynchrony_diagnostics(same, prev)
    for name in diag.index:
        rec(sec.key, "asynchrony", name, **diag.loc[name].to_dict())
    series = {"prev_ny": prev["premium"], "same_day": same["premium"]}
    mr = mean_reversion(sec, series)
    panel, betas = factors(sec, prev, market, raw["adr"])
    st = strategy(sec, raw, fx, div, panel, market)
    hyp = []
    h1 = diag.loc["prev_ny", "daily_change_autocorr_lag1"]
    hyp.append(dict(hypothesis="H1", value=f"{h1:.3f}", criterion="< -0.2", confirmed=bool(h1 < -0.2), qualifier=""))
    ok2 = bool(mr["h2_arma"] > 2 * mr["h2_ar1"])
    ok2m = bool(mr["h2_arma"] > 2 * mr["h2_ar1_matched"])
    hyp.append(dict(hypothesis="H2", value=f"ARMA {mr['h2_arma']:.1f} d vs AR(1) {mr['h2_ar1']:.1f} d "
                    f"({mr['h2_windows']} windows)", criterion="ARMA > 2 x AR(1)", confirmed=ok2,
                    qualifier="" if ok2 == ok2m else f"matched-window check disagrees (AR(1) {mr['h2_ar1_matched']:.1f} d)"))
    b = betas["dq"]
    hyp.append(dict(hypothesis="H3", value=f"{b['coef']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}] "
                    f"(ADR {betas['r_adr']['coef']:.2f}, local {betas['r_local_usd']['coef']:.2f})",
                    criterion="difference > 0, 95% CI above 0", confirmed=bool(b["ci_low"] > 0), qualifier=""))
    s_net, lo, hi = st["sharpe_net"]
    ok4 = bool(lo <= 0)
    hyp.append(dict(hypothesis="H4", value=f"net Sharpe {s_net:.2f} [{lo:.2f}, {hi:.2f}], width {hi - lo:.2f}",
                    criterion="95% CI contains 0 or is below 0", confirmed=ok4,
                    qualifier="weak (low power)" if ok4 and hi - lo > WEAK_CI_WIDTH else ""))
    alt = None
    if with_h5:
        h = h5(sec, raw, fx, div, st["res"])
        alt = h["alt"]
        ok5 = h["diff"] > 0
        hyp.append(dict(hypothesis="H5", value=f"{h['diff']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}]",
                        criterion="difference > 0", confirmed=bool(ok5),
                        qualifier=("strongly confirmed" if ok5 and h["lo"] > 0 else "") if ok5 else ""))
    for hrow in hyp:
        hrow["security"] = sec.key
    return {"hyp": hyp, "checks": checks, "res": st["res"], "alt": alt, "prev": prev, "raw": raw, "fx": fx,
            "div": div}


# ---------------------------------------------------------------------------
def exploratory(out: dict) -> None:
    """Analyses not in the plan, all labelled exploratory."""
    print("\n=== EXPLORATORY (not in the pre-registration) ===")
    # 1. Mean reversion on q = ln(1 + premium) instead of the premium (the plan's text
    #    says q, the TSMC code uses the premium).
    for key, o in out.items():
        q = np.log1p(o["prev"]["premium"])
        roll = pd.DataFrame(rolling_arma({"prev_ny": q}))
        wk = roll[roll["frequency"] == "weekly"]
        arma = wk[(wk["model"] == "arma11") & ~wk["excluded"]]["half_life_days"].median()
        ar1 = wk[wk["model"] == "ar1"]["half_life_days"].median()
        rec(key, "exploratory", "H2 on q = ln(1 + premium)", arma=arma, ar1=ar1, confirmed=bool(arma > 2 * ar1))
        print(f"  {key}: H2 on q: ARMA {arma:.1f} d vs AR(1) {ar1:.1f} d -> {'confirmed' if arma > 2 * ar1 else 'not confirmed'}")
    # 2. Chunghwa: Yahoo has no local dividend for the 2013-07-15 ADR dividend.
    o = out.get("cht")
    if o is not None:
        t = o["res"].trades
        ex = pd.Timestamp("2013-07-15")
        hit = t[(t["adr_entry_date"] < ex) & (t["adr_exit_date"] >= ex)] if len(t) else t
        print(f"  cht: trades holding the ADR short over 2013-07-15: {len(hit)}")
        rec("cht", "exploratory", "missing 2013 local dividend", trades_affected=len(hit))


def main() -> None:
    pd.set_option("display.width", 220)
    out, hyps, checks = {}, [], []
    for sec in [TSMC] + STEP6:
        print(f"\n######## {sec.name} ({sec.adr} / {sec.local}, ratio {sec.ratio}) ########")
        o = analyse(sec, with_h5=sec.key != TSMC.key)
        out[sec.key] = o
        hyps += o["hyp"]
        checks += o["checks"]
        print(pd.DataFrame(o["hyp"])[["hypothesis", "value", "criterion", "confirmed", "qualifier"]].to_string(index=False))

    h = pd.DataFrame(hyps)[["security", "hypothesis", "value", "criterion", "confirmed", "qualifier"]]
    step6 = h[h["security"] != TSMC.key]
    agg = step6.groupby("hypothesis")["confirmed"].sum().astype(int)
    print("\n=== AGGREGATION (UMC, ASE, Chunghwa; generalised if confirmed for at least 2 of 3) ===")
    for hyp, n in agg.items():
        verdict = "generalised" if n >= 2 else "not generalised"
        print(f"  {hyp}: confirmed for {n} of 3 -> {verdict}")
        rec("step6", "aggregation", hyp, confirmed_count=int(n), verdict=verdict)

    exploratory({k: v for k, v in out.items() if k != TSMC.key})

    h.to_csv(ROOT / "data" / "step6_hypotheses.csv", index=False)
    pd.DataFrame(RESULTS).to_csv(ROOT / "data" / "step6_results.csv", index=False)
    pd.DataFrame(checks).to_csv(ROOT / "data" / "step6_data_checks.csv", index=False)
    trades = []
    for key, o in out.items():
        if len(o["res"].trades):
            trades.append(o["res"].trades.assign(security=key, signal="prev_ny"))
        if o["alt"] is not None and len(o["alt"].trades):
            trades.append(o["alt"].trades.assign(security=key, signal="same_day"))
    pd.concat(trades, ignore_index=True).to_csv(ROOT / "data" / "step6_trades.csv", index=False)

    keys = [s.key for s in STEP6]
    names = {s.key: f"{s.name} ({s.adr} / {s.local})" for s in STEP6}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, k in zip(axes, keys):
        p = to_weekly(out[k]["prev"]["premium"]) * 100
        ax.plot(p.index, p, lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(names[k])
        ax.set_ylabel("Premium (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "step6_premiums.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, k in zip(axes, keys):
        d = out[k]["res"].daily
        ax.plot(d.index, d["gross"].cumsum(), lw=1, label="prev_ny, gross")
        ax.plot(d.index, d["net"].cumsum(), lw=1, label="prev_ny, net")
        a = out[k]["alt"].daily
        ax.plot(a.index, a["net"].cumsum(), lw=1, ls="--", label="same_day (H5), net")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(names[k])
        ax.set_ylabel("USD per 1 USD per leg")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "step6_cumulative_pnl.png", dpi=150)
    plt.close(fig)
    print("\nOutputs: data/step6_hypotheses.csv, data/step6_results.csv, data/step6_data_checks.csv, "
          "data/step6_trades.csv, figures/step6_premiums.png, figures/step6_cumulative_pnl.png")


if __name__ == "__main__":
    main()
