"""Robustness of the premium strategy backtest (roadmap step 5b).

Everything uses the rules fixed a priori (adr_premium.backtest) and, unless a cost
scenario says otherwise, the default costs.

1. Block bootstrap of the weekly Sharpe ratio, gross and net.
2. Monte-Carlo with random entry dates on the real prices.
3. Execution fragility: one-day delays at entry and at exit, and the asynchrony test.
4. Cost scenarios and break-even cost per round trip.
5. S&P 500 beta hedge.
6. Stress periods.

Usage: python scripts/backtest_robustness.py
Output: data/backtest_robustness.csv, figures/robustness_*.png
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.backtest import (COST_COMPONENTS, PNL_COMPONENTS, Costs, Market, Rules,
                                  execution_dates, metrics, run_backtest, trade_final)
from adr_premium.data import download_market
from adr_premium.factors import build_panel
from adr_premium.premium import premium_prev_ny, premium_same_day
from adr_premium.robustness import (block_bootstrap_sharpe, hedged_daily, monte_carlo_random_entries,
                                    p_value_above, pair_beta_at, sharpe, stress_summary, weekly_pnl)
from backtest import load

SEED = 20261120
# Block length for the bootstrap, in weeks. Trades last 6.4 weeks on average (26 at
# most) and the strategy is flat in between, so the weekly P&L is dependent within a
# trade (in position or not, and the same premium cycle). 13 weeks, about twice the
# average holding period, keeps a typical trade and its neighbourhood inside one block;
# the autocorrelation of the weekly P&L printed by the script is small beyond a few
# lags, so longer blocks would mostly add variance.
BLOCK_WEEKS = 13
N_BOOTSTRAP = 5000
N_MONTE_CARLO = 2000
JULY_2020_ENTRY = pd.Timestamp("2020-07-10")
HEDGE_WINDOW = 156
HEDGE_MIN_WEEKS = 52
HEDGE_COST = 1e-4
STRESS = [("March 2020", "2020-03-01", "2020-03-31"),
          ("August 2024", "2024-08-01", "2024-08-31"),
          ("April 2025", "2025-04-01", "2025-04-30"),
          ("January-May 2026", "2026-01-01", "2026-05-31")]
COST_SCENARIOS = {
    "default": Costs(),
    "institutional": Costs(adr_commission=2e-4, local_commission=3e-4, fx_conversion=2e-4,
                           adr_borrow_fee=0.0025),
    "tax_only": replace(Costs.zero(), local_sale_tax=Costs().local_sale_tax),
}

ROWS: list[dict] = []


def record(section: str, scenario: str, **values) -> None:
    for k, v in values.items():
        ROWS.append({"section": section, "scenario": scenario, "metric": k, "value": v})


def fmt(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
def bootstrap(res) -> None:
    print("\n=== 1. Block bootstrap of the weekly Sharpe ratio "
          f"(blocks of {BLOCK_WEEKS} weeks, {N_BOOTSTRAP} draws, seed {SEED}) ===")
    acf = [weekly_pnl(res.daily["net"]).autocorr(k) for k in range(1, 9)]
    print("Autocorrelation of the weekly net P&L, lags 1-8: " + ", ".join(fmt(a, 2) for a in acf))
    print(f"Average holding period: {res.trades['holding_weeks'].mean():.1f} weeks, "
          f"longest {res.trades['holding_weeks'].max()}")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ("gross", "net"):
        w = weekly_pnl(res.daily[col])
        draws = block_bootstrap_sharpe(w, BLOCK_WEEKS, N_BOOTSTRAP, SEED)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        point = sharpe(w.to_numpy())
        share = float(np.mean(draws <= 0))
        print(f"{col:5s}: Sharpe {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
              f"share of draws <= 0: {share:.1%}")
        record("bootstrap", col, sharpe=point, ci_low=lo, ci_high=hi, share_le_zero=share)
        ax.hist(draws, bins=80, alpha=0.5, label=f"{col} (actual {point:.2f})")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Annualised weekly Sharpe ratio")
    ax.set_title("Block bootstrap of the Sharpe ratio (13-week blocks)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "robustness_bootstrap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def monte_carlo(res, mkt: Market) -> None:
    print(f"\n=== 2. Monte-Carlo: random entry dates ({N_MONTE_CARLO} replications) ===")
    sig = res.signal
    decisions = sig[sig["z"].notna() & sig["adr_exec"].notna() & sig["local_exec"].notna()]
    holdings = res.trades["holding_weeks"].to_numpy()
    sim = monte_carlo_random_entries(decisions, holdings, mkt, Costs(), N_MONTE_CARLO, SEED)
    actual = {"gross": res.trades["gross_pnl"].sum(), "net": res.trades["net_pnl"].sum()}
    print(f"{len(holdings)} trades, holding periods kept, {len(decisions)} weekly decision dates")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ("gross", "net")):
        p = p_value_above(sim[col], actual[col])
        q = sim[col].quantile([0.05, 0.5, 0.95])
        print(f"{col:5s}: actual {actual[col]:+.4f}; random entries median {q[0.5]:+.4f}, "
              f"5%-95% [{q[0.05]:+.4f}, {q[0.95]:+.4f}]; one-sided p-value {p:.4f}")
        record("monte_carlo", col, actual=actual[col], sim_mean=sim[col].mean(),
               sim_median=q[0.5], sim_p05=q[0.05], sim_p95=q[0.95], p_value=p)
        ax.hist(sim[col], bins=60, color="grey", alpha=0.7)
        ax.axvline(actual[col], color="C3", lw=1.5, label=f"actual (p = {p:.3f})")
        ax.set_title(f"Total {col} P&L, random entries")
        ax.set_xlabel("USD per 1 USD per leg")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "robustness_monte_carlo.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def describe_diff(name: str, base: pd.Series, other: pd.Series) -> dict:
    d = base - other
    t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
    return {"variant": name, "n": len(d), "mean_pnl": other.mean(), "median_pnl": other.median(),
            "std_pnl": other.std(), "total_pnl": other.sum(), "mean_loss_vs_normal": d.mean(),
            "t_loss": t, "share_worse": float((d > 0).mean())}


def execution(res, mkt: Market, raw, fx) -> None:
    print("\n=== 3. Execution fragility (per-trade gross P&L, one-day delays) ===")
    sig, trades = res.signal, res.trades
    lag = execution_dates(sig.index, mkt.adr.index, mkt.local.index, 1)

    def variant(entry: bool, exit_: bool) -> pd.DataFrame:
        t = trades.copy()
        if entry:
            t["adr_entry_date"] = lag.loc[t["entry_decision"], "adr_exec"].to_numpy()
            t["local_entry_date"] = lag.loc[t["entry_decision"], "local_exec"].to_numpy()
        if exit_:
            t["adr_exit_date"] = lag.loc[t["exit_decision"], "adr_exec"].to_numpy()
            t["local_exit_date"] = lag.loc[t["exit_decision"], "local_exec"].to_numpy()
        final = pd.DataFrame([trade_final(r, mkt, Costs()) for _, r in t.iterrows()], index=t.index)
        return pd.DataFrame({"gross": final["price_fx"] + final["dividends_gross"],
                             "net": final[PNL_COMPONENTS].sum(axis=1)})

    pnl = {"normal": variant(False, False), "entry_delayed": variant(True, False),
           "exit_delayed": variant(False, True), "both_delayed": variant(True, True)}
    keep_all = pd.Series(True, index=trades.index)
    no_july = trades["entry_decision"] != JULY_2020_ENTRY
    for label, mask in (("all trades", keep_all), ("without July 2020", no_july)):
        print(f"\n{label} (n = {mask.sum()}):")
        table = [describe_diff(k, pnl["normal"].loc[mask, "gross"], v.loc[mask, "gross"])
                 for k, v in pnl.items()]
        table = pd.DataFrame(table)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        for r in table.to_dict("records"):
            record("execution", f"{label}|{r['variant']}",
                   **{k: v for k, v in r.items() if k != "variant"})

    # Hypothesis: the gross gain lost with a delayed exit comes from asynchrony noise.
    # Measure of that noise: same_day minus prev_ny log premium at the exit decision date
    # (both use the FRED rate of their own ADR date).
    q_same = np.log1p(premium_same_day(raw["adr"], raw["local"], fx)["premium"])
    q_prev = np.log1p(premium_prev_ny(raw["adr"], raw["local"], fx)["premium"])
    gap = (q_same - q_prev).dropna()
    print("\nGap q_same_day - q_prev_ny at the decision dates (the ADR's own move on the "
          "decision date, i.e. part of the signal already reversed before the ADR trades):")
    for side, col_date in (("entry", "entry_decision"), ("exit", "exit_decision")):
        g = gap.reindex(pd.DatetimeIndex(trades[col_date])).dropna()
        t_stat = g.mean() / g.std() * np.sqrt(len(g))
        print(f"  {side:5s}: n={len(g)}, mean {g.mean():+.4f}, median {g.median():+.4f}, t={t_stat:+.2f} "
              f"(all days: mean {gap.mean():+.4f}, std {gap.std():.4f})")
        record("asynchrony_gap", side, n=len(g), mean=g.mean(), median=g.median(), t=t_stat)

    print("\nAsynchrony test: loss from a one-day delay vs gap = q_same_day - q_prev_ny")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (side, col_date, variant_name) in zip(
            axes, (("exit", "exit_decision", "exit_delayed"), ("entry", "entry_decision", "entry_delayed"))):
        loss = pnl["normal"]["gross"] - pnl[variant_name]["gross"]
        g = gap.reindex(pd.DatetimeIndex(trades[col_date])).to_numpy()
        for label, mask in (("all trades", keep_all.to_numpy()), ("without July 2020", no_july.to_numpy())):
            ok = mask & np.isfinite(g)
            x, y = g[ok], loss.to_numpy()[ok]
            pear = stats.pearsonr(x, y)
            spear = stats.spearmanr(x, y)
            slope = stats.linregress(x, y)
            print(f"  {side:5s} {label:18s} n={ok.sum():2d}  Pearson {pear.statistic:+.2f} "
                  f"(p={pear.pvalue:.2f})  Spearman {spear.statistic:+.2f} (p={spear.pvalue:.2f})  "
                  f"slope {slope.slope:+.3f} (t={slope.slope / slope.stderr:+.2f})")
            record("asynchrony_test", f"{side}|{label}", n=int(ok.sum()), pearson=pear.statistic,
                   pearson_p=pear.pvalue, spearman=spear.statistic, spearman_p=spear.pvalue,
                   slope=slope.slope, slope_t=slope.slope / slope.stderr)
        ax.scatter(g * 100, loss * 100, s=18)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(f"q_same_day - q_prev_ny at the {side} decision (%)")
        ax.set_ylabel(f"gross P&L lost with a delayed {side} (%)")
        ax.set_title(f"Delayed {side}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "robustness_execution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def costs(adr, local, fx, div) -> None:
    print("\n=== 4. Cost scenarios (a priori rules) ===")
    out = []
    for name, c in COST_SCENARIOS.items():
        res = run_backtest(adr, local, fx, div, rules=Rules(), costs=c)
        t = res.trades
        n = len(t)
        gross, net = t["gross_pnl"].sum(), t["net_pnl"].sum()
        total_cost = gross - net
        tax = -t["local_tax"].sum()
        m = metrics(res.daily, t, "net")
        row = {"scenario": name, "gross": gross, "net": net, "ann_return_net": m["ann_return"],
               "sharpe_weekly_net": m["sharpe_weekly"], "cost_per_trade_bp": total_cost / n * 1e4,
               "breakeven_cost_per_trade_bp": gross / n * 1e4,
               "tax_share_of_costs": tax / total_cost if total_cost else np.nan}
        out.append(row)
        record("costs", name, **{k: v for k, v in row.items() if k != "scenario"})
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Costs per trade are all-in (commissions, tax, FX, borrow, withholding), in bp of "
          "1 USD per leg; the break-even cost is the all-in cost per round trip that sets "
          "the net P&L to zero (gross P&L / number of trades).")


# ---------------------------------------------------------------------------
def hedge(res, raw, fx) -> pd.DataFrame:
    print(f"\n=== 5. S&P 500 beta hedge (beta on the {HEDGE_WINDOW} weeks before each entry, "
          f"{HEDGE_COST * 1e4:.0f} bp per hedge transaction) ===")
    market = download_market()
    panel = build_panel(premium_prev_ny(raw["adr"], raw["local"], fx), market, raw["adr"].dropna())
    betas = pd.Series([pair_beta_at(panel, d, HEDGE_WINDOW, HEDGE_MIN_WEEKS)
                       for d in res.trades["entry_decision"]], index=res.trades.index)
    print(f"Pair beta at entry: mean {betas.mean():+.3f}, min {betas.min():+.3f}, "
          f"max {betas.max():+.3f}; trades without a hedge (< {HEDGE_MIN_WEEKS} weeks of "
          f"history): {betas.isna().sum()}")
    hedged = hedged_daily(res.daily, res.trades, betas, market["spx"], HEDGE_COST)
    spx_w = np.log(market["spx"].dropna()).resample("W-FRI").last().diff()
    rows = []
    for name, d in (("unhedged", res.daily), ("hedged", hedged)):
        for col in ("gross", "net"):
            m = metrics(d, res.trades.assign(net_pnl=np.nan, gross_pnl=np.nan), col)
            w = weekly_pnl(d[col])
            both = pd.concat([w, spx_w], axis=1, join="inner").dropna()
            realised = stats.linregress(both.iloc[:, 1], both.iloc[:, 0]).slope
            rows.append({"version": name, "pnl": col, "ann_return": m["ann_return"],
                         "ann_vol_weekly": m["ann_vol_weekly"], "sharpe_weekly": m["sharpe_weekly"],
                         "max_drawdown": m["max_drawdown"], "total_pnl": m["total_pnl"],
                         "beta_to_spx_weekly": realised})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Hedge totals: price {hedged['hedge_price'].sum():+.4f}, cost {hedged['hedge_cost'].sum():+.4f}")
    for r in rows:
        record("hedge", f"{r['version']}|{r['pnl']}",
               **{k: v for k, v in r.items() if k not in ("version", "pnl")})

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(res.daily.index, res.daily["net"].cumsum(), lw=1, label="Net, unhedged")
    ax.plot(hedged.index, hedged["net"].cumsum(), lw=1, label="Net, S&P 500 hedged")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Cumulative P&L (USD per 1 USD per leg)")
    ax.set_title("Beta hedge with the S&P 500 (beta from the 156 weeks before each entry)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "robustness_hedge.png", dpi=150)
    plt.close(fig)
    return hedged


# ---------------------------------------------------------------------------
def stress(res, hedged: pd.DataFrame) -> None:
    print("\n=== 6. Stress periods ===")
    rows = []
    for name, start, end in STRESS:
        for version, d in (("unhedged", res.daily), ("hedged", hedged)):
            s = stress_summary(d, start, end)
            rows.append({"period": name, "version": version, **s})
            record("stress", f"{name}|{version}", **{k: v for k, v in s.items() if k != "worst_week_end"},
                   worst_week_end=str(s["worst_week_end"].date()) if pd.notna(s["worst_week_end"]) else "")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # Which trades overlap each period.
    t = res.trades
    for name, start, end in STRESS:
        first = t[["adr_entry_date", "local_entry_date"]].min(axis=1)
        last = t[["adr_exit_date", "local_exit_date"]].max(axis=1)
        hit = t[(first <= pd.Timestamp(end)) & (last >= pd.Timestamp(start))]
        trades_txt = ", ".join(f"{a.date()} -> {b.date()}" for a, b in
                               zip(hit["entry_decision"], hit["exit_decision"])) or "none"
        print(f"  {name}: trades overlapping the period: {trades_txt}")


def main() -> None:
    adr, local, fx, div = load()
    raw = pd.DataFrame({"adr": adr, "local": local})
    res = run_backtest(adr, local, fx, div, rules=Rules(), costs=Costs())
    mkt = Market.build(adr, local, fx, div)
    print(f"A priori rules, default costs: {len(res.trades)} trades, "
          f"gross {res.trades['gross_pnl'].sum():+.4f}, net {res.trades['net_pnl'].sum():+.4f}")

    bootstrap(res)
    monte_carlo(res, mkt)
    execution(res, mkt, raw, fx)
    costs(adr, local, fx, div)
    hedged = hedge(res, raw, fx)
    stress(res, hedged)

    pd.DataFrame(ROWS).to_csv(ROOT / "data" / "backtest_robustness.csv", index=False)
    print("\nOutputs: data/backtest_robustness.csv, figures/robustness_bootstrap.png, "
          "figures/robustness_monte_carlo.png, figures/robustness_execution.png, "
          "figures/robustness_hedge.png")


if __name__ == "__main__":
    main()
