"""Backtest of the premium mean-reversion strategy (roadmap step 5a).

Headline result: the rules fixed a priori in adr_premium.backtest (z-score over 26
weeks, entry z > 1.5, exit z < 0 or 26 weeks, short premium only) with the default
costs. The execution-lag variant and the sensitivity grids are reported WITHOUT any
selection: no grid point replaces the headline result.

P&L is in USD per 1 USD per leg; divide by 2 for a return on gross exposure.

Usage: python scripts/backtest.py
Output: data/backtest_results.csv, data/backtest_trades.csv,
        figures/backtest_cumulative_pnl.png, figures/backtest_signal.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.backtest import (FINANCING_ASSUMPTION, Costs, Rules, check_dividends,
                                  cost_breakdown, metrics, run_backtest)
from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_dividends, download_fred_fx
from adr_premium.securities import TSMC, Security

# Same sub-periods as scripts/mean_reversion.py.
SUBPERIODS = [
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020-2023", "2020-01-01", "2023-12-31"),
    ("2024-Apr 2026", "2024-01-01", "2026-04-30"),
    ("May 2026-end", "2026-05-01", None),
]
ENTRY_GRID = [1.0, 1.5, 2.0]
WINDOW_GRID = [13, 26, 52]
BORROW_GRID = [0.0025, 0.005, 0.01, 0.03]


def load(security: Security = TSMC):
    raw = download(security=security)
    with contextlib.redirect_stdout(io.StringIO()):  # the FX checks are reported in step 1
        fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    return raw["adr"], raw["local"], fx, download_dividends(security=security)


def summarise(res, scenario: str, rules: Rules, costs: Costs) -> list[dict]:
    """One row per (period, gross/net), with the parameters and the cost breakdown."""
    rows = []
    periods = [("full", None, None)] + SUBPERIODS
    for name, start, end in periods:
        daily = res.daily.loc[start:end]
        if daily.empty:
            continue
        trades = res.trades
        if len(trades):
            entry = trades["entry_decision"]
            mask = np.ones(len(trades), bool)
            if start:
                mask &= entry >= pd.Timestamp(start)
            if end:
                mask &= entry <= pd.Timestamp(end)
            trades = trades[mask]
        breakdown = cost_breakdown(daily)
        for col in ("gross", "net"):
            rows.append({
                "scenario": scenario, "period": name, "pnl": col,
                "entry_z": rules.entry_z, "window": rules.window, "extra_lag": rules.extra_lag,
                "borrow_fee": costs.adr_borrow_fee,
                **metrics(daily, trades, col),
                **{f"sum_{k}": v for k, v in breakdown.items()},
            })
    return rows


def plot(res, path_pnl: Path, path_signal: Path) -> None:
    daily = res.daily
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily["gross"].cumsum(), lw=1, label="Gross (price, FX, dividends)")
    ax.plot(daily.index, daily["net"].cumsum(), lw=1, label="Net of all costs")
    for _, t in res.trades.iterrows():
        ax.axvspan(min(t["adr_entry_date"], t["local_entry_date"]),
                   max(t["adr_exit_date"], t["local_exit_date"]), color="grey", alpha=0.15, lw=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Cumulative P&L (USD per 1 USD per leg)")
    ax.set_title("Short-premium strategy, rules fixed a priori (shaded: in position)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_pnl, dpi=150)
    plt.close(fig)

    sig = res.signal
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(sig.index, np.expm1(sig["q"]) * 100, lw=0.8, label="Weekly premium (prev_ny)")
    a1.plot(sig.index, np.expm1(sig["mean"]) * 100, lw=0.8, label="Trailing mean")
    a2.plot(sig.index, sig["z"], lw=0.8, color="grey")
    a2.axhline(Rules().entry_z, color="C3", lw=0.6, ls="--")
    a2.axhline(Rules().exit_z, color="C2", lw=0.6, ls="--")
    t = res.trades
    for ax, col in ((a1, "q"), (a2, "z")):
        ent = sig.loc[t["entry_decision"], col]
        ext = sig.loc[t["exit_decision"], col]
        if col == "q":
            ent, ext = np.expm1(ent) * 100, np.expm1(ext) * 100
        ax.scatter(ent.index, ent, marker="v", color="C3", s=22, zorder=3, label="Entry")
        ax.scatter(ext.index, ext, marker="^", color="C2", s=22, zorder=3, label="Exit")
        ax.grid(alpha=0.3)
    a1.set_ylabel("Premium (%)")
    a2.set_ylabel("z-score")
    a1.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path_signal, dpi=150)
    plt.close(fig)


def show(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")


def main() -> None:
    adr, local, fx, div = load()

    # Dividend consistency: 1 ADR = 5 shares, so the ADR dividend ~ 5 x local / FX.
    check = check_dividends(div.loc["2010":], fx)
    print(f"Dividends checked: {len(check)}, flagged (gap > 5% or unmatched): {check['flag'].sum()}, "
          f"largest gap {check['rel_diff'].abs().max():.2%} on "
          f"{check.loc[check['rel_diff'].abs().idxmax(), 'adr_ex_date'].date()}")
    if check["flag"].any():
        print(check[check["flag"]].to_string(index=False))

    rows, trade_tables = [], []
    base_rules, base_costs = Rules(), Costs()
    runs = [("a_priori", base_rules, base_costs),
            ("short_pays_net_dividend", base_rules, replace(base_costs, short_pays_net_dividend=True)),
            ("execution_lag_1d", replace(base_rules, extra_lag=1), base_costs)]
    runs += [(f"grid_entry{e}_window{w}", replace(base_rules, entry_z=e, window=w), base_costs)
             for e in ENTRY_GRID for w in WINDOW_GRID]
    runs += [(f"borrow_{b:.4f}", base_rules, replace(base_costs, adr_borrow_fee=b))
             for b in BORROW_GRID]
    results = {}
    for name, rules, costs in runs:
        res = run_backtest(adr, local, fx, div, rules=rules, costs=costs)
        results[name] = res
        rows += summarise(res, name, rules, costs)
        if name in ("a_priori", "execution_lag_1d"):
            trade_tables.append(res.trades.assign(scenario=name))

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "data" / "backtest_results.csv", index=False)
    trades = pd.concat(trade_tables, ignore_index=True)
    trades.to_csv(ROOT / "data" / "backtest_trades.csv", index=False)

    main_res = results["a_priori"]
    print("\n=== HEADLINE RESULT: rules fixed a priori (entry z > 1.5, window 26, exit z < 0 "
          "or 26 weeks), default costs ===")
    print(f"Sample: {main_res.start.date()} to {main_res.daily.index[-1].date()}")
    print(FINANCING_ASSUMPTION)
    print("FX: the local leg is converted at the FRED rate of its own execution dates. "
          "The ADR short pays the GROSS ADR dividend (see the net variant below).")
    m_cols = ["pnl", "ann_return", "ann_vol", "sharpe", "ann_vol_weekly", "sharpe_weekly",
              "max_drawdown", "total_pnl", "n_trades", "avg_holding_weeks", "win_rate", "exposure"]
    head = table[(table["scenario"] == "a_priori") & (table["period"] == "full")]
    print(show(head, m_cols))

    print("\nP&L decomposition, full sample (USD per 1 USD per leg):")
    print(cost_breakdown(main_res.daily).to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nVariant: the ADR short pays the dividend NET of withholding:")
    netdiv = table[(table["scenario"] == "short_pays_net_dividend") & (table["period"] == "full")]
    print(show(netdiv, m_cols + ["sum_withholding_tax"]))

    print("\nBy sub-period (a priori rules):")
    sub = table[(table["scenario"] == "a_priori") & (table["period"] != "full")]
    print(show(sub, ["period", "pnl", "ann_return", "sharpe_weekly", "max_drawdown", "total_pnl",
                     "n_trades", "win_rate", "exposure"]))

    print("\nExecution one trading day later on each leg (robustness):")
    lag = table[(table["scenario"] == "execution_lag_1d") & (table["period"] == "full")]
    print(show(lag, m_cols))

    full_net = table[(table["period"] == "full") & (table["pnl"] == "net")]
    full_gross = table[(table["period"] == "full") & (table["pnl"] == "gross")]
    grid = full_net[full_net["scenario"].str.startswith("grid_")]
    print("\nSensitivity, entry threshold x window (net; reported without selection):")
    print(grid.pivot(index="entry_z", columns="window", values="sharpe_weekly")
          .to_string(float_format=lambda x: f"{x:.3f}").replace("window", "Sharpe (weekly) window"))
    print(grid.pivot(index="entry_z", columns="window", values="ann_return")
          .to_string(float_format=lambda x: f"{x:.4f}").replace("window", "ann. return   window"))
    print(grid.pivot(index="entry_z", columns="window", values="n_trades")
          .to_string().replace("window", "n trades      window"))
    g_gross = full_gross[full_gross["scenario"].str.startswith("grid_")]
    print(g_gross.pivot(index="entry_z", columns="window", values="ann_return")
          .to_string(float_format=lambda x: f"{x:.4f}").replace("window", "gross return  window"))

    print("\nSensitivity, ADR borrow fee (net):")
    bor = full_net[full_net["scenario"].str.startswith("borrow_")]
    print(show(bor, ["borrow_fee", "ann_return", "sharpe_weekly", "total_pnl", "sum_borrow_fee"]))

    print("\nThe headline result is the a priori rule set above; the grids only show how "
          "fragile it is and are not used to choose parameters.")

    plot(main_res, ROOT / "figures" / "backtest_cumulative_pnl.png",
         ROOT / "figures" / "backtest_signal.png")
    print("\nOutputs: data/backtest_results.csv, data/backtest_trades.csv, "
          "figures/backtest_cumulative_pnl.png, figures/backtest_signal.png")


if __name__ == "__main__":
    main()
