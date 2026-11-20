"""Robustness tools for the premium strategy backtest (roadmap step 5b).

- block bootstrap of the weekly Sharpe ratio;
- Monte-Carlo with random entry dates (same number of trades, same holding periods);
- S&P 500 beta hedge with a beta estimated on past data only;
- stress-period summaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adr_premium.backtest import PNL_COMPONENTS, Costs, Market, trade_final

# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def weekly_pnl(daily: pd.Series) -> pd.Series:
    """Weekly sums of a daily P&L series (weeks ending on Friday)."""
    return daily.resample("W-FRI").sum()


def sharpe(x: np.ndarray, periods_per_year: int = 52) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def block_bootstrap_sharpe(weekly: pd.Series, block: int = 13, n_draws: int = 5000,
                           seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap of the annualised weekly Sharpe ratio.

    Each draw glues blocks of ``block`` consecutive weeks, starting at uniform random
    weeks (wrapping around the end), until the original length is reached. Blocks keep
    the dependence inside a trade (several weeks in position, then weeks at zero).
    """
    x = weekly.to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_draws, -1)[:, :n] % n
    samples = x[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, samples.mean(axis=1) / sd * np.sqrt(52), np.nan)


# ---------------------------------------------------------------------------
# Monte-Carlo with random entries
# ---------------------------------------------------------------------------
def random_placements(n_slots: int, holdings: np.ndarray, rng) -> list[tuple[int, int]]:
    """Random non-overlapping (entry, exit) index pairs with the given holding periods.

    A trade held h weeks occupies slots [i, i + h]; the next entry is strictly after
    the previous exit, as in the real strategy. The order of the holding periods is
    shuffled, and the free slots are spread uniformly at random between the trades.
    """
    h = rng.permutation(np.asarray(holdings, dtype=int))
    lengths = h + 1
    free = n_slots - lengths.sum()
    if free < 0:
        raise ValueError("the trades do not fit in the sample")
    k = len(h)
    # Uniform composition of the free slots into k + 1 gaps.
    cuts = np.sort(rng.choice(free + k, size=k, replace=False))
    gaps = cuts - np.arange(k)
    out, offset = [], 0
    for j in range(k):
        start = gaps[j] + offset
        out.append((int(start), int(start + h[j])))
        offset += lengths[j]
    return out


def monte_carlo_random_entries(decisions: pd.DataFrame, holdings: np.ndarray, mkt: Market,
                               costs: Costs, n_reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Total gross and net P&L of ``n_reps`` strategies with random entry dates.

    ``decisions`` holds the executable weekly decision dates (index) with their
    ``adr_exec`` and ``local_exec`` dates, under the same execution rules as the real
    strategy. Each replication keeps the number of trades and their holding periods.
    """
    rng = np.random.default_rng(seed)
    adr_exec = decisions["adr_exec"].to_numpy()
    local_exec = decisions["local_exec"].to_numpy()
    cache: dict[tuple[int, int], pd.Series] = {}
    rows = []
    for _ in range(n_reps):
        gross = net = 0.0
        for i, j in random_placements(len(decisions), holdings, rng):
            if (i, j) not in cache:
                trade = {"adr_entry_date": adr_exec[i], "local_entry_date": local_exec[i],
                         "adr_exit_date": adr_exec[j], "local_exit_date": local_exec[j]}
                cache[(i, j)] = trade_final(trade, mkt, costs)
            r = cache[(i, j)]
            gross += r["price_fx"] + r["dividends_gross"]
            net += r[PNL_COMPONENTS].sum()
        rows.append({"gross": gross, "net": net})
    return pd.DataFrame(rows)


def p_value_above(simulated: np.ndarray, actual: float) -> float:
    """One-sided p-value: share of simulations at least as good as the actual result."""
    simulated = np.asarray(simulated)
    return float((1 + np.sum(simulated >= actual)) / (1 + len(simulated)))


# ---------------------------------------------------------------------------
# Beta hedge
# ---------------------------------------------------------------------------
def pair_beta_at(panel: pd.DataFrame, decision, window: int = 156, min_obs: int = 52) -> float:
    """S&P 500 beta of the pair (long local, short ADR) estimated before ``decision``.

    The pair's weekly log return is -dq = r_local_usd - r_adr. Only the ``window``
    weekly rows whose Taipei date is strictly before the decision date are used, so
    the decision week itself and anything later are excluded. NaN with fewer than
    ``min_obs`` usable weeks.
    """
    past = panel[pd.DatetimeIndex(panel["date"]) < pd.Timestamp(decision)].tail(window)
    d = pd.DataFrame({"y": -past["dq"], "x": past["r_sp500"]}).dropna()
    if len(d) < min_obs:
        return np.nan
    x = d["x"] - d["x"].mean()
    return float((x * (d["y"] - d["y"].mean())).sum() / (x ** 2).sum())


def hedge_path(trade, beta: float, spx: pd.Series, calendar: pd.DatetimeIndex,
               cost: float = 1e-4) -> pd.DataFrame:
    """Cumulative P&L of the S&P 500 hedge of one trade (USD per 1 USD per leg).

    The hedge holds -beta USD of the index (a long position when the pair's beta is
    negative) from the ADR entry close to the ADR exit close, on NY closes. Index
    returns stand in for a futures contract; ``cost`` is charged on each hedge
    transaction. No hedge when beta is NaN.
    """
    out = pd.DataFrame(0.0, index=calendar, columns=["hedge_price", "hedge_cost"])
    if not np.isfinite(beta) or beta == 0:
        return out
    a_in, a_out = pd.Timestamp(trade["adr_entry_date"]), pd.Timestamp(trade["adr_exit_date"])
    s = spx.dropna()
    s_cal = s.reindex(s.index.union(calendar)).ffill().reindex(calendar)
    s_in, s_out = float(s.asof(a_in)), float(s.asof(a_out))
    h = -beta
    pnl = h * (s_cal / s_in - 1)
    pnl[calendar < a_in] = 0.0
    pnl[calendar >= a_out] = h * (s_out / s_in - 1)
    out["hedge_price"] = pnl
    out.loc[calendar >= a_in, "hedge_cost"] -= cost * abs(h)
    out.loc[calendar >= a_out, "hedge_cost"] -= cost * abs(h) * s_out / s_in
    return out


def hedged_daily(daily: pd.DataFrame, trades: pd.DataFrame, betas: pd.Series, spx: pd.Series,
                 cost: float = 1e-4) -> pd.DataFrame:
    """Add the hedge to a backtest's daily P&L: columns hedge_price, hedge_cost,
    gross (with the hedge's price P&L) and net (with both)."""
    cal = daily.index
    cum = pd.DataFrame(0.0, index=cal, columns=["hedge_price", "hedge_cost"])
    for k, t in trades.iterrows():
        cum += hedge_path(t, betas.loc[k], spx, cal, cost)
    inc = cum.diff()
    inc.iloc[0] = cum.iloc[0]
    out = daily.copy()
    out["hedge_price"] = inc["hedge_price"]
    out["hedge_cost"] = inc["hedge_cost"]
    out["gross"] = daily["gross"] + inc["hedge_price"]
    out["net"] = daily["net"] + inc["hedge_price"] + inc["hedge_cost"]
    return out


# ---------------------------------------------------------------------------
# Stress periods
# ---------------------------------------------------------------------------
def stress_summary(daily: pd.DataFrame, start, end) -> dict:
    """Position, P&L and worst week of a daily P&L table over [start, end]."""
    d = daily.loc[start:end]
    weekly = d["net"].resample("W-FRI").sum()
    worst = weekly.idxmin() if len(weekly) else pd.NaT
    return {
        "in_position_share": float(d["in_position"].mean()) if len(d) else np.nan,
        "gross_pnl": float(d["gross"].sum()),
        "net_pnl": float(d["net"].sum()),
        "worst_week_net": float(weekly.min()) if len(weekly) else np.nan,
        "worst_week_end": worst,
    }
