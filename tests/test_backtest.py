import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.backtest import (Costs, Market, Rules, execution_dates, run_backtest,
                                  trade_path, weekly_signal)


def _world(seed=0, n=1500, phi=0.97, sigma_q=0.01, sigma_l=0.012):
    """Synchronous prices: ADR = local in USD * exp(q), same calendar, constant FX.

    With phi < 1, q is a daily AR(1) around 0.1. With phi = 1 it is a random walk; the
    drifts then make both prices martingales, so any bounded strategy has zero
    expected P&L.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n)
    eps_q = rng.normal(0, sigma_q, n)
    q = np.empty(n)
    q[0] = 0.1
    for t in range(1, n):
        if phi < 1:
            q[t] = 0.1 + phi * (q[t - 1] - 0.1) + eps_q[t]
        else:
            q[t] = q[t - 1] - sigma_q ** 2 / 2 + eps_q[t]
    log_l = np.cumsum(rng.normal(-sigma_l ** 2 / 2, sigma_l, n))
    local_usd = 100 * np.exp(log_l)
    fx = pd.Series(30.0, index=dates)
    local = pd.Series(local_usd * 30 / 5, index=dates)
    adr = pd.Series(local_usd * np.exp(q), index=dates)
    return adr, local, fx


# --- no look-ahead -----------------------------------------------------------
def test_signal_and_decisions_ignore_future_data():
    adr, local, fx = _world(1)
    cut = adr.index[900]
    full = run_backtest(adr, local, fx, rules=Rules(entry_z=1.0), costs=Costs.zero())

    # Replace everything after the cut with unrelated prices.
    rng = np.random.default_rng(99)
    after = adr.index > cut
    adr2, local2 = adr.copy(), local.copy()
    adr2[after] = adr2[after] * np.exp(rng.normal(0, 0.3, after.sum()))
    local2[after] = local2[after] * np.exp(rng.normal(0, 0.3, after.sum()))
    alt = run_backtest(adr2, local2, fx, rules=Rules(entry_z=1.0), costs=Costs.zero())

    known = full.signal.index <= cut
    pd.testing.assert_series_equal(full.signal.loc[known, "z"], alt.signal.loc[known, "z"])
    # Every decision taken on or before the cut is identical.
    a, b = (r.trades[r.trades["entry_decision"] <= cut] for r in (full, alt))
    assert len(a) > 3
    pd.testing.assert_series_equal(a["entry_decision"].reset_index(drop=True),
                                   b["entry_decision"].reset_index(drop=True))
    closed = a["exit_decision"] <= cut
    pd.testing.assert_series_equal(a.loc[closed, "exit_decision"].reset_index(drop=True),
                                   b.loc[closed, "exit_decision"].reset_index(drop=True))


def test_z_uses_only_previous_weeks():
    dates = pd.bdate_range("2020-01-06", periods=200)
    prem = pd.Series(np.random.default_rng(3).normal(0.1, 0.02, 200), index=dates)
    sig = weekly_signal(prem, window=5)
    q = np.log1p(prem)
    weekly = q.groupby(q.index.to_period("W-FRI")).tail(1)
    for k in range(5, len(weekly)):
        prev = weekly.iloc[k - 5:k]  # the five weeks strictly before week k
        expected = (weekly.iloc[k] - prev.mean()) / prev.std()
        assert sig["z"].iloc[k] == pytest.approx(expected)
    assert sig["z"].iloc[:5].isna().all()


# --- execution ---------------------------------------------------------------
def test_each_leg_executes_after_the_decision():
    ny = pd.bdate_range("2021-01-04", periods=60)
    tw = ny.delete([4, 5, 20])  # Taipei holidays
    ny = ny.delete([10, 30])  # NY holidays
    decisions = tw[::5]
    for lag in (0, 1):
        ex = execution_dates(decisions, ny, tw, extra_lag=lag).dropna()
        # NY closes ~15 h after Taipei on the same date: >= is strictly later in time.
        assert (ex["adr_exec"] >= ex.index).all()
        assert (ex["local_exec"] > ex.index).all()
        assert ex["adr_exec"].isin(ny).all() and ex["local_exec"].isin(tw).all()
    ex0 = execution_dates(decisions, ny, tw, 0).dropna()
    ex1 = execution_dates(decisions, ny, tw, 1).dropna()
    common = ex0.index.intersection(ex1.index)
    assert (ex1.loc[common, "adr_exec"] > ex0.loc[common, "adr_exec"]).all()
    assert (ex1.loc[common, "local_exec"] > ex0.loc[common, "local_exec"]).all()


def test_trades_execute_after_decisions_in_backtest():
    adr, local, fx = _world(2)
    res = run_backtest(adr, local, fx, rules=Rules(entry_z=1.0, extra_lag=1), costs=Costs())
    t = res.trades
    assert len(t) > 3
    assert (t["adr_entry_date"] >= t["entry_decision"]).all()
    assert (t["local_entry_date"] > t["entry_decision"]).all()
    assert (t["adr_exit_date"] >= t["exit_decision"]).all()
    assert (t["local_exit_date"] > t["exit_decision"]).all()
    assert (t["entry_decision"].iloc[1:].to_numpy() > t["exit_decision"].iloc[:-1].to_numpy()).all()


# --- the strategy on simulated premiums --------------------------------------
def test_wins_when_premium_mean_reverts():
    pnl = [run_backtest(*_world(s, phi=0.97), costs=Costs.zero()).trades["gross_pnl"].sum()
           for s in range(10)]
    assert all(p > 0 for p in pnl)


def test_zero_expected_gain_on_random_walk():
    per_trade = pd.concat([run_backtest(*_world(s, phi=1.0), costs=Costs.zero()).trades["gross_pnl"]
                           for s in range(30)])
    se = per_trade.std() / np.sqrt(len(per_trade))
    assert len(per_trade) > 100
    assert abs(per_trade.mean()) < 3 * se
    assert abs(per_trade.mean()) < 0.01


# --- costs -------------------------------------------------------------------
def _flat_market(dividends=None, local_exit=600.0):
    dates = pd.date_range("2021-01-04", "2022-01-14", freq="B")
    adr = pd.Series(100.0, index=dates)
    local = pd.Series(600.0, index=dates)  # 5 * 600 / 30 = 100 USD: no premium
    local[dates >= "2022-01-05"] = local_exit
    fx = pd.Series(30.0, index=dates)
    return Market.build(adr, local, fx, dividends)


def _trade():
    return pd.Series({"adr_entry_date": pd.Timestamp("2021-01-04"),
                      "local_entry_date": pd.Timestamp("2021-01-05"),
                      "adr_exit_date": pd.Timestamp("2022-01-04"),  # 365 days later
                      "local_exit_date": pd.Timestamp("2022-01-05")})


def test_costs_by_hand():
    costs = Costs()
    end = trade_path(_trade(), _flat_market(), costs).iloc[-1]
    assert end["price_fx"] == pytest.approx(0.0)
    assert end["adr_commission"] == pytest.approx(-2 * 5e-4)
    assert end["local_commission"] == pytest.approx(-2 * 10e-4)
    assert end["local_tax"] == pytest.approx(-30e-4)
    assert end["fx_conversion"] == pytest.approx(-2 * 5e-4)
    assert end["borrow_fee"] == pytest.approx(-0.005)  # one full year
    assert end.sum() == pytest.approx(-(0.001 + 0.002 + 0.003 + 0.001 + 0.005))


def test_dividends_by_hand():
    div = pd.DataFrame({"adr": [1.5, np.nan], "local": [np.nan, 10.0]},
                       index=pd.DatetimeIndex(["2021-06-15", "2021-06-16"]))
    end = trade_path(_trade(), _flat_market(div), Costs()).iloc[-1]
    shares = 30 / 600  # 1 USD at 30 TWD, bought at 600 TWD
    local_gross = shares * 10 / 30
    assert end["dividends_gross"] == pytest.approx(local_gross - 1.5 / 100)
    assert end["withholding_tax"] == pytest.approx(-0.21 * local_gross)


def test_dividend_on_entry_date_not_received():
    # The local share is bought on the close of its ex-date: no dividend.
    div = pd.DataFrame({"adr": [np.nan], "local": [10.0]}, index=pd.DatetimeIndex(["2021-01-05"]))
    end = trade_path(_trade(), _flat_market(div), Costs()).iloc[-1]
    assert end["dividends_gross"] == 0.0


def test_tax_only_on_local_sales():
    path = trade_path(_trade(), _flat_market(local_exit=660.0), Costs())
    before_exit = path.loc[: "2022-01-04"]
    assert (before_exit["local_tax"] == 0).all()  # buying the local share is not taxed
    exit_value = (30 / 600) * 660 / 30  # 1.1 USD
    assert path["local_tax"].iloc[-1] == pytest.approx(-30e-4 * exit_value)
    # Without the tax, nothing else changes.
    no_tax = trade_path(_trade(), _flat_market(local_exit=660.0),
                        Costs(local_sale_tax=0.0)).iloc[-1]
    assert no_tax["local_tax"] == 0.0
    assert no_tax.drop("local_tax").equals(path.iloc[-1].drop("local_tax"))
