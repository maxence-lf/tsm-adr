import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.robustness import (block_bootstrap_sharpe, hedge_path, p_value_above,
                                    pair_beta_at, random_placements)


def _panel(n=400, beta=-0.2, seed=0):
    """Weekly rows with a pair return -dq = beta * r_sp500 + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    r = rng.normal(0, 0.02, n)
    pair = beta * r + rng.normal(0, 0.01, n)
    return pd.DataFrame({"date": dates, "dq": -pair, "r_sp500": r}, index=dates)


def test_beta_ignores_the_decision_week_and_later():
    panel = _panel()
    decision = panel["date"].iloc[300]
    beta = pair_beta_at(panel, decision)
    assert beta == pytest.approx(-0.2, abs=0.05)

    rng = np.random.default_rng(1)
    altered = panel.copy()
    later = altered["date"] >= decision  # the decision week and every later week
    altered.loc[later, "dq"] = rng.normal(0, 0.5, later.sum())
    altered.loc[later, "r_sp500"] = rng.normal(0, 0.5, later.sum())
    assert pair_beta_at(altered, decision) == beta


def test_beta_uses_the_last_156_weeks_only():
    panel = _panel()
    decision = panel["date"].iloc[300]
    altered = panel.copy()
    old = altered["date"] < panel["date"].iloc[300 - 156]
    altered.loc[old, "dq"] = 99.0  # older than the window: must not matter
    assert pair_beta_at(altered, decision) == pair_beta_at(panel, decision)


def test_beta_needs_enough_history():
    panel = _panel()
    assert np.isnan(pair_beta_at(panel, panel["date"].iloc[30]))


def test_hedge_by_hand():
    cal = pd.bdate_range("2021-01-04", periods=30)
    spx = pd.Series(100.0, index=cal)
    spx[cal >= cal[10]] = 110.0
    trade = {"adr_entry_date": cal[2], "adr_exit_date": cal[20]}
    path = hedge_path(trade, beta=-0.2, spx=spx, calendar=cal, cost=1e-4)
    end = path.iloc[-1]
    assert end["hedge_price"] == pytest.approx(0.2 * 0.1)  # long 0.2 USD of the index
    assert end["hedge_cost"] == pytest.approx(-1e-4 * 0.2 * (1 + 1.1))
    assert (path.loc[: cal[1]] == 0).all().all()
    nohedge = hedge_path(trade, beta=np.nan, spx=spx, calendar=cal)
    assert (nohedge == 0).all().all()


def test_random_placements_do_not_overlap():
    rng = np.random.default_rng(0)
    holdings = np.array([1, 5, 26, 3, 3, 10])
    for _ in range(500):
        pairs = sorted(random_placements(100, holdings, rng))
        assert sorted(j - i for i, j in pairs) == sorted(holdings)
        assert pairs[0][0] >= 0 and pairs[-1][1] < 100
        for (_, exit_), (entry, _) in zip(pairs, pairs[1:]):
            assert entry > exit_


def test_random_placements_cover_the_sample():
    rng = np.random.default_rng(1)
    starts = [min(p[0] for p in random_placements(60, np.array([2, 2]), rng)) for _ in range(3000)]
    assert min(starts) == 0 and max(starts) > 40


def test_bootstrap_sharpe_is_reproducible_and_centred():
    weekly = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 800))
    a = block_bootstrap_sharpe(weekly, n_draws=2000, seed=7)
    b = block_bootstrap_sharpe(weekly, n_draws=2000, seed=7)
    np.testing.assert_array_equal(a, b)
    point = weekly.mean() / weekly.std() * np.sqrt(52)
    lo, hi = np.percentile(a, [2.5, 97.5])
    assert lo < point < hi
    assert abs(np.median(a) - point) < 0.15


def test_p_value():
    assert p_value_above(np.arange(99), 98.5) == pytest.approx(1 / 100)
    assert p_value_above(np.arange(99), -1) == pytest.approx(1.0)
