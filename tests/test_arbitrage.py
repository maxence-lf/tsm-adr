import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.arbitrage import breakeven_deviation, deviations, round_trip_cost, share_above
from adr_premium.backtest import Costs


def test_round_trip_cost_by_hand():
    # 2 x 5 + 2 x 10 + 30 + 2 x 5 = 70 bp with the default costs.
    assert round_trip_cost(Costs()) == pytest.approx(0.0070)
    inst = Costs(adr_commission=2e-4, local_commission=3e-4, fx_conversion=2e-4, adr_borrow_fee=0.0025)
    # 2 x 2 + 2 x 3 + 30 + 2 x 2 = 44 bp.
    assert round_trip_cost(inst) == pytest.approx(0.0044)


def test_breakeven_deviation_by_hand():
    # T = h: half the deviation is captured, so D* = 2 (C + b h / 252).
    assert breakeven_deviation(0.007, 0.005, 50, 50) == pytest.approx(2 * (0.007 + 0.005 * 50 / 252))
    # T = 2h: three quarters are captured.
    assert breakeven_deviation(0.007, 0.005, 50, 100) == pytest.approx((0.007 + 0.005 * 100 / 252) / 0.75)
    # Without costs nothing is needed; a slower reversion needs a larger deviation.
    assert breakeven_deviation(0.0, 0.0, 50, 50) == 0.0
    assert breakeven_deviation(0.007, 0.005, 100, 50) > breakeven_deviation(0.007, 0.005, 50, 50)


def test_deviations_use_previous_weeks_only():
    dates = pd.bdate_range("2020-01-06", periods=300)
    prem = pd.Series(np.random.default_rng(0).normal(0.05, 0.02, 300), index=dates)
    dev = deviations(prem, window=4)
    q = np.log1p(prem)
    weekly = q.groupby(q.index.to_period("W-FRI")).tail(1)
    k = 10
    assert dev.loc[weekly.index[k]] == pytest.approx(weekly.iloc[k] - weekly.iloc[k - 4:k].mean())


def test_share_above():
    dev = pd.Series([-0.03, -0.01, 0.0, 0.01, 0.02, 0.04])
    out = share_above(dev, 0.015)
    assert out["share_above"] == pytest.approx(2 / 6)
    assert out["share_abs_above"] == pytest.approx(3 / 6)
