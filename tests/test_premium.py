import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.clean import fill_fx
from adr_premium.premium import premium_prev_ny, premium_same_day


def _s(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))


def test_zero_premium_when_prices_consistent():
    d = ["2026-03-02", "2026-03-03"]
    fx = _s(d, [32.0, 32.0])
    local = _s(d, [1000.0, 1100.0])
    adr = 5 * local / fx  # exact parity
    out = premium_same_day(adr, local, fx)
    assert np.allclose(out["premium"], 0.0)


def test_known_premium_value():
    # 5 * 1000 / 32 = 156.25 USD; ADR at 187.5 -> 20% premium
    d = ["2026-03-02"]
    out = premium_same_day(_s(d, [187.5]), _s(d, [1000.0]), _s(d, [32.0]))
    assert out["premium"].iloc[0] == pytest.approx(0.20)


def test_prev_ny_uses_strictly_previous_ny_date():
    # Taipei's Monday must use the previous Friday's NY close, not Monday's.
    local = _s(["2026-03-06", "2026-03-09"], [1000.0, 1000.0])          # Fri, Mon
    fx = _s(["2026-03-05", "2026-03-06"], [32.0, 32.0])
    adr = _s(["2026-03-05", "2026-03-06", "2026-03-09"], [150.0, 160.0, 999.0])
    out = premium_prev_ny(adr, local, fx)
    assert out.loc["2026-03-06", "adr"] == 150.0   # Thu
    assert out.loc["2026-03-09", "adr"] == 160.0   # Fri, never 999 (same day)


def test_prev_ny_drops_long_gaps():
    # No NY quote for more than 5 days: the Taipei date must be excluded.
    local = _s(["2026-03-20"], [1000.0])
    fx = _s(["2026-03-02"], [32.0])
    adr = _s(["2026-03-02"], [150.0])
    assert premium_prev_ny(adr, local, fx).empty


def test_same_day_ignores_non_common_holidays():
    local = _s(["2026-03-02", "2026-03-03"], [1000.0, 1000.0])
    fx = _s(["2026-03-02", "2026-03-03"], [32.0, 32.0])
    adr = _s(["2026-03-02"], [160.0])  # NY closed on the 3rd
    assert list(premium_same_day(adr, local, fx).index) == [pd.Timestamp("2026-03-02")]


def test_prev_ny_takes_fx_on_adr_date():
    local = _s(["2026-03-09"], [1000.0])                                # Monday
    adr = _s(["2026-03-06"], [160.0])                                   # Friday
    fx = _s(["2026-03-06", "2026-03-09"], [31.0, 33.0])
    out = premium_prev_ny(adr, local, fx)
    assert out.loc["2026-03-09", "fx"] == 31.0
    assert out.loc["2026-03-09", "premium"] == pytest.approx(160.0 * 31.0 / 5000.0 - 1)


def test_prev_ny_can_keep_fx_on_taipei_date():
    local = _s(["2026-03-09"], [1000.0])
    adr = _s(["2026-03-06"], [160.0])
    fx = _s(["2026-03-06", "2026-03-09"], [31.0, 33.0])
    assert premium_prev_ny(adr, local, fx, fx_on_adr_date=False).loc["2026-03-09", "fx"] == 33.0


def test_prev_ny_drops_day_without_fx_on_adr_date():
    local = _s(["2026-03-09"], [1000.0])
    adr = _s(["2026-03-06"], [160.0])
    assert premium_prev_ny(adr, local, _s(["2026-03-09"], [33.0])).empty


def test_fx_is_never_later_than_the_adr_date():
    # Every rate after a given date is absurdly high: any look-ahead would show up.
    days = pd.bdate_range("2026-03-02", "2026-03-20")
    adr = pd.Series(160.0, index=days)
    local = pd.Series(1000.0, index=days)
    for cutoff in days[1:-1]:
        fx = pd.Series(32.0, index=days)
        fx[fx.index > cutoff] = 999.0
        fx[cutoff] = np.nan  # the rate of the cutoff date itself is missing and filled
        filled, _ = fill_fx(fx)

        same = premium_same_day(adr, local, filled)
        assert (same.loc[:cutoff, "fx"] == 32.0).all()

        prev = premium_prev_ny(adr, local, filled)
        assert (prev[prev["adr_date"] <= cutoff]["fx"] == 32.0).all()
        assert (prev["fx"] == filled.reindex(prev["adr_date"]).to_numpy()).all()
