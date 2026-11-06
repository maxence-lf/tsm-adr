import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.premium import premium_prev_ny, premium_same_day

# Convention for the factor analysis: q = ln(1 + premium). With FX in TWD per USD,
# 1 + premium = P_ADR * FX / (5 * P_local), so q = ln P_ADR + ln FX - ln P_local - ln 5
# and dq = r_ADR - r_local + d ln FX exactly, each term taken on the dates the premium uses.


def _prices(seed):
    rng = np.random.default_rng(seed)
    ny = pd.bdate_range("2026-01-05", periods=120)
    taipei = ny.delete([10, 11, 40])  # Taipei holidays differ from New York's
    adr = pd.Series(150 * np.exp(np.cumsum(rng.normal(0, 0.02, len(ny)))), index=ny)
    local = pd.Series(900 * np.exp(np.cumsum(rng.normal(0, 0.02, len(taipei)))), index=taipei)
    fx = pd.Series(31 * np.exp(np.cumsum(rng.normal(0, 0.003, len(ny)))), index=ny)
    return adr, local, fx


def _check(frame):
    q = np.log1p(frame["premium"])
    rhs = np.log(frame["adr"]).diff() - np.log(frame["local"]).diff() + np.log(frame["fx"]).diff()
    np.testing.assert_allclose(q.diff().dropna(), rhs.dropna(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(q, np.log(frame["adr"]) + np.log(frame["fx"]) - np.log(frame["local"]) - np.log(5),
                               rtol=0, atol=1e-12)


def test_log_premium_decomposition_prev_ny():
    _check(premium_prev_ny(*_prices(1)))


def test_log_premium_decomposition_same_day():
    _check(premium_same_day(*_prices(2)))


def test_log_premium_decomposition_holds_on_weekly_samples():
    frame = premium_prev_ny(*_prices(3))
    weekly = frame.groupby(frame.index.to_period("W-FRI")).tail(1)
    _check(weekly)
