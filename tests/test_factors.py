import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.factors import build_panel, check_alignment, identity_error, ols_newey_west
from adr_premium.premium import premium_prev_ny


def _world(seed=0, n=400):
    """Simulated NY and Taipei prices, FX and market data with different calendars."""
    rng = np.random.default_rng(seed)
    ny = pd.bdate_range("2020-01-06", periods=n)
    taipei = ny.delete(np.arange(30, n, 47))  # Taipei holidays differ from New York's
    adr = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=ny)
    local = pd.Series(600 * np.exp(np.cumsum(rng.normal(0, 0.02, len(taipei)))), index=taipei)
    fx = pd.Series(30 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=ny)
    market = pd.DataFrame({
        "spx": pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=ny),
        "ndx": pd.Series(15000 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=ny),
        "adr_volume": pd.Series(rng.uniform(5e6, 1e7, n), index=ny),
        "local_volume": pd.Series(rng.uniform(2e7, 4e7, len(taipei)), index=taipei),
    })
    return adr, local, fx, market


def _panel(seed=0):
    adr, local, fx, market = _world(seed)
    return build_panel(premium_prev_ny(adr, local, fx), market, adr), market


def test_weekly_decomposition_is_exact():
    panel, _ = _panel(1)
    assert identity_error(panel) < 1e-12
    p = panel.dropna(subset=["dq"])
    np.testing.assert_allclose(p["dq"], p["r_adr"] - p["r_local_usd"], rtol=0, atol=1e-12)


def test_regression_recovers_known_beta():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=2000))
    y = 0.01 + 0.5 * x + rng.normal(scale=1.0, size=2000)
    result = ols_newey_west(y, x.to_frame("x"))
    row = result["table"].loc["x"]
    # The sd of the estimate is about 1 / sqrt(2000) = 0.022: 0.5 must lie in the 95% CI.
    assert row["ci_low"] < 0.5 < row["ci_high"]
    assert row["coef"] == pytest.approx(0.5, abs=0.07)
    assert result["n"] == 2000 and result["lags"] >= 1


def test_newey_west_se_exceeds_classical_se_with_autocorrelation():
    # Positively autocorrelated regressor and residuals: the classical se is too small.
    rng = np.random.default_rng(3)
    n = 3000

    def ar1(phi):
        e = rng.normal(size=n)
        out = np.empty(n)
        out[0] = e[0]
        for t in range(1, n):
            out[t] = phi * out[t - 1] + e[t]
        return out

    x = pd.Series(ar1(0.8))
    y = 0.3 * x + pd.Series(ar1(0.8))
    row = ols_newey_west(y, x.to_frame("x"), lags=12)["table"].loc["x"]
    assert row["se"] > 1.5 * row["se_classic"]


def test_us_factors_are_read_on_the_adr_date():
    panel, market = _panel(4)
    check_alignment(panel, market)
    rows = panel.dropna(subset=["adr_date"])
    assert (rows["sp500_date"] == rows["adr_date"]).all()
    assert (rows["sp500_date"] < rows["date"]).all()  # prev_ny: always before the Taipei date


def test_no_factor_value_comes_from_after_the_adr_date():
    # Each index level encodes its own date: any look-ahead would show a later date.
    adr, local, fx, market = _world(5)
    market["spx"] = market.index.map(pd.Timestamp.toordinal).astype(float)
    market.loc[market.index[100:110], "spx"] = np.nan  # a gap must be filled from the past only
    panel = build_panel(premium_prev_ny(adr, local, fx), market, adr)
    rows = panel.dropna(subset=["ln_sp500"])
    read = np.exp(rows["ln_sp500"]).round().astype(int)
    assert (read <= rows["adr_date"].map(pd.Timestamp.toordinal)).all()


def test_alignment_check_detects_a_factor_date_shifted_into_the_future():
    panel, market = _panel(6)
    shifted = panel.copy()
    row = shifted["adr_date"].first_valid_index()
    shifted.loc[row, "sp500_date"] = shifted.loc[row, "adr_date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="after the ADR date"):
        check_alignment(shifted, market)


def test_alignment_check_detects_a_factor_value_taken_from_a_later_day():
    panel, market = _panel(7)
    shifted = panel.copy()
    later = np.log(market["spx"]).shift(-1)  # the next session's level, a future value
    known = shifted["sp500_date"].notna()
    shifted.loc[known, "ln_sp500"] = later.reindex(shifted.loc[known, "sp500_date"]).to_numpy()
    with pytest.raises(ValueError, match="do not match"):
        check_alignment(shifted, market)


def test_pair_trade_beta_is_minus_the_premium_beta():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from factors import pair_returns

    panel, _ = _panel(8)
    pair = pair_returns(panel)
    beta_dq = ols_newey_west(panel["dq"], panel[["r_sp500"]])["table"].loc["r_sp500", "coef"]
    beta_log = ols_newey_west(pair["pair_log"], pair[["r_sp500"]])["table"].loc["r_sp500", "coef"]
    beta_simple = ols_newey_west(pair["pair_simple"], pair[["r_sp500"]])["table"].loc["r_sp500", "coef"]
    assert beta_log == pytest.approx(-beta_dq, abs=1e-12)
    assert beta_simple == pytest.approx(-beta_dq, abs=0.02)  # simple and log returns differ slightly
