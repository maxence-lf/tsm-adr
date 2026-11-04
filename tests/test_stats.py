import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.stats import (_half_life, adf_kpss, ar1_half_life, arma11_half_life, chow_test,
                               to_weekly, to_weekly_mean)


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)  # start from the stationary distribution
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _daily(values):
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))


def test_half_life_of_ar1():
    # True half-life: ln(0.5) / ln(0.95) = 13.51. With n = 4000 the sd of phi is
    # sqrt((1 - 0.95**2) / 4000) = 0.0049, and d(half-life)/d(phi) = 277 at 0.95,
    # so the sd of the estimate is about 1.37 days (1.36 over 200 simulations).
    # A tolerance of 3 sd keeps the test meaningful without being fragile.
    out = ar1_half_life(_daily(_ar1(0.95, 4000, seed=1)))
    assert out["half_life"] == pytest.approx(np.log(0.5) / np.log(0.95), abs=4.1)
    assert out["half_life_ci_low"] < out["half_life"] < out["half_life_ci_high"]
    assert out["n"] == 3999


def test_unit_root_gives_infinite_half_life_without_error():
    trend = _daily(np.arange(50.0) ** 1.5)  # explosive: phi > 1
    out = ar1_half_life(trend)
    assert out["phi"] >= 1
    assert np.isinf(out["half_life"])


def test_adf_does_not_reject_on_random_walk():
    walk = _daily(np.cumsum(np.random.default_rng(2).normal(size=4000)))
    assert adf_kpss(walk)["adf_pvalue"] > 0.05


def test_adf_and_kpss_agree_on_stationary_series():
    out = adf_kpss(_daily(_ar1(0.5, 2000, seed=3)))
    assert out["adf_pvalue"] < 0.05
    assert out["kpss_pvalue"] > 0.05


def test_measurement_noise_shortens_daily_half_life():
    # The latent premium is an AR(1) with phi = 0.98 (half-life 34.3 days). White noise
    # with sd 2 (innovation sd 1) gives daily changes an autocorrelation near -0.44,
    # the value measured on the real premium. The noise biases phi down: the daily
    # estimate is far too short, and the weekly one, where the noise weighs less
    # against five days of true moves, is closer to the truth.
    true = np.log(0.5) / np.log(0.98)
    n = 8000
    latent = _ar1(0.98, n, seed=4)
    observed = _daily(latent + np.random.default_rng(5).normal(scale=2.0, size=n))

    daily = ar1_half_life(observed)["half_life"]
    weekly_in_days = ar1_half_life(to_weekly(observed))["half_life"] * 5

    assert observed.diff().autocorr() == pytest.approx(-0.44, abs=0.05)
    assert daily < 0.5 * true
    assert abs(weekly_in_days - true) < abs(daily - true)


def _noisy_ar1(latent_seed, noise_seed, n=8000):
    """Same set-up as the bias test: AR(1) phi = 0.98 plus white noise with sd 2."""
    latent = _ar1(0.98, n, seed=latent_seed)
    return _daily(latent + np.random.default_rng(noise_seed).normal(scale=2.0, size=n))


# (4, 5) are the seeds of the bias test; the others come from the calibration run.
# Over 20 seed pairs, the ARMA half-life had a mean of 36.2 days (sd 4.7) daily and
# 35.1 days (sd 5.0) weekly, against a true 34.3 days: a tolerance of 15 days is
# 3 sd. The AR(1) stayed at 3 to 5 days daily and 11 to 17 days weekly, always at
# least 17 days from the truth: inside the band for the ARMA, outside for the AR(1).
@pytest.mark.parametrize("latent_seed, noise_seed", [(4, 5), (102, 103), (104, 105)])
@pytest.mark.parametrize("frequency, days_per_step", [("daily", 1), ("weekly", 5)])
def test_arma_recovers_half_life_under_measurement_noise(latent_seed, noise_seed, frequency, days_per_step):
    true = np.log(0.5) / np.log(0.98)
    observed = _noisy_ar1(latent_seed, noise_seed)
    x = observed if frequency == "daily" else to_weekly(observed)

    arma = arma11_half_life(x)
    arma_days = arma["half_life"] * days_per_step
    ar1_days = ar1_half_life(x)["half_life"] * days_per_step

    assert arma["converged"]
    assert arma["theta"] < 0
    assert arma_days == pytest.approx(true, abs=15)
    assert ar1_days != pytest.approx(true, abs=15)
    assert abs(arma_days - true) < abs(ar1_days - true)
    assert arma["half_life_ci_low"] * days_per_step < true < arma["half_life_ci_high"] * days_per_step


def test_arma_whitens_residuals_that_ar1_leaves_correlated():
    # On a correctly specified model the Ljung-Box p-value is uniform, so p > 0.05 on
    # a single seed would fail 5% of the time by design (these seeds give p = 0.01).
    # The test checks instead that the AR(1) is rejected massively and the ARMA is not.
    observed = _noisy_ar1(4, 5)
    ar1 = ar1_half_life(observed)
    arma = arma11_half_life(observed)
    assert ar1["resid_autocorr"] < -0.3
    assert ar1["ljungbox_pvalue"] < 1e-10
    assert abs(arma["resid_autocorr"]) < 0.05
    assert arma["ljungbox_pvalue"] > 1e-3


def test_arma_handles_empty_weeks():
    weekly = to_weekly(_noisy_ar1(4, 5))
    holed = weekly.copy()
    holed.iloc[[100, 500, 900]] = np.nan
    out = arma11_half_life(holed)
    assert out["n"] == weekly.notna().sum() - 3
    assert np.isfinite(out["half_life"])


def test_half_life_is_infinite_without_mean_reversion():
    assert np.isinf(_half_life(1.0))
    assert np.isinf(_half_life(1.02))
    walk = _daily(np.cumsum(np.random.default_rng(10).normal(size=1000)))
    out = arma11_half_life(walk)  # must not raise near a unit root
    assert out["phi"] > 0.95
    assert np.isinf(out["half_life_ci_high"]) or out["half_life_ci_high"] > 100


def test_weekly_mean_averages_and_does_not_fill():
    s = pd.Series([1.0, 3.0, 5.0], index=pd.to_datetime(["2026-03-02", "2026-03-05", "2026-03-16"]))
    weekly = to_weekly_mean(s)
    assert weekly.iloc[0] == 2.0
    assert np.isnan(weekly.iloc[1])
    assert weekly.iloc[2] == 5.0


def test_weekly_keeps_last_value_and_does_not_fill():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-03-02", "2026-03-05", "2026-03-16"]))
    weekly = to_weekly(s)
    assert list(weekly.index) == list(pd.to_datetime(["2026-03-06", "2026-03-13", "2026-03-20"]))
    assert weekly.iloc[0] == 2.0  # Thursday, the last value of that week
    assert np.isnan(weekly.iloc[1])  # empty week stays empty
    assert weekly.iloc[2] == 3.0


def test_missing_week_does_not_create_a_two_week_change():
    values = pd.Series(_ar1(0.9, 300, seed=6), index=pd.date_range("2000-01-07", periods=300, freq="W-FRI"))
    holed = values.copy()
    holed.iloc[100] = np.nan
    assert ar1_half_life(holed)["n"] == ar1_half_life(values)["n"] - 2


def test_chow_detects_mean_shift():
    x = _ar1(0.5, 600, seed=7)
    x[300:] += 2.0
    s = _daily(x)
    out = chow_test(s, s.index[300])
    assert out["chow_pvalue"] < 0.01
    assert (out["n_before"], out["n_after"]) == (299, 300)


def test_chow_no_break_on_homogeneous_series():
    s = _daily(_ar1(0.5, 600, seed=8))
    assert chow_test(s, s.index[300])["chow_pvalue"] > 0.05


def test_chow_returns_nan_when_a_regime_is_too_short():
    s = _daily(_ar1(0.5, 100, seed=9))
    out = chow_test(s, s.index[-2])
    assert out["n_after"] == 2
    assert np.isnan(out["chow_pvalue"])
