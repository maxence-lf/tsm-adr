import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from adr_premium.stats import level_shift
from simulate_level_shift import calibration, simulate


@pytest.mark.parametrize("half_life", [32, 70])
def test_calibration_matches_targets(half_life):
    # Targets close to the real prev_ny series: daily change std 1.97, autocorrelation -0.45.
    cal = calibration(half_life, change_std=1.97, change_autocorr=-0.45)
    assert cal["phi"] ** half_life == pytest.approx(0.5)
    series = simulate(pd.bdate_range("2000-01-03", periods=40_000), cal, seed=1)
    changes = series.diff()
    assert changes.std() == pytest.approx(1.97, rel=0.03)
    assert changes.autocorr() == pytest.approx(-0.45, abs=0.02)


def test_level_shift_compares_last_and_first_year():
    weeks = pd.date_range("2020-01-03", periods=156, freq="W-FRI")
    x = pd.Series(np.r_[np.zeros(52), np.ones(52), np.full(52, 3.0)], index=weeks)
    assert level_shift(x, weeks[0], weeks[-1]) == pytest.approx(3.0, abs=0.1)
    assert level_shift(pd.Series(1.0, index=weeks), weeks[0], weeks[-1]) == 0
