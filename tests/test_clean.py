import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.clean import clean_fx, flag_fx


def _fx(values):
    return pd.Series(values, index=pd.date_range("2026-03-02", periods=len(values), freq="B"))


def test_impossible_value_is_removed_and_reported():
    fx = _fx([30.0, 30.1, 1.8, 30.2, 30.0])
    cleaned, removed = clean_fx(fx)
    assert fx.index[2] not in cleaned.index
    assert list(removed.columns) == ["date", "value", "reason"]
    assert len(removed) == 1
    assert removed.loc[0, "date"] == fx.index[2]
    assert removed.loc[0, "value"] == 1.8
    assert "outside" in removed.loc[0, "reason"]


def test_normal_values_are_kept_unchanged():
    fx = _fx([27.5, 30.123456, 33.9, 25.0, 40.0])
    cleaned, removed = clean_fx(fx)
    pd.testing.assert_series_equal(cleaned, fx)
    assert removed.empty


def test_missing_values_are_not_reported_as_removed():
    fx = _fx([30.0, np.nan, 30.1])
    cleaned, removed = clean_fx(fx)
    assert removed.empty
    pd.testing.assert_series_equal(cleaned, fx)


def test_three_percent_deviation_is_flagged_but_not_removed():
    fx = _fx([30.0, 30.0, 30.9, 30.0, 30.0])
    cleaned, removed = clean_fx(fx)
    pd.testing.assert_series_equal(cleaned, fx)
    assert removed.empty

    flagged = flag_fx(cleaned)
    assert list(flagged["date"]) == [fx.index[2]]
    assert flagged.loc[0, "value"] == 30.9
    assert flagged.loc[0, "median"] == 30.0
    assert flagged.loc[0, "deviation"] == pytest.approx(0.03)


def test_small_moves_are_not_flagged():
    assert flag_fx(_fx([30.0, 30.3, 30.1, 30.4, 30.2])).empty
