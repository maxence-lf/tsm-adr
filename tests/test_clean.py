import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.clean import clean_fx, fill_fx, flag_fx, prepare_fred_fx


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


def test_fill_stops_after_three_business_days():
    fx = _fx([30.0, np.nan, np.nan, np.nan, np.nan, np.nan, 31.0])
    filled, dates = fill_fx(fx)
    assert list(filled.iloc[1:4]) == [30.0, 30.0, 30.0]
    assert filled.iloc[4:6].isna().all()
    assert filled.iloc[6] == 31.0
    assert list(dates) == list(fx.index[1:4])


def test_fill_counts_business_days_not_calendar_days():
    fx = pd.Series([30.0, np.nan, 31.0], index=pd.to_datetime(["2026-03-06", "2026-03-09", "2026-03-10"]))
    filled, dates = fill_fx(fx)
    assert list(filled.index) == list(fx.index)  # no weekend row
    assert filled.loc["2026-03-09"] == 30.0
    assert len(dates) == 1


def test_fill_only_uses_past_values():
    filled, _ = fill_fx(_fx([np.nan, 30.0, np.nan, 31.0]))
    assert filled.loc[filled.index[1]] == 30.0
    assert filled.iloc[1] == 30.0
    assert filled.index[0] == pd.Timestamp("2026-03-03")  # a leading gap has no past value


def test_fill_does_not_extend_past_the_last_value():
    filled, dates = fill_fx(_fx([30.0, 31.0, np.nan, np.nan]))
    assert len(filled) == 2
    assert len(dates) == 0


def test_prepare_fred_ignores_history_before_the_sample(capsys):
    fred = pd.Series([40.5, 30.0, 30.1], index=pd.to_datetime(["1985-01-02", "2026-03-02", "2026-03-03"]))
    fx, dates = prepare_fred_fx(fred, pd.Timestamp("2010-01-04"))
    assert "FX points removed: 0" in capsys.readouterr().out
    assert list(fx) == [30.0, 30.1]
    assert len(dates) == 0
