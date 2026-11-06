import sys
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium import data
from adr_premium.data import download_fred_fx, read_fred_csv


def test_fred_missing_values_become_nan():
    csv = "observation_date,DEXTAUS\n2026-03-02,32.10\n2026-03-03,.\n2026-03-04,\n2026-03-05,32.20\n"
    fx = read_fred_csv(StringIO(csv))
    assert len(fx) == 4
    assert fx.isna().sum() == 2
    assert np.isnan(fx.loc["2026-03-03"])
    assert np.isnan(fx.loc["2026-03-04"])
    assert fx.dtype == float
    assert fx.loc["2026-03-05"] == 32.20


def test_fred_dates_are_parsed_and_sorted():
    csv = "observation_date,DEXTAUS\n2026-03-03,32.2\n2026-03-02,32.1\n"
    fx = read_fred_csv(StringIO(csv))
    assert isinstance(fx.index, pd.DatetimeIndex)
    assert list(fx.index) == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")]


def test_fred_column_names_do_not_matter():
    # FRED has used both DATE and observation_date as the first header.
    fx = read_fred_csv(StringIO("DATE,VALUE\n2026-03-02,32.1\n"))
    assert fx.name == "fx_fred"
    assert fx.iloc[0] == 32.1


def test_fred_unexpected_text_is_rejected():
    with pytest.raises(ValueError):
        read_fred_csv(StringIO("observation_date,DEXTAUS\n2026-03-02,n/a value\n"))


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fred_bad_response_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: _Response(b"<html>error</html>"))
    with pytest.raises((ValueError, IndexError)):
        download_fred_fx(refresh=True)
    assert not (tmp_path / "fx_fred.csv").exists()


def test_fred_cache_is_written_then_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: _Response(b"observation_date,DEXTAUS\n2026-03-02,32.1\n"))
    assert download_fred_fx().iloc[0] == 32.1

    def fail(*a, **k):
        raise AssertionError("network must not be used when the cache exists")

    monkeypatch.setattr(data, "urlopen", fail)
    assert download_fred_fx().iloc[0] == 32.1


def test_market_cache_is_reused_and_zero_volume_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    days = pd.to_datetime(["2026-03-02", "2026-03-03"])
    values = {"Close": pd.Series([100.0, 101.0], index=days), "Volume": pd.Series([0.0, 5.0], index=days)}
    monkeypatch.setattr(data, "_yahoo_column", lambda ticker, column, start, end: values[column])
    market = data.download_market()
    assert list(market.columns) == ["spx", "ndx", "adr_volume", "local_volume"]
    assert np.isnan(market.loc["2026-03-02", "adr_volume"])
    assert market.loc["2026-03-03", "local_volume"] == 5.0

    def fail(*a, **k):
        raise AssertionError("network must not be used when the cache exists")

    monkeypatch.setattr(data, "_yahoo_column", fail)
    pd.testing.assert_frame_equal(data.download_market(), market, check_freq=False, check_names=False)
