"""Download and local cache of the data (yfinance, FRED).

Limitations to document in the README:
- yfinance is an unofficial Yahoo Finance API (no quality guarantee).
- Closing prices are UNADJUSTED (auto_adjust=False): ex-dividend dates differ between
  Taipei and the ADR, and Yahoo's retroactive adjustment would distort the premium level.
- The exact timestamp of the TWD=X "Close" is not documented by Yahoo: check it against
  an official fixing (e.g. Taipei Forex) before drawing fine conclusions.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

TICKERS = {"adr": "TSM", "local": "2330.TW", "fx": "TWD=X"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTAUS"


def download(start: str = "2010-01-01", end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame (columns adr, local, fx) of unadjusted closes, cached in CSV."""
    cache = DATA_DIR / "raw_closes.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    out = {}
    for key, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {ticker}")
        close = df["Close"]
        out[key] = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    # outer: each market keeps its own trading days, the alignment is done later.
    raw = pd.concat(out, axis=1, join="outer").sort_index()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(cache)
    return raw


def read_fred_csv(source) -> pd.Series:
    """Parse a FRED CSV (date, value) into a float series; "." and blanks become NaN."""
    df = pd.read_csv(source, na_values=["."], index_col=0, parse_dates=True)
    fx = pd.to_numeric(df.iloc[:, 0]).astype(float).sort_index()
    fx.index.name = "date"
    fx.name = "fx_fred"
    return fx


def download_fred_fx(refresh: bool = False) -> pd.Series:
    """Return DEXTAUS (Fed H.10, TWD per USD, noon New York), cached in CSV."""
    cache = DATA_DIR / "fx_fred.csv"
    if cache.exists() and not refresh:
        return read_fred_csv(cache)

    with urlopen(FRED_URL, timeout=60) as response:
        payload = response.read()
    fx = read_fred_csv(BytesIO(payload))  # parse first: a bad response must not be cached
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_bytes(payload)
    return fx
