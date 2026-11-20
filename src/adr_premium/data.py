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

from adr_premium.securities import TSMC, Security

TICKERS = {"adr": "TSM", "local": "2330.TW", "fx": "TWD=X"}  # TSMC, kept for reference
# Market factors (closes) and volumes for the factor analysis, kept in a separate cache
# so that raw_closes.csv, used by every earlier result, is never rewritten.
MARKET_CLOSES = {"spx": "^GSPC", "ndx": "^NDX"}
MARKET_VOLUMES = {"adr_volume": "TSM", "local_volume": "2330.TW"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTAUS"


def _tickers(security: Security) -> dict[str, str]:
    return {"adr": security.adr, "local": security.local, "fx": "TWD=X"}


def download(start: str | None = None, end: str | None = None, refresh: bool = False,
             security: Security = TSMC) -> pd.DataFrame:
    """Return a DataFrame (columns adr, local, fx) of unadjusted closes, cached in CSV.

    ``start`` and ``end`` default to the security's sample (end exclusive, as in
    yfinance). TSMC uses the original cache file, raw_closes.csv.
    """
    cache = DATA_DIR / f"raw_closes{security.cache_suffix}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    start = security.start if start is None else start
    end = security.download_end if end is None else end

    import yfinance as yf

    out = {}
    for key, ticker in _tickers(security).items():
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


def _yahoo_column(ticker: str, column: str, start: str, end: str | None) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    out = df[column]
    return out.iloc[:, 0] if isinstance(out, pd.DataFrame) else out


def download_market(start: str = "2010-01-01", end: str | None = None, refresh: bool = False,
                    security: Security = TSMC) -> pd.DataFrame:
    """S&P 500 and Nasdaq-100 closes, ADR and local volumes, cached in CSV.

    Volumes are in ADRs and in local shares. A zero volume is a missing value (no
    trading reported), not an actual zero, so it is stored as NaN. For a security other
    than TSMC, the index closes are taken from the TSMC cache, so that every security
    uses the same index data, and only its volumes are downloaded.
    """
    cache = DATA_DIR / f"market_extra{security.cache_suffix}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    volumes = {"adr_volume": security.adr, "local_volume": security.local}
    if security.key == TSMC.key:
        out = {key: _yahoo_column(t, "Close", start, end) for key, t in MARKET_CLOSES.items()}
    else:
        index_closes = download_market()[list(MARKET_CLOSES)]
        out = {key: index_closes[key].dropna() for key in MARKET_CLOSES}
        start, end = security.start, security.download_end
    out |= {key: _yahoo_column(t, "Volume", start, end) for key, t in volumes.items()}
    market = pd.concat(out, axis=1, join="outer").sort_index()
    market.index = pd.DatetimeIndex(market.index).tz_localize(None)
    for key in MARKET_VOLUMES:
        market[key] = market[key].where(market[key] > 0)
    DATA_DIR.mkdir(exist_ok=True)
    market.to_csv(cache)
    return market


def _dividend_series(ticker: str) -> pd.Series:
    """Cash dividends per share from yfinance, indexed by ex-date in the exchange's own time zone."""
    import yfinance as yf

    div = yf.Ticker(ticker).dividends
    if div.empty:
        raise RuntimeError(f"No dividends for {ticker}")
    # The timestamps are local to each exchange (09:30 New York, 09:00 Taipei): dropping
    # the time zone keeps the local calendar date of the ex-date.
    div.index = pd.DatetimeIndex(div.index).tz_localize(None).normalize()
    return div.astype(float)


def download_dividends(refresh: bool = False, security: Security = TSMC) -> pd.DataFrame:
    """Ex-date cash dividends (columns adr in USD per ADR, local in TWD per share), cached in CSV.

    Amounts are as reported by Yahoo (gross, before any withholding tax); a missing value
    means no ex-date on that day for that line.
    """
    cache = DATA_DIR / f"dividends{security.cache_suffix}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    out = {key: _dividend_series(_tickers(security)[key]) for key in ("adr", "local")}
    div = pd.concat(out, axis=1, join="outer").sort_index()
    div.index.name = "ex_date"
    DATA_DIR.mkdir(exist_ok=True)
    div.to_csv(cache)
    return div
