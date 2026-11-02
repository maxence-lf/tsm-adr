"""Téléchargement et cache local des données (yfinance).

Limites à documenter dans le README :
- yfinance est une API non officielle de Yahoo Finance (pas de garantie de qualité).
- On utilise les prix de clôture NON ajustés (auto_adjust=False) : les dates de
  détachement de dividende diffèrent entre Taipei et l'ADR, et l'ajustement
  rétroactif de Yahoo déformerait le niveau de la prime.
- L'horodatage exact du "Close" de TWD=X n'est pas documenté par Yahoo : à vérifier
  contre un fixing officiel (ex. Taipei Forex) avant toute conclusion fine.
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
    """Renvoie un DataFrame (colonnes adr, local, fx) des clôtures non ajustées, mis en cache en CSV."""
    cache = DATA_DIR / "raw_closes.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    out = {}
    for key, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"Aucune donnée pour {ticker}")
        close = df["Close"]
        out[key] = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    # outer : chaque marché garde ses propres jours de bourse, l'alignement se fait ensuite.
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
