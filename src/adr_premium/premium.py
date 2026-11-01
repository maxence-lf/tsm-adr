"""Construction de la prime ADR / action locale.

Convention : prime = prix ADR (USD) / (ratio * prix local (TWD) / USDTWD) - 1

Problème central : Taipei clôture à 13:30 heure de Taipei (05:30 UTC),
New York à 16:00 heure de New York (20:00 ou 21:00 UTC selon l'heure d'été).
Les deux clôtures d'une même date calendaire sont séparées d'environ 15 h :
la clôture new-yorkaise intègre de l'information que Taipei n'a pas encore vue.

Deux définitions sont donc calculées :
- ``same_day``   : clôture NY du jour d  vs clôture Taipei du jour d  (écart ~15 h,
                   NY "en avance" sur Taipei)
- ``prev_ny``    : clôture NY du dernier jour de bourse NY strictement antérieur à d
                   vs clôture Taipei du jour d (écart ~8,5 h, Taipei "en avance")

Aucune des deux n'est synchrone. Comparer leurs propriétés (volatilité des
variations quotidiennes, autocorrélation) mesure le bruit dû à l'asynchronie.
"""
from __future__ import annotations

import pandas as pd

ADR_RATIO = 5  # 1 ADR TSM = 5 actions ordinaires 2330.TW


def _as_series(s: pd.Series, name: str) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = name
    return s


def premium_same_day(adr_usd: pd.Series, local_twd: pd.Series, usdtwd: pd.Series,
                     ratio: int = ADR_RATIO) -> pd.DataFrame:
    """Prime avec les clôtures de la même date calendaire (jours communs uniquement)."""
    df = pd.concat(
        [_as_series(adr_usd, "adr"), _as_series(local_twd, "local"), _as_series(usdtwd, "fx")],
        axis=1, join="inner",
    )
    df["local_in_usd"] = ratio * df["local"] / df["fx"]
    df["premium"] = df["adr"] / df["local_in_usd"] - 1
    return df


def premium_prev_ny(adr_usd: pd.Series, local_twd: pd.Series, usdtwd: pd.Series,
                    ratio: int = ADR_RATIO) -> pd.DataFrame:
    """Prime avec, pour chaque date Taipei d, la dernière clôture NY strictement avant d.

    L'ADR utilisé est indexé par la date Taipei. La date NY effectivement utilisée
    est conservée dans la colonne ``adr_date`` pour audit.
    """
    local = _as_series(local_twd, "local").to_frame()
    fx = _as_series(usdtwd, "fx")
    adr = _as_series(adr_usd, "adr").to_frame()
    adr["adr_date"] = adr.index

    # Décaler d'un jour garantit "strictement antérieur" lors du merge_asof.
    adr_shifted = adr.copy()
    adr_shifted.index = adr_shifted.index + pd.Timedelta(days=1)

    df = pd.merge_asof(
        local.reset_index(names="date"),
        adr_shifted.reset_index(names="date"),
        on="date", direction="backward",
    ).set_index("date")
    df = df.join(fx, how="inner").dropna(subset=["adr", "local", "fx"])

    # Écart max toléré entre date Taipei et date NY utilisée (week-ends, fériés).
    gap = (df.index - pd.DatetimeIndex(df["adr_date"])).days
    df["ny_gap_days"] = gap
    df = df[df["ny_gap_days"] <= 5]

    df["local_in_usd"] = ratio * df["local"] / df["fx"]
    df["premium"] = df["adr"] / df["local_in_usd"] - 1
    return df


def asynchrony_diagnostics(same_day: pd.DataFrame, prev_ny: pd.DataFrame) -> pd.DataFrame:
    """Compare les deux définitions : niveau, bruit des variations, autocorrélation."""
    rows = {}
    for name, df in {"same_day": same_day, "prev_ny": prev_ny}.items():
        p = df["premium"]
        dp = p.diff().dropna()
        rows[name] = {
            "n_obs": len(p),
            "premium_mean": p.mean(),
            "premium_std": p.std(),
            "daily_change_std": dp.std(),
            # Autocorrélation négative des variations = signe de bruit de mesure
            # (un écart dû à l'asynchronie se corrige le lendemain).
            "daily_change_autocorr_lag1": dp.autocorr(lag=1),
        }
    return pd.DataFrame(rows).T
