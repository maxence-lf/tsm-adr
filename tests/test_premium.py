import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adr_premium.premium import premium_prev_ny, premium_same_day


def _s(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))


def test_zero_premium_when_prices_consistent():
    d = ["2026-03-02", "2026-03-03"]
    fx = _s(d, [32.0, 32.0])
    local = _s(d, [1000.0, 1100.0])
    adr = 5 * local / fx  # parité exacte
    out = premium_same_day(adr, local, fx)
    assert np.allclose(out["premium"], 0.0)


def test_known_premium_value():
    # 5 * 1000 / 32 = 156.25 USD ; ADR à 187.5 -> prime de 20 %
    d = ["2026-03-02"]
    out = premium_same_day(_s(d, [187.5]), _s(d, [1000.0]), _s(d, [32.0]))
    assert out["premium"].iloc[0] == pytest.approx(0.20)


def test_prev_ny_uses_strictly_previous_ny_date():
    # Lundi Taipei doit utiliser la clôture NY du vendredi précédent, pas celle du lundi.
    local = _s(["2026-03-06", "2026-03-09"], [1000.0, 1000.0])          # ven, lun
    fx = _s(["2026-03-06", "2026-03-09"], [32.0, 32.0])
    adr = _s(["2026-03-05", "2026-03-06", "2026-03-09"], [150.0, 160.0, 999.0])
    out = premium_prev_ny(adr, local, fx)
    assert out.loc["2026-03-06", "adr"] == 150.0   # jeu.
    assert out.loc["2026-03-09", "adr"] == 160.0   # ven., jamais 999 (même jour)


def test_prev_ny_drops_long_gaps():
    # Pas de cotation NY pendant > 5 jours : la date Taipei doit être exclue.
    local = _s(["2026-03-20"], [1000.0])
    fx = _s(["2026-03-20"], [32.0])
    adr = _s(["2026-03-02"], [150.0])
    assert premium_prev_ny(adr, local, fx).empty


def test_same_day_ignores_non_common_holidays():
    local = _s(["2026-03-02", "2026-03-03"], [1000.0, 1000.0])
    fx = _s(["2026-03-02", "2026-03-03"], [32.0, 32.0])
    adr = _s(["2026-03-02"], [160.0])  # NY fermé le 03
    assert list(premium_same_day(adr, local, fx).index) == [pd.Timestamp("2026-03-02")]
