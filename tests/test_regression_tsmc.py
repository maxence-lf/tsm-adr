"""Non-regression test: the TSMC results must not change when the code is refactored.

The reference values in tests/fixtures/tsmc_reference.json were computed from the
local data caches at commit 912658a (tag prereg-step6), before the code was made
configurable by security. The test recomputes them through the public API with its
TSMC defaults and requires them to be identical (to floating-point round-off).

The raw data is not redistributed, so the test is skipped when the caches are absent.
Regenerate the reference (only on purpose) with: python tests/test_regression_tsmc.py --write
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURE = ROOT / "tests" / "fixtures" / "tsmc_reference.json"
CACHES = ["raw_closes.csv", "fx_fred.csv", "market_extra.csv", "dividends.csv"]


def _series_summary(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(len(s)), "first_date": str(s.index[0].date()), "last_date": str(s.index[-1].date()),
            "sum": float(s.sum()), "sum_sq": float((s ** 2).sum()), "first": float(s.iloc[0]),
            "last": float(s.iloc[-1])}


def compute() -> dict:
    from adr_premium.backtest import Costs, Rules, metrics, run_backtest
    from adr_premium.clean import prepare_fred_fx
    from adr_premium.data import download, download_dividends, download_fred_fx, download_market
    from adr_premium.factors import build_panel, ols_newey_west
    from adr_premium.premium import asynchrony_diagnostics, premium_prev_ny, premium_same_day
    from adr_premium.stats import ar1_half_life, arma11_half_life, to_weekly

    raw = download()
    with contextlib.redirect_stdout(io.StringIO()):
        fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    prev = premium_prev_ny(raw["adr"], raw["local"], fx)
    same = premium_same_day(raw["adr"], raw["local"], fx)
    out = {"prev_ny": _series_summary(prev["premium"]), "same_day": _series_summary(same["premium"])}
    diag = asynchrony_diagnostics(same, prev)
    out["autocorr"] = {k: float(diag.loc[k, "daily_change_autocorr_lag1"]) for k in diag.index}

    q = np.log1p(prev["premium"])
    for name, x in (("daily", q), ("weekly", to_weekly(q))):
        out[f"ar1_{name}_phi"] = float(ar1_half_life(x)["phi"])
        out[f"arma_{name}_phi"] = float(arma11_half_life(x)["phi"])

    panel = build_panel(prev, download_market(), raw["adr"], "W-FRI")
    main = ols_newey_west(panel["dq"], panel[["r_sp500", "dln_fx"]])["table"]
    out["factor_main"] = {t: [float(main.loc[t, "coef"]), float(main.loc[t, "se"])] for t in main.index}
    p = panel[["r_adr", "r_local_usd", "r_sp500"]].dropna()
    out["beta_adr"] = float(ols_newey_west(p["r_adr"], p[["r_sp500"]])["table"].loc["r_sp500", "coef"])
    out["beta_local_usd"] = float(ols_newey_west(p["r_local_usd"], p[["r_sp500"]])["table"].loc["r_sp500", "coef"])

    res = run_backtest(raw["adr"], raw["local"], fx, download_dividends(), rules=Rules(), costs=Costs())
    t = res.trades
    out["backtest"] = {
        "n_trades": int(len(t)),
        "entries": [str(d.date()) for d in t["entry_decision"]],
        "exits": [str(d.date()) for d in t["exit_decision"]],
        "gross_pnl": [float(x) for x in t["gross_pnl"]],
        "net_pnl": [float(x) for x in t["net_pnl"]],
        "daily_net_sum_sq": float((res.daily["net"] ** 2).sum()),
        "metrics_net": {k: float(v) for k, v in metrics(res.daily, t, "net").items()},
    }
    return out


def _compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys differ"
        for k in expected:
            _compare(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12, nan_ok=True), path
    else:
        assert actual == expected, path


@pytest.mark.skipif(not all((ROOT / "data" / c).exists() for c in CACHES),
                    reason="local data caches not available")
def test_tsmc_results_unchanged():
    _compare(compute(), json.loads(FIXTURE.read_text()))


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(compute(), indent=1))
        print(f"written {FIXTURE}")
