"""Generate every table of the interim report from the data in data/.

Each table is written as Markdown to report/tables/<name>.md. If the report exists,
the content between the markers <!-- table:<name> --> and <!-- /table:<name> --> is
replaced by the table, so no number in a report table is typed by hand.

Inputs: data/raw_closes.csv and data/fx_fred.csv (through the project loaders),
data/premium_same_day.csv, data/premium_prev_ny.csv, data/premium_table.csv and
data/mean_reversion_results.csv. Run build_premium.py, export_premium_table.py and
mean_reversion.py first.

Usage: python scripts/make_report_tables.py
"""
import contextlib
import functools
import io
import re
import sys
from datetime import datetime, time
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from adr_premium.clean import FX_MAX, FX_MIN, clean_fx, flag_fx, prepare_fred_fx
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import ADR_RATIO, asynchrony_diagnostics, premium_prev_ny, premium_same_day
from adr_premium.stats import ar1_half_life, arma11_half_life, to_weekly
from compare_fx import GAP_THRESHOLD, compare_flagged, persistent_gaps, relative_gap

DATA = ROOT / "data"
TABLES = ROOT / "report" / "tables"
REPORT = ROOT / "report" / "interim_report.md"

MANUAL_CHECK_DATE = "2026-06-23"
# Values reported by two news articles of 23 June 2026 (secondary sources):
# https://ts2.tech/en/tsmc-drops-6-key-investor-trend-cant-be-missed/
# https://stockti.com/tsmc-adr-premium-shrinks-sparking-6-slide-in-us-shares
# Their ADR price differs from the official close in our data and is not timestamped.
ARTICLE = {"adr": 439.78, "local": 2490.0, "fx": 31.625, "premium": 0.117, "premium_previous_close": 0.179}
EXAMPLE_START, EXAMPLE_END = "2025-04-07", "2025-04-16"
JUMP_THRESHOLD = 10.0  # percentage points of daily change
NOISE_PERIOD = ("2011", "2016")
SUBPERIODS = ["2010-2019", "2020-2023", "2024-Apr 2026"]
ROLLING_GROUPS = [("2013-2016", "2013", "2016"), ("2017-2019", "2017", "2019"),
                  ("2020-2023", "2020", "2023"), ("2024-2026", "2024", "2026")]
# Simulation of the bias: same set-up as the calibration of the tests.
SIM_PHI, SIM_N, SIM_NOISE, SIM_SEEDS = 0.98, 8000, 2.0, range(30)

KEY = []  # (quantity, value, source) quoted in the text of the report


def key(quantity: str, value: str, source: str) -> str:
    KEY.append((quantity, value, source))
    return value


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "n/a"
    if np.isinf(x):
        return "inf"
    return f"{x:,.{digits}f}"


def pval(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def md(df: pd.DataFrame, align: list[str] | None = None) -> str:
    """Markdown table; ``align`` holds 'l' or 'r' for each column."""
    cols = list(df.columns)
    align = align or ["l"] + ["r"] * (len(cols) - 1)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- inputs

def load() -> dict:
    raw = download()
    fred = download_fred_fx()
    with contextlib.redirect_stdout(io.StringIO()):
        fred_fx, filled = prepare_fred_fx(fred, raw.index[0])
    read = lambda name: pd.read_csv(DATA / name, index_col=0, parse_dates=True)
    mr = pd.read_csv(DATA / "mean_reversion_results.csv")
    yahoo, removed = clean_fx(raw["fx"])
    return {
        "raw": raw, "fred": fred, "fred_fx": fred_fx, "filled": filled,
        "same": read("premium_same_day.csv"), "prev": read("premium_prev_ny.csv"),
        "table": read("premium_table.csv"), "mr": mr,
        "yahoo": yahoo, "removed": removed, "flagged": flag_fx(yahoo),
    }


def yahoo_premiums(raw: pd.DataFrame, fx: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The earlier Yahoo version: rate on the Taipei date for prev_ny."""
    return (premium_same_day(raw["adr"], raw["local"], fx),
            premium_prev_ny(raw["adr"], raw["local"], fx, fx_on_adr_date=False))


# ---------------------------------------------------------------- tables

def t_data_sources(d: dict) -> str:
    raw, fred = d["raw"], d["fred"]
    rows = []
    for name, ticker, source, unit, s in [
        ("TSMC ADR", "TSM", "Yahoo Finance (yfinance)", "USD per ADR", raw["adr"]),
        ("TSMC ordinary share", "2330.TW", "Yahoo Finance (yfinance)", "TWD per share", raw["local"]),
        ("USD/TWD, comparison only", "TWD=X", "Yahoo Finance (yfinance)", "TWD per USD", raw["fx"]),
        ("USD/TWD, used for the premium", "DEXTAUS", "FRED (Federal Reserve H.10)", "TWD per USD",
         fred.loc[raw.index[0]:]),
    ]:
        s = s.dropna()
        rows.append([name, f"`{ticker}`", source, unit, s.index[0].date(), s.index[-1].date(), f"{len(s):,}"])
    key("Last FRED value", str(fred.dropna().index[-1].date()), "fx_fred.csv")
    filled = d["filled"]
    key("FRED values forward-filled since 2010", str(len(filled)), "fx_fred.csv")
    key("Forward-filled values used by prev_ny / same_day",
        f"{int(pd.DatetimeIndex(d['prev']['adr_date']).isin(filled).sum())} / {int(d['same'].index.isin(filled).sum())}",
        "premium CSVs")
    weekly = to_weekly(d["prev"]["premium"])
    key("Empty weeks in the weekly prev_ny series", str(int(weekly.isna().sum())), "premium_prev_ny.csv")
    key("Last Yahoo price date", str(raw.dropna(subset=["adr"]).index[-1].date()), "raw_closes.csv")
    return md(pd.DataFrame(rows, columns=["Series", "Ticker", "Source", "Unit", "First", "Last", "Observations"]),
              ["l", "l", "l", "l", "l", "l", "r"])


def t_manual_check(d: dict) -> str:
    t = d["table"].loc[MANUAL_CHECK_DATE]
    raw = d["raw"]
    adr_prev, local, fx_prev = t["adr_prev_ny"], t["local"], t["fx_prev_ny"]
    local_usd = ADR_RATIO * local / fx_prev
    premium = adr_prev / local_usd - 1
    local_usd_same = ADR_RATIO * local / t["fx_same_day"]
    yahoo_fx = raw.loc[t["adr_date_used_prev_ny"], "fx"]
    rows = [
        ["Taipei date d", MANUAL_CHECK_DATE, "`premium_table.csv`"],
        ["NY date used by `prev_ny` (`adr_date`)", str(t["adr_date_used_prev_ny"]), "previous NY session"],
        ["TSM close on `adr_date` (USD)", num(adr_prev, 2), "`raw_closes.csv`"],
        ["2330.TW close on d (TWD)", num(local, 1), "`raw_closes.csv`"],
        ["FRED USD/TWD on `adr_date`", num(fx_prev, 2), "`fx_fred.csv`"],
        [f"Local value of one ADR: {ADR_RATIO} x {num(local, 1)} / {num(fx_prev, 2)} (USD)", num(local_usd, 2), "computed"],
        [f"`prev_ny` premium: {num(adr_prev, 2)} / {num(local_usd, 2)} - 1", pct(premium, 2), "computed"],
        ["`prev_ny` premium in `premium_table.csv`", f"{t['premium_prev_ny_pct']:.2f}%", "pipeline"],
        ["TSM close on d (USD)", num(t["adr_same_day"], 2), "`raw_closes.csv`"],
        ["FRED USD/TWD on d", num(t["fx_same_day"], 2), "`fx_fred.csv`"],
        [f"`same_day` premium: {num(t['adr_same_day'], 2)} / {num(local_usd_same, 2)} - 1",
         pct(t["adr_same_day"] / local_usd_same - 1, 2), "computed"],
        ["Yahoo TWD=X on `adr_date`, for comparison", num(yahoo_fx, 4), "`raw_closes.csv`"],
    ]
    # Comparison with the news reports, which use Taipei d against NY d: the same_day definition.
    a = ARTICLE
    article_local_usd = ADR_RATIO * a["local"] / a["fx"]
    article_premium = a["adr"] / article_local_usd - 1
    adr_effect = a["adr"] / local_usd_same - 1
    fx_effect = t["adr_same_day"] / article_local_usd - 1
    previous = d["table"].loc[str(t["adr_date_used_prev_ny"]), "premium_same_day_pct"]
    rows += [
        ["News reports: TSM price (USD), not timestamped", num(a["adr"], 2), "news reports"],
        ["News reports: 2330.TW close (TWD) / USD/TWD", f"{num(a['local'], 1)} / {num(a['fx'], 3)}", "news reports"],
        ["News reports: premium", pct(a["premium"], 1), "news reports"],
        [f"Recomputed from the reported inputs: {num(a['adr'], 2)} / ({ADR_RATIO} x {num(a['local'], 1)} / "
         f"{num(a['fx'], 3)}) - 1", pct(article_premium, 2), "computed"],
        ["Reported ADR price with FRED rate on d", pct(adr_effect, 2), "computed"],
        ["TSM close with reported rate", pct(fx_effect, 2), "computed"],
        ["`same_day` premium (TSM close, FRED rate)", pct(t["adr_same_day"] / local_usd_same - 1, 2), "pipeline"],
        ["News reports: premium at the previous NY close", pct(a["premium_previous_close"], 1), "news reports"],
        [f"`same_day` premium on {t['adr_date_used_prev_ny']}", f"{previous:.2f}%", "`premium_table.csv`"],
    ]
    key("Article premium recomputed from reported inputs", pct(article_premium, 2), "computed")
    key("Reported ADR price with FRED rate", pct(adr_effect, 2), "computed")
    key("TSM close with reported rate", pct(fx_effect, 2), "computed")
    key("TSM close vs reported ADR price", pct(t["adr_same_day"] / a["adr"] - 1, 2), "raw_closes.csv")
    key("same_day premium on previous NY date", f"{previous:.2f}%", "premium_table.csv")
    key("Manual check prev_ny premium", pct(premium, 2), "computed from raw_closes.csv, fx_fred.csv")
    key("Manual check same_day premium", pct(t["adr_same_day"] / local_usd_same - 1, 2), "computed")
    key("Manual check TSM change on d", pct(t["adr_same_day"] / adr_prev - 1, 1), "raw_closes.csv")
    key("Manual check Yahoo vs FRED gap", pct(yahoo_fx / fx_prev - 1, 2), "raw_closes.csv, fx_fred.csv")
    return md(pd.DataFrame(rows, columns=["Item", "Value", "Source"]), ["l", "r", "l"])


def t_fx_gross_errors(d: dict) -> str:
    raw, fred = d["raw"], d["fred"]
    same_raw, prev_raw = yahoo_premiums(raw, raw["fx"])
    rows = []
    for r in d["removed"].itertuples():
        rows.append([r.date.date(), num(r.value, 4), num(fred.get(r.date, np.nan), 2),
                     pct(same_raw.loc[r.date, "premium"]), pct(prev_raw.loc[r.date, "premium"])])
        key(f"Gross error {r.date.date()} prev_ny premium", pct(prev_raw.loc[r.date, "premium"]), "recomputed")
    return md(pd.DataFrame(rows, columns=["Date", "Yahoo TWD=X", "FRED DEXTAUS",
                                          "`same_day` premium (raw Yahoo)", "`prev_ny` premium (raw Yahoo)"]))


def fx_comparison(d: dict) -> tuple[pd.Series, pd.DataFrame]:
    gap = relative_gap(d["yahoo"], d["fred"])
    return gap, compare_flagged(d["flagged"], gap, d["fred"])


def t_fx_gap_by_year(d: dict) -> str:
    gap, table = fx_comparison(d)
    flagged_year = pd.Series(table.index.year).value_counts()
    rows = []
    for year, g in gap.groupby(gap.index.year):
        rows.append([year, len(g), pct(g.abs().mean(), 2), pct((g.abs() > GAP_THRESHOLD).mean(), 0),
                     int(flagged_year.get(year, 0))])
    large = gap[gap.abs() > GAP_THRESHOLD]
    key("Dates with |gap| > 1%", str(len(large)), "recomputed")
    key("of which flagged", str(int(large.index.isin(table.index).sum())), "recomputed")
    key("Flagged dates", str(len(table)), "recomputed")
    for y in (2011, 2013, 2014, 2015, 2016):
        key(f"Share |gap|>1% in {y}", pct((gap.loc[str(y)].abs() > GAP_THRESHOLD).mean(), 0), "recomputed")
    key("Share of flagged dates in 2011-2016",
        str(int(((table.index.year >= 2011) & (table.index.year <= 2016)).sum())), "recomputed")
    return md(pd.DataFrame(rows, columns=["Year", "Common dates", "Mean abs gap", "Share abs gap > 1%",
                                          "Dates flagged in Yahoo"]))


def t_fx_verdicts(d: dict) -> str:
    _, table = fx_comparison(d)
    labels = {
        "yahoo_point_off": "Yahoo point off (gap > 1%, FRED flat)",
        "yahoo_neighbours_off": "Yahoo point close to FRED, its neighbours off",
        "no_fred_value": "No FRED value on that date",
        "fred_confirms": "FRED shows the same move",
        "unclear": "Unclear",
    }
    counts = table["verdict"].value_counts()
    rows = [[labels[v], int(counts[v])] for v in labels if v in counts]
    compared = table.dropna(subset=["gap"])
    for v in counts.index:
        key(f"Verdict {v}", str(int(counts[v])), "recomputed")
    key("Flagged dates comparable", str(len(compared)), "recomputed")
    key("Mean |FRED deviation| on flagged dates", pct(compared["fred_dev"].abs().mean(), 2), "recomputed")
    key("Mean |Yahoo deviation| on flagged dates", pct(compared["yahoo_dev"].abs().mean(), 2), "recomputed")
    key("FRED values flagged by the same rule", str(len(flag_fx(d["fred"].loc[d["yahoo"].index[0]:]))), "recomputed")
    return md(pd.DataFrame(rows, columns=["Verdict for the dates flagged in Yahoo", "Dates"]), ["l", "r"])


def t_fx_shifts(d: dict) -> str:
    gap, _ = fx_comparison(d)
    runs = persistent_gaps(gap)
    key("Persistent shifts", str(len(runs)), "recomputed")
    key("Longest persistent shift (dates)", str(int(runs["n"].max())), "recomputed")
    key("Shortest persistent shift (dates)", str(int(runs["n"].min())), "recomputed")
    rows = [[r.start, r.end, r.n, f"{r.mean_gap_pct:.2f}%"] for r in runs.itertuples()]
    return md(pd.DataFrame(rows, columns=["Start", "End", "Consecutive dates", "Mean gap Yahoo/FRED"]),
              ["l", "l", "r", "r"])


def noise_comparison(d: dict) -> dict:
    """Daily change std and autocorrelation: cleaned Yahoo against FRED."""
    y_same, y_prev = yahoo_premiums(d["raw"], d["yahoo"])
    out = {}
    for name, ys, fs in [("same_day", y_same, d["same"]), ("prev_ny", y_prev, d["prev"])]:
        dy, df = ys["premium"].diff(), fs["premium"].diff()
        common = dy.dropna().index.intersection(df.dropna().index)
        period = common[(common >= NOISE_PERIOD[0]) & (common <= f"{NOISE_PERIOD[1]}-12-31")]
        out[name] = {
            "yahoo": ys, "fred": fs,
            "noise_yahoo": dy.loc[period].std(), "noise_fred": df.loc[period].std(),
        }
    return out


def t_fx_impact(d: dict) -> str:
    comp = noise_comparison(d)
    diag_y = asynchrony_diagnostics(comp["same_day"]["yahoo"], comp["prev_ny"]["yahoo"])
    diag_f = asynchrony_diagnostics(d["same"], d["prev"])
    rows = []
    for name in ("prev_ny", "same_day"):
        c = comp[name]
        change = c["noise_fred"] / c["noise_yahoo"] - 1
        rows.append([f"`{name}`", f"{int(diag_y.loc[name, 'n_obs']):,} / {int(diag_f.loc[name, 'n_obs']):,}",
                     f"{pct(diag_y.loc[name, 'premium_mean'], 2)} / {pct(diag_f.loc[name, 'premium_mean'], 2)}",
                     f"{pct(diag_y.loc[name, 'daily_change_std'], 2)} / {pct(diag_f.loc[name, 'daily_change_std'], 2)}",
                     f"{pct(c['noise_yahoo'], 2)} / {pct(c['noise_fred'], 2)} ({change * 100:+.0f}%)",
                     f"{diag_y.loc[name, 'daily_change_autocorr_lag1']:.3f} / "
                     f"{diag_f.loc[name, 'daily_change_autocorr_lag1']:.3f}"])
        key(f"{name} 2011-2016 noise change", f"{change * 100:+.0f}%", "recomputed")
        key(f"{name} autocorr Yahoo / FRED", f"{diag_y.loc[name, 'daily_change_autocorr_lag1']:.3f} / "
            f"{diag_f.loc[name, 'daily_change_autocorr_lag1']:.3f}", "recomputed / premium CSVs")
    cols = ["Definition", "Observations", "Mean premium", "Daily change std",
            f"Daily change std, {NOISE_PERIOD[0]}-{NOISE_PERIOD[1]}", "Daily change autocorrelation"]
    for name, frame in (("prev_ny", d["prev"]), ("same_day", d["same"])):
        key(f"{name} min premium FRED", pct(frame["premium"].min()), "premium CSVs")
    return md(pd.DataFrame(rows, columns=[c + (" (Yahoo / FRED)" if i else "") for i, c in enumerate(cols)]))


def t_closing_times(_: dict) -> str:
    utc, tpe, nyc = ZoneInfo("UTC"), ZoneInfo("Asia/Taipei"), ZoneInfo("America/New_York")
    rows = []
    for label, day in [("Winter (example 2026-01-15)", datetime(2026, 1, 15)),
                       ("Summer (example 2026-07-15)", datetime(2026, 7, 15))]:
        taipei = datetime.combine(day, time(13, 30), tpe).astimezone(utc)
        ny = datetime.combine(day, time(16, 0), nyc).astimezone(utc)
        noon = datetime.combine(day, time(12, 0), nyc).astimezone(utc)
        ny_prev = ny - pd.Timedelta(days=1)
        same_gap = (ny - taipei).total_seconds() / 3600
        prev_gap = (taipei - ny_prev).total_seconds() / 3600
        rows.append([label, taipei.strftime("%H:%M"), ny.strftime("%H:%M"), noon.strftime("%H:%M"),
                     f"{same_gap:.1f} h", f"{prev_gap:.1f} h"])
    return md(pd.DataFrame(rows, columns=["Season", "Taipei close (13:30 local), UTC",
                                          "NY close (16:00 local), UTC", "FRED rate (noon NY), UTC",
                                          "`same_day` gap: NY close d - Taipei close d",
                                          "`prev_ny` gap: Taipei close d - NY close d-1"]))


def t_asynchrony(d: dict) -> str:
    diag = asynchrony_diagnostics(d["same"], d["prev"])
    rows = []
    for name in ("same_day", "prev_ny"):
        r = diag.loc[name]
        rows.append([f"`{name}`", f"{int(r['n_obs']):,}", pct(r["premium_mean"], 2), pct(r["premium_std"], 2),
                     pct(r["daily_change_std"], 2), f"{r['daily_change_autocorr_lag1']:.3f}"])
        key(f"{name} daily change std (FRED)", pct(r["daily_change_std"], 2), "premium CSVs")
        key(f"{name} autocorr (FRED)", f"{r['daily_change_autocorr_lag1']:.3f}", "premium CSVs")
    start = min(d["same"].index[0], d["prev"].index[0]).date()
    end = max(d["same"].index[-1], d["prev"].index[-1]).date()
    key("Premium sample", f"{start} to {end}", "premium CSVs")
    return md(pd.DataFrame(rows, columns=["Definition", "Observations", "Mean premium", "Premium std",
                                          "Daily change std", "Daily change autocorrelation (lag 1)"]))


def t_jumps(d: dict) -> str:
    t = d["table"]
    cols = ["premium_same_day_pct", "premium_prev_ny_pct"]
    jumps = t[(t[cols].diff().abs() > JUMP_THRESHOLD).any(axis=1)]
    nxt = t[cols].shift(-1).loc[jumps.index]
    rows = [[day.date(), f"{r.premium_same_day_pct:.2f}%", f"{r.premium_prev_ny_pct:.2f}%",
             f"{nxt.loc[day, cols[0]]:.2f}%", f"{nxt.loc[day, cols[1]]:.2f}%"]
            for day, r in jumps.iterrows()]
    key("Jumps above 10 points", str(len(jumps)), "premium_table.csv")
    return md(pd.DataFrame(rows, columns=["Date", "`same_day`", "`prev_ny`",
                                          "`same_day` next day", "`prev_ny` next day"]))


def t_example(d: dict) -> str:
    t = d["table"].loc[EXAMPLE_START:EXAMPLE_END]
    rows = [[day.date(), num(r.adr_same_day, 2), num(r.local, 1), num(r.fx_same_day, 2),
             f"{r.premium_same_day_pct:.2f}%", num(r.adr_prev_ny, 2), f"{r.premium_prev_ny_pct:.2f}%"]
            for day, r in t.iterrows()]
    for day in ("2025-04-08", "2025-04-09", "2025-04-10"):
        r = d["table"].loc[day]
        key(f"Example {day} same_day / prev_ny",
            f"{r.premium_same_day_pct:.2f}% / {r.premium_prev_ny_pct:.2f}%", "premium_table.csv")
    a = d["table"]
    key("Example TSM change 2025-04-09", pct(a.loc["2025-04-09", "adr_same_day"] / a.loc["2025-04-08", "adr_same_day"] - 1),
        "premium_table.csv")
    key("Example 2330.TW change 2025-04-10", pct(a.loc["2025-04-10", "local"] / a.loc["2025-04-09", "local"] - 1),
        "premium_table.csv")
    return md(pd.DataFrame(rows, columns=["Taipei date d", "TSM close on d (USD)", "2330.TW close on d (TWD)",
                                          "FRED on d", "`same_day`", "TSM close on `adr_date` (USD)", "`prev_ny`"]))


def mr_rows(d: dict, analysis: str) -> pd.DataFrame:
    return d["mr"][d["mr"]["analysis"] == analysis]


def t_stationarity(d: dict) -> str:
    rows = []
    for r in mr_rows(d, "full_sample").itertuples():
        bound = " (bound)" if r.kpss_pvalue_is_bound else ""
        rows.append([f"`{r.definition}`", r.frequency, f"{int(r.n):,}", f"{r.adf_stat:.2f}", pval(r.adf_pvalue),
                     int(r.adf_lags), f"{r.kpss_stat:.2f}", f"{r.kpss_pvalue:.2f}{bound}"])
        key(f"ADF p {r.definition} {r.frequency}", f"{r.adf_pvalue:.2f}", "mean_reversion_results.csv")
    sub = mr_rows(d, "subperiod")
    for r in sub[sub["definition"] == "prev_ny"].itertuples():
        key(f"Weekly mean premium {r.label}", pct(r.mean), "mean_reversion_results.csv")
    return md(pd.DataFrame(rows, columns=["Definition", "Frequency", "Observations", "ADF statistic",
                                          "ADF p-value", "ADF lags (AIC)", "KPSS statistic", "KPSS p-value"]))


def model(d: dict, label: str, frequency: str, name: str, definition: str = "prev_ny") -> pd.Series:
    m = mr_rows(d, "model_comparison")
    sel = m[(m["definition"] == definition) & (m["label"] == label) & (m["frequency"] == frequency)
            & (m["model"] == name)]
    return sel.iloc[0]


def hl(r: pd.Series, ci: bool = True) -> str:
    out = num(r["half_life_days"])
    return f"{out} [{num(r['half_life_ci_low'])}, {num(r['half_life_ci_high'])}]" if ci else out


def t_ar1_naive(d: dict) -> str:
    rows = []
    ratios = []
    for label in SUBPERIODS + ["full"]:
        dly, wk = model(d, label, "daily", "ar1"), model(d, label, "weekly_friday", "ar1")
        ratio = wk["half_life_days"] / dly["half_life_days"]
        if label != "full":
            ratios.append(ratio)
        rows.append([label if label != "full" else "Full sample", hl(dly), hl(wk), f"{ratio:.1f}",
                     f"{dly['resid_autocorr']:.2f}", f"{wk['resid_autocorr']:.2f}"])
    key("AR(1) weekly/daily ratio by sub-period", ", ".join(f"{r:.1f}" for r in ratios), "mean_reversion_results.csv")
    return md(pd.DataFrame(rows, columns=["Period (`prev_ny`)", "AR(1) daily [95% CI]", "AR(1) weekly [95% CI]",
                                          "Weekly / daily", "Residual autocorrelation, daily",
                                          "Residual autocorrelation, weekly"]))


@functools.cache
def simulate() -> pd.DataFrame:
    """AR(1) with phi = 0.98 plus white noise (sd 2, innovation sd 1), 8000 business days."""
    true = np.log(0.5) / np.log(SIM_PHI)
    rows = []
    for seed in SIM_SEEDS:
        rng = np.random.default_rng(seed)
        e = rng.normal(0, 1.0, SIM_N)
        x = np.empty(SIM_N)
        x[0] = e[0] / np.sqrt(1 - SIM_PHI**2)
        for t in range(1, SIM_N):
            x[t] = SIM_PHI * x[t - 1] + e[t]
        obs = pd.Series(x + rng.normal(0, SIM_NOISE, SIM_N), index=pd.bdate_range("2000-01-03", periods=SIM_N))
        weekly = to_weekly(obs)
        arma_d, arma_w = arma11_half_life(obs), arma11_half_life(weekly)
        rows.append({
            "ar1_daily": ar1_half_life(obs)["half_life"], "ar1_weekly": ar1_half_life(weekly)["half_life"] * 5,
            "arma_daily": arma_d["half_life"], "arma_weekly": arma_w["half_life"] * 5,
            "theta_daily": arma_d["theta"], "theta_weekly": arma_w["theta"],
            "autocorr": obs.diff().autocorr(),
        })
    out = pd.DataFrame(rows)
    out.attrs["true"] = true
    return out


def t_simulation(_: dict) -> str:
    sim = simulate()
    true = sim.attrs["true"]
    rows = []
    for label, col in [("AR(1), daily", "ar1_daily"), ("AR(1), weekly", "ar1_weekly"),
                       ("ARMA(1,1), daily", "arma_daily"), ("ARMA(1,1), weekly", "arma_weekly")]:
        s = sim[col]
        rows.append([label, num(s.mean()), num(s.std()), f"{num(s.min())} to {num(s.max())}", num(true)])
        key(f"Simulation {label} mean", num(s.mean()), "recomputed simulation")
    key("Simulation true half-life", num(true), "ln(0.5)/ln(0.98)")
    key("Simulation daily change autocorr", f"{sim['autocorr'].mean():.2f}", "recomputed simulation")
    key("Simulation theta daily / weekly", f"{sim['theta_daily'].mean():.2f} / {sim['theta_weekly'].mean():.2f}",
        "recomputed simulation")
    key("Simulation replications", str(len(sim)), "recomputed simulation")
    return md(pd.DataFrame(rows, columns=["Estimator (half-life in days)", "Mean", "Std",
                                          f"Range over {len(sim)} replications", "True value"]))


def t_arma(d: dict) -> str:
    rows = []
    sub = mr_rows(d, "subperiod")
    for label in SUBPERIODS:
        dly, wk = model(d, label, "daily", "arma11"), model(d, label, "weekly_friday", "arma11")
        mean = sub[(sub["definition"] == "prev_ny") & (sub["label"] == label)].iloc[0]["mean"]
        rows.append([label, f"{int(wk['n'])}", pct(mean), hl(dly), hl(wk), f"{dly['theta']:.2f}",
                     f"{wk['theta']:.2f}", pval(dly["ljungbox_pvalue"]), pval(wk["ljungbox_pvalue"])])
        key(f"ARMA {label} daily / weekly", f"{num(dly['half_life_days'])} / {num(wk['half_life_days'])}",
            "mean_reversion_results.csv")
    for freq in ("daily", "weekly_friday"):
        th = [model(d, l, freq, "arma11")["theta"] for l in SUBPERIODS]
        key(f"ARMA theta range {freq}", f"{min(th):.2f} to {max(th):.2f}", "mean_reversion_results.csv")
    same = [f"{num(model(d, l, 'daily', 'arma11', 'same_day')['half_life_days'])} / "
            f"{num(model(d, l, 'weekly_friday', 'arma11', 'same_day')['half_life_days'])}" for l in SUBPERIODS]
    key("same_day ARMA daily / weekly by sub-period", "; ".join(same), "mean_reversion_results.csv")
    full_d, full_w = model(d, "full", "daily", "arma11"), model(d, "full", "weekly_friday", "arma11")
    key("Full-sample ARMA daily / weekly", f"{num(full_d['half_life_days'])} / {num(full_w['half_life_days'])}",
        "mean_reversion_results.csv")
    return md(pd.DataFrame(rows, columns=["Period (`prev_ny`)", "Weeks", "Mean premium", "ARMA daily [95% CI]",
                                          "ARMA weekly [95% CI]", "Theta daily", "Theta weekly",
                                          "Ljung-Box p, daily", "Ljung-Box p, weekly"]))


def t_rolling(d: dict) -> str:
    r = mr_rows(d, "rolling_arma").copy()
    r["end"] = pd.to_datetime(r["end"])
    rows = []

    def summary(label, g):
        kept = g[~g["excluded"]]
        h = kept["half_life_days"]
        q1, med, q3 = h.quantile([0.25, 0.5, 0.75])
        return [label, len(g), int(g["excluded"].sum()), num(med), f"{num(q1)} to {num(q3)}",
                pct((h > 20).mean(), 0), pct((kept["half_life_ci_low"] > 20).mean(), 0)]

    for label, freq, mdl in [("ARMA(1,1) weekly, 3-year windows", "weekly", "arma11"),
                             ("ARMA(1,1) daily, 2-year windows", "daily", "arma11"),
                             ("AR(1) weekly, 3-year windows", "weekly", "ar1")]:
        g = r[(r["frequency"] == freq) & (r["model"] == mdl)]
        rows.append(summary(label, g))
        key(f"Rolling {label}", f"median {rows[-1][3]}, IQR {rows[-1][4]}, >20 days {rows[-1][5]}, "
            f"lower bound >20 {rows[-1][6]}, windows {rows[-1][1]}, excluded {rows[-1][2]}", "mean_reversion_results.csv")
    weekly = r[(r["frequency"] == "weekly") & (r["model"] == "arma11")]
    for label, a, b in ROLLING_GROUPS:
        g = weekly[(weekly["end"] >= a) & (weekly["end"] <= f"{b}-12-31")]
        rows.append(summary(f"ARMA(1,1) weekly, windows ending {label}", g))
        key(f"Rolling weekly ARMA median, windows ending {label}", rows[-1][3], "mean_reversion_results.csv")
    below = weekly[~weekly["excluded"] & (weekly["half_life_days"] < 20)]
    key("Weekly ARMA windows below 20 days", f"{len(below)}, ending {below['end'].min().date()} to "
        f"{below['end'].max().date()}", "mean_reversion_results.csv")
    daily = r[(r["frequency"] == "daily") & (r["model"] == "arma11") & ~r["excluded"]].set_index("end")
    both = weekly[~weekly["excluded"]].set_index("end").join(daily, lsuffix="_w", rsuffix="_d", how="inner")
    key("Median weekly/daily ratio, rolling", num((both["half_life_days_w"] / both["half_life_days_d"]).median(), 1),
        "mean_reversion_results.csv")
    return md(pd.DataFrame(rows, columns=["Estimate (`prev_ny`, half-life in trading days)", "Windows", "Excluded",
                                          "Median", "Interquartile range", "Share > 20 days",
                                          "Share with lower 95% bound > 20 days"]))


def t_chow(d: dict) -> str:
    titles = {"chow": "Candidate date, full history", "chow_placebo": "Placebo date, full history",
              "chow_since_2024": "Sample from 2024 only"}
    rows = []
    for analysis, title in titles.items():
        for r in mr_rows(d, analysis).itertuples():
            rows.append([title, r.label, f"{r.chow_f:.2f}", pval(r.chow_pvalue), int(r.n_before), int(r.n_after)])
    placebo = mr_rows(d, "chow_placebo")
    key("Placebo Chow p-value range", f"{pval(placebo['chow_pvalue'].min())} to {pval(placebo['chow_pvalue'].max())}",
        "mean_reversion_results.csv")
    cand = mr_rows(d, "chow")
    key("Candidate Chow p-value range", f"{pval(cand['chow_pvalue'].min())} to {pval(cand['chow_pvalue'].max())}",
        "mean_reversion_results.csv")
    key("Candidate weeks after", f"{int(cand['n_after'].min())} to {int(cand['n_after'].max())}",
        "mean_reversion_results.csv")
    loc = mr_rows(d, "chow_since_2024").set_index("label")
    key("Since-2024 Chow p at 2026-01-02", pval(loc.loc["2026-01-02", "chow_pvalue"]), "mean_reversion_results.csv")
    key("Since-2024 Chow p in 2025", ", ".join(pval(loc.loc[x, "chow_pvalue"]) for x in ("2025-01-03", "2025-07-04")),
        "mean_reversion_results.csv")
    return md(pd.DataFrame(rows, columns=["Test", "Break date", "F", "p-value", "Weeks before", "Weeks after"]),
              ["l", "l", "r", "r", "r", "r"])


def t_issues(d: dict) -> str:
    """One row per methodological issue, numbers from the other computations."""
    _, table = fx_comparison(d)
    gap, _ = fx_comparison(d)
    large = gap[gap.abs() > GAP_THRESHOLD]
    comp = noise_comparison(d)
    diag_y = asynchrony_diagnostics(comp["same_day"]["yahoo"], comp["prev_ny"]["yahoo"])
    diag_f = asynchrony_diagnostics(d["same"], d["prev"])
    same_raw, prev_raw = yahoo_premiums(d["raw"], d["raw"]["fx"])
    sim = simulate()
    ar1_w = [model(d, l, "weekly_friday", "ar1")["half_life_days"] for l in SUBPERIODS]
    ar1_d = [model(d, l, "daily", "ar1")["half_life_days"] for l in SUBPERIODS]
    arma_w = [model(d, l, "weekly_friday", "arma11")["half_life_days"] for l in SUBPERIODS]
    r = mr_rows(d, "rolling_arma")
    wk = r[(r["frequency"] == "weekly") & (r["model"] == "arma11") & ~r["excluded"]].copy()
    wk["end"] = pd.to_datetime(wk["end"])
    grp = {lab: wk[(wk["end"] >= a) & (wk["end"] <= f"{b}-12-31")]["half_life_days"].median()
           for lab, a, b in ROLLING_GROUPS}
    placebo, cand = mr_rows(d, "chow_placebo"), mr_rows(d, "chow")
    noise = comp["prev_ny"]
    rows = [
        ["a) Errors in the Yahoo exchange rate",
         f"{len(d['removed'])} impossible values (outside [{FX_MIN:g}, {FX_MAX:g}]) gave premiums down to "
         f"{pct(prev_raw.loc[d['removed']['date'], 'premium'].min())}; a local rule then flagged {len(table)} "
         f"dates; comparison with FRED found {len(large)} dates with a gap above 1%, of which "
         f"{int(large.index.isin(table.index).sum())} flagged, and {len(persistent_gaps(gap))} persistent shifts",
         "FRED DEXTAUS, taken on the NY date of the ADR used",
         f"`prev_ny` daily change std over {NOISE_PERIOD[0]}-{NOISE_PERIOD[1]}: {pct(noise['noise_yahoo'], 2)} "
         f"-> {pct(noise['noise_fred'], 2)}; minimum premium {pct(prev_raw['premium'].min())} -> "
         f"{pct(d['prev']['premium'].min())}"],
        ["b) Measurement noise from non-synchronous closes",
         f"Daily changes have a lag-1 autocorrelation of {diag_f.loc['prev_ny', 'daily_change_autocorr_lag1']:.2f}; "
         f"it is unchanged when Yahoo is replaced by FRED",
         "`prev_ny` as reference; weekly sampling and a signal-plus-noise model downstream",
         f"Autocorrelation `prev_ny`: {diag_y.loc['prev_ny', 'daily_change_autocorr_lag1']:.3f} (Yahoo) -> "
         f"{diag_f.loc['prev_ny', 'daily_change_autocorr_lag1']:.3f} (FRED): the noise is not from the exchange rate"],
        ["c) AR(1) half-life biased downwards",
         f"Weekly AR(1) half-lives {num(min(ar1_w))}-{num(max(ar1_w))} days against {num(min(ar1_d))}-"
         f"{num(max(ar1_d))} daily; a calibrated simulation (true {num(sim.attrs['true'])} days) gives "
         f"{num(sim['ar1_daily'].mean())} daily and {num(sim['ar1_weekly'].mean())} weekly",
         "ARMA(1,1): AR(1) signal plus white measurement noise",
         f"Weekly half-life by sub-period: {num(min(ar1_w))}-{num(max(ar1_w))} (AR(1)) -> "
         f"{num(min(arma_w))}-{num(max(arma_w))} days (ARMA)"],
        ["d) Sub-periods chosen after looking at the data",
         "Sub-period boundaries were read from the premium chart, so agreement within them may be selected",
         "Rolling 3-year windows with a monthly step, no manual split",
         f"Weekly ARMA median {num(wk['half_life_days'].median())} days, above 20 days in "
         f"{pct((wk['half_life_days'] > 20).mean(), 0)} of windows, but from {num(grp['2017-2019'])} "
         f"(windows ending 2017-2019) to {num(grp['2024-2026'])} (ending 2024-2026)"],
        ["e) Chow test at chosen break dates",
         f"Rejection at the April-May 2026 dates (p {pval(cand['chow_pvalue'].min())} to "
         f"{pval(cand['chow_pvalue'].max())}); placebo dates reject too (all {len(placebo)}, "
         f"p at most {pval(placebo['chow_pvalue'].max())})",
         "Results reported as not conclusive; no break claimed",
         f"Break at the candidate dates: claimed -> not conclusive ({len(placebo)} of {len(placebo)} "
         f"placebo dates reject)"],
    ]
    return md(pd.DataFrame(rows, columns=["Issue", "How it was detected", "Fix", "Impact (before -> after)"]),
              ["l", "l", "l", "l"])


def t_versions(_: dict) -> str:
    rows = [["Python", sys.version.split()[0]]]
    for pkg in ("pandas", "numpy", "statsmodels", "scipy", "matplotlib", "yfinance", "pytest"):
        rows.append([pkg, version(pkg)])
    return md(pd.DataFrame(rows, columns=["Component", "Version"]), ["l", "l"])


def t_key_figures(_: dict) -> str:
    return md(pd.DataFrame(KEY, columns=["Quantity quoted in the text", "Value", "Source"]), ["l", "l", "l"])


TABLES_IN_ORDER = [
    ("data_sources", t_data_sources), ("manual_check", t_manual_check), ("fx_gross_errors", t_fx_gross_errors),
    ("fx_gap_by_year", t_fx_gap_by_year), ("fx_verdicts", t_fx_verdicts), ("fx_shifts", t_fx_shifts),
    ("fx_impact", t_fx_impact), ("closing_times", t_closing_times), ("asynchrony", t_asynchrony),
    ("jumps", t_jumps), ("example", t_example), ("stationarity", t_stationarity), ("ar1_naive", t_ar1_naive),
    ("simulation", t_simulation), ("arma", t_arma), ("rolling", t_rolling), ("chow", t_chow),
    ("issues", t_issues), ("versions", t_versions), ("key_figures", t_key_figures),
]


def fill_report(tables: dict[str, str]) -> None:
    text = REPORT.read_text()
    for name, content in tables.items():
        pattern = re.compile(rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", re.S)
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m: m.group(1) + content + m.group(2), text)
    missing = set(re.findall(r"<!-- table:(\w+) -->", text)) - set(tables)
    if missing:
        raise ValueError(f"Unknown tables referenced in the report: {sorted(missing)}")
    REPORT.write_text(text)


def main() -> None:
    d = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, fn in TABLES_IN_ORDER:
        tables[name] = fn(d)
        (TABLES / f"{name}.md").write_text(tables[name])
        print(f"{name}: {tables[name].count(chr(10)) - 2} rows")
    if REPORT.exists():
        fill_report(tables)
        print(f"Tables inserted into {REPORT}")


if __name__ == "__main__":
    main()
