"""Stationarity and mean reversion of the premium (FRED FX).

prev_ny is the reference definition, same_day the robustness check. Half-lives are
expressed in trading days; weekly ones are converted at 5 trading days per week.

Usage: python scripts/mean_reversion.py
Output: data/mean_reversion_results.csv, figures/rolling_mean_half_life.png,
        figures/rolling_arma_half_life.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr_premium.clean import prepare_fred_fx
from adr_premium.data import download, download_fred_fx
from adr_premium.premium import premium_prev_ny, premium_same_day
from adr_premium.stats import adf_kpss, ar1_half_life, arma11_half_life, chow_test, to_weekly, to_weekly_mean

DAYS_PER_WEEK = 5
ROLLING_DAYS = 504  # 2 years of trading days
ROLLING_WEEKS = 104  # the same 2 years in weeks
CHOW_DATES = ["2026-04-01", "2026-04-15", "2026-04-24", "2026-05-01"]
# Below one year of weekly data, the AR(1) slope of a regime whose half-life is
# itself several weeks long is too imprecise for a break test to be conclusive.
MIN_WEEKS = 52
# Rolling ARMA(1,1) on prev_ny, without any hand-picked sub-period.
ARMA_WINDOW_WEEKS = 156  # 3 years
ARMA_WINDOW_DAYS = 504  # 2 years, as a comparison
PHI_MAX = 0.999  # beyond this the half-life is not identified from one window
HALF_LIFE_THRESHOLD = 20  # trading days
# The "before" regime of a full-history Chow test spans premiums from 1% to 20%:
# placebo dates show whether the test rejects everywhere, and a sample starting in
# 2024 compares the candidate dates with the recent regime only.
CHOW_PLACEBO_DATES = ["2013-01-04", "2016-01-01", "2018-01-05", "2020-01-03",
                      "2022-01-07", "2024-01-05", "2025-01-03"]
CHOW_LOCAL_START = "2024-01-01"
CHOW_LOCAL_DATES = ["2025-01-03", "2025-07-04", "2026-01-02"] + CHOW_DATES
# Every Chow result is reported as not conclusive: the full-history test rejects at
# every placebo date, and the AR(1) residuals are autocorrelated, which breaks the
# assumptions behind its F distribution.
CHOW_VERDICT = "not conclusive: rejects at every placebo date, AR(1) residuals autocorrelated"
SUBPERIODS = [
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020-2023", "2020-01-01", "2023-12-31"),
    ("2024-Apr 2026", "2024-01-01", "2026-04-30"),
    ("May 2026-end", "2026-05-01", None),
]


# Sampling schemes for the AR(1) / ARMA(1,1) comparison, with days per step.
# Weekly means are a control only: averaging creates autocorrelation of its own
# (smoother levels, positively autocorrelated changes), so their half-lives are not
# directly comparable with point samples.
SAMPLINGS = [
    ("daily", lambda s: s, 1),
    ("weekly_friday", to_weekly, DAYS_PER_WEEK),
    ("weekly_mean", to_weekly_mean, DAYS_PER_WEEK),
]
MODELS = [("ar1", ar1_half_life), ("arma11", arma11_half_life)]


def premiums() -> dict[str, pd.Series]:
    raw = download()
    fx, _ = prepare_fred_fx(download_fred_fx(), raw.index[0])
    return {
        "prev_ny": premium_prev_ny(raw["adr"], raw["local"], fx)["premium"],
        "same_day": premium_same_day(raw["adr"], raw["local"], fx)["premium"],
    }


def half_life_row(x: pd.Series, scale: int) -> dict:
    """AR(1) results with half-lives converted to trading days."""
    ar = ar1_half_life(x)
    return {
        "n_ar1": ar["n"],
        "phi": ar["phi"],
        "half_life_days": ar["half_life"] * scale,
        "half_life_ci_low": ar["half_life_ci_low"] * scale,
        "half_life_ci_high": ar["half_life_ci_high"] * scale,
        "r2": ar["r2"],
    }


def full_sample(series: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for name, s in series.items():
        for freq, x, scale in (("daily", s, 1), ("weekly", to_weekly(s), DAYS_PER_WEEK)):
            x = x.dropna() if freq == "daily" else x
            rows.append({
                "analysis": "full_sample", "definition": name, "frequency": freq,
                "start": x.first_valid_index().date(), "end": x.last_valid_index().date(),
                "mean": x.mean(), "std": x.std(),
                **adf_kpss(x), **half_life_row(x, scale),
            })
    return rows


def rolling(series: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for name, s in series.items():
        weekly = to_weekly(s)
        month_ends = s.groupby(s.index.to_period("M")).tail(1).index
        for end in month_ends:
            window = s.loc[:end].iloc[-ROLLING_DAYS:]
            if len(window) < ROLLING_DAYS:
                continue
            for freq, x, scale in (("daily", window, 1),
                                   ("weekly", weekly.loc[:end].iloc[-ROLLING_WEEKS:], DAYS_PER_WEEK)):
                rows.append({
                    "analysis": "rolling", "definition": name, "frequency": freq,
                    "start": x.first_valid_index().date(), "end": end.date(),
                    "mean": x.mean(), "std": x.std(), **half_life_row(x, scale),
                })
    return rows


def chow(series: dict[str, pd.Series]) -> list[dict]:
    weekly = to_weekly(series["prev_ny"])
    runs = [("chow", weekly, date) for date in CHOW_DATES]
    runs += [("chow_placebo", weekly, date) for date in CHOW_PLACEBO_DATES]
    runs += [("chow_since_2024", weekly.loc[CHOW_LOCAL_START:], date) for date in CHOW_LOCAL_DATES]
    rows = []
    for analysis, x, date in runs:
        out = chow_test(x, date)
        short = f"; only {out['n_after']} weeks after the date (< {MIN_WEEKS})" if out["n_after"] < MIN_WEEKS else ""
        rows.append({
            "analysis": analysis, "definition": "prev_ny", "frequency": "weekly", "label": date,
            "start": x.first_valid_index().date(), **out,
            "conclusive": False, "note": CHOW_VERDICT + short,
        })
    return rows


def subperiods(series: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for name, s in series.items():
        weekly = to_weekly(s)
        for label, start, end in SUBPERIODS:
            x = weekly.loc[start:end]
            n = int(x.notna().sum())
            rows.append({
                "analysis": "subperiod", "definition": name, "frequency": "weekly", "label": label,
                "start": x.first_valid_index().date(), "end": x.last_valid_index().date(),
                "mean": x.mean(), "std": x.std(), **half_life_row(x, DAYS_PER_WEEK),
                "note": f"only {n} weeks (< {MIN_WEEKS}): half-life imprecise" if n < MIN_WEEKS else "",
            })
    return rows


def rolling_arma(series: dict[str, pd.Series]) -> list[dict]:
    """ARMA(1,1) on rolling prev_ny windows, monthly step, plus the weekly AR(1).

    A window is excluded from the statistics when the fit did not converge or when
    phi > PHI_MAX.
    """
    s = series["prev_ny"]
    weekly = to_weekly(s)
    month_ends = s.groupby(s.index.to_period("M")).tail(1).index
    rows = []
    for end in month_ends:
        windows = [
            ("weekly", "arma11", weekly.loc[:end].iloc[-ARMA_WINDOW_WEEKS:], ARMA_WINDOW_WEEKS, DAYS_PER_WEEK),
            ("weekly", "ar1", weekly.loc[:end].iloc[-ARMA_WINDOW_WEEKS:], ARMA_WINDOW_WEEKS, DAYS_PER_WEEK),
            ("daily", "arma11", s.loc[:end].iloc[-ARMA_WINDOW_DAYS:], ARMA_WINDOW_DAYS, 1),
        ]
        for freq, model, x, size, scale in windows:
            if len(x) < size:
                continue
            out = arma11_half_life(x) if model == "arma11" else ar1_half_life(x)
            converged = out.get("converged", True)
            rows.append({
                "analysis": "rolling_arma", "definition": "prev_ny", "frequency": freq, "model": model,
                "start": x.first_valid_index().date(), "end": end.date(), "n": int(x.notna().sum()),
                "phi": out["phi"], "theta": out.get("theta", np.nan),
                "half_life_days": out["half_life"] * scale,
                "half_life_ci_low": out["half_life_ci_low"] * scale,
                "half_life_ci_high": out["half_life_ci_high"] * scale,
                "converged": converged,
                "excluded": model == "arma11" and (not converged or out["phi"] > PHI_MAX),
            })
    return rows


def summarize_rolling_arma(rows: list[dict]) -> None:
    t = pd.DataFrame(rows)
    print(f"\nRolling ARMA(1,1), prev_ny, monthly step (half-life in trading days; windows not "
          f"converged or with phi > {PHI_MAX} excluded)")
    for (freq, model), g in t.groupby(["frequency", "model"], sort=False):
        kept = g[~g["excluded"]]
        hl = kept["half_life_days"]
        q1, med, q3 = hl.quantile([0.25, 0.5, 0.75])
        low = kept["half_life_ci_low"].median()
        theta = f", theta median {kept['theta'].median():.2f}" if model == "arma11" else ""
        print(f"  {freq:6s} {model:6s} windows {len(g)}, excluded {int(g['excluded'].sum())}: "
              f"median {med:.1f} [Q1 {q1:.1f}, Q3 {q3:.1f}], range {hl.min():.1f} to {hl.max():.1f}, "
              f"median CI low {low:.1f}, share > {HALF_LIFE_THRESHOLD} days "
              f"{(hl > HALF_LIFE_THRESHOLD).mean():.0%}{theta}")
    excluded = t[t["excluded"]]
    if len(excluded):
        print("  excluded windows: " + ", ".join(f"{r.frequency} {r.end} (phi {r.phi:.4f}, "
              f"{'converged' if r.converged else 'not converged'})" for r in excluded.itertuples()))


def plot_rolling_arma(rows: list[dict], out: Path) -> None:
    t = pd.DataFrame(rows)
    t["end"] = pd.to_datetime(t["end"])
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = {
        ("weekly", "arma11"): ("#d95f02", "-", "ARMA(1,1) weekly, 3-year window"),
        ("daily", "arma11"): ("#1f77b4", "-", "ARMA(1,1) daily, 2-year window"),
        ("weekly", "ar1"): ("#7f7f7f", "--", "AR(1) weekly, 3-year window"),
    }
    for (freq, model), (color, ls, label) in styles.items():
        g = t[(t["frequency"] == freq) & (t["model"] == model)]
        shown = g["half_life_days"].where(~g["excluded"])
        ax.plot(g["end"], shown, color=color, ls=ls, lw=1.4, label=label)
        if model == "arma11":
            ax.plot(g["end"], g["half_life_ci_low"].where(~g["excluded"]), color=color, ls=":", lw=1.0,
                    label=f"ARMA(1,1) {freq}, lower 95% bound")
    for end in t.loc[t["excluded"], "end"].unique():
        ax.axvspan(end - pd.Timedelta(days=15), end + pd.Timedelta(days=15), color="0.85", lw=0, zorder=0)
    ax.axhline(HALF_LIFE_THRESHOLD, color="black", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Half-life (trading days)")
    ax.set_title("prev_ny half-life on rolling windows, monthly step (grey: excluded windows)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def model_comparison(series: dict[str, pd.Series]) -> list[dict]:
    """AR(1) vs ARMA(1,1) on the full sample and each sub-period, for every sampling."""
    periods = [("full", None, None)] + SUBPERIODS
    rows = []
    for name, s in series.items():
        for label, start, end in periods:
            for sampling, transform, scale in SAMPLINGS:
                x = transform(s).loc[start:end]
                n = int(x.notna().sum())
                for model, fit in MODELS:
                    out = fit(x)
                    rows.append({
                        "analysis": "model_comparison", "definition": name, "frequency": sampling,
                        "label": label, "model": model,
                        "start": x.first_valid_index().date(), "end": x.last_valid_index().date(),
                        "n": n, "phi": out["phi"], "theta": out.get("theta", np.nan),
                        "half_life_days": out["half_life"] * scale,
                        "half_life_ci_low": out["half_life_ci_low"] * scale,
                        "half_life_ci_high": out["half_life_ci_high"] * scale,
                        "resid_autocorr": out["resid_autocorr"],
                        "ljungbox_pvalue": out["ljungbox_pvalue"],
                        "converged": out.get("converged", True),
                        "note": f"only {n} observations" if sampling != "daily" and n < MIN_WEEKS else "",
                    })
    return rows


def print_model_comparison(rows: list[dict]) -> None:
    t = pd.DataFrame(rows)
    t["half-life [95% CI]"] = t.apply(fmt_hl, axis=1)
    t["warn"] = np.where(~t["converged"], "not converged ", "") + t["note"]
    cols = ["label", "frequency", "model", "n", "phi", "theta", "half-life [95% CI]",
            "resid_autocorr", "ljungbox_pvalue", "warn"]
    for name, g in t.groupby("definition", sort=False):
        print(f"\nAR(1) vs ARMA(1,1), {name} (half-life in trading days; Ljung-Box at 4 lags)")
        print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def plot_rolling(table: pd.DataFrame, out: Path) -> None:
    ref = table[table["definition"] == "prev_ny"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"daily": "#1f77b4", "weekly": "#d95f02"}
    hidden = []
    for freq, g in ref.groupby("frequency"):
        dates = pd.to_datetime(g["end"])
        top.plot(dates, g["mean"] * 100, lw=1.2, color=colors[freq], label=freq.capitalize())
        finite = np.isfinite(g["half_life_days"])
        bottom.plot(dates[finite], g["half_life_days"][finite], lw=1.2, color=colors[freq],
                    label=f"{freq.capitalize()} AR(1)")
        if (~finite).any():
            hidden.append(f"{freq}: {(~finite).sum()}")
    top.set_ylabel("Mean premium (%)")
    top.set_title("prev_ny premium, rolling 2-year windows, monthly step")
    bottom.set_ylabel("Half-life (trading days)")
    bottom.set_yscale("log")
    if hidden:
        bottom.set_title("Windows with no mean reversion (phi >= 1) not shown: " + ", ".join(hidden),
                         fontsize=9, loc="left")
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fmt_hl(row) -> str:
    def one(v):
        return "inf" if np.isinf(v) else f"{v:.1f}"
    return f"{one(row.half_life_days)} [{one(row.half_life_ci_low)}, {one(row.half_life_ci_high)}]"


def main() -> None:
    pd.set_option("display.width", 170)
    series = premiums()
    full, roll, breaks, subs = full_sample(series), rolling(series), chow(series), subperiods(series)
    models = model_comparison(series)
    roll_arma = rolling_arma(series)
    table = pd.DataFrame(full + roll + breaks + subs + models + roll_arma)
    out_csv = ROOT / "data" / "mean_reversion_results.csv"
    table.to_csv(out_csv, index=False)

    print("\nFull sample (half-life in trading days, 95% CI)")
    for r in pd.DataFrame(full).itertuples():
        bound = " (bound)" if r.kpss_pvalue_is_bound else ""
        print(f"  {r.definition:8s} {r.frequency:6s} n={r.n:4d}  ADF stat {r.adf_stat:6.2f} p={r.adf_pvalue:.3f} "
              f"lags={r.adf_lags:2d} | KPSS stat {r.kpss_stat:5.2f} p={r.kpss_pvalue:.3f}{bound} | "
              f"phi={r.phi:.4f} half-life {fmt_hl(r)}")

    print("\nRolling 2-year windows (trading days)")
    r = pd.DataFrame(roll)
    for (name, freq), g in r.groupby(["definition", "frequency"], sort=False):
        hl = g["half_life_days"]
        fin = hl[np.isfinite(hl)]
        print(f"  {name:8s} {freq:6s} windows={len(g)}  mean premium {g['mean'].min():.1%} to {g['mean'].max():.1%}  "
              f"half-life median {fin.median():.1f}, range {fin.min():.1f} to {fin.max():.1f}, "
              f"no reversion in {np.isinf(hl).sum()}")
        last = g.iloc[-1]
        print(f"  {'':15s} last window to {last['end']}: mean {last['mean']:.1%}, half-life {fmt_hl(last)}")

    titles = {"chow": "Chow test, prev_ny weekly AR(1), full history",
              "chow_placebo": "Placebo dates, full history",
              "chow_since_2024": f"Sample from {CHOW_LOCAL_START} only"}
    print(f"\nCHOW TESTS ARE NOT CONCLUSIVE: {CHOW_VERDICT.split(': ', 1)[1]}.")
    for analysis, title in titles.items():
        print(f"\n{title} (not conclusive)")
        for b in (b for b in breaks if b["analysis"] == analysis):
            short = f", only {b['n_after']} weeks after" if b["n_after"] < MIN_WEEKS else ""
            print(f"  {b['label']}: F={b['chow_f']:.2f} p={b['chow_pvalue']:.2g} "
                  f"(weeks before {b['n_before']}, after {b['n_after']}{short})")

    print("\nSub-periods, weekly")
    for s in pd.DataFrame(subs).itertuples():
        print(f"  {s.definition:8s} {s.label:14s} {s.start} to {s.end}  n={s.n_ar1 + 1:3d}  mean {s.mean:6.2%}  "
              f"std {s.std:6.2%}  phi={s.phi:.3f}  half-life {fmt_hl(s)} {s.note}")

    print_model_comparison(models)
    summarize_rolling_arma(roll_arma)
    plot_rolling_arma(roll_arma, ROOT / "figures" / "rolling_arma_half_life.png")

    out_fig = ROOT / "figures" / "rolling_mean_half_life.png"
    plot_rolling(r, out_fig)
    print(f"\nResults: {out_csv} ({len(table)} rows)\nChart: {out_fig}")


if __name__ == "__main__":
    main()
