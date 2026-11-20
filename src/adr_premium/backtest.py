"""Backtest engine for the premium mean-reversion strategy (roadmap step 5a).

Signal: z-score of q = ln(1 + premium), with the prev_ny premium and the FRED rate,
sampled weekly on the last Taipei date of each week. The mean and standard deviation
come from the ``window`` PREVIOUS weekly observations only: the current week enters q
but not its own benchmark, and no later data is ever read.

Execution. The signal is known at the Taipei close of the decision date d (05:30 UTC).
Each leg trades at the first closing price strictly after that moment:
- ADR: NY close of the first NY trading date >= d (NY closes after Taipei on the
  same calendar date);
- local share: Taipei close of the first Taipei trading date > d;
- FX: FRED rate (noon New York) of the local leg's execution date, for both the
  conversion into TWD at entry and the conversion of the sale proceeds at exit.
The ``extra_lag`` option moves each leg by that many further trading days on its own
exchange.

P&L. The P&L is computed from the execution prices and from each leg's actual daily
closes, NEVER from changes of the prev_ny premium. That premium pairs a Taipei close
with an ADR close that is already in the past at decision time and carries the
asynchrony noise measured in step 1: a P&L built on it would book a convergence that
no order could have captured.

Units: each trade puts 1 USD in each leg (long local, short ADR). P&L is in USD per
1 USD per leg, without compounding; divide by 2 for a return on gross exposure.

Financing: the proceeds of the ADR short sale are assumed to fund the purchase of the
local share. Neither interest on the short proceeds nor a financing cost of the long
leg is modelled; the only carry cost is the ADR borrow fee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adr_premium.premium import ADR_RATIO, premium_prev_ny

FINANCING_ASSUMPTION = ("Financing: the ADR short-sale proceeds fund the local purchase; no interest "
                        "income or financing cost is modelled, only the ADR borrow fee.")

# ---------------------------------------------------------------------------
# Rules fixed a priori: chosen BEFORE any backtest was run and never tuned on the
# results. The sensitivity grid in scripts/backtest.py is reported without selection;
# the headline result always uses these values.
# ---------------------------------------------------------------------------
WINDOW_WEEKS = 26  # weekly observations in the rolling mean and std
ENTRY_Z = 1.5  # enter "short premium" when z > ENTRY_Z
EXIT_Z = 0.0  # exit when z < EXIT_Z
MAX_HOLD_WEEKS = 26  # or after this many weekly decisions in position
# One direction only (no "long premium" trade): short selling in Taiwan is more
# constrained (not verified), whereas shorting the ADR is easy according to
# Lamont (2024).

COST_COMPONENTS = ["adr_commission", "local_commission", "local_tax", "fx_conversion",
                   "borrow_fee", "withholding_tax"]
PNL_COMPONENTS = ["price_fx", "dividends_gross"] + COST_COMPONENTS


@dataclass(frozen=True)
class Rules:
    window: int = WINDOW_WEEKS
    entry_z: float = ENTRY_Z
    exit_z: float = EXIT_Z
    max_hold: int = MAX_HOLD_WEEKS
    extra_lag: int = 0  # extra trading days of execution delay on each leg


@dataclass(frozen=True)
class Costs:
    """Trading frictions, as fractions of traded value unless stated otherwise."""
    # ADR commission per transaction. Assumption, not verified.
    adr_commission: float = 5e-4
    # Local commission per transaction. Assumption, not verified (the regulatory
    # ceiling is 0.1425%; discounted rates are common).
    local_commission: float = 10e-4
    # Taiwan securities transaction tax, sales of shares only. 0.3% of the sale
    # amount; assumption, not verified against an official source here.
    local_sale_tax: float = 30e-4
    # Each USD/TWD conversion. Assumption, not verified.
    fx_conversion: float = 5e-4
    # ADR borrow fee, per year, on the initial short notional, pro rata on calendar
    # days. Assumption, not verified.
    adr_borrow_fee: float = 0.005
    # Withholding tax on local dividends received by a non-resident. Assumption, not
    # verified (21% general rate since 2018, 20% before; treaties can lower it).
    dividend_withholding: float = 0.21
    # Whether the ADR short pays the lender the dividend net of withholding (True) or
    # the gross ADR dividend (False, default). Assumption, not verified.
    short_pays_net_dividend: bool = False

    @classmethod
    def zero(cls) -> "Costs":
        return cls(adr_commission=0.0, local_commission=0.0, local_sale_tax=0.0, fx_conversion=0.0,
                   adr_borrow_fee=0.0, dividend_withholding=0.0)


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def weekly_signal(premium: pd.Series, window: int = WINDOW_WEEKS) -> pd.DataFrame:
    """Weekly q, its trailing mean and std, and z, on the last Taipei date of each week.

    The benchmark uses the ``window`` previous non-empty weeks (``shift(1)``): an empty
    week (a market closure) is skipped, not counted.
    """
    q = np.log1p(_clean(premium))
    weekly = q.groupby(q.index.to_period("W-FRI")).tail(1)
    prior = weekly.shift(1).rolling(window, min_periods=window)
    out = pd.DataFrame({"q": weekly, "mean": prior.mean(), "std": prior.std()})
    out["z"] = (out["q"] - out["mean"]) / out["std"]
    out.index.name = "decision_date"
    return out


# ---------------------------------------------------------------------------
# Execution dates
# ---------------------------------------------------------------------------
def execution_dates(decisions: pd.DatetimeIndex, ny_dates: pd.DatetimeIndex,
                    tw_dates: pd.DatetimeIndex, extra_lag: int = 0) -> pd.DataFrame:
    """First executable close of each leg after a decision taken at the Taipei close.

    ADR: first NY date >= d (the NY close of d is ~15 h after the Taipei close of d).
    Local: first Taipei date > d. NaT when the data ends before.
    """
    ny = pd.DatetimeIndex(ny_dates).sort_values()
    tw = pd.DatetimeIndex(tw_dates).sort_values()
    i_ny = ny.searchsorted(decisions, side="left") + extra_lag
    i_tw = tw.searchsorted(decisions, side="right") + extra_lag

    def pick(idx, i):
        return pd.DatetimeIndex([idx[k] if k < len(idx) else pd.NaT for k in i])

    return pd.DataFrame({"adr_exec": pick(ny, i_ny), "local_exec": pick(tw, i_tw)},
                        index=pd.DatetimeIndex(decisions, name="decision_date"))


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
def generate_trades(signal: pd.DataFrame, execs: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Entry and exit decisions from the weekly signal (one position at a time).

    Every decision at week t reads only z_t, which uses data up to the Taipei close of
    t. A decision whose execution falls after the end of the data is skipped; a
    position still open at the end is closed at the last executable decision.
    """
    executable = execs["adr_exec"].notna() & execs["local_exec"].notna()
    rows, open_trade = [], None
    dates = signal.index
    for i, d in enumerate(dates):
        z = signal["z"].iloc[i]
        if not executable.loc[d]:
            continue
        if open_trade is None:
            if np.isfinite(z) and z > rules.entry_z:
                open_trade = {"entry_decision": d, "entry_z": z, "i0": i}
            continue
        held = i - open_trade["i0"]
        reason = None
        if np.isfinite(z) and z < rules.exit_z:
            reason = "z_below_exit"
        elif held >= rules.max_hold:
            reason = "max_hold"
        if reason:
            rows.append({**open_trade, "exit_decision": d, "exit_z": z, "exit_reason": reason,
                         "holding_weeks": held})
            open_trade = None
    if open_trade is not None:
        last = executable[executable].index.max()
        if last > open_trade["entry_decision"]:
            i = dates.get_loc(last)
            rows.append({**open_trade, "exit_decision": last, "exit_z": signal["z"].iloc[i],
                         "exit_reason": "end_of_sample", "holding_weeks": i - open_trade["i0"]})
    cols = ["entry_decision", "exit_decision", "exit_reason", "entry_z", "exit_z", "holding_weeks"]
    trades = pd.DataFrame(rows, columns=cols + ["i0"]).drop(columns="i0")
    if trades.empty:
        return trades
    entry = execs.loc[trades["entry_decision"]]
    exit_ = execs.loc[trades["exit_decision"]]
    trades["adr_entry_date"] = entry["adr_exec"].to_numpy()
    trades["local_entry_date"] = entry["local_exec"].to_numpy()
    trades["adr_exit_date"] = exit_["adr_exec"].to_numpy()
    trades["local_exit_date"] = exit_["local_exec"].to_numpy()
    return trades


# ---------------------------------------------------------------------------
# P&L of one trade
# ---------------------------------------------------------------------------
@dataclass
class Market:
    """Daily inputs on a common calendar (union of NY and Taipei trading dates)."""
    adr: pd.Series  # USD, NY dates
    local: pd.Series  # TWD, Taipei dates
    fx: pd.Series  # TWD per USD, FRED dates
    dividends: pd.DataFrame  # columns adr (USD/ADR), local (TWD/share), by ex-date
    calendar: pd.DatetimeIndex
    ratio: int = ADR_RATIO

    def __post_init__(self):
        self._fx_dates = self.fx.index.to_numpy()
        self._fx_values = self.fx.to_numpy()
        self._div = {}
        for col in ("adr", "local"):
            d = self.dividends[col].dropna() if col in self.dividends else pd.Series(dtype=float)
            d.index = pd.DatetimeIndex(d.index)
            self._div[col] = d.astype(float)

    @classmethod
    def build(cls, adr, local, fx, dividends=None, ratio: int = ADR_RATIO) -> "Market":
        adr, local, fx = _clean(adr), _clean(local), _clean(fx)
        cal = adr.index.union(local.index)
        if dividends is None:
            dividends = pd.DataFrame(columns=["adr", "local"], dtype=float)
        return cls(adr, local, fx, dividends, cal, ratio)

    def fx_at(self, date) -> float:
        """FRED rate at or before ``date``."""
        i = self._fx_dates.searchsorted(np.datetime64(date), side="right") - 1
        return float(self._fx_values[i]) if i >= 0 else np.nan

    def dividends_between(self, col: str, start, end) -> pd.Series:
        """Ex-dates in (start, end]: a holder at the close before the ex-date is paid."""
        d = self._div[col]
        return d[(d.index > start) & (d.index <= end)]

    def on_calendar(self, s: pd.Series) -> pd.Series:
        """Last value at or before each calendar date."""
        return s.reindex(s.index.union(self.calendar)).ffill().reindex(self.calendar)


@dataclass
class TradeTerms:
    """Execution prices, rates and dated one-off flows of one trade."""
    a_in: pd.Timestamp
    a_out: pd.Timestamp
    l_in: pd.Timestamp
    l_out: pd.Timestamp
    pa_in: float
    pa_out: float
    pl_in: float
    pl_out: float
    fx_in: float
    fx_out: float
    shares: float  # local shares bought with 1 USD
    local_exit_usd: float  # USD proceeds of the local sale
    flows: list  # (date, component, signed amount)
    borrow_total: float  # signed, over the whole holding period


def trade_terms(trade, mkt: Market, costs: Costs) -> TradeTerms:
    """Execution terms of one trade.

    The local leg is converted at the FRED rate of its own execution dates: 1 USD is
    converted to TWD on the local entry date, and the sale proceeds back to USD on the
    local exit date.

    Financing assumption: the proceeds of the ADR short sale fund the purchase of the
    local share. No interest is earned on the short proceeds and no financing cost is
    charged on the long leg; the only carry modelled is the ADR borrow fee.
    """
    a_in, a_out = pd.Timestamp(trade["adr_entry_date"]), pd.Timestamp(trade["adr_exit_date"])
    l_in, l_out = pd.Timestamp(trade["local_entry_date"]), pd.Timestamp(trade["local_exit_date"])
    pa_in, pa_out = float(mkt.adr.loc[a_in]), float(mkt.adr.loc[a_out])
    pl_in, pl_out = float(mkt.local.loc[l_in]), float(mkt.local.loc[l_out])
    fx_in, fx_out = mkt.fx_at(l_in), mkt.fx_at(l_out)
    shares = fx_in / pl_in  # 1 USD converted to TWD, then local shares bought
    local_exit_usd = shares * pl_out / fx_out
    wht = costs.dividend_withholding

    flows = []
    for ex, amount in mkt.dividends_between("local", l_in, l_out).items():
        gross = shares * amount / mkt.fx_at(ex)
        flows += [(ex, "dividends_gross", gross), (ex, "withholding_tax", -wht * gross)]
    for ex, amount in mkt.dividends_between("adr", a_in, a_out).items():
        gross = amount / pa_in
        flows.append((ex, "dividends_gross", -gross))
        if costs.short_pays_net_dividend:
            # The short owes the dividend net of withholding: the saving is booked
            # against withholding_tax, so the gross P&L is the same in both cases.
            flows.append((ex, "withholding_tax", wht * gross))

    flows += [
        (a_in, "adr_commission", -costs.adr_commission * 1.0),
        (a_out, "adr_commission", -costs.adr_commission * pa_out / pa_in),
        (l_in, "local_commission", -costs.local_commission * 1.0),
        (l_out, "local_commission", -costs.local_commission * local_exit_usd),
        # The transaction tax applies to sales of the local share only (here: the exit).
        (l_out, "local_tax", -costs.local_sale_tax * local_exit_usd),
        (l_in, "fx_conversion", -costs.fx_conversion * 1.0),  # USD -> TWD
        (l_out, "fx_conversion", -costs.fx_conversion * local_exit_usd),  # TWD -> USD
    ]
    borrow = -costs.adr_borrow_fee * (a_out - a_in).days / 365.0
    return TradeTerms(a_in, a_out, l_in, l_out, pa_in, pa_out, pl_in, pl_out, fx_in, fx_out,
                      shares, local_exit_usd, flows, borrow)


def trade_final(trade, mkt: Market, costs: Costs) -> pd.Series:
    """Final P&L components of one trade (fast: no daily path)."""
    t = trade_terms(trade, mkt, costs)
    out = dict.fromkeys(PNL_COMPONENTS, 0.0)
    out["price_fx"] = (1 - t.pa_out / t.pa_in) + (t.local_exit_usd - 1)
    for _, col, amount in t.flows:
        out[col] += amount
    out["borrow_fee"] = t.borrow_total
    return pd.Series(out)


def trade_path(trade, mkt: Market, costs: Costs) -> pd.DataFrame:
    """Cumulative P&L components of one trade on every calendar date (USD per 1 USD leg).

    Components (costs are negative):
    - price_fx: mark-to-market of both legs on their own closes, the local leg
      converted at the FRED rate of each date while held;
    - dividends_gross: local dividends received (gross) minus ADR dividends paid by the
      short, booked on the ex-date. Both legs drop by their dividend on that date, so
      price_fx + dividends_gross is the total return before frictions;
    - withholding_tax, adr_commission, local_commission, local_tax (sales only),
      fx_conversion, borrow_fee.
    """
    t = trade_terms(trade, mkt, costs)
    cal = mkt.calendar
    out = pd.DataFrame(0.0, index=cal, columns=PNL_COMPONENTS)

    # Short ADR: 1 USD sold at pa_in.
    adr_pnl = 1 - mkt.on_calendar(mkt.adr) / t.pa_in
    adr_pnl[cal < t.a_in] = 0.0
    adr_pnl[cal >= t.a_out] = 1 - t.pa_out / t.pa_in
    # Long local: valued at the FRED rate of each date while held.
    loc_pnl = t.shares * mkt.on_calendar(mkt.local) / mkt.on_calendar(mkt.fx) - 1
    loc_pnl[cal < t.l_in] = 0.0
    loc_pnl[cal >= t.l_out] = t.local_exit_usd - 1
    out["price_fx"] = adr_pnl + loc_pnl

    for date, col, amount in t.flows:
        out.loc[cal >= date, col] += amount

    # Borrow fee on the initial short notional, pro rata on calendar days.
    days = np.clip((cal - t.a_in).days, 0, (t.a_out - t.a_in).days)
    out["borrow_fee"] = -costs.adr_borrow_fee * days / 365.0
    return out


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------
@dataclass
class Result:
    signal: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame  # daily P&L by component, plus gross, net and in_position
    start: pd.Timestamp  # first date with a valid signal


def run_backtest(adr: pd.Series, local: pd.Series, fx: pd.Series, dividends=None,
                 rules: Rules = Rules(), costs: Costs = Costs(),
                 ratio: int = ADR_RATIO) -> Result:
    mkt = Market.build(adr, local, fx, dividends, ratio)
    prem = premium_prev_ny(mkt.adr, mkt.local, mkt.fx, ratio=ratio)
    signal = weekly_signal(prem["premium"], rules.window)
    execs = execution_dates(signal.index, mkt.adr.index, mkt.local.index, rules.extra_lag)
    signal = signal.join(execs)
    trades = generate_trades(signal, execs, rules)

    cum = pd.DataFrame(0.0, index=mkt.calendar, columns=PNL_COMPONENTS)
    in_pos = pd.Series(False, index=mkt.calendar)
    records = []
    for _, t in trades.iterrows():
        path = trade_path(t, mkt, costs)
        cum += path
        first = min(t["adr_entry_date"], t["local_entry_date"])
        last = max(t["adr_exit_date"], t["local_exit_date"])
        in_pos |= (mkt.calendar >= first) & (mkt.calendar < last)
        final = path.iloc[-1]
        records.append({
            "adr_entry_price": float(mkt.adr.loc[t["adr_entry_date"]]),
            "local_entry_price": float(mkt.local.loc[t["local_entry_date"]]),
            "fx_entry": mkt.fx_at(t["local_entry_date"]),
            "adr_exit_price": float(mkt.adr.loc[t["adr_exit_date"]]),
            "local_exit_price": float(mkt.local.loc[t["local_exit_date"]]),
            "fx_exit": mkt.fx_at(t["local_exit_date"]),
            "holding_days": (last - first).days,
            **final.to_dict(),
        })
    if len(trades):
        trades = pd.concat([trades, pd.DataFrame(records, index=trades.index)], axis=1)
        trades["gross_pnl"] = trades["price_fx"] + trades["dividends_gross"]
        trades["net_pnl"] = trades[PNL_COMPONENTS].sum(axis=1)

    daily = cum.diff()
    daily.iloc[0] = cum.iloc[0]
    daily["gross"] = daily["price_fx"] + daily["dividends_gross"]
    daily["net"] = daily[PNL_COMPONENTS].sum(axis=1)
    daily["in_position"] = in_pos
    valid = signal["z"].dropna()
    start = valid.index[0] if len(valid) else mkt.calendar[-1]
    return Result(signal, trades, daily.loc[start:], start)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(daily: pd.DataFrame, trades: pd.DataFrame, col: str = "net") -> dict:
    """Performance of a daily P&L column (USD per 1 USD per leg, no compounding)."""
    pnl = daily[col]
    n = len(pnl)
    years = (daily.index[-1] - daily.index[0]).days / 365.25 if n > 1 else np.nan
    per_year = n / years if years and years > 0 else np.nan
    mean, std = pnl.mean(), pnl.std()
    cum = pnl.cumsum()
    pnl_col = "net_pnl" if col == "net" else "gross_pnl"
    # The daily marks of the two legs are asynchronous (Taipei vs NY closes), which adds
    # day-to-day noise that reverses: weekly sums give a less noisy volatility.
    weekly = pnl.resample("W-FRI").sum()
    w_std = weekly.std()
    return {
        "ann_return": mean * per_year,
        "ann_vol": std * np.sqrt(per_year),
        "sharpe": mean / std * np.sqrt(per_year) if std > 0 else np.nan,
        "ann_vol_weekly": w_std * np.sqrt(52),
        "sharpe_weekly": weekly.mean() / w_std * np.sqrt(52) if w_std > 0 else np.nan,
        "max_drawdown": float((cum - cum.cummax()).min()) if n else np.nan,
        "total_pnl": float(pnl.sum()),
        "n_trades": len(trades),
        "avg_holding_weeks": trades["holding_weeks"].mean() if len(trades) else np.nan,
        "win_rate": (trades[pnl_col] > 0).mean() if len(trades) else np.nan,
        "exposure": daily["in_position"].mean() if n else np.nan,
        "years": years,
    }


def cost_breakdown(daily: pd.DataFrame) -> pd.Series:
    """Total gross P&L, each cost's impact and net P&L over the sample."""
    s = daily[["gross"] + COST_COMPONENTS + ["net"]].sum()
    return s


def check_dividends(dividends: pd.DataFrame, fx: pd.Series, ratio: int = ADR_RATIO,
                    tol: float = 0.05, max_gap_days: int = 5) -> pd.DataFrame:
    """Compare each ADR dividend with ``ratio`` times the local one converted to USD.

    The local ex-date is matched to the nearest one within ``max_gap_days``; the rate
    is FRED at or before the ADR ex-date. ``flag`` marks relative gaps above ``tol`` and
    ADR dividends without a matching local one.
    """
    adr = dividends["adr"].dropna()
    loc = dividends["local"].dropna()
    fx = _clean(fx)
    rows = []
    for ex, amount in adr.items():
        gaps = np.abs((loc.index - ex).days) if len(loc) else np.array([])
        if len(gaps) and gaps.min() <= max_gap_days:
            k = int(gaps.argmin())
            implied = ratio * loc.iloc[k] / float(fx.asof(ex))
            rows.append({"adr_ex_date": ex, "local_ex_date": loc.index[k], "adr_dividend": amount,
                         "local_dividend": loc.iloc[k], "implied_usd": implied,
                         "rel_diff": amount / implied - 1})
        else:
            rows.append({"adr_ex_date": ex, "local_ex_date": pd.NaT, "adr_dividend": amount,
                         "local_dividend": np.nan, "implied_usd": np.nan, "rel_diff": np.nan})
    out = pd.DataFrame(rows)
    out["flag"] = out["rel_diff"].isna() | (out["rel_diff"].abs() > tol)
    return out
