"""No-arbitrage band of the premium (roadmap step 7): a derived calculation, not a test.

If the premium's deviation D from its mean decays with half-life h (trading days), a
position held T trading days captures on average D (1 - 0.5^(T / h)). It pays the
round-trip trading cost C and the ADR borrow fee b (per year) for T days. The smallest
deviation that covers these costs is

    D* = (C + b T / 252) / (1 - 0.5^(T / h)).

D and C are in the units of the backtest: USD per 1 USD per leg, which for a short-
premium trade is approximately the change in q = ln(1 + premium). The calculation
ignores risk, financing other than the borrow fee, dividends and the asynchrony noise
of the observed premium; it is an order of magnitude, not a trading rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adr_premium.backtest import Costs, weekly_signal

TRADING_DAYS_PER_YEAR = 252


def round_trip_cost(costs: Costs) -> float:
    """Trading cost of one round trip per 1 USD per leg, at unchanged prices.

    Two ADR commissions, two local commissions, the sale tax on the local leg and two
    FX conversions; the borrow fee and dividend withholding are not included.
    """
    return (2 * costs.adr_commission + 2 * costs.local_commission + costs.local_sale_tax
            + 2 * costs.fx_conversion)


def breakeven_deviation(cost: float, borrow_fee: float, half_life: float, holding: float) -> float:
    """D* for a holding period ``holding`` and a half-life ``half_life``, both in trading days."""
    captured = 1 - 0.5 ** (holding / half_life)
    return (cost + borrow_fee * holding / TRADING_DAYS_PER_YEAR) / captured


def deviations(premium: pd.Series, window: int = 26) -> pd.Series:
    """Weekly q minus its mean over the ``window`` previous weeks, as in the strategy signal."""
    sig = weekly_signal(premium, window)
    return (sig["q"] - sig["mean"]).dropna()


def share_above(dev: pd.Series, threshold: float) -> dict:
    """Share of weeks with a deviation above ``threshold`` (premium too high: the
    short-premium side) and with an absolute deviation above it (both sides)."""
    x = dev.to_numpy()
    return {"share_above": float(np.mean(x > threshold)),
            "share_abs_above": float(np.mean(np.abs(x) > threshold))}
