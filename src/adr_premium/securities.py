"""Securities covered by the project: tickers, ADS ratio and sample dates.

Ratios and dates for UMC, ASE and Chunghwa are those of the step-6 pre-registration
(docs/preregistration_step6.md, tag prereg-step6), checked in 20-F filings. TSMC keeps
the original cache file names, so its results do not depend on this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STEP6_END = "2026-10-30"  # last date of the TSMC sample


@dataclass(frozen=True)
class Security:
    key: str
    name: str
    adr: str  # NYSE ticker
    local: str  # TWSE ticker on Yahoo
    ratio: int  # common shares per ADS
    start: str  # first date of the sample
    end: str | None = None  # last date of the sample (inclusive); None = no limit

    @property
    def cache_suffix(self) -> str:
        """'' for TSMC, so that its original cache files are used unchanged."""
        return "" if self.key == "tsmc" else f"_{self.key}"

    @property
    def download_end(self) -> str | None:
        """yfinance end date (exclusive) covering ``end``."""
        return None if self.end is None else str((pd.Timestamp(self.end) + pd.Timedelta(days=1)).date())

    def clip(self, df):
        """Restrict a date-indexed frame or series to [start, end]."""
        return df.loc[self.start:self.end]


TSMC = Security("tsmc", "TSMC", "TSM", "2330.TW", 5, "2010-01-01")
UMC = Security("umc", "United Microelectronics", "UMC", "2303.TW", 5, "2010-01-01", STEP6_END)
# ASE Technology Holding exists since 2018-04-30; earlier "ASX" data is ASE Inc., a different security.
ASE = Security("ase", "ASE Technology Holding", "ASX", "3711.TW", 2, "2018-04-30", STEP6_END)
# Chunghwa's last capital reduction was paid on 2011-01-25; the sample starts after it.
CHT = Security("cht", "Chunghwa Telecom", "CHT", "2412.TW", 10, "2011-03-01", STEP6_END)

SECURITIES = {s.key: s for s in (TSMC, UMC, ASE, CHT)}
STEP6 = [UMC, ASE, CHT]
