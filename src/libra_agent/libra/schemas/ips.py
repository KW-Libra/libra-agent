from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class IPSConfig:
    single_ticker_limit_pct: float = 25.0
    sector_limit_pct: float = 40.0
    annual_volatility_limit: float = 0.20
    asset_class_target: dict[str, float] = field(default_factory=dict)
    asset_class_band_pct: float = 10.0
    min_cash_pct: float = 5.0
    max_market_impact_pct_of_adv: float = 5.0
    excluded_tickers: list[str] = field(default_factory=list)
    excluded_sectors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KYCProfile:
    risk_tolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] = "MODERATE"
    investment_horizon_years: int = 15
    max_drawdown_tolerance_pct: float = 15.0
