from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Severity(str, Enum):
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


@dataclass(slots=True)
class ComplianceViolation:
    rule_id: str
    severity: Severity
    description: str
    affected_subjects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "description": self.description,
            "affected_subjects": list(self.affected_subjects),
        }


@dataclass(slots=True)
class ComplianceCheck:
    can_proceed: bool
    state: Literal["BEFORE", "AFTER"]
    violations: list[ComplianceViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "can_proceed": self.can_proceed,
            "violations": [violation.to_dict() for violation in self.violations],
            "state": self.state,
        }


@dataclass(slots=True)
class MarketSnapshot:
    prices: dict[str, float] = field(default_factory=dict)
    avg_daily_volume: dict[str, float] = field(default_factory=dict)
    krx_status: dict[str, str] = field(default_factory=dict)
    sector_map: dict[str, str] = field(default_factory=dict)
    esg_score: dict[str, float] = field(default_factory=dict)
    volatility: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ComplianceContext:
    proposed_trades: list[object]
    before_portfolio: dict[str, float]
    after_portfolio: dict[str, float]
    cash_balance_pct: float
    user_ips: object
    user_profile: object
    user_exclusions: list[str] = field(default_factory=list)
    market_data: MarketSnapshot = field(default_factory=MarketSnapshot)
