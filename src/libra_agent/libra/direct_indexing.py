from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..libra_models import PortfolioSnapshot

RISK_PROFILE_THRESHOLDS = {
    "안정형": 0.02,
    "conservative": 0.02,
    "안정추구형": 0.03,
    "moderately_conservative": 0.03,
    "위험중립형": 0.05,
    "neutral": 0.05,
    "적극투자형": 0.07,
    "aggressive": 0.07,
    "공격투자형": 0.10,
    "very_aggressive": 0.10,
}


def _normalize_ticker(value: str) -> str:
    return "".join(char for char in str(value).upper() if char.isalnum())


@dataclass(slots=True, frozen=True)
class TargetWeight:
    ticker: str
    company_name: str
    weight: float
    market: str = "KR"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetWeight:
        stock = payload.get("stock") if isinstance(payload.get("stock"), Mapping) else {}
        ticker = _normalize_ticker(str(payload.get("ticker") or stock.get("ticker") or ""))
        company_name = str(
            payload.get("company_name")
            or payload.get("name")
            or stock.get("company_name")
            or stock.get("name")
            or ticker
        ).strip()
        market = str(payload.get("market") or stock.get("market") or "KR").strip() or "KR"
        try:
            weight = float(payload.get("weight", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("target_weights[].weight must be a number.") from exc
        if not ticker:
            raise ValueError("target_weights[].ticker is required.")
        if weight < 0 or weight > 1:
            raise ValueError("target_weights[].weight must be between 0 and 1.")
        return cls(ticker=ticker, company_name=company_name or ticker, weight=weight, market=market)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "weight": round(float(self.weight), 6),
            "market": self.market,
        }


@dataclass(slots=True, frozen=True)
class PortfolioDefinition:
    name: str
    target_weights: tuple[TargetWeight, ...]
    description: str = ""
    risk_profile: str = "위험중립형"
    drift_threshold: float = 0.05
    rebalancing_frequency: str = "임계치 도달 시"
    threshold_overridden: bool = False
    created_at: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioDefinition:
        profile = payload.get("profile") if isinstance(payload.get("profile"), Mapping) else {}
        risk_profile = str(
            payload.get("risk_profile") or profile.get("risk_profile") or "위험중립형"
        )
        default_threshold = RISK_PROFILE_THRESHOLDS.get(risk_profile.casefold(), 0.05)
        try:
            threshold = float(
                payload.get("drift_threshold")
                or profile.get("drift_threshold")
                or default_threshold
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("portfolio_definition.drift_threshold must be a number.") from exc

        raw_targets = payload.get("target_weights", [])
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("portfolio_definition.target_weights must contain at least one item.")
        targets = tuple(
            TargetWeight.from_dict(item) for item in raw_targets if isinstance(item, Mapping)
        )
        if not targets:
            raise ValueError(
                "portfolio_definition.target_weights must contain at least one valid item."
            )

        total_weight = sum(item.weight for item in targets)
        if not (0.999 <= total_weight <= 1.001):
            raise ValueError(
                f"portfolio_definition target weights must sum to 1.0. current={total_weight:.4f}"
            )
        if len({item.ticker for item in targets}) != len(targets):
            raise ValueError("portfolio_definition target_weights contains duplicate tickers.")
        if threshold <= 0 or threshold > 0.5:
            raise ValueError("portfolio_definition drift_threshold must be in (0, 0.5].")

        return cls(
            name=str(payload.get("name") or "Direct Index").strip(),
            description=str(payload.get("description") or ""),
            target_weights=targets,
            risk_profile=risk_profile,
            drift_threshold=threshold,
            rebalancing_frequency=str(
                payload.get("rebalancing_frequency")
                or profile.get("rebalancing_frequency")
                or "임계치 도달 시"
            ),
            threshold_overridden=bool(
                payload.get("threshold_overridden") or profile.get("threshold_overridden")
            ),
            created_at=str(payload.get("created_at"))
            if payload.get("created_at") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_weights": [item.to_dict() for item in self.target_weights],
            "risk_profile": self.risk_profile,
            "drift_threshold": round(float(self.drift_threshold), 6),
            "rebalancing_frequency": self.rebalancing_frequency,
            "threshold_overridden": self.threshold_overridden,
            "created_at": self.created_at,
        }


@dataclass(slots=True, frozen=True)
class StockDrift:
    ticker: str
    company_name: str
    target_weight: float
    current_weight: float
    drift_abs: float
    drift_rel: float
    direction: str
    severity: str
    market_value_krw: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "target_weight": round(float(self.target_weight), 6),
            "current_weight": round(float(self.current_weight), 6),
            "drift_abs": round(float(self.drift_abs), 6),
            "drift_rel": self.drift_rel
            if math.isinf(self.drift_rel)
            else round(float(self.drift_rel), 6),
            "direction": self.direction,
            "severity": self.severity,
            "market_value_krw": self.market_value_krw,
        }


@dataclass(slots=True, frozen=True)
class DriftReport:
    definition_name: str
    threshold: float
    stock_drifts: tuple[StockDrift, ...]
    portfolio_drift_l1: float
    portfolio_drift_l2: float
    portfolio_drift_rms: float
    portfolio_drift_max: float
    computed_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_name": self.definition_name,
            "threshold": round(float(self.threshold), 6),
            "stock_drifts": [item.to_dict() for item in self.stock_drifts],
            "portfolio_drift_l1": round(float(self.portfolio_drift_l1), 6),
            "portfolio_drift_l2": round(float(self.portfolio_drift_l2), 6),
            "portfolio_drift_rms": round(float(self.portfolio_drift_rms), 6),
            "portfolio_drift_max": round(float(self.portfolio_drift_max), 6),
            "computed_at": self.computed_at,
        }


def compute_drift(definition: PortfolioDefinition, portfolio: PortfolioSnapshot) -> DriftReport:
    target_map = {item.ticker: item for item in definition.target_weights}
    holding_map = {_normalize_ticker(item.ticker): item for item in portfolio.holdings}
    all_tickers = sorted({*target_map.keys(), *holding_map.keys()})
    stock_drifts: list[StockDrift] = []

    for ticker in all_tickers:
        target = target_map.get(ticker)
        holding = holding_map.get(ticker)
        target_weight = target.weight if target else 0.0
        current_weight = holding.weight if holding else 0.0
        if target_weight == 0.0 and current_weight == 0.0:
            continue
        drift_abs = current_weight - target_weight
        drift_rel = math.inf if target_weight == 0 else drift_abs / target_weight
        stock_drifts.append(
            StockDrift(
                ticker=ticker,
                company_name=(
                    target.company_name if target else holding.company_name if holding else ticker
                ),
                target_weight=target_weight,
                current_weight=current_weight,
                drift_abs=drift_abs,
                drift_rel=drift_rel,
                direction=_direction(drift_abs, definition.drift_threshold),
                severity=_severity(drift_abs, definition.drift_threshold),
                market_value_krw=getattr(holding, "market_value_krw", None) if holding else None,
            )
        )

    l1 = sum(abs(item.drift_abs) for item in stock_drifts)
    l2 = math.sqrt(sum(item.drift_abs**2 for item in stock_drifts))
    rms = (
        math.sqrt(sum(item.drift_abs**2 for item in stock_drifts) / len(stock_drifts))
        if stock_drifts
        else 0.0
    )
    max_drift = max((abs(item.drift_abs) for item in stock_drifts), default=0.0)
    return DriftReport(
        definition_name=definition.name,
        threshold=definition.drift_threshold,
        stock_drifts=tuple(stock_drifts),
        portfolio_drift_l1=l1,
        portfolio_drift_l2=l2,
        portfolio_drift_rms=rms,
        portfolio_drift_max=max_drift,
    )


def candidate_plan_from_drift(report: DriftReport) -> dict[str, float]:
    plan: dict[str, float] = {}
    for item in report.stock_drifts:
        if item.severity not in {"MODERATE", "SEVERE"}:
            continue
        trade_delta = -item.drift_abs
        if abs(trade_delta) >= report.threshold:
            plan[item.ticker] = round(trade_delta, 6)
    return plan


def compact_drift_context(report: DriftReport | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    payload = report.to_dict() if isinstance(report, DriftReport) else dict(report)
    stock_drifts = payload.get("stock_drifts", [])
    return {
        "portfolio_drift_l1": round(float(payload.get("portfolio_drift_l1") or 0.0), 6),
        "portfolio_drift_max": round(float(payload.get("portfolio_drift_max") or 0.0), 6),
        "threshold": round(float(payload.get("threshold") or 0.0), 6),
        "stock_drifts": [
            {
                "ticker": str(item.get("ticker") or ""),
                "target_weight": round(float(item.get("target_weight") or 0.0), 4),
                "current_weight": round(float(item.get("current_weight") or 0.0), 4),
                "drift_abs": round(float(item.get("drift_abs") or 0.0), 4),
                "direction": str(item.get("direction") or ""),
                "severity": str(item.get("severity") or ""),
            }
            for item in stock_drifts
            if isinstance(item, Mapping)
            and str(item.get("severity") or "") in {"MODERATE", "SEVERE"}
        ][:8],
    }


def _direction(drift_abs: float, threshold: float) -> str:
    if abs(drift_abs) < threshold:
        return "IN_BAND"
    return "OVERWEIGHT" if drift_abs > 0 else "UNDERWEIGHT"


def _severity(drift_abs: float, threshold: float) -> str:
    abs_drift = abs(drift_abs)
    if abs_drift < threshold:
        return "MINOR"
    if abs_drift < threshold * 2:
        return "MODERATE"
    return "SEVERE"
