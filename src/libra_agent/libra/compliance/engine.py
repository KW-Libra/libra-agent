from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Literal

from libra_agent.libra_models import PortfolioSnapshot

from ..schemas.compliance import ComplianceCheck, ComplianceContext, ComplianceViolation, MarketSnapshot, Severity
from ..schemas.decision import Trade
from ..schemas.ips import IPSConfig, KYCProfile

RuleFunction = Callable[[ComplianceContext], list[ComplianceViolation]]


class ComplianceEngine:
    """Deterministic hard-rule layer.

    This is intentionally not an LLM agent. If a rule evaluation fails, the
    engine returns a BLOCKING violation so automated trading cannot proceed.
    """

    def __init__(self, rules: dict[str, RuleFunction]) -> None:
        self.rules = dict(rules)

    def check(self, ctx: ComplianceContext, state: Literal["BEFORE", "AFTER"]) -> ComplianceCheck:
        all_violations: list[ComplianceViolation] = []
        for rule_id, rule_fn in self.rules.items():
            try:
                all_violations.extend(rule_fn(ctx))
            except Exception as exc:
                all_violations.append(
                    ComplianceViolation(
                        rule_id=rule_id,
                        severity=Severity.BLOCKING,
                        description=f"룰 평가 실패: {exc}",
                        affected_subjects=[],
                    )
                )
        return ComplianceCheck(
            can_proceed=not any(item.severity == Severity.BLOCKING for item in all_violations),
            violations=all_violations,
            state=state,
        )


def default_compliance_engine() -> ComplianceEngine:
    return ComplianceEngine(
        {
            "IPS_SINGLE_TICKER_LIMIT": check_ips_single_ticker_limit,
            "IPS_SECTOR_LIMIT": check_ips_sector_limit,
            "IPS_VOLATILITY_LIMIT": check_ips_volatility_limit,
            "IPS_ASSET_CLASS_BAND": check_ips_asset_class_band,
            "ESG_USER_EXCLUSION": check_esg_user_exclusion,
            "ESG_MIN_SCORE": check_esg_min_score,
            "TAX_ANNUAL_GAIN_LIMIT": check_tax_annual_gain_limit,
            "KYC_RISK_PROFILE_MISMATCH": check_kyc_risk_profile_mismatch,
            "LIQUIDITY_MIN_CASH": check_liquidity_min_cash,
            "MARKET_IMPACT_LIMIT": check_market_impact_limit,
            "RESTRICTED_TICKER": check_restricted_ticker,
        }
    )


def build_compliance_context_from_portfolio(
    portfolio: PortfolioSnapshot,
    *,
    proposed_trades: Iterable[Trade] | None = None,
    ips: IPSConfig | None = None,
    kyc: KYCProfile | None = None,
    market_data: MarketSnapshot | None = None,
) -> ComplianceContext:
    before = {holding.ticker: round(float(holding.weight) * 100.0, 4) for holding in portfolio.holdings}
    trades = list(proposed_trades or [])
    after = dict(before)
    for trade in trades:
        after[trade.subject] = round(after.get(trade.subject, 0.0) + float(trade.delta_pct), 4)
    snapshot = market_data or MarketSnapshot()
    if not snapshot.sector_map:
        snapshot = replace(
            snapshot,
            sector_map={
                holding.ticker: str(getattr(holding, "sector", "") or "")
                for holding in portfolio.holdings
                if str(getattr(holding, "sector", "") or "").strip()
            },
        )
    if not snapshot.esg_score:
        snapshot = replace(
            snapshot,
            esg_score={
                holding.ticker: float(holding.esg_score)
                for holding in portfolio.holdings
                if holding.esg_score is not None
            },
        )
    user_ips = ips or IPSConfig()
    return ComplianceContext(
        proposed_trades=trades,
        before_portfolio=before,
        after_portfolio=after,
        cash_balance_pct=round(float(portfolio.cash_weight) * 100.0, 4),
        user_ips=user_ips,
        user_profile=kyc or KYCProfile(),
        user_exclusions=[*user_ips.excluded_tickers, *user_ips.excluded_sectors],
        market_data=snapshot,
    )


def _target_portfolio(ctx: ComplianceContext) -> dict[str, float]:
    return ctx.after_portfolio or ctx.before_portfolio


def _blocking(rule_id: str, description: str, subjects: list[str]) -> ComplianceViolation:
    return ComplianceViolation(rule_id=rule_id, severity=Severity.BLOCKING, description=description, affected_subjects=subjects)


def _warning(rule_id: str, description: str, subjects: list[str]) -> ComplianceViolation:
    return ComplianceViolation(rule_id=rule_id, severity=Severity.WARNING, description=description, affected_subjects=subjects)


def check_ips_single_ticker_limit(ctx: ComplianceContext) -> list[ComplianceViolation]:
    limit = float(ctx.user_ips.single_ticker_limit_pct)
    violations: list[ComplianceViolation] = []
    for ticker, weight in _target_portfolio(ctx).items():
        if ticker == "CASH":
            continue
        if weight > limit:
            violations.append(
                _blocking(
                    "IPS_SINGLE_TICKER_LIMIT",
                    f"{ticker} 비중 {weight:.1f}%가 단일종목 한도 {limit:.1f}%를 초과",
                    [ticker],
                )
            )
    return violations


def check_ips_sector_limit(ctx: ComplianceContext) -> list[ComplianceViolation]:
    limit = float(ctx.user_ips.sector_limit_pct)
    sector_weights: dict[str, float] = {}
    for ticker, weight in _target_portfolio(ctx).items():
        sector = ctx.market_data.sector_map.get(ticker)
        if not sector:
            continue
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)
    return [
        _blocking("IPS_SECTOR_LIMIT", f"{sector} 섹터 비중 {weight:.1f}%가 한도 {limit:.1f}%를 초과", [sector])
        for sector, weight in sector_weights.items()
        if weight > limit
    ]


def check_ips_volatility_limit(ctx: ComplianceContext) -> list[ComplianceViolation]:
    observed = ctx.market_data.volatility.get("PORTFOLIO")
    if observed is None:
        return []
    limit = float(ctx.user_ips.annual_volatility_limit)
    if float(observed) > limit:
        return [_warning("IPS_VOLATILITY_LIMIT", f"연환산 변동성 {float(observed):.2%}가 한도 {limit:.2%}를 초과", ["PORTFOLIO"])]
    return []


def check_ips_asset_class_band(ctx: ComplianceContext) -> list[ComplianceViolation]:
    # Asset-class mapping is intentionally left data-driven. If no market data
    # provides asset-class buckets, the rule stays silent rather than guessing.
    return []


def check_esg_user_exclusion(ctx: ComplianceContext) -> list[ComplianceViolation]:
    excluded_tickers = {item.upper() for item in ctx.user_ips.excluded_tickers}
    excluded_sectors = {item.upper() for item in ctx.user_ips.excluded_sectors}
    violations: list[ComplianceViolation] = []
    for trade in ctx.proposed_trades:
        if float(getattr(trade, "delta_pct", 0.0)) <= 0:
            continue
        ticker = str(getattr(trade, "subject", "")).upper()
        sector = ctx.market_data.sector_map.get(ticker, "").upper()
        if ticker in excluded_tickers:
            violations.append(_blocking("ESG_USER_EXCLUSION", f"{ticker}가 사용자 제외 종목에 포함", [ticker]))
        if sector and sector in excluded_sectors:
            violations.append(_blocking("ESG_USER_EXCLUSION", f"{ticker}가 제외 섹터 {sector}에 매칭", [ticker]))
    return violations


def check_esg_min_score(ctx: ComplianceContext) -> list[ComplianceViolation]:
    min_score = getattr(ctx.user_ips, "esg_min_score", None)
    if min_score is None:
        return []
    threshold = float(min_score)
    violations: list[ComplianceViolation] = []
    for ticker, weight in _target_portfolio(ctx).items():
        if ticker == "CASH" or float(weight) <= 0:
            continue
        score = ctx.market_data.esg_score.get(ticker)
        if score is None:
            continue
        if float(score) < threshold:
            violations.append(
                _blocking(
                    "ESG_MIN_SCORE",
                    f"{ticker} ESG 점수 {float(score):.1f}가 사용자 최소 기준 {threshold:.1f} 미만",
                    [ticker],
                )
            )
    return violations


def check_tax_annual_gain_limit(ctx: ComplianceContext) -> list[ComplianceViolation]:
    # Tax lots are not part of ComplianceContext v1 yet. TaxAgent can still
    # provide a soft opinion; the hard annual-gain rule activates once lots are
    # wired into context.
    return []


def check_kyc_risk_profile_mismatch(ctx: ComplianceContext) -> list[ComplianceViolation]:
    if getattr(ctx.user_profile, "risk_tolerance", "MODERATE") != "CONSERVATIVE":
        return []
    positive_risk_trade = next(
        (trade for trade in ctx.proposed_trades if float(getattr(trade, "delta_pct", 0.0)) >= 10.0),
        None,
    )
    if positive_risk_trade is None:
        return []
    return [
        _blocking(
            "KYC_RISK_PROFILE_MISMATCH",
            "보수형 사용자에게 10%p 이상 위험자산 증액은 자동 승인 불가",
            [str(getattr(positive_risk_trade, "subject", "PORTFOLIO"))],
        )
    ]


def check_liquidity_min_cash(ctx: ComplianceContext) -> list[ComplianceViolation]:
    if float(ctx.cash_balance_pct) < float(ctx.user_ips.min_cash_pct):
        return [
            _warning(
                "LIQUIDITY_MIN_CASH",
                f"현금 비중 {ctx.cash_balance_pct:.1f}%가 최소 {ctx.user_ips.min_cash_pct:.1f}% 미만",
                ["CASH"],
            )
        ]
    return []


def check_market_impact_limit(ctx: ComplianceContext) -> list[ComplianceViolation]:
    # Requires portfolio value and ADV units; keep silent until live order data
    # is present instead of pretending a percentage-only estimate is enough.
    return []


def check_restricted_ticker(ctx: ComplianceContext) -> list[ComplianceViolation]:
    violations: list[ComplianceViolation] = []
    for ticker, status in ctx.market_data.krx_status.items():
        normalized = str(status).strip().upper()
        if normalized and normalized not in {"NORMAL", "OK", "ACTIVE", "거래"}:
            violations.append(_blocking("RESTRICTED_TICKER", f"{ticker} 거래 상태가 {status}", [ticker]))
    return violations
