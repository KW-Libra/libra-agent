from __future__ import annotations

from libra_agent.libra.schemas.ips import IPSConfig, KYCProfile


def persona_v1_kyc() -> KYCProfile:
    return KYCProfile(
        risk_tolerance="MODERATE",
        investment_horizon_years=15,
        max_drawdown_tolerance_pct=15.0,
    )


def persona_v1_ips() -> IPSConfig:
    return IPSConfig(
        single_ticker_limit_pct=25.0,
        sector_limit_pct=40.0,
        annual_volatility_limit=0.20,
        asset_class_target={"EQUITY": 60.0, "BOND": 35.0, "ALT": 5.0},
        asset_class_band_pct=10.0,
        min_cash_pct=5.0,
        max_market_impact_pct_of_adv=5.0,
        excluded_tickers=[],
        excluded_sectors=["TOBACCO", "WEAPONS"],
        esg_min_score=60.0,
    )


PERSONA_V1 = {
    "id": "persona_v1_default",
    "user_id": "demo_user_001",
    "description": "35세 직장인, 위험 중립, 장기 적립",
    "kyc": persona_v1_kyc(),
    "ips": persona_v1_ips(),
    "initial_portfolio": {
        "069500": 25.0,
        "379800": 25.0,
        "152380": 20.0,
        "153130": 15.0,
        "132030": 10.0,
        "CASH": 5.0,
    },
}
