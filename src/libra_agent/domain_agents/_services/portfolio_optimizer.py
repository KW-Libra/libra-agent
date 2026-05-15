"""
Portfolio Optimizer — 금융공학 엔진

기존 equal-weight 단순 드리프트 감지를 대체하는 정량적 최적화 엔진.

구현된 기능:
  1. Mean-Variance 최적화 (Markowitz, 최소 분산 포트폴리오)
  2. 리스크 패리티 (Equal Risk Contribution)
  3. 히스토리컬 VaR / CVaR (Expected Shortfall)
  4. Maximum Drawdown (MDD) 계산
  5. Tracking Error (vs KOSPI 200 또는 커스텀 벤치마크)
  6. FIFO 세금 로트 추적 (Tax-Loss Harvesting 지원)
  7. HHI 집중도 지수
  8. Almgren-Chriss 시장충격 모델

주의:
  - scipy, numpy 필요 (requirements.txt에 포함됨)
  - 실제 공분산 행렬은 일별 수익률 히스토리가 필요
  - 히스토리 없으면 equal-weight 폴백 (FALLBACK 표시)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── 데이터 클래스 ────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """최적화 결과"""

    method: str  # "mean_variance" | "risk_parity" | "equal_weight_fallback"
    target_weights: dict[str, float]  # symbol → 목표 비중
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """포트폴리오 리스크 지표 집합"""

    var_95: float  # 95% 신뢰수준 VaR (일별, KRW)
    var_99: float  # 99% 신뢰수준 VaR (일별, KRW)
    cvar_95: float  # 95% CVaR / Expected Shortfall
    mdd: float  # Maximum Drawdown (음수, -0.15 = -15%)
    volatility: float  # 연환산 변동성
    tracking_error: float  # 벤치마크 대비 TE
    hhi: float  # Herfindahl-Hirschman Index (0~1, 낮을수록 분산)
    beta: float = 1.0  # 시장 베타
    notes: list[str] = field(default_factory=list)


@dataclass
class TaxLot:
    """세금 로트 (FIFO 추적용)"""

    symbol: str
    quantity: float
    cost_basis: float  # 취득 단가 (KRW)
    acquired_date: str  # ISO 8601
    lot_id: str = ""


@dataclass
class HarvestCandidate:
    """TLH 수확 후보"""

    symbol: str
    unrealized_loss_krw: float
    wash_sale_safe: bool  # 30일 이내 동일 종목 매수 없으면 True
    replacement_symbol: str  # 유사 팩터 대체 종목
    estimated_tax_saving_krw: float


# ── 포트폴리오 최적화기 ──────────────────────────────────────────


class PortfolioOptimizer:
    """
    다이렉트 인덱싱 + 자동 리밸런싱을 위한 정량 최적화 엔진.

    사용법:
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(holdings, method="risk_parity")
        trades  = optimizer.compute_trades(holdings, result.target_weights)
        risks   = optimizer.compute_risk_metrics(holdings, returns_matrix)
    """

    def __init__(self, tax_rate: float = 0.22) -> None:
        """
        Args:
            tax_rate: 양도소득세율 (기본 22% — 국내 주식 대주주 기준)
        """
        self._tax_rate = tax_rate

    # ════════════════════════════════════════════════════════════
    # 1. 포트폴리오 최적화
    # ════════════════════════════════════════════════════════════

    def optimize(
        self,
        holdings: list[dict[str, Any]],
        method: str = "risk_parity",
        returns_matrix: np.ndarray | None = None,
        risk_free_rate: float = 0.035,  # 한국 무위험 수익률 ~3.5%
    ) -> OptimizationResult:
        """
        포트폴리오 최적 비중 계산.

        Args:
            holdings:       보유 종목 리스트 [{"symbol": ..., "weight": ..., ...}]
            method:         "mean_variance" | "risk_parity" | "equal_weight"
            returns_matrix: 일별 수익률 행렬 (n_days × n_assets). None이면 equal-weight 폴백
            risk_free_rate: 연환산 무위험 수익률

        Returns:
            OptimizationResult
        """
        symbols = [h["symbol"] for h in holdings]
        n = len(symbols)

        if n == 0:
            return OptimizationResult(method="empty", target_weights={})

        if method == "equal_weight" or returns_matrix is None:
            return self._equal_weight(
                symbols, note="히스토리 데이터 없음" if returns_matrix is None else ""
            )

        cov_matrix = self._compute_covariance(returns_matrix)

        if method == "risk_parity":
            return self._risk_parity(symbols, cov_matrix)
        elif method == "mean_variance":
            mean_returns = returns_matrix.mean(axis=0) * 252  # 연환산
            return self._mean_variance(symbols, mean_returns, cov_matrix, risk_free_rate)
        else:
            return self._equal_weight(symbols, note=f"알 수 없는 method: {method}")

    def _equal_weight(self, symbols: list[str], note: str = "") -> OptimizationResult:
        n = len(symbols)
        weights = {s: 1.0 / n for s in symbols}
        notes = [f"Equal-weight (1/{n})"]
        if note:
            notes.append(note)
        return OptimizationResult(
            method="equal_weight_fallback",
            target_weights=weights,
            notes=notes,
        )

    def _compute_covariance(self, returns: np.ndarray) -> np.ndarray:
        """연환산 공분산 행렬 (Ledoit-Wolf 축소 추정 미지원 시 표본 공분산)"""
        try:
            cov = np.cov(returns, rowvar=False) * 252
            # 양정치 행렬(PD) 보장 — 수치 안정성
            min_eig = np.linalg.eigvalsh(cov).min()
            if min_eig < 1e-8:
                cov += np.eye(cov.shape[0]) * (abs(min_eig) + 1e-8)
            return cov
        except Exception as e:
            logger.warning(f"[Optimizer] 공분산 계산 오류: {e}")
            return np.eye(returns.shape[1])

    def _risk_parity(self, symbols: list[str], cov: np.ndarray) -> OptimizationResult:
        """
        Equal Risk Contribution (ERC) — 리스크 패리티.

        각 자산의 '한계 리스크 기여도'가 균등하도록 비중 결정.
        고변동성 자산은 낮은 비중, 저변동성 자산은 높은 비중.

        수식:
          MRC_i = (Σ·w)_i / √(w^T Σ w)
          목표: w_i · MRC_i = 1/n (모든 i에 대해)
        """
        from scipy.optimize import minimize

        n = len(symbols)
        w0 = np.ones(n) / n

        def risk_contribution_diff(w):
            port_vol = math.sqrt(max(w @ cov @ w, 1e-12))
            mrc = cov @ w / port_vol
            rc = w * mrc  # 각 자산 리스크 기여
            target = port_vol / n
            return float(np.sum((rc - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.40)] * n  # 최소 1%, 최대 40% (집중도 제한)

        result = minimize(
            risk_contribution_diff,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        weights = dict(zip(symbols, result.x, strict=False))
        port_vol = math.sqrt(max(result.x @ cov @ result.x, 0))

        return OptimizationResult(
            method="risk_parity",
            target_weights=weights,
            expected_volatility=port_vol,
            notes=["Equal Risk Contribution 최적화", f"포트폴리오 변동성: {port_vol:.2%}"],
        )

    def _mean_variance(
        self,
        symbols: list[str],
        mean_returns: np.ndarray,
        cov: np.ndarray,
        risk_free_rate: float,
    ) -> OptimizationResult:
        """
        Mean-Variance 최적화 (최대 Sharpe Ratio 포트폴리오).

        수식:
          max  (μ^T w - r_f) / √(w^T Σ w)
          s.t. Σ w_i = 1,  0 ≤ w_i ≤ 0.40
        """
        from scipy.optimize import minimize

        n = len(symbols)
        w0 = np.ones(n) / n

        def neg_sharpe(w):
            ret = float(mean_returns @ w)
            vol = math.sqrt(max(float(w @ cov @ w), 1e-12))
            return -(ret - risk_free_rate) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.40)] * n

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        w = result.x
        ret = float(mean_returns @ w)
        vol = math.sqrt(max(float(w @ cov @ w), 0))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        weights = dict(zip(symbols, w, strict=False))

        return OptimizationResult(
            method="mean_variance",
            target_weights=weights,
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            notes=[f"Max Sharpe: {sharpe:.3f}", f"기대수익: {ret:.2%}, 변동성: {vol:.2%}"],
        )

    # ════════════════════════════════════════════════════════════
    # 2. 거래 계획 생성
    # ════════════════════════════════════════════════════════════

    def compute_trades(
        self,
        holdings: list[dict[str, Any]],
        target_weights: dict[str, float],
        drift_threshold: float = 0.02,  # 2% 미만 드리프트는 무시
        total_value: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        현재 비중 → 목표 비중 달성을 위한 거래 목록 생성.

        Args:
            drift_threshold: 이 미만의 드리프트는 거래 생성 안 함 (거래 비용 절감)

        Returns:
            trades: [{"symbol", "action", "delta", "currentWeight", "targetWeight", "reason"}]
        """
        trades = []
        tv = total_value or sum(h.get("market_value", 0) for h in holdings)
        current: dict[str, float] = {h["symbol"]: h.get("weight", 0) for h in holdings}

        for sym, target in target_weights.items():
            current_w = current.get(sym, 0.0)
            delta = target - current_w

            if abs(delta) < drift_threshold:
                continue

            trades.append(
                {
                    "symbol": sym,
                    "action": "BUY" if delta > 0 else "SELL",
                    "quantity": 0,  # 수량은 Execution Agent가 계산
                    "delta": round(delta, 6),
                    "currentWeight": round(current_w, 6),
                    "targetWeight": round(target, 6),
                    "estimated_value_krw": abs(delta) * tv,
                    "reason": f"드리프트 {delta:+.2%} (현재 {current_w:.2%} → 목표 {target:.2%})",
                }
            )

        # 매수 먼저, 매도 나중 (현금 부족 방지)
        trades.sort(key=lambda t: 0 if t["action"] == "BUY" else 1)
        return trades

    # ════════════════════════════════════════════════════════════
    # 3. 리스크 지표 계산
    # ════════════════════════════════════════════════════════════

    def compute_risk_metrics(
        self,
        holdings: list[dict[str, Any]],
        returns_matrix: np.ndarray | None = None,
        benchmark_returns: np.ndarray | None = None,
        total_value: float | None = None,
    ) -> RiskMetrics:
        """
        포트폴리오 리스크 지표 통합 계산.

        Args:
            returns_matrix:     일별 수익률 행렬 (n_days × n_assets)
            benchmark_returns:  벤치마크 일별 수익률 (n_days,) — KOSPI200
        """
        n = len(holdings)
        tv = total_value or sum(h.get("market_value", 0) for h in holdings)
        weights = np.array([h.get("weight", 1.0 / n) for h in holdings])

        notes: list[str] = []

        # ── HHI ─────────────────────────────────────────────────
        # HHI = Σ w_i² (0 ~ 1, 1에 가까울수록 집중)
        hhi = float(np.sum(weights**2))

        if returns_matrix is None or returns_matrix.shape[0] < 30:
            notes.append("수익률 히스토리 부족 — VaR 추정치 신뢰도 낮음")
            return RiskMetrics(
                var_95=tv * 0.02,
                var_99=tv * 0.03,
                cvar_95=tv * 0.025,
                mdd=-0.10,
                volatility=0.18,
                tracking_error=0.05,
                hhi=hhi,
                notes=notes + ["히스토리 부족 — 보수적 추정치 사용"],
            )

        # ── 포트폴리오 일별 수익률 ────────────────────────────────
        port_returns = returns_matrix @ weights  # 가중합

        # ── 히스토리컬 VaR ────────────────────────────────────────
        # 과거 시뮬레이션 방식 — 분포 가정 불필요
        var_95 = abs(float(np.percentile(port_returns, 5))) * tv
        var_99 = abs(float(np.percentile(port_returns, 1))) * tv

        # ── CVaR (Expected Shortfall) ────────────────────────────
        # VaR 초과 손실의 평균 — VaR보다 tail risk 더 잘 포착
        threshold_95 = float(np.percentile(port_returns, 5))
        tail_losses = port_returns[port_returns <= threshold_95]
        cvar_95 = abs(float(tail_losses.mean())) * tv if len(tail_losses) > 0 else var_95

        # ── Maximum Drawdown ─────────────────────────────────────
        # MDD = max(peak - trough) / peak
        cumulative = np.cumprod(1 + port_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        mdd = float(drawdowns.min())

        # ── 연환산 변동성 ────────────────────────────────────────
        volatility = float(np.std(port_returns)) * math.sqrt(252)

        # ── Tracking Error ────────────────────────────────────────
        if benchmark_returns is not None and len(benchmark_returns) == len(port_returns):
            active_returns = port_returns - benchmark_returns
            tracking_error = float(np.std(active_returns)) * math.sqrt(252)
        else:
            tracking_error = 0.0
            notes.append("벤치마크 수익률 없음 — Tracking Error 미계산")

        # ── 베타 ──────────────────────────────────────────────────
        beta = 1.0
        if benchmark_returns is not None and len(benchmark_returns) > 1:
            cov_pb = float(np.cov(port_returns, benchmark_returns)[0, 1])
            var_b = float(np.var(benchmark_returns))
            beta = cov_pb / var_b if var_b > 0 else 1.0

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            mdd=mdd,
            volatility=volatility,
            tracking_error=tracking_error,
            hhi=hhi,
            beta=beta,
            notes=notes,
        )

    # ════════════════════════════════════════════════════════════
    # 4. FIFO 세금 로트 추적 (TLH 지원)
    # ════════════════════════════════════════════════════════════

    def find_harvest_candidates(
        self,
        holdings: list[dict[str, Any]],
        tax_lots: list[TaxLot],
        recent_buys: list[dict[str, Any]],  # 최근 30일 매수 내역
        replacement_map: dict[str, str] | None,  # symbol → 대체 종목
    ) -> list[HarvestCandidate]:
        """
        Tax-Loss Harvesting 수확 후보 목록 생성.

        Wash-Sale Rule (30일):
          - 손실 실현 후 30일 이내 동일/실질적으로 동일한 종목 재매수 금지
          - 한국 세법상 명시적 규정 없으나 세무 리스크 고려 권장

        수식:
          Tax Alpha = unrealized_loss × tax_rate
        """
        candidates: list[HarvestCandidate] = []

        # 최근 30일 매수 종목 집합
        recent_buy_symbols = {r["symbol"] for r in recent_buys}

        holdings_map = {h["symbol"]: h for h in holdings}

        for lot in tax_lots:
            holding = holdings_map.get(lot.symbol)
            if not holding:
                continue

            current_price = holding.get("current_price", lot.cost_basis)
            quantity = lot.quantity

            unrealized_pnl = (current_price - lot.cost_basis) * quantity

            # 손실이 없으면 패스
            if unrealized_pnl >= -10_000:  # 1만원 미만 손실은 무시 (거래비용 > 절세효과)
                continue

            unrealized_loss_krw = abs(unrealized_pnl)
            tax_saving = unrealized_loss_krw * self._tax_rate
            wash_sale_safe = lot.symbol not in recent_buy_symbols
            replacement = (replacement_map or {}).get(lot.symbol, "")

            candidates.append(
                HarvestCandidate(
                    symbol=lot.symbol,
                    unrealized_loss_krw=unrealized_loss_krw,
                    wash_sale_safe=wash_sale_safe,
                    replacement_symbol=replacement,
                    estimated_tax_saving_krw=tax_saving,
                )
            )

        # 절세 효과 큰 순으로 정렬
        candidates.sort(key=lambda c: c.estimated_tax_saving_krw, reverse=True)
        return candidates

    def compute_fifo_cost_basis(self, buy_history: list[dict[str, Any]]) -> list[TaxLot]:
        """
        매수 내역에서 FIFO 방식으로 세금 로트 생성.

        Args:
            buy_history: [{"symbol", "quantity", "price", "date", "lot_id"}]
        """
        lots: list[TaxLot] = []
        for trade in sorted(buy_history, key=lambda t: t.get("date", "")):
            lots.append(
                TaxLot(
                    symbol=trade["symbol"],
                    quantity=trade["quantity"],
                    cost_basis=trade["price"],
                    acquired_date=trade.get("date", ""),
                    lot_id=trade.get("lot_id", ""),
                )
            )
        return lots

    # ════════════════════════════════════════════════════════════
    # 5. Almgren-Chriss 시장충격 모델
    # ════════════════════════════════════════════════════════════

    def almgren_chriss_cost(
        self,
        quantity: float,  # 목표 거래량 (주)
        adv: float,  # 일평균 거래량 (주)
        sigma: float,  # 일별 변동성
        price: float,  # 현재 가격 (KRW)
        T: float = 1.0,  # 총 거래 기간 (일)
        eta: float = 0.1,  # 일시적 충격 계수
        gamma: float = 0.1,  # 영구 충격 계수
    ) -> dict[str, float]:
        """
        Almgren-Chriss (2001) 최적 체결 비용 모델.

        수식:
          일시적 충격 (temporary): η × (v/V)^α   (α ≈ 0.5~1.0)
          영구  충격 (permanent):  γ × (v/V)

          총 비용 = 영구충격 × Q/2 + 일시충격 × Σ(n_k²)
        """
        if adv <= 0:
            return {"temporary_cost": 0.0, "permanent_cost": 0.0, "total_cost": 0.0}

        participation_rate = quantity / adv if adv > 0 else 0.0

        # 일시적 충격 (단순 선형 근사)
        temp_impact = eta * participation_rate * sigma * price
        temporary_cost = temp_impact * quantity / 2

        # 영구적 충격
        perm_impact = gamma * participation_rate * sigma * price
        permanent_cost = perm_impact * quantity / 2

        total_cost = temporary_cost + permanent_cost

        return {
            "temporary_cost": temporary_cost,
            "permanent_cost": permanent_cost,
            "total_cost": total_cost,
            "cost_bps": (total_cost / (quantity * price) * 10_000) if quantity * price > 0 else 0,
            "participation_rate": participation_rate,
        }


# ── 글로벌 싱글턴 ────────────────────────────────────────────────
_optimizer: PortfolioOptimizer | None = None


def get_optimizer(tax_rate: float = 0.22) -> PortfolioOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer(tax_rate=tax_rate)
    return _optimizer
