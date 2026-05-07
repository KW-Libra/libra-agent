"""LIBRA Domain Agents — JY 7-에이전트 합의 시스템.

JYlibra-sample_v1 의 도메인 전문 에이전트 7명을 흡수한 모듈.
패러다임: 병렬 deliberate → AgentVerdict (vote: approve/reject/abstain) → 합의.

본 모듈의 BaseAgent / PortfolioContext / AgentVerdict 는 JY 원본과 호환되지만,
LIBRA Judge 시스템(libra_agent.libra.agents) 의 InformationAgentProtocol /
AgentResponse / PortfolioSnapshot 과는 ``_adapter`` 모듈을 거쳐 매핑된다.

에이전트:
  - RiskAgent      (Vora)   — 정량 리스크 감시 (HHI/VaR/MDD)
  - TaxAgent       (Reed)   — 손익통산 후보 식별
  - ComplianceAgent(Clarke) — IPS 거부권 (단독 차단 가능)
  - MacroAgent     (Halden) — 거시 충격 + Gemini × Claude 교차검증
  - SentimentAgent (Imo)    — FinBERT + Gemini × Claude 적대 검토
  - ExecutionAgent (Tien)   — Almgren-Chriss 시장충격
  - ESGAgent       (Esme)   — ESG 점수 + 탄소강도
"""

from .base import (
    AgentVerdict as DomainAgentVerdict,
    BaseAgent as DomainBaseAgent,
    PortfolioContext as DomainPortfolioContext,
)
from .compliance import ComplianceAgent
from .esg_agent import ESGAgent
from .execution_agent import ExecutionAgent
from .macro_agent import MacroAgent
from .risk import RiskAgent
from .sentiment_agent import SentimentAgent
from .tax import TaxAgent

__all__ = [
    "DomainAgentVerdict",
    "DomainBaseAgent",
    "DomainPortfolioContext",
    "ComplianceAgent",
    "ESGAgent",
    "ExecutionAgent",
    "MacroAgent",
    "RiskAgent",
    "SentimentAgent",
    "TaxAgent",
]
