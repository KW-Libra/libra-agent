# LIBRA 7-에이전트 풀 머지 설계서 (v1.0, 2026-05-07)

> 본 문서는 **JY 7-에이전트 합의 시스템** 과 **LIBRA Judge LangGraph 시스템**
> 을 단일 코드베이스로 통합하기 위한 정밀 설계서다.
>
> Phase 1-4 는 이미 양쪽 레포(`D:\libra-agent`, `D:\Libra`) 의
> `feat/seven-agent-merge` 브랜치에서 구현되어 검증 완료.
> Phase 5-7 이 본 문서의 **구현 대상**이며 Codex 또는 후속 작업으로 진행한다.

---

## 0. 현재 상태 (2026-05-07 22:30 기준)

| Phase | 작업 | 상태 |
|---|---|---|
| 1 | 인벤토리 + 의존성 트리 | DONE |
| 2 | JY 7 + services 흡수 (`libra_agent/domain_agents/`) | DONE (양쪽) |
| 3 | 어댑터 (`_adapter.py`) — 동적 dataclass 필드 필터 | DONE (양쪽) |
| 4 | 단위 검증 (`scripts/test_seven_agents.py`) — 4/4 PASS | DONE |
| 5 | `AgentBundle` 확장 + Judge prompt + 합의 로직 | **TODO** |
| 6 | backend `AgentSignal` 스키마 + Migration `V7` | **TODO** |
| 7 | frontend Decision Trace UI 7-에이전트 표시 | **TODO** |

> 풀 머지 작업 브랜치: `feat/seven-agent-merge` (양쪽 레포). 기존 `main` 은
> 미팅 시연 안전판으로 보존.

---

## 1. 통합 목표

1. JY 7-에이전트(Risk/Tax/Compliance/Macro/Sentiment/Execution/ESG) 를
   LIBRA Judge 의 sub-agent 로 호출 가능하게 만든다.
2. **Compliance 단독 거부권** 을 Judge 결정 위에 적용한다.
3. **Macro / Sentiment 의 Gemini × Claude 적대 검토** 를 Judge run 의
   Decision Trace 에 노출한다.
4. 기존 6 에이전트(Disclosure/News/Report/Profit/Cost/Evaluation) 흐름은
   회귀 없이 그대로 유지한다.
5. 양쪽 레포 (`libra-agent` 분리, `Libra` 모노레포) 를 동기화한다.

## 2. 통합 원칙 (불변)

- **추상 수준에서 호환**: JY `BaseAgent.deliberate(ctx) → DomainAgentVerdict`
  는 어댑터를 통해 LIBRA `InformationAgentProtocol.run(...) → AgentResponse`
  로 노출된다. JY 원본 코드는 변경하지 않는다.
- **양쪽 레포의 `AgentResponse` 필드 차이는 어댑터가 흡수**한다
  (`dataclass.fields()` 동적 필터). 둘 중 어느 한쪽으로 합치지 않는다.
- **Compliance 의 hard rule (LLM 미사용)** 은 LLM/Network 장애 시에도
  반드시 동작한다.
- **Sentiment 의 Phase 2 파이프라인** (FinBERT → Gemini → Claude 적대 검토)
  은 `libra_agent.sentiment` 모듈에 단일 진입점 (`analyze_news`) 으로 유지한다.
- **JY services 의존성 (LLMRouter, portfolio_optimizer 등)** 은
  `libra_agent.domain_agents._services` 하위에 격리한다.
  외부 코드는 `_services` 를 직접 import 하지 않는다.

---

## 3. 통합 후 시스템 경계

```
사용자 (Web)
  ↓ HTTPS
libra-frontend (Vue 3.5)
  ↓ REST + JWT
libra-backend (Spring Boot 3.5)
  ↓ HTTP  (POST /v1/judge-runs)
libra-agent (Python 3.11+, FastAPI)
  │
  ├─ Judge (LangGraph 1.1)
  │    │
  │    ├─ Information Agents (6, 기존)
  │    │   - DisclosureAgent / NewsAgent / ReportAgent
  │    │   - ProfitAgent / CostAgent / EvaluationAgent
  │    │
  │    └─ Domain Agents (7, 신규, JyDomainAgentAdapter 경유)
  │        - RiskAgent (Vora) — Claude Sonnet
  │        - TaxAgent (Reed) — Claude Haiku
  │        - ComplianceAgent (Clarke) — rule-based, 단독 거부권
  │        - MacroAgent (Halden) — Gemini Flash + Claude cross-validate
  │        - SentimentAgent (Imo) — FinBERT + Gemini × Claude 적대 검토
  │        - ExecutionAgent (Tien) — Claude Haiku, Almgren-Chriss
  │        - ESGAgent (Esme) — Gemini Flash, ESG 점수 + 탄소강도
  │
  ├─ libra_agent.sentiment (PR 통합)
  │    └─ analyze_news() — Phase 2 파이프라인
  │
  └─ LLM 호출
       - Anthropic Claude (Sonnet/Haiku)
       - Google Gemini (Flash/Pro)
       - Ollama (로컬 폴백)
```

호출 방향은 단방향이며, `libra-agent` 가 backend·frontend·ingest 의 어느 것도
직접 호출하지 않는다.

---

## 4. 모듈 트리 (정확한 파일 경로)

양쪽 레포 모두 동일 구조. `<repo>` = `D:\libra-agent` 또는 `D:\Libra`.

```
<repo>/
├── src/libra_agent/
│   ├── libra_models.py              # AgentResponse, PortfolioSnapshot, JudgeDecision
│   ├── libra_runtime.py             # LLMAgent, LocalKnowledgeBase, ChatClient
│   ├── libra_graph.py               # LangGraph Judge 그래프
│   ├── libra_api.py                 # FastAPI HTTP 어댑터
│   │
│   ├── libra/                       # ← 기존 Judge 시스템
│   │   ├── agents/
│   │   │   ├── base.py              # DelegatingInformationAgent, AgentBundle
│   │   │   ├── factory.py           # build_default_agent_bundle()  ← Phase 5에서 확장
│   │   │   ├── disclosure_agent.py
│   │   │   ├── news_agent.py        # Phase 2 통합본: analyze_with_collab() 헬퍼
│   │   │   ├── report_agent.py
│   │   │   ├── profit_agent.py
│   │   │   ├── cost_agent.py
│   │   │   └── evaluation_agent.py
│   │   ├── prompts/
│   │   │   ├── base.py              # InformationAgentPromptProfile
│   │   │   ├── judge.py             # ← Phase 5에서 도메인 에이전트 등록
│   │   │   ├── disclosure.py / news.py / report.py
│   │   │   └── domain.py            # ← Phase 5 신규: 7개 도메인 prompt_profile
│   │   ├── llm_clients/             # ChatClientProtocol (Anthropic/Ollama/llama.cpp)
│   │   ├── direct_indexing.py / signals.py / constraints.py / ...
│   │
│   ├── domain_agents/               # ← Phase 2 신규 (JY 7)
│   │   ├── __init__.py              # 7개 클래스 + DomainAgentVerdict, DomainBaseAgent, DomainPortfolioContext export
│   │   ├── base.py                  # JY BaseAgent / AgentVerdict / PortfolioContext
│   │   ├── compliance.py            # ComplianceAgent (rule-based)
│   │   ├── esg_agent.py             # ESGAgent
│   │   ├── execution_agent.py       # ExecutionAgent
│   │   ├── macro_agent.py           # MacroAgent (cross_validate=True)
│   │   ├── risk.py                  # RiskAgent
│   │   ├── sentiment_agent.py       # SentimentAgent (Phase 2 파이프라인 사용)
│   │   ├── tax.py                   # TaxAgent
│   │   ├── _adapter.py              # JyDomainAgentAdapter, build_domain_agent_adapters()
│   │   ├── _consensus.py            # ← Phase 5 신규: apply_compliance_veto, compute_domain_consensus
│   │   └── _services/
│   │       ├── __init__.py
│   │       ├── llm_router.py        # Claude+Gemini 라우터
│   │       ├── portfolio_optimizer.py  # Almgren-Chriss, VaR, MDD
│   │       └── market_data_injector.py # RSS/BOK API (선택적)
│   │
│   └── sentiment/                   # ← PR 통합 (Phase 2 파이프라인)
│       ├── __init__.py
│       ├── gemini_claude_collab.py  # Gemini × Claude 적대 검토 (PR 패치 적용본)
│       ├── news_analyzer.py         # SENTIMENT_MODE 분기 + analyze_news()
│       ├── finbert_service.py       # ProsusAI/finbert 헤드라인 분류
│       ├── ollama_client.py         # 로컬 폴백
│       └── ticker_names.py          # 종목코드 → 표시명
│
├── scripts/
│   ├── test_news_pipeline.py        # Sentiment 4-step 단위 검증 (Phase 2 결과)
│   ├── test_seven_agents.py         # 7-에이전트 어댑터 4-step 단위 검증 (Phase 4 결과)
│   └── test_judge_thirteen.py       # ← Phase 5 신규: Judge → 13 에이전트 호출 통합 검증
│
└── pyproject.toml                   # optional [sentiment], [domain_agents] 추가
```

`libra-backend` 와 `libra-frontend` 의 변경 위치는 §9, §10 참조.

---

## 5. 데이터 모델 매핑

### 5.1 PortfolioSnapshot (LIBRA) ↔ PortfolioContext (JY)

LIBRA `PortfolioSnapshot` (frozen dataclass, `libra_agent.libra_models`):

| 필드 | 타입 |
|---|---|
| generated_at | datetime |
| holdings | tuple[PortfolioHolding, ...] |
| total_value_krw | float \| None |
| cash_weight | float |
| user_preferences | tuple[str, ...] |

`PortfolioHolding`:

| 필드 | 타입 |
|---|---|
| ticker | str |
| company_name | str |
| weight | float |
| aliases | tuple[str, ...] |
| shares | float \| None |
| last_price | float \| None |
| average_price | float \| None |
| market_value_krw | float \| None |
| unrealized_pnl_krw | float \| None |

JY `PortfolioContext` (mutable dataclass, `libra_agent.domain_agents.base`):

| 필드 | 타입 |
|---|---|
| user_id | str |
| holdings | list[dict] |
| preferences | dict[str, Any] |
| total_value | float |
| proposed_trades | list[dict] |
| market_context_str | str |
| returns_data | dict[str, list[float]] \| None |
| router | LLMRouter \| None |

**매핑 규칙** — `_adapter.portfolio_snapshot_to_domain_context`:

| LIBRA 필드 | JY 필드 | 변환 |
|---|---|---|
| `h.ticker` | `holdings[i]["symbol"]` | 직접 |
| `h.company_name` | `holdings[i]["name"]` | 직접 |
| `h.weight` | `holdings[i]["weight"]` | float() |
| `h.shares` | `holdings[i]["quantity"]` | float(or 0) |
| `h.last_price` | `holdings[i]["current_price"]` | float(or 0) |
| `h.average_price` | `holdings[i]["average_price"]` | float(or 0) |
| `h.market_value_krw` | `holdings[i]["market_value"]` | float(or 0) |
| (없음) | `holdings[i]["sector"]` | 기본 `"기타"` (외부 주입 권장) |
| `total_value_krw` | `total_value` | float(or 0) |
| `user_preferences` | `preferences` | `{k: True for k in tuple}` |
| (외부 인자) | `proposed_trades` | adapter 호출 시점에 주입 |
| (외부 인자) | `market_context_str` | `context` 파라미터 그대로 |

**알려진 정보 손실** (Phase 5 또는 후속에서 보강):
- LIBRA `PortfolioHolding` 에 `sector` 가 없음 → ESG/Macro 의 sector-based
  룰이 부정확. 보강 방법:
  1. backend 가 KIS 종목 마스터에서 sector 를 채워 응답
  2. `PortfolioHolding` 에 `sector: str | None` 필드 추가
  3. `_adapter` 가 sector 자동 lookup (ticker → sector 사전 내장)
- LIBRA `user_preferences` 가 `tuple[str]` → JY `preferences` 의 풍부한
  dict 와 불일치. 보강 방법:
  - `PortfolioSnapshot.user_preferences` 를 `dict[str, Any]` 로 확장
  - 또는 `PortfolioSnapshot` 에 별도 `domain_preferences: dict` 필드 추가

### 5.2 DomainAgentVerdict (JY) → AgentResponse (LIBRA)

JY `DomainAgentVerdict` (mutable dataclass):

| 필드 | 타입 |
|---|---|
| agent_id | str |
| vote | "approve" \| "reject" \| "abstain" |
| confidence | float (0~1) |
| rationale | str |
| signals | list[dict {label, value, ...}] |
| llm_used | str |

LIBRA `AgentResponse` (mutable dataclass): `agent_id`, `opinion_id`,
`turn_number`, `query_understood`, `verdict` (enum), `evidence`, `direction`,
`strength`, `urgency`, `confidence`, `reasoning_for_judge_agent`,
(`signal_score`, `source_trust`, `opinion`, `risk_level` 은 `D:\libra-agent`
측에만 존재, `D:\Libra` 측에는 없을 수도).

**매핑 규칙** — `_adapter.domain_verdict_to_agent_response`:

| JY vote | LIBRA verdict | direction | opinion |
|---|---|---|---|
| approve | `DIRECT_ANSWER` | +0.6 | POSITIVE |
| reject | `DIRECT_ANSWER` | −0.7 | NEGATIVE |
| abstain | `QUIET` | 0.0 | NEUTRAL |

| JY 필드 | LIBRA 필드 |
|---|---|
| confidence | confidence + strength (동일 값) |
| rationale | reasoning_for_judge_agent |
| signals[].label → value | evidence dict 항목 |
| llm_used | (현재 미매핑, evidence 에 추가 권장) |

**양쪽 레포 호환**: `dataclass.fields(AgentResponse)` 로 실제 받는 필드만
전달. 모노레포에 없는 필드(`signal_score` 등)는 자동으로 drop.

---

## 6. 어댑터 인터페이스

### 6.1 `JyDomainAgentAdapter`

위치: `libra_agent.domain_agents._adapter.JyDomainAgentAdapter`

```python
class JyDomainAgentAdapter:
    """JY BaseAgent 를 LIBRA InformationAgentProtocol 로 노출."""

    agent_id: str
    owner_scope: str

    def __init__(
        self,
        jy_agent: DomainBaseAgent,
        *,
        agent_id: str,
        owner_scope: str,
    ) -> None: ...

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: "LocalKnowledgeBase",
        depth: str = "medium",
    ) -> AgentResponse: ...
        # 동기 호출. 내부에서 asyncio.run, loop 충돌 시 별 thread 폴백.

    async def deliberate(self, ctx: DomainPortfolioContext) -> DomainAgentVerdict: ...
        # 원본 JY 패턴 직통 패스.
```

빌더:

```python
def build_domain_agent_adapters() -> dict[str, JyDomainAgentAdapter]:
    """7개 도메인 에이전트 인스턴스를 반환.

    키: "risk" | "tax" | "compliance" | "macro" | "sentiment" | "execution" | "esg"
    """
```

### 6.2 헬퍼 함수

```python
def portfolio_snapshot_to_domain_context(
    portfolio: PortfolioSnapshot,
    *,
    user_id: str = "libra",
    proposed_trades: list[dict[str, Any]] | None = None,
    market_context_str: str = "",
) -> DomainPortfolioContext: ...

def domain_verdict_to_agent_response(
    verdict: Any,  # DomainAgentVerdict
    *,
    agent_id: str,
    turn_number: int,
    query: str,
) -> AgentResponse: ...
```

---

## 7. Phase 5 — `AgentBundle` 확장 + Judge 통합

### 7.1 `AgentBundle` 변경

위치: `libra_agent.libra.agents.base`

```python
@dataclass(slots=True)
class AgentBundle:
    # 기존 6 (Phase 5 이후에도 그대로 필수)
    disclosure: InformationAgentProtocol
    news: InformationAgentProtocol
    report: InformationAgentProtocol
    profit: TradeAgentProtocol
    cost: TradeAgentProtocol
    evaluation: EvaluationAgentProtocol

    # 신규 7 (도메인 에이전트, optional)
    risk: InformationAgentProtocol | None = None
    tax: InformationAgentProtocol | None = None
    compliance: InformationAgentProtocol | None = None
    macro: InformationAgentProtocol | None = None
    sentiment: InformationAgentProtocol | None = None
    execution: InformationAgentProtocol | None = None
    esg: InformationAgentProtocol | None = None

    def domain_agents(self) -> dict[str, InformationAgentProtocol]:
        """도메인 에이전트 dict 만 반환 (None 제거)."""
        return {
            name: agent for name, agent in {
                "risk": self.risk, "tax": self.tax,
                "compliance": self.compliance, "macro": self.macro,
                "sentiment": self.sentiment, "execution": self.execution,
                "esg": self.esg,
            }.items() if agent is not None
        }
```

### 7.2 `factory.build_default_agent_bundle` 변경

위치: `libra_agent.libra.agents.factory`

```python
import os

def build_default_agent_bundle(*, client: "ChatClient") -> AgentBundle:
    bundle_kwargs = dict(
        disclosure=DisclosureAgent(client=client),
        news=NewsAgent(client=client),
        report=ReportAgent(client=client),
        profit=ProfitAgent(),
        cost=CostAgent(),
        evaluation=EvaluationAgent(client=client),
    )

    if os.environ.get("LIBRA_DOMAIN_AGENTS_ENABLED", "false").lower() == "true":
        from libra_agent.domain_agents._adapter import build_domain_agent_adapters
        adapters = build_domain_agent_adapters()
        bundle_kwargs.update(adapters)  # risk/tax/compliance/macro/sentiment/execution/esg

    return AgentBundle(**bundle_kwargs)
```

기본값은 `false` 라서 기존 회귀 없음. 미팅·운영에서는 `true` 로 설정.

### 7.3 Judge prompt 변경

위치: `libra_agent.libra.prompts.judge` (`JUDGE_PROMPT_PROFILE` 또는 동등 객체)

기존 prompt 의 "Available agents" 섹션에 7개 도메인 에이전트 owner_scope 추가:

```
Available agents (call any subset, in any order):
- DisclosureAgent: 공시 사실 확인
- NewsAgent: 시장 반응·뉴스 정성 신호
- ReportAgent: 증권사 리포트·컨센서스
- ProfitAgent: 후보 plan 의 expected payoff 검토
- CostAgent: 거래비용·유동성 검토
- EvaluationAgent: 사후 결과 평가

Domain agents (optional, when LIBRA_DOMAIN_AGENTS_ENABLED=true):
- RiskAgent (Vora): HHI/VaR/MDD/단일 delta 위험 한도
- TaxAgent (Reed): 손익통산 후보 식별
- ComplianceAgent (Clarke): 사용자 IPS/ESG 제외 룰 — 거부권 있음
- MacroAgent (Halden): 거시 충격 + Gemini × Claude cross-validate
- SentimentAgent (Imo): FinBERT + Gemini × Claude 적대 검토
- ExecutionAgent (Tien): Almgren-Chriss 시장충격 + 체결 전략
- ESGAgent (Esme): ESG 점수 + 탄소강도 + 사용자 esg_exclusions
```

도메인 에이전트의 `owner_task_brief` 는 `libra/prompts/domain.py` 신규 파일에
정의하거나, `domain_agents/_adapter.py` 의 빌더가 직접 주입한다.

### 7.4 합의 로직 (`_consensus.py`)

위치: `libra_agent.domain_agents._consensus`

```python
from libra_agent.libra_models import AgentResponse, JudgeDecision, DecisionType

def compute_domain_consensus(responses: list[AgentResponse]) -> dict:
    """7-에이전트 다수결 + confidence 가중 점수.

    Returns:
        {
          "n_approve": int,    # opinion=POSITIVE 카운트
          "n_reject":  int,    # opinion=NEGATIVE 카운트
          "n_abstain": int,
          "score":     float,  # Σ(direction × confidence) / Σ confidence
          "compliance_veto": bool,
          "rejecting_agents": list[str],
        }
    """

def apply_compliance_veto(
    judge_decision: JudgeDecision,
    domain_responses: list[AgentResponse],
) -> JudgeDecision:
    """Compliance 가 reject 한 경우 Judge 결정을 USER_DECISION_REQUIRED 로 강제 전환.

    Compliance reject 의 reasoning 을 JudgeDecision.reasoning 앞에 prepend.
    """
```

### 7.5 Judge 흐름 변경

위치: `libra_agent.libra_graph` 또는 동등 LangGraph 노드 정의

기존:
```
information_gathering → deliberation → consensus → decision
```

변경 후:
```
information_gathering
  → deliberation (정보 에이전트)
  → candidate_plan_generation
  → domain_consensus  (← Phase 5 신규: 7개 도메인 에이전트 병렬 호출)
  → compliance_check  (← Phase 5 신규: Compliance reject 시 강제 전환)
  → profit_cost_review
  → decision (+ Decision Trace 에 도메인 결과 추가)
```

병렬 호출 구현 가이드:
- `asyncio.gather(*[adapter.deliberate(ctx) for adapter in bundle.domain_agents().values()])`
  로 7개 동시 호출
- 어댑터의 sync `run()` 대신 async `deliberate()` 직통 사용
- 각 응답을 `domain_verdict_to_agent_response` 로 변환 후 Decision Trace 노드 추가

### 7.6 Phase 5 검증 스크립트

위치: `scripts/test_judge_thirteen.py` (신규)

```python
"""Judge → 13 에이전트 (6 정보/거래 + 7 도메인) 호출 통합 검증."""

# Step 1: LIBRA_DOMAIN_AGENTS_ENABLED=true 환경에서 build_default_agent_bundle()
#         실행 → 13 에이전트 모두 등록 확인
# Step 2: 샘플 PortfolioSnapshot + 가짜 Judge run → Decision Trace 에 도메인 verdict 7개 노출 확인
# Step 3: ComplianceAgent 가 reject 하는 케이스 → JudgeDecision.decision == USER_DECISION_REQUIRED
# Step 4: Macro adversarial cross-validate 결과 (model_used="gemini-flash+claude") 가 evidence 에 기록 확인
```

---

## 8. Phase 6 — backend 영향 (`libra-backend`)

### 8.1 데이터 모델 변경

위치: `libra-backend/src/main/java/com/libra/api/decision/AgentSignal.java`
(혹은 동등 엔티티)

```java
@Entity
public class AgentSignal {
    // 기존 필드
    Long id;
    UUID decisionRunId;
    String agentId;          // "disclosure" | "news" | ... | "risk" | "tax" | ...
    String verdict;           // DIRECT_ANSWER, PARTIAL_ANSWER, ...
    Double confidence;
    String reasoning;

    // 신규 필드 (Phase 6)
    String agentKind;         // "information" | "trade" | "domain"
    String vote;              // "approve" | "reject" | "abstain" | NULL (information 에이전트는 NULL)
    String domainSignalsJson; // JY signals[] dict 의 JSON (선택적)
    String llmUsed;           // "claude-sonnet-4-6" | "gemini-2.5-flash" | "finbert+gemini-flash+claude" | ...
}
```

### 8.2 Migration

위치: `libra-backend/src/main/resources/db/migration/V7__seven_agent_merge.sql`

```sql
ALTER TABLE agent_signals
    ADD COLUMN agent_kind         VARCHAR(20) NULL,
    ADD COLUMN vote               VARCHAR(10) NULL,
    ADD COLUMN domain_signals_json JSON      NULL,
    ADD COLUMN llm_used           VARCHAR(80) NULL;

-- 기존 record 백필: Disclosure/News/Report = 'information', Profit/Cost = 'trade', Evaluation = 'evaluation'
UPDATE agent_signals
   SET agent_kind = CASE
       WHEN agent_id IN ('disclosure', 'news', 'report')   THEN 'information'
       WHEN agent_id IN ('profit', 'cost')                 THEN 'trade'
       WHEN agent_id = 'evaluation'                        THEN 'evaluation'
       ELSE 'domain'
   END
 WHERE agent_kind IS NULL;
```

### 8.3 DTO / 직렬화

위치: `libra-backend/src/main/java/com/libra/api/decision/DecisionRunResponse.java`

`agent_signals` 배열에 `agent_kind`, `vote`, `domain_signals` 필드 추가.

`DecisionRunService.persistAgentResponse(...)` 가 LIBRA `AgentResponse` 의
`evidence` 안에 `vote`, `llm_used` 가 있으면 새 컬럼에 저장.

---

## 9. Phase 7 — frontend 영향 (`libra-frontend`)

### 9.1 Decision Trace UI

위치: `libra-frontend/src/pages/DecisionDetailPage.vue` (또는 동등)

- 기존 정보/거래 에이전트 6개 카드 영역 그대로 유지
- 새 섹션 "도메인 에이전트 합의 (7)":
  - 카드 1행 7개 (Risk / Tax / Compliance / Macro / Sentiment / Execution / ESG)
  - 카드 내용: vote 배지 (approve=green / reject=red / abstain=gray), confidence,
    rationale (1줄 truncate, hover 전체), signals 토글
- Compliance 가 reject 한 경우: 화면 상단 빨간 배너 "Compliance 거부 →
  사용자 결정 필요" 표시
- Macro / Sentiment 의 `model_used` 가 `+claude` 포함 시 카드에
  "Gemini × Claude 적대 검토" 배지 노출

### 9.2 라우팅

새 페이지 추가 안 함. 기존 `/decision/:runId` 에 도메인 에이전트 섹션만 확장.

### 9.3 API 호출

`libra-frontend/src/api.ts` 의 `getDecisionRun(runId)` 가 새 필드
(`agent_kind`, `vote`, `domain_signals`) 를 받도록 타입 정의 업데이트.

---

## 10. 의존성

### 10.1 `pyproject.toml` (양쪽 레포 동일)

```toml
[project.optional-dependencies]
sentiment = [
  "anthropic>=0.39",          # Claude SDK (gemini_claude_collab 은 httpx 직접 호출이라 필요 없음, 다만 LLMRouter 가 사용)
  "python-dotenv>=1.0",       # .env 로딩
  "transformers>=4.40",       # FinBERT
  "torch>=2.2",               # FinBERT 백엔드
]

domain_agents = [
  "anthropic>=0.39",          # LLMRouter Claude
  "google-generativeai>=0.7", # LLMRouter Gemini
  "numpy>=1.26",              # Risk / portfolio_optimizer
  "scipy>=1.11",              # portfolio_optimizer Mean-Variance / Risk Parity
  "aiohttp>=3.9",             # market_data_injector (선택적)
]
```

설치: `pip install -e ".[sentiment,domain_agents]"`

### 10.2 환경변수 (`.env`, libra-agent 측)

| 변수 | 용도 | 기본값 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude (Judge + Sentiment + 도메인) | (필수) |
| `GEMINI_API_KEY` | Gemini (Sentiment 적대 검토 + Macro cross-validate) | (없으면 Ollama 폴백) |
| `SENTIMENT_MODE` | `gemini-claude` / `finbert-ollama` / `fingpt-local` | `gemini-claude` |
| `ANTHROPIC_MODEL` | Anthropic 모델 ID | `claude-haiku-4-5-20251001` |
| `OLLAMA_BASE_URL` | Ollama 폴백 endpoint | `http://ollama:11434` |
| `OLLAMA_MODEL` | Ollama 모델 | `llama3.2:3b` |
| `LIBRA_DOMAIN_AGENTS_ENABLED` | 7개 도메인 에이전트 활성 | `false` (운영에서 `true`) |
| `LLM_ROUTING_POLICY` | LLMRouter 정책 | `balanced` |

KIS / OAuth / DB 키는 `libra-backend` env 측 (변경 없음).

---

## 11. 테스트 전략

| 레벨 | 도구 | 위치 | 커버 |
|---|---|---|---|
| 단위 | pytest | `tests/unit/test_adapter.py` (신규) | `domain_verdict_to_agent_response`, `portfolio_snapshot_to_domain_context` |
| 단위 | pytest | `tests/unit/test_consensus.py` (신규) | `apply_compliance_veto`, `compute_domain_consensus` |
| 통합 단위 | python | `scripts/test_news_pipeline.py` (Phase 2 결과) | FinBERT + Gemini × Claude 적대 검토 |
| 통합 단위 | python | `scripts/test_seven_agents.py` (Phase 4 결과) | 7개 어댑터 + AgentResponse 변환 |
| 통합 단위 | python | `scripts/test_judge_thirteen.py` (Phase 5 신규) | Judge → 13 에이전트 + Compliance veto + Decision Trace |
| 통합 E2E | docker-compose.local + frontend | `docs/local-demo.md` | Judge run → 13 에이전트 호출 → backend 영속화 → frontend Decision Trace UI |

검증 명령:

```powershell
# Phase 1-4 검증 (이미 완료)
python scripts/test_news_pipeline.py            # GEMINI 키 있을 때 step 3 PASS
python scripts/test_seven_agents.py             # 4/4 PASS

# Phase 5 검증 (구현 후)
$env:LIBRA_DOMAIN_AGENTS_ENABLED = "true"
python scripts/test_judge_thirteen.py           # 4/4 PASS 목표

# Phase 6 검증
cd libra-backend
.\gradlew.bat test                              # AgentSignal V7 마이그레이션 통과

# Phase 7 검증
cd libra-frontend
npm run build                                   # build successful
npm run dev                                     # /decision/:runId 에서 도메인 카드 7개 표시 확인
```

---

## 12. 리스크 및 완화

| # | 리스크 | 영향 | 완화 |
|---|---|---|---|
| R1 | `PortfolioHolding` 에 sector 없음 | ESG/Macro 룰 부정확 | T1: PortfolioHolding 확장 + KIS 마스터 채움. 임시: 어댑터에 ticker→sector 사전 내장 |
| R2 | `asyncio.run` 중첩 (이미 loop 안에서 호출) | adapter `run()` RuntimeError | 별 thread 폴백 (이미 구현). Phase 5 의 LangGraph 노드는 async 직통 (`deliberate`) 사용 |
| R3 | `anthropic` / `google-generativeai` SDK 미설치 | 도메인 에이전트 LLM 호출 실패 | optional 의존 + LLMRouter graceful degrade. `pip install -e ".[domain_agents]"` 필수 |
| R4 | Compliance 거부권이 Judge 결정과 충돌 | 사용자 혼란 | Decision Trace 에 거부 사유 명시 + `USER_DECISION_REQUIRED` 라우팅 + frontend 빨간 배너 |
| R5 | `AgentSignal` 스키마 변경으로 기존 record 깨짐 | Migration 실패 | V7 에 백필 SQL 포함 (§8.2) + DEFAULT NULL |
| R6 | 모노레포(`Libra`) ↔ 분리레포(`libra-agent`) 코드 분기 | 어댑터 동작 차이 | 어댑터의 `dataclass.fields()` 동적 필터 + CI 양쪽 빌드 |
| R7 | 7-에이전트 동시 호출 시 LLM API rate limit | 응답 지연 / 실패 | LLMRouter 의 폴백 체인 활용 + 도메인 에이전트별 timeout 설정 |
| R8 | Phase 5/6/7 진행 중 미팅 시연 깨짐 | 발표 영향 | 작업 브랜치 `feat/seven-agent-merge` 격리. `main` 은 시연 안전판 유지 |

---

## 13. 미해결 / 향후 과제 (T2)

- [ ] 합의 로직에 **confidence 가중 평균** (현재 단순 다수결 + 거부권만)
- [ ] **Reflection 라운드**: 도메인 에이전트 의견 충돌 시 Judge 가 재호출
- [ ] **LangFuse / LangSmith 관측성** (토큰 비용 / 지연 / 실패율)
- [ ] **Reflection 학습 루프**를 도메인 에이전트로 확장 (현재 EvaluationAgent 만)
- [ ] **DART 공시 → DisclosureAgent 통합** (HJ-agent 잔여 이식)
- [ ] **PortfolioHolding.sector** 추가 + KIS 종목 마스터 자동 채움
- [ ] `LIBRA_DOMAIN_AGENTS_ENABLED` 를 사용자별 토글로 (DB 컬럼)
- [ ] backend `kis_credentials` 테이블 (multi-tenant KIS 키)

---

## 14. 참고 자료 (현재 레포 안)

- `D:\Libra\docs\PROJECT_OVERVIEW.md` — LIBRA 본체 설계 (Judge 기준)
- `D:\Libra\docs\midterm-deliverables.md` — 중간발표 문서 (Judge 기준)
- `D:\Libra\docs\team-contribution-integration-map.md` — 5/7 통합 매핑 (sentiment 까지)
- `D:\KW-Libra-team-repos\JYlibra-sample_v1` — JY 7-에이전트 원본
- `D:\KW-Libra-team-repos\JYlibra-sample_v1\LIBRA_v2_Agent_Design.docx` — JY 정본 설계
- PR 코멘트 (PRAHE, 2026-05-07) — `fix/sentiment-gemini-collab`

---

(끝)
