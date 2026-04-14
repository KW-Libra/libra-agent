# LIBRA 에이전트 상세 설계

> **문서 목적**: LIBRA 멀티 에이전트 시스템의 에이전트별 상세 설계. LIBRA 시나리오 및 명세 v1의 24개 명세를 구조적으로 강제하는 스키마와 책임 경계를 정의한다.
>
> **단일 소스 원칙**: 본 문서가 에이전트 설계의 유일한 기준 문서. 변경점은 섹션 12 "변경 이력"에 기록한다.
>
> **범위**: Judge 오케스트레이터 + 5개 하위 에이전트 (Disclosure / News / Report / Profit / Cost). Portfolio Service(에이전트 아님)는 별도.
>
> **상태**: 단계 1~4 완료 / 단계 5 (confidence 산정) · 6 (references 권한) · Judge 별도 섹션 미완 / Profit-Cost 경계 3개 항목 팀 미팅 합의 필요.
>
> **관련**: LIBRA 시나리오 및 명세 v1 / LIBRA 4월 Work Items

---

## 1. 책임 경계

> **NOTE: 분할 원칙**
> 네 가지 기준으로 에이전트를 분할했다: Single Responsibility, Information Source Cohesion, Reasoning Pattern Similarity, Independent Failure Domain.

### 1.1 책임 경계 테이블 (6개 에이전트)

| 에이전트 | **책임 (YES)** | **책임 밖 (NO)** |
|---|---|---|
| **Disclosure Agent** | `libra-ingest`가 전달한 공시 payload 해석 / 공시 유형 분류 / 핵심 사실 추출 + 자연어 요약 / raw_signal 라벨링(점수 아님) / 예정 공시 일정 알림 | OpenDART 원천 수집/재시도/스케줄링 → `libra-ingest` / 시장 반응 해석 → News / 투자 판단 → Judge / 사업부별 원인 추론 → Report / 재무 데이터 재계산 → Portfolio Service |
| **News** | 뉴스/매체 검색(한/영) / 매체 cross-check / 시장 반응 관찰(정성) / 섹터 ETF 동향 / 백그라운드 모니터링 + push 트리거 / 임계치 기반 wake-up / **(macro 서브)** 지수/환율/금리 | 공시 원본 → Disclosure / 리포트 분석 → Report / 재무 데이터 해석 → Report / 가격 미래 예측 → Judge 합의 형성 |
| **Report** | 증권사 리포트 검색/분류 / 목표주가·투자의견·논지 추출 / 사업부별 원인 분해 / 여러 증권사 컨센서스 집계 / fallback 간접 단서 추출 | 뉴스 검색 → News / 공시 원본 → Disclosure / 최종 투자 결정 → Judge / 매크로 → News(macro 서브) |
| **Profit** | **리밸런싱 plan의 기대수익 시뮬레이션(1m/3m/...) / 샤프비율·최대낙폭 추정 / 시나리오 비교 평가(예: "대기의 기회비용") / 과거 base rate 조회(어닝 쇼크 등) / 팩터 노출도 분석** | 거래 실행 비용 계산 → Cost / 방향성 판단(매수/매도) → News/Report / 플랜 자체 수립 → Judge / 리밸런싱 결정 확정 → Judge |
| **Cost** | 거래 수수료 계산 / 세금 계산 / 슬리피지 추정 / 호가 스프레드 분석 / 트립와이어 **가격** 계산 / 실시간 유동성 상태 | 기대수익/리스크 시뮬레이션 → Profit / 과거 이벤트 base rate → Profit / 방향성 판단 → News/Report / 리밸런싱 결정 → Judge / 실제 주문 실행 → 3번 컴포넌트 |

### 1.2 빈틈 처리 (Judge로 흡수)

다음 책임은 6개 에이전트에 속하지 않고 Judge가 직접 처리한다:

- 사용자 사전 설정 해석
- 사용자 자율성 경계 판단 (USER_DECISION_REQUIRED 트리거)
- Decision Trace 기록 및 자연어 변환
- feedback_checkpoint 예약 및 미래 재평가
- **실행 방식(TWAP/VWAP/즉시) 선택** — Profit의 시뮬레이션 + Cost의 유동성을 종합해서 Judge가 결정

### 1.3 겹침 점검 결과

| 의심 지점 | 검토 결과 |
|---|---|
| "시장 반응"을 News vs Cost가 보는가 | News는 "왜 움직이는가"(정성), Cost는 "얼마나 거래 가능한가"(유동성). 관찰 각도 다름. |
| "과거 base rate" Profit vs Report | Profit은 가격/수익률 히스토리 기반(어닝 쇼크 후 평균 수익률 등), Report는 리포트 아카이브. DB/목적 다름. |
| 공시 vs 리포트 | Disclosure는 원본, Report는 증권사 해석. 깔끔하게 갈림. |
| **Profit vs Cost (신규)** | Profit은 "리밸런싱이 좋은 결정인가"(수익률/리스크 관점), Cost는 "이 리밸런싱을 얼마에 실행할 수 있나"(실행 마찰 관점). 입력(rebalance_plan)이 같아도 출력 차원이 다름. |

### 1.4 Profit-Cost 경계의 미정 영역

다음 항목은 Profit/Cost 중 어느 에이전트에 귀속할지 팀 미팅에서 최종 합의 필요. 본 v2에서는 Profit으로 잠정 배치:

- **TWAP/VWAP/분할 실행 계획 수립** (시나리오 D 턴 4) — 시간대별 분할 전략은 기대수익 시뮬레이션에 가까워 Profit 배치. 단 Cost가 분할 스케줄별 슬리피지를 제공해야 가능.
- **"대기의 기회비용" 평가** (시나리오 A 턴 4) — 미래 가격 변동 추정이 핵심이므로 Profit. 단 Cost가 시점별 거래비용 차이를 제공.
- **과거 어닝 쇼크 base rate** — 과거 수익률 분포이므로 Profit.

이 세 항목은 4월 팀 미팅 안건으로 LIBRA 4월 Work Items에 추가.

---

## 2. 공통 응답 스키마

> **✓ 이 스키마가 강제하는 명세**
> 1, 2, 3, 4 (confidence / reasoning / 빈손 금지 / 한계 명시) + C-1, C-2, C-3, C-4 (direction 축 / references / turn marker / 도구 로깅)

```python
class AgentResponse(BaseModel):
    # === 메타 ===
    agent_id: str                    # "disclosure" | "news" | "report" | "profit" | "cost"
    opinion_id: str                  # 이 응답 고유 ID (references에서 참조됨)
    turn_number: int                 # Judge가 부여한 턴 번호 (C-3)
    
    # === 질문 이해 ===
    query_understood: str
    
    # === 핵심 답 ===
    verdict: Literal[
        "DIRECT_ANSWER",
        "PARTIAL_ANSWER",
        "DIRECT_ANSWER_UNAVAILABLE",
        "QUIET"
    ]
    evidence: dict                   # 에이전트별 서브 스키마
    
    # === 방향성 투표 (C-1) ===
    direction: float                 # -1.0 ~ +1.0
    strength: float                  # 0.0 ~ 1.0
    urgency: Literal["immediate", "scheduled", "watch", "defer"]
    
    # === 메타 추론 ===
    confidence: float
    reasoning_for_judge_agent: str
    limits_acknowledged: Optional[str]
    
    # === 협업 (C-2) ===
    references: List[Reference] = []
    
    # === 추적성 (C-4) ===
    tools_called: List[ToolCall]
    depth_used: Literal["shallow", "medium", "deep"]
```

```python
class Reference(BaseModel):
    agent_id: str
    opinion_id: str  
    relation: Literal["request", "rebut", "agree", "depend"]
    note: Optional[str]

class ToolCall(BaseModel):
    tool_name: str
    purpose: str
    summary: str
```

### 2.1 Profit의 특수 케이스

Profit은 "방향성 투표자"가 아닌 "시뮬레이션 제공자"라서, `direction` 필드의 의미가 다른 에이전트와 다르다:

- **다른 에이전트의 direction**: "내 판단으로는 이 종목 매수/매도가 나음"
- **Profit의 direction**: "주어진 rebalance_plan을 실행했을 때 기대되는 수익 방향 (plan이 매수 비중이면 +, 매도 비중이면 -로 자연스럽게 나옴)"

즉 Profit의 direction은 **plan에 대한 평가**이지 **종목에 대한 판단**이 아니다. Judge가 합의 형성 시 이 차이를 반영해야 한다. 명세 보강 필요 (Judge 섹션에서 재논의).

### 2.2 Cost의 특수 케이스

Cost도 방향성을 내지 않는다. `direction: 0, strength: 0`이 기본. 단 `urgency`는 유동성 상태에 따라 줄 수 있다 (예: 현재 호가 스프레드가 평소의 3배면 `urgency: watch` = "지금 실행은 불리").

---

## 3. 에이전트별 Evidence 스키마

### 3.1 Disclosure

```python
class DisclosureEvidence(BaseModel):
    found_count: int
    items: List[DisclosureItem]
    upcoming_disclosures: List[UpcomingEvent] = []  # 빈손 금지

class DisclosureItem(BaseModel):
    ticker: str
    company_name: str
    disclosure_type: str             # "잠정실적공시" / "지분변동" 등
    timestamp: datetime
    summary: str
    raw_signal: Optional[str]        # "어닝 미스" 같은 정성 라벨. 점수 아님.
```

### 3.2 News

```python
class NewsEvidence(BaseModel):
    sub_role: Literal["macro", "company_specific", "mixed"]
    
    company_findings: Dict[str, CompanyNewsFinding] = {}
    macro_findings: Optional[MacroFinding] = None
    
    source_reliability: Literal["high", "medium", "low"]
    cross_check_count: int

class CompanyNewsFinding(BaseModel):
    sentiment: Literal["positive", "negative", "mixed", "neutral"]
    key_headlines: List[str]
    market_reaction: Optional[str]   # "주가 -3.4%, 거래량 6.8x"
    sector_comparison: Optional[str]

class MacroFinding(BaseModel):
    index_movements: Dict[str, float]
    fx_state: Optional[str]
    notable_events: List[str]
```

### 3.3 Report

```python
class ReportEvidence(BaseModel):
    coverage_reports_count: int
    preview_reports_count: int
    items: List[ReportItem]
    consensus: Optional[ConsensusData] = None
    indirect_inference: Optional[str]

class ReportItem(BaseModel):
    broker: str
    published_at: datetime
    report_type: Literal["preview", "coverage", "in_depth"]
    target_price: Optional[float]
    opinion: Optional[str]
    key_thesis: str
    business_segment_analysis: Optional[Dict[str, str]]

class ConsensusData(BaseModel):
    target_price_avg: float
    opinion_distribution: Dict[str, int]
```

### 3.4 Profit (신규)

```python
class ProfitEvidence(BaseModel):
    mode: Literal[
        "plan_simulation",       # rebalance_plan 받아서 기대수익 추정
        "scenario_compare",      # X vs Y 시나리오 비교 (대기 기회비용 등)
        "base_rate_query",       # 과거 유사 이벤트 base rate
        "execution_plan_build"   # TWAP/VWAP 스케줄 수립 (미정, 팀 미팅 안건)
    ]
    
    plan_simulation: Optional[PlanSimulation] = None
    scenario_compare: Optional[List[ScenarioResult]] = None
    base_rate: Optional[BaseRateResult] = None
    execution_plan: Optional[ExecutionPlanProposal] = None

class PlanSimulation(BaseModel):
    rebalance_plan: Dict[str, float]   # {"005930": -0.10, "000660": +0.10}
    expected_return_1m: float           # %
    expected_return_3m: float
    sharpe_ratio: float
    max_drawdown: float
    recommendation_text: str            # 자연어 요약

class ScenarioResult(BaseModel):
    scenario_name: str                  # "X_immediate" / "Y_wait_1hr"
    expected_price_move: float
    worst_case: float
    information_quality_note: Optional[str]

class BaseRateResult(BaseModel):
    event_type: str                     # "earnings_miss_10pct_plus"
    sample_count: int
    avg_1day_return: float
    avg_5day_return: float
    avg_30day_return: float
    caveat: Optional[str]               # "표본 작음" 등

class ExecutionPlanProposal(BaseModel):
    method: Literal["immediate", "TWAP", "VWAP", "sliced"]
    schedule: List[ScheduleSlice]
    expected_total_impact_bp: float     # Cost의 slippage와 결합 필요
```

### 3.5 Cost (축소된 정의)

```python
class CostEvidence(BaseModel):
    mode: Literal[
        "trade_cost",        # 단순 거래 비용 계산
        "liquidity_check",   # 현재 유동성 상태
        "tripwire_price"     # 트립와이어 발동 가격 계산
    ]
    
    trade_cost: Optional[TradeCostResult] = None
    liquidity: Optional[LiquiditySnapshot] = None
    tripwire_prices: Optional[TripwirePriceConfig] = None

class TradeCostResult(BaseModel):
    rebalance_plan: Dict[str, float]
    commission_krw: float
    tax_krw: float
    estimated_slippage_bp: float
    spread_state_bp: float
    total_friction_bp: float

class LiquiditySnapshot(BaseModel):
    bid_ask_spread_bp: float
    current_volume_vs_average: float    # 6.8 = 평균의 6.8배
    interpretation: str                 # "평소의 2.8배 스프레드" 등

class TripwirePriceConfig(BaseModel):
    levels: List[TripwirePrice]

class TripwirePrice(BaseModel):
    threshold_pct: float                # -5% / -8%
    trigger_price: float
    distance_from_current_pct: float
```

> **NOTE: Profit과 Cost의 경계 (축소된 Cost)**
> Cost는 "이 거래가 지금 얼마에 가능한가"만 답한다. 기대수익/리스크 시뮬레이션, 시나리오 비교, 과거 이벤트 base rate는 모두 Profit 영역.

### 3.6 시나리오별 Evidence 사용 매핑

| 시나리오-턴 | 에이전트 | Evidence mode | 비고 |
|---|---|---|---|
| A-턴4 (대기 기회비용) | Profit | `scenario_compare` | 가격 변동 추정은 Profit 영역 |
| D-턴4 (TWAP 실행 계획) | Profit | `execution_plan_build` | 실행 스케줄은 수익 최적화 관점 (팀 미팅 안건) |
| D-추가 (최종 friction 계산) | Cost | `trade_cost` | Profit의 execution_plan + Cost의 slippage 결합 |

시나리오 D는 **Profit → Cost 순으로 두 에이전트 호출**이 필요. 단일 호출의 단순성은 잃었지만 책임 분리가 깔끔.

---

## 4. 호출 도구 + depth별 동작

### 4.1 공통 원칙

- `shallow`: "특이사항 yes/no" 수준. 도구 1~2개.
- `medium`: "무엇이 있는가" 수준. 도구 3~4개. **default.**
- `deep`: "왜 그런가" 수준. 도구 5~6개 + 교차 검증.

같은 턴 내 depth 재호출 금지 — 다음 턴에서 deep으로 재호출. 에이전트는 `reasoning_for_judge_agent`로 제안만 가능.

### 4.2 Disclosure

> **repo scope note**
> 아래 도구 표는 Disclosure Agent가 필요로 하는 논리적 입력 능력을 설명한다. 실제 OpenDART 호출, 수집 스케줄링, 소스별 재시도는 `libra-ingest` 또는 별도 데이터 서비스의 책임이며, 현재 `libra-agent` repo는 정규화된 disclosure payload와 예정 일정 payload를 소비한다.

| 도구 | 제공처 | 용도 |
|---|---|---|
| `knowledge.disclosure_search` | `libra-ingest` / data service | 기간/종목 필터 disclosure payload |
| `knowledge.disclosure_detail` | `libra-ingest` / data service | 정규화된 공시 본문 + 첨부 요약 |
| `knowledge.company_profile` | reference service | 기업 기본 정보 |
| `knowledge.upcoming_events` | calendar service | 예정 일정 (빈손 금지) |

- shallow: list_disclosures (24h)
- medium: + get_disclosure_detail (7d)
- deep: + get_company_overview + upcoming_events (30d + 미래)

### 4.3 News

> **repo scope note**
> News Agent 역시 소스 수집기를 직접 포함하지 않는다. 여기서 말하는 도구는 논리적 정보원이며, push 트리거 자체의 감지와 원천 수집은 agent repo 밖에서 수행된다.

| 도구 | 제공처 | 용도 |
|---|---|---|
| `naver_news.search` | Naver News API | 한국어 헤드라인 |
| `naver_news.get_article` | Naver News API | 본문 분석 |
| `foreign_news.search` | Bing News / GDELT | 영문 뉴스 |
| `price_snapshot.get` | KIS | 실시간 가격/거래량(정성 관찰용) |
| `sector_etf.get_state` | KIS | 섹터 ETF 동향 |
| `conference_call.search_transcript` | 후보 미정 | 콜 트랜스크립트 |
| `push_trigger_stream` | 검증된 trigger 입력 | push 이벤트 컨텍스트 |
| `macro_snapshot.get` | BOK ECOS | 지수/환율/금리 (macro 서브) |

- shallow: naver_news.search + price_snapshot.get
- medium: + foreign_news.search + sector_etf.get_state
- deep: + naver_news.get_article + conference_call.search_transcript

### 4.4 Report

| 도구 | 제공처 | 용도 |
|---|---|---|
| `report_db.search` | 한경 컨센서스 / HTS | 메타 리스트 |
| `report_db.get_content` | 동상 | 본문 / PDF 파싱 |
| `report_db.get_consensus` | 한경 컨센서스 | 목표주가/의견 집계 |
| `report_db.get_historical_pattern` | 내부 DB | 과거 유사 이벤트 리포트 패턴 |

- shallow: search
- medium: + get_consensus
- deep: + get_content + get_historical_pattern

### 4.5 Profit (신규)

| 도구 | 제공처 | 용도 |
|---|---|---|
| `portfolio_simulator.monte_carlo` | 내부 모듈 | plan 시뮬레이션 (기대수익/낙폭) |
| `factor_model.exposure_check` | 내부 모듈 | 팩터 노출도 분석 |
| `historical_events.query` | 내부 DB (v1 Cost에서 이전) | 과거 유사 이벤트 수익률 분포 |
| `execution_planner.build_twap` | 내부 모듈 (v1 Cost에서 이전, 잠정) | TWAP 분할 스케줄 |
| `scenario_compare.evaluate` | 내부 모듈 | 시나리오 X vs Y 비교 |

- shallow: portfolio_simulator.monte_carlo (간단 모드)
- medium: + factor_model.exposure_check
- deep: + historical_events.query + execution_planner.build_twap + scenario_compare.evaluate

### 4.6 Cost (좁은 정의)

| 도구 | 제공처 | 용도 |
|---|---|---|
| `kis.get_orderbook` | KIS Open API | 호가창, 스프레드 |
| `kis.get_trade_history` | KIS Open API | 일중 거래량 패턴 |
| `market_microstructure.estimate_slippage` | 내부 모듈 | 슬리피지 추정 |
| `fee_calculator` | 정적 룰셋 | 수수료/세금 |
| `tripwire_price_calc` | 내부 모듈 | 트립와이어 가격 계산 |

- shallow: fee_calculator + kis.get_orderbook
- medium: + estimate_slippage + kis.get_trade_history
- deep: + tripwire_price_calc (필요 시)

### 4.7 4월 위험도 분류

| 에이전트 | 위험도 | 비고 |
|---|---|---|
| Disclosure | 🟢 안전 | 입력 스키마와 핵심 사실 추출 범위가 비교적 명확 |
| Cost | 🟢 안전 | KIS API 익숙 + 정적 룰셋 위주 (축소로 더 단순해짐) |
| Profit | 🟡 주의 | 몬테카를로 시뮬레이터 구축 필요, 과거 수익률 DB 구축 필요 |
| News | 🟡 주의 | push 입력 품질과 conference_call 대체 자료 범위가 불확실 |
| Report | 🔴 경고 | 데이터 소스 자체 미정. **4월 초 즉시 결정 필요** |

> **⚠️ 주의: Profit 에이전트의 4월 설계 함정**
> v1에서는 Cost 안에 "확장 역할"로 섞여 있어 작업량이 과소평가됐음. Profit으로 분리되면서 실제 구현 부담이 드러남:
> - 몬테카를로 시뮬레이터 (v1: 수동 룰 기반, v2+: 실제 모델)
> - 과거 수익률 DB 구축 (어닝 쇼크 등 이벤트 분류 + 당일/5일/30일 수익률 계산)
> - 팩터 모델 (4월 설계 범위 밖, v2 이후)
>
> **4월 v1 구현 권고**: 시뮬레이터는 정적 룰셋으로 충분. 과거 DB는 어닝 쇼크 이벤트만 먼저.

---

## 5. 시나리오별 호출 흐름

LIBRA 시나리오 및 명세 v1의 4개 시나리오에서 6개 에이전트가 어떻게 호출되는지 요약.

### 시나리오 A (정기 + 모호) — DEFER
Disclosure → News(deep) → Report → **Profit**(대기 기회비용 평가) → Judge 합의 → DEFER

### 시나리오 B (속보) — USER_DECISION_REQUIRED
News(push 트리거) → **Profit**(매도 시나리오별 수익 영향) → **Cost**(실행 friction + 트립와이어 가격) → Judge → USER_DECISION_REQUIRED

### 시나리오 C (평범한 날) — HOLD silent
Disclosure → News(shallow) → Judge → HOLD  
*Profit / Cost / Report 호출 안 됨 — 동적 호출의 비용 절감 효과*

### 시나리오 D (자동 REBALANCE)
Disclosure → Report → News → **Profit**(실행 계획 + 시뮬레이션) → **Cost**(friction 결합) → Judge → REBALANCE

> **⚠️ 주의: B, D에서 턴 하나씩 늘어남**
> v1의 Cost 단독 호출이 Profit + Cost 순차 호출로 변경. 책임 분리는 깔끔해졌지만 Judge 합의 시점이 뒤로 밀림. 사용자 알림 타이밍 재검토 필요 (팀 미팅 안건).

---

## 6. 팀 코드 ↔ 본 설계 필드 매핑표

> **목적**: 팀원의 `orchestrator.py`에서 사용한 필드명과 본 설계 문서의 필드명이 다름. 5월 구현 단계에서 통일 시 참조. 지금 통일하지 않고 매핑표로 기록만 함 (팀 미팅 안건).

| 팀 코드 (`orchestrator.py`) | 본 설계 (v2) | 비고 |
|---|---|---|
| `signal_score` (-1.0 ~ +1.0) | `direction` | 의미 동일 |
| `confidence` | `confidence` | 동일 |
| `urgency` (immediate/scheduled/watch) | `urgency` (+ defer) | 본 설계에 DEFER 추가 |
| `opinion` ("SELL_BIAS"/"BUY_BIAS"/"NEUTRAL"/...) | 대체: `direction` + `strength`로 분해 | enum → 수치로 일반화 |
| `trigger` (bool) | `verdict` 및 `urgency`로 대체 | 단순 bool보다 풍부한 상태 표현 |
| `risk_level` ("high"/"mid"/"low") | (미정) | 본 설계에는 대응 필드 없음. 통합 필요 |
| `event_type` ("regulation"/"earnings"/...) | Disclosure `raw_signal` 또는 News `CompanyNewsFinding.sentiment` 맥락 | 에이전트별 evidence로 분산 |
| `horizon` ("short"/"mid"/"long") | `urgency`와 의미 겹침, 정리 필요 | 통합 후보 |
| `affected_tickers` | evidence 내부 필드 | 스키마 자연스러움 |
| `source_count` | `NewsEvidence.cross_check_count` 등 | 에이전트별 evidence 필드 |
| `status` (ok/error/skipped) | `verdict` + 에러 핸들링 래퍼 | 구조 유지하며 내부 필드로 |
| 공통 래퍼 `{agent, status, reason, data}` | `AgentResponse` Pydantic 모델 | 정신 동일, 필드 확장 |

### 6.1 코드 구조적 장점 (유지 권고)

- **공통 래퍼 `{agent, status, reason, data}`**: 에러 핸들링이 깔끔. 우리 `AgentResponse`에도 `status` 필드 추가 고려 가치 있음.
- **Claude tool_use API 직접 활용**: `ORCHESTRATOR_TOOLS` 정의 + `tools=` 파라미터로 LLM이 직접 도구 호출. 우리 Judge 구현의 baseline으로 적합.
- **mock/real 이중 모드**: 테스트 용이성. 4월 설계 검증에 그대로 사용 가능.
- **bull/bear/neutral 시나리오 분기**: 우리 시나리오 A/B/C/D의 단순화 버전. 통합 시 우리 시나리오로 확장.

### 6.2 본 설계의 구조적 보강 (코드에 없는 것)

- 6개 에이전트 중 Profit 추가 (Cost 분리)
- `references` 필드 (에이전트 간 협업 관계 명시)
- `depth` 파라미터 (shallow/medium/deep)
- `verdict` enum (DIRECT_ANSWER / PARTIAL / UNAVAILABLE / QUIET)
- DEFER / USER_DECISION_REQUIRED 결정 타입
- Decision Trace 노드 리스트 구조
- Self-imposed deadline, 자율성 경계, 사용자 알림 4단계 강도

---

## 7. 작업 상태

### 완료
- [x] 6개 에이전트 책임 경계 정의
- [x] 공통 응답 스키마 + 에이전트별 Evidence 스키마
- [x] 호출 도구 + depth별 동작 정의
- [x] 시나리오 4개 × 에이전트 호출 흐름 매핑
- [x] 팀 코드(orchestrator.py 등) ↔ 본 설계 필드 매핑

### 미완 (다음 세션)
- [ ] 단계 5: 에이전트별 confidence 산정 방법 (가장 어려움)
- [ ] 단계 6: references 권한 매트릭스
- [ ] Judge 오케스트레이터 별도 섹션 (시스템 프롬프트, 합의 알고리즘, 호출 전략)

### 외부 의존
- [ ] LIBRA 시나리오 및 명세 v1 1.3 섹션 6개 에이전트 반영
- [ ] LIBRA 4월 Work Items D-6 (Profit 에이전트) 추가
- [ ] **팀 미팅 — 섹션 8의 4개 안건 합의**

---

## 8. 팀 미팅 안건

1. **Profit vs Cost 경계 — 3개 미정 항목 합의**
   - TWAP/VWAP 실행 계획 수립 → Profit? Cost? 공동?
   - 대기 기회비용 평가 → Profit 확정?
   - 과거 어닝 쇼크 base rate → Profit 확정?

2. **`risk_level`, `horizon`, `event_type` 필드의 본 스키마 흡수 방식**
   - 팀 코드에 있는데 본 스키마에 대응 없음
   - 삭제? 통합? 별도 필드로 유지?

3. **Profit 에이전트 4월 구현 범위 확정**
   - 4월 구현: 정적 룰셋 시뮬레이터 + 어닝 쇼크 이벤트 DB만
   - 팩터 모델, 몬테카를로는 5월 이후 — 팀 동의 필요

4. **Judge 호출 순서 변경에 따른 사용자 알림 타이밍**
   - 시나리오 B, D에서 Profit → Cost 순차로 턴 수 증가
   - 사용자가 체감할 응답 지연 검토

---

## 9. 변경 이력

**2026-04-13** — 단일 소스로 통합
- 이전 v1/v2 분리 구조 폐기. 본 문서가 유일 기준.

**2026-04-13** — Profit 에이전트 추가
- 팀 합의에 이미 존재했으나 누락. 팀원 `orchestrator.py`에서 확인되어 반영.
- Cost 책임 축소: 거래 실행 시 빠져나가는 비용(세금/수수료/슬리피지/스프레드)과 트립와이어 가격만 담당.
- 다음 항목이 Cost → Profit으로 이동: 기대수익/리스크 시뮬레이션, 시나리오 비교, 과거 base rate, TWAP 실행 계획(잠정, 팀 미팅 안건).
- 시나리오 A / B / D의 호출 흐름에 Profit 추가.
- 팀 코드와의 필드 매핑표 추가(섹션 8).

**2026-04-11** — 초기 작성
- 5개 에이전트(Judge + Disclosure / News / Report / Cost) 책임 경계 정의.
- 공통 응답 스키마 + Evidence 서브 스키마.
- 호출 도구 목록 + depth별 동작.
