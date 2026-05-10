# LIBRA v1 Governance Runtime Implementation

작성일: 2026-05-10

## 목적

`design_intuition_v1.md`, `design_spec_v1.md`, `prompts_v1.md`의 핵심 설계를 코드에 옮기기 위한 1차 구현이다.

이번 변경은 기존 `JudgeOrchestrator`를 즉시 제거하지 않는다. 대신 새 설계의 핵심 불변 조건을 별도 v1 runtime으로 구현한다.

- Compliance는 LLM agent가 아니라 deterministic rule engine이다.
- 기존 `AgentResponse`는 새 `AgentOpinion` schema로 변환된다.
- Cost, Execution, Disclosure 같은 정보성 발화는 consensus 계산에서 제외된다.
- Final decision branch는 LLM이 아니라 코드가 결정한다.
- `COMPLIANCE_VETO`는 자동 거래 실행을 막고 `USER_DECISION_REQUIRED`로 전환한다.

## 추가된 코드

| 경로 | 역할 |
|---|---|
| `src/libra_agent/libra/schemas/` | v1 AgentOpinion, Vote, ComplianceCheck, FinalDecision, IPS/KYC schema |
| `src/libra_agent/libra/compliance/engine.py` | deterministic Compliance Rule Engine |
| `src/libra_agent/libra/mediator/consensus.py` | consensus score, branch classification, target selection |
| `src/libra_agent/libra/judge/final.py` | 4분기 final decision branch와 tentative trade 산정 |
| `src/libra_agent/libra/committee.py` | 기존 AgentResponse를 v1 governance runtime으로 연결하는 어댑터 |
| `src/libra_agent/libra/personas/v1.py` | PERSONA_V1 IPS/KYC fixture |
| `src/libra_agent/libra_api.py` | 기존 Judge API 응답에 `governance_v1` 결과 부착 |
| `scripts/run_benchmark.py` | v1 decision/branch/compliance를 decision matrix와 발표 보고서에 출력 |
| `tests/test_libra_v1_governance.py` | v1 governance 핵심 경로 테스트 |
| `tests/test_libra_agent_api.py` | Judge API v1 attachment 테스트 |

## 현재 구현 범위

### 완료

- `AgentOpinion` / `Vote` schema
- `ComplianceEngine`
- 10개 rule id의 기본 함수
- `IPS_SINGLE_TICKER_LIMIT` hard BLOCKING
- `IPS_SECTOR_LIMIT` hard BLOCKING
- `ESG_USER_EXCLUSION` hard BLOCKING
- `ESG_MIN_SCORE` hard BLOCKING
- `LIQUIDITY_MIN_CASH`, `IPS_VOLATILITY_LIMIT` WARNING
- `AgentResponse -> AgentOpinion` adapter
- legacy `ComplianceAgent`를 v1 opinion set에서 제외
- informational vote consensus 제외
- conflict target selection
- deterministic final branch
- `COMPLIANCE_VETO -> USER_DECISION_REQUIRED + 3 options`
- 기존 `/judge` API 응답에 `governance_v1` 부착
- scenario benchmark `decision_matrix.csv`와 `presentation_report.md`에 v1 branch/compliance 표시
- 09 ESG 충돌 시나리오에서 `ESG_MIN_SCORE -> COMPLIANCE_VETO` 실제 API 확인

### 아직 남음

- Round 1 실제 병렬 agent execution
- Mediator Judge LLM prompt/tool-use 연결
- Round 2 targeted recall prompt 연결
- Final Judge LLM prompt/tool-use 연결
- 기존 Judge API의 주 decision source를 v1 runtime으로 완전 전환
- scenario backtest와 4차원 metric wiring
- Tax lot 기반 `TAX_ANNUAL_GAIN_LIMIT`
- ADV/호가 기반 `MARKET_IMPACT_LIMIT`
- asset class mapping 기반 `IPS_ASSET_CLASS_BAND`

## 구현 판단

전체 시스템을 한 번에 교체하면 API, 프론트, benchmark가 동시에 깨질 위험이 크다. 그래서 v1 runtime은 기존 AgentResponse를 입력으로 받아 동작하게 만들었다.

즉, 기존 에이전트 구현을 버리지 않고 다음 순서로 갈아엎는다.

1. 기존 agent 실행 결과를 v1 schema로 변환
2. Compliance Rule Engine을 외부 hard-rule 레이어로 적용
3. consensus와 branch를 코드로 결정
4. 이후 Mediator/Final Judge LLM을 prompt/tool-use 기반으로 연결
5. 마지막에 기존 sequential Judge API를 v1 committee runtime으로 교체

## 검증

```powershell
D:\Libra\.venv\Scripts\python.exe -m unittest tests.test_libra_v1_governance -v
```

현재 5개 v1 governance 테스트가 통과한다.

API attachment와 benchmark reporting까지 포함한 smoke set:

```powershell
D:\Libra\.venv\Scripts\python.exe -m unittest tests.test_libra_v1_governance tests.test_libra_agent_api tests.test_benchmark_scenarios -v
```

현재 18개 테스트가 통과한다.

실제 API 벤치마크 확인:

```powershell
D:\Libra\.venv\Scripts\python.exe scripts\run_benchmark.py --scenario-id 09_esg_conflict --out-dir outputs\benchmark\live_anthropic_0510_v1_esg_veto_check --timeout-seconds 420
```

확인 결과 `decision_matrix.csv`에 다음 값이 기록된다.

- `governance_v1_decision=USER_DECISION_REQUIRED`
- `governance_v1_branch=COMPLIANCE_VETO`
- `governance_v1_compliance=BLOCKING`
