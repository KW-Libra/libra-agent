# 2026-05-08 LIBRA 팀 공유용 작업 정리

이 문서는 `feat/seven-agent-merge` 브랜치에 반영된 작업을 팀원들이 빠르게 확인할 수 있도록 정리한 것이다. 핵심은 교수님 피드백에 대응하기 위해 LIBRA를 "개인 투자자를 위한 설명 가능한 AI 멀티 에이전트 기반 자동 리밸런싱 판단 시스템"으로 재정의하고, 그에 맞는 데모 흐름과 구현 근거를 네 개 활성 레포에 나누어 반영한 것이다.

## 한 줄 요약

이번 작업은 단순 화면 수정이 아니라, 회원가입 이후 초기 포트폴리오 설정, KIS 기반 종목/잔고/가격 연동, Judge 중심 멀티 에이전트 판단, Decision Trace 시각화, 정성 신호 정량화 검증까지 이어지는 중간발표용 통합 브랜치를 만드는 작업이었다.

## GitHub 브랜치와 PR

| 레포 | 브랜치 | 커밋 | PR |
|---|---|---:|---|
| `libra-frontend` | `feat/seven-agent-merge` | `65acbe8` | https://github.com/KW-Libra/libra-frontend/pull/new/feat/seven-agent-merge |
| `libra-backend` | `feat/seven-agent-merge` | `556c7ae` | https://github.com/KW-Libra/libra-backend/pull/new/feat/seven-agent-merge |
| `libra-agent` | `feat/seven-agent-merge` | `02fe8a1` | https://github.com/KW-Libra/libra-agent/pull/new/feat/seven-agent-merge |
| `libra-ingest` | `feat/seven-agent-merge` | `2b63ee4` | https://github.com/KW-Libra/libra-ingest/pull/new/feat/seven-agent-merge |

## 왜 이 작업을 했는가

교수님 피드백에서 가장 크게 지적된 부분은 다음 네 가지였다.

1. "다이렉트 인덱싱"만으로는 프로젝트 목적이 직관적이지 않다.
2. ETF, 로보어드바이저, 다이렉트 인덱싱 비교표보다 문제 정의와 필요성 흐름이 먼저 필요하다.
3. 멀티 에이전트라고 부르려면 각 관점 에이전트가 무엇을 판단하고 Judge가 어떻게 종합하는지 보여야 한다.
4. 성능 검증은 서비스 안정성이 아니라 Buy & Hold, 기계적 리밸런싱, LIBRA 판단 기반 리밸런싱 비교로 가야 한다.

그래서 이번 브랜치에서는 "AI 에이전트 기반 자동 포트폴리오 리밸런싱 판단 시스템"이라는 방향에 맞춰 기능, 화면, 에이전트 구조, 검증 문서를 함께 정리했다.

## 내가 진행한 주요 작업

### 1. 중간발표와 교수님 미팅 대응 자료 정리

`libra-agent`에 미팅용 자료를 만들었다.

- `docs/meetings/2026-05-08/LIBRA_2026-05-08_professor_meeting.pptx`
- `docs/meetings/2026-05-08/meeting-brief.md`
- `docs/meetings/2026-05-08/requirements-spec.md`
- `docs/meetings/2026-05-08/qa-defense.md`
- `docs/meetings/2026-05-08/verification-plan.md`
- `docs/meetings/2026-05-08/team-integration-plan.md`
- `docs/architecture/seven-agent-merge-design.md`

자료 내용은 제목 재정의, 배경 스토리라인, 요구사항 명세, 펑셔널 아키텍처, 멀티 에이전트 구조, 검증 계획, 예상 질문 답변으로 구성했다.

### 2. 멀티 에이전트 판단 구조 통합

`libra-agent`에 기존 Judge 흐름과 도메인 에이전트들을 연결했다.

- 기존 판단 축: `disclosure`, `news`, `report`, `profit`, `cost`
- 신규 도메인 에이전트: `risk`, `tax`, `compliance`, `macro`, `sentiment`, `execution`, `esg`
- 신규 모듈:
  - `src/libra_agent/domain_agents/`
  - `src/libra_agent/domain_agents/_adapter.py`
  - `src/libra_agent/domain_agents/_consensus.py`
  - `src/libra_agent/domain_agents/_services/llm_router.py`
  - `src/libra_agent/gemini_client.py`
  - `src/libra_agent/sentiment/`

핵심 의도는 단일 함수가 결론을 내리는 구조가 아니라, 관점별 에이전트가 판단을 내고 Judge가 합의, 충돌, 거부권, 실행 가능성을 종합하는 구조를 보여주는 것이다.

### 3. Claude/Gemini/로컬 모델 혼합 실행 기반 정리

에이전트 레포에서 LLM 호출 경로를 정리했다.

- Claude 계열 호출
- Gemini 호출
- FinBERT 또는 로컬 감성 분석 경로
- LLM 라우팅 정책
- 모델 간 불일치 감지 로그

이 작업은 "휴리스틱만으로 판단한다"는 인상을 줄이기 위한 것이다. 수치 계산이 필요한 부분은 결정론적으로 처리하되, 정성 판단과 설명 생성은 LLM 기반 에이전트가 담당하도록 분리했다.

### 4. 백엔드 통합

`libra-backend`에는 프론트와 에이전트를 실제 서비스 흐름으로 연결하는 API들을 추가했다.

- 회원가입/로그인/JWT/OAuth 인증 흐름
- 사용자 단위 리소스 분리
- KIS 인증 정보 등록/상태 조회
- KIS 잔고 동기화
- KIS 현재가 조회
- KIS 종목 마스터/검색 흐름
- 모의투자 주문 요청 경로
- 목표 포트폴리오 정의 저장
- Decision Run 상세 응답 확장
- Agent Signal 저장 구조 확장
- Decision Execution/Order Proposal 구조
- Flyway 마이그레이션 `V3` ~ `V7`

이 작업으로 "회원가입 -> 초기 설정 -> 종목 선택 -> 목표 비중 설정 -> 판단 실행 -> 주문 후보 확인" 흐름의 백엔드 기반을 만들었다.

### 5. 프론트엔드 UX와 디자인 재구성

`libra-frontend`에는 기존 화면을 단순 색상 변경이 아니라 서비스 흐름 중심으로 다시 구성했다.

- 로그인/회원가입 화면 재구성
- 공통 앱 셸과 네비게이션 정리
- 온보딩 화면 추가
- KIS 종목 검색 컴포넌트 추가
- 대시보드 재구성
- Decision Trace 화면 추가
- History, Profile, Agents, News, Indexing, Symbol Detail 화면 정리
- Agent Flow, Confidence Gauge, Plan Delta, Portfolio Cells 등 시각화 컴포넌트 추가
- 401 에러가 날 때 원문 `Unauthorized` 대신 사용자용 한국어 메시지 표시
- 모바일/데스크톱에서 가로 overflow가 생기지 않도록 주요 화면 확인

디자인 방향은 "금융 SaaS/운용 대시보드"에 맞춰, 마케팅 페이지처럼 보이기보다 실제 판단과 실행 흐름을 읽을 수 있는 작업 화면에 가깝게 잡았다.

### 6. Ingest 쪽 정성 신호 정량화 근거 추가

`libra-ingest`에는 교수님이 물을 가능성이 높은 "뉴스, 공시, 리포트 같은 정성 데이터를 어떻게 정량화하나"에 대한 구현 근거를 추가했다.

- `src/libra_ingest/pipeline/signal_scorer.py`
- `src/libra_ingest/pipeline/signal_validation.py`
- `contracts/signal-validation.schema.json`
- `contracts/signal-calibration.schema.json`
- `examples/price-history.sample.csv`

주요 기능은 다음과 같다.

- 문서의 긍정/부정/리스크 단어와 문서 타입을 기반으로 `sentiment_score`, `impact_score`, `risk_score`, `time_decay` 생성
- 이벤트 신호를 가격 히스토리와 연결해 forward return, abnormal return, 방향 적중률 계산
- 여러 가중치 후보 중 in-sample 기준으로 가장 나은 보정 후보 산출
- CLI 옵션으로 `signal_validation.json`, `signal_calibration.json` 출력

이 부분은 당장 수익률 우위를 주장하기 위한 것이 아니라, "검증 가능한 신호 체계로 발전시키고 있다"는 근거다.

## 팀원 작업과 연결되는 부분

다른 팀원 레포의 아이디어는 활성 레포에 그대로 복사하는 방식이 아니라, 현재 서비스 구조에 맞게 흡수하는 방향으로 정리했다.

| 연결 영역 | 반영 위치 | 팀 공유 시 설명 |
|---|---|---|
| 다이렉트 인덱싱 입력/초기 설정 | `libra-frontend`, `libra-backend` | 사용자가 직접 종목과 목표 비중을 설정하는 흐름으로 연결 |
| 멀티 에이전트 판단 | `libra-agent` | 관점별 에이전트를 Judge가 종합하는 구조로 통합 |
| 뉴스/감성 분석 | `libra-agent`, `libra-ingest` | 감성 분석 결과를 Agent 판단과 정량 신호에 연결 |
| 리포트/공시/뉴스 수집 | `libra-ingest` | 정규화 이벤트와 신호 검증 대상으로 연결 |
| 사용자 시연 화면 | `libra-frontend` | Decision Trace, Dashboard, Profile, Onboarding으로 확인 |

팀원들에게는 "각자 만든 기능이 어느 활성 레포의 어느 책임 영역으로 들어갔는지"를 이 표 기준으로 확인해달라고 요청하면 된다.

## 검증한 내용

| 레포 | 검증 명령 | 결과 |
|---|---|---|
| `libra-frontend` | `npm run build` | 통과 |
| `libra-frontend` | Playwright 주요 화면 점검 | 데스크톱/모바일 주요 화면 가로 overflow 없음 |
| `libra-backend` | `.\gradlew test` | 통과 |
| `libra-agent` | `python -m pytest` | 51개 통과 |
| `libra-ingest` | `python -m pytest` | 16개 통과 |

커밋 전에는 staged diff 공백 검사와 명백한 API 키 패턴 검색도 수행했다.

## 아직 남은 일

이번 브랜치는 "중간발표와 팀 통합을 위한 기준선"이지 최종 완성본은 아니다. 남은 일은 다음과 같다.

1. 팀원들이 각 PR에서 자기 담당 기능이 제대로 반영됐는지 리뷰한다.
2. KIS 실계정/모의투자 키는 각자 로컬 환경에서 다시 등록해 확인한다.
3. Agent API, Backend, Frontend를 동시에 띄운 상태에서 실제 계정 기반 E2E를 다시 확인한다.
4. Buy & Hold, 기계적 리밸런싱, LIBRA 판단 리밸런싱 비교 검증 데이터를 더 채운다.
5. 리포트 수집 결과가 Report Agent 판단에 얼마나 잘 연결되는지 샘플을 늘린다.
6. 프론트 디자인은 현재 1차 정리 상태이므로, 팀 피드백 후 더 다듬는다.

## 팀에 보낼 메시지 초안

아래 문장을 그대로 공유해도 된다.

```text
feat/seven-agent-merge 브랜치로 frontend/backend/agent/ingest 4개 레포에 중간발표용 통합 작업을 올렸습니다.

이번 작업 범위는 회원가입/초기 설정/KIS 연동/목표 포트폴리오/멀티 에이전트 Judge/Decision Trace/정성 신호 검증까지입니다. 각자 담당했던 기능이 활성 레포에 어떻게 반영됐는지 PR에서 확인해 주세요.

검증은 frontend npm run build, backend gradlew test, agent pytest 51개, ingest pytest 16개까지 통과했습니다.

특히 교수님 미팅 대비 문서는 libra-agent의 docs/meetings/2026-05-08, 전체 작업 요약은 docs/handoff/2026-05-08-team-progress-summary.md에 정리해뒀습니다.
```
