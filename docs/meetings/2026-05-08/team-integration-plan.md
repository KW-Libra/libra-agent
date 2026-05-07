# 팀원 repo 통합 계획

## 원칙

팀원 repo는 단순 참고가 아니라 LIBRA의 5개 기능 컴포넌트에 배치해서 설명한다. 내일 미팅에서는 “모든 repo가 완전히 하나로 합쳐졌다”가 아니라, **핵심 런타임에 흡수한 항목**과 **추가 검증이 필요한 항목**을 구분해서 말한다.

## 기능 컴포넌트별 통합표

| 컴포넌트 | 목적 | 통합 대상 | 현재 상태 | 다음 작업 |
| --- | --- | --- | --- | --- |
| 데이터 수집 | 가격, 뉴스, 공시, 리포트 수집 | `libra-ingest`, 팀원 수집 repo | ingest 기반 문서/이벤트 입력 구조 존재. Report 문서는 `doc_type=REPORT` 입력으로 Agent가 해석하는 구조로 정리 | 팀원 collector 결과를 공통 `KnowledgeDocument`로 매핑하고 실제 샘플 ingest 검증 |
| AI 판단 | 멀티 에이전트 판단 | `libra-agent`, `HJ-agent`, `JYlibra-sample_v1` | 기존 6개 Agent + JY식 7개 도메인 Agent를 Judge 흐름에 흡수. Gemini-only 정책, 도메인 합의, domain signal 저장 구조 반영 | 실제 API key quota 상황에서 재시도/저장 포함 E2E 재검증 |
| 리밸런싱 처리 | 주문 후보, KIS 모의주문 | `libra-backend`, `libra-direct` | KIS credential, demo/real 환경 분기, 현재가 조회, 잔고 동기화, 모의주문 후보 API 흐름 연결 | 실제 모의투자 주문 실행 결과 저장과 실패 케이스 검증 |
| 사용자 리포팅 | 판단 이력, trace, 평가 | `libra-backend`, `libra-frontend` | Decision Trace에 7개 도메인 Agent 카드, compliance reject 배너, Gemini/Claude 검토 배지 반영 | 저장된 run이 없을 때의 empty state와 교수님용 설명 문구 정리 |
| 사용자 UI | 온보딩, 설정, 실행 게이트 | `JYlibra-sample_v1`, `libra-frontend` | JY의 검색/토론/신호 패널 흐름을 Vue로 이식. Topbar KIS 종목 검색, `/symbol/:ticker`, `/news` 신호 워크벤치, Dashboard command center 반영 | 온보딩 목표비중 저장 UX와 검색 결과 선택 흐름을 실제 KIS 종목 목록으로 추가 검증 |

## 이번 통합에서 실제 반영한 항목

| repo | 흡수한 내용 | 반영 위치 |
| --- | --- | --- |
| `JYlibra-sample_v1` | Agent debate, 7 domain agent verdict, market signal workbench, stock search/detail UX를 Vue 구조로 재구현 | `D:\libra-frontend\src\pages\DashboardPage.vue`, `DecisionTracePage.vue`, `NewsPage.vue`, `SymbolDetailPage.vue`, `components\search\StockSearch.vue` |
| `HJ-agent` | 7개 관점 Agent를 Judge 판단 보강 레이어로 통합하는 구조 | `D:\libra-agent\src\libra_agent\domain_agents`, `D:\libra-agent\src\libra_agent\libra_graph.py` |
| `libra-direct` | 사용자 초기 종목 선택, 목표 비중, KIS 가격/잔고 기반 direct indexing 입력 흐름 | `D:\libra-frontend\src\pages\OnboardingPage.vue`, `D:\libra-backend\src\main\java`의 portfolio/KIS API |

## 내일 말할 표현

“팀원들이 만든 repo를 기능 컴포넌트 단위로 재배치했습니다. 현재 핵심 런타임인 agent, backend, frontend에 7개 관점 Agent, KIS 종목 검색, 종목 상세, 신호 워크벤치, Decision Trace 시각화를 반영했습니다. 다만 실제 모의주문 저장과 ingest 샘플 검증은 다음 체크포인트로 남겨 두었습니다.”

## 말하지 말아야 할 표현

- “다 통합됐습니다.”
- “다른 팀원 repo는 필요 없습니다.”
- “제가 다 했습니다.”
- “나중에 어떻게든 붙이면 됩니다.”
