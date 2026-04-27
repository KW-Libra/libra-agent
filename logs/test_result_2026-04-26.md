
### [2026-04-26 20:52] orchestrator.py — 판단 결과 (BEAR)

**최종 결정**
- decision      : BLOCKED
- urgency       : watch
- confidence    : 0.78
- rebalance_plan: {'005930': -0.15, '000660': 0.15}
- reason        : 제약 조건 위반으로 차단: 005930 리밸런싱 후 비중 45%이 최소 허용 비중 45%보다 낮습니다.

**에이전트별 시그널**
- call_dart_agent
  signal_score=-0.52  opinion=SELL_BIAS  confidence=0.81
  reason: 공시 기반 시그널 분석 완료 (stub)
- call_news_agent
  signal_score=-0.38  opinion=SELL_BIAS  confidence=0.7
  reason: 뉴스 감성 분석 완료 (stub)
- call_cost_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 거래비용 계산 완료 (stub)
- call_profit_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 수익 시뮬레이션 완료 (stub)


### [2026-04-26 20:57] orchestrator.py — 판단 결과 (NEUTRAL)

**최종 결정**
- decision      : REBALANCE
- urgency       : scheduled
- confidence    : 0.68
- rebalance_plan: {'005930': 0.05, '000660': -0.05}
- reason        : 증권사 리포트에서 삼성전자(005930)에 대해 MILD_BUY 의견(신뢰도 0.72)이 확인되었으며, 공시 및 뉴스 시그널은 전반적으로 NEUTRAL로 리스크가 낮습니다. 리밸런싱 비용(2,450원, 0.025%)이 매우 낮고 3개월 기대수익 +2.4% 개선 효과가 예상되므로, 005930 비중을 60%→65%로 소폭 확대하고 000660을 40%→35%로 조정하는 예정 리밸런싱을 권고합니다.

**에이전트별 시그널**
- call_dart_agent
  signal_score=0.08  opinion=NEUTRAL  confidence=0.62
  reason: 공시 기반 시그널 분석 완료 (stub)
- call_news_agent
  signal_score=-0.05  opinion=NEUTRAL  confidence=0.55
  reason: 뉴스 감성 분석 완료 (stub)
- call_report_agent
  signal_score=0.12  opinion=MILD_BUY  confidence=0.72
  reason: 증권사 리포트 분석 완료 (stub)
- call_profit_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 수익 시뮬레이션 완료 (stub)
- call_cost_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 거래비용 계산 완료 (stub)


### [2026-04-26 20:59] orchestrator.py — 판단 결과 (BEAR)

**최종 결정**
- decision      : REBALANCE
- urgency       : immediate
- confidence    : 0.79
- rebalance_plan: {'005930': -0.15, '000660': 0.15}
- reason        : 삼성전자(005930)에 대해 DART 공시(규제 리스크, 신뢰도 81%), 뉴스 감성(-0.38), 증권사 리포트(MILD_SELL, 신뢰도 85%) 등 전 에이전트에서 일관된 매도 편향 시그널이 확인되어 비중을 60%→45%로 축소합니다. SK하이닉스(000660)는 상대적으로 부정 시그널이 없어 비중을 40%→55%로 확대하며 리스크를 분산합니다. 거래비용은 총 7,350원(비중 대비 0.073%)으로 리밸런싱 효익 대비 비용 부담이 낮아 즉시 실행이 적절합니다.

**에이전트별 시그널**
- call_dart_agent
  signal_score=-0.52  opinion=SELL_BIAS  confidence=0.81
  reason: 공시 기반 시그널 분석 완료 (stub)
- call_news_agent
  signal_score=-0.38  opinion=SELL_BIAS  confidence=0.7
  reason: 뉴스 감성 분석 완료 (stub)
- call_report_agent
  signal_score=-0.29  opinion=MILD_SELL  confidence=0.85
  reason: 증권사 리포트 분석 완료 (stub)
- call_profit_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 수익 시뮬레이션 완료 (stub)
- call_cost_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 거래비용 계산 완료 (stub)


### [2026-04-26 21:00] orchestrator.py — 판단 결과 (BULL)

**최종 결정**
- decision      : REBALANCE
- urgency       : scheduled
- confidence    : 0.82
- rebalance_plan: {'005930': -0.1, '000660': 0.1}
- reason        : 증권사 리포트(신뢰도 0.88)와 공시 분석(신뢰도 0.76) 모두 두 종목에 대해 BUY_BIAS 의견을 제시하였으며, 뉴스 감성 분석에서 000660(SK하이닉스)에 대한 긍정적 제품 이슈가 추가 확인되었습니다. 리밸런싱 후 3개월 기대수익 +9.1%, Sharpe 1.23으로 위험 대비 수익이 우수하고, 거래비용은 0.049%로 매우 낮아 005930 비중을 60%→50%로 줄이고 000660 비중을 40%→50%로 확대하는 예정 리밸런싱을 권고합니다.

**에이전트별 시그널**
- call_dart_agent
  signal_score=0.44  opinion=BUY_BIAS  confidence=0.76
  reason: 공시 기반 시그널 분석 완료 (stub)
- call_news_agent
  signal_score=0.31  opinion=MILD_BUY  confidence=0.68
  reason: 뉴스 감성 분석 완료 (stub)
- call_report_agent
  signal_score=0.58  opinion=BUY_BIAS  confidence=0.88
  reason: 증권사 리포트 분석 완료 (stub)
- call_profit_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 수익 시뮬레이션 완료 (stub)
- call_cost_agent
  signal_score=N/A  opinion=N/A  confidence=N/A
  reason: 거래비용 계산 완료 (stub)

