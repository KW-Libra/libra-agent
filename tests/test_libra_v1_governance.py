from __future__ import annotations

import unittest

from libra_agent.libra.committee import CommitteeRuntime, agent_response_to_opinion, responses_to_opinions, run_agent_callables_parallel
from libra_agent.libra.compliance import build_compliance_context_from_portfolio, default_compliance_engine
from libra_agent.libra.judge.final import determine_branch, render_rule_based_final_decision
from libra_agent.libra.mediator import classify_branch, compute_consensus, consensus_by_subject, select_targets
from libra_agent.libra.personas import persona_v1_ips, persona_v1_kyc
from libra_agent.libra.schemas import AgentOpinion, DecisionBranch, DecisionType, Direction, IPSConfig, MarketSnapshot, Severity, Trade, Vote
from libra_agent.libra_models import AgentResponse, AgentVerdict, PortfolioSnapshot, Urgency


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-05-10T09:00:00+09:00",
            "holdings": [
                {"ticker": "069500", "company_name": "KODEX 200", "weight": 0.32, "sector": "EQUITY"},
                {"ticker": "153130", "company_name": "KODEX 단기채권", "weight": 0.18, "sector": "BOND"},
                {"ticker": "379800", "company_name": "KODEX 미국S&P500", "weight": 0.25, "sector": "EQUITY"},
                {"ticker": "132030", "company_name": "KODEX 골드선물", "weight": 0.10, "sector": "ALT"},
            ],
            "cash_weight": 0.15,
        }
    )


def _response(
    agent_id: str,
    *,
    direction: float,
    confidence: float = 0.8,
    strength: float = 0.8,
    focus_tickers: list[str] | None = None,
) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        opinion_id=f"{agent_id}_1",
        turn_number=1,
        query_understood="테스트",
        verdict=AgentVerdict.PARTIAL_ANSWER,
        evidence={},
        direction=direction,
        strength=strength,
        urgency=Urgency.DEFER,
        confidence=confidence,
        reasoning_for_judge_agent=f"{agent_id} 판단",
        focus_tickers=focus_tickers or ["069500"],
    )


class LibraV1GovernanceTests(unittest.TestCase):
    def test_compliance_engine_blocks_existing_single_ticker_violation(self) -> None:
        ctx = build_compliance_context_from_portfolio(_portfolio(), ips=persona_v1_ips(), kyc=persona_v1_kyc())
        check = default_compliance_engine().check(ctx, "BEFORE")

        self.assertFalse(check.can_proceed)
        self.assertTrue(any(v.rule_id == "IPS_SINGLE_TICKER_LIMIT" and v.severity == Severity.BLOCKING for v in check.violations))

    def test_compliance_veto_is_external_to_committee_votes(self) -> None:
        portfolio = _portfolio()
        ips = persona_v1_ips()
        market = MarketSnapshot(sector_map={"449450": "WEAPONS", "069500": "EQUITY"})
        ctx = build_compliance_context_from_portfolio(
            portfolio,
            proposed_trades=[Trade("449450", 2.0, "방산 ETF 매수 후보")],
            ips=ips,
            kyc=persona_v1_kyc(),
            market_data=market,
        )
        compliance_after = default_compliance_engine().check(ctx, "AFTER")
        consensus = {
            "449450": consensus_by_subject(
                [
                    AgentOpinion("Profit", votes=[Vote("449450", Direction.INCREASE, 4.0, 0.8)]),
                    AgentOpinion("ESG", votes=[Vote("449450", Direction.DECREASE, -5.0, 0.95)]),
                ]
            )["449450"]
        }

        decision, branch = determine_branch(consensus, compliance_after)
        rendered = render_rule_based_final_decision(
            consensus_per_subject=consensus,
            votes=[Vote("449450", Direction.INCREASE, 4.0, 0.8)],
            compliance_after=compliance_after,
        )

        self.assertEqual(decision, DecisionType.USER_DECISION_REQUIRED)
        self.assertEqual(branch, DecisionBranch.COMPLIANCE_VETO)
        self.assertEqual(rendered.trades, [])
        self.assertEqual(len(rendered.user_options or []), 3)

    def test_compliance_engine_blocks_esg_min_score(self) -> None:
        ctx = build_compliance_context_from_portfolio(
            _portfolio(),
            ips=IPSConfig(esg_min_score=70.0),
            kyc=persona_v1_kyc(),
            market_data=MarketSnapshot(esg_score={"069500": 58.0}),
        )

        check = default_compliance_engine().check(ctx, "AFTER")

        self.assertFalse(check.can_proceed)
        self.assertTrue(any(v.rule_id == "ESG_MIN_SCORE" for v in check.violations))

    def test_consensus_ignores_informational_votes_and_selects_conflict_targets(self) -> None:
        opinions = [
            AgentOpinion("Profit", votes=[Vote("069500", Direction.INCREASE, 5.0, 0.9)]),
            AgentOpinion("Risk", votes=[Vote("069500", Direction.DECREASE, -5.0, 0.9)]),
            AgentOpinion("Cost", votes=[Vote("069500", Direction.HOLD, 0.0, 1.0, informational=True)]),
        ]
        scores = consensus_by_subject(opinions)

        self.assertAlmostEqual(compute_consensus([vote for op in opinions for vote in op.votes]), 0.0)
        self.assertEqual(classify_branch([vote for op in opinions for vote in op.votes]).value, "CONFLICT")
        self.assertEqual(scores["069500"].branch.value, "CONFLICT")
        self.assertEqual(select_targets(scores, opinions), ["Profit", "Risk"])

    def test_agent_response_adapter_excludes_legacy_compliance_agent(self) -> None:
        responses = [
            _response("profit", direction=0.8),
            _response("cost", direction=0.0),
            _response("compliance", direction=-1.0),
        ]
        opinions = responses_to_opinions(responses)
        cost = agent_response_to_opinion(responses[1])

        self.assertEqual([opinion.agent for opinion in opinions], ["Profit", "Cost"])
        self.assertTrue(cost.votes[0].informational)

    def test_committee_runtime_returns_v1_final_decision(self) -> None:
        result = CommitteeRuntime().run_from_agent_responses(
            portfolio=_portfolio(),
            responses=[
                _response("profit", direction=-0.9, focus_tickers=["069500"]),
                _response("risk", direction=-0.8, focus_tickers=["069500"]),
                _response("cost", direction=0.0, focus_tickers=["069500"]),
            ],
            ips=persona_v1_ips(),
            kyc=persona_v1_kyc(),
        )

        self.assertFalse(result.compliance_before.can_proceed)
        self.assertEqual(result.final_decision.decision, DecisionType.USER_DECISION_REQUIRED)
        self.assertEqual(result.final_decision.branch, DecisionBranch.COMPLIANCE_VETO)

    def test_round1_agent_callables_run_in_declared_order_after_parallel_execution(self) -> None:
        responses = run_agent_callables_parallel(
            {
                "profit": lambda: _response("profit", direction=0.7),
                "risk": lambda: _response("risk", direction=-0.7),
                "cost": lambda: _response("cost", direction=0.0),
            }
        )

        self.assertEqual([response.agent_id for response in responses], ["profit", "risk", "cost"])


if __name__ == "__main__":
    unittest.main()
