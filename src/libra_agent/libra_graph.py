from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from libra_agent.runtime.debate_events import publish_debate_event

from .libra.direct_indexing import (
    PortfolioDefinition,
    candidate_plan_from_drift,
    compute_drift,
)
from .libra_models import (
    AgentResponse,
    DecisionType,
    JudgeDecision,
    PortfolioSnapshot,
    TriggerEvent,
    Urgency,
    UserNotification,
)
from .libra_runtime import (
    LocalKnowledgeBase,
    PlannedAgentCall,
    RunState,
    candidate_plan_to_proposed_trades,
    canonical_agent_id,
)
from .libra_validation import sanitize_agent_response_payload, sanitize_judge_payload
from .utils import stable_hash


class LibraGraphState(TypedDict, total=False):
    query: str
    portfolio: dict[str, Any]
    knowledge_base: dict[str, Any]
    depth: str
    trigger: str
    trigger_event: dict[str, Any] | None
    portfolio_definition: dict[str, Any] | None
    drift_report: dict[str, Any] | None
    deadline_seconds: int | None
    thread_id: str
    enable_human_interrupts: bool
    run_state: dict[str, Any]
    called_agents: list[str]
    skipped_agents: list[str]
    skip_rationale: dict[str, str]
    responses: list[dict[str, Any]]
    executed_calls: list[dict[str, Any]]
    judge_actions: list[dict[str, Any]]
    domain_consensus: dict[str, Any]
    pending_call: dict[str, Any] | None
    turn_number: int
    candidate_plan: dict[str, float]
    needs_trade_eval: bool
    final_result: dict[str, Any]
    pending_call_layer: str | None


class LibraLangGraphRuntime:
    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator
        self._compiled_no_checkpoint = None

    def run(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        portfolio_definition: PortfolioDefinition | None = None,
        depth: str = "medium",
        trigger: str = "pull",
        trigger_event: TriggerEvent | None = None,
        deadline_seconds: int | None = None,
        thread_id: str | None = None,
        enable_human_interrupts: bool = False,
    ) -> dict[str, Any]:
        runtime_thread_id = thread_id or self._default_thread_id(
            query=query,
            trigger=trigger,
            trigger_event=trigger_event,
            portfolio=portfolio,
        )
        runtime_portfolio = self._portfolio_with_definition_targets(portfolio, portfolio_definition)
        input_state: LibraGraphState = {
            "query": query,
            "portfolio": runtime_portfolio.to_dict(),
            "knowledge_base": knowledge_base.to_state_payload(),
            "portfolio_definition": portfolio_definition.to_dict()
            if portfolio_definition
            else None,
            "depth": depth,
            "trigger": trigger,
            "trigger_event": trigger_event.to_dict() if trigger_event else None,
            "deadline_seconds": deadline_seconds,
            "thread_id": runtime_thread_id,
            "enable_human_interrupts": enable_human_interrupts,
        }

        result = self._invoke_graph(payload=input_state, thread_id=runtime_thread_id)
        return self._normalize_result(result=result, thread_id=runtime_thread_id)

    def resume(
        self,
        *,
        thread_id: str,
        resume_payload: Any,
    ) -> dict[str, Any]:
        if self.coordinator.checkpoint_path is None:
            raise RuntimeError("LangGraph resume requires a checkpoint_path.")
        runtime_thread_id = thread_id.strip()
        if not runtime_thread_id:
            raise RuntimeError("thread_id is required to resume a LangGraph run.")
        result = self._invoke_graph(
            payload=Command(resume=resume_payload), thread_id=runtime_thread_id
        )
        return self._normalize_result(result=result, thread_id=runtime_thread_id)

    def _compiled_graph(self):
        if self._compiled_no_checkpoint is None:
            self._compiled_no_checkpoint = self._build_graph().compile(name="libra_judge")
        return self._compiled_no_checkpoint

    def _invoke_graph(self, *, payload: Any, thread_id: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        if self.coordinator.checkpoint_path is None:
            graph = self._compiled_graph()
            return graph.invoke(payload, config=config)

        checkpoint_path = Path(self.coordinator.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            graph = self._build_graph().compile(checkpointer=saver, name="libra_judge")
            return graph.invoke(payload, config=config)

    def _normalize_result(self, *, result: Mapping[str, Any], thread_id: str) -> dict[str, Any]:
        final_result = dict(result.get("final_result", {}))
        runtime = (
            dict(final_result.get("runtime", {}))
            if isinstance(final_result.get("runtime"), Mapping)
            else {}
        )
        runtime.update(
            {
                "engine": "langgraph",
                "thread_id": thread_id,
                "checkpoint_path": str(self.coordinator.checkpoint_path)
                if self.coordinator.checkpoint_path
                else None,
            }
        )
        interrupts = self._serialize_interrupts(result.get("__interrupt__", ()))
        runtime["interrupted"] = bool(interrupts)
        runtime["resume_required"] = bool(interrupts)
        if interrupts:
            runtime["interrupts"] = interrupts
        final_result["runtime"] = runtime
        return final_result

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(LibraGraphState)
        builder.add_node("initialize", self._initialize_node)
        builder.add_node("judge_orchestrate", self._judge_orchestrate)
        builder.add_node("domain_orchestrate", self._domain_orchestrate)
        builder.add_node("execute_agent", self._execute_agent)
        builder.add_node("domain_consensus", self._domain_consensus)
        builder.add_node("final_judge", self._final_judge)
        builder.add_node("route_human_review", self._route_human_review)
        builder.add_node("human_review", self._human_review)
        builder.add_node("deadline_terminal", self._deadline_terminal)

        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "judge_orchestrate")
        builder.add_edge("domain_consensus", "final_judge")
        builder.add_edge("final_judge", "route_human_review")
        builder.add_edge("human_review", END)
        builder.add_edge("deadline_terminal", END)
        return builder

    def _initialize_node(self, state: LibraGraphState) -> dict[str, Any]:
        trigger_event = self._trigger_event_from_state(state)
        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        portfolio_definition = self._portfolio_definition_from_state(state)
        drift_report = (
            compute_drift(portfolio_definition, portfolio) if portfolio_definition else None
        )
        candidate_plan = candidate_plan_from_drift(drift_report) if drift_report else {}
        run_state = self.coordinator._initialize_run_state(
            trigger=state["trigger"],
            trigger_event=trigger_event,
            deadline_seconds=state.get("deadline_seconds"),
        )
        return {
            "run_state": self._serialize_run_state(run_state),
            "called_agents": [],
            "skipped_agents": [],
            "skip_rationale": {},
            "responses": [],
            "executed_calls": [],
            "judge_actions": [],
            "domain_consensus": {},
            "pending_call": None,
            "pending_call_layer": None,
            "turn_number": 1,
            "candidate_plan": candidate_plan,
            "needs_trade_eval": bool(candidate_plan),
            "drift_report": drift_report.to_dict() if drift_report else None,
            "final_result": {},
        }

    def _judge_orchestrate(
        self, state: LibraGraphState
    ) -> Command[Literal["deadline_terminal", "execute_agent", "domain_orchestrate"]]:
        run_state = self._deserialize_run_state(state["run_state"])
        if self.coordinator._deadline_exceeded(run_state):
            return Command(goto="deadline_terminal")

        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        trigger_event = self._trigger_event_from_state(state)
        responses = [AgentResponse.from_dict(item) for item in state["responses"]]
        action = self.coordinator._judge_next_action(
            query=state["query"],
            portfolio=portfolio,
            responses=responses,
            called_agents=list(state["called_agents"]),
            depth=state["depth"],
            trigger=state["trigger"],
            trigger_event=trigger_event,
            candidate_plan=state.get("candidate_plan", {}),
        )
        action_plan = action.get("candidate_rebalance_plan", {})
        candidate_plan = (
            dict(action_plan)
            if isinstance(action_plan, Mapping) and action_plan
            else dict(state.get("candidate_plan", {}))
        )
        judge_actions = list(state.get("judge_actions", []))
        judge_actions.append(
            self._serialize_judge_action(
                action,
                turn_number=int(state.get("turn_number", 1)),
                called_agents=list(state["called_agents"]),
                response_count=len(responses),
            )
        )
        update: dict[str, Any] = {
            "candidate_plan": candidate_plan,
            "needs_trade_eval": bool(candidate_plan),
            "judge_actions": judge_actions,
            "pending_call": None,
            "pending_call_layer": None,
        }
        if action.get("action") == "CALL_AGENT":
            update["pending_call"] = self._serialize_planned_call(
                PlannedAgentCall(
                    agent_id=str(action["agent_id"]),
                    query=str(action["query"]),
                    context=str(action["context"]),
                    depth=str(action["depth"]),
                    fallback=str(action["fallback"])
                    if action.get("fallback") is not None
                    else None,
                    note=str(action["note"]) if action.get("note") is not None else None,
                )
            )
            update["pending_call_layer"] = "core"
            return Command(update=update, goto="execute_agent")
        return Command(update=update, goto="domain_orchestrate")

    def _domain_orchestrate(
        self, state: LibraGraphState
    ) -> Command[Literal["deadline_terminal", "execute_agent", "domain_consensus"]]:
        run_state = self._deserialize_run_state(state["run_state"])
        if self.coordinator._deadline_exceeded(run_state):
            return Command(goto="deadline_terminal")

        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        trigger_event = self._trigger_event_from_state(state)
        responses = [AgentResponse.from_dict(item) for item in state["responses"]]
        action = self.coordinator._domain_next_action(
            query=state["query"],
            portfolio=portfolio,
            responses=responses,
            called_agents=list(state["called_agents"]),
            depth=state["depth"],
            trigger=state["trigger"],
            trigger_event=trigger_event,
            candidate_plan=state.get("candidate_plan", {}),
        )
        action_plan = action.get("candidate_rebalance_plan", {})
        candidate_plan = (
            dict(action_plan)
            if isinstance(action_plan, Mapping) and action_plan
            else dict(state.get("candidate_plan", {}))
        )
        judge_actions = list(state.get("judge_actions", []))
        judge_actions.append(
            self._serialize_judge_action(
                action,
                turn_number=int(state.get("turn_number", 1)),
                called_agents=list(state["called_agents"]),
                response_count=len(responses),
            )
        )
        update: dict[str, Any] = {
            "candidate_plan": candidate_plan,
            "needs_trade_eval": bool(candidate_plan),
            "judge_actions": judge_actions,
            "pending_call": None,
            "pending_call_layer": None,
        }
        if action.get("action") == "CALL_AGENT":
            update["pending_call"] = self._serialize_planned_call(
                PlannedAgentCall(
                    agent_id=str(action["agent_id"]),
                    query=str(action["query"]),
                    context=str(action["context"]),
                    depth=str(action["depth"]),
                    fallback=str(action["fallback"])
                    if action.get("fallback") is not None
                    else None,
                    note=str(action["note"]) if action.get("note") is not None else None,
                )
            )
            update["pending_call_layer"] = "domain"
            return Command(update=update, goto="execute_agent")
        return Command(update=update, goto="domain_consensus")

    def _execute_agent(
        self, state: LibraGraphState
    ) -> Command[Literal["judge_orchestrate", "domain_orchestrate"]]:
        pending_call_payload = state.get("pending_call")
        return_target = (
            "domain_orchestrate"
            if state.get("pending_call_layer") == "domain"
            else "judge_orchestrate"
        )
        if not isinstance(pending_call_payload, Mapping):
            return Command(goto=return_target)

        planned_call = self._deserialize_planned_call(pending_call_payload)
        agent_id = canonical_agent_id(planned_call.agent_id)
        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        knowledge_base = LocalKnowledgeBase.from_state_payload(state["knowledge_base"])
        responses = list(state["responses"])
        executed_calls = list(state["executed_calls"])
        called_agents = list(state["called_agents"])
        layer = str(state.get("pending_call_layer") or "core")
        publish_debate_event(
            "agent_started",
            {
                "agent_id": agent_id,
                "layer": layer,
                "turn_number": state["turn_number"],
                "query": planned_call.query,
                "depth": planned_call.depth,
            },
        )

        try:
            if agent_id in {"disclosure", "news", "report"}:
                agent = self.coordinator._agent_by_id(agent_id)
                response = agent.run(
                    query=planned_call.query,
                    context=planned_call.context,
                    fallback=planned_call.fallback,
                    note=planned_call.note,
                    turn_number=state["turn_number"],
                    portfolio=portfolio,
                    knowledge_base=knowledge_base,
                    depth=planned_call.depth,
                )
            elif agent_id == "profit":
                response = self.coordinator.profit_agent.run(
                    query=planned_call.query,
                    turn_number=state["turn_number"],
                    portfolio=portfolio,
                    knowledge_base=knowledge_base,
                    rebalance_plan=dict(state.get("candidate_plan", {})),
                )
            elif agent_id == "cost":
                response = self.coordinator.cost_agent.run(
                    query=planned_call.query,
                    turn_number=state["turn_number"],
                    portfolio=portfolio,
                    rebalance_plan=dict(state.get("candidate_plan", {})),
                )
            elif agent_id in getattr(self.coordinator, "domain_agents", {}):
                from libra_agent.domain_agents._adapter import (
                    _run_async,
                    domain_verdict_to_agent_response,
                    portfolio_snapshot_to_domain_context,
                )

                candidate_plan = dict(state.get("candidate_plan", {}))
                context_parts = [str(planned_call.context or f"원 사용자 요청: {state['query']}")]
                if candidate_plan:
                    context_parts.append(f"후보 리밸런싱 초안: {candidate_plan}")
                for prior in [AgentResponse.from_dict(item) for item in responses[-6:]]:
                    context_parts.append(
                        f"{prior.agent_id}: {prior.reasoning_for_judge_agent or prior.query_understood}"
                    )
                proposed_trades = candidate_plan_to_proposed_trades(candidate_plan)
                ctx = portfolio_snapshot_to_domain_context(
                    portfolio,
                    user_id="libra",
                    proposed_trades=proposed_trades,
                    market_context_str="\n".join(context_parts),
                )
                ctx.router = getattr(self.coordinator, "domain_router", None)
                domain_agent = self.coordinator.domain_agents[agent_id]
                verdict = _run_async(domain_agent.deliberate(ctx))
                response = domain_verdict_to_agent_response(
                    verdict,
                    agent_id=agent_id,
                    turn_number=state["turn_number"],
                    query=planned_call.query,
                )
            else:
                return Command(
                    update={"pending_call": None, "pending_call_layer": None}, goto=return_target
                )
        except Exception as exc:
            publish_debate_event(
                "agent_failed",
                {
                    "agent_id": agent_id,
                    "layer": layer,
                    "turn_number": state["turn_number"],
                    "error": str(exc)[:500],
                },
            )
            raise
        response = sanitize_agent_response_payload(
            response.to_dict(),
            agent_id=agent_id,
            portfolio=portfolio,
            query=planned_call.query,
            turn_number=state["turn_number"],
            opinion_id=response.opinion_id,
            depth=planned_call.depth,
        )
        publish_debate_event(
            "agent_completed",
            _agent_response_event(response, layer=layer, turn_number=state["turn_number"]),
        )

        responses.append(response.to_dict())
        executed_calls.append(self._serialize_planned_call(planned_call))
        called_agents.append(agent_id)
        return Command(
            update={
                "knowledge_base": knowledge_base.to_state_payload(),
                "responses": responses,
                "executed_calls": executed_calls,
                "called_agents": called_agents,
                "pending_call": None,
                "pending_call_layer": None,
                "turn_number": state["turn_number"] + 1,
            },
            goto=return_target,
        )

    def _domain_consensus(self, state: LibraGraphState) -> dict[str, Any]:
        from libra_agent.domain_agents._consensus import compute_domain_consensus

        responses = [AgentResponse.from_dict(item) for item in state["responses"]]
        consensus = compute_domain_consensus(responses)
        publish_debate_event("consensus_updated", {"domain_consensus": consensus})
        return {"domain_consensus": consensus}

    def _final_judge(self, state: LibraGraphState) -> dict[str, Any]:
        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        knowledge_base = LocalKnowledgeBase.from_state_payload(state["knowledge_base"])
        trigger_event = self._trigger_event_from_state(state)
        run_state = self._deserialize_run_state(state["run_state"])
        responses = [AgentResponse.from_dict(item) for item in state["responses"]]
        executed_calls = [self._deserialize_planned_call(item) for item in state["executed_calls"]]
        judge_actions = list(state.get("judge_actions", []))
        called_agents = list(state["called_agents"])
        skipped_agents = list(state["skipped_agents"])
        skip_rationale = dict(state["skip_rationale"])
        candidate_plan = dict(state["candidate_plan"])
        needs_trade_eval = bool(state["needs_trade_eval"])
        inferred_skip_rationale = self.coordinator._infer_skip_rationale(
            query=state["query"],
            trigger=state["trigger"],
            trigger_event=trigger_event,
            called_agents=called_agents,
            responses=responses,
            candidate_plan=candidate_plan,
            depth=state["depth"],
        )
        for agent_id, reason in inferred_skip_rationale.items():
            skip_rationale.setdefault(agent_id, reason)
        skipped_agents = sorted({*skipped_agents, *inferred_skip_rationale})

        final_payload = self.coordinator._judge_phase(
            query=state["query"],
            portfolio=portfolio,
            responses=responses,
            stage="final",
            candidate_plan=candidate_plan,
            drift_report=state.get("drift_report"),
        )
        final_payload = sanitize_judge_payload(final_payload, portfolio=portfolio, stage="final")
        consensus_score, divergence_score = self.coordinator._consensus_metrics(responses)
        decision = JudgeDecision.from_dict(
            {
                **final_payload,
                "called_agents": called_agents,
                "skipped_agents": skipped_agents,
                "skip_rationale": skip_rationale,
                "candidate_rebalance_plan": candidate_plan,
                "needs_trade_evaluation": needs_trade_eval,
                "consensus_score": consensus_score,
                "divergence_score": divergence_score,
                "trigger": state["trigger"],
                "trigger_event": trigger_event.to_dict() if trigger_event else None,
                "deadline_at": run_state.deadline_at.isoformat(timespec="seconds")
                if run_state.deadline_at
                else None,
                "elapsed_seconds": self.coordinator._elapsed_seconds(run_state),
                "notification_log": [item.to_dict() for item in run_state.notification_log],
            }
        )
        decision.called_agents = called_agents
        decision.skipped_agents = skipped_agents
        decision.skip_rationale = skip_rationale
        decision.candidate_rebalance_plan = candidate_plan
        decision.consensus_score = consensus_score
        decision.divergence_score = divergence_score
        decision.needs_trade_evaluation = needs_trade_eval
        decision.trigger = state["trigger"]
        decision.trigger_event = trigger_event
        decision.deadline_at = (
            run_state.deadline_at.isoformat(timespec="seconds") if run_state.deadline_at else None
        )
        decision.elapsed_seconds = self.coordinator._elapsed_seconds(run_state)
        called_agent_set = {canonical_agent_id(agent_id) for agent_id in called_agents}
        query_lower = state["query"].casefold()
        rebalance_intent = any(
            token in query_lower
            for token in ("리밸런싱", "리밸런스", "rebalance", "초안", "비중", "매수", "매도")
        )
        if (
            state["trigger"] == "pull"
            and decision.decision == DecisionType.DEFER
            and not candidate_plan
            and {"disclosure", "news"}.issubset(called_agent_set)
        ):
            response_map = {canonical_agent_id(item.agent_id): item for item in responses}
            if self.coordinator._should_finalize_after_basic_scan(
                query=state["query"],
                depth=state["depth"],
                disclosure_response=response_map.get("disclosure"),
                news_response=response_map.get("news"),
                candidate_plan=candidate_plan,
            ):
                decision.decision = DecisionType.HOLD
                decision.urgency = Urgency.DEFER
                decision.summary = (
                    "신규 공시나 유의미한 뉴스 신호가 없어 현재 포트폴리오를 유지합니다."
                )
                decision.reasoning = (
                    "공시와 뉴스 점검에서 리밸런싱을 정당화할 신호가 없었고, "
                    "리포트·수익·비용 에이전트 호출도 필요하지 않았습니다."
                )
                decision.feedback_checkpoint = None
                decision.follow_up_at = self.coordinator._default_follow_up_at(
                    query=state["query"],
                    responses=responses,
                )
                decision.options = []
                decision.auto_safeguards = {}
        if decision.decision == DecisionType.REBALANCE:
            guardrail_reason: str | None = None
            if not candidate_plan:
                guardrail_reason = (
                    "판단 에이전트가 구체적인 리밸런싱 초안 없이 REBALANCE를 제안했습니다."
                )
                decision.summary = "지금 단계 근거만으로는 실행 가능한 리밸런싱 초안을 만들 수 없어 결정을 보류합니다."
            elif not {"profit", "cost"}.issubset(called_agent_set):
                guardrail_reason = (
                    "수익·비용 검토가 끝나기 전에 판단 에이전트가 REBALANCE를 제안했습니다."
                )
                if rebalance_intent:
                    decision.summary = "후보 리밸런싱 초안은 있지만 수익·비용 검토가 끝나지 않아 지금은 실행 결정을 보류합니다."
                else:
                    decision.summary = (
                        "실행 초안은 잡혔지만 거래 전 검토가 덜 끝나서 지금은 보류합니다."
                    )
            if guardrail_reason:
                decision.decision = DecisionType.DEFER
                decision.urgency = Urgency.DEFER
                decision.reasoning = guardrail_reason
                decision.feedback_checkpoint = None
                decision.follow_up_at = self.coordinator._default_follow_up_at(
                    query=state["query"],
                    responses=responses,
                )
                decision.options = []
                decision.auto_safeguards = {}
        empty_portfolio_no_trade = not portfolio.holdings and not candidate_plan
        if empty_portfolio_no_trade:
            decision.decision = DecisionType.DEFER
            decision.urgency = Urgency.DEFER
            decision.summary = (
                "포트폴리오가 비어 있고 후보 리밸런싱 초안도 없어 지금 실행할 매수·매도 조정은 없습니다."
            )
            decision.reasoning = (
                "공시·뉴스의 로컬 근거 0건은 시장이 조용하다는 뜻이 아니라 현재 주입된 근거가 "
                "부족하다는 뜻입니다. 보유 종목과 주문 후보가 없으므로 유동성·기술·체결·세금 "
                "검토 대상도 없고, 투자 판단을 내려면 시장/종목 데이터 수집 후 초기 분산 "
                "포트폴리오 후보를 먼저 생성해야 합니다."
            )
            decision.candidate_rebalance_plan = {}
            decision.needs_trade_evaluation = False
            decision.follow_up_at = None
            decision.feedback_checkpoint = None
            decision.options = []
            decision.user_notification = UserNotification(
                level="info",
                body=decision.summary,
                action_required=False,
                kind="empty_portfolio_needs_candidate",
                estimated_followup=None,
                sent_at=self.coordinator._notification_timestamp(),
            )
        self.coordinator._apply_push_guardrails(
            decision=decision,
            portfolio=portfolio,
            run_state=run_state,
        )
        from libra_agent.domain_agents._consensus import (
            DOMAIN_AGENT_IDS,
            apply_compliance_veto,
            compute_domain_consensus,
        )

        domain_responses = [
            response for response in responses if response.agent_id in DOMAIN_AGENT_IDS
        ]
        if domain_responses:
            domain_consensus = dict(
                state.get("domain_consensus") or compute_domain_consensus(domain_responses)
            )
            decision.auto_safeguards = {
                **dict(decision.auto_safeguards),
                "domain_consensus": domain_consensus,
            }
            decision = apply_compliance_veto(decision, domain_responses)
        if (
            not empty_portfolio_no_trade
            and not decision.follow_up_at
            and decision.decision.value == "DEFER"
        ):
            decision.follow_up_at = self.coordinator._default_follow_up_at(
                query=state["query"], responses=responses
            )
        if not decision.feedback_checkpoint and decision.decision.value == "REBALANCE":
            decision.feedback_checkpoint = self.coordinator._default_feedback_checkpoint()
        decision.follow_up_at = self.coordinator._sanitize_future_timestamp(
            decision.follow_up_at,
            default=None
            if empty_portfolio_no_trade
            else (
                self.coordinator._default_follow_up_at(query=state["query"], responses=responses)
                if decision.decision.value == "DEFER"
                else None
            ),
        )
        decision.feedback_checkpoint = self.coordinator._sanitize_future_timestamp(
            decision.feedback_checkpoint,
            default=self.coordinator._default_feedback_checkpoint()
            if decision.decision.value == "REBALANCE"
            else None,
        )
        if (
            decision.decision == DecisionType.DEFER
            and not candidate_plan
            and decision.reasoning.startswith("판단 에이전트가 구체적인 리밸런싱 초안 없이")
            and "report" in skip_rationale
        ):
            decision.reasoning = "이번 판단에는 공시와 뉴스 점검만으로 충분했고, 평가할 구체적인 리밸런싱 초안은 없었습니다."
        if not decision.reasoning.strip():
            decision.reasoning = (
                "판단 에이전트가 하위 에이전트 방향성, 신호 충돌, 거래 검토 결과를 종합했습니다."
            )
        if decision.decision != DecisionType.USER_DECISION_REQUIRED:
            decision.options = []
        decision.decision_trace = self.coordinator._decision_trace(
            query=state["query"],
            executed_calls=executed_calls,
            judge_actions=judge_actions,
            responses=responses,
            decision=decision,
        )
        if decision.user_notification is None or not decision.user_notification.body.strip():
            decision.user_notification = UserNotification(
                level=self.coordinator._notification_level(decision.decision, decision.urgency),
                body=decision.summary,
                action_required=decision.decision.value == "USER_DECISION_REQUIRED",
                kind="final_decision",
                estimated_followup=decision.follow_up_at,
                sent_at=self.coordinator._notification_timestamp(),
            )
        else:
            decision.user_notification = UserNotification(
                level=self.coordinator._notification_level(decision.decision, decision.urgency),
                body=decision.summary,
                action_required=decision.decision.value == "USER_DECISION_REQUIRED",
                kind=decision.user_notification.kind or "final_decision",
                estimated_followup=decision.user_notification.estimated_followup
                or decision.follow_up_at,
                sent_at=decision.user_notification.sent_at
                or self.coordinator._notification_timestamp(),
            )
        decision.notification_log = list(run_state.notification_log)
        if not any(
            item.kind == decision.user_notification.kind
            and item.body == decision.user_notification.body
            for item in decision.notification_log
        ):
            decision.notification_log.append(decision.user_notification)
        publish_debate_event(
            "final_decision_draft",
            {
                "decision": decision.decision.value,
                "summary": decision.summary,
                "confidence": decision.confidence,
                "urgency": decision.urgency.value,
                "called_agents": list(called_agents),
                "skipped_agents": list(skipped_agents),
                "requires_approval": decision.user_notification.action_required
                if decision.user_notification
                else decision.decision.value == "USER_DECISION_REQUIRED",
            },
        )

        return {
            "final_result": {
                "model": self.coordinator.client.model,
                "query": state["query"],
                "portfolio": portfolio.to_dict(),
                "agent_responses": [response.to_dict() for response in responses],
                "decision": decision.to_dict(),
                "knowledge_sources": dict(knowledge_base.source_paths),
                "direct_indexing": {
                    "portfolio_definition": state.get("portfolio_definition"),
                    "drift_report": state.get("drift_report"),
                    "candidate_rebalance_plan": candidate_plan,
                },
            }
        }

    def _route_human_review(
        self, state: LibraGraphState
    ) -> Command[Literal["human_review", "__end__"]]:
        if not state.get("enable_human_interrupts"):
            return Command(goto=END)
        final_result = state.get("final_result", {})
        decision_payload = (
            final_result.get("decision", {}) if isinstance(final_result, Mapping) else {}
        )
        if not isinstance(decision_payload, Mapping):
            return Command(goto=END)
        user_notification = (
            decision_payload.get("user_notification", {})
            if isinstance(decision_payload.get("user_notification"), Mapping)
            else {}
        )
        requires_human = bool(user_notification.get("action_required")) or (
            decision_payload.get("decision") == "USER_DECISION_REQUIRED"
        )
        return Command(goto="human_review" if requires_human else END)

    def _human_review(self, state: LibraGraphState) -> dict[str, Any]:
        final_result = dict(state.get("final_result", {}))
        decision_payload = (
            dict(final_result.get("decision", {}))
            if isinstance(final_result.get("decision"), Mapping)
            else {}
        )
        user_notification = (
            dict(decision_payload.get("user_notification", {}))
            if isinstance(decision_payload.get("user_notification"), Mapping)
            else {}
        )
        human_response = interrupt(
            {
                "kind": "human_review",
                "thread_id": state["thread_id"],
                "query": state["query"],
                "decision": decision_payload.get("decision"),
                "summary": decision_payload.get("summary"),
                "urgency": decision_payload.get("urgency"),
                "prompt": user_notification.get("body") or decision_payload.get("summary"),
            }
        )
        runtime = (
            dict(final_result.get("runtime", {}))
            if isinstance(final_result.get("runtime"), Mapping)
            else {}
        )
        runtime["human_response"] = human_response
        final_result["runtime"] = runtime
        return {"final_result": final_result}

    def _deadline_terminal(self, state: LibraGraphState) -> dict[str, Any]:
        portfolio = PortfolioSnapshot.from_dict(state["portfolio"])
        knowledge_base = LocalKnowledgeBase.from_state_payload(state["knowledge_base"])
        run_state = self._deserialize_run_state(state["run_state"])
        responses = [AgentResponse.from_dict(item) for item in state["responses"]]
        executed_calls = [self._deserialize_planned_call(item) for item in state["executed_calls"]]
        final_result = self.coordinator._deadline_result(
            query=state["query"],
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            called_agents=list(state["called_agents"]),
            skipped_agents=list(state["skipped_agents"]),
            skip_rationale=dict(state["skip_rationale"]),
            executed_calls=executed_calls,
            judge_actions=list(state.get("judge_actions", [])),
            responses=responses,
            run_state=run_state,
            candidate_plan=dict(state.get("candidate_plan", {})),
        )
        return {"final_result": final_result}

    def _serialize_run_state(self, run_state: RunState) -> dict[str, Any]:
        return {
            "trigger": run_state.trigger,
            "trigger_event": run_state.trigger_event.to_dict() if run_state.trigger_event else None,
            "started_at": run_state.started_at.isoformat(),
            "deadline_at": run_state.deadline_at.isoformat() if run_state.deadline_at else None,
            "notification_log": [item.to_dict() for item in run_state.notification_log],
        }

    def _deserialize_run_state(self, payload: Mapping[str, Any]) -> RunState:
        started_at = self.coordinator._coerce_datetime_or_now(payload.get("started_at"))
        deadline_at = self.coordinator._coerce_datetime(payload.get("deadline_at"))
        trigger_payload = payload.get("trigger_event")
        trigger_event = (
            TriggerEvent.from_dict(trigger_payload)
            if isinstance(trigger_payload, Mapping)
            else None
        )
        notifications = [
            UserNotification.from_dict(item)
            for item in payload.get("notification_log", [])
            if isinstance(item, Mapping)
        ]
        return RunState(
            trigger=str(payload.get("trigger", "pull")),
            trigger_event=trigger_event,
            started_at=started_at,
            deadline_at=deadline_at,
            notification_log=notifications,
        )

    def _serialize_planned_call(self, call: PlannedAgentCall) -> dict[str, Any]:
        return {
            "agent_id": canonical_agent_id(call.agent_id),
            "query": call.query,
            "context": call.context,
            "depth": call.depth,
            "fallback": call.fallback,
            "note": call.note,
        }

    def _deserialize_planned_call(self, payload: Mapping[str, Any]) -> PlannedAgentCall:
        return PlannedAgentCall(
            agent_id=canonical_agent_id(str(payload.get("agent_id", ""))),
            query=str(payload.get("query", "")),
            context=str(payload.get("context", "")),
            depth=str(payload.get("depth", "medium")),
            fallback=str(payload.get("fallback")) if payload.get("fallback") is not None else None,
            note=str(payload.get("note")) if payload.get("note") is not None else None,
        )

    def _serialize_judge_action(
        self,
        action: Mapping[str, Any],
        *,
        turn_number: int,
        called_agents: list[str],
        response_count: int,
    ) -> dict[str, Any]:
        agent_id = str(action.get("agent_id") or "").strip()
        candidate_plan = action.get("candidate_rebalance_plan", {})
        return {
            "turn_number": turn_number,
            "action": str(action.get("action") or "").strip().upper(),
            "agent_id": canonical_agent_id(agent_id) if agent_id else None,
            "layer": str(action.get("layer") or "core").strip(),
            "reason": str(action.get("reason") or "").strip(),
            "query": str(action.get("query") or "").strip(),
            "context": str(action.get("context") or "").strip(),
            "depth": str(action.get("depth") or "").strip(),
            "candidate_rebalance_plan": dict(candidate_plan)
            if isinstance(candidate_plan, Mapping)
            else {},
            "called_agents_before": list(called_agents),
            "response_count_before": response_count,
        }

    def _trigger_event_from_state(self, state: LibraGraphState) -> TriggerEvent | None:
        payload = state.get("trigger_event")
        if isinstance(payload, Mapping):
            return TriggerEvent.from_dict(payload)
        return None

    def _portfolio_definition_from_state(
        self, state: LibraGraphState
    ) -> PortfolioDefinition | None:
        payload = state.get("portfolio_definition")
        if isinstance(payload, Mapping):
            return PortfolioDefinition.from_dict(payload)
        return None

    def _portfolio_with_definition_targets(
        self,
        portfolio: PortfolioSnapshot,
        definition: PortfolioDefinition | None,
    ) -> PortfolioSnapshot:
        if definition is None:
            return portfolio
        payload = portfolio.to_dict()
        holdings = list(payload.get("holdings", []))
        existing = {
            "".join(char for char in str(item.get("ticker", "")).upper() if char.isalnum())
            for item in holdings
            if isinstance(item, Mapping)
        }
        for target in definition.target_weights:
            if target.ticker in existing:
                continue
            holdings.append(
                {
                    "ticker": target.ticker,
                    "company_name": target.company_name,
                    "weight": 0.0,
                    "aliases": [target.company_name],
                }
            )
        payload["holdings"] = holdings
        return PortfolioSnapshot.from_dict(payload)

    def _default_thread_id(
        self,
        *,
        query: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        portfolio: PortfolioSnapshot,
    ) -> str:
        fingerprint = {
            "query": query,
            "trigger": trigger,
            "trigger_event": trigger_event.to_dict() if trigger_event else None,
            "portfolio": portfolio.to_dict(),
        }
        return f"libra_{stable_hash(fingerprint)[:20]}"

    def _serialize_interrupts(self, payload: Any) -> list[dict[str, Any]]:
        if not payload:
            return []
        serialized: list[dict[str, Any]] = []
        for item in payload:
            serialized.append(
                {
                    "id": getattr(item, "id", None),
                    "value": getattr(item, "value", None),
                }
            )
        return serialized


def _agent_response_event(
    response: AgentResponse,
    *,
    layer: str,
    turn_number: int,
) -> dict[str, Any]:
    return {
        "agent_id": response.agent_id,
        "layer": layer,
        "turn_number": turn_number,
        "verdict": response.verdict.value,
        "opinion": response.opinion,
        "direction": response.direction,
        "strength": response.strength,
        "confidence": response.confidence,
        "urgency": response.urgency.value,
        "risk_level": response.risk_level,
        "focus_tickers": list(response.focus_tickers),
        "reasoning": _limit_text(response.reasoning_for_judge_agent, 900),
        "limits_acknowledged": _limit_text(response.limits_acknowledged or "", 500) or None,
        "tools_called": [
            {
                "tool_name": item.tool_name,
                "purpose": item.purpose,
                "summary": _limit_text(item.summary, 300),
            }
            for item in response.tools_called[:6]
        ],
    }


def _limit_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
