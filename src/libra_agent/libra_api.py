from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException

from .libra.agents.evaluation_agent import EvaluationAgent
from .libra.direct_indexing import PortfolioDefinition
from .libra.llm_clients import open_chat_client_from_env
from .libra_models import PortfolioSnapshot, TriggerEvent
from .libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from .libra_store import LibraDecisionStore


DEFAULT_STATE_DIR = Path("outputs") / "libra_agent_api"

app = FastAPI(title="LIBRA Agent API", version="0.1.0")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LIBRA Agent HTTP API")
    parser.add_argument("--host", default=os.getenv("LIBRA_AGENT_HOST", "0.0.0.0"), help="API bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("LIBRA_AGENT_PORT", "8010")), help="API bind port")
    parser.add_argument(
        "--state-dir",
        default=os.getenv("LIBRA_AGENT_STATE_DIR", str(DEFAULT_STATE_DIR)),
        help="Directory for run outputs and LangGraph checkpoints",
    )
    parser.add_argument(
        "--provider",
        choices=("llama_cpp", "ollama", "anthropic", "gemini"),
        help="Override LIBRA_LLM_PROVIDER for this API process",
    )
    return parser


def _state_dir() -> Path:
    return Path(os.getenv("LIBRA_AGENT_STATE_DIR", str(DEFAULT_STATE_DIR)))


_EVALUATION_CLIENT: Any = None
_EVALUATION_CLIENT_RESOLVED: bool = False


def _evaluation_client() -> Any:
    """Return a cached ChatClient for reflection generation, or None if unavailable.

    The reflection step is best-effort — if the LLM provider is misconfigured we
    let evaluation fall back to bare metric output rather than failing the call.
    """
    global _EVALUATION_CLIENT, _EVALUATION_CLIENT_RESOLVED
    if _EVALUATION_CLIENT_RESOLVED:
        return _EVALUATION_CLIENT
    _EVALUATION_CLIENT_RESOLVED = True
    try:
        _EVALUATION_CLIENT = open_chat_client_from_env()
    except Exception:
        _EVALUATION_CLIENT = None
    return _EVALUATION_CLIENT


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object.")
    return value


def _optional_mapping(value: Any, *, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _as_mapping(value, field_name=field_name)


def _as_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc


def _as_float(value: Any, *, field_name: str, default: float | None = None) -> float:
    if value is None or value == "":
        if default is not None:
            return default
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a number.") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _existing_path(value: Any, *, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"{field_name} does not exist: {path}")
    return str(path)


def _ingest_refresh_source_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    refresh = _optional_mapping(payload.get("ingest_refresh"), field_name="ingest_refresh") or {}
    mapped_keys = {
        "enabled": "ingest_refresh_enabled",
        "mode": "ingest_refresh_mode",
        "root": "ingest_root",
        "out_dir": "ingest_out_dir",
        "timeout_seconds": "ingest_refresh_timeout_seconds",
        "live_date": "ingest_live_date",
        "rss_limit": "ingest_rss_limit",
        "dart_limit": "ingest_dart_limit",
        "report_limit": "ingest_report_limit",
        "report_pdf_pages": "ingest_report_pdf_pages",
        "report_min_body_chars": "ingest_report_min_body_chars",
        "skip_article_body": "ingest_skip_article_body",
    }
    source_paths: dict[str, str] = {}
    if "allow_ingest_refresh" in payload:
        source_paths["ingest_refresh_enabled"] = str(_as_bool(payload.get("allow_ingest_refresh"))).lower()
    for raw_key, source_key in mapped_keys.items():
        value = refresh.get(raw_key)
        if value is not None and value != "":
            source_paths[source_key] = str(value)
    return source_paths


def _build_knowledge_base(payload: Mapping[str, Any]) -> LocalKnowledgeBase:
    ingest_source_paths = _ingest_refresh_source_paths(payload)
    inline_knowledge = _optional_mapping(payload.get("knowledge_base"), field_name="knowledge_base")
    if inline_knowledge is not None:
        knowledge_base = LocalKnowledgeBase.from_state_payload(inline_knowledge)
        knowledge_base.source_paths.update(ingest_source_paths)
        return knowledge_base

    sources = _optional_mapping(payload.get("knowledge_sources"), field_name="knowledge_sources") or {}
    events_path = _existing_path(sources.get("events") or payload.get("events"), field_name="events")
    normalized_documents_path = _existing_path(
        sources.get("normalized_documents") or payload.get("normalized_documents"),
        field_name="normalized_documents",
    )
    enriched_documents_path = _existing_path(
        sources.get("enriched_documents") or payload.get("enriched_documents"),
        field_name="enriched_documents",
    )
    if not any((events_path, normalized_documents_path, enriched_documents_path)):
        raise HTTPException(
            status_code=400,
            detail="Pass knowledge_base or knowledge_sources with events/normalized_documents/enriched_documents.",
        )
    knowledge_base = LocalKnowledgeBase.from_files(
        events_path=events_path,
        normalized_documents_path=normalized_documents_path,
        enriched_documents_path=enriched_documents_path,
    )
    knowledge_base.source_paths.update(ingest_source_paths)
    return knowledge_base


def _record_result(result: dict[str, Any], *, state_dir: Path) -> dict[str, Any]:
    store = LibraDecisionStore(state_dir)
    runtime = result.get("runtime", {})
    if isinstance(runtime, Mapping) and runtime.get("interrupted"):
        result["state_record"] = {
            "run_path": None,
            "follow_up_queue": None,
            "feedback_queue": None,
        }
        return result
    result["state_record"] = store.record_result(result)
    return result


def _as_plain_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        try:
            result[str(raw_key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return result


def _portfolio_with_definition_targets(
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


def _decision_payload_from_evaluation_request(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    decision_run_result = payload.get("decision_run_result")
    if isinstance(decision_run_result, Mapping):
        decision = decision_run_result.get("decision")
        if isinstance(decision, Mapping):
            return decision
    decision = payload.get("decision")
    if isinstance(decision, Mapping):
        return decision
    return payload


def _agent_responses_from_evaluation_request(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    decision_run_result = payload.get("decision_run_result")
    responses = decision_run_result.get("agent_responses") if isinstance(decision_run_result, Mapping) else payload.get("agent_responses")
    if not isinstance(responses, list):
        return []
    return [item for item in responses if isinstance(item, Mapping)]


def _evaluation_signal_score(payload: Mapping[str, Any]) -> float:
    explicit = payload.get("signal_score")
    if explicit is not None:
        return _as_float(explicit, field_name="signal_score")
    responses = _agent_responses_from_evaluation_request(payload)
    scored = []
    for response in responses:
        try:
            scored.append(float(response.get("signal_score")))
        except (TypeError, ValueError):
            try:
                scored.append(float(response.get("direction", 0.0)) * float(response.get("strength", 0.0)) * float(response.get("confidence", 0.0)))
            except (TypeError, ValueError):
                continue
    if scored:
        return max(scored, key=lambda item: abs(item))
    decision_run_result = payload.get("decision_run_result")
    decision = _decision_payload_from_evaluation_request(payload)
    consensus = decision.get("consensus_score")
    if consensus is None and isinstance(decision_run_result, Mapping):
        consensus = decision_run_result.get("consensus_score")
    return _as_float(consensus, field_name="signal_score", default=0.0)


def _prepend_prior_reflections(query: str, raw: Any) -> str:
    """Inject the user's recent reflections into the Judge query, if any.

    The backend supplies ``prior_reflections`` as a list of objects coming from
    ``DecisionEvaluationEntity.metrics_payload``. We render them as a short
    Korean preamble so the Judge can read them as part of its initial context
    without changing the LangGraph state shape.
    """
    if not isinstance(raw, list) or not raw:
        return query
    lines: list[str] = []
    for item in raw[:5]:
        if not isinstance(item, Mapping):
            continue
        reflection = str(item.get("reflection") or "").strip()
        if not reflection:
            continue
        decision = str(item.get("decision") or "?").strip() or "?"
        verdict = str(item.get("verdict") or "?").strip() or "?"
        try:
            realized = float(item.get("realized_return_pct") or 0.0)
        except (TypeError, ValueError):
            realized = 0.0
        lines.append(
            f"- [결정 {decision} / 평가 {verdict} / 실현 {realized:+.2f}%] {reflection}"
        )
    if not lines:
        return query
    header = "[과거 결정 회고 — 같은 사용자, 최근 호출 결과]"
    footer = "[위 회고를 참고하여 이번 판단의 근거에 반영하십시오.]"
    return header + "\n" + "\n".join(lines) + "\n" + footer + "\n\n원 질의: " + query


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/judge-runs")
def create_judge_run(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required.")

    portfolio_definition_payload = _optional_mapping(
        payload.get("portfolio_definition"),
        field_name="portfolio_definition",
    )
    try:
        portfolio_definition = (
            PortfolioDefinition.from_dict(portfolio_definition_payload)
            if portfolio_definition_payload is not None
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    portfolio_payload = _as_mapping(payload.get("portfolio"), field_name="portfolio")
    portfolio = _portfolio_with_definition_targets(
        PortfolioSnapshot.from_dict(portfolio_payload),
        portfolio_definition,
    )
    if not portfolio.holdings:
        raise HTTPException(
            status_code=400,
            detail="portfolio.holdings or portfolio_definition.target_weights must contain at least one stock.",
        )

    query = _prepend_prior_reflections(query, payload.get("prior_reflections"))

    knowledge_base = _build_knowledge_base(payload)
    trigger = str(payload.get("trigger") or "pull")
    trigger_event_payload = _optional_mapping(payload.get("trigger_event"), field_name="trigger_event")
    trigger_event = TriggerEvent.from_dict(trigger_event_payload) if trigger_event_payload else None
    state_dir = _state_dir()

    try:
        with ExitStack() as stack:
            client = open_chat_client_from_env(stack=stack)
            client.ensure_available()
            orchestrator = JudgeOrchestrator(
                client=client,
                checkpoint_path=state_dir / "langgraph.sqlite",
            )
            result = orchestrator.run(
                query=query,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                portfolio_definition=portfolio_definition,
                depth=str(payload.get("depth") or "medium"),
                trigger=trigger,
                trigger_event=trigger_event,
                deadline_seconds=_as_optional_int(payload.get("deadline_seconds"), field_name="deadline_seconds"),
                thread_id=str(payload.get("thread_id")) if payload.get("thread_id") else None,
                enable_human_interrupts=_as_bool(payload.get("enable_human_interrupts", False)),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _record_result(result, state_dir=state_dir)


@app.post("/v1/evaluations")
def create_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    decision_payload = _decision_payload_from_evaluation_request(payload)
    decision = str(decision_payload.get("decision") or "").strip().upper()
    if not decision:
        raise HTTPException(status_code=400, detail="decision or decision_run_result.decision.decision is required.")
    rebalance_plan = _as_plain_float_map(
        payload.get("rebalance_plan") or decision_payload.get("candidate_rebalance_plan")
    )
    realized_return_pct = _as_float(payload.get("realized_return_pct"), field_name="realized_return_pct")
    cost_pct = _as_float(payload.get("cost_pct"), field_name="cost_pct", default=0.0)
    signal_score = _evaluation_signal_score(payload)
    return EvaluationAgent(client=_evaluation_client()).run(
        decision=decision,
        rebalance_plan=rebalance_plan,
        signal_score=signal_score,
        user_feedback=str(payload.get("user_feedback") or "").strip() or None,
        realized_return_pct=realized_return_pct,
        cost_pct=cost_pct,
        horizon=str(payload.get("horizon") or "1w"),
    )


def main() -> None:
    import uvicorn

    args = build_argument_parser().parse_args()
    os.environ["LIBRA_AGENT_STATE_DIR"] = str(args.state_dir)
    if args.provider:
        os.environ["LIBRA_LLM_PROVIDER"] = str(args.provider)
    uvicorn.run("libra_agent.libra_api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
