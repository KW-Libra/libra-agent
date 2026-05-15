from __future__ import annotations

import argparse
import ast
import os
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .libra.agents.evaluation_agent import EvaluationAgent
from .libra.committee import CommitteeRuntime
from .libra.direct_indexing import PortfolioDefinition
from .libra.llm_clients import open_chat_client_from_env
from .libra.schemas import IPSConfig, KYCProfile, MarketSnapshot
from .libra_models import AgentResponse, PortfolioSnapshot, TriggerEvent
from .libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from .libra_store import LibraDecisionStore

DEFAULT_STATE_DIR = Path("outputs") / "libra_agent_api"

app = FastAPI(title="LIBRA Agent API", version="0.1.0")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LIBRA Agent HTTP API")
    parser.add_argument(
        "--host", default=os.getenv("LIBRA_AGENT_HOST", "0.0.0.0"), help="API bind host"
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("LIBRA_AGENT_PORT", "8010")), help="API bind port"
    )
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


def _safe_exception_detail(exc: BaseException) -> str:
    """Return a short exception chain for local diagnostics without leaking keys."""
    secrets = [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("ANTHROPIC_API_KEY", ""),
    ]
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(parts) < 4:
        text = f"{type(current).__name__}: {current}"
        for secret in secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        parts.append(text)
        current = current.__cause__
    return " <- ".join(parts)


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
        source_paths["ingest_refresh_enabled"] = str(
            _as_bool(payload.get("allow_ingest_refresh"))
        ).lower()
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

    sources = (
        _optional_mapping(payload.get("knowledge_sources"), field_name="knowledge_sources") or {}
    )
    events_path = _existing_path(
        sources.get("events") or payload.get("events"), field_name="events"
    )
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


def _literal_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _structured_preferences(portfolio: PortfolioSnapshot) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in portfolio.user_preferences:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key:
            result[key] = _literal_value(raw_value)
    return result


def _percent_from_preference(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def _ips_from_payload(portfolio: PortfolioSnapshot, payload: Mapping[str, Any]) -> IPSConfig:
    preferences = _structured_preferences(portfolio)
    raw_governance = (
        _optional_mapping(payload.get("governance_v1"), field_name="governance_v1") or {}
    )
    raw_ips = _optional_mapping(raw_governance.get("ips"), field_name="governance_v1.ips") or {}
    excluded_sectors = (
        raw_ips.get("excluded_sectors", preferences.get("excluded_sectors", [])) or []
    )
    excluded_tickers = (
        raw_ips.get("excluded_tickers", preferences.get("excluded_tickers", [])) or []
    )
    if isinstance(excluded_sectors, str):
        excluded_sectors = [excluded_sectors]
    if isinstance(excluded_tickers, str):
        excluded_tickers = [excluded_tickers]

    sector_limit = raw_ips.get("sector_limit_pct")
    if sector_limit is None:
        max_sector_weight = preferences.get("max_sector_weight")
        if isinstance(max_sector_weight, Mapping) and max_sector_weight:
            sector_limit = max(float(value) for value in max_sector_weight.values()) * 100.0

    return IPSConfig(
        single_ticker_limit_pct=_percent_from_preference(
            raw_ips.get("single_ticker_limit_pct", preferences.get("max_single_weight")),
            default=25.0,
        ),
        sector_limit_pct=_percent_from_preference(sector_limit, default=40.0),
        annual_volatility_limit=_percent_from_preference(
            raw_ips.get("annual_volatility_limit"), default=20.0
        )
        / 100.0,
        asset_class_target=dict(
            raw_ips.get("asset_class_target") or {"EQUITY": 60.0, "BOND": 35.0, "ALT": 5.0}
        ),
        asset_class_band_pct=_percent_from_preference(
            raw_ips.get("asset_class_band_pct"), default=10.0
        ),
        min_cash_pct=_percent_from_preference(
            raw_ips.get("min_cash_pct", preferences.get("cash_min_weight")), default=5.0
        ),
        max_market_impact_pct_of_adv=_percent_from_preference(
            raw_ips.get("max_market_impact_pct_of_adv"), default=5.0
        ),
        excluded_tickers=[str(item).upper() for item in excluded_tickers],
        excluded_sectors=[str(item).upper() for item in excluded_sectors],
        esg_min_score=(
            float(raw_ips.get("esg_min_score", preferences.get("esg_min_score")))
            if raw_ips.get("esg_min_score", preferences.get("esg_min_score")) is not None
            else None
        ),
    )


def _kyc_from_payload(payload: Mapping[str, Any]) -> KYCProfile:
    raw_governance = (
        _optional_mapping(payload.get("governance_v1"), field_name="governance_v1") or {}
    )
    raw_kyc = _optional_mapping(raw_governance.get("kyc"), field_name="governance_v1.kyc") or {}
    risk = str(raw_kyc.get("risk_tolerance") or "MODERATE").strip().upper()
    if risk not in {"CONSERVATIVE", "MODERATE", "AGGRESSIVE"}:
        risk = "MODERATE"
    return KYCProfile(
        risk_tolerance=risk,  # type: ignore[arg-type]
        investment_horizon_years=int(raw_kyc.get("investment_horizon_years") or 15),
        max_drawdown_tolerance_pct=float(raw_kyc.get("max_drawdown_tolerance_pct") or 15.0),
    )


def _market_snapshot_from_portfolio(portfolio: PortfolioSnapshot) -> MarketSnapshot:
    return MarketSnapshot(
        prices={
            holding.ticker: float(holding.last_price)
            for holding in portfolio.holdings
            if holding.last_price is not None
        },
        sector_map={
            holding.ticker: str(holding.sector) for holding in portfolio.holdings if holding.sector
        },
        esg_score={
            holding.ticker: float(holding.esg_score)
            for holding in portfolio.holdings
            if holding.esg_score is not None
        },
    )


def _attach_governance_v1(
    result: dict[str, Any], *, payload: Mapping[str, Any], portfolio: PortfolioSnapshot
) -> dict[str, Any]:
    raw_governance = (
        _optional_mapping(payload.get("governance_v1"), field_name="governance_v1") or {}
    )
    if raw_governance.get("enabled") is False:
        return result
    response_payloads = result.get("agent_responses")
    if not isinstance(response_payloads, list):
        return result
    responses = [
        AgentResponse.from_dict(item) for item in response_payloads if isinstance(item, Mapping)
    ]
    governance_result = CommitteeRuntime().run_from_agent_responses(
        portfolio=portfolio,
        responses=responses,
        ips=_ips_from_payload(portfolio, payload),
        kyc=_kyc_from_payload(payload),
        market_data=_market_snapshot_from_portfolio(portfolio),
    )
    result["governance_v1"] = governance_result.to_dict()
    return result


def _governance_v1_execution_mode(payload: Mapping[str, Any]) -> str:
    raw_governance = (
        _optional_mapping(payload.get("governance_v1"), field_name="governance_v1") or {}
    )
    raw_mode = raw_governance.get("execution_mode") or raw_governance.get("mode")
    env_mode = os.environ.get("LIBRA_GOVERNANCE_V1_EXECUTION_MODE")
    mode = str(raw_mode or env_mode or "attach").strip().lower()
    if mode in {"primary", "committee", "v1"}:
        return "primary"
    return "attach"


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
    responses = (
        decision_run_result.get("agent_responses")
        if isinstance(decision_run_result, Mapping)
        else payload.get("agent_responses")
    )
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
                scored.append(
                    float(response.get("direction", 0.0))
                    * float(response.get("strength", 0.0))
                    * float(response.get("confidence", 0.0))
                )
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
        lines.append(f"- [결정 {decision} / 평가 {verdict} / 실현 {realized:+.2f}%] {reflection}")
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
    trigger_event_payload = _optional_mapping(
        payload.get("trigger_event"), field_name="trigger_event"
    )
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
            if _governance_v1_execution_mode(payload) == "primary":
                result = orchestrator.run_v1_committee(
                    query=query,
                    portfolio=portfolio,
                    knowledge_base=knowledge_base,
                    portfolio_definition=portfolio_definition,
                    depth=str(payload.get("depth") or "medium"),
                    trigger=trigger,
                    trigger_event=trigger_event,
                    deadline_seconds=_as_optional_int(
                        payload.get("deadline_seconds"), field_name="deadline_seconds"
                    ),
                    thread_id=str(payload.get("thread_id")) if payload.get("thread_id") else None,
                    enable_human_interrupts=_as_bool(payload.get("enable_human_interrupts", False)),
                    ips=_ips_from_payload(portfolio, payload),
                    kyc=_kyc_from_payload(payload),
                    market_data=_market_snapshot_from_portfolio(portfolio),
                )
            else:
                result = orchestrator.run(
                    query=query,
                    portfolio=portfolio,
                    knowledge_base=knowledge_base,
                    portfolio_definition=portfolio_definition,
                    depth=str(payload.get("depth") or "medium"),
                    trigger=trigger,
                    trigger_event=trigger_event,
                    deadline_seconds=_as_optional_int(
                        payload.get("deadline_seconds"), field_name="deadline_seconds"
                    ),
                    thread_id=str(payload.get("thread_id")) if payload.get("thread_id") else None,
                    enable_human_interrupts=_as_bool(payload.get("enable_human_interrupts", False)),
                )
                result = _attach_governance_v1(result, payload=payload, portfolio=portfolio)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_exception_detail(exc)) from exc

    return _record_result(result, state_dir=state_dir)


@app.post("/v1/evaluations")
def create_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    decision_payload = _decision_payload_from_evaluation_request(payload)
    decision = str(decision_payload.get("decision") or "").strip().upper()
    if not decision:
        raise HTTPException(
            status_code=400, detail="decision or decision_run_result.decision.decision is required."
        )
    rebalance_plan = _as_plain_float_map(
        payload.get("rebalance_plan") or decision_payload.get("candidate_rebalance_plan")
    )
    realized_return_pct = _as_float(
        payload.get("realized_return_pct"), field_name="realized_return_pct"
    )
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
