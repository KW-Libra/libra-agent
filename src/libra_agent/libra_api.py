from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException

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
        choices=("llama_cpp", "ollama", "anthropic"),
        help="Override LIBRA_LLM_PROVIDER for this API process",
    )
    return parser


def _state_dir() -> Path:
    return Path(os.getenv("LIBRA_AGENT_STATE_DIR", str(DEFAULT_STATE_DIR)))


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


def _build_knowledge_base(payload: Mapping[str, Any]) -> LocalKnowledgeBase:
    inline_knowledge = _optional_mapping(payload.get("knowledge_base"), field_name="knowledge_base")
    if inline_knowledge is not None:
        return LocalKnowledgeBase.from_state_payload(inline_knowledge)

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
    return LocalKnowledgeBase.from_files(
        events_path=events_path,
        normalized_documents_path=normalized_documents_path,
        enriched_documents_path=enriched_documents_path,
    )


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/judge-runs")
def create_judge_run(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required.")

    portfolio_payload = _as_mapping(payload.get("portfolio"), field_name="portfolio")
    portfolio = PortfolioSnapshot.from_dict(portfolio_payload)
    if not portfolio.holdings:
        raise HTTPException(status_code=400, detail="portfolio.holdings must contain at least one holding.")

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


def main() -> None:
    import uvicorn

    args = build_argument_parser().parse_args()
    os.environ["LIBRA_AGENT_STATE_DIR"] = str(args.state_dir)
    if args.provider:
        os.environ["LIBRA_LLM_PROVIDER"] = str(args.provider)
    uvicorn.run("libra_agent.libra_api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
