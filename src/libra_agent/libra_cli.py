from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from .libra.config import add_backend_arguments
from .libra.llm_clients import open_chat_client_from_args
from .libra.portfolio_sources import KIS_DEFAULT_CONFIG_PATH, build_kis_portfolio_snapshot
from .libra_models import PortfolioSnapshot, TriggerEvent
from .libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from .libra_store import LibraDecisionStore


def _infer_batch_file(batch_dir: Path, stem: str) -> Path | None:
    for candidate in (batch_dir / f"{stem}.jsonl", batch_dir / f"{stem}.json"):
        if candidate.exists():
            return candidate
    return None


def _load_portfolio(path: str | Path) -> PortfolioSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio file must be a JSON object.")
    return PortfolioSnapshot.from_dict(payload)


def _build_trigger_event(args: argparse.Namespace) -> TriggerEvent | None:
    if args.trigger != "push":
        return None
    payload = {
        "trigger_type": "news_push",
        "headline": args.event_headline or args.query,
        "summary": args.event_summary,
        "ticker": args.event_ticker,
        "company_name": args.event_company,
        "source": args.event_source,
        "event_time": args.event_time,
        "cross_check_count": args.event_cross_check_count,
        "market_reaction": args.event_market_reaction,
        "severity": "watch",
    }
    return TriggerEvent.from_dict(payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LIBRA MVP with a local LLM backend")
    parser.add_argument("--query", help="User question or instruction for LIBRA")
    parser.add_argument("--portfolio", help="Path to a portfolio snapshot JSON file")
    parser.add_argument(
        "--portfolio-source",
        default="file",
        choices=("file", "kis"),
        help="Portfolio input source. Use 'kis' to bootstrap holdings directly from Korea Investment Open API.",
    )
    parser.add_argument(
        "--kis-config",
        default=str(KIS_DEFAULT_CONFIG_PATH),
        help="Path to kis_devlp.yaml used for KIS portfolio bootstrap",
    )
    parser.add_argument(
        "--kis-env",
        default="real",
        choices=("real", "demo"),
        help="KIS environment for portfolio bootstrap",
    )
    parser.add_argument("--kis-app-key", help="Override KIS app key for portfolio bootstrap")
    parser.add_argument("--kis-app-secret", help="Override KIS app secret for portfolio bootstrap")
    parser.add_argument("--kis-account-no", help="Override KIS account number (first 8 digits)")
    parser.add_argument("--kis-product-code", help="Override KIS product code (last 2 digits)")
    parser.add_argument("--kis-user-agent", help="Override KIS User-Agent header")
    parser.add_argument(
        "--batch-dir", help="Local knowledge directory containing events/normalized documents"
    )
    parser.add_argument("--events", help="Path to events.json or events.jsonl")
    parser.add_argument(
        "--normalized-documents",
        help="Path to normalized_documents.json or normalized_documents.jsonl",
    )
    parser.add_argument("--enriched-documents", help="Fallback path to enriched_documents.json")
    parser.add_argument(
        "--trigger", default="pull", choices=("pull", "push"), help="Invocation mode for the Judge"
    )
    parser.add_argument("--event-headline", help="Headline for a push-triggered event")
    parser.add_argument("--event-summary", help="Summary for a push-triggered event")
    parser.add_argument("--event-ticker", help="Primary ticker for a push-triggered event")
    parser.add_argument("--event-company", help="Primary company name for a push-triggered event")
    parser.add_argument(
        "--event-source", default="news_push", help="Source label for a push-triggered event"
    )
    parser.add_argument("--event-time", help="Timestamp for a push-triggered event")
    parser.add_argument(
        "--event-cross-check-count",
        type=int,
        default=1,
        help="Cross-check count for a push-triggered event",
    )
    parser.add_argument(
        "--event-market-reaction", help="Observed market reaction for a push-triggered event"
    )
    parser.add_argument(
        "--deadline-seconds", type=int, help="Optional self-imposed analysis deadline in seconds"
    )
    parser.add_argument(
        "--state-dir",
        default=str(Path("outputs") / "libra_state"),
        help="Directory used to persist LIBRA run outputs, follow-ups, and feedback checkpoints",
    )
    parser.add_argument(
        "--thread-id",
        help="Optional thread id used to correlate a committee run.",
    )
    parser.add_argument(
        "--enable-human-interrupts",
        action="store_true",
        help="Pause on USER_DECISION_REQUIRED outcomes (handled by the production graph path).",
    )
    add_backend_arguments(
        parser, default_backend="ollama", backend_help="Local LLM backend/provider"
    )
    parser.add_argument("--depth", default="medium", choices=("shallow", "medium", "deep"))
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    batch_dir = Path(args.batch_dir) if args.batch_dir else None
    events_path = Path(args.events) if args.events else None
    normalized_path = Path(args.normalized_documents) if args.normalized_documents else None
    enriched_path = Path(args.enriched_documents) if args.enriched_documents else None

    if batch_dir is not None:
        events_path = events_path or _infer_batch_file(batch_dir, "events")
        normalized_path = normalized_path or _infer_batch_file(batch_dir, "normalized_documents")
        enriched_path = enriched_path or _infer_batch_file(batch_dir, "enriched_documents")

    if events_path is not None and not events_path.exists():
        raise RuntimeError(f"Events file does not exist: {events_path}")
    if normalized_path is not None and not normalized_path.exists():
        raise RuntimeError(f"Normalized documents file does not exist: {normalized_path}")
    if enriched_path is not None and not enriched_path.exists():
        raise RuntimeError(f"Enriched documents file does not exist: {enriched_path}")
    if events_path is None and normalized_path is None and enriched_path is None:
        raise RuntimeError(
            "No local knowledge files were found. Pass --batch-dir or explicit file paths."
        )
    return events_path, normalized_path, enriched_path


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.query:
        parser.error("--query is required.")
    if args.portfolio_source == "file" and not args.portfolio:
        parser.error("--portfolio is required when --portfolio-source is 'file'.")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with ExitStack() as stack:
        client = open_chat_client_from_args(args, stack=stack)
        client.ensure_available()
        orchestrator = JudgeOrchestrator(
            client=client,
            checkpoint_path=Path(args.state_dir) / "langgraph.sqlite",
        )
        events_path, normalized_path, enriched_path = resolve_inputs(args)
        portfolio = (
            _load_portfolio(args.portfolio)
            if args.portfolio_source == "file"
            else build_kis_portfolio_snapshot(args)
        )
        trigger_event = _build_trigger_event(args)
        knowledge_base = LocalKnowledgeBase.from_files(
            events_path=events_path,
            normalized_documents_path=normalized_path,
            enriched_documents_path=enriched_path,
        )
        result = orchestrator.run_v1_committee(
            query=args.query,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=args.depth,
            trigger=args.trigger,
            trigger_event=trigger_event,
            deadline_seconds=args.deadline_seconds,
            thread_id=args.thread_id,
            enable_human_interrupts=args.enable_human_interrupts,
        )
    store = LibraDecisionStore(args.state_dir)
    runtime = result.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("interrupted"):
        result["state_record"] = {
            "run_path": None,
            "follow_up_queue": None,
            "feedback_queue": None,
        }
    else:
        result["state_record"] = store.record_result(result)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
