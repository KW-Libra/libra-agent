from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from libra_agent.ingest_bundle import knowledge_payload_from_ingest_bundle
from libra_agent.libra.config import add_backend_arguments
from libra_agent.libra.direct_indexing import (
    PortfolioDefinition,
    candidate_plan_from_drift,
    compute_drift,
)
from libra_agent.libra.llm_clients import open_chat_client_from_args
from libra_agent.libra.schemas import IPSConfig, KYCProfile, MarketSnapshot
from libra_agent.libra_models import PortfolioSnapshot
from libra_agent.libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from libra_agent.runtime.debate_events import debate_event_publisher


DEFAULT_QUERY = "현재 포트폴리오를 점검하고 유지/조정 필요성을 판단해줘."
DEFAULT_USER_PREFERENCES = (
    "백테스트 재생 기준",
    "무리한 회전율 회피",
    "리스크 우선",
    "같은 시점에 관측 가능한 근거만 사용",
    "cash_min_weight=0.0",
)
DEFAULT_COMPANY_NAMES = {
    "003490": "대한항공",
    "005380": "현대차",
    "005930": "삼성전자",
    "035420": "NAVER",
    "105560": "KB금융",
}
CORE_AGENT_IDS = ("disclosure", "news", "report", "profit", "cost")
DOMAIN_AGENT_IDS = (
    "risk",
    "tax",
    "macro",
    "sentiment",
    "execution",
    "esg",
    "liquidity",
    "technical",
)
TRACE_KEYS = (
    "actor",
    "phase",
    "model",
    "tool_name",
    "agent_id",
    "layer",
    "turn_number",
    "verdict",
    "opinion",
    "decision",
    "error",
    "reason",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"JSONL row is not an object: {path}")
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _load_dotenv(path: Path | None) -> None:
    if path is None:
        return
    if not path.exists():
        raise RuntimeError(f"Env file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _set_default_env(args: argparse.Namespace) -> None:
    _load_dotenv(Path(args.env_file) if args.env_file else None)
    os.environ.setdefault("LIBRA_LLM_PROVIDER", "anthropic")
    os.environ.setdefault("LIBRA_ANTHROPIC_MODEL", str(args.anthropic_model or "claude-sonnet-4-6"))
    os.environ.setdefault("LIBRA_DOMAIN_AGENTS_ENABLED", "true")
    os.environ.setdefault("LLM_ROUTING_POLICY", "claude")
    os.environ.setdefault("LIBRA_DISABLE_AGENT_FALLBACKS", "true")
    os.environ.setdefault("LIBRA_SENTIMENT_PHASE2_ENABLED", "false")
    os.environ.setdefault("LIBRA_LLM_TIMEOUT_SECONDS", "180")
    os.environ.setdefault("LIBRA_LLM_REQUEST_TIMEOUT_SECONDS", "180")
    if args.usage_log:
        os.environ["LIBRA_LLM_USAGE_LOG"] = str(Path(args.usage_log))


def _target_weights(fixture: dict[str, Any]) -> dict[str, float]:
    return {str(ticker): float(weight) for ticker, weight in fixture["target_weights"].items()}


def _initial_shares(fixture: dict[str, Any], first_price_row: dict[str, Any]) -> dict[str, float]:
    initial_value = float(fixture["initial_value_krw"])
    return {
        ticker: (initial_value * weight) / float(first_price_row[ticker])
        for ticker, weight in _target_weights(fixture).items()
    }


def _company_name(ticker: str) -> str:
    return DEFAULT_COMPANY_NAMES.get(ticker, ticker)


def _portfolio_payload(
    *,
    day: str,
    price_row: dict[str, Any],
    shares: dict[str, float],
    user_preferences: tuple[str, ...],
) -> dict[str, Any]:
    values = {ticker: float(shares[ticker]) * float(price_row[ticker]) for ticker in shares}
    total_value = sum(values.values())
    holdings = []
    for ticker, market_value in sorted(values.items()):
        price = float(price_row[ticker])
        holdings.append(
            {
                "ticker": ticker,
                "company_name": _company_name(ticker),
                "weight": market_value / total_value if total_value > 0 else 0.0,
                "aliases": [_company_name(ticker), ticker],
                "shares": shares[ticker],
                "last_price": price,
                "market_value_krw": market_value,
                "sector": "EQUITY",
            }
        )
    return {
        "generated_at": f"{day}T15:30:00+09:00",
        "total_value_krw": total_value,
        "cash_weight": 0.0,
        "holdings": holdings,
        "user_preferences": list(user_preferences),
    }


def _portfolio_definition(fixture: dict[str, Any]) -> PortfolioDefinition:
    targets = [
        {
            "ticker": ticker,
            "company_name": _company_name(ticker),
            "weight": weight,
            "market": "KR",
        }
        for ticker, weight in _target_weights(fixture).items()
    ]
    return PortfolioDefinition.from_dict(
        {
            "name": "LIBRA Backtest Target Portfolio",
            "description": "Point-in-time replay target weights for service-runtime backtest.",
            "risk_profile": "위험중립형",
            "drift_threshold": float(fixture.get("threshold_band", 0.05)),
            "target_weights": targets,
        }
    )


def _ips_from_args(args: argparse.Namespace) -> IPSConfig:
    return IPSConfig(
        single_ticker_limit_pct=float(args.single_ticker_limit_pct),
        sector_limit_pct=float(args.sector_limit_pct),
        annual_volatility_limit=float(args.annual_volatility_limit),
        asset_class_target={"EQUITY": 100.0},
        asset_class_band_pct=100.0,
        min_cash_pct=float(args.min_cash_pct),
        max_market_impact_pct_of_adv=float(args.max_market_impact_pct_of_adv),
        excluded_tickers=[],
        excluded_sectors=[],
        esg_min_score=None,
    )


def _kyc_from_args(args: argparse.Namespace) -> KYCProfile:
    return KYCProfile(
        risk_tolerance=str(args.risk_tolerance).upper(),
        investment_horizon_years=int(args.investment_horizon_years),
        max_drawdown_tolerance_pct=float(args.max_drawdown_tolerance_pct),
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
    )


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {ticker: max(0.0, float(weight)) for ticker, weight in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("target weights must sum to a positive value")
    return {ticker: value / total for ticker, value in clipped.items()}


def _apply_rebalance(
    *,
    shares: dict[str, float],
    price_row: dict[str, Any],
    base_weights: dict[str, float],
    candidate_plan: dict[str, Any],
    cost_rate: float,
) -> dict[str, float]:
    if not candidate_plan:
        return dict(shares)
    current_values = {ticker: float(shares[ticker]) * float(price_row[ticker]) for ticker in shares}
    total_value = sum(current_values.values())
    current_weights = {
        ticker: (current_values.get(ticker, 0.0) / total_value if total_value > 0 else 0.0)
        for ticker in base_weights
    }
    target = dict(current_weights)
    for ticker, delta in candidate_plan.items():
        if ticker in target:
            target[ticker] = target.get(ticker, 0.0) + float(delta)
    target = _normalize_weights(target)
    desired_values = {ticker: total_value * target[ticker] for ticker in target}
    turnover = sum(abs(desired_values[ticker] - current_values.get(ticker, 0.0)) for ticker in target)
    investable_value = max(0.0, total_value - turnover * cost_rate)
    return {
        ticker: (investable_value * target[ticker]) / float(price_row[ticker])
        for ticker in target
    }


def _bundle_rows(bundles_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(bundles_dir / "index.json")
    rows = payload.get("bundles", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid bundle index: {bundles_dir / 'index.json'}")
    return [row for row in rows if isinstance(row, dict)]


def _load_bundle(bundles_dir: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    file_name = str(row.get("file") or "")
    if not file_name:
        raise RuntimeError(f"Bundle row missing file: {row}")
    path = bundles_dir / file_name
    return path, _read_json(path)


def _decision_schedule(bundle_rows: list[dict[str, Any]], args: argparse.Namespace) -> set[str]:
    frequency = str(args.decision_frequency or "daily").strip().lower()
    if frequency == "daily":
        return {str(row.get("prices_until")) for row in bundle_rows}
    if frequency == "every-n-trading-days":
        interval = int(args.decision_interval)
        if interval < 1:
            raise RuntimeError("--decision-interval must be >= 1")
        return {
            str(row.get("prices_until"))
            for index, row in enumerate(bundle_rows)
            if index % interval == 0
        }
    if frequency == "weekly":
        selected: set[str] = set()
        seen_weeks: set[tuple[int, int]] = set()
        for row in bundle_rows:
            day = str(row.get("prices_until"))
            iso = datetime.fromisoformat(day).date().isocalendar()
            key = (iso.year, iso.week)
            if key in seen_weeks:
                continue
            seen_weeks.add(key)
            selected.add(day)
        return selected
    raise RuntimeError(f"Unsupported decision frequency: {args.decision_frequency}")


def _scheduled_skip_result(
    *,
    day: str,
    bundle: dict[str, Any],
    portfolio: PortfolioSnapshot,
    portfolio_definition: PortfolioDefinition,
    drift_report: dict[str, Any],
    candidate_plan: dict[str, float],
    schedule_reason: str,
) -> dict[str, Any]:
    summary = f"SCHEDULED_SKIP: {schedule_reason}; no LLM committee run for {day}."
    return {
        "model": "scheduled-skip/no-llm",
        "query": DEFAULT_QUERY,
        "portfolio": portfolio.to_dict(),
        "agent_responses": [],
        "decision": {
            "decision": "DEFER",
            "summary": summary,
            "confidence": 1.0,
            "urgency": "defer",
            "called_agents": [],
            "skipped_agents": [*CORE_AGENT_IDS, *DOMAIN_AGENT_IDS],
            "skip_rationale": {
                "schedule": "This trading day is outside the configured backtest decision cadence."
            },
            "candidate_rebalance_plan": {},
            "decision_trace": [],
            "reasoning": summary,
            "user_notification": {
                "level": "silent",
                "body": summary,
                "action_required": False,
                "kind": "scheduled_backtest_skip",
                "estimated_followup": None,
                "sent_at": f"{day}T15:30:00+09:00",
            },
            "follow_up_at": None,
            "feedback_checkpoint": None,
            "consensus_score": 0.0,
            "divergence_score": 0.0,
            "needs_trade_evaluation": bool(candidate_plan),
            "trigger": "schedule_skip",
            "trigger_event": None,
            "deadline_at": None,
            "elapsed_seconds": 0.0,
            "options": [],
            "auto_safeguards": {
                "scheduled_skip": True,
                "schedule_reason": schedule_reason,
                "direct_indexing_candidate_plan": dict(candidate_plan),
            },
            "notification_log": [],
        },
        "knowledge_sources": {
            "ingest_bundle": str(bundle.get("bundle_id") or bundle.get("as_of") or day)
        },
        "governance_v1": {
            "round1_opinions": [],
            "round2_opinions": [],
            "consensus_per_subject": {},
            "targets_to_recall": [],
            "mediator_decision": {
                "consensus_per_subject": {},
                "targets_to_recall": [],
                "skip_round_2": True,
                "rationale": "Scheduled skip; mediator not called.",
            },
            "compliance_before": {"can_proceed": True, "violations": [], "state": "BEFORE"},
            "compliance_after": {"can_proceed": True, "violations": [], "state": "AFTER"},
            "round1_responses": [],
            "round2_responses": [],
            "tentative_trades": [],
            "execution_plan": None,
            "final_decision": {
                "decision": "DEFER",
                "branch": "NO_EXECUTABLE_TRADE",
                "trades": [],
                "compliance_check": {"can_proceed": True, "violations": [], "state": "AFTER"},
                "reasoning": summary,
                "user_question": None,
                "user_options": None,
            },
        },
        "direct_indexing": {
            "portfolio_definition": portfolio_definition.to_dict(),
            "drift_report": drift_report,
            "candidate_rebalance_plan": dict(candidate_plan),
        },
        "runtime": {
            "engine": "scheduled_skip",
            "round1_agent_count": 0,
            "round2_agent_count": 0,
        },
        "backtest_schedule": {
            "decision_executed": False,
            "reason": schedule_reason,
            "scheduled_skip": True,
        },
    }


def _decision_row(*, day: str, bundle: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    governance = result.get("governance_v1") if isinstance(result.get("governance_v1"), dict) else {}
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    round1 = governance.get("round1_responses") if isinstance(governance.get("round1_responses"), list) else []
    round2 = governance.get("round2_responses") if isinstance(governance.get("round2_responses"), list) else []
    notification = decision.get("user_notification") if isinstance(decision.get("user_notification"), dict) else {}
    return {
        "date": day,
        "as_of": bundle.get("as_of"),
        "decision": decision.get("decision"),
        "summary": decision.get("summary"),
        "confidence": decision.get("confidence"),
        "urgency": decision.get("urgency"),
        "candidate_rebalance_plan": decision.get("candidate_rebalance_plan") or {},
        "user_handoff": bool(notification.get("action_required"))
        or decision.get("decision") == "USER_DECISION_REQUIRED",
        "called_agents": decision.get("called_agents") or [],
        "round1_agents": [str(item.get("agent_id")) for item in round1 if isinstance(item, dict)],
        "round2_agents": [str(item.get("agent_id")) for item in round2 if isinstance(item, dict)],
        "runtime_engine": runtime.get("engine"),
        "round1_agent_count": runtime.get("round1_agent_count"),
        "round2_agent_count": runtime.get("round2_agent_count"),
        "source_bundle_id": bundle.get("bundle_id"),
        "observed_count": bundle.get("observed_count"),
        "portfolio_relevant_count": bundle.get("portfolio_relevant_count"),
        "document_count": bundle.get("document_count"),
        "scheduled_skip": runtime.get("engine") == "scheduled_skip",
    }


def _sanitize_replay_metadata(result: dict[str, Any], day: str) -> dict[str, Any]:
    sanitized = deepcopy(result)
    decision = sanitized.get("decision") if isinstance(sanitized.get("decision"), dict) else None
    if not isinstance(decision, dict):
        return sanitized
    replay_timestamp = f"{day}T15:30:00+09:00"
    decision["follow_up_at"] = None
    notification = decision.get("user_notification")
    if isinstance(notification, dict):
        notification["estimated_followup"] = None
        notification["sent_at"] = replay_timestamp
    notification_log = decision.get("notification_log")
    if isinstance(notification_log, list):
        for item in notification_log:
            if isinstance(item, dict):
                item["estimated_followup"] = None
                item["sent_at"] = replay_timestamp
    direct_indexing = sanitized.get("direct_indexing")
    if isinstance(direct_indexing, dict):
        drift_report = direct_indexing.get("drift_report")
        if isinstance(drift_report, dict):
            drift_report["computed_at"] = replay_timestamp
    return sanitized


def _expected_round1_agents() -> set[str]:
    if os.environ.get("LIBRA_DOMAIN_AGENTS_ENABLED", "false").strip().lower() == "true":
        return {*CORE_AGENT_IDS, *DOMAIN_AGENT_IDS}
    return set(CORE_AGENT_IDS)


def _assert_service_runtime(result: dict[str, Any], *, day: str) -> None:
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    if runtime.get("engine") != "governance_v1_committee":
        raise RuntimeError(f"{day}: not service v1 committee runtime: {runtime}")
    governance = result.get("governance_v1") if isinstance(result.get("governance_v1"), dict) else {}
    round1 = governance.get("round1_responses") if isinstance(governance.get("round1_responses"), list) else []
    actual = {str(item.get("agent_id")) for item in round1 if isinstance(item, dict)}
    missing = _expected_round1_agents() - actual
    if missing:
        raise RuntimeError(f"{day}: missing round1 agents: {sorted(missing)}; actual={sorted(actual)}")


def _resolve_raw_bundle_path(value: Any, *, raw_dir: Path, bundles_dir: Path) -> Path:
    candidate = Path(str(value))
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend([raw_dir / candidate, bundles_dir / candidate, bundles_dir / candidate.name])
    for path in candidates:
        if path.exists():
            return path
    return candidate


def _resume_from_raw(
    *,
    raw_path: Path,
    bundles_dir: Path,
    expected_dates: list[str],
    price_by_date: dict[str, dict[str, Any]],
    base_weights: dict[str, float],
    initial_shares: dict[str, float],
    cost_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, float], str | None]:
    raw_rows = _read_jsonl(raw_path)
    if not raw_rows:
        return [], dict(initial_shares), None

    raw_dates = [str(row.get("date")) for row in raw_rows]
    if raw_dates != expected_dates[: len(raw_dates)]:
        raise RuntimeError("Resume raw dates do not match the selected replay date prefix.")

    shares = dict(initial_shares)
    decisions: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        day = str(raw_row.get("date"))
        price_row = price_by_date[day]
        bundle_path = _resolve_raw_bundle_path(
            raw_row.get("bundle"),
            raw_dir=raw_path.parent,
            bundles_dir=bundles_dir,
        )
        bundle = _read_json(bundle_path)
        result = raw_row.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Replay row result is not an object for {day}")
        row = _decision_row(day=day, bundle=bundle, result=result)
        decisions.append(row)
        if row["decision"] == "REBALANCE":
            shares = _apply_rebalance(
                shares=shares,
                price_row=price_row,
                base_weights=base_weights,
                candidate_plan=row["candidate_rebalance_plan"],
                cost_rate=cost_rate,
            )
    return decisions, shares, raw_dates[-1]


class TraceRecorder:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.counts: Counter[str] = Counter()
        self.actor_counts: Counter[str] = Counter()
        self.fallback_events = 0
        self._lock = threading.Lock()

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        compact = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        for key in TRACE_KEYS:
            if key in payload:
                compact[key] = payload[key]
        with self._lock:
            self.counts[event] += 1
            actor = str(payload.get("actor") or payload.get("agent_id") or "")
            if actor:
                self.actor_counts[actor] += 1
            phase = str(payload.get("phase") or "")
            if (
                "fallback" in event
                or "fallback" in phase
                or "fallback" in str(payload.get("reason") or "")
            ):
                self.fallback_events += 1
            if self.path is not None:
                _append_jsonl(self.path, compact)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_counts": dict(sorted(self.counts.items())),
                "actor_counts": dict(sorted(self.actor_counts.items())),
                "fallback_events": self.fallback_events,
            }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    _set_default_env(args)
    fixture_path = Path(args.fixture)
    bundles_dir = Path(args.bundles_dir)
    fixture = _read_json(fixture_path)
    prices = fixture.get("prices") or []
    if not prices:
        raise RuntimeError("Fixture has no prices.")
    price_by_date = {str(row["date"]): row for row in prices if isinstance(row, dict) and row.get("date")}
    bundle_rows = _bundle_rows(bundles_dir)
    if args.start_date:
        bundle_rows = [row for row in bundle_rows if str(row.get("prices_until")) >= args.start_date]
    if args.end_date:
        bundle_rows = [row for row in bundle_rows if str(row.get("prices_until")) <= args.end_date]
    if args.limit:
        bundle_rows = bundle_rows[: int(args.limit)]
    if not bundle_rows:
        raise RuntimeError("No replay rows selected. Check --start-date, --end-date, and --limit.")

    selected_dates = [str(row.get("prices_until")) for row in bundle_rows]
    first_price_row = price_by_date.get(selected_dates[0])
    if first_price_row is None:
        raise RuntimeError(f"Selected first replay date is missing from fixture prices: {selected_dates[0]}")

    base_weights = _target_weights(fixture)
    shares = _initial_shares(fixture, first_price_row)
    cost_rate = float(fixture.get("transaction_cost_bp", 0.0)) / 10_000.0
    portfolio_definition = _portfolio_definition(fixture)
    decision_dates = _decision_schedule(bundle_rows, args)
    user_preferences = tuple(args.user_preference or DEFAULT_USER_PREFERENCES)
    ips = _ips_from_args(args)
    kyc = _kyc_from_args(args)
    decisions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_path = Path(args.raw_out) if args.raw_out else None
    resume_raw_path = Path(args.resume_raw) if args.resume_raw else None
    if resume_raw_path:
        resumed_decisions, shares, last_resume_date = _resume_from_raw(
            raw_path=resume_raw_path,
            bundles_dir=bundles_dir,
            expected_dates=selected_dates,
            price_by_date=price_by_date,
            base_weights=base_weights,
            initial_shares=shares,
            cost_rate=cost_rate,
        )
        decisions.extend(resumed_decisions)
        if last_resume_date:
            bundle_rows = [row for row in bundle_rows if str(row.get("prices_until")) > last_resume_date]
    if raw_path and raw_path.exists() and not args.append_raw:
        if resume_raw_path and raw_path.resolve() == resume_raw_path.resolve():
            args.append_raw = True
        else:
            raw_path.unlink()

    trace = TraceRecorder(Path(args.trace_out) if args.trace_out else None)
    with ExitStack() as stack:
        client = open_chat_client_from_args(args, stack=stack)
        client.ensure_available()
        orchestrator = JudgeOrchestrator(
            client=client,
            checkpoint_path=Path(args.state_dir) / "langgraph.sqlite",
        )
        total_replay_count = len(decisions) + len(bundle_rows)
        token = debate_event_publisher.set(trace.publish)
        try:
            for index, bundle_row in enumerate(bundle_rows, start=len(decisions) + 1):
                day = str(bundle_row.get("prices_until"))
                price_row = price_by_date.get(day)
                if price_row is None:
                    continue
                bundle_path, bundle = _load_bundle(bundles_dir, bundle_row)
                portfolio = PortfolioSnapshot.from_dict(
                    _portfolio_payload(
                        day=day,
                        price_row=price_row,
                        shares=shares,
                        user_preferences=user_preferences,
                    )
                )
                drift = compute_drift(portfolio_definition, portfolio)
                candidate_plan = candidate_plan_from_drift(drift)
                if day not in decision_dates:
                    result = _scheduled_skip_result(
                        day=day,
                        bundle=bundle,
                        portfolio=portfolio,
                        portfolio_definition=portfolio_definition,
                        drift_report=drift.to_dict(),
                        candidate_plan=candidate_plan,
                        schedule_reason=(
                            f"decision_frequency={args.decision_frequency}; "
                            f"decision_interval={args.decision_interval}"
                        ),
                    )
                    result = _sanitize_replay_metadata(result, day)
                    row = _decision_row(day=day, bundle=bundle, result=result)
                    decisions.append(row)
                    if raw_path:
                        _append_jsonl(raw_path, {"date": day, "bundle": str(bundle_path), "result": result})
                    if args.progress_every and index % int(args.progress_every) == 0:
                        print(f"replayed {index}/{total_replay_count} through {day}", flush=True)
                    continue

                knowledge_base = LocalKnowledgeBase.from_state_payload(
                    knowledge_payload_from_ingest_bundle(bundle, source_path=str(bundle_path))
                )
                try:
                    result = orchestrator.run_v1_committee(
                        query=args.query,
                        portfolio=portfolio,
                        knowledge_base=knowledge_base,
                        portfolio_definition=portfolio_definition,
                        depth=args.depth,
                        trigger="pull",
                        deadline_seconds=args.deadline_seconds,
                        thread_id=f"{args.thread_prefix}-{day}",
                        enable_human_interrupts=False,
                        ips=ips,
                        kyc=kyc,
                        market_data=_market_snapshot_from_portfolio(portfolio),
                    )
                    _assert_service_runtime(result, day=day)
                except Exception as exc:
                    error = {"date": day, "bundle": str(bundle_path), "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    if not args.continue_on_error:
                        raise
                    continue

                result = _sanitize_replay_metadata(result, day)
                result["backtest_schedule"] = {
                    "decision_executed": True,
                    "decision_frequency": args.decision_frequency,
                    "decision_interval": args.decision_interval,
                }
                row = _decision_row(day=day, bundle=bundle, result=result)
                decisions.append(row)
                if raw_path:
                    _append_jsonl(raw_path, {"date": day, "bundle": str(bundle_path), "result": result})
                if row["decision"] == "REBALANCE":
                    shares = _apply_rebalance(
                        shares=shares,
                        price_row=price_row,
                        base_weights=base_weights,
                        candidate_plan=row["candidate_rebalance_plan"],
                        cost_rate=cost_rate,
                    )
                if args.sleep_seconds:
                    time.sleep(float(args.sleep_seconds))
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"replayed {index}/{total_replay_count} through {day}", flush=True)
        finally:
            debate_event_publisher.reset(token)

    if args.fail_on_fallback_events and trace.fallback_events:
        raise RuntimeError(f"Fallback events observed: {trace.fallback_events}")

    _write_json(Path(args.out), decisions)
    breakdown = Counter(str(row.get("decision")) for row in decisions)
    round1_agent_union = sorted({agent for row in decisions for agent in row.get("round1_agents", [])})
    scheduled_skip_count = sum(1 for row in decisions if row.get("scheduled_skip"))
    llm_decision_count = len(decisions) - scheduled_skip_count
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_repo": str(Path(__file__).resolve().parents[1]),
        "fixture": str(fixture_path),
        "bundles_dir": str(bundles_dir),
        "out": str(Path(args.out)),
        "raw_out": str(raw_path) if raw_path else None,
        "usage_log": str(Path(args.usage_log)) if args.usage_log else None,
        "backend": getattr(args, "backend", None),
        "anthropic_model": args.anthropic_model or os.environ.get("LIBRA_ANTHROPIC_MODEL"),
        "runtime": "JudgeOrchestrator.run_v1_committee",
        "domain_agents_enabled": os.environ.get("LIBRA_DOMAIN_AGENTS_ENABLED"),
        "llm_routing_policy": os.environ.get("LLM_ROUTING_POLICY"),
        "disable_agent_fallbacks": os.environ.get("LIBRA_DISABLE_AGENT_FALLBACKS"),
        "sentiment_phase2_enabled": os.environ.get("LIBRA_SENTIMENT_PHASE2_ENABLED"),
        "committee_round1_max_workers": os.environ.get("LIBRA_COMMITTEE_ROUND1_MAX_WORKERS"),
        "committee_round2_max_workers": os.environ.get("LIBRA_COMMITTEE_ROUND2_MAX_WORKERS"),
        "committee_llm_repair_attempts": os.environ.get("LIBRA_COMMITTEE_LLM_REPAIR_ATTEMPTS"),
        "drop_invalid_mediator_targets": os.environ.get("LIBRA_DROP_INVALID_MEDIATOR_TARGETS"),
        "committee_opinion_reasoning_chars": os.environ.get("LIBRA_COMMITTEE_OPINION_REASONING_CHARS"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "decision_frequency": args.decision_frequency,
        "decision_interval": args.decision_interval,
        "decision_count": len(decisions),
        "llm_decision_count": llm_decision_count,
        "scheduled_skip_count": scheduled_skip_count,
        "selected_first_date": selected_dates[0],
        "selected_last_date": selected_dates[-1],
        "errors": errors,
        "decision_breakdown": dict(sorted(breakdown.items())),
        "rebalance_count": sum(1 for row in decisions if row.get("decision") == "REBALANCE"),
        "user_handoff_count": sum(1 for row in decisions if row.get("user_handoff")),
        "round1_agent_union": round1_agent_union,
        "expected_round1_agents": sorted(_expected_round1_agents()),
        "trace": trace.summary(),
    }
    if args.summary_out:
        _write_json(Path(args.summary_out), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the real libra-agent v1 committee service runtime over point-in-time backtest bundles."
    )
    parser.add_argument("--fixture", required=True, help="Comparison fixture JSON containing prices and target weights.")
    parser.add_argument("--bundles-dir", required=True, help="Directory containing ingest-bundles/index.json.")
    parser.add_argument("--out", required=True, help="Output JSON array for compact LIBRA decisions.")
    parser.add_argument("--summary-out", help="Optional JSON replay summary path.")
    parser.add_argument("--raw-out", help="Optional JSONL path for full raw LIBRA results.")
    parser.add_argument("--append-raw", action="store_true", help="Append to --raw-out instead of replacing it.")
    parser.add_argument("--resume-raw", help="Existing raw JSONL prefix to resume from without recomputing prior days.")
    parser.add_argument("--usage-log", help="Anthropic usage JSONL path. Also sets LIBRA_LLM_USAGE_LOG.")
    parser.add_argument("--trace-out", help="Optional compact debate event JSONL path.")
    parser.add_argument("--env-file", help="Optional dotenv file loaded before opening the LLM backend.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--depth", default="shallow", choices=("shallow", "medium", "deep"))
    parser.add_argument("--deadline-seconds", type=int)
    parser.add_argument("--state-dir", default="outputs/backtests/service-committee-replay-state")
    parser.add_argument("--thread-prefix", default="service-committee-backtest")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--decision-frequency",
        default="daily",
        choices=("daily", "every-n-trading-days", "weekly"),
        help="How often to run the LLM committee inside the selected date range.",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=1,
        help="Trading-day interval used when --decision-frequency every-n-trading-days is selected.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--fail-on-fallback-events", action="store_true")
    parser.add_argument("--user-preference", action="append", help="Repeatable user preference injected into each replay portfolio.")
    parser.add_argument("--single-ticker-limit-pct", type=float, default=100.0)
    parser.add_argument("--sector-limit-pct", type=float, default=100.0)
    parser.add_argument("--annual-volatility-limit", type=float, default=1.0)
    parser.add_argument("--min-cash-pct", type=float, default=0.0)
    parser.add_argument("--max-market-impact-pct-of-adv", type=float, default=100.0)
    parser.add_argument("--risk-tolerance", default="MODERATE", choices=("CONSERVATIVE", "MODERATE", "AGGRESSIVE"))
    parser.add_argument("--investment-horizon-years", type=int, default=15)
    parser.add_argument("--max-drawdown-tolerance-pct", type=float, default=50.0)
    add_backend_arguments(parser, default_backend="anthropic", backend_help="LLM backend/provider for service-runtime replay")
    return parser


def main() -> None:
    replay(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
