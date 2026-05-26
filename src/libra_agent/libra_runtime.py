from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from libra_agent.runtime.debate_events import (
    publish_debate_event,
    publish_llm_error,
    publish_llm_prompt,
    publish_llm_response,
    publish_llm_skipped,
    publish_tool_observation,
)

from .libra.constraints import default_constraints_for, validate_rebalance_plan
from .libra.direct_indexing import (
    PortfolioDefinition,
    candidate_plan_from_drift,
    compact_drift_context,
    compute_drift,
)
from .libra.llm_clients.base import ChatClientError, ChatClientProtocol
from .libra.prompts import (
    JUDGE_ACTION_RULES,
    JUDGE_ACTION_SYSTEM_PROMPT,
    JUDGE_DOMAIN_ACTION_SYSTEM_PROMPT,
    JUDGE_NOTIFICATION_LEVELS,
    JUDGE_PHASE_REQUIRED_KEYS,
    JUDGE_PHASE_SYSTEM_PROMPT,
    InformationAgentPromptProfile,
    default_agent_fallback,
    default_agent_note,
    default_agent_query,
    get_information_prompt_profile,
)
from .libra_models import (
    AgentResponse,
    AgentVerdict,
    DecisionPhase,
    DecisionTraceNode,
    DecisionType,
    JudgeDecision,
    KnowledgeDocument,
    KnowledgeEntity,
    KnowledgeEvent,
    PortfolioSnapshot,
    ToolCall,
    TriggerEvent,
    Urgency,
    UserNotification,
)
from .libra_validation import (
    EMPTY_PORTFOLIO_NO_TRADE_REASONING,
    EMPTY_PORTFOLIO_NO_TRADE_SUMMARY,
    sanitize_agent_evidence,
    sanitize_agent_response_payload,
    sanitize_judge_payload,
)
from .utils import coerce_datetime, collapse_whitespace, contains_japanese_kana, stable_hash

ChatClient = ChatClientProtocol

CORE_ROUTING_AGENT_IDS = ("disclosure", "news", "report", "profit", "cost")
DOMAIN_ROUTING_AGENT_IDS = (
    "risk",
    "tax",
    "compliance",
    "macro",
    "sentiment",
    "execution",
    "esg",
    "liquidity",
    "technical",
)


POSITIVE_KEYWORDS = (
    "상회",
    "서프라이즈",
    "개선",
    "상향",
    "회복",
    "증가",
    "호조",
    "수주",
    "확대",
    "강세",
    "흑자",
    "improving",
    "beat",
    "upgrade",
    "recovery",
    "growth",
)

NEGATIVE_KEYWORDS = (
    "하회",
    "미스",
    "악화",
    "하향",
    "감소",
    "우려",
    "약세",
    "조사",
    "리콜",
    "소송",
    "규제",
    "화재",
    "적자",
    "miss",
    "downgrade",
    "probe",
    "investigation",
    "recall",
    "lawsuit",
    "weakness",
)

PUSH_RISK_KEYWORDS = (
    "조사",
    "규제",
    "리콜",
    "화재",
    "소송",
    "probe",
    "investigation",
    "lawsuit",
    "recall",
    "fire",
)

EVENT_TYPE_BIAS = {
    "EARNINGS": 0.35,
    "CAPEX": 0.15,
    "RESEARCH": 0.22,
    "DISCLOSURE": 0.08,
    "PRODUCT": 0.12,
    "FUNDING": 0.1,
    "GOVERNANCE": -0.08,
    "REGULATION": -0.28,
    "LEGAL": -0.35,
    "GEOPOLITICAL": -0.22,
    "MACRO": 0.0,
    "MNA": 0.18,
    "OTHER": 0.0,
}


def normalize_ticker(value: str) -> str:
    return "".join(char for char in value.upper() if char.isalnum())


def canonical_agent_id(value: str) -> str:
    normalized = str(value).strip().casefold()
    compact = normalized.replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "dart": "disclosure",
        "disclosureagent": "disclosure",
        "newsagent": "news",
        "reportagent": "report",
        "profitagent": "profit",
        "costagent": "cost",
        "riskagent": "risk",
        "taxagent": "tax",
        "complianceagent": "compliance",
        "macroagent": "macro",
        "sentimentagent": "sentiment",
        "executionagent": "execution",
        "esgagent": "esg",
        "liquidityagent": "liquidity",
        "technicalanalysisagent": "technical",
        "technicalagent": "technical",
        "ta": "technical",
    }
    if compact in aliases:
        return aliases[compact]
    if normalized == "dart":
        return "disclosure"
    return normalized


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def truncate(value: str, limit: int = 700) -> str:
    text = collapse_whitespace(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def event_direction_score(event: KnowledgeEvent) -> float:
    text = f"{event.headline}\n{event.summary}".casefold()
    score = EVENT_TYPE_BIAS.get(event.event_type, 0.0)
    positive_hits = sum(1 for token in POSITIVE_KEYWORDS if token in text)
    negative_hits = sum(1 for token in NEGATIVE_KEYWORDS if token in text)
    score += positive_hits * 0.16
    score -= negative_hits * 0.2
    score *= 0.5 + (event.confidence * 0.5)
    return clamp(score, -1.0, 1.0)


def candidate_plan_to_proposed_trades(candidate_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for ticker, delta in candidate_plan.items():
        weight_delta = float(delta)
        side = "buy" if weight_delta > 0 else "sell" if weight_delta < 0 else "hold"
        trades.append(
            {
                "symbol": ticker,
                "weight_delta": weight_delta,
                "side": side,
                "delta": weight_delta,
                "action": side.upper(),
            }
        )
    return trades


@dataclass(slots=True, frozen=True)
class KnowledgeSlice:
    events: list[KnowledgeEvent]
    documents: list[KnowledgeDocument]
    tools_called: list[ToolCall]


@dataclass(slots=True, frozen=True)
class AgentToolLoopResult:
    knowledge_slice: KnowledgeSlice
    stop_reason: str


@dataclass(slots=True, frozen=True)
class IngestRefreshResult:
    tool_call: ToolCall
    changed: bool


@dataclass(slots=True, frozen=True)
class PlannedAgentCall:
    agent_id: str
    query: str
    context: str
    depth: str
    fallback: str | None = None
    note: str | None = None


@dataclass(slots=True)
class RunState:
    trigger: str
    trigger_event: TriggerEvent | None
    started_at: datetime
    deadline_at: datetime | None
    notification_log: list[UserNotification]


class LocalKnowledgeBase:
    def __init__(
        self,
        *,
        events: list[KnowledgeEvent],
        documents: list[KnowledgeDocument],
        source_paths: dict[str, str],
    ) -> None:
        self.events = events
        self.documents = documents
        self.source_paths = source_paths
        self.documents_by_id = {document.doc_id: document for document in documents}

    @classmethod
    def from_files(
        cls,
        *,
        events_path: str | Path | None = None,
        enriched_documents_path: str | Path | None = None,
        normalized_documents_path: str | Path | None = None,
    ) -> LocalKnowledgeBase:
        events: list[KnowledgeEvent] = []
        source_paths: dict[str, str] = {}
        if events_path and Path(events_path).exists():
            events = [cls._event_from_payload(item) for item in cls._read_records(events_path)]
            source_paths["events"] = str(Path(events_path))

        documents: list[KnowledgeDocument] = []
        if normalized_documents_path and Path(normalized_documents_path).exists():
            source_paths["normalized_documents"] = str(Path(normalized_documents_path))
            documents = [
                cls._document_from_normalized_payload(item)
                for item in cls._read_records(normalized_documents_path)
            ]
        elif enriched_documents_path and Path(enriched_documents_path).exists():
            source_paths["enriched_documents"] = str(Path(enriched_documents_path))
            documents = [
                cls._document_from_enriched_payload(item)
                for item in cls._read_records(enriched_documents_path)
            ]
        return cls(events=events, documents=documents, source_paths=source_paths)

    @classmethod
    def from_state_payload(cls, payload: Mapping[str, Any]) -> LocalKnowledgeBase:
        events = [
            KnowledgeEvent.from_dict(item)
            for item in payload.get("events", [])
            if isinstance(item, Mapping)
        ]
        documents = [
            KnowledgeDocument.from_dict(item)
            for item in payload.get("documents", [])
            if isinstance(item, Mapping)
        ]
        source_paths = (
            dict(payload.get("source_paths", {}))
            if isinstance(payload.get("source_paths"), Mapping)
            else {}
        )
        return cls(events=events, documents=documents, source_paths=source_paths)

    def to_state_payload(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "documents": [document.to_dict() for document in self.documents],
            "source_paths": dict(self.source_paths),
        }

    def refresh_from_ingest(self, *, agent_id: str) -> IngestRefreshResult:
        agent_id = canonical_agent_id(agent_id)
        tool_name = f"ingest.refresh_{agent_id}"
        if not self._ingest_refresh_enabled():
            return IngestRefreshResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    purpose="근거 부족 시 upstream ingest 갱신",
                    summary="ingest refresh가 비활성화되어 있어 로컬 캐시 안에서만 판단합니다.",
                ),
                changed=False,
            )

        ingest_root = self._resolve_ingest_root()
        if ingest_root is None:
            return IngestRefreshResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    purpose="근거 부족 시 upstream ingest 갱신",
                    summary="libra-ingest 루트를 찾지 못해 refresh를 실행하지 못했습니다.",
                ),
                changed=False,
            )

        out_dir = self._ingest_out_dir(agent_id)
        command = self._ingest_command(agent_id=agent_id, ingest_root=ingest_root, out_dir=out_dir)
        env = os.environ.copy()
        src_path = str(ingest_root / "src")
        env["PYTHONPATH"] = src_path + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        try:
            completed = subprocess.run(
                command,
                cwd=ingest_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._ingest_timeout_seconds(),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return IngestRefreshResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    purpose="근거 부족 시 upstream ingest 갱신",
                    summary=f"ingest refresh 실행 실패: {type(exc).__name__}.",
                ),
                changed=False,
            )

        if completed.returncode != 0:
            stderr = truncate(completed.stderr or completed.stdout or "unknown error", 220)
            if "Live baseline fetch returned no documents" in (
                completed.stderr or completed.stdout or ""
            ):
                return IngestRefreshResult(
                    tool_call=ToolCall(
                        tool_name=tool_name,
                        purpose="근거 부족 시 upstream ingest 갱신",
                        summary=f"ingest refresh는 실행됐지만 새 문서가 없습니다. out_dir={out_dir}",
                    ),
                    changed=False,
                )
            return IngestRefreshResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    purpose="근거 부족 시 upstream ingest 갱신",
                    summary=f"ingest refresh가 실패했습니다. exit={completed.returncode}; {stderr}",
                ),
                changed=False,
            )

        refreshed = LocalKnowledgeBase.from_files(
            events_path=out_dir / "events.json",
            normalized_documents_path=out_dir / "normalized_documents.json",
        )
        if not refreshed.events and not refreshed.documents:
            return IngestRefreshResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    purpose="근거 부족 시 upstream ingest 갱신",
                    summary=f"ingest refresh는 완료됐지만 새 이벤트나 문서가 없습니다. out_dir={out_dir}",
                ),
                changed=False,
            )

        self._merge_from(refreshed)
        self.source_paths["last_ingest_refresh_agent"] = agent_id
        self.source_paths["last_ingest_refresh_out_dir"] = str(out_dir)
        self.source_paths["events"] = str(out_dir / "events.json")
        self.source_paths["normalized_documents"] = str(out_dir / "normalized_documents.json")
        return IngestRefreshResult(
            tool_call=ToolCall(
                tool_name=tool_name,
                purpose="근거 부족 시 upstream ingest 갱신",
                summary=(
                    f"ingest refresh 성공. 새 이벤트 {len(refreshed.events)}건, "
                    f"정규화 문서 {len(refreshed.documents)}건을 로컬 지식 캐시에 병합했습니다."
                ),
            ),
            changed=True,
        )

    def _merge_from(self, other: LocalKnowledgeBase) -> None:
        event_ids = {event.event_id or stable_hash(event.to_dict()) for event in self.events}
        for event in other.events:
            key = event.event_id or stable_hash(event.to_dict())
            if key not in event_ids:
                self.events.append(event)
                event_ids.add(key)

        document_ids = {
            document.doc_id or stable_hash(document.to_dict()) for document in self.documents
        }
        for document in other.documents:
            key = document.doc_id or stable_hash(document.to_dict())
            if key not in document_ids:
                self.documents.append(document)
                document_ids.add(key)
                self.documents_by_id[document.doc_id] = document

    def _ingest_refresh_enabled(self) -> bool:
        raw = (
            os.getenv("LIBRA_INGEST_REFRESH_ENABLED")
            or self.source_paths.get("ingest_refresh_enabled")
            or "false"
        )
        return str(raw).strip().casefold() in {"1", "true", "yes", "y", "on"}

    def _resolve_ingest_root(self) -> Path | None:
        candidates = [
            os.getenv("LIBRA_INGEST_ROOT"),
            self.source_paths.get("ingest_root"),
            str(Path.cwd().parent / "libra-ingest"),
            r"D:\libra-ingest",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if (path / "src" / "libra_ingest" / "ingest_cli.py").is_file():
                return path.resolve()
        return None

    def _ingest_out_dir(self, agent_id: str) -> Path:
        base = (
            os.getenv("LIBRA_INGEST_OUT_DIR")
            or self.source_paths.get("ingest_out_dir")
            or str(Path("outputs") / "ingest_refresh")
        )
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return Path(base).expanduser().resolve() / agent_id / timestamp

    def _ingest_timeout_seconds(self) -> float:
        raw = os.getenv("LIBRA_INGEST_REFRESH_TIMEOUT_SECONDS") or self.source_paths.get(
            "ingest_refresh_timeout_seconds"
        )
        try:
            return max(1.0, float(raw)) if raw else 90.0
        except (TypeError, ValueError):
            return 90.0

    def _ingest_mode(self) -> str:
        raw = (
            os.getenv("LIBRA_INGEST_REFRESH_MODE")
            or self.source_paths.get("ingest_refresh_mode")
            or "sample"
        )
        return "live" if str(raw).strip().casefold() == "live" else "sample"

    def _ingest_command(self, *, agent_id: str, ingest_root: Path, out_dir: Path) -> list[str]:
        command = [
            os.getenv("LIBRA_INGEST_PYTHON") or sys.executable,
            "-m",
            "libra_ingest.ingest_cli",
            "--out-dir",
            str(out_dir),
            "--emit-push-candidates",
            "--pretty",
        ]
        mode = self._ingest_mode()
        if mode == "live":
            return [*command, *self._live_ingest_args(agent_id)]
        return [*command, *self._sample_ingest_args(agent_id, ingest_root)]

    def _sample_ingest_args(self, agent_id: str, ingest_root: Path) -> list[str]:
        examples = ingest_root / "examples"
        if agent_id == "disclosure":
            return ["--dart-records", str(examples / "dart-records.sample.json")]
        if agent_id == "report":
            return ["--report-rows", str(examples / "report-rows.sample.json")]
        return ["--rss-items", str(examples / "rss-items.sample.json")]

    def _live_ingest_args(self, agent_id: str) -> list[str]:
        rss_limit = self._env_int("LIBRA_INGEST_RSS_LIMIT", "ingest_rss_limit", 5)
        dart_limit = self._env_int("LIBRA_INGEST_DART_LIMIT", "ingest_dart_limit", 20)
        report_limit = self._env_int("LIBRA_INGEST_REPORT_LIMIT", "ingest_report_limit", 10)
        if agent_id == "news":
            dart_limit = 0
            report_limit = 0
        elif agent_id == "disclosure":
            rss_limit = 0
            report_limit = 0
        elif agent_id == "report":
            rss_limit = 0
            dart_limit = 0
        args = [
            "--live-baseline",
            "--rss-limit",
            str(rss_limit),
            "--dart-limit",
            str(dart_limit),
            "--report-limit",
            str(report_limit),
            "--report-pdf-pages",
            str(self._env_int("LIBRA_INGEST_REPORT_PDF_PAGES", "ingest_report_pdf_pages", 5)),
            "--report-min-body-chars",
            str(
                self._env_int(
                    "LIBRA_INGEST_REPORT_MIN_BODY_CHARS", "ingest_report_min_body_chars", 500
                )
            ),
        ]
        live_date = os.getenv("LIBRA_INGEST_LIVE_DATE") or self.source_paths.get("ingest_live_date")
        if live_date:
            args.extend(["--live-date", str(live_date)])
        if self._env_bool(
            "LIBRA_INGEST_SKIP_ARTICLE_BODY", "ingest_skip_article_body", default=True
        ):
            args.append("--skip-article-body")
        return args

    def _env_int(self, env_key: str, source_key: str, default: int) -> int:
        raw = os.getenv(env_key) or self.source_paths.get(source_key)
        try:
            return max(0, int(float(raw))) if raw is not None else default
        except (TypeError, ValueError):
            return default

    def _env_bool(self, env_key: str, source_key: str, *, default: bool) -> bool:
        raw = os.getenv(env_key) or self.source_paths.get(source_key)
        if raw is None:
            return default
        return str(raw).strip().casefold() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _read_json(path: str | Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def _read_records(cls, path: str | Path) -> list[Mapping[str, Any]]:
        file_path = Path(path)
        if file_path.suffix.lower() == ".jsonl":
            records: list[Mapping[str, Any]] = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if isinstance(payload, Mapping):
                        records.append(payload)
            return records
        return cls._as_records(cls._read_json(file_path))

    @staticmethod
    def _as_records(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            for key in ("documents", "events"):
                records = payload.get(key)
                if isinstance(records, list):
                    return [item for item in records if isinstance(item, Mapping)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        return []

    @classmethod
    def _entities_from_payload(cls, payload: Any) -> tuple[KnowledgeEntity, ...]:
        if not isinstance(payload, list):
            return ()
        entities: list[KnowledgeEntity] = []
        for item in payload:
            if isinstance(item, Mapping):
                entities.append(KnowledgeEntity.from_dict(item))
        return tuple(entities)

    @classmethod
    def _document_from_enriched_payload(cls, payload: Mapping[str, Any]) -> KnowledgeDocument:
        document_payload = payload.get("document", {})
        source_info = (
            document_payload.get("source_info", {}) if isinstance(document_payload, Mapping) else {}
        )
        normalized_content = (
            document_payload.get("normalized_content", {})
            if isinstance(document_payload, Mapping)
            else {}
        )
        timing_info = (
            document_payload.get("timing_info", {}) if isinstance(document_payload, Mapping) else {}
        )
        return KnowledgeDocument(
            doc_id=str(document_payload.get("doc_id", "")),
            doc_type=str(document_payload.get("doc_type", "")),
            title=str(normalized_content.get("title", "")),
            body=str(normalized_content.get("body", "")),
            publisher=str(source_info.get("publisher", "")),
            source_name=str(source_info.get("source_name", "")),
            source_url=str(source_info.get("source_url", "")),
            region=str(source_info.get("region", "")),
            published_at=coerce_datetime(str(timing_info.get("published_at", "")) or None),
            relevance_score=float(payload.get("relevance_score", 0.0))
            if payload.get("relevance_score") is not None
            else None,
            event_type=str(payload.get("event_type", "")) or None,
            event_type_score=float(payload.get("event_type_score", 0.0))
            if payload.get("event_type_score") is not None
            else None,
            entities=cls._entities_from_payload(payload.get("entities")),
            metadata=dict(payload.get("cluster_metadata", {}))
            if isinstance(payload.get("cluster_metadata"), Mapping)
            else {},
        )

    @classmethod
    def _document_from_normalized_payload(cls, payload: Mapping[str, Any]) -> KnowledgeDocument:
        source_info = payload.get("source_info", {}) if isinstance(payload, Mapping) else {}
        normalized_content = (
            payload.get("normalized_content", {}) if isinstance(payload, Mapping) else {}
        )
        timing_info = payload.get("timing_info", {}) if isinstance(payload, Mapping) else {}
        return KnowledgeDocument(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            title=str(normalized_content.get("title", "")),
            body=str(normalized_content.get("body", "")),
            publisher=str(source_info.get("publisher", "")),
            source_name=str(source_info.get("source_name", "")),
            source_url=str(source_info.get("source_url", "")),
            region=str(source_info.get("region", "")),
            published_at=coerce_datetime(str(timing_info.get("published_at", "")) or None),
            entities=cls._entities_from_payload(payload.get("entities")),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    @classmethod
    def _event_from_payload(cls, payload: Mapping[str, Any]) -> KnowledgeEvent:
        return KnowledgeEvent(
            event_id=str(payload.get("event_id", "")),
            event_type=str(payload.get("event_type", "")),
            event_time=coerce_datetime(str(payload.get("event_time", "")) or None),
            headline=str(payload.get("headline", "")),
            summary=str(payload.get("summary", "")),
            confidence=clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0),
            source_documents=tuple(
                str(item) for item in payload.get("source_documents", []) if str(item)
            ),
            entities=cls._entities_from_payload(payload.get("entities")),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    def slice_for_agent(
        self,
        *,
        agent_id: str,
        portfolio: PortfolioSnapshot,
        query: str,
        depth: str = "medium",
    ) -> KnowledgeSlice:
        agent_id = canonical_agent_id(agent_id)
        alias_map = self._alias_map(portfolio)
        relevant_events = self._filter_events(agent_id=agent_id, alias_map=alias_map, query=query)
        relevant_documents = self._filter_documents(
            agent_id=agent_id, alias_map=alias_map, query=query
        )

        event_limit = {"shallow": 4, "medium": 8, "deep": 12}.get(depth, 8)
        document_limit = {"shallow": 3, "medium": 6, "deep": 9}.get(depth, 6)
        relevant_events = relevant_events[:event_limit]
        relevant_documents = relevant_documents[:document_limit]

        tools_called = [
            ToolCall(
                tool_name="local_knowledge.load_events",
                purpose=f"{agent_id} 에이전트 이벤트 근거 확인",
                summary=f"관련 로컬 이벤트 {len(relevant_events)}건을 불러왔습니다.",
            ),
            ToolCall(
                tool_name="local_knowledge.load_documents",
                purpose=f"{agent_id} 에이전트 문서 근거 확인",
                summary=f"정규화 문서 캐시에서 관련 문서 {len(relevant_documents)}건을 불러왔습니다.",
            ),
        ]
        return KnowledgeSlice(
            events=relevant_events, documents=relevant_documents, tools_called=tools_called
        )

    def ticker_signal(self, ticker: str, portfolio: PortfolioSnapshot) -> float:
        alias_map = self._alias_map(portfolio)
        normalized_ticker = normalize_ticker(ticker)
        matches = []
        for event in self.events:
            matched = self._match_tickers(
                headline=event.headline,
                body=event.summary,
                entities=event.entities,
                alias_map=alias_map,
            )
            if normalized_ticker in matched:
                matches.append(self._event_direction(event))
        if not matches:
            return 0.0
        return clamp(sum(matches) / len(matches), -1.0, 1.0)

    def portfolio_signal_scan(
        self, portfolio: PortfolioSnapshot
    ) -> tuple[dict[str, float], ToolCall]:
        signals = {
            holding.ticker: self.ticker_signal(holding.ticker, portfolio)
            for holding in portfolio.holdings
        }
        non_zero_count = sum(1 for value in signals.values() if abs(value) >= 0.01)
        return signals, ToolCall(
            tool_name="local_knowledge.portfolio_signal_scan",
            purpose="보유 종목별 로컬 이벤트 방향성 재점검",
            summary=f"보유 종목 {len(signals)}개 중 방향성 신호가 있는 종목 {non_zero_count}개를 확인했습니다.",
        )

    def _filter_events(
        self,
        *,
        agent_id: str,
        alias_map: dict[str, set[str]],
        query: str,
    ) -> list[KnowledgeEvent]:
        results: list[KnowledgeEvent] = []
        agent_id = canonical_agent_id(agent_id)
        market_wide = not alias_map
        wants_macro = any(
            token in query.casefold()
            for token in ("거시", "매크로", "macro", "환율", "금리", "지수")
        )
        for event in sorted(self.events, key=lambda item: item.event_time, reverse=True):
            matched = self._match_tickers(
                headline=event.headline,
                body=event.summary,
                entities=event.entities,
                alias_map=alias_map,
            )
            if agent_id == "disclosure" and event.event_type not in {"DISCLOSURE", "EARNINGS"}:
                continue
            if agent_id == "report" and event.event_type not in {
                "RESEARCH",
                "EARNINGS",
                "DISCLOSURE",
            }:
                continue
            if agent_id == "news" and event.event_type == "RESEARCH":
                continue
            if market_wide:
                if agent_id == "news" and event.event_type not in {"NEWS", "NEWS_FLOW", "MACRO"}:
                    continue
                if agent_id == "report" and event.event_type != "RESEARCH":
                    continue
            if matched or event.event_type == "MACRO" or wants_macro:
                results.append(
                    KnowledgeEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        event_time=event.event_time,
                        headline=event.headline,
                        summary=event.summary,
                        confidence=event.confidence,
                        source_documents=event.source_documents,
                        matched_holdings=tuple(sorted(matched)),
                        entities=event.entities,
                        metadata=event.metadata,
                    )
                )
            elif market_wide:
                results.append(
                    KnowledgeEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        event_time=event.event_time,
                        headline=event.headline,
                        summary=event.summary,
                        confidence=event.confidence,
                        source_documents=event.source_documents,
                        matched_holdings=(),
                        entities=event.entities,
                        metadata=event.metadata,
                    )
                )
        return results

    def _filter_documents(
        self,
        *,
        agent_id: str,
        alias_map: dict[str, set[str]],
        query: str,
    ) -> list[KnowledgeDocument]:
        agent_id = canonical_agent_id(agent_id)
        doc_type_filter = {
            "disclosure": {"DISCLOSURE"},
            "news": {"NEWS"},
            "report": {"REPORT"},
        }.get(agent_id, set())
        results: list[KnowledgeDocument] = []
        market_wide = not alias_map
        wants_macro = any(
            token in query.casefold()
            for token in ("거시", "매크로", "macro", "환율", "금리", "지수")
        )
        for document in sorted(self.documents, key=lambda item: item.published_at, reverse=True):
            if doc_type_filter and document.doc_type not in doc_type_filter:
                continue
            matched = self._match_tickers(
                headline=document.title,
                body=document.body,
                entities=document.entities,
                alias_map=alias_map,
            )
            if (
                matched
                or market_wide
                or (agent_id == "news" and wants_macro and document.doc_type == "NEWS")
            ):
                results.append(
                    KnowledgeDocument(
                        doc_id=document.doc_id,
                        doc_type=document.doc_type,
                        title=document.title,
                        body=document.body,
                        publisher=document.publisher,
                        source_name=document.source_name,
                        source_url=document.source_url,
                        region=document.region,
                        published_at=document.published_at,
                        relevance_score=document.relevance_score,
                        event_type=document.event_type,
                        event_type_score=document.event_type_score,
                        entities=document.entities,
                        matched_holdings=tuple(sorted(matched)),
                        metadata=document.metadata,
                    )
                )
        return results

    def _alias_map(self, portfolio: PortfolioSnapshot) -> dict[str, set[str]]:
        alias_map: dict[str, set[str]] = {}
        for holding in portfolio.holdings:
            normalized = normalize_ticker(holding.ticker)
            aliases = {
                holding.company_name.casefold(),
                holding.ticker.casefold(),
                normalized.casefold(),
            }
            short_numeric = "".join(char for char in holding.ticker if char.isdigit())
            if short_numeric:
                aliases.add(short_numeric.casefold())
            for alias in holding.aliases:
                aliases.add(alias.casefold())
            alias_map[normalized] = {alias for alias in aliases if alias}
        return alias_map

    def _match_tickers(
        self,
        *,
        headline: str,
        body: str,
        entities: tuple[KnowledgeEntity, ...],
        alias_map: dict[str, set[str]],
    ) -> set[str]:
        matched: set[str] = set()
        haystack = f"{headline}\n{body}".casefold()
        entity_tickers = {normalize_ticker(entity.ticker) for entity in entities if entity.ticker}
        for ticker, aliases in alias_map.items():
            if ticker in entity_tickers:
                matched.add(ticker)
                continue
            for alias in aliases:
                if alias and alias in haystack:
                    matched.add(ticker)
                    break
        return matched

    def _event_direction(self, event: KnowledgeEvent) -> float:
        return event_direction_score(event)


def _agent_fallbacks_disabled() -> bool:
    return os.getenv("LIBRA_DISABLE_AGENT_FALLBACKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


class LLMAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        client: ChatClient,
        prompt_profile: InformationAgentPromptProfile | None = None,
    ) -> None:
        self.agent_id = canonical_agent_id(agent_id)
        self.client = client
        self.prompt_profile = prompt_profile or get_information_prompt_profile(self.agent_id)

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
    ) -> AgentResponse:
        strict_no_fallback = _agent_fallbacks_disabled()
        if strict_no_fallback:
            fallback = None
        tool_loop = self._run_tool_loop(
            query=query,
            context=context,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=depth,
        )
        knowledge_slice = tool_loop.knowledge_slice
        publish_tool_observation(
            actor=self.agent_id,
            phase="agent_tool_loop",
            tools=knowledge_slice.tools_called,
        )
        opinion_id = f"{self.agent_id}_{stable_hash({'agent': self.agent_id, 'turn': turn_number, 'query': query})[:12]}"

        if not knowledge_slice.events and not knowledge_slice.documents:
            publish_llm_skipped(
                actor=self.agent_id,
                phase="agent_response",
                reason="이 에이전트가 판단할 관련 로컬 근거가 없어 LLM 호출을 생략했습니다.",
                context={
                    "query": query,
                    "depth": depth,
                    "stop_reason": tool_loop.stop_reason,
                    "events": 0,
                    "documents": 0,
                },
            )
            verdict = (
                AgentVerdict.QUIET
                if self.agent_id == "news"
                else AgentVerdict.DIRECT_ANSWER_UNAVAILABLE
            )
            response = AgentResponse(
                agent_id=self.agent_id,
                opinion_id=opinion_id,
                turn_number=turn_number,
                query_understood=query,
                verdict=verdict,
                evidence=sanitize_agent_evidence(
                    agent_id=self.agent_id, evidence={}, portfolio=portfolio
                ),
                direction=0.0,
                strength=0.0,
                urgency=Urgency.DEFER,
                confidence=0.2,
                reasoning_for_judge_agent="도구 루프를 실행했지만 이 에이전트가 판단할 관련 로컬 근거가 없습니다.",
                limits_acknowledged=f"현재 로컬 캐시에 보유 종목과 일치하는 항목이 없습니다. stop_reason={tool_loop.stop_reason}",
                tools_called=knowledge_slice.tools_called,
                depth_used=depth,
            )
            return sanitize_agent_response_payload(
                response.to_dict(),
                agent_id=self.agent_id,
                portfolio=portfolio,
                query=query,
                turn_number=turn_number,
                opinion_id=opinion_id,
                depth=depth,
            )

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            query=query,
            context=context,
            fallback=fallback,
            note=note,
            portfolio=portfolio,
            knowledge_slice=knowledge_slice,
            depth=depth,
            turn_number=turn_number,
        )
        raw_response: Mapping[str, Any] = {}
        try:
            publish_llm_prompt(
                actor=self.agent_id,
                phase="agent_response",
                model=str(getattr(self.client, "model", "unknown")),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            raw_response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            publish_llm_response(
                actor=self.agent_id,
                phase="agent_response",
                model=str(getattr(self.client, "model", "unknown")),
                output=raw_response,
            )
            response = self._normalize_llm_response(
                raw_response,
                query=query,
                turn_number=turn_number,
                portfolio=portfolio,
                knowledge_slice=knowledge_slice,
                opinion_id=opinion_id,
                depth=depth,
            )
        except (ChatClientError, ValueError, TypeError) as exc:
            publish_llm_error(
                actor=self.agent_id,
                phase="agent_response",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            if strict_no_fallback:
                raise ChatClientError(
                    f"{self.agent_id} agent LLM response failed and local fallback is disabled."
                ) from exc
            response = self._local_fallback_response(
                query=query,
                turn_number=turn_number,
                portfolio=portfolio,
                knowledge_slice=knowledge_slice,
                opinion_id=opinion_id,
                depth=depth,
                reason=f"agent_response failed: {type(exc).__name__}",
            )

        if self._is_low_signal_response(response):
            try:
                repaired_raw_response = self._repair_agent_response(
                    raw_response,
                    original_user_prompt=user_prompt,
                )
                response = self._normalize_llm_response(
                    repaired_raw_response,
                    query=query,
                    turn_number=turn_number,
                    portfolio=portfolio,
                    knowledge_slice=knowledge_slice,
                    opinion_id=opinion_id,
                    depth=depth,
                )
            except (ChatClientError, ValueError, TypeError) as exc:
                if strict_no_fallback:
                    raise ChatClientError(
                        f"{self.agent_id} agent LLM repair failed and local fallback is disabled."
                    ) from exc
                response = self._local_fallback_response(
                    query=query,
                    turn_number=turn_number,
                    portfolio=portfolio,
                    knowledge_slice=knowledge_slice,
                    opinion_id=opinion_id,
                    depth=depth,
                    reason=f"agent_response_repair failed: {type(exc).__name__}",
                )
        if self._is_low_signal_response(response):
            if strict_no_fallback:
                raise ChatClientError(
                    f"{self.agent_id} agent response remained sparse and local fallback is disabled."
                )
            response = self._local_fallback_response(
                query=query,
                turn_number=turn_number,
                portfolio=portfolio,
                knowledge_slice=knowledge_slice,
                opinion_id=opinion_id,
                depth=depth,
                reason="agent_response remained sparse after validation",
            )
        response = sanitize_agent_response_payload(
            response.to_dict(),
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            turn_number=turn_number,
            opinion_id=opinion_id,
            depth=depth,
        )
        return response

    def _normalize_llm_response(
        self,
        raw_response: Mapping[str, Any],
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_slice: KnowledgeSlice,
        opinion_id: str,
        depth: str,
    ) -> AgentResponse:
        response = sanitize_agent_response_payload(
            raw_response,
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            turn_number=turn_number,
            opinion_id=opinion_id,
            depth=depth,
        )
        response.agent_id = self.agent_id
        response.opinion_id = response.opinion_id or opinion_id
        response.turn_number = turn_number
        response.query_understood = response.query_understood or query
        response.direction = clamp(response.direction, -1.0, 1.0)
        response.strength = clamp(response.strength, 0.0, 1.0)
        response.confidence = clamp(response.confidence, 0.0, 1.0)
        response.depth_used = depth
        response.tools_called = knowledge_slice.tools_called
        self._annotate_observation_counts(response, knowledge_slice)
        if not response.evidence:
            response.evidence = {
                "events": [event.to_dict() for event in knowledge_slice.events[:4]],
                "documents": [document.to_dict() for document in knowledge_slice.documents[:3]],
            }
        if (
            response.confidence <= 0
            and response.reasoning_for_judge_agent.strip()
            and self._has_substantive_evidence(response.evidence)
        ):
            # A zero confidence with cited evidence is a malformed scalar, not an LLM outage.
            response.confidence = 0.35
        elif (
            response.confidence <= 0
            and response.reasoning_for_judge_agent.strip()
            and self._observed_count(knowledge_slice) > 0
        ):
            # The agent may correctly report zero portfolio-relevant evidence while still
            # using local observations. Keep the low confidence but do not treat it as an
            # LLM outage that needs a fallback summary.
            response.confidence = 0.2
        if not response.focus_tickers:
            focus_tickers = set()
            for event in knowledge_slice.events:
                focus_tickers.update(event.matched_holdings)
            for document in knowledge_slice.documents:
                focus_tickers.update(document.matched_holdings)
            response.focus_tickers = sorted(focus_tickers)
        return response

    def _repair_agent_response(
        self,
        invalid_payload: Mapping[str, Any],
        *,
        original_user_prompt: str,
    ) -> dict[str, Any]:
        repair_payload = {
            "invalid_response": dict(invalid_payload),
            "validator_error": (
                "The previous agent response was rejected because it did not contain enough "
                "usable reasoning, confidence, or signal for Judge to trust it."
            ),
            "repair_rules": [
                "Return only one JSON object with the exact normal agent response keys.",
                "Do not invent external data; use only the original supplied observations.",
                "Always fill reasoning_for_judge_agent with one or two Korean sentences.",
                "If the observations are outside this agent's scope, say so explicitly and keep direction and strength at 0.",
                "If at least one supplied observation was considered, confidence must be at least 0.2; describe uncertainty in limits_acknowledged.",
                "If there is truly no usable observation, return QUIET or DIRECT_ANSWER_UNAVAILABLE with a clear Korean reason.",
            ],
            "original_task": original_user_prompt,
        }
        system_prompt = (
            self._system_prompt()
            + " The previous JSON failed validation. Repair it without changing the supplied evidence."
        )
        user_prompt = json.dumps(repair_payload, ensure_ascii=False, separators=(",", ":"))
        publish_llm_prompt(
            actor=self.agent_id,
            phase="agent_response_repair",
            model=str(getattr(self.client, "model", "unknown")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        try:
            response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except (ChatClientError, ValueError, TypeError) as exc:
            publish_llm_error(
                actor=self.agent_id,
                phase="agent_response_repair",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            raise
        publish_llm_response(
            actor=self.agent_id,
            phase="agent_response_repair",
            model=str(getattr(self.client, "model", "unknown")),
            output=response,
        )
        return response

    def _local_fallback_response(
        self,
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_slice: KnowledgeSlice,
        opinion_id: str,
        depth: str,
        reason: str,
    ) -> AgentResponse:
        event_count = len(knowledge_slice.events)
        document_count = len(knowledge_slice.documents)
        observed_count = self._observed_count(knowledge_slice)
        portfolio_relevant_count = self._portfolio_relevant_count(knowledge_slice)
        has_evidence = bool(event_count or document_count)
        evidence = self._fallback_evidence(knowledge_slice)
        fallback_message = (
            "LLM 응답이 검증 기준을 통과하지 못해 로컬 근거 기반 자동 요약 폴백을 사용합니다."
            if "remained sparse" in reason
            else "LLM 응답 실패로 로컬 근거 기반 자동 요약 폴백을 사용합니다."
        )
        publish_llm_skipped(
            actor=self.agent_id,
            phase="agent_response_fallback",
            reason=fallback_message,
            context={
                "fallback_reason": reason,
                "events": event_count,
                "documents": document_count,
                "observed_count": observed_count,
                "portfolio_relevant_count": portfolio_relevant_count,
            },
        )
        if has_evidence:
            verdict = AgentVerdict.PARTIAL_ANSWER
        elif self.agent_id == "news":
            verdict = AgentVerdict.QUIET
        else:
            verdict = AgentVerdict.DIRECT_ANSWER_UNAVAILABLE
        raw_response = {
            "agent_id": self.agent_id,
            "opinion_id": opinion_id,
            "turn_number": turn_number,
            "query_understood": query,
            "verdict": verdict.value,
            "evidence": evidence,
            "direction": 0.0,
            "strength": 0.0,
            "urgency": Urgency.DEFER.value,
            "confidence": 0.25 if has_evidence else 0.2,
            "reasoning_for_judge_agent": (
                f"로컬 관찰은 이벤트 {event_count}건과 문서 {document_count}건이며, "
                f"이 중 현재 포트폴리오와 직접 매칭되는 근거는 {portfolio_relevant_count}건입니다. "
                "자동 요약 폴백으로 대체합니다. "
                "최종 판단에는 근거 존재 여부와 에이전트 범위 제한만 참고하세요."
            ),
            "limits_acknowledged": (
                f"{fallback_message} reason={reason}; "
                "본문 해석이나 투자 방향성 평가는 수행하지 않았습니다."
            ),
            "references": [],
            "focus_tickers": [],
        }
        response = sanitize_agent_response_payload(
            raw_response,
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            turn_number=turn_number,
            opinion_id=opinion_id,
            depth=depth,
        )
        response.tools_called = knowledge_slice.tools_called
        response.depth_used = depth
        return response

    def _observed_count(self, knowledge_slice: KnowledgeSlice) -> int:
        return len(knowledge_slice.events) + len(knowledge_slice.documents)

    def _portfolio_relevant_count(self, knowledge_slice: KnowledgeSlice) -> int:
        return sum(1 for event in knowledge_slice.events if event.matched_holdings) + sum(
            1 for document in knowledge_slice.documents if document.matched_holdings
        )

    def _annotate_observation_counts(
        self, response: AgentResponse, knowledge_slice: KnowledgeSlice
    ) -> None:
        if self.agent_id not in {"disclosure", "news", "report"}:
            return
        observed_count = self._observed_count(knowledge_slice)
        portfolio_relevant_count = self._portfolio_relevant_count(knowledge_slice)
        existing_observed = response.evidence.get("observed_count")
        if isinstance(existing_observed, (int, float)):
            observed_count = max(observed_count, int(existing_observed))
        response.evidence["observed_count"] = observed_count
        response.evidence["portfolio_relevant_count"] = portfolio_relevant_count
        response.evidence["usable_for_trade_decision"] = portfolio_relevant_count > 0

    def _fallback_evidence(self, knowledge_slice: KnowledgeSlice) -> dict[str, Any]:
        observed_count = self._observed_count(knowledge_slice)
        portfolio_relevant_count = self._portfolio_relevant_count(knowledge_slice)
        if self.agent_id == "disclosure":
            items = [self._fallback_event_item(event) for event in knowledge_slice.events[:6]]
            if not items:
                items = [
                    self._fallback_document_item(document)
                    for document in knowledge_slice.documents[:6]
                ]
            return {
                "found_count": portfolio_relevant_count,
                "observed_count": observed_count,
                "portfolio_relevant_count": portfolio_relevant_count,
                "usable_for_trade_decision": portfolio_relevant_count > 0,
                "items": items,
                "upcoming_disclosures": [],
            }
        if self.agent_id == "news":
            headlines = [
                item
                for item in (
                    [
                        truncate(event.headline or event.summary, 160)
                        for event in knowledge_slice.events[:6]
                    ]
                    + [
                        truncate(document.title or document.body, 160)
                        for document in knowledge_slice.documents[:4]
                    ]
                )
                if item
            ]
            return {
                "sub_role": "mixed",
                "observed_count": observed_count,
                "portfolio_relevant_count": portfolio_relevant_count,
                "usable_for_trade_decision": portfolio_relevant_count > 0,
                "company_findings": {
                    "portfolio": {
                        "sentiment": "neutral",
                        "key_headlines": headlines[:5],
                        "market_reaction": None,
                        "sector_comparison": None,
                    }
                },
                "macro_findings": {
                    "events_count": len(knowledge_slice.events),
                    "documents_count": len(knowledge_slice.documents),
                    "headlines": headlines[:8],
                },
                "source_reliability": "medium",
                "cross_check_count": 0,
            }
        if self.agent_id == "report":
            items = [
                self._fallback_document_item(document) for document in knowledge_slice.documents[:6]
            ]
            return {
                "coverage_reports_count": len(items),
                "preview_reports_count": 0,
                "items": items,
                "consensus": None,
            }
        if self.agent_id == "profit":
            return {
                "plan_simulation": {
                    "rebalance_plan": {},
                    "ticker_signals": {},
                    "expected_return_1m": 0,
                    "expected_return_3m": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "recommendation_text": "LLM 실패로 수익성 시뮬레이션을 수행하지 못했습니다.",
                }
            }
        if self.agent_id == "cost":
            return {
                "trade_cost": {
                    "rebalance_plan": {},
                    "commission_krw": 0,
                    "tax_krw": 0,
                    "estimated_slippage_bp": 0,
                    "spread_state_bp": 0,
                    "total_friction_bp": 0,
                }
            }
        return {
            "events": [event.to_dict() for event in knowledge_slice.events[:4]],
            "documents": [document.to_dict() for document in knowledge_slice.documents[:3]],
        }

    def _fallback_event_item(self, event: KnowledgeEvent) -> dict[str, Any]:
        ticker = next(iter(event.matched_holdings), None)
        primary_entity = event.entities[0] if event.entities else None
        return {
            "ticker": ticker or (primary_entity.ticker if primary_entity else None),
            "company_name": primary_entity.entity_name if primary_entity else None,
            "type": event.event_type,
            "headline": event.headline,
            "summary": event.summary,
            "date": event.event_time.isoformat(),
        }

    def _fallback_document_item(self, document: KnowledgeDocument) -> dict[str, Any]:
        ticker = next(iter(document.matched_holdings), None)
        primary_entity = document.entities[0] if document.entities else None
        return {
            "ticker": ticker or (primary_entity.ticker if primary_entity else None),
            "company_name": primary_entity.entity_name if primary_entity else None,
            "type": document.event_type or document.doc_type,
            "headline": document.title,
            "summary": truncate(document.body, 320),
            "date": document.published_at.isoformat(),
            "broker": document.publisher or document.source_name,
            "publisher": document.publisher or document.source_name,
            "matched_holdings": list(document.matched_holdings),
            "key_thesis": truncate(document.body or document.title, 320),
            "published_at": document.published_at.isoformat(),
            "report_type": document.doc_type.casefold() or "other",
        }

    def _run_tool_loop(
        self,
        *,
        query: str,
        context: str | None,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str,
    ) -> AgentToolLoopResult:
        tools_called: list[ToolCall] = [
            ToolCall(
                tool_name=f"{self.agent_id}.observe_request",
                purpose="요청, 포트폴리오, 직전 Judge 맥락 관찰",
                summary=(
                    f"보유 종목 {len(portfolio.holdings)}개와 요청 '{truncate(query, 80)}'를 보고 "
                    f"{self.agent_id} 에이전트가 필요한 근거 조회를 계획했습니다."
                ),
            )
        ]
        primary = knowledge_base.slice_for_agent(
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            depth=depth,
        )
        tools_called.extend(primary.tools_called)
        observed = self._with_tools(primary, tools_called)
        tools_called.append(
            ToolCall(
                tool_name=f"{self.agent_id}.observe_tool_result",
                purpose="1차 근거 조회 결과 관찰",
                summary=f"1차 조회 결과 이벤트 {len(observed.events)}건, 문서 {len(observed.documents)}건을 관찰했습니다.",
            )
        )

        stop_reason = "primary_evidence_sufficient"
        if self._should_expand_search(observed, depth=depth):
            tools_called.append(
                ToolCall(
                    tool_name=f"{self.agent_id}.replan_search",
                    purpose="근거 부족 또는 얕은 조회 결과에 따른 추가 행동 선택",
                    summary="1차 관찰만으로 충분하지 않아 deep 범위로 같은 책임 영역을 재조회합니다.",
                )
            )
            expanded = knowledge_base.slice_for_agent(
                agent_id=self.agent_id,
                portfolio=portfolio,
                query=self._expanded_query(query=query, context=context),
                depth="deep",
            )
            observed = self._merge_slices(observed, expanded, tools_called + expanded.tools_called)
            tools_called = observed.tools_called
            tools_called.append(
                ToolCall(
                    tool_name=f"{self.agent_id}.observe_tool_result",
                    purpose="추가 근거 조회 결과 관찰",
                    summary=f"추가 조회 후 누적 이벤트 {len(observed.events)}건, 문서 {len(observed.documents)}건을 확보했습니다.",
                )
            )
            observed = self._with_tools(observed, tools_called)
            stop_reason = "expanded_search_completed"

        if self._should_refresh_ingest(observed):
            refresh_result = knowledge_base.refresh_from_ingest(agent_id=self.agent_id)
            tools_called = list(observed.tools_called) + [refresh_result.tool_call]
            if refresh_result.changed:
                refreshed_slice = knowledge_base.slice_for_agent(
                    agent_id=self.agent_id,
                    portfolio=portfolio,
                    query=self._expanded_query(query=query, context=context),
                    depth="deep",
                )
                observed = self._merge_slices(
                    observed, refreshed_slice, tools_called + refreshed_slice.tools_called
                )
                tools_called = observed.tools_called
                tools_called.append(
                    ToolCall(
                        tool_name=f"{self.agent_id}.observe_ingest_refresh",
                        purpose="ingest refresh 이후 새 근거 관찰",
                        summary=(
                            f"refresh 이후 누적 이벤트 {len(observed.events)}건, "
                            f"문서 {len(observed.documents)}건을 관찰했습니다."
                        ),
                    )
                )
                observed = self._with_tools(observed, tools_called)
                stop_reason = "ingest_refresh_completed"
            else:
                observed = self._with_tools(observed, tools_called)
                stop_reason = "ingest_refresh_unavailable"

        if self._should_scan_portfolio(observed):
            signals, tool_call = knowledge_base.portfolio_signal_scan(portfolio)
            tools_called = list(observed.tools_called) + [tool_call]
            signal_summary = (
                ", ".join(
                    f"{ticker}:{score:.2f}"
                    for ticker, score in signals.items()
                    if abs(score) >= 0.01
                )
                or "뚜렷한 종목별 방향성 없음"
            )
            tools_called.append(
                ToolCall(
                    tool_name=f"{self.agent_id}.observe_portfolio_scan",
                    purpose="보유 종목 방향성 스캔 결과 관찰",
                    summary=f"포트폴리오 스캔 관찰 결과: {signal_summary}.",
                )
            )
            observed = self._with_tools(observed, tools_called)
            stop_reason = "portfolio_scan_completed"

        tools_called = list(observed.tools_called) + [
            ToolCall(
                tool_name=f"{self.agent_id}.stop",
                purpose="하위 에이전트 도구 루프 종료 판단",
                summary=(
                    f"stop_reason={stop_reason}; 최종 근거 이벤트 {len(observed.events)}건, "
                    f"문서 {len(observed.documents)}건으로 Judge에게 응답합니다."
                ),
            )
        ]
        return AgentToolLoopResult(
            knowledge_slice=self._with_tools(observed, tools_called),
            stop_reason=stop_reason,
        )

    def _should_expand_search(self, knowledge_slice: KnowledgeSlice, *, depth: str) -> bool:
        if depth == "deep":
            return False
        if not knowledge_slice.events and not knowledge_slice.documents:
            return True
        if self.agent_id == "report" and not knowledge_slice.documents:
            return True
        if (
            self.agent_id == "news"
            and len(knowledge_slice.events) + len(knowledge_slice.documents) <= 1
        ):
            return True
        return False

    def _should_refresh_ingest(self, knowledge_slice: KnowledgeSlice) -> bool:
        if knowledge_slice.events or knowledge_slice.documents:
            return False
        return self.agent_id in {"disclosure", "news", "report"}

    def _should_scan_portfolio(self, knowledge_slice: KnowledgeSlice) -> bool:
        if knowledge_slice.events or knowledge_slice.documents:
            return self.agent_id in {"news", "report"}
        return True

    def _expanded_query(self, *, query: str, context: str | None) -> str:
        parts = [query]
        if context:
            parts.append(context)
        parts.append("보유 종목 전체 관련 단서")
        return " | ".join(part for part in parts if part)

    def _with_tools(
        self, knowledge_slice: KnowledgeSlice, tools_called: list[ToolCall]
    ) -> KnowledgeSlice:
        return KnowledgeSlice(
            events=list(knowledge_slice.events),
            documents=list(knowledge_slice.documents),
            tools_called=list(tools_called),
        )

    def _merge_slices(
        self,
        first: KnowledgeSlice,
        second: KnowledgeSlice,
        tools_called: list[ToolCall],
    ) -> KnowledgeSlice:
        events: list[KnowledgeEvent] = []
        event_ids: set[str] = set()
        for event in [*first.events, *second.events]:
            key = event.event_id or stable_hash(event.to_dict())
            if key in event_ids:
                continue
            event_ids.add(key)
            events.append(event)

        documents: list[KnowledgeDocument] = []
        document_ids: set[str] = set()
        for document in [*first.documents, *second.documents]:
            key = document.doc_id or stable_hash(document.to_dict())
            if key in document_ids:
                continue
            document_ids.add(key)
            documents.append(document)

        return KnowledgeSlice(events=events, documents=documents, tools_called=list(tools_called))

    def _is_low_signal_response(self, response: AgentResponse) -> bool:
        if response.confidence > 0 and response.reasoning_for_judge_agent.strip():
            return False
        if response.direction != 0 or response.strength != 0:
            return False
        if response.verdict not in {
            AgentVerdict.PARTIAL_ANSWER,
            AgentVerdict.DIRECT_ANSWER_UNAVAILABLE,
            AgentVerdict.QUIET,
        }:
            return False
        return True

    def _has_substantive_evidence(self, evidence: Mapping[str, Any]) -> bool:
        if not isinstance(evidence, Mapping):
            return False
        found_count = evidence.get("found_count")
        if isinstance(found_count, (int, float)) and found_count > 0:
            return True
        for key in (
            "items",
            "events",
            "documents",
            "reports",
            "company_findings",
            "source_documents",
        ):
            value = evidence.get(key)
            if isinstance(value, list) and value:
                return True
        return False

    def _system_prompt(self) -> str:
        return self.prompt_profile.system_prompt

    def _user_prompt(
        self,
        *,
        query: str,
        context: str | None,
        fallback: str | None,
        note: str | None,
        portfolio: PortfolioSnapshot,
        knowledge_slice: KnowledgeSlice,
        depth: str,
        turn_number: int,
    ) -> str:
        title_limit, body_limit, event_summary_limit = self._snippet_limits(depth)
        sections = [
            f"agent={self.agent_id}",
            f"turn={turn_number}",
            f"depth={depth}",
            f"query={query}",
        ]
        if context:
            sections.append(f"context={context}")
        if fallback:
            sections.append(f"fallback={fallback}")
        if note:
            sections.append(f"note={note}")
        sections.append("agent_tool_observations:")
        sections.extend(
            f"- {tool.tool_name}: {tool.summary}" for tool in knowledge_slice.tools_called
        )
        sections.append("portfolio:")
        sections.extend(
            f"- {holding['ticker']} {holding['company_name']} weight={holding['weight']}"
            for holding in self._compact_portfolio(portfolio)["holdings"]
        )
        sections.append("preferences:")
        sections.extend(
            f"- {item}" for item in self._compact_portfolio(portfolio)["user_preferences"]
        )
        sections.append("events:")
        for event in knowledge_slice.events:
            sections.append(
                "- "
                f"{event.event_type} "
                f"tickers={','.join(event.matched_holdings) or 'none'} "
                f"headline={truncate(event.headline, title_limit)} "
                f"summary={truncate(event.summary, event_summary_limit)}"
            )
        sections.append("documents:")
        for document in knowledge_slice.documents:
            sections.append(
                "- "
                f"{document.doc_type} "
                f"tickers={','.join(document.matched_holdings) or 'none'} "
                f"title={truncate(document.title, title_limit)} "
                f"excerpt={truncate(document.body, body_limit)}"
            )
        guidance = self._agent_guidance()
        sections.append(f"focus={guidance['focus']}")
        sections.append(
            f"evidence_hint={json.dumps(guidance['evidence_shape_hint'], ensure_ascii=False, separators=(',', ':'))}"
        )
        sections.append(
            "return_json="
            + json.dumps(self._response_template(), ensure_ascii=False, separators=(",", ":"))
        )
        return "\n".join(sections)

    def _snippet_limits(self, depth: str) -> tuple[int, int, int]:
        if depth == "deep":
            return (90, 160, 120)
        if depth == "medium":
            return (80, 120, 90)
        return (72, 84, 68)

    def _compact_portfolio(self, portfolio: PortfolioSnapshot) -> dict[str, Any]:
        return {
            "generated_at": portfolio.generated_at.isoformat(),
            "holdings": [
                {
                    "ticker": holding.ticker,
                    "company_name": holding.company_name,
                    "weight": round(float(holding.weight), 4),
                }
                for holding in portfolio.holdings
            ],
            "cash_weight": round(float(portfolio.cash_weight), 4),
            "user_preferences": list(portfolio.user_preferences[:4]),
        }

    def _agent_guidance(self) -> dict[str, Any]:
        return {
            "focus": self.prompt_profile.focus,
            "evidence_shape_hint": dict(self.prompt_profile.evidence_shape_hint),
        }

    def _response_template(self) -> dict[str, Any]:
        return {
            **dict(self.prompt_profile.response_template),
            "evidence": dict(self.prompt_profile.evidence_shape_hint),
        }


class CoreChatDomainRouter:
    """Expose the active Judge ChatClient through the domain-agent router API."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def ask(
        self,
        *,
        agent_id: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cross_validate: bool = False,
    ) -> str:
        del max_tokens, cross_validate
        payload = {
            "agent_id": agent_id,
            "domain_system_prompt": system,
            "domain_user_prompt": user,
            "instructions": (
                "Execute the domain agent task. Return JSON with exactly one key, "
                "rationale, containing the answer text."
            ),
        }
        system_prompt = (
            "You are a LIBRA domain-agent LLM adapter. "
            "Follow domain_system_prompt and domain_user_prompt, but return only "
            'one JSON object: {"rationale":"..."}. '
            "Write rationale in Korean unless the domain prompt explicitly asks otherwise."
        )
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        model_name = self.model_name_for(agent_id)
        publish_llm_prompt(
            actor=agent_id,
            phase="domain_agent_response",
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        try:
            response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except Exception as exc:
            publish_llm_error(
                actor=agent_id,
                phase="domain_agent_response",
                model=model_name,
                error=exc,
            )
            raise
        publish_llm_response(
            actor=agent_id,
            phase="domain_agent_response",
            model=model_name,
            output=response,
        )
        rationale = response.get("rationale") or response.get("answer") or response.get("summary")
        if isinstance(rationale, str) and rationale.strip():
            return rationale.strip()
        return json.dumps(response, ensure_ascii=False)

    def model_name_for(self, agent_id: str) -> str:
        del agent_id
        return str(getattr(self.client, "model", "active-chat-client"))


class JudgeOrchestrator:
    def __init__(self, *, client: ChatClient, checkpoint_path: str | Path | None = None) -> None:
        self.client = client
        from .libra.agents import build_default_agent_bundle

        agent_bundle = build_default_agent_bundle(client=client)
        self.disclosure_agent = agent_bundle.disclosure
        self.news_agent = agent_bundle.news
        self.report_agent = agent_bundle.report
        self.profit_agent = agent_bundle.profit
        self.cost_agent = agent_bundle.cost
        self.evaluation_agent = agent_bundle.evaluation
        self.domain_agents = agent_bundle.domain_agents()
        self.domain_router = CoreChatDomainRouter(client)
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        from .libra_graph import LibraLangGraphRuntime

        self._graph_runtime = LibraLangGraphRuntime(self)

    def _routing_agent_ids(self) -> tuple[str, ...]:
        return CORE_ROUTING_AGENT_IDS

    def _next_domain_agent(self, called_agents: set[str]) -> str | None:
        for agent_id in DOMAIN_ROUTING_AGENT_IDS:
            if agent_id in self.domain_agents and agent_id not in called_agents:
                return agent_id
        return None

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
        return self._graph_runtime.run(
            query=query,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            portfolio_definition=portfolio_definition,
            depth=depth,
            trigger=trigger,
            trigger_event=trigger_event,
            deadline_seconds=deadline_seconds,
            thread_id=thread_id,
            enable_human_interrupts=enable_human_interrupts,
        )

    def run_v1_committee(
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
        ips: Any | None = None,
        kyc: Any | None = None,
        market_data: Any | None = None,
    ) -> dict[str, Any]:
        """Run the v1 committee design as the primary Judge path.

        This path follows the v1 design docs: Compliance BEFORE, Round 1
        committee calls in parallel, Mediator target selection, optional Round
        2 targeted recall, Compliance AFTER, then code-locked Final Judge.
        """
        del thread_id, enable_human_interrupts
        started_at = datetime.now().astimezone()
        run_state = self._initialize_run_state(
            trigger=trigger,
            trigger_event=trigger_event,
            deadline_seconds=deadline_seconds,
        )
        drift_report = (
            compute_drift(portfolio_definition, portfolio) if portfolio_definition else None
        )
        candidate_plan = candidate_plan_from_drift(drift_report) if drift_report else {}
        if not candidate_plan and trigger == "push":
            candidate_plan = self._candidate_plan_from_trigger(
                portfolio=portfolio,
                trigger_event=trigger_event,
            )

        from .libra.committee import CommitteeRuntime

        agent_order = (
            *CORE_ROUTING_AGENT_IDS,
            *(
                agent_id
                for agent_id in DOMAIN_ROUTING_AGENT_IDS
                if agent_id != "compliance" and agent_id in self.domain_agents
            ),
        )
        round1_calls = {
            agent_id: self._v1_agent_call(
                agent_id=agent_id,
                query=query,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                candidate_plan=candidate_plan,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                turn_number=index,
                round2_context=None,
            )
            for index, agent_id in enumerate(agent_order, start=1)
        }

        def round2_factory(agent_id: str, round2_context: str):
            return self._v1_agent_call(
                agent_id=agent_id,
                query=query,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                candidate_plan=candidate_plan,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                turn_number=len(agent_order) + 1,
                round2_context=round2_context,
            )

        governance = CommitteeRuntime().run_from_agent_rounds(
            portfolio=portfolio,
            round1_agent_calls=round1_calls,
            round2_agent_call_factory=round2_factory,
            ips=ips,
            kyc=kyc,
            market_data=market_data,
            candidate_plan=candidate_plan,
            mediator_client=self.client,
            final_judge_client=self.client,
        )
        elapsed = (datetime.now().astimezone() - started_at).total_seconds()
        all_responses = [*governance.round1_responses, *governance.round2_responses]
        decision = self._decision_from_v1_governance(
            query=query,
            portfolio=portfolio,
            governance=governance,
            responses=all_responses,
            run_state=run_state,
            candidate_plan=candidate_plan,
            elapsed_seconds=elapsed,
        )
        return {
            "model": self.client.model,
            "query": query,
            "portfolio": portfolio.to_dict(),
            "agent_responses": [response.to_dict() for response in all_responses],
            "decision": decision.to_dict(),
            "knowledge_sources": dict(knowledge_base.source_paths),
            "governance_v1": governance.to_dict(),
            "direct_indexing": {
                "portfolio_definition": portfolio_definition.to_dict()
                if portfolio_definition
                else None,
                "drift_report": drift_report.to_dict() if drift_report else None,
                "candidate_rebalance_plan": dict(candidate_plan),
            },
            "runtime": {
                "engine": "governance_v1_committee",
                "round1_agent_count": len(governance.round1_opinions),
                "round2_agent_count": len(governance.round2_opinions),
            },
        }

    def resume(
        self,
        *,
        thread_id: str,
        resume_payload: Any,
    ) -> dict[str, Any]:
        return self._graph_runtime.resume(
            thread_id=thread_id,
            resume_payload=resume_payload,
        )

    def _v1_agent_call(
        self,
        *,
        agent_id: str,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        candidate_plan: Mapping[str, float],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        turn_number: int,
        round2_context: str | None,
    ):
        def call() -> AgentResponse:
            return self._execute_v1_committee_agent(
                agent_id=agent_id,
                query=query,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                candidate_plan=candidate_plan,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                turn_number=turn_number,
                round2_context=round2_context,
            )

        return call

    def _execute_v1_committee_agent(
        self,
        *,
        agent_id: str,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        candidate_plan: Mapping[str, float],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        turn_number: int,
        round2_context: str | None,
    ) -> AgentResponse:
        agent_id = canonical_agent_id(agent_id)
        context = self._default_agent_context(
            agent_id=agent_id,
            query=query,
            responses=[],
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        if round2_context:
            context = f"{context}\n\n[Round 2 targeted recall]\n{round2_context}"
        note = self._default_agent_note(
            agent_id=agent_id,
            query=query,
            responses=[],
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        note_prefix = (
            "v1 Committee Round 2 표적 재호출"
            if round2_context
            else "v1 Committee Round 1 독립 발화"
        )
        note = f"{note_prefix}. {note}" if note else note_prefix
        if agent_id in {"disclosure", "news", "report"}:
            agent = self._agent_by_id(agent_id)
            response = agent.run(
                query=self._default_agent_query(agent_id=agent_id, trigger=trigger),
                context=context,
                fallback=self._default_agent_fallback(agent_id=agent_id, trigger=trigger),
                note=note,
                turn_number=turn_number,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                depth=depth,
            )
        elif agent_id == "profit":
            response = self.profit_agent.run(
                query=query,
                turn_number=turn_number,
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                rebalance_plan=dict(candidate_plan),
            )
        elif agent_id == "cost":
            response = self.cost_agent.run(
                query=query,
                turn_number=turn_number,
                portfolio=portfolio,
                rebalance_plan=dict(candidate_plan),
            )
        elif agent_id in self.domain_agents:
            response = self._execute_v1_domain_agent(
                agent_id=agent_id,
                query=query,
                context=context,
                turn_number=turn_number,
                portfolio=portfolio,
                candidate_plan=candidate_plan,
            )
        else:
            raise ChatClientError(f"Unknown v1 committee agent: {agent_id}")
        return sanitize_agent_response_payload(
            response.to_dict(),
            agent_id=agent_id,
            portfolio=portfolio,
            query=query,
            turn_number=turn_number,
            opinion_id=response.opinion_id,
            depth=depth,
        )

    def _execute_v1_domain_agent(
        self,
        *,
        agent_id: str,
        query: str,
        context: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        candidate_plan: Mapping[str, float],
    ) -> AgentResponse:
        from libra_agent.domain_agents._adapter import (
            _run_async,
            domain_verdict_to_agent_response,
            portfolio_snapshot_to_domain_context,
        )

        proposed_trades = candidate_plan_to_proposed_trades(candidate_plan)
        ctx = portfolio_snapshot_to_domain_context(
            portfolio,
            user_id="libra",
            proposed_trades=proposed_trades,
            market_context_str=context,
        )
        ctx.router = getattr(self, "domain_router", None)
        verdict = _run_async(self.domain_agents[agent_id].deliberate(ctx))
        return domain_verdict_to_agent_response(
            verdict,
            agent_id=agent_id,
            turn_number=turn_number,
            query=query,
        )

    def _decision_from_v1_governance(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        governance: Any,
        responses: list[AgentResponse],
        run_state: RunState,
        candidate_plan: Mapping[str, float],
        elapsed_seconds: float,
    ) -> JudgeDecision:
        final = governance.final_decision
        trade_plan = {
            trade.subject: round(float(trade.delta_pct) / 100.0, 6)
            for trade in final.trades
            if abs(float(trade.delta_pct)) >= 0.1
        }
        called_agents = [response.agent_id for response in responses]
        skip_rationale = {
            "compliance": "Compliance는 LLM 위원회가 아니라 BEFORE/AFTER 코드 룰 엔진으로 실행되었습니다."
        }
        options = [option.label for option in (final.user_options or [])]
        urgency = (
            Urgency.WATCH
            if final.decision.value == "USER_DECISION_REQUIRED"
            else Urgency.SCHEDULED
            if final.decision.value == "REBALANCE"
            else Urgency.DEFER
        )
        consensus_values = [
            abs(score.weighted_score) for score in governance.consensus_per_subject.values()
        ]
        consensus_score = max(consensus_values, default=0.0)
        conflict_count = sum(
            1
            for score in governance.consensus_per_subject.values()
            if score.branch.value in {"CONFLICT", "INSUFFICIENT_VOTES"}
        )
        divergence_score = conflict_count / max(1, len(governance.consensus_per_subject))
        trace = [
            {
                "turn_number": 0,
                "phase": DecisionPhase.INFORMATION_GATHERING.value,
                "actor": "Compliance Rule Engine",
                "query": "BEFORE check",
                "summary": "현재 포트폴리오가 사용자 IPS/KYC hard rule을 위반하는지 코드로 검사했습니다.",
                "context": governance.compliance_before.state,
                "note": "LLM 호출 없음",
            },
            {
                "turn_number": 1,
                "phase": DecisionPhase.DELIBERATION.value,
                "actor": "Round 1 Committee",
                "query": "11 LLM agent parallel review",
                "summary": f"{len(governance.round1_opinions)}개 LLM 에이전트가 독립 의견을 제출했습니다.",
                "context": ", ".join(opinion.agent for opinion in governance.round1_opinions),
            },
            {
                "turn_number": 2,
                "phase": DecisionPhase.CONSENSUS.value,
                "actor": "Mediator Judge",
                "query": "conflict selection",
                "summary": governance.mediator_decision.rationale,
                "context": f"targets={governance.mediator_decision.targets_to_recall}",
            },
        ]
        if governance.round2_opinions:
            trace.append(
                {
                    "turn_number": 3,
                    "phase": DecisionPhase.DELIBERATION.value,
                    "actor": "Round 2 Targeted Recall",
                    "query": "targeted re-check",
                    "summary": f"{len(governance.round2_opinions)}개 충돌 에이전트를 표적 재호출했습니다.",
                    "context": ", ".join(opinion.agent for opinion in governance.round2_opinions),
                }
            )
        trace.append(
            {
                "turn_number": 4,
                "phase": DecisionPhase.DECISION.value,
                "actor": "Final Judge",
                "query": final.branch.value,
                "summary": final.reasoning,
                "context": f"decision={final.decision.value}",
            }
        )
        summary = f"{final.decision.value}: {final.reasoning}"
        notification = UserNotification(
            level=self._notification_level(DecisionType(final.decision.value), urgency),
            body=final.user_question or summary,
            action_required=final.decision.value == "USER_DECISION_REQUIRED",
            kind="governance_v1_final_decision",
            estimated_followup=None,
            sent_at=self._notification_timestamp(),
        )
        decision = JudgeDecision.from_dict(
            {
                "decision": final.decision.value,
                "summary": summary,
                "confidence": min(1.0, max(0.5, consensus_score)),
                "urgency": urgency.value,
                "called_agents": called_agents,
                "skipped_agents": ["compliance"],
                "skip_rationale": skip_rationale,
                "candidate_rebalance_plan": trade_plan,
                "decision_trace": trace,
                "reasoning": final.reasoning,
                "user_notification": notification.to_dict(),
                "consensus_score": consensus_score,
                "divergence_score": divergence_score,
                "needs_trade_evaluation": bool(candidate_plan or trade_plan),
                "trigger": run_state.trigger,
                "trigger_event": run_state.trigger_event.to_dict()
                if run_state.trigger_event
                else None,
                "deadline_at": run_state.deadline_at.isoformat(timespec="seconds")
                if run_state.deadline_at
                else None,
                "elapsed_seconds": elapsed_seconds,
                "options": options,
                "auto_safeguards": {
                    "governance_v1_branch": final.branch.value,
                    "compliance_after": governance.compliance_after.to_dict(),
                },
                "notification_log": [
                    item.to_dict() for item in [*run_state.notification_log, notification]
                ],
            }
        )
        return decision

    def _initialize_run_state(
        self,
        *,
        trigger: str,
        trigger_event: TriggerEvent | None,
        deadline_seconds: int | None,
    ) -> RunState:
        started_at = datetime.now().astimezone()
        if deadline_seconds is None and trigger == "push":
            deadline_seconds = 300
        deadline_at = (
            (started_at + timedelta(seconds=deadline_seconds)) if deadline_seconds else None
        )
        notification_log: list[UserNotification] = []
        if trigger == "push" and trigger_event is not None:
            notification_log.append(
                self._acknowledgement_notification(
                    trigger_event=trigger_event,
                    deadline_at=deadline_at,
                    sent_at=started_at,
                )
            )
        return RunState(
            trigger=trigger,
            trigger_event=trigger_event,
            started_at=started_at,
            deadline_at=deadline_at,
            notification_log=notification_log,
        )

    def _agent_by_id(self, agent_id: str) -> LLMAgent:
        agent_id = canonical_agent_id(agent_id)
        return {
            "disclosure": self.disclosure_agent,
            "news": self.news_agent,
            "report": self.report_agent,
            **self.domain_agents,
        }[agent_id]

    def _should_skip_agent(
        self,
        *,
        planned_call: PlannedAgentCall,
        original_query: str,
        depth: str,
        trigger: str,
        responses: list[AgentResponse],
    ) -> bool:
        del original_query
        if trigger == "push":
            return False
        if canonical_agent_id(planned_call.agent_id) != "report":
            return False
        if depth != "shallow":
            return False
        if not responses:
            return False
        low_signal = all(
            abs(response.direction) < 0.08
            and response.urgency in {Urgency.DEFER, Urgency.SCHEDULED}
            for response in responses
        )
        return low_signal

    def _candidate_plan_from_trigger(
        self,
        *,
        portfolio: PortfolioSnapshot,
        trigger_event: TriggerEvent | None,
    ) -> dict[str, float]:
        if trigger_event is None or not trigger_event.ticker:
            return {}
        holdings = {normalize_ticker(item.ticker): item for item in portfolio.holdings}
        target = holdings.get(normalize_ticker(trigger_event.ticker))
        if target is None:
            return {}
        text = "\n".join(
            part
            for part in (
                trigger_event.headline,
                trigger_event.summary or "",
                trigger_event.market_reaction or "",
            )
            if part
        ).casefold()
        is_risk_event = any(token in text for token in PUSH_RISK_KEYWORDS)
        if not is_risk_event:
            return {target.ticker: 0.05}

        plan: dict[str, float] = {target.ticker: -0.1}
        other_holdings = sorted(
            (
                holding
                for holding in portfolio.holdings
                if normalize_ticker(holding.ticker) != normalize_ticker(target.ticker)
            ),
            key=lambda item: float(item.weight),
            reverse=True,
        )
        for holding in other_holdings[:2]:
            plan[holding.ticker] = -0.05
        return plan

    def _trade_agent_order(
        self,
        *,
        query: str,
        planning: Mapping[str, Any],
        trigger: str,
    ) -> tuple[str, ...]:
        lowered = query.casefold()
        if (
            trigger == "push"
            or planning.get("decision") == DecisionType.USER_DECISION_REQUIRED.value
        ):
            return ("cost", "profit")
        if any(token in lowered for token in ("비용", "세금", "슬리피지", "실행")):
            return ("cost", "profit")
        return ("profit", "cost")

    def _judge_next_action(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        already_called = [canonical_agent_id(item) for item in called_agents]
        fast_finalize = self._empty_portfolio_fast_finalize_action(
            query=query,
            portfolio=portfolio,
            responses=responses,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        if fast_finalize is not None:
            publish_llm_skipped(
                actor="judge",
                phase="core_routing",
                reason="빈 포트폴리오 점검 fast-path로 Core 라우팅 LLM 호출을 생략합니다.",
                context={
                    "holdings": 0,
                    "candidate_rebalance_plan": {},
                    "called_agents": list(called_agents),
                },
            )
            publish_debate_event(
                "judge_action",
                _compact_judge_action_event(
                    fast_finalize,
                    layer="core",
                    turn_number=len(responses) + 1,
                    called_agents=already_called,
                    response_count=len(responses),
                ),
            )
            return fast_finalize
        valid_next_agents = [
            agent_id
            for agent_id in self._routing_agent_ids()
            if agent_id not in set(already_called)
        ]
        payload = {
            "query": query,
            "trigger": trigger,
            "trigger_event": trigger_event.to_dict() if trigger_event else None,
            "depth": depth,
            "called_agents": list(called_agents),
            "already_called_agent_values": already_called,
            "valid_next_agent_values": valid_next_agents,
            "candidate_rebalance_plan": dict(candidate_plan or {}),
            "portfolio": {
                "holdings": [
                    {
                        "ticker": item.ticker,
                        "company_name": item.company_name,
                        "weight": round(float(item.weight), 4),
                    }
                    for item in portfolio.holdings
                ],
                "user_preferences": list(portfolio.user_preferences[:4]),
            },
            "agent_responses": [self._compact_agent_response(response) for response in responses],
            "instructions": {
                "action_values": ["CALL_AGENT", "FINALIZE"],
                "agent_values": list(self._routing_agent_ids()),
                "depth_values": ["shallow", "medium", "deep"],
                "rules": JUDGE_ACTION_RULES,
                "validator_contract": [
                    "agent_id must be one of the exact lowercase agent_values.",
                    "agent_id must be one of valid_next_agent_values unless action is FINALIZE.",
                    "CALL_AGENT profit or cost is rejected when candidate_rebalance_plan is empty.",
                    "candidate_rebalance_plan must use portfolio tickers with nonzero weight deltas.",
                    "If no safe next agent is justified, use FINALIZE.",
                ],
            },
        }
        system_prompt = JUDGE_ACTION_SYSTEM_PROMPT
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publish_llm_prompt(
            actor="judge",
            phase="core_routing",
            model=str(getattr(self.client, "model", "unknown")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        try:
            raw = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except ChatClientError as exc:
            publish_llm_error(
                actor="judge",
                phase="core_routing",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            raise ChatClientError(
                "Judge routing LLM failed; deterministic routing fallback is disabled."
            ) from exc
        publish_llm_response(
            actor="judge",
            phase="core_routing",
            model=str(getattr(self.client, "model", "unknown")),
            output=raw,
        )
        normalized = self._normalize_judge_action(
            raw,
            query=query,
            portfolio=portfolio,
            responses=responses,
            called_agents=called_agents,
            depth=depth,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        initial_raw = dict(raw)
        if normalized is None:
            raw = self._repair_judge_action(raw, context_payload=payload)
            normalized = self._normalize_judge_action(
                raw,
                query=query,
                portfolio=portfolio,
                responses=responses,
                called_agents=called_agents,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                candidate_plan=candidate_plan,
            )
        if normalized is None:
            detail = self._judge_action_rejection_detail(
                raw,
                initial_payload=initial_raw,
                called_agents=called_agents,
                candidate_plan=candidate_plan,
            )
            raise ChatClientError(
                f"Judge routing LLM returned an invalid or unsafe next action. {detail}"
            )
        publish_debate_event(
            "judge_action",
            _compact_judge_action_event(
                normalized,
                layer="core",
                turn_number=len(responses) + 1,
                called_agents=already_called,
                response_count=len(responses),
            ),
        )
        return normalized

    def _empty_portfolio_fast_finalize_action(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None,
    ) -> dict[str, Any] | None:
        if (
            trigger != "pull"
            or trigger_event is not None
            or responses
            or portfolio.holdings
            or candidate_plan
        ):
            return None
        lowered = query.casefold()
        exploration_tokens = (
            "초기",
            "후보",
            "구성",
            "설계",
            "추천",
            "관심",
            "종목",
            "시장",
            "뉴스",
            "공시",
            "기회",
            "탐색",
            "찾",
            "만들",
            "편입",
            "매수",
            "매도",
            "리밸런싱",
            "리밸런스",
            "비중",
            "분산",
        )
        if any(token in lowered for token in exploration_tokens):
            return None
        check_tokens = ("점검", "유지", "조정", "리뷰", "확인", "판단")
        if not any(token in lowered for token in check_tokens):
            return None
        return {
            "action": "FINALIZE",
            "reason": (
                "보유 종목과 후보 리밸런싱 초안이 없어 실행 가능한 매수·매도 조정이 없습니다. "
                "초기 포트폴리오 후보가 생성된 뒤 투자 검토를 진행합니다."
            ),
            "note": (
                "빈 포트폴리오의 단순 유지·조정 점검 요청이므로 공시·뉴스 조회 없이 종료합니다. "
                "시장 스캔이나 초기 후보 생성을 명시한 요청은 별도 정보 수집 경로로 처리합니다."
            ),
            "candidate_rebalance_plan": {},
        }

    def _repair_judge_action(
        self,
        invalid_payload: Mapping[str, Any],
        *,
        context_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        already_called = [
            canonical_agent_id(item) for item in context_payload.get("called_agents", [])
        ]
        all_agents = [
            canonical_agent_id(item)
            for item in (context_payload.get("instructions", {}) or {}).get("agent_values", [])
        ]
        valid_next_agents = [
            agent_id for agent_id in all_agents if agent_id not in set(already_called)
        ]
        repair_payload = {
            "invalid_response": dict(invalid_payload),
            "invalid_agent_id": canonical_agent_id(str(invalid_payload.get("agent_id", ""))),
            "already_called_agent_values": already_called,
            "valid_next_agent_values": valid_next_agents,
            "validator_error": (
                "The previous action was rejected by LIBRA's safety validator. "
                "Return a corrected JSON action only; do not explain outside JSON."
            ),
            "repair_rules": [
                "Use action CALL_AGENT or FINALIZE only.",
                "For CALL_AGENT, use exact lowercase agent_id values from valid_next_agent_values only.",
                "Do not call an already-called agent.",
                "CALL_AGENT profit or cost is invalid if candidate_rebalance_plan is empty.",
                "If you want profit or cost, include a concrete nonempty candidate_rebalance_plan with portfolio tickers and nonzero weight deltas.",
                "If you wanted a domain agent such as risk, tax, compliance, macro, sentiment, execution, esg, liquidity, or technical, choose FINALIZE so the separate domain council layer can route it.",
                "If the desired agent is already called or no valid trade-review action exists, choose FINALIZE.",
            ],
            "state": context_payload,
        }
        system_prompt = (
            JUDGE_ACTION_SYSTEM_PROMPT
            + " The previous JSON was rejected by the validator; repair it into a valid safe action."
        )
        user_prompt = json.dumps(repair_payload, ensure_ascii=False, separators=(",", ":"))
        publish_llm_prompt(
            actor="judge",
            phase="core_routing_repair",
            model=str(getattr(self.client, "model", "unknown")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        try:
            response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except ChatClientError as exc:
            publish_llm_error(
                actor="judge",
                phase="core_routing_repair",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            raise ChatClientError(
                "Judge routing LLM repair failed; deterministic routing fallback is disabled."
            ) from exc
        publish_llm_response(
            actor="judge",
            phase="core_routing_repair",
            model=str(getattr(self.client, "model", "unknown")),
            output=response,
        )
        return response

    def _judge_action_rejection_detail(
        self,
        payload: Mapping[str, Any],
        *,
        initial_payload: Mapping[str, Any],
        called_agents: list[str],
        candidate_plan: Mapping[str, float] | None,
    ) -> str:
        already_called = [canonical_agent_id(item) for item in called_agents]
        valid_next_agents = [
            agent_id
            for agent_id in self._routing_agent_ids()
            if agent_id not in set(already_called)
        ]

        def _compact(raw: Mapping[str, Any]) -> dict[str, Any]:
            raw_plan = raw.get("candidate_rebalance_plan")
            return {
                "action": truncate(str(raw.get("action", "")), 80),
                "agent_id": truncate(str(raw.get("agent_id", "")), 80),
                "normalized_agent_id": canonical_agent_id(str(raw.get("agent_id", ""))),
                "candidate_plan_keys": list(raw_plan.keys())[:8]
                if isinstance(raw_plan, Mapping)
                else [],
            }

        detail = {
            "initial": _compact(initial_payload),
            "repaired": _compact(payload),
            "already_called": already_called,
            "valid_next_agents": valid_next_agents,
            "candidate_plan_keys": list((candidate_plan or {}).keys())[:8],
        }
        return "detail=" + json.dumps(detail, ensure_ascii=False, separators=(",", ":"))

    def _domain_next_action(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        normalized_plan = self._draft_candidate_plan(
            portfolio=portfolio,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        if not portfolio.holdings and not normalized_plan:
            normalized = {
                "action": "FINALIZE_DOMAIN_REVIEW",
                "reason": (
                    "보유 종목과 후보 리밸런싱 초안이 없어 도메인 심의 대상이 없습니다. "
                    "유동성·기술·체결·세금 검토는 구체적인 종목 또는 주문 후보가 생긴 뒤 수행합니다."
                ),
                "candidate_rebalance_plan": {},
                "layer": "domain",
            }
            publish_debate_event(
                "judge_action",
                _compact_judge_action_event(
                    normalized,
                    layer="domain",
                    turn_number=len(responses) + 1,
                    called_agents=[canonical_agent_id(item) for item in called_agents],
                    response_count=len(responses),
                ),
            )
            return normalized

        payload = {
            "query": query,
            "trigger": trigger,
            "trigger_event": trigger_event.to_dict() if trigger_event else None,
            "depth": depth,
            "called_agents": list(called_agents),
            "called_domain_agents": [
                agent_id
                for agent_id in called_agents
                if canonical_agent_id(agent_id) in DOMAIN_ROUTING_AGENT_IDS
            ],
            "candidate_rebalance_plan": dict(normalized_plan),
            "portfolio": {
                "holdings": [
                    {
                        "ticker": item.ticker,
                        "company_name": item.company_name,
                        "weight": round(float(item.weight), 4),
                    }
                    for item in portfolio.holdings
                ],
                "user_preferences": list(portfolio.user_preferences[:4]),
            },
            "agent_responses": [self._compact_agent_response(response) for response in responses],
            "instructions": {
                "action_values": ["CALL_AGENT", "FINALIZE_DOMAIN_REVIEW"],
                "agent_values": [
                    agent_id
                    for agent_id in DOMAIN_ROUTING_AGENT_IDS
                    if agent_id in self.domain_agents
                ],
                "depth_values": ["medium", "deep"],
                "rules": [
                    "Domain council agents review the Judge candidate decision; they do not gather first-layer facts.",
                    "Choose at most one domain agent per round, then wait for its response.",
                    "Do not call a domain agent that already answered.",
                    "If every useful domain view has answered, choose FINALIZE_DOMAIN_REVIEW.",
                    "Use Compliance for policy constraints, Risk for concentration/downside exposure, Liquidity for ADV/spread/free-float constraints, Technical for price/volume momentum, Execution for market impact, Tax for tax effects, Macro for regime risk, Sentiment for market mood, and ESG for sustainability constraints.",
                ],
            },
        }
        system_prompt = JUDGE_DOMAIN_ACTION_SYSTEM_PROMPT
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publish_llm_prompt(
            actor="judge",
            phase="domain_routing",
            model=str(getattr(self.client, "model", "unknown")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        try:
            raw = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except ChatClientError as exc:
            publish_llm_error(
                actor="judge",
                phase="domain_routing",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            raise ChatClientError(
                "Domain council routing LLM failed; deterministic routing fallback is disabled."
            ) from exc
        publish_llm_response(
            actor="judge",
            phase="domain_routing",
            model=str(getattr(self.client, "model", "unknown")),
            output=raw,
        )
        normalized = self._normalize_domain_action(
            raw,
            query=query,
            portfolio=portfolio,
            responses=responses,
            called_agents=called_agents,
            depth=depth,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        if normalized is None:
            raise ChatClientError(
                "Domain council routing LLM returned an invalid or unsafe next action."
            )
        publish_debate_event(
            "judge_action",
            _compact_judge_action_event(
                normalized,
                layer="domain",
                turn_number=len(responses) + 1,
                called_agents=[canonical_agent_id(item) for item in called_agents],
                response_count=len(responses),
            ),
        )
        return normalized

    def _normalize_judge_action(
        self,
        payload: Mapping[str, Any],
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None,
    ) -> dict[str, Any] | None:
        action = str(payload.get("action", "")).strip().upper()
        if action not in {"CALL_AGENT", "FINALIZE"}:
            return None
        raw_candidate_plan = payload.get("candidate_rebalance_plan")
        incoming_candidate_plan = (
            raw_candidate_plan
            if isinstance(raw_candidate_plan, Mapping) and raw_candidate_plan
            else candidate_plan
        )
        normalized_plan = self._draft_candidate_plan(
            portfolio=portfolio,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=incoming_candidate_plan,
        )
        result: dict[str, Any] = {
            "action": action,
            "reason": str(payload.get("reason", "")).strip(),
            "candidate_rebalance_plan": normalized_plan,
        }
        if action == "FINALIZE":
            if not portfolio.holdings and not normalized_plan:
                result["reason"] = (
                    "보유 종목과 후보 리밸런싱 초안이 없어 실행 가능한 매수·매도 조정이 없습니다. "
                    "초기 포트폴리오 후보가 생성된 뒤 투자 검토를 진행합니다."
                )
            return result

        agent_id = canonical_agent_id(str(payload.get("agent_id", "")))
        allowed_agent_ids = set(self._routing_agent_ids())
        if agent_id not in allowed_agent_ids:
            return None
        called_set = {canonical_agent_id(item) for item in called_agents}
        if agent_id in called_set:
            return None
        action_depth = str(payload.get("depth", depth)).strip().lower()
        if action_depth not in {"shallow", "medium", "deep"}:
            action_depth = depth

        if agent_id in {"profit", "cost"} and not normalized_plan:
            normalized_plan = self._draft_candidate_plan(
                portfolio=portfolio,
                trigger=trigger,
                trigger_event=trigger_event,
                candidate_plan=candidate_plan,
            )
            if not normalized_plan:
                return None
            result["candidate_rebalance_plan"] = normalized_plan
        disclosure_response = next(
            (item for item in responses if canonical_agent_id(item.agent_id) == "disclosure"),
            None,
        )
        normalized_call_depth = (
            action_depth if agent_id in {"disclosure", "news", "report"} else "medium"
        )

        result.update(
            {
                "agent_id": agent_id,
                "query": self._default_agent_query(
                    agent_id=agent_id,
                    trigger=trigger,
                    disclosure_response=disclosure_response,
                    responses=responses,
                ),
                "context": self._default_agent_context(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": normalized_call_depth,
                "fallback": self._default_agent_fallback(agent_id=agent_id, trigger=trigger),
                "note": self._default_agent_note(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
            }
        )
        return result

    def _normalize_domain_action(
        self,
        payload: Mapping[str, Any],
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None,
    ) -> dict[str, Any] | None:
        action = str(payload.get("action", "")).strip().upper()
        if action not in {"CALL_AGENT", "FINALIZE_DOMAIN_REVIEW"}:
            return None
        normalized_plan = self._draft_candidate_plan(
            portfolio=portfolio,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        result: dict[str, Any] = {
            "action": action,
            "reason": str(payload.get("reason", "")).strip(),
            "candidate_rebalance_plan": normalized_plan,
            "layer": "domain",
        }
        enabled_domain_ids = {
            agent_id for agent_id in DOMAIN_ROUTING_AGENT_IDS if agent_id in self.domain_agents
        }
        if not enabled_domain_ids:
            result["action"] = "FINALIZE_DOMAIN_REVIEW"
            result["reason"] = (
                result["reason"] or "활성화된 도메인 에이전트가 없어 도메인 심의를 건너뜁니다."
            )
            return result
        if action == "FINALIZE_DOMAIN_REVIEW":
            return result

        agent_id = canonical_agent_id(str(payload.get("agent_id", "")))
        if agent_id not in enabled_domain_ids:
            return None
        called_set = {canonical_agent_id(item) for item in called_agents}
        if agent_id in called_set:
            return None
        action_depth = str(payload.get("depth", "medium")).strip().lower()
        if action_depth not in {"medium", "deep"}:
            action_depth = "medium"

        result.update(
            {
                "agent_id": agent_id,
                "query": self._default_agent_query(
                    agent_id=agent_id, trigger=trigger, responses=responses
                ),
                "context": self._default_agent_context(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": action_depth,
                "fallback": self._default_agent_fallback(agent_id=agent_id, trigger=trigger),
                "note": self._default_agent_note(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
            }
        )
        return result

    def _draft_candidate_plan(
        self,
        *,
        portfolio: PortfolioSnapshot,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if isinstance(candidate_plan, Mapping):
            sanitized = self._sanitize_plan(candidate_plan, portfolio)
            if sanitized:
                return sanitized
        if trigger == "push":
            push_plan = self._candidate_plan_from_trigger(
                portfolio=portfolio,
                trigger_event=trigger_event,
            )
            if push_plan:
                return push_plan
        return {}

    def _default_agent_query(
        self,
        *,
        agent_id: str,
        trigger: str,
        disclosure_response: AgentResponse | None = None,
        responses: list[AgentResponse] | None = None,
    ) -> str:
        del responses
        return default_agent_query(
            agent_id=agent_id,
            trigger=trigger,
            has_disclosure_context=disclosure_response is not None,
        )

    def _default_agent_context(
        self,
        *,
        agent_id: str,
        query: str,
        responses: list[AgentResponse],
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> str:
        context_parts: list[str] = []
        trigger_context = self._trigger_context_text(trigger_event)
        if trigger_context:
            context_parts.append(trigger_context)
        if responses:
            latest = responses[-1]
            context_parts.append(
                f"직전 {latest.agent_id} 관찰: {truncate(latest.reasoning_for_judge_agent or latest.query_understood, 180)}"
            )
        if agent_id in {"profit", "cost"}:
            context_parts.append("판단 에이전트는 현재 리밸런싱 초안을 실행할지 검토하고 있습니다.")
            if candidate_plan:
                context_parts.append(f"후보 리밸런싱 초안: {dict(candidate_plan)}")
        else:
            context_parts.append(f"원 사용자 요청: {query}")
            if agent_id in DOMAIN_ROUTING_AGENT_IDS and candidate_plan:
                context_parts.append(f"후보 리밸런싱 초안: {dict(candidate_plan)}")
        return " | ".join(part for part in context_parts if part)

    def _default_agent_fallback(self, *, agent_id: str, trigger: str) -> str | None:
        return default_agent_fallback(agent_id=agent_id, trigger=trigger)

    def _default_agent_note(
        self,
        *,
        agent_id: str,
        query: str,
        responses: list[AgentResponse],
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> str | None:
        del query, trigger_event
        latest = responses[-1] if responses else None
        return default_agent_note(
            agent_id=agent_id,
            latest_agent_id=canonical_agent_id(latest.agent_id) if latest is not None else None,
            trigger=trigger,
            has_candidate_plan=bool(candidate_plan),
        )

    def _recommended_news_depth(
        self,
        *,
        default_depth: str,
        disclosure_response: AgentResponse | None,
    ) -> str:
        if disclosure_response is None:
            return default_depth
        text = f"{disclosure_response.reasoning_for_judge_agent}\n{disclosure_response.limits_acknowledged or ''}".casefold()
        if any(
            token in text for token in ("어닝", "earnings", "실적", "화재", "규제", "조사", "리콜")
        ):
            return "deep"
        return default_depth

    def _has_rebalance_intent(self, query: str) -> bool:
        lowered = query.casefold()
        return any(
            token in lowered
            for token in ("리밸런싱", "리밸런스", "rebalance", "초안", "비중", "매수", "매도")
        )

    def _is_explicit_report_request(self, query: str) -> bool:
        lowered = query.casefold()
        return any(
            token in lowered for token in ("리포트", "report", "컨센서스", "증권사", "목표주가")
        )

    def _is_explicit_news_request(self, query: str) -> bool:
        lowered = query.casefold()
        return any(
            token in lowered
            for token in (
                "뉴스",
                "기사",
                "보도",
                "속보",
                "장중",
                "시장 반응",
                "breaking",
                "headline",
                "급락",
                "급등",
            )
        )

    def _initial_pull_agent(self, query: str) -> str:
        if self._is_explicit_news_request(query):
            return "news"
        if self._is_explicit_report_request(query):
            return "report"
        return "disclosure"

    def _push_prescreen_is_sufficient(self, trigger_event: TriggerEvent | None) -> bool:
        if trigger_event is None:
            return False
        if not (trigger_event.ticker or trigger_event.company_name or trigger_event.headline):
            return False
        text = f"{trigger_event.headline} {trigger_event.summary or ''}".casefold()
        cross_check_count = int(trigger_event.cross_check_count or 0)
        return (
            cross_check_count >= 2
            or bool(trigger_event.market_reaction)
            or any(
                token in text
                for token in (
                    "조사",
                    "규제",
                    "리콜",
                    "화재",
                    "안전",
                    "investigation",
                    "recall",
                    "probe",
                    "fire",
                )
            )
        )

    def _should_finalize_after_basic_scan(
        self,
        *,
        query: str,
        depth: str,
        disclosure_response: AgentResponse | None,
        news_response: AgentResponse | None,
        candidate_plan: Mapping[str, float],
    ) -> bool:
        if disclosure_response is None or news_response is None:
            return False
        if candidate_plan:
            return False
        if self._is_explicit_report_request(query):
            return False
        if self._should_call_report(
            query=query,
            depth=depth,
            disclosure_response=disclosure_response,
            news_response=news_response,
        ):
            return False
        responses = (disclosure_response, news_response)
        reasoning = "\n".join(
            collapse_whitespace(
                f"{response.reasoning_for_judge_agent} {response.limits_acknowledged or ''}"
            ).casefold()
            for response in responses
        )
        if any(
            token in reasoning
            for token in (
                "리포트 필요",
                "리포트 확인",
                "report needed",
                "call report",
                "컨센서스",
                "preview",
                "추가 확인",
                "추가 정보",
            )
        ):
            return False
        max_signal = max(abs(response.direction) for response in responses)
        min_confidence = min(response.confidence for response in responses)
        if max_signal < 0.08 and all(response.urgency == Urgency.DEFER for response in responses):
            return True
        if max_signal < 0.14 and min_confidence >= 0.5:
            return True
        if depth == "shallow" and max_signal < 0.18 and min_confidence >= 0.45:
            return True
        return False

    def _should_call_report(
        self,
        *,
        query: str,
        depth: str,
        disclosure_response: AgentResponse | None,
        news_response: AgentResponse | None,
    ) -> bool:
        if self._is_explicit_report_request(query):
            return True
        if disclosure_response is None or news_response is None:
            return False
        responses = (disclosure_response, news_response)
        for response in responses:
            if response is None:
                continue
            reasoning = f"{response.reasoning_for_judge_agent}\n{response.limits_acknowledged or ''}".casefold()
            if any(
                token in reasoning
                for token in (
                    "리포트 필요",
                    "리포트 확인",
                    "report needed",
                    "call report",
                    "사업부",
                    "컨센서스",
                    "추가 정보",
                    "preview",
                )
            ):
                return True
        directions = [response.direction for response in responses]
        confidences = [response.confidence for response in responses]
        if any(
            response.verdict == AgentVerdict.DIRECT_ANSWER_UNAVAILABLE for response in responses
        ):
            if depth == "deep" and max(abs(value) for value in directions) >= 0.12:
                return True
            if max(abs(value) for value in directions) >= 0.18:
                return True
            return False
        if directions[0] * directions[1] < -0.01 and abs(directions[0] - directions[1]) >= 0.14:
            return True
        if max(abs(value) for value in directions) >= 0.25 and min(confidences) < 0.55:
            return True
        if (
            depth == "deep"
            and max(abs(value) for value in directions) >= 0.2
            and min(confidences) < 0.7
        ):
            return True
        return False

    def _infer_skip_rationale(
        self,
        *,
        query: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        called_agents: list[str],
        responses: list[AgentResponse],
        candidate_plan: Mapping[str, float],
        depth: str,
    ) -> dict[str, str]:
        called_set = {canonical_agent_id(item) for item in called_agents}
        rationale: dict[str, str] = {}
        response_map = {canonical_agent_id(item.agent_id): item for item in responses}

        if trigger == "push" and self._push_prescreen_is_sufficient(trigger_event):
            for agent_id in ("disclosure", "news", "report"):
                if agent_id not in called_set:
                    rationale[agent_id] = (
                        "속보 트리거에 사전 확인 정보가 포함되어 추가 정보 수집을 건너뛰었습니다."
                    )

        if trigger == "pull":
            disclosure_response = response_map.get("disclosure")
            news_response = response_map.get("news")
            if (
                "report" not in called_set
                and disclosure_response is not None
                and news_response is not None
            ):
                if self._should_finalize_after_basic_scan(
                    query=query,
                    depth=depth,
                    disclosure_response=disclosure_response,
                    news_response=news_response,
                    candidate_plan=candidate_plan,
                ):
                    rationale["report"] = "이번 판단에는 공시와 뉴스 점검만으로 충분했습니다."
                elif not self._should_call_report(
                    query=query,
                    depth=depth,
                    disclosure_response=disclosure_response,
                    news_response=news_response,
                ):
                    rationale["report"] = (
                        "이번 판단에서 증권사 해석을 추가로 확인할 근거가 부족했습니다."
                    )

        if not candidate_plan:
            if "profit" not in called_set:
                rationale["profit"] = "수익과 위험을 평가할 구체적인 리밸런싱 초안이 없었습니다."
            if "cost" not in called_set:
                rationale["cost"] = "실행 초안이 없어 거래비용과 유동성 점검이 필요하지 않았습니다."

        return rationale

    def _notification_timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return coerce_datetime(text)
        except (TypeError, ValueError):
            return None

    def _coerce_datetime_or_now(self, value: Any) -> datetime:
        return self._coerce_datetime(value) or datetime.now().astimezone()

    def _deadline_exceeded(self, run_state: RunState) -> bool:
        if run_state.deadline_at is None:
            return False
        return datetime.now().astimezone(run_state.deadline_at.tzinfo) > run_state.deadline_at

    def _elapsed_seconds(self, run_state: RunState) -> float:
        return max(0.0, (datetime.now().astimezone() - run_state.started_at).total_seconds())

    def _deadline_result(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        called_agents: list[str],
        skipped_agents: list[str],
        skip_rationale: dict[str, str],
        executed_calls: list[PlannedAgentCall],
        responses: list[AgentResponse],
        run_state: RunState,
        candidate_plan: Mapping[str, float] | None = None,
        judge_actions: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_plan = dict(candidate_plan or {})
        summary = "분석 deadline을 넘겨 자동 판단보다 사용자 확인이 우선인 상태로 전환합니다."
        decision = JudgeDecision(
            decision=DecisionType.USER_DECISION_REQUIRED,
            summary=summary,
            confidence=0.55,
            urgency=Urgency.WATCH,
            called_agents=list(called_agents),
            skipped_agents=list(skipped_agents),
            skip_rationale=dict(skip_rationale),
            candidate_rebalance_plan=normalized_plan,
            reasoning="Self-imposed deadline expired before the orchestration finished.",
            follow_up_at=self._default_follow_up_at(query=query, responses=responses),
            consensus_score=0.0,
            divergence_score=0.0,
            needs_trade_evaluation=bool(normalized_plan),
            trigger=run_state.trigger,
            trigger_event=run_state.trigger_event,
            deadline_at=run_state.deadline_at.isoformat(timespec="seconds")
            if run_state.deadline_at
            else None,
            elapsed_seconds=self._elapsed_seconds(run_state),
            options=["권고안 승인", "전량 매도", "유지", "직접 비율 설정"]
            if run_state.trigger == "push"
            else [],
            auto_safeguards=self._build_auto_safeguards(
                portfolio=portfolio,
                trigger_event=run_state.trigger_event,
                candidate_plan=normalized_plan,
            ),
        )
        self._apply_push_guardrails(
            decision=decision,
            portfolio=portfolio,
            run_state=run_state,
        )
        final_notification = UserNotification(
            level="push",
            body=decision.summary,
            action_required=True,
            kind="deadline_expired",
            estimated_followup=decision.follow_up_at,
            sent_at=self._notification_timestamp(),
        )
        decision.user_notification = final_notification
        decision.notification_log = list(run_state.notification_log) + [final_notification]
        decision.decision_trace = self._decision_trace(
            query=query,
            executed_calls=executed_calls,
            judge_actions=judge_actions,
            responses=responses,
            decision=decision,
        )
        return {
            "model": self.client.model,
            "query": query,
            "portfolio": portfolio.to_dict(),
            "agent_responses": [response.to_dict() for response in responses],
            "decision": decision.to_dict(),
            "knowledge_sources": dict(knowledge_base.source_paths),
        }

    def _apply_push_guardrails(
        self,
        *,
        decision: JudgeDecision,
        portfolio: PortfolioSnapshot,
        run_state: RunState,
    ) -> None:
        if run_state.trigger != "push" or run_state.trigger_event is None:
            return
        trigger_event = run_state.trigger_event
        if not trigger_event.ticker:
            return
        normalized_ticker = normalize_ticker(trigger_event.ticker)
        if normalized_ticker not in {normalize_ticker(item.ticker) for item in portfolio.holdings}:
            return
        text = "\n".join(
            part
            for part in (
                trigger_event.headline,
                trigger_event.summary or "",
                trigger_event.market_reaction or "",
            )
            if part
        ).casefold()
        is_risk_event = (
            any(token in text for token in PUSH_RISK_KEYWORDS)
            or trigger_event.cross_check_count >= 3
        )
        if not is_risk_event:
            return
        label = trigger_event.company_name or trigger_event.ticker or "해당 종목"
        decision.decision = DecisionType.USER_DECISION_REQUIRED
        decision.urgency = Urgency.WATCH
        decision.summary = f"{label} 관련 규제·안전 리스크 이벤트는 사전 위임 범위를 벗어나므로 사용자 확인이 먼저 필요합니다."
        decision.reasoning = "Push-triggered risk event crossed LIBRA's autonomy boundary, so the Judge is handing authority back to the user."
        decision.options = ["권고안 승인", "전량 매도", "유지", "직접 비율 설정"]
        decision.auto_safeguards = self._build_auto_safeguards(
            portfolio=portfolio,
            trigger_event=trigger_event,
            candidate_plan=decision.candidate_rebalance_plan,
        )

    def _trigger_context_text(self, trigger_event: TriggerEvent | None) -> str:
        if trigger_event is None:
            return ""
        parts = [
            f"Triggered by {trigger_event.source or trigger_event.trigger_type}: {trigger_event.headline}",
        ]
        if trigger_event.company_name:
            parts.append(f"company={trigger_event.company_name}")
        if trigger_event.ticker:
            parts.append(f"ticker={trigger_event.ticker}")
        if trigger_event.summary:
            parts.append(f"summary={trigger_event.summary}")
        if trigger_event.cross_check_count:
            parts.append(f"cross_check_count={trigger_event.cross_check_count}")
        if trigger_event.market_reaction:
            parts.append(f"market_reaction={trigger_event.market_reaction}")
        return " | ".join(parts)

    def _build_auto_safeguards(
        self,
        *,
        portfolio: PortfolioSnapshot,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float],
    ) -> dict[str, Any]:
        if trigger_event is None or not trigger_event.ticker:
            return dict(candidate_plan=candidate_plan)
        ticker = normalize_ticker(trigger_event.ticker)
        if ticker not in {normalize_ticker(item.ticker) for item in portfolio.holdings}:
            return dict(candidate_plan=candidate_plan)
        return {
            "tripwire_1": f"{trigger_event.ticker} 장중 -5% 시 보유 수량의 절반 자동 축소",
            "tripwire_2": f"{trigger_event.ticker} 장중 -8% 시 잔여 보유분 추가 축소",
            "candidate_plan": dict(candidate_plan),
        }

    def _acknowledgement_notification(
        self,
        *,
        trigger_event: TriggerEvent,
        deadline_at: datetime | None,
        sent_at: datetime,
    ) -> UserNotification:
        label = trigger_event.company_name or trigger_event.ticker or "이벤트"
        if deadline_at is not None:
            local_deadline = deadline_at.astimezone().strftime("%H:%M")
            body = f"{label} 관련 이벤트 감지. {local_deadline} 전후로 분석 결과를 알려드립니다. 액션은 잠시 보류해 주세요."
        else:
            body = f"{label} 관련 이벤트 감지. 분석 중이며, 결과가 준비되면 바로 알려드립니다."
        return UserNotification(
            level="watch",
            body=body,
            action_required=False,
            kind="event_acknowledged",
            estimated_followup=deadline_at.isoformat(timespec="seconds") if deadline_at else None,
            sent_at=sent_at.isoformat(timespec="seconds"),
        )

    def _judge_phase(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        stage: str,
        candidate_plan: Mapping[str, float] | None = None,
        drift_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        deterministic_payload = self._empty_portfolio_final_decision_payload(
            portfolio=portfolio,
            stage=stage,
            candidate_plan=candidate_plan,
            drift_report=drift_report,
        )
        if deterministic_payload is not None:
            publish_llm_skipped(
                actor="judge",
                phase=f"final_decision_{stage}",
                reason="빈 포트폴리오 no-trade fast-path로 최종 판단 LLM 호출을 생략합니다.",
                context={
                    "holdings": 0,
                    "candidate_rebalance_plan": {},
                    "direct_indexing": None,
                },
            )
            return deterministic_payload

        response_payloads = [self._compact_agent_response(response) for response in responses]
        prompt = json.dumps(
            {
                "stage": stage,
                "query": query,
                "portfolio": {
                    "holdings": [
                        {
                            "ticker": holding.ticker,
                            "company_name": holding.company_name,
                            "weight": round(float(holding.weight), 4),
                        }
                        for holding in portfolio.holdings
                    ],
                    "user_preferences": list(portfolio.user_preferences[:4]),
                },
                "agent_responses": response_payloads,
                "direct_indexing": self._compact_direct_indexing_context(
                    candidate_plan=candidate_plan,
                    drift_report=drift_report,
                ),
                "instructions": {
                    "required_keys": JUDGE_PHASE_REQUIRED_KEYS,
                    "decision_values": [item.value for item in DecisionType],
                    "urgency_values": [item.value for item in Urgency],
                    "notification_levels": JUDGE_NOTIFICATION_LEVELS,
                    "plan_format": {"ticker": "weight_delta"},
                    "notes": [
                        "Use only supplied agent responses.",
                        "Keep summary concise and write all natural-language text only in Korean.",
                        "Do not use Japanese kana in summary, reasoning, notifications, or options.",
                        "If no trade is justified, return an empty candidate_rebalance_plan object.",
                        "Do not translate zero local evidence, empty caches, QUIET, or DIRECT_ANSWER_UNAVAILABLE into a claim that the market is quiet or stable.",
                        "The HOLD-over-DEFER preference does not apply when holdings and candidate_rebalance_plan are both empty.",
                        "If holdings and candidate_rebalance_plan are both empty, decision must be DEFER and state that there is no executable trade and an initial portfolio candidate is needed before investment review.",
                        "When there is no executable trade and no action is required, set follow_up_at to null.",
                        "A direct_indexing candidate_rebalance_plan is a target-vs-current drift draft, not a free-form heuristic.",
                        "If a direct_indexing candidate plan exists and profit/cost agents did not block it, REBALANCE may be justified even when disclosure/news are quiet.",
                    ],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_prompt = JUDGE_PHASE_SYSTEM_PROMPT
        publish_llm_prompt(
            actor="judge",
            phase=f"final_decision_{stage}",
            model=str(getattr(self.client, "model", "unknown")),
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.0,
        )
        try:
            payload = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.0,
            )
        except ChatClientError as exc:
            publish_llm_error(
                actor="judge",
                phase=f"final_decision_{stage}",
                model=str(getattr(self.client, "model", "unknown")),
                error=exc,
            )
            raise ChatClientError(
                "Judge final-decision LLM failed; deterministic decision fallback is disabled."
            ) from exc
        payload = sanitize_judge_payload(payload, portfolio=portfolio, stage=stage)
        publish_llm_response(
            actor="judge",
            phase=f"final_decision_{stage}",
            model=str(getattr(self.client, "model", "unknown")),
            output=payload,
        )
        if self._is_low_signal_judge_payload(
            payload
        ) or self._judge_payload_has_unsupported_language(payload):
            raise ChatClientError(
                "Judge final-decision LLM returned an invalid or unsupported-language payload."
            )
        return payload

    def _empty_portfolio_final_decision_payload(
        self,
        *,
        portfolio: PortfolioSnapshot,
        stage: str,
        candidate_plan: Mapping[str, float] | None,
        drift_report: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stage != "final" or portfolio.holdings or candidate_plan or drift_report:
            return None
        return sanitize_judge_payload(
            {
                "decision": DecisionType.DEFER.value,
                "summary": EMPTY_PORTFOLIO_NO_TRADE_SUMMARY,
                "confidence": 0.95,
                "urgency": Urgency.DEFER.value,
                "reasoning": EMPTY_PORTFOLIO_NO_TRADE_REASONING,
                "candidate_rebalance_plan": {},
                "needs_trade_evaluation": False,
                "follow_up_at": None,
                "feedback_checkpoint": None,
                "user_notification": {
                    "level": "info",
                    "body": EMPTY_PORTFOLIO_NO_TRADE_SUMMARY,
                    "action_required": False,
                    "kind": "final_decision",
                    "estimated_followup": None,
                    "sent_at": None,
                },
                "options": [],
                "auto_safeguards": {},
            },
            portfolio=portfolio,
            stage=stage,
        )

    def _compact_direct_indexing_context(
        self,
        *,
        candidate_plan: Mapping[str, float] | None,
        drift_report: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not candidate_plan and not drift_report:
            return None
        return {
            "candidate_rebalance_plan": dict(candidate_plan or {}),
            "drift": compact_drift_context(drift_report),
        }

    def _compact_agent_response(self, response: AgentResponse) -> dict[str, Any]:
        evidence = response.evidence
        evidence_summary: dict[str, Any]
        if response.agent_id == "disclosure":
            evidence_summary = {
                "found_count": evidence.get("found_count", 0),
                "observed_count": evidence.get("observed_count", evidence.get("found_count", 0)),
                "portfolio_relevant_count": evidence.get(
                    "portfolio_relevant_count", evidence.get("found_count", 0)
                ),
                "usable_for_trade_decision": evidence.get("usable_for_trade_decision", False),
                "upcoming_disclosures": evidence.get("upcoming_disclosures", []),
                "items": [
                    {
                        "ticker": item.get("ticker"),
                        "disclosure_type": item.get("disclosure_type"),
                        "timestamp": item.get("timestamp"),
                    }
                    for item in evidence.get("items", [])[:3]
                    if isinstance(item, Mapping)
                ],
            }
        elif response.agent_id == "report":
            evidence_summary = {
                "coverage_reports_count": evidence.get("coverage_reports_count", 0),
                "preview_reports_count": evidence.get("preview_reports_count", 0),
                "items": [
                    {
                        "broker": item.get("broker"),
                        "report_type": item.get("report_type"),
                        "matched_holdings": item.get("matched_holdings"),
                    }
                    for item in evidence.get("items", [])[:3]
                    if isinstance(item, Mapping)
                ],
            }
        elif response.agent_id == "news":
            company_findings = evidence.get("company_findings")
            evidence_summary = {
                "sub_role": evidence.get("sub_role"),
                "observed_count": evidence.get("observed_count", 0),
                "portfolio_relevant_count": evidence.get("portfolio_relevant_count", 0),
                "usable_for_trade_decision": evidence.get("usable_for_trade_decision", False),
                "company_findings": list(company_findings.keys())[:4]
                if isinstance(company_findings, Mapping)
                else [],
                "cross_check_count": evidence.get("cross_check_count", 0),
            }
        else:
            evidence_summary = {
                "mode": evidence.get("mode"),
                "keys": list(evidence.keys())[:6],
            }
        return {
            "agent_id": response.agent_id,
            "verdict": response.verdict.value,
            "direction": response.direction,
            "strength": response.strength,
            "urgency": response.urgency.value,
            "confidence": response.confidence,
            "signal_score": response.signal_score,
            "source_trust": response.source_trust,
            "event_type": response.event_type,
            "horizon": response.horizon,
            "risk_level": response.risk_level,
            "opinion": response.opinion,
            "reasoning_for_judge_agent": truncate(response.reasoning_for_judge_agent, 220),
            "limits_acknowledged": truncate(response.limits_acknowledged or "", 140) or None,
            "focus_tickers": list(response.focus_tickers),
            "evidence_summary": evidence_summary,
        }

    def _sanitize_plan(
        self,
        candidate_plan: Mapping[str, Any],
        portfolio: PortfolioSnapshot,
    ) -> dict[str, float]:
        allowed = {
            normalize_ticker(holding.ticker): holding.ticker for holding in portfolio.holdings
        }
        sanitized: dict[str, float] = {}
        for raw_ticker, raw_delta in candidate_plan.items():
            normalized = normalize_ticker(str(raw_ticker))
            if normalized not in allowed:
                continue
            try:
                delta = clamp(float(raw_delta), -0.1, 0.1)
            except (TypeError, ValueError):
                continue
            if abs(delta) < 0.005:
                continue
            sanitized[allowed[normalized]] = round(delta, 4)
        constraint_check = validate_rebalance_plan(portfolio=portfolio, plan=sanitized)
        if not constraint_check.passed:
            constraint_set = default_constraints_for(portfolio)
            total_trade = sum(abs(delta) for delta in constraint_check.adjusted_plan.values())
            if (
                constraint_check.reason.startswith("Total daily trade weight")
                and total_trade > constraint_set.max_trade_per_day
            ):
                scale = constraint_set.max_trade_per_day / total_trade
                staged_plan = {
                    ticker: round(delta * scale, 4)
                    for ticker, delta in constraint_check.adjusted_plan.items()
                    if abs(delta * scale) >= 0.005
                }
                staged_check = validate_rebalance_plan(portfolio=portfolio, plan=staged_plan)
                if staged_check.passed:
                    return staged_check.adjusted_plan
            return {}
        return constraint_check.adjusted_plan

    def _consensus_metrics(self, responses: list[AgentResponse]) -> tuple[float, float]:
        weighted_total = 0.0
        weight_sum = 0.0
        directional_values: list[float] = []
        for response in responses:
            if response.agent_id == "cost":
                continue
            weight = max(0.05, response.confidence * max(response.strength, 0.2))
            weighted_total += response.direction * weight
            weight_sum += weight
            directional_values.append(response.direction)
        consensus = weighted_total / weight_sum if weight_sum else 0.0
        if len(directional_values) <= 1:
            divergence = 0.0
        else:
            mean_value = sum(directional_values) / len(directional_values)
            variance = sum((value - mean_value) ** 2 for value in directional_values) / len(
                directional_values
            )
            divergence = variance**0.5
        return round(clamp(consensus, -1.0, 1.0), 4), round(clamp(divergence, 0.0, 1.0), 4)

    def _decision_trace(
        self,
        *,
        query: str,
        executed_calls: list[PlannedAgentCall],
        responses: list[AgentResponse],
        decision: JudgeDecision,
        judge_actions: list[Mapping[str, Any]] | None = None,
    ) -> list[DecisionTraceNode]:
        trace: list[DecisionTraceNode] = []
        response_index = 0

        def append_agent_response(
            response: AgentResponse, planned_call: PlannedAgentCall | None
        ) -> None:
            trace.append(
                DecisionTraceNode(
                    turn_number=len(trace) + 1,
                    phase=DecisionPhase.INFORMATION_GATHERING
                    if response.agent_id in {"disclosure", "news", "report"}
                    else DecisionPhase.DELIBERATION,
                    actor=response.agent_id,
                    query=planned_call.query if planned_call else query,
                    summary=response.reasoning_for_judge_agent or response.query_understood,
                    context=planned_call.context if planned_call else None,
                    note=planned_call.note if planned_call else None,
                    references=tuple(response.references),
                    tools_called=tuple(response.tools_called),
                )
            )

        for action in judge_actions or []:
            action_name = str(action.get("action") or "").strip().upper()
            agent_id = str(action.get("agent_id") or "").strip()
            layer = str(action.get("layer") or "core").strip().lower()
            reason = str(action.get("reason") or "").strip()
            candidate_plan = action.get("candidate_rebalance_plan", {})
            called_before = action.get("called_agents_before", [])
            called_before_text = (
                ", ".join(str(item) for item in called_before) if called_before else "없음"
            )
            if action_name == "CALL_AGENT" and agent_id:
                if layer == "domain":
                    summary = (
                        f"Judge가 Core 판단안을 {agent_id} 도메인 관점으로 심의하기로 결정했습니다."
                    )
                    routing_query = f"도메인 심의 호출: {agent_id}"
                else:
                    summary = f"Judge가 현재 관찰을 바탕으로 {agent_id} 에이전트만 호출하기로 결정했습니다."
                    routing_query = f"다음 호출 결정: {agent_id}"
            elif action_name == "FINALIZE_DOMAIN_REVIEW":
                summary = "Judge가 도메인 심의를 마치고 합의 계산으로 이동하기로 결정했습니다."
                routing_query = "도메인 심의 종료"
            else:
                summary = "Judge가 추가 호출 없이 최종 판단으로 이동하기로 결정했습니다."
                routing_query = "추가 호출 여부 판단"
            if reason:
                summary = f"{summary} 이유: {reason}"
            if isinstance(candidate_plan, Mapping) and candidate_plan:
                summary = f"{summary} 후보 리밸런싱 초안: {dict(candidate_plan)}"
            trace.append(
                DecisionTraceNode(
                    turn_number=len(trace) + 1,
                    phase=DecisionPhase.DELIBERATION,
                    actor="judge",
                    query=routing_query,
                    summary=summary,
                    context=f"호출 전 완료 에이전트: {called_before_text}",
                )
            )
            if action_name == "CALL_AGENT" and response_index < len(responses):
                response = responses[response_index]
                planned_call = (
                    executed_calls[response_index] if response_index < len(executed_calls) else None
                )
                append_agent_response(response, planned_call)
                response_index += 1

        for index in range(response_index, len(responses)):
            response = responses[index]
            planned_call = executed_calls[index] if index < len(executed_calls) else None
            append_agent_response(response, planned_call)
        domain_consensus = (
            dict(decision.auto_safeguards.get("domain_consensus", {}))
            if isinstance(decision.auto_safeguards.get("domain_consensus"), Mapping)
            else {}
        )
        domain_suffix = ""
        if domain_consensus:
            domain_suffix = (
                f" 도메인 합의 점수 {float(domain_consensus.get('score') or 0.0):.2f}, "
                f"approve {domain_consensus.get('n_approve', 0)}, reject {domain_consensus.get('n_reject', 0)}, "
                f"abstain {domain_consensus.get('n_abstain', 0)}."
            )
            if domain_consensus.get("compliance_veto"):
                domain_suffix += " Compliance 거부권이 적용되었습니다."
        trace.append(
            DecisionTraceNode(
                turn_number=len(trace) + 1,
                phase=DecisionPhase.CONSENSUS,
                actor="judge",
                query="합의 형성",
                summary=(
                    f"합의 점수 {decision.consensus_score:.2f}, 충돌 점수 {decision.divergence_score:.2f}, "
                    f"후보 리밸런싱 초안 {decision.candidate_rebalance_plan or '{}'}."
                    f"{domain_suffix}"
                ),
            )
        )
        trace.append(
            DecisionTraceNode(
                turn_number=len(trace) + 1,
                phase=DecisionPhase.DECISION,
                actor="judge",
                query=query,
                summary=decision.summary,
                references=(),
                tools_called=(),
            )
        )
        return trace

    def _is_low_signal_judge_payload(self, payload: Mapping[str, Any]) -> bool:
        decision = str(payload.get("decision", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        confidence = payload.get("confidence", 0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        return not decision or not summary or confidence_value <= 0.0

    def _judge_payload_has_unsupported_language(self, payload: Mapping[str, Any]) -> bool:
        user_visible_fields = {
            "summary": payload.get("summary"),
            "reasoning": payload.get("reasoning"),
            "user_notification": payload.get("user_notification"),
            "options": payload.get("options"),
        }
        return contains_japanese_kana(user_visible_fields)

    def _notification_level(self, decision: DecisionType, urgency: Urgency) -> str:
        if decision == DecisionType.USER_DECISION_REQUIRED:
            return "push"
        if decision == DecisionType.DEFER:
            return "info"
        if urgency in {Urgency.IMMEDIATE, Urgency.WATCH}:
            return "watch"
        if decision == DecisionType.HOLD:
            return "silent"
        return "info"

    def _default_follow_up_at(self, *, query: str, responses: list[AgentResponse]) -> str:
        now = datetime.now().astimezone()
        lowered = query.casefold()
        if any(
            response.agent_id == "report"
            and response.verdict == AgentVerdict.DIRECT_ANSWER_UNAVAILABLE
            for response in responses
        ):
            delta = timedelta(hours=1)
        elif any(token in lowered for token in ("장중", "속보", "breaking")):
            delta = timedelta(minutes=30)
        else:
            delta = timedelta(hours=4)
        return (now + delta).isoformat(timespec="seconds")

    def _default_feedback_checkpoint(self) -> str:
        return (datetime.now().astimezone() + timedelta(days=7)).isoformat(timespec="seconds")

    def _sanitize_future_timestamp(self, value: str | None, *, default: str | None) -> str | None:
        if not value:
            return default
        candidate = coerce_datetime(value)
        now = datetime.now().astimezone(candidate.tzinfo)
        if candidate < now - timedelta(minutes=5):
            return default
        return candidate.isoformat(timespec="seconds")


def _compact_judge_action_event(
    action: Mapping[str, Any],
    *,
    layer: str,
    turn_number: int,
    called_agents: list[str],
    response_count: int,
) -> dict[str, Any]:
    raw_plan = action.get("candidate_rebalance_plan")
    return {
        "layer": layer,
        "turn_number": turn_number,
        "action": str(action.get("action") or "").strip().upper(),
        "agent_id": canonical_agent_id(str(action.get("agent_id") or ""))
        if action.get("agent_id")
        else None,
        "reason": truncate(str(action.get("reason") or ""), 700),
        "query": truncate(str(action.get("query") or ""), 500),
        "depth": str(action.get("depth") or "").strip() or None,
        "candidate_rebalance_plan": dict(raw_plan) if isinstance(raw_plan, Mapping) else {},
        "called_agents": list(called_agents),
        "response_count": response_count,
    }
