from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SOURCE_TRUST = {
    "disclosure": 1.0,
    "dart": 1.0,
    "report": 1.0,
    "research": 1.0,
    "news": 0.7,
    "profit": 0.5,
    "cost": 0.5,
}

VALID_EVENT_TYPES = {
    "capex",
    "disclosure",
    "earnings",
    "funding",
    "geopolitical",
    "governance",
    "legal",
    "macro",
    "mna",
    "other",
    "personnel",
    "product",
    "regulation",
    "research",
}

VALID_HORIZONS = {"intraday", "short", "mid", "long"}


@dataclass(frozen=True, slots=True)
class SignalProfile:
    signal_score: float
    source_trust: float
    event_type: str | None
    horizon: str | None
    risk_level: str
    opinion: str


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def source_trust_for(agent_id: str) -> float:
    return SOURCE_TRUST.get(agent_id.strip().casefold(), 0.5)


def infer_signal_profile(
    *,
    agent_id: str,
    direction: float,
    strength: float,
    confidence: float,
    evidence: Mapping[str, Any] | None = None,
    explicit_signal_score: Any = None,
    explicit_source_trust: Any = None,
    explicit_event_type: Any = None,
    explicit_horizon: Any = None,
) -> SignalProfile:
    source_trust = _coerce_float(explicit_source_trust, source_trust_for(agent_id))
    event_type = normalize_event_type(explicit_event_type) or _find_first_text(
        evidence or {}, ("event_type", "dominant_event_type")
    )
    event_type = normalize_event_type(event_type)
    horizon = normalize_horizon(explicit_horizon) or _find_first_text(
        evidence or {}, ("horizon", "time_horizon")
    )
    horizon = normalize_horizon(horizon)

    if explicit_signal_score is None:
        explicit_signal_score = _find_first_number(evidence or {}, ("signal_score", "final_score"))
    if explicit_signal_score is None:
        score = direction * strength * confidence * source_trust
    else:
        score = _coerce_float(explicit_signal_score, 0.0)
    signal_score = round(clamp(score, -1.0, 1.0), 4)

    return SignalProfile(
        signal_score=signal_score,
        source_trust=round(clamp(source_trust, 0.0, 1.0), 4),
        event_type=event_type,
        horizon=horizon,
        risk_level=risk_level_for(signal_score),
        opinion=opinion_for(signal_score),
    )


def opinion_for(signal_score: float) -> str:
    if signal_score > 0.35:
        return "BUY_BIAS"
    if signal_score > 0.10:
        return "MILD_BUY"
    if signal_score < -0.35:
        return "SELL_BIAS"
    if signal_score < -0.10:
        return "MILD_SELL"
    return "NEUTRAL"


def risk_level_for(signal_score: float) -> str:
    abs_score = abs(signal_score)
    if abs_score > 0.60:
        return "high"
    if abs_score > 0.35:
        return "mid"
    return "low"


def normalize_event_type(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    normalized = text.replace("-", "_").strip("_").casefold()
    return normalized if normalized in VALID_EVENT_TYPES else None


def normalize_horizon(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    normalized = text.replace("-", "_").strip("_").casefold()
    return normalized if normalized in VALID_HORIZONS else None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _find_first_text(payload: Any, keys: tuple[str, ...]) -> str | None:
    found = _find_first(payload, keys, want_number=False)
    return _clean_text(found)


def _find_first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    found = _find_first(payload, keys, want_number=True)
    if found is None:
        return None
    try:
        return float(found)
    except (TypeError, ValueError):
        return None


def _find_first(payload: Any, keys: tuple[str, ...], *, want_number: bool) -> Any:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                value = payload[key]
                if want_number and isinstance(value, (int, float)):
                    return value
                if not want_number and _clean_text(value):
                    return value
        for value in payload.values():
            found = _find_first(value, keys, want_number=want_number)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            found = _find_first(item, keys, want_number=want_number)
            if found is not None:
                return found
    return None
