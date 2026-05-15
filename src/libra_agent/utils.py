from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime_or_none(value: str | None) -> datetime | None:
    if not value:
        return None

    candidate = value.strip()
    if not candidate:
        return None

    try:
        parsed = parsedate_to_datetime(candidate)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, IndexError):
        pass

    normalized = candidate.replace("Z", "+00:00")
    for parser in (
        lambda text: datetime.fromisoformat(text),
        lambda text: datetime.strptime(text, "%Y%m%d"),
        lambda text: datetime.strptime(text, "%Y-%m-%d"),
        lambda text: datetime.strptime(text, "%Y/%m/%d"),
        lambda text: datetime.strptime(text, "%Y.%m.%d"),
    ):
        try:
            parsed = parser(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            continue

    return None


def coerce_datetime(value: str | None, default: datetime | None = None) -> datetime:
    parsed = parse_datetime_or_none(value)
    if parsed is not None:
        return parsed
    return default or utc_now()


def stable_hash(payload: Mapping[str, Any] | str) -> str:
    if isinstance(payload, str):
        encoded = payload.encode("utf-8", errors="ignore")
    else:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    return hashlib.sha256(encoded).hexdigest()


def collapse_whitespace(value: str) -> str:
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def contains_japanese_kana(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_JAPANESE_KANA_RE.search(value))
    if isinstance(value, Mapping):
        return any(contains_japanese_kana(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(contains_japanese_kana(item) for item in value)
    return False
