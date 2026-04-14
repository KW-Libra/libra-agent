from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime_or_none(value: str | None) -> datetime | None:
    if not value:
        return None

    candidate = value.strip()
    if not candidate:
        return None

    try:
        parsed = parsedate_to_datetime(candidate)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
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
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
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
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collapse_whitespace(value: str) -> str:
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
