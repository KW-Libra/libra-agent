from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import boto3

from libra_agent.config import settings

KnowledgeSource = Literal["local", "s3", "missing"]

KNOWLEDGE_FILES = {
    "manifest": "manifest.json",
    "normalized_documents": "normalized_documents.json",
    "events": "events.json",
    "push_candidates": "push_candidates.json",
    "consensus_snapshot": "consensus_snapshot.json",
    "financial_statement": "financial_statement.json",
}

REQUIRED_FILES = {"manifest", "normalized_documents", "events"}


@dataclass(slots=True)
class KnowledgeSnapshot:
    source: KnowledgeSource
    payloads: dict[str, Any] = field(default_factory=dict)
    file_locations: dict[str, str] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)
    error: str | None = None
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def available(self) -> bool:
        return not self.error and not any(item in REQUIRED_FILES for item in self.missing_files)

    def summary(self) -> dict[str, Any]:
        manifest = self.payloads.get("manifest") if isinstance(self.payloads.get("manifest"), dict) else {}
        counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
        return {
            "available": self.available,
            "source": self.source,
            "loaded_at": self.loaded_at,
            "generated_at": manifest.get("generated_at") if isinstance(manifest, dict) else None,
            "counts": counts,
            "available_payloads": sorted(self.payloads.keys()),
            "missing_files": list(self.missing_files),
            "error": self.error,
        }

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": self.summary(),
            "file_locations": dict(self.file_locations),
        }
        if include_payloads:
            payload["payloads"] = dict(self.payloads)
        return payload


class KnowledgeReader:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        s3_bucket: str | None = None,
        s3_prefix: str = "knowledge/current",
        aws_region: str = "ap-northeast-2",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip().strip("/")
        self.aws_region = aws_region

    @classmethod
    def from_settings(cls) -> "KnowledgeReader":
        return cls(
            cache_dir=settings.knowledge_cache_dir,
            s3_bucket=settings.s3_bucket,
            s3_prefix=settings.knowledge_s3_prefix,
            aws_region=settings.aws_region,
        )

    def load_current(self) -> KnowledgeSnapshot:
        local_snapshot = self._load_local()
        if local_snapshot.available:
            return local_snapshot
        if self.s3_bucket:
            s3_snapshot = self._load_s3()
            if s3_snapshot.available:
                return s3_snapshot
            if local_snapshot.payloads:
                return local_snapshot
            return s3_snapshot
        return local_snapshot

    def _load_local(self) -> KnowledgeSnapshot:
        payloads: dict[str, Any] = {}
        locations: dict[str, str] = {}
        missing: list[str] = []

        for logical_name, file_name in KNOWLEDGE_FILES.items():
            path = self.cache_dir / file_name
            if not path.exists():
                if logical_name in REQUIRED_FILES:
                    missing.append(logical_name)
                continue
            payloads[logical_name] = json.loads(path.read_text(encoding="utf-8"))
            locations[logical_name] = str(path)

        return KnowledgeSnapshot(
            source="local" if payloads else "missing",
            payloads=payloads,
            file_locations=locations,
            missing_files=missing,
        )

    def _load_s3(self) -> KnowledgeSnapshot:
        payloads: dict[str, Any] = {}
        locations: dict[str, str] = {}
        missing: list[str] = []
        client = boto3.client("s3", region_name=self.aws_region)

        for logical_name, file_name in KNOWLEDGE_FILES.items():
            key = "/".join(part for part in (self.s3_prefix, file_name) if part)
            location = f"s3://{self.s3_bucket}/{key}"
            try:
                response = client.get_object(Bucket=self.s3_bucket, Key=key)
            except Exception as exc:
                if logical_name in REQUIRED_FILES:
                    missing.append(logical_name)
                if logical_name == "manifest" and not payloads:
                    return KnowledgeSnapshot(
                        source="missing",
                        payloads=payloads,
                        file_locations=locations,
                        missing_files=missing,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                continue
            payloads[logical_name] = json.loads(response["Body"].read().decode("utf-8"))
            locations[logical_name] = location

        return KnowledgeSnapshot(
            source="s3" if payloads else "missing",
            payloads=payloads,
            file_locations=locations,
            missing_files=missing,
        )
