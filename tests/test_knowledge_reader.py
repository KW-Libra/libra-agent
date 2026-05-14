from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

from libra_agent.knowledge.reader import KnowledgeReader


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reader_loads_local_current_artifacts(tmp_path: Path):
    _write_json(tmp_path / "manifest.json", {"generated_at": "2026-05-14T00:00:00Z", "counts": {"events": 1}})
    _write_json(tmp_path / "normalized_documents.json", {"documents": []})
    _write_json(tmp_path / "events.json", {"events": [{}]})
    _write_json(tmp_path / "consensus_snapshot.json", {"snapshots": [{}]})

    snapshot = KnowledgeReader(cache_dir=tmp_path, s3_bucket=None).load_current()
    payload = snapshot.to_dict(include_payloads=True)

    assert payload["summary"]["available"] is True
    assert payload["summary"]["source"] == "local"
    assert payload["summary"]["counts"]["events"] == 1
    assert "consensus_snapshot" in payload["payloads"]


def test_reader_falls_back_to_s3_when_local_missing(tmp_path: Path):
    objects = {
        "knowledge/current/manifest.json": {"generated_at": "2026-05-14T00:00:00Z", "counts": {"events": 1}},
        "knowledge/current/normalized_documents.json": {"documents": []},
        "knowledge/current/events.json": {"events": []},
    }

    def get_object(*, Bucket: str, Key: str):
        if Key not in objects:
            raise RuntimeError("missing")
        return {"Body": io.BytesIO(json.dumps(objects[Key]).encode("utf-8"))}

    client = Mock()
    client.get_object.side_effect = get_object

    with patch("libra_agent.knowledge.reader.boto3.client", return_value=client):
        snapshot = KnowledgeReader(
            cache_dir=tmp_path,
            s3_bucket="bucket",
            s3_prefix="knowledge/current",
            aws_region="ap-northeast-2",
        ).load_current()

    assert snapshot.available is True
    assert snapshot.summary()["source"] == "s3"
    assert snapshot.file_locations["manifest"] == "s3://bucket/knowledge/current/manifest.json"


def test_reader_reports_missing_required_artifacts(tmp_path: Path):
    snapshot = KnowledgeReader(cache_dir=tmp_path, s3_bucket=None).load_current()

    assert snapshot.available is False
    assert snapshot.summary()["source"] == "missing"
    assert snapshot.summary()["missing_files"] == ["manifest", "normalized_documents", "events"]
