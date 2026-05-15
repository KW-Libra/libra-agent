from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from libra_agent.api.routes import ResumeRequest, RunStartRequest

ROOT = Path(__file__).resolve().parents[1]


def _portfolio() -> dict[str, object]:
    return {
        "generated_at": "2026-05-15T00:00:00+09:00",
        "holdings": [],
    }


def _schema(name: str) -> dict[str, object]:
    with (ROOT / "contracts" / name).open(encoding="utf-8") as file:
        return json.load(file)


def _schema_errors(name: str, payload: dict[str, object]) -> list[object]:
    schema = _schema(name)
    try:
        import jsonschema
    except ModuleNotFoundError:
        return _minimal_schema_errors(schema, payload)

    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(validator.iter_errors(payload), key=lambda error: list(error.path))


def _validate_schema_payload(name: str, payload: dict[str, object]) -> None:
    errors = _schema_errors(name, payload)
    assert errors == []


def _minimal_schema_errors(schema: dict[str, object], payload: object) -> list[str]:
    """Small fallback for the schema features used by these contract samples."""

    def resolve(ref: str) -> dict[str, object]:
        if not ref.startswith("#/$defs/"):
            return {}
        current: object = schema
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(current, dict):
                return {}
            current = current.get(part, {})
        return current if isinstance(current, dict) else {}

    def matches_type(expected: object, value: object) -> bool:
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "null":
            return value is None
        return True

    def validate(node: dict[str, object], value: object, path: str) -> list[str]:
        if "$ref" in node:
            return validate(resolve(str(node["$ref"])), value, path)

        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            if any(
                not validate(candidate, value, path)
                for candidate in any_of
                if isinstance(candidate, dict)
            ):
                return []
            return [f"{path}: did not match anyOf"]

        expected_type = node.get("type")
        if expected_type is not None and not matches_type(expected_type, value):
            return [f"{path}: expected {expected_type}"]

        enum = node.get("enum")
        if isinstance(enum, list) and value not in enum:
            return [f"{path}: not in enum"]

        errors: list[str] = []
        if expected_type == "object" and isinstance(value, dict):
            required = node.get("required")
            if isinstance(required, list):
                for key in required:
                    if isinstance(key, str) and key not in value:
                        errors.append(f"{path}.{key}: required")

            properties = node.get("properties")
            properties = properties if isinstance(properties, dict) else {}
            additional = node.get("additionalProperties", True)
            if additional is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path}.{key}: additional property")
            for key, child in properties.items():
                if key in value and isinstance(child, dict):
                    errors.extend(validate(child, value[key], f"{path}.{key}"))
            if isinstance(additional, dict):
                for key, child_value in value.items():
                    if key not in properties:
                        errors.extend(validate(additional, child_value, f"{path}.{key}"))

        if expected_type == "array" and isinstance(value, list):
            items = node.get("items")
            if isinstance(items, dict):
                for index, item in enumerate(value):
                    errors.extend(validate(items, item, f"{path}[{index}]"))

        return errors

    return validate(schema, payload, "$")


def test_run_start_request_accepts_contract_human_interrupt_flag():
    request = RunStartRequest(
        query="포트폴리오 점검",
        portfolio=_portfolio(),
        enable_human_interrupts=True,
    )

    assert request.human_review_enabled() is True


def test_run_start_request_accepts_runtime_contract_fields():
    request = RunStartRequest(
        query="포트폴리오 점검",
        portfolio={
            "holdings": [
                {
                    "ticker": "005930",
                    "company_name": "삼성전자",
                    "weight": 0.5,
                    "sector": "TECH",
                    "esg_score": 0.72,
                    "carbon_intensity": 1.5,
                }
            ],
        },
        knowledge_sources={
            "events": "events.json",
            "normalized_documents": "normalized_documents.json",
            "ingest_refresh_enabled": False,
        },
        knowledge_base={
            "events": [],
            "documents": [],
            "source_paths": {"ingest_refresh_enabled": "true"},
        },
        portfolio_definition={
            "name": "Core",
            "target_weights": [{"ticker": "005930", "company_name": "삼성전자", "weight": 1.0}],
        },
        governance_v1={
            "execution_mode": "primary",
            "ips": {"single_ticker_limit_pct": 25.0},
            "kyc": {"risk_tolerance": "MODERATE"},
        },
        approval_required=True,
    )

    assert request.human_review_enabled() is True
    assert request.knowledge_sources["ingest_refresh_enabled"] is False
    assert request.knowledge_base["source_paths"]["ingest_refresh_enabled"] == "true"
    assert request.portfolio_definition["target_weights"][0]["ticker"] == "005930"
    assert request.governance_v1["execution_mode"] == "primary"


def test_judge_run_request_schema_accepts_route_runtime_fields():
    _validate_schema_payload(
        "judge-run-request.schema.json",
        {
            "query": "포트폴리오 점검",
            "portfolio": {
                "holdings": [
                    {
                        "ticker": "005930",
                        "company_name": "삼성전자",
                        "weight": 0.5,
                        "sector": "TECH",
                        "esg_score": 0.72,
                        "carbon_intensity": 1.5,
                    }
                ]
            },
            "knowledge_sources": {
                "events": "events.json",
                "normalized_documents": "normalized_documents.json",
                "ingest_refresh_enabled": False,
            },
            "knowledge_base": {
                "events": [
                    {
                        "event_id": "evt-1",
                        "event_type": "NEWS",
                        "event_time": "2026-05-15T00:00:00+09:00",
                        "headline": "뉴스",
                        "summary": "요약",
                        "confidence": 0.8,
                        "metadata": {"source": "test"},
                    }
                ],
                "documents": [],
                "source_paths": {"ingest_refresh_enabled": "true"},
            },
            "portfolio_definition": {
                "name": "Core",
                "target_weights": [{"ticker": "005930", "company_name": "삼성전자", "weight": 1.0}],
            },
            "governance_v1": {
                "execution_mode": "primary",
                "ips": {"single_ticker_limit_pct": 25.0},
                "kyc": {"risk_tolerance": "MODERATE"},
            },
            "depth": "medium",
            "trigger": "pull",
            "deadline_seconds": 30,
            "thread_id": "thread-1",
            "enable_human_interrupts": True,
            "approval_required": True,
        },
    )


def test_judge_run_request_schema_rejects_unknown_top_level_fields():
    errors = _schema_errors(
        "judge-run-request.schema.json",
        {
            "query": "포트폴리오 점검",
            "portfolio": {},
            "unknown_field": True,
        },
    )

    assert errors


def test_run_start_request_requires_portfolio():
    with pytest.raises(ValidationError):
        RunStartRequest(query="포트폴리오 점검")


def test_run_start_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RunStartRequest(
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            unknown_field=True,
        )


def test_resume_request_keeps_contract_and_ui_fields():
    request = ResumeRequest(
        approved=True,
        decision="APPROVE",
        interrupt_id="interrupt-1",
        option_index=0,
        override_decision="HOLD",
        metadata={"source": "test"},
    )

    assert request.interrupt_id == "interrupt-1"
    assert request.option_index == 0
    assert request.override_decision == "HOLD"
    assert request.metadata == {"source": "test"}


def test_user_approval_response_schema_accepts_resume_runtime_fields():
    _validate_schema_payload(
        "user-approval-response.schema.json",
        {
            "approved": True,
            "decision": "APPROVE",
            "interrupt_id": "interrupt-1",
            "option_index": 0,
            "override_decision": "HOLD",
            "override_plan": {"005930": 0.1},
            "note": "ok",
            "effective_at": None,
            "responder": "user-1",
            "metadata": {"source": "ui"},
            "client_extra": "allowed",
        },
    )
