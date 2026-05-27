from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MISSING_PATTERNS = (
    "OHLCV 또는 수익률 히스토리",
    "ADV, 호가",
    "데이터가 없어",
    "데이터 없음",
    "문서 0건",
    "리포트 0건",
    "로컬 캐시에 없어",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _agent_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = row.get("result")
    if not isinstance(result, dict):
        return []
    agents = result.get("agent_responses")
    return agents if isinstance(agents, list) else []


def _decision(row: dict[str, Any]) -> str | None:
    result = row.get("result")
    if not isinstance(result, dict):
        return None
    for key_path in (
        ("governance_v1", "final_decision", "decision"),
        ("decision", "decision"),
    ):
        cursor: Any = result
        for key in key_path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def _branch(row: dict[str, Any]) -> str | None:
    result = row.get("result")
    if not isinstance(result, dict):
        return None
    for key_path in (
        ("governance_v1", "final_decision", "branch"),
        ("decision", "auto_safeguards", "governance_v1_branch"),
    ):
        cursor: Any = result
        for key in key_path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def _safe_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def audit(raw: Path) -> dict[str, Any]:
    rows = _read_jsonl(raw)
    agent_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "opinions": Counter(),
            "verdicts": Counter(),
            "missing_pattern_rows": 0,
            "sample_missing": None,
        }
    )
    decision_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    report_coverage_rows = 0
    report_coverage_total = 0
    technical_indicator_rows = 0
    liquidity_observed_rows = 0
    liquidity_missing_rows = 0

    for row in rows:
        if decision := _decision(row):
            decision_counts[decision] += 1
        if branch := _branch(row):
            branch_counts[branch] += 1

        date = str(row.get("date") or "")
        for agent in _agent_rows(row):
            agent_id = str(agent.get("agent_id") or agent.get("agent") or "").strip()
            if not agent_id:
                continue
            stats = agent_stats[agent_id]
            stats["rows"] += 1
            stats["opinions"][agent.get("opinion")] += 1
            stats["verdicts"][agent.get("verdict")] += 1
            text = _safe_text(agent)
            if any(pattern in text for pattern in MISSING_PATTERNS):
                stats["missing_pattern_rows"] += 1
                if stats["sample_missing"] is None:
                    stats["sample_missing"] = {
                        "date": date,
                        "reasoning": str(
                            agent.get("reasoning_for_judge_agent")
                            or agent.get("limits_acknowledged")
                            or ""
                        )[:500],
                    }

            evidence = agent.get("evidence")
            evidence = evidence if isinstance(evidence, dict) else {}
            if agent_id == "report":
                count = int(evidence.get("coverage_reports_count") or 0)
                report_coverage_total += count
                if count > 0:
                    report_coverage_rows += 1
            elif agent_id == "technical":
                if any(token in text for token in ("RSI14", "price vs MA20", "price vs MA60", "volume ratio 20d")):
                    technical_indicator_rows += 1
            elif agent_id == "liquidity":
                if "ADV 사용률" in text:
                    liquidity_observed_rows += 1
                if "ADV, 호가" in text:
                    liquidity_missing_rows += 1

    serializable_agents = {}
    for agent_id, stats in sorted(agent_stats.items()):
        serializable_agents[agent_id] = {
            "rows": stats["rows"],
            "opinions": dict(stats["opinions"].most_common()),
            "verdicts": dict(stats["verdicts"].most_common()),
            "missing_pattern_rows": stats["missing_pattern_rows"],
            "sample_missing": stats["sample_missing"],
        }

    return {
        "raw": str(raw),
        "rows": len(rows),
        "first_date": rows[0].get("date") if rows else None,
        "last_date": rows[-1].get("date") if rows else None,
        "decision_counts": dict(decision_counts.most_common()),
        "branch_counts": dict(branch_counts.most_common()),
        "coverage": {
            "liquidity_observed_rows": liquidity_observed_rows,
            "liquidity_missing_rows": liquidity_missing_rows,
            "technical_indicator_rows": technical_indicator_rows,
            "report_coverage_rows": report_coverage_rows,
            "report_coverage_total": report_coverage_total,
        },
        "agents": serializable_agents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence coverage in a LIBRA replay raw JSONL.")
    parser.add_argument("--raw", required=True, help="Replay raw JSONL path.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    payload = audit(Path(args.raw))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
