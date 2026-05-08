from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import httpx
import yaml


AGENTS = [
    "disclosure",
    "news",
    "report",
    "profit",
    "cost",
    "risk",
    "tax",
    "compliance",
    "macro",
    "sentiment",
    "execution",
    "esg",
]

BASELINES = [
    "buy_hold",
    "calendar_monthly",
    "threshold_5pct",
    "equal_weight_calendar",
]


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path
    profiles: Path
    universe: Path
    scenarios: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def load_benchmark(paths: BenchmarkPaths) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    profiles = load_yaml(paths.profiles).get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("profiles.yaml must contain profiles mapping.")
    stocks = load_yaml(paths.universe).get("stocks", [])
    if not isinstance(stocks, list):
        raise ValueError("stock_universe.yaml must contain stocks list.")
    universe = {str(item["symbol"]): item for item in stocks if isinstance(item, Mapping) and item.get("symbol")}
    scenarios = []
    for path in sorted(paths.scenarios.glob("*.yaml")):
        scenario = load_yaml(path)
        scenario["_path"] = str(path)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"No scenario YAML files found in {paths.scenarios}.")
    return dict(profiles), universe, scenarios


def simulated_date(scenario: Mapping[str, Any]) -> date:
    raw = str(scenario.get("simulated_date") or "").strip()
    if not raw:
        raise ValueError(f"{scenario.get('scenario_id')} missing simulated_date.")
    return date.fromisoformat(raw)


def is_month_end(day: date) -> bool:
    return (day + timedelta(days=1)).month != day.month


def holdings(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    portfolio = scenario.get("portfolio")
    if not isinstance(portfolio, Mapping):
        raise ValueError(f"{scenario.get('scenario_id')} missing portfolio mapping.")
    raw_holdings = portfolio.get("holdings")
    if not isinstance(raw_holdings, list) or not raw_holdings:
        raise ValueError(f"{scenario.get('scenario_id')} must contain portfolio.holdings.")
    return [dict(item) for item in raw_holdings if isinstance(item, Mapping)]


def max_abs_drift(scenario: Mapping[str, Any]) -> float:
    values = []
    for item in holdings(scenario):
        current = float(item.get("current_weight", 0.0) or 0.0)
        target = float(item.get("target_weight", current) or 0.0)
        values.append(abs(current - target))
    return max(values or [0.0])


def apply_buy_hold(_: Mapping[str, Any]) -> str:
    return "HOLD"


def apply_calendar_monthly(scenario: Mapping[str, Any]) -> str:
    return "REBALANCE" if is_month_end(simulated_date(scenario)) else "HOLD"


def apply_threshold_5pct(scenario: Mapping[str, Any]) -> str:
    return "REBALANCE" if max_abs_drift(scenario) > 0.05 else "HOLD"


def apply_equal_weight_calendar(scenario: Mapping[str, Any]) -> str:
    return "REBALANCE" if is_month_end(simulated_date(scenario)) else "HOLD"


def apply_baselines(scenario: Mapping[str, Any]) -> dict[str, str]:
    return {
        "buy_hold": apply_buy_hold(scenario),
        "calendar_monthly": apply_calendar_monthly(scenario),
        "threshold_5pct": apply_threshold_5pct(scenario),
        "equal_weight_calendar": apply_equal_weight_calendar(scenario),
    }


def profile_preferences(profile: Mapping[str, Any]) -> list[str]:
    preferences = [str(item) for item in profile.get("preferences", []) if str(item).strip()]
    structured = [
        f"excluded_sectors={profile.get('excluded_sectors', [])}",
        f"max_single_weight={profile.get('max_single_weight')}",
        f"esg_min_score={profile.get('esg_min_score')}",
        f"approval_required_above_krw={profile.get('approval_required_above_krw')}",
        f"tax_loss_harvesting_pref={profile.get('tax_loss_harvesting_pref')}",
    ]
    return preferences + structured


def build_portfolio(scenario: Mapping[str, Any], profile: Mapping[str, Any], universe: Mapping[str, Any]) -> dict[str, Any]:
    raw_portfolio = scenario["portfolio"]
    total_value = float(raw_portfolio.get("total_value_krw", 0.0) or 0.0)
    result_holdings = []
    for item in holdings(scenario):
        symbol = str(item["symbol"])
        stock = universe.get(symbol, {})
        current_weight = float(item.get("current_weight", 0.0) or 0.0)
        market_value = total_value * current_weight if total_value else None
        result_holdings.append(
            {
                "ticker": symbol,
                "company_name": str(stock.get("name") or item.get("name") or symbol),
                "weight": current_weight,
                "aliases": list(stock.get("aliases", [])),
                "shares": item.get("shares"),
                "last_price": item.get("last_price"),
                "average_price": item.get("average_price"),
                "market_value_krw": market_value,
                "unrealized_pnl_krw": _unrealized_pnl(item),
            }
        )
    return {
        "generated_at": f"{scenario['simulated_date']}T09:00:00+09:00",
        "holdings": result_holdings,
        "total_value_krw": total_value,
        "cash_weight": float(raw_portfolio.get("cash_weight", 0.0) or 0.0),
        "user_preferences": profile_preferences(profile),
    }


def _unrealized_pnl(item: Mapping[str, Any]) -> float | None:
    shares = item.get("shares")
    last_price = item.get("last_price")
    average_price = item.get("average_price")
    if shares is None or last_price is None or average_price is None:
        return None
    try:
        return (float(last_price) - float(average_price)) * float(shares)
    except (TypeError, ValueError):
        return None


def build_portfolio_definition(scenario: Mapping[str, Any], universe: Mapping[str, Any]) -> dict[str, Any] | None:
    if not bool(scenario.get("portfolio_definition_enabled")):
        return None
    raw_targets = []
    total_target = 0.0
    for item in holdings(scenario):
        target = float(item.get("target_weight", 0.0) or 0.0)
        if target <= 0:
            continue
        total_target += target
        raw_targets.append((item, target))
    if not raw_targets or total_target <= 0:
        return None
    target_weights = []
    for item, target in raw_targets:
        symbol = str(item["symbol"])
        stock = universe.get(symbol, {})
        target_weights.append(
            {
                "ticker": symbol,
                "company_name": str(stock.get("name") or item.get("name") or symbol),
                "weight": round(target / total_target, 6),
                "market": str(stock.get("market") or "KR"),
            }
        )
    # Keep validation happy after rounding.
    drift = round(1.0 - sum(float(item["weight"]) for item in target_weights), 6)
    target_weights[-1]["weight"] = round(float(target_weights[-1]["weight"]) + drift, 6)
    return {
        "name": f"{scenario['scenario_id']} target weights",
        "description": str(scenario.get("description") or ""),
        "risk_profile": "위험중립형",
        "drift_threshold": 0.05,
        "rebalancing_frequency": "benchmark scenario",
        "target_weights": target_weights,
    }


def entity(symbol: str, universe: Mapping[str, Any]) -> dict[str, Any]:
    stock = universe.get(symbol, {})
    return {
        "entity_id": symbol,
        "entity_type": "STOCK",
        "entity_name": str(stock.get("name") or symbol),
        "ticker": symbol,
        "confidence": 1.0,
    }


def build_knowledge_base(scenario: Mapping[str, Any], universe: Mapping[str, Any]) -> dict[str, Any]:
    events = []
    documents = []
    scenario_id = str(scenario["scenario_id"])
    day = str(scenario["simulated_date"])
    for index, raw_event in enumerate(scenario.get("events") or [], start=1):
        tickers = [str(item) for item in raw_event.get("tickers", [])]
        doc_id = f"{scenario_id}_event_doc_{index}"
        events.append(
            {
                "event_id": f"{scenario_id}_event_{index}",
                "event_type": str(raw_event.get("event_type") or "OTHER"),
                "event_time": f"{day}T09:00:00+09:00",
                "headline": str(raw_event.get("headline") or ""),
                "summary": str(raw_event.get("summary") or ""),
                "confidence": float(raw_event.get("confidence", 0.75) or 0.75),
                "source_documents": [doc_id],
                "matched_holdings": tickers,
                "entities": [entity(symbol, universe) for symbol in tickers],
                "metadata": {"scenario_id": scenario_id},
            }
        )
    for index, raw_doc in enumerate(scenario.get("documents") or [], start=1):
        tickers = [str(item) for item in raw_doc.get("tickers", [])]
        documents.append(
            {
                "doc_id": f"{scenario_id}_doc_{index}",
                "doc_type": str(raw_doc.get("doc_type") or "NEWS"),
                "title": str(raw_doc.get("title") or ""),
                "body": str(raw_doc.get("body") or ""),
                "publisher": str(raw_doc.get("publisher") or "BenchmarkFixture"),
                "source_name": "benchmark",
                "source_url": f"benchmark://{scenario_id}/{index}",
                "region": "KR",
                "published_at": f"{day}T09:00:00+09:00",
                "relevance_score": 0.9,
                "event_type": str(raw_doc.get("doc_type") or "NEWS"),
                "event_type_score": 0.9,
                "entities": [entity(symbol, universe) for symbol in tickers],
                "matched_holdings": tickers,
                "metadata": {"scenario_id": scenario_id},
            }
        )
    return {
        "events": events,
        "documents": documents,
        "source_paths": {"benchmark_scenario": scenario_id},
    }


def build_agent_payload(scenario: Mapping[str, Any], profiles: Mapping[str, Any], universe: Mapping[str, Any]) -> dict[str, Any]:
    profile_id = str(scenario.get("profile_id") or "")
    if profile_id not in profiles:
        raise ValueError(f"{scenario.get('scenario_id')} references unknown profile_id={profile_id!r}.")
    profile = profiles[profile_id]
    payload = {
        "query": str(scenario.get("query") or scenario.get("title") or scenario["scenario_id"]),
        "portfolio": build_portfolio(scenario, profile, universe),
        "knowledge_base": build_knowledge_base(scenario, universe),
        "depth": str(scenario.get("depth") or "medium"),
        "trigger": str(scenario.get("trigger") or "pull"),
    }
    definition = build_portfolio_definition(scenario, universe)
    if definition is not None:
        payload["portfolio_definition"] = definition
    return payload


def call_libra(base_url: str, payload: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{base_url.rstrip('/')}/v1/judge-runs", json=payload)
        response.raise_for_status()
        return response.json()


def summarize_libra_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "libra_decision": "",
            "called_agents": [],
            "skipped_agents": [],
            "confidence": "",
            "summary": "",
            "trace_turns": 0,
        }
    decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
    trace = decision.get("decision_trace", []) if isinstance(decision, Mapping) else []
    return {
        "libra_decision": decision.get("decision", ""),
        "called_agents": list(decision.get("called_agents", []) or []),
        "skipped_agents": list(decision.get("skipped_agents", []) or []),
        "confidence": decision.get("confidence", ""),
        "summary": decision.get("summary", ""),
        "trace_turns": len(trace) if isinstance(trace, list) else 0,
    }


def build_row(
    scenario: Mapping[str, Any],
    computed_baselines: Mapping[str, str],
    result: Mapping[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    expected = scenario.get("expected") if isinstance(scenario.get("expected"), Mapping) else {}
    summary = summarize_libra_result(result)
    return {
        "scenario_id": scenario.get("scenario_id"),
        "title": scenario.get("title"),
        "simulated_date": scenario.get("simulated_date"),
        "profile_id": scenario.get("profile_id"),
        "expected_agents": list(expected.get("agents", []) or []),
        "expected_decision": expected.get("decision", ""),
        "computed_baselines": dict(computed_baselines),
        "declared_baselines": dict(scenario.get("baseline_decisions", {}) or {}),
        "libra_decision": summary["libra_decision"],
        "called_agents": summary["called_agents"],
        "skipped_agents": summary["skipped_agents"],
        "confidence": summary["confidence"],
        "trace_turns": summary["trace_turns"],
        "summary": summary["summary"],
        "error": error or "",
        "libra_only_output": ["decision", "called_agents", "decision_trace", "agent_rationales", "user_notification"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_decision_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "title", *BASELINES, "LIBRA", "called_agents", "LIBRA_only_output", "note"])
        for row in rows:
            baselines = row["computed_baselines"]
            writer.writerow(
                [
                    row["scenario_id"],
                    row["title"],
                    *(baselines.get(name, "") for name in BASELINES),
                    row["libra_decision"] or "(not run)",
                    " ".join(row["called_agents"] or row["expected_agents"]),
                    "+".join(row["libra_only_output"]),
                    row["summary"] or row["error"],
                ]
            )


def write_call_heatmap_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "mode", *AGENTS])
        for row in rows:
            called = set(row["called_agents"] or row["expected_agents"])
            mode = "actual" if row["called_agents"] else "expected"
            writer.writerow([row["scenario_id"], mode, *(1 if agent in called else 0 for agent in AGENTS)])


def write_call_heatmap_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    cell = 24
    label_w = 190
    top = 120
    width = label_w + len(AGENTS) * cell + 20
    height = top + len(rows) * cell + 30
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf7"/>',
        '<style>text{font-family:Arial,"Malgun Gothic",sans-serif;font-size:11px;fill:#111827}.small{font-size:9px;fill:#6b7280}.on{fill:#2563eb}.off{fill:#e5e7eb;stroke:#d1d5db}</style>',
        '<text x="16" y="24" font-size="16" font-weight="700">LIBRA call pattern heatmap</text>',
    ]
    for idx, agent in enumerate(AGENTS):
        x = label_w + idx * cell + 15
        lines.append(f'<text class="small" x="{x}" y="112" transform="rotate(-55 {x} 112)">{agent}</text>')
    for r_idx, row in enumerate(rows):
        y = top + r_idx * cell
        called = set(row["called_agents"] or row["expected_agents"])
        mode = "actual" if row["called_agents"] else "expected"
        lines.append(f'<text x="16" y="{y + 16}">{row["scenario_id"]}</text>')
        lines.append(f'<text class="small" x="126" y="{y + 16}">{mode}</text>')
        for c_idx, agent in enumerate(AGENTS):
            x = label_w + c_idx * cell
            cls = "on" if agent in called else "off"
            lines.append(f'<rect class="{cls}" x="{x}" y="{y}" width="18" height="18" rx="3"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LIBRA Benchmark Summary",
        "",
        "| scenario | LIBRA | called agents | baselines | note |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        baselines = ", ".join(f"{name}={row['computed_baselines'].get(name)}" for name in BASELINES)
        called = ", ".join(row["called_agents"] or row["expected_agents"])
        note = row["summary"] or row["error"] or ""
        lines.append(f"| {row['scenario_id']} | {row['libra_decision'] or '(not run)'} | {called} | {baselines} | {note} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_root = Path(args.benchmark_root or repo_root / "benchmarks").resolve()
    paths = BenchmarkPaths(
        root=benchmark_root,
        profiles=benchmark_root / "profiles.yaml",
        universe=benchmark_root / "stock_universe.yaml",
        scenarios=benchmark_root / "scenarios",
    )
    profiles, universe, scenarios = load_benchmark(paths)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir or repo_root / "outputs" / "benchmark" / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected = scenarios
    if args.scenario_id:
        wanted = {item.strip() for item in args.scenario_id.split(",") if item.strip()}
        selected = [item for item in selected if str(item.get("scenario_id")) in wanted]
        missing = wanted - {str(item.get("scenario_id")) for item in selected}
        if missing:
            raise ValueError(f"Unknown scenario_id values: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: args.limit]

    for scenario in selected:
        baselines = apply_baselines(scenario)
        result: dict[str, Any] | None = None
        error = None
        if not args.skip_libra:
            payload = build_agent_payload(scenario, profiles, universe)
            try:
                result = call_libra(args.base_url, payload, timeout_seconds=args.timeout_seconds)
            except Exception as exc:  # keep the benchmark moving so failures become analysis rows
                error = f"{type(exc).__name__}: {exc}"
            write_json(out_dir / f"{scenario['scenario_id']}.payload.json", payload)
            if result is not None:
                write_json(out_dir / f"{scenario['scenario_id']}.result.json", result)
        rows.append(build_row(scenario, baselines, result, error=error))

    write_json(out_dir / "rows.json", rows)
    write_decision_matrix(out_dir / "decision_matrix.csv", rows)
    write_call_heatmap_csv(out_dir / "call_heatmap.csv", rows)
    write_call_heatmap_svg(out_dir / "call_heatmap.svg", rows)
    write_summary_md(out_dir / "summary.md", rows)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIBRA controlled scenario benchmark.")
    parser.add_argument("--benchmark-root", help="Benchmark root directory. Defaults to ./benchmarks.")
    parser.add_argument("--out-dir", help="Output directory. Defaults to ./outputs/benchmark/<timestamp>.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="libra-agent API base URL.")
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--skip-libra", action="store_true", help="Only compute baseline decisions and expected heatmap.")
    parser.add_argument("--scenario-id", help="Comma-separated scenario ids to run.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected scenarios.")
    args = parser.parse_args()
    out_dir = run(args)
    print(f"benchmark outputs: {out_dir}")


if __name__ == "__main__":
    main()
