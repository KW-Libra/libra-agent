from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import date, datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _price_rows(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    prices = fixture.get("prices")
    if not isinstance(prices, list) or not prices:
        raise ValueError("Fixture must contain a nonempty prices array.")
    return sorted((row for row in prices if isinstance(row, dict)), key=lambda row: str(row["date"]))


def _target_weights(fixture: dict[str, Any]) -> dict[str, float]:
    raw = fixture.get("target_weights")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Fixture must contain target_weights.")
    return {str(ticker): float(weight) for ticker, weight in raw.items()}


def _tickers(fixture: dict[str, Any]) -> list[str]:
    return list(_target_weights(fixture))


def _cost_rate(fixture: dict[str, Any]) -> float:
    return float(fixture.get("transaction_cost_bp", 0.0)) / 10_000.0


def _annualization_factor(fixture: dict[str, Any]) -> int:
    return int(fixture.get("annualization_factor", 252))


def _initial_positions(fixture: dict[str, Any]) -> dict[str, float]:
    first = _price_rows(fixture)[0]
    initial_value = float(fixture["initial_value_krw"])
    return {
        ticker: (initial_value * weight) / float(first[ticker])
        for ticker, weight in _target_weights(fixture).items()
    }


def _position_values(positions: dict[str, float], prices: dict[str, Any]) -> dict[str, float]:
    return {ticker: shares * float(prices[ticker]) for ticker, shares in positions.items()}


def _total_value(positions: dict[str, float], prices: dict[str, Any]) -> float:
    return sum(_position_values(positions, prices).values())


def _weights(positions: dict[str, float], prices: dict[str, Any]) -> dict[str, float]:
    values = _position_values(positions, prices)
    total = sum(values.values())
    if total <= 0:
        return {ticker: 0.0 for ticker in positions}
    return {ticker: value / total for ticker, value in values.items()}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {ticker: max(0.0, float(weight)) for ticker, weight in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {ticker: value / total for ticker, value in clipped.items()}


def _rebalance_to_weights(
    positions: dict[str, float],
    prices: dict[str, Any],
    target_weights: dict[str, float],
    cost_rate: float,
) -> tuple[dict[str, float], float, float]:
    current_values = _position_values(positions, prices)
    current_total = sum(current_values.values())
    desired_values = {
        ticker: current_total * float(weight)
        for ticker, weight in target_weights.items()
    }
    turnover = sum(abs(desired_values[ticker] - current_values.get(ticker, 0.0)) for ticker in target_weights)
    cost = turnover * cost_rate
    after_cost_total = max(0.0, current_total - cost)
    next_positions = {
        ticker: (after_cost_total * float(weight)) / float(prices[ticker])
        for ticker, weight in target_weights.items()
    }
    return next_positions, turnover, cost


def _returns(values: list[float]) -> list[float]:
    result: list[float] = []
    for previous, current in zip(values, values[1:]):
        result.append(0.0 if previous <= 0 else (current / previous) - 1.0)
    return result


def _cagr(values: list[float], annualization_factor: int) -> float:
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    years = (len(values) - 1) / float(annualization_factor)
    if years <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1.0 / years) - 1.0


def _annualized_volatility(values: list[float], annualization_factor: int) -> float:
    returns = _returns(values)
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(annualization_factor)


def _sharpe_ratio(values: list[float], annualization_factor: int) -> float | None:
    returns = _returns(values)
    if len(returns) < 2:
        return None
    volatility = statistics.stdev(returns)
    if volatility == 0:
        return None
    return (statistics.mean(returns) / volatility) * math.sqrt(annualization_factor)


def _max_drawdown(values: list[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            continue
        max_drawdown = min(max_drawdown, (value / peak) - 1.0)
    return max_drawdown


def _summary(
    *,
    name: str,
    group: str,
    value_history: list[float],
    initial_value: float,
    trades: int,
    turnover: float,
    transaction_cost: float,
    annualization_factor: int,
    decision_count: int = 0,
    trace_complete_count: int = 0,
    user_handoff_count: int = 0,
    avoided_trade_count: int = 0,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ending_value = value_history[-1]
    sharpe = _sharpe_ratio(value_history, annualization_factor)
    trace_completeness = None
    if decision_count:
        trace_completeness = trace_complete_count / decision_count
    return {
        "strategy": name,
        "group": group,
        "starting_value_krw": round(initial_value, 2),
        "ending_value_krw": round(ending_value, 2),
        "total_return_pct": round(((ending_value / initial_value) - 1.0) * 100.0, 3),
        "cagr_pct": round(_cagr(value_history, annualization_factor) * 100.0, 3),
        "annualized_volatility_pct": round(_annualized_volatility(value_history, annualization_factor) * 100.0, 3),
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(_max_drawdown(value_history) * 100.0, 3),
        "trades": trades,
        "turnover_krw": round(turnover, 2),
        "transaction_cost_krw": round(transaction_cost, 2),
        "trace_completeness_pct": round(trace_completeness * 100.0, 1) if trace_completeness is not None else None,
        "user_handoff_count": user_handoff_count,
        "avoided_trade_count": avoided_trade_count,
        "parameters": parameters or {},
    }


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _is_last_observed_date_of_month(index: int, prices: list[dict[str, Any]]) -> bool:
    current = _parse_date(str(prices[index]["date"]))
    if index == len(prices) - 1:
        return True
    next_date = _parse_date(str(prices[index + 1]["date"]))
    return current.month != next_date.month or current.year != next_date.year


def _is_last_observed_date_of_quarter(index: int, prices: list[dict[str, Any]]) -> bool:
    current = _parse_date(str(prices[index]["date"]))
    if current.month not in {3, 6, 9, 12}:
        return False
    if index == len(prices) - 1:
        return True
    next_date = _parse_date(str(prices[index + 1]["date"]))
    return current.month != next_date.month or current.year != next_date.year


def _inverse_vol_weights(
    price_window: list[dict[str, Any]],
    tickers: list[str],
    epsilon: float = 1e-9,
) -> dict[str, float]:
    vol_by_ticker: dict[str, float] = {}
    for ticker in tickers:
        ticker_returns = [
            (float(current[ticker]) / float(previous[ticker])) - 1.0
            for previous, current in zip(price_window, price_window[1:])
            if float(previous[ticker]) > 0
        ]
        if len(ticker_returns) >= 2:
            vol_by_ticker[ticker] = statistics.stdev(ticker_returns)
        elif ticker_returns:
            vol_by_ticker[ticker] = abs(ticker_returns[0])
        else:
            vol_by_ticker[ticker] = 0.0

    if all(value <= epsilon for value in vol_by_ticker.values()):
        equal_weight = 1.0 / len(tickers)
        return {ticker: equal_weight for ticker in tickers}

    inverse = {ticker: 1.0 / max(volatility, epsilon) for ticker, volatility in vol_by_ticker.items()}
    total = sum(inverse.values())
    return {ticker: value / total for ticker, value in inverse.items()}


def _trace_complete(result: dict[str, Any]) -> bool:
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    governance = result.get("governance_v1") if isinstance(result.get("governance_v1"), dict) else {}
    round1 = governance.get("round1_responses") if isinstance(governance.get("round1_responses"), list) else []
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    return bool(runtime.get("engine") == "governance_v1_committee" and round1 and decision)


def _extract_decision(raw_row: dict[str, Any]) -> dict[str, Any]:
    day = str(raw_row.get("date") or "")
    if not day:
        raise ValueError("Raw replay row is missing date.")
    result = raw_row.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{day}: raw replay result is not an object.")
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    governance = result.get("governance_v1") if isinstance(result.get("governance_v1"), dict) else {}
    final_decision = (
        governance.get("final_decision") if isinstance(governance.get("final_decision"), dict) else {}
    )
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    round1 = governance.get("round1_responses") if isinstance(governance.get("round1_responses"), list) else []
    round2 = governance.get("round2_responses") if isinstance(governance.get("round2_responses"), list) else []
    notification = decision.get("user_notification") if isinstance(decision.get("user_notification"), dict) else {}
    raw_plan = decision.get("candidate_rebalance_plan") if isinstance(decision.get("candidate_rebalance_plan"), dict) else {}
    candidate_plan: dict[str, float] = {}
    for ticker, delta in raw_plan.items():
        candidate_plan[str(ticker)] = float(delta)
    return {
        "date": day,
        "decision": decision.get("decision"),
        "branch": final_decision.get("branch"),
        "summary": decision.get("summary"),
        "confidence": decision.get("confidence"),
        "urgency": decision.get("urgency"),
        "candidate_rebalance_plan": candidate_plan,
        "committee_trades": final_decision.get("trades") or [],
        "user_handoff": bool(notification.get("action_required"))
        or decision.get("decision") == "USER_DECISION_REQUIRED"
        or bool(final_decision.get("user_question")),
        "called_agents": decision.get("called_agents") or [],
        "round1_agents": [str(item.get("agent_id")) for item in round1 if isinstance(item, dict)],
        "round2_agents": [str(item.get("agent_id")) for item in round2 if isinstance(item, dict)],
        "runtime_engine": runtime.get("engine"),
        "round1_agent_count": runtime.get("round1_agent_count"),
        "round2_agent_count": runtime.get("round2_agent_count"),
        "trace_complete": _trace_complete(result),
    }


def build_replay_fixture(
    source_fixture: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    *,
    require_full: bool,
) -> dict[str, Any]:
    prices = _price_rows(source_fixture)
    fixture_dates = [str(row["date"]) for row in prices]
    raw_dates = [str(row.get("date")) for row in raw_rows]
    if raw_dates != fixture_dates[: len(raw_dates)]:
        raise ValueError("Replay raw dates do not match the source fixture date prefix.")
    if require_full and len(raw_rows) != len(prices):
        raise ValueError(f"Replay is incomplete: {len(raw_rows)} raw rows != {len(prices)} fixture rows.")
    decisions = [_extract_decision(row) for row in raw_rows]
    replay_fixture = deepcopy(source_fixture)
    replay_fixture["prices"] = prices[: len(raw_rows)]
    replay_fixture["libra_decisions"] = decisions
    replay_fixture["replay_validation"] = {
        "raw_rows": len(raw_rows),
        "source_fixture_rows": len(prices),
        "full_match": len(raw_rows) == len(prices),
        "first_date": raw_dates[0] if raw_dates else None,
        "last_date": raw_dates[-1] if raw_dates else None,
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return replay_fixture


def simulate_buy_and_hold(fixture: dict[str, Any]) -> dict[str, Any]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    value_history = [_total_value(positions, row) for row in prices]
    return _summary(
        name="Buy & Hold",
        group="baseline",
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=0,
        turnover=0.0,
        transaction_cost=0.0,
        annualization_factor=_annualization_factor(fixture),
    )


def simulate_quarterly_rebalancing(fixture: dict[str, Any]) -> dict[str, Any]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    target_weights = _target_weights(fixture)
    cost_rate = _cost_rate(fixture)
    explicit_dates = {str(value) for value in fixture.get("quarterly_rebalance_dates", [])}
    trades = 0
    turnover = 0.0
    transaction_cost = 0.0
    rebalance_dates: list[str] = []
    value_history = [_total_value(positions, prices[0])]
    for index, row in enumerate(prices[1:], start=1):
        should_rebalance = str(row["date"]) in explicit_dates if explicit_dates else _is_last_observed_date_of_quarter(index, prices)
        if should_rebalance:
            positions, step_turnover, step_cost = _rebalance_to_weights(positions, row, target_weights, cost_rate)
            trades += 1
            turnover += step_turnover
            transaction_cost += step_cost
            rebalance_dates.append(str(row["date"]))
        value_history.append(_total_value(positions, row))
    return _summary(
        name="Quarterly Rebalancing",
        group="baseline",
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=trades,
        turnover=turnover,
        transaction_cost=transaction_cost,
        annualization_factor=_annualization_factor(fixture),
        parameters={"rebalance_dates": rebalance_dates},
    )


def simulate_threshold_band_rebalancing(fixture: dict[str, Any]) -> dict[str, Any]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    target_weights = _target_weights(fixture)
    threshold = float(fixture.get("threshold_band", fixture.get("mechanical_drift_threshold", 0.05)))
    cost_rate = _cost_rate(fixture)
    trades = 0
    turnover = 0.0
    transaction_cost = 0.0
    rebalance_dates: list[str] = []
    value_history = [_total_value(positions, prices[0])]
    for row in prices[1:]:
        current_weights = _weights(positions, row)
        max_drift = max(abs(current_weights[ticker] - target_weights[ticker]) for ticker in target_weights)
        if max_drift >= threshold:
            positions, step_turnover, step_cost = _rebalance_to_weights(positions, row, target_weights, cost_rate)
            trades += 1
            turnover += step_turnover
            transaction_cost += step_cost
            rebalance_dates.append(str(row["date"]))
        value_history.append(_total_value(positions, row))
    return _summary(
        name="Threshold 5% Rebalancing",
        group="baseline",
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=trades,
        turnover=turnover,
        transaction_cost=transaction_cost,
        annualization_factor=_annualization_factor(fixture),
        parameters={"threshold": threshold, "rebalance_dates": rebalance_dates},
    )


def simulate_monthly_inverse_vol_risk_parity(fixture: dict[str, Any]) -> dict[str, Any]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    tickers = _tickers(fixture)
    lookback_days = int(fixture.get("risk_parity_lookback_days", 60))
    cost_rate = _cost_rate(fixture)
    explicit_dates = {str(value) for value in fixture.get("risk_parity_rebalance_dates", [])}
    trades = 0
    turnover = 0.0
    transaction_cost = 0.0
    rebalance_dates: list[str] = []
    value_history = [_total_value(positions, prices[0])]
    for index, row in enumerate(prices[1:], start=1):
        should_rebalance = str(row["date"]) in explicit_dates if explicit_dates else _is_last_observed_date_of_month(index, prices)
        if should_rebalance:
            start = max(0, index - lookback_days)
            target = _inverse_vol_weights(prices[start : index + 1], tickers)
            positions, step_turnover, step_cost = _rebalance_to_weights(positions, row, target, cost_rate)
            trades += 1
            turnover += step_turnover
            transaction_cost += step_cost
            rebalance_dates.append(str(row["date"]))
        value_history.append(_total_value(positions, row))
    return _summary(
        name="Monthly Risk Parity",
        group="baseline",
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=trades,
        turnover=turnover,
        transaction_cost=transaction_cost,
        annualization_factor=_annualization_factor(fixture),
        parameters={"lookback_days": lookback_days, "rebalance_dates": rebalance_dates},
    )


def _mechanical_trade_dates(fixture: dict[str, Any]) -> set[str]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    target_weights = _target_weights(fixture)
    threshold = float(fixture.get("threshold_band", fixture.get("mechanical_drift_threshold", 0.05)))
    trade_dates: set[str] = set()
    for row in prices[1:]:
        current_weights = _weights(positions, row)
        max_drift = max(abs(current_weights[ticker] - target_weights[ticker]) for ticker in target_weights)
        if max_drift >= threshold:
            trade_dates.add(str(row["date"]))
            positions, _turnover, _cost = _rebalance_to_weights(positions, row, target_weights, 0.0)
    return trade_dates


def _decision_counts(fixture: dict[str, Any]) -> tuple[int, int, int, int]:
    decisions = fixture.get("libra_decisions", [])
    trace_complete_count = sum(1 for decision in decisions if decision.get("trace_complete"))
    user_handoff_count = sum(
        1
        for decision in decisions
        if decision.get("user_handoff") or decision.get("decision") == "USER_DECISION_REQUIRED"
    )
    rebalance_count = sum(
        1
        for decision in decisions
        if decision.get("decision") == "REBALANCE" and decision.get("candidate_rebalance_plan")
    )
    return len(decisions), trace_complete_count, user_handoff_count, rebalance_count


def _target_from_candidate_plan(
    positions: dict[str, float],
    row: dict[str, Any],
    base_weights: dict[str, float],
    plan: dict[str, float],
) -> dict[str, float]:
    target = deepcopy(_weights(positions, row))
    for ticker in base_weights:
        target.setdefault(ticker, 0.0)
    for ticker, delta in plan.items():
        if ticker in target:
            target[ticker] = target.get(ticker, 0.0) + float(delta)
    return _normalize_weights(target)


def simulate_libra_immediate(fixture: dict[str, Any]) -> dict[str, Any]:
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    base_weights = _target_weights(fixture)
    cost_rate = _cost_rate(fixture)
    decisions_by_date: dict[str, list[dict[str, Any]]] = {}
    for decision in fixture.get("libra_decisions", []):
        decisions_by_date.setdefault(str(decision.get("date")), []).append(decision)

    trades = 0
    turnover = 0.0
    transaction_cost = 0.0
    executed_dates: list[str] = []
    mechanical_dates = _mechanical_trade_dates(fixture)
    avoided_trade_count = 0
    for decision in fixture.get("libra_decisions", []):
        if decision.get("decision") in {"HOLD", "DEFER"} and str(decision.get("date")) in mechanical_dates:
            avoided_trade_count += 1

    value_history = [_total_value(positions, prices[0])]
    for row in prices[1:]:
        for decision in decisions_by_date.get(str(row["date"]), []):
            if decision.get("decision") != "REBALANCE":
                continue
            plan = {str(ticker): float(delta) for ticker, delta in decision.get("candidate_rebalance_plan", {}).items()}
            if not plan:
                continue
            target = _target_from_candidate_plan(positions, row, base_weights, plan)
            positions, step_turnover, step_cost = _rebalance_to_weights(positions, row, target, cost_rate)
            trades += 1
            turnover += step_turnover
            transaction_cost += step_cost
            executed_dates.append(str(row["date"]))
        value_history.append(_total_value(positions, row))

    decision_count, trace_complete_count, user_handoff_count, _rebalance_count = _decision_counts(fixture)
    return _summary(
        name="LIBRA v1 Immediate Execution",
        group="libra_v1",
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=trades,
        turnover=turnover,
        transaction_cost=transaction_cost,
        annualization_factor=_annualization_factor(fixture),
        decision_count=decision_count,
        trace_complete_count=trace_complete_count,
        user_handoff_count=user_handoff_count,
        avoided_trade_count=avoided_trade_count,
        parameters={"executed_dates": executed_dates, "execution_target": "candidate_rebalance_plan"},
    )


def _execute_rebalance_target(
    *,
    positions: dict[str, float],
    row: dict[str, Any],
    base_weights: dict[str, float],
    cost_rate: float,
    execution_target: str,
    source_plan: dict[str, float] | None,
) -> tuple[dict[str, float], float, float]:
    if execution_target == "policy_weights":
        target = base_weights
    elif execution_target == "candidate_rebalance_plan":
        if not source_plan:
            return positions, 0.0, 0.0
        target = _target_from_candidate_plan(positions, row, base_weights, source_plan)
    else:
        raise ValueError(f"Unsupported execution target: {execution_target}")
    return _rebalance_to_weights(positions, row, target, cost_rate)


def simulate_execution_policy(
    fixture: dict[str, Any],
    *,
    delay_days: int,
    threshold: float,
    mode: str,
    confirmation_execution_lag_days: int,
    execution_target: str,
) -> dict[str, Any]:
    if mode not in {"delayed_execution", "confirmation_gate"}:
        raise ValueError(f"Unsupported execution policy mode: {mode}")
    prices = _price_rows(fixture)
    positions = _initial_positions(fixture)
    base_weights = _target_weights(fixture)
    cost_rate = _cost_rate(fixture)
    decisions_by_date: dict[str, list[dict[str, Any]]] = {}
    for decision in fixture.get("libra_decisions", []):
        decisions_by_date.setdefault(str(decision.get("date")), []).append(decision)

    pending_confirmation_index: int | None = None
    pending_source_date: str | None = None
    pending_source_plan: dict[str, float] | None = None
    scheduled_execution_index: int | None = None
    scheduled_source_date: str | None = None
    scheduled_source_plan: dict[str, float] | None = None
    trades = 0
    turnover = 0.0
    transaction_cost = 0.0
    trigger_dates: list[str] = []
    confirmation_dates: list[str] = []
    executed_dates: list[str] = []
    skipped_due_dates: list[dict[str, Any]] = []
    value_history = [_total_value(positions, prices[0])]

    for index, row in enumerate(prices[1:], start=1):
        if scheduled_execution_index is not None and index >= scheduled_execution_index:
            positions, step_turnover, step_cost = _execute_rebalance_target(
                positions=positions,
                row=row,
                base_weights=base_weights,
                cost_rate=cost_rate,
                execution_target=execution_target,
                source_plan=scheduled_source_plan,
            )
            if step_turnover > 0:
                trades += 1
                turnover += step_turnover
                transaction_cost += step_cost
                executed_dates.append(str(row["date"]))
            scheduled_execution_index = None
            scheduled_source_date = None
            scheduled_source_plan = None

        if pending_confirmation_index is not None and index >= pending_confirmation_index:
            current_weights = _weights(positions, row)
            max_drift = max(abs(current_weights[ticker] - base_weights[ticker]) for ticker in base_weights)
            confirmation_dates.append(str(row["date"]))
            if max_drift >= threshold:
                if confirmation_execution_lag_days <= 0:
                    positions, step_turnover, step_cost = _execute_rebalance_target(
                        positions=positions,
                        row=row,
                        base_weights=base_weights,
                        cost_rate=cost_rate,
                        execution_target=execution_target,
                        source_plan=pending_source_plan,
                    )
                    if step_turnover > 0:
                        trades += 1
                        turnover += step_turnover
                        transaction_cost += step_cost
                        executed_dates.append(str(row["date"]))
                else:
                    scheduled_execution_index = min(index + confirmation_execution_lag_days, len(prices) - 1)
                    scheduled_source_date = pending_source_date
                    scheduled_source_plan = pending_source_plan
            else:
                skipped_due_dates.append(
                    {
                        "source_date": pending_source_date,
                        "confirmation_date": str(row["date"]),
                        "max_drift": round(max_drift, 8),
                    }
                )
            pending_confirmation_index = None
            pending_source_date = None
            pending_source_plan = None

        if pending_confirmation_index is None and scheduled_execution_index is None:
            for decision in decisions_by_date.get(str(row["date"]), []):
                if decision.get("decision") != "REBALANCE" or not decision.get("candidate_rebalance_plan"):
                    continue
                source_plan = {
                    str(ticker): float(delta)
                    for ticker, delta in decision.get("candidate_rebalance_plan", {}).items()
                }
                trigger_dates.append(str(row["date"]))
                if mode == "delayed_execution":
                    scheduled_execution_index = min(index + delay_days, len(prices) - 1)
                    scheduled_source_date = str(row["date"])
                    scheduled_source_plan = source_plan
                else:
                    pending_confirmation_index = min(index + delay_days, len(prices) - 1)
                    pending_source_date = str(row["date"])
                    pending_source_plan = source_plan
                break

        value_history.append(_total_value(positions, row))

    decision_count, trace_complete_count, user_handoff_count, _rebalance_count = _decision_counts(fixture)
    if mode == "delayed_execution":
        strategy_name = f"LIBRA-v3 T+{delay_days} Delayed Execution"
        group = "libra_v3_delayed_execution"
    else:
        strategy_name = f"LIBRA-v3 T+{delay_days} Confirmation Gate"
        group = "libra_v3_confirmation_gate"
    return _summary(
        name=strategy_name,
        group=group,
        value_history=value_history,
        initial_value=float(fixture["initial_value_krw"]),
        trades=trades,
        turnover=turnover,
        transaction_cost=transaction_cost,
        annualization_factor=_annualization_factor(fixture),
        decision_count=decision_count,
        trace_complete_count=trace_complete_count,
        user_handoff_count=user_handoff_count,
        avoided_trade_count=len(skipped_due_dates),
        parameters={
            "delay_days": delay_days,
            "threshold": threshold,
            "mode": mode,
            "confirmation_execution_lag_days": confirmation_execution_lag_days if mode == "confirmation_gate" else 0,
            "execution_target": execution_target,
            "trigger_dates": trigger_dates,
            "confirmation_dates": confirmation_dates,
            "executed_dates": executed_dates,
            "skipped_due_dates": skipped_due_dates,
            "scheduled_source_date": scheduled_source_date,
            "look_ahead_bias_control": (
                "Confirmation observes data through T+N and executes after the configured lag; "
                "the default lag=1 avoids same-close confirmation fills."
                if mode == "confirmation_gate"
                else "Delayed execution changes timing only and is reported as sensitivity, not confirmation."
            ),
        },
    )


def _decision_breakdown(fixture: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in fixture.get("libra_decisions", []):
        key = str(decision.get("decision"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _find_strategy(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("strategy") == strategy:
            return row
    return None


def build_results(
    fixture: dict[str, Any],
    *,
    source_fixture_path: Path,
    raw_path: Path,
    main_delay_days: int,
    sensitivity_delay_days: list[int],
    threshold: float,
    confirmation_execution_lag_days: int,
    execution_target: str,
) -> dict[str, Any]:
    baselines = [
        simulate_buy_and_hold(fixture),
        simulate_quarterly_rebalancing(fixture),
        simulate_threshold_band_rebalancing(fixture),
        simulate_monthly_inverse_vol_risk_parity(fixture),
        simulate_libra_immediate(fixture),
    ]
    candidates = [
        simulate_execution_policy(
            fixture,
            delay_days=main_delay_days,
            threshold=threshold,
            mode="confirmation_gate",
            confirmation_execution_lag_days=confirmation_execution_lag_days,
            execution_target=execution_target,
        )
    ]
    for delay in sensitivity_delay_days:
        candidates.append(
            simulate_execution_policy(
                fixture,
                delay_days=delay,
                threshold=threshold,
                mode="confirmation_gate",
                confirmation_execution_lag_days=confirmation_execution_lag_days,
                execution_target=execution_target,
            )
        )
        candidates.append(
            simulate_execution_policy(
                fixture,
                delay_days=delay,
                threshold=threshold,
                mode="delayed_execution",
                confirmation_execution_lag_days=confirmation_execution_lag_days,
                execution_target=execution_target,
            )
        )

    rows = baselines + candidates
    libra_v1 = _find_strategy(rows, "LIBRA v1 Immediate Execution")
    threshold_row = _find_strategy(rows, "Threshold 5% Rebalancing")
    quarterly_row = _find_strategy(rows, "Quarterly Rebalancing")
    main_v3 = _find_strategy(rows, f"LIBRA-v3 T+{main_delay_days} Confirmation Gate")
    if libra_v1:
        for row in rows:
            row["return_gap_vs_libra_v1_pct_points"] = round(
                float(row["total_return_pct"]) - float(libra_v1["total_return_pct"]),
                3,
            )
            row["ending_gap_vs_libra_v1_krw"] = round(
                float(row["ending_value_krw"]) - float(libra_v1["ending_value_krw"]),
                2,
            )

    performance_checks: dict[str, Any] = {}
    if main_v3 and libra_v1 and threshold_row and quarterly_row:
        performance_checks = {
            "main_strategy": main_v3["strategy"],
            "beats_libra_v1_return": float(main_v3["total_return_pct"]) > float(libra_v1["total_return_pct"]),
            "beats_threshold_return": float(main_v3["total_return_pct"]) > float(threshold_row["total_return_pct"]),
            "beats_quarterly_return": float(main_v3["total_return_pct"]) > float(quarterly_row["total_return_pct"]),
            "sharpe_at_least_threshold": (
                main_v3["sharpe_ratio"] is not None
                and threshold_row["sharpe_ratio"] is not None
                and float(main_v3["sharpe_ratio"]) >= float(threshold_row["sharpe_ratio"])
            ),
            "mdd_not_worse_than_threshold": float(main_v3["max_drawdown_pct"]) >= float(threshold_row["max_drawdown_pct"]),
            "cost_not_worse_than_threshold": float(main_v3["transaction_cost_krw"]) <= float(threshold_row["transaction_cost_krw"]),
            "trades_not_worse_than_threshold": int(main_v3["trades"]) <= int(threshold_row["trades"]),
            "is_performance_proof": False,
        }
    ranked = sorted(rows, key=lambda item: float(item["ending_value_krw"]), reverse=True)
    decision_count, trace_complete_count, user_handoff_count, rebalance_count = _decision_counts(fixture)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_fixture": str(source_fixture_path),
        "raw_replay": str(raw_path),
        "price_rows": len(_price_rows(fixture)),
        "decision_count": decision_count,
        "decision_breakdown": _decision_breakdown(fixture),
        "rebalance_count": rebalance_count,
        "user_handoff_count": user_handoff_count,
        "trace_complete_count": trace_complete_count,
        "main_policy": {
            "name": f"LIBRA-v3 T+{main_delay_days} Confirmation Gate",
            "delay_days": main_delay_days,
            "confirmation_execution_lag_days": confirmation_execution_lag_days,
            "threshold": threshold,
            "execution_target": execution_target,
            "pre_registered_role": (
                "sensitivity reference only; not a validated product policy unless a "
                "separate in-loop replay is run with this execution rule"
            ),
        },
        "baselines": baselines,
        "candidates": candidates,
        "ranked_by_ending_value": ranked,
        "performance_checks": performance_checks,
        "interpretation_guardrails": [
            "The replay raw is the single LLM decision source; strategy rows do not rerun Claude.",
            "LIBRA v1 Immediate Execution uses final decision candidate_rebalance_plan deltas on the signal date.",
            "LIBRA-v3 Confirmation Gate keeps the REBALANCE signal fixed, observes residual drift after T+N, and executes after the configured lag.",
            "T+N rows are sensitivity analysis only. They are not valid proof of a v3 service backtest because later LLM inputs would differ after skipped or delayed executions.",
            "Do not present any T+N policy as final unless it is replayed in-loop so portfolio state changes feed into subsequent LLM calls.",
            "Delayed Execution rows are timing-only sensitivity checks and should not be presented as confirmation.",
            "Risk Parity is a high-turnover quantitative benchmark, not the product's low-frequency target behavior.",
        ],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "strategy",
        "group",
        "ending_value_krw",
        "total_return_pct",
        "cagr_pct",
        "annualized_volatility_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "trades",
        "turnover_krw",
        "transaction_cost_krw",
        "trace_completeness_pct",
        "user_handoff_count",
        "avoided_trade_count",
        "return_gap_vs_libra_v1_pct_points",
        "ending_gap_vs_libra_v1_krw",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _write_md(payload: dict[str, Any], path: Path) -> None:
    headers = [
        "strategy",
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "trades",
        "transaction_cost_krw",
        "return_gap_vs_libra_v1_pct_points",
    ]
    checks = payload.get("performance_checks", {})
    lines = [
        "# Replay Strategy Evaluation",
        "",
        f"Raw replay: `{payload['raw_replay']}`",
        f"Source fixture: `{payload['source_fixture']}`",
        "",
        "## Sensitivity Reference",
        "",
        f"- {payload['main_policy']['name']}",
        "- Status: not a final validated policy",
        f"- Confirmation lag: {payload['main_policy']['confirmation_execution_lag_days']} trading day(s)",
        f"- Execution target: `{payload['main_policy']['execution_target']}`",
        "",
        "## Reference Checks",
        "",
    ]
    if checks:
        for key, value in checks.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "## Ranked Results",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
    )
    for row in payload["ranked_by_ending_value"]:
        lines.append("| " + " | ".join(_format(row.get(header)) for header in headers) + " |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in payload["interpretation_guardrails"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one replay raw file against baselines and LIBRA-v3 execution policies without rerunning LLMs."
    )
    parser.add_argument("--raw", required=True, help="Replay raw JSONL from replay_full_committee_backtest.py.")
    parser.add_argument("--fixture", required=True, help="Source comparison fixture JSON.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-fixture", help="Optional fixture JSON with extracted libra_decisions.")
    parser.add_argument("--require-full", action="store_true", help="Require raw rows to cover the full source fixture.")
    parser.add_argument("--main-delay-days", type=int, default=2)
    parser.add_argument("--sensitivity-delay-days", type=int, nargs="*", default=[1, 3])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--confirmation-execution-lag-days",
        type=int,
        default=1,
        help="Trading-day lag between confirmation observation and execution. Default 1 avoids same-close confirmation fills.",
    )
    parser.add_argument(
        "--execution-target",
        choices=("policy_weights", "candidate_rebalance_plan"),
        default="policy_weights",
        help="Target used by v3 execution after confirmation. policy_weights mirrors the product execution gate.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result payload as JSON.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    raw_path = Path(args.raw)
    source_fixture_path = Path(args.fixture)
    source_fixture = _read_json(source_fixture_path)
    raw_rows = _read_jsonl(raw_path)
    replay_fixture = build_replay_fixture(source_fixture, raw_rows, require_full=bool(args.require_full))
    if args.out_fixture:
        _write_json(Path(args.out_fixture), replay_fixture)
    payload = build_results(
        replay_fixture,
        source_fixture_path=source_fixture_path,
        raw_path=raw_path,
        main_delay_days=int(args.main_delay_days),
        sensitivity_delay_days=list(args.sensitivity_delay_days),
        threshold=float(args.threshold),
        confirmation_execution_lag_days=int(args.confirmation_execution_lag_days),
        execution_target=str(args.execution_target),
    )
    _write_json(Path(args.out_json), payload)
    _write_csv(payload["ranked_by_ending_value"], Path(args.out_csv))
    _write_md(payload, Path(args.out_md))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote {args.out_json}")
        print(f"Wrote {args.out_csv}")
        print(f"Wrote {args.out_md}")
        if args.out_fixture:
            print(f"Wrote {args.out_fixture}")


if __name__ == "__main__":
    main()
