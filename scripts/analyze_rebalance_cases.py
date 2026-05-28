"""Post-hoc case study of LIBRA REBALANCE events.

For each REBALANCE decision, capture:
- date, consensus/divergence
- candidate_rebalance_plan (deltas)
- pre-trade portfolio weights and drift
- post-trade price moves (T+5, T+20) of the trimmed asset vs portfolio average
  to evaluate whether the trim direction was correct in hindsight.

LLM-free analysis using only raw + fixture.
"""

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _price_index(price_rows):
    return {str(row["date"]): row for row in price_rows}


def _trading_dates(price_rows):
    return [str(row["date"]) for row in price_rows]


def _date_offset(dates: list[str], anchor: str, offset: int) -> str | None:
    try:
        i = dates.index(anchor)
    except ValueError:
        return None
    j = i + offset
    if 0 <= j < len(dates):
        return dates[j]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    fixture_path = Path(args.fixture)
    fixture = _read_json(fixture_path)
    price_rows = fixture.get("price_rows") or fixture.get("prices") or []
    if not price_rows:
        # try other shapes
        if isinstance(fixture.get("price_history"), list):
            price_rows = fixture["price_history"]
    price_by_date = _price_index(price_rows)
    dates = _trading_dates(price_rows)
    tickers = sorted({k for row in price_rows for k in row.keys() if k not in {"date", "volumes"}})

    cases = []
    for j in _read_jsonl(raw_path):
        d = j["result"]["decision"]
        if d.get("decision") != "REBALANCE":
            continue
        date = str(j["date"])
        plan = d.get("candidate_rebalance_plan") or {}
        # plan deltas are weight adjustments per ticker
        deltas = {t: float(v) for t, v in plan.items() if isinstance(v, (int, float))}
        if not deltas:
            continue
        # primary trim = largest negative delta (most reduced asset)
        trim_ticker, trim_delta = min(deltas.items(), key=lambda kv: kv[1])
        # forward returns of the trimmed asset vs basket average
        results_per_horizon = {}
        for horizon in (5, 10, 20, 60):
            future_date = _date_offset(dates, date, horizon)
            if not future_date:
                results_per_horizon[f"T+{horizon}"] = None
                continue
            row_now = price_by_date.get(date) or {}
            row_fut = price_by_date.get(future_date) or {}
            # trimmed asset return
            try:
                p0 = float(row_now[trim_ticker])
                p1 = float(row_fut[trim_ticker])
                trim_ret = (p1 / p0 - 1.0) * 100.0
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                trim_ret = None
            # basket equal-weight return across all tickers in price_rows
            basket_rets = []
            for t in tickers:
                try:
                    p0 = float(row_now[t])
                    p1 = float(row_fut[t])
                    basket_rets.append((p1 / p0 - 1.0) * 100.0)
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    pass
            basket_ret = sum(basket_rets) / len(basket_rets) if basket_rets else None
            edge = (
                trim_ret - basket_ret
                if (trim_ret is not None and basket_ret is not None)
                else None
            )
            results_per_horizon[f"T+{horizon}"] = {
                "future_date": future_date,
                "trim_asset_return_pct": round(trim_ret, 3) if trim_ret is not None else None,
                "basket_return_pct": round(basket_ret, 3) if basket_ret is not None else None,
                "trim_vs_basket_pct_points": round(edge, 3) if edge is not None else None,
                "trim_correct": (edge < 0) if edge is not None else None,
            }
        cases.append(
            {
                "date": date,
                "consensus_score": d.get("consensus_score"),
                "divergence_score": d.get("divergence_score"),
                "trigger": d.get("trigger"),
                "plan_deltas_pct": {t: round(v * 100, 2) for t, v in deltas.items()},
                "primary_trim": {
                    "ticker": trim_ticker,
                    "delta_pct": round(trim_delta * 100, 2),
                },
                "forward": results_per_horizon,
            }
        )

    # summary: how many of N trims beat (i.e. trimmed asset underperformed) at each horizon
    horizons = ("T+5", "T+10", "T+20", "T+60")
    summary = {}
    for h in horizons:
        correct = sum(1 for c in cases if c["forward"].get(h) and c["forward"][h].get("trim_correct") is True)
        wrong = sum(1 for c in cases if c["forward"].get(h) and c["forward"][h].get("trim_correct") is False)
        avg_edge_values = [
            c["forward"][h]["trim_vs_basket_pct_points"]
            for c in cases
            if c["forward"].get(h) and c["forward"][h].get("trim_vs_basket_pct_points") is not None
        ]
        avg_edge = sum(avg_edge_values) / len(avg_edge_values) if avg_edge_values else None
        summary[h] = {
            "trim_correct_count": correct,
            "trim_wrong_count": wrong,
            "trim_correct_pct": round(100.0 * correct / max(1, correct + wrong), 1),
            "avg_trim_minus_basket_pct_points": round(avg_edge, 3) if avg_edge is not None else None,
            "interpretation": (
                "negative = trimmed asset underperformed basket on average "
                "= LIBRA's trim direction was correct in hindsight"
            ),
        }

    out = {
        "raw": str(raw_path),
        "fixture": str(fixture_path),
        "rebalance_case_count": len(cases),
        "horizon_summary": summary,
        "cases": cases,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# LIBRA REBALANCE Case Study")
    lines.append("")
    lines.append(f"- raw: `{raw_path.name}`")
    lines.append(f"- fixture: `{fixture_path.name}`")
    lines.append(f"- REBALANCE events: **{len(cases)}**")
    lines.append("")
    lines.append("## Horizon summary (was trim direction correct in hindsight?)")
    lines.append("")
    lines.append("| Horizon | correct | wrong | hit rate | avg (trim - basket) pct pt |")
    lines.append("|---|---:|---:|---:|---:|")
    for h in horizons:
        s = summary[h]
        avg = s["avg_trim_minus_basket_pct_points"]
        avg_str = f"{avg:+.2f}" if avg is not None else "n/a"
        lines.append(
            f"| {h} | {s['trim_correct_count']} | {s['trim_wrong_count']} | "
            f"{s['trim_correct_pct']:.1f}% | {avg_str} |"
        )
    lines.append("")
    lines.append("Negative `avg (trim - basket)` means the trimmed asset underperformed the "
                 "equal-weight basket — i.e. LIBRA's trim direction was correct in hindsight.")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    for i, c in enumerate(cases, 1):
        lines.append(f"### Case {i} — {c['date']}")
        lines.append("")
        lines.append(f"- consensus: {c['consensus_score']}, divergence: {c['divergence_score']}")
        delta_str = ", ".join(
            f"{t} {v:+.2f}%p" for t, v in sorted(c["plan_deltas_pct"].items(), key=lambda kv: kv[1])
        )
        lines.append(f"- plan deltas: {delta_str}")
        pt = c["primary_trim"]
        lines.append(f"- primary trim: **{pt['ticker']} {pt['delta_pct']:+.2f}%p**")
        lines.append("")
        lines.append("| horizon | trimmed return | basket return | edge | correct? |")
        lines.append("|---|---:|---:|---:|---:|")
        for h in horizons:
            fwd = c["forward"].get(h)
            if not fwd:
                lines.append(f"| {h} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {h} ({fwd['future_date']}) | "
                f"{fwd['trim_asset_return_pct']:+.2f}% | "
                f"{fwd['basket_return_pct']:+.2f}% | "
                f"{fwd['trim_vs_basket_pct_points']:+.2f}pp | "
                f"{'OK' if fwd['trim_correct'] else 'NO'} |"
            )
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cases: {len(cases)}")
    for h in horizons:
        s = summary[h]
        avg = s["avg_trim_minus_basket_pct_points"]
        avg_str = f"{avg:+.2f}pp" if avg is not None else "n/a"
        print(
            f"  {h}: trim-correct {s['trim_correct_count']}/"
            f"{s['trim_correct_count'] + s['trim_wrong_count']} "
            f"({s['trim_correct_pct']}%), avg edge {avg_str}"
        )


if __name__ == "__main__":
    main()
