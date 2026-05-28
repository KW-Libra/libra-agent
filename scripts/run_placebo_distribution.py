"""Generate placebo distribution by running N seeds of same-count random placebo.

LLM-free post-hoc analysis. Reads existing raw + fixture, generates placebo rows
for seed 1..N, then reports where LIBRA v2 execution-only lands inside the
placebo return/Sharpe distribution.

This tests whether LIBRA's trigger timing matters beyond random chance.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_replay_strategies import (  # noqa: E402
    _read_json,
    _read_jsonl,
    _write_json,
    build_replay_fixture,
    simulate_libra_v2_execution_only,
    simulate_same_count_random_placebo,
)


def _percentile_below(sorted_values: list[float], target: float) -> float:
    n = len(sorted_values)
    below = sum(1 for x in sorted_values if x < target)
    return 100.0 * below / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--execution-mode", default="policy_target")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    fixture_path = Path(args.fixture)
    source_fixture = _read_json(fixture_path)
    raw_rows = _read_jsonl(raw_path)
    fixture = build_replay_fixture(source_fixture, raw_rows, require_full=True)

    libra_v2 = simulate_libra_v2_execution_only(fixture, execution_mode=args.execution_mode)
    libra_return = float(libra_v2["total_return_pct"])
    libra_sharpe = float(libra_v2["sharpe_ratio"])
    libra_mdd = float(libra_v2["max_drawdown_pct"])
    libra_trades = int(libra_v2["trades"])
    libra_cost = float(libra_v2["transaction_cost_krw"])

    rows = []
    for seed in range(1, args.seeds + 1):
        row = simulate_same_count_random_placebo(
            fixture,
            execution_mode=args.execution_mode,
            seed=seed,
        )
        rows.append(
            {
                "seed": seed,
                "total_return_pct": float(row["total_return_pct"]),
                "sharpe_ratio": float(row["sharpe_ratio"]),
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "trades": int(row["trades"]),
                "transaction_cost_krw": float(row["transaction_cost_krw"]),
            }
        )

    returns = sorted(r["total_return_pct"] for r in rows)
    sharpes = sorted(r["sharpe_ratio"] for r in rows)
    n = len(returns)

    def _p(values: list[float], q: float) -> float:
        idx = max(0, min(n - 1, int(round(q * (n - 1)))))
        return values[idx]

    summary = {
        "config": {
            "raw": str(raw_path),
            "fixture": str(fixture_path),
            "seeds": args.seeds,
            "execution_mode": args.execution_mode,
        },
        "libra_v2_execution_only": {
            "total_return_pct": libra_return,
            "sharpe_ratio": libra_sharpe,
            "max_drawdown_pct": libra_mdd,
            "trades": libra_trades,
            "transaction_cost_krw": libra_cost,
        },
        "placebo_return_distribution_pct": {
            "min": min(returns),
            "p05": _p(returns, 0.05),
            "p25": _p(returns, 0.25),
            "median": statistics.median(returns),
            "mean": statistics.mean(returns),
            "p75": _p(returns, 0.75),
            "p95": _p(returns, 0.95),
            "max": max(returns),
            "stdev": statistics.stdev(returns) if n > 1 else 0.0,
        },
        "placebo_sharpe_distribution": {
            "min": min(sharpes),
            "p05": _p(sharpes, 0.05),
            "median": statistics.median(sharpes),
            "mean": statistics.mean(sharpes),
            "p95": _p(sharpes, 0.95),
            "max": max(sharpes),
        },
        "libra_percentile_in_placebo": {
            "return_percentile": _percentile_below(returns, libra_return),
            "return_seeds_below": sum(1 for x in returns if x < libra_return),
            "return_seeds_above_or_equal": sum(1 for x in returns if x >= libra_return),
            "sharpe_percentile": _percentile_below(sharpes, libra_sharpe),
        },
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_json, summary)

    out_csv = Path(args.out_csv)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("seed,total_return_pct,sharpe_ratio,max_drawdown_pct,trades,transaction_cost_krw\n")
        for r in rows:
            f.write(
                f"{r['seed']},{r['total_return_pct']},{r['sharpe_ratio']},"
                f"{r['max_drawdown_pct']},{r['trades']},{r['transaction_cost_krw']}\n"
            )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(
        f"LIBRA v2 ({args.execution_mode}) return={libra_return:.3f}% "
        f"→ placebo percentile={summary['libra_percentile_in_placebo']['return_percentile']:.1f} "
        f"(seeds_below={summary['libra_percentile_in_placebo']['return_seeds_below']}/{n})"
    )
    print(
        f"LIBRA sharpe={libra_sharpe:.3f} "
        f"→ placebo percentile={summary['libra_percentile_in_placebo']['sharpe_percentile']:.1f}"
    )
    print(
        f"Placebo return mean={summary['placebo_return_distribution_pct']['mean']:.3f}%, "
        f"stdev={summary['placebo_return_distribution_pct']['stdev']:.3f}%, "
        f"range=[{summary['placebo_return_distribution_pct']['min']:.3f}, "
        f"{summary['placebo_return_distribution_pct']['max']:.3f}]"
    )


if __name__ == "__main__":
    main()
