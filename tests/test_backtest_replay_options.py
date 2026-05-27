from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BacktestReplayOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replay = _load_script_module("replay_full_committee_backtest_script", "scripts/replay_full_committee_backtest.py")
        cls.evaluate = _load_script_module("evaluate_replay_strategies_script", "scripts/evaluate_replay_strategies.py")

    def test_decision_schedule_every_n_trading_days(self) -> None:
        rows = [
            {"prices_until": "2021-01-04"},
            {"prices_until": "2021-01-05"},
            {"prices_until": "2021-01-06"},
            {"prices_until": "2021-01-07"},
            {"prices_until": "2021-01-08"},
        ]
        args = argparse.Namespace(decision_frequency="every-n-trading-days", decision_interval=2)

        self.assertEqual(
            self.replay._decision_schedule(rows, args),
            {"2021-01-04", "2021-01-06", "2021-01-08"},
        )

    def test_decision_schedule_weekly_uses_first_observed_trading_day(self) -> None:
        rows = [
            {"prices_until": "2021-01-04"},
            {"prices_until": "2021-01-05"},
            {"prices_until": "2021-01-08"},
            {"prices_until": "2021-01-11"},
            {"prices_until": "2021-01-12"},
        ]
        args = argparse.Namespace(decision_frequency="weekly", decision_interval=1)

        self.assertEqual(
            self.replay._decision_schedule(rows, args),
            {"2021-01-04", "2021-01-11"},
        )

    def test_evaluation_accepts_contiguous_mid_fixture_range(self) -> None:
        source_fixture = {
            "initial_value_krw": 1_000_000,
            "target_weights": {"A": 0.6, "B": 0.4},
            "prices": [
                {"date": "2021-01-04", "A": 100, "B": 100},
                {"date": "2021-01-05", "A": 101, "B": 99},
                {"date": "2021-01-06", "A": 102, "B": 98},
                {"date": "2021-01-07", "A": 103, "B": 97},
            ],
        }
        raw_rows = [
            {
                "date": "2021-01-06",
                "result": {
                    "decision": {"decision": "HOLD"},
                    "governance_v1": {"final_decision": {"branch": "STRONG_HOLD"}},
                    "runtime": {"engine": "governance_v1_committee"},
                },
            },
            {
                "date": "2021-01-07",
                "result": {
                    "decision": {"decision": "DEFER"},
                    "governance_v1": {"final_decision": {"branch": "NO_EXECUTABLE_TRADE"}},
                    "runtime": {"engine": "governance_v1_committee"},
                },
            },
        ]

        replay_fixture = self.evaluate.build_replay_fixture(
            source_fixture,
            raw_rows,
            require_full=True,
            start_date="2021-01-06",
            end_date="2021-01-07",
        )

        self.assertEqual([row["date"] for row in replay_fixture["prices"]], ["2021-01-06", "2021-01-07"])
        self.assertEqual(replay_fixture["replay_validation"]["source_start_index"], 2)
        self.assertTrue(replay_fixture["replay_validation"]["selected_range_full_match"])


if __name__ == "__main__":
    unittest.main()
