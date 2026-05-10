from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_benchmark", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_benchmark
SPEC.loader.exec_module(run_benchmark)


class BenchmarkScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = run_benchmark.BenchmarkPaths(
            root=REPO_ROOT / "benchmarks",
            profiles=REPO_ROOT / "benchmarks" / "profiles.yaml",
            universe=REPO_ROOT / "benchmarks" / "stock_universe.yaml",
            scenarios=REPO_ROOT / "benchmarks" / "scenarios",
        )
        cls.profiles, cls.universe, cls.scenarios = run_benchmark.load_benchmark(cls.paths)

    def test_loads_ten_controlled_scenarios(self) -> None:
        self.assertEqual(len(self.scenarios), 10)
        self.assertIn("balanced_kr", self.profiles)
        self.assertIn("005930", self.universe)

    def test_declared_baselines_match_rule_functions(self) -> None:
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario["scenario_id"]):
                declared = scenario.get("baseline_decisions", {})
                computed = run_benchmark.apply_baselines(scenario)
                self.assertEqual(computed, declared)

    def test_agent_payload_contains_inline_knowledge_base(self) -> None:
        scenario = next(item for item in self.scenarios if item["scenario_id"] == "02_semiconductor_concentration")

        payload = run_benchmark.build_agent_payload(scenario, self.profiles, self.universe)

        self.assertEqual(payload["trigger"], "pull")
        self.assertEqual(payload["governance_v1"]["execution_mode"], "primary")
        self.assertEqual(payload["portfolio"]["holdings"][0]["ticker"], "005930")
        self.assertEqual(payload["portfolio"]["holdings"][0]["sector"], "반도체")
        self.assertEqual(payload["portfolio"]["holdings"][0]["esg_score"], 78)
        self.assertTrue(
            any(item.startswith("max_single_weight=") for item in payload["portfolio"]["user_preferences"])
        )
        self.assertGreater(len(payload["knowledge_base"]["events"]), 0)

    def test_portfolio_definition_targets_are_normalized_when_enabled(self) -> None:
        enabled = [item for item in self.scenarios if item.get("portfolio_definition_enabled")]
        self.assertGreaterEqual(len(enabled), 3)
        for scenario in enabled:
            with self.subTest(scenario=scenario["scenario_id"]):
                definition = run_benchmark.build_portfolio_definition(scenario, self.universe)
                self.assertIsNotNone(definition)
                assert definition is not None
                total = sum(float(item["weight"]) for item in definition["target_weights"])
                self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
