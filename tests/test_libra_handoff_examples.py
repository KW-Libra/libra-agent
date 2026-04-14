from __future__ import annotations

import json
import unittest
from pathlib import Path

from libra_agent.libra_models import PortfolioSnapshot
from libra_agent.libra_validation import sanitize_agent_response_payload


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "agent-responses"
PORTFOLIO_PATH = ROOT / "examples" / "portfolio.sample.json"


class LibraHandoffExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio = PortfolioSnapshot.from_dict(json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8")))

    def test_agent_response_examples_match_runtime_shape(self) -> None:
        expected_files = {
            "disclosure.sample.json": "disclosure",
            "news.sample.json": "news",
            "report.sample.json": "report",
            "profit.sample.json": "profit",
            "cost.sample.json": "cost",
        }

        for filename, agent_id in expected_files.items():
            with self.subTest(filename=filename):
                payload = json.loads((EXAMPLE_DIR / filename).read_text(encoding="utf-8"))
                response = sanitize_agent_response_payload(
                    payload,
                    agent_id=agent_id,
                    portfolio=self.portfolio,
                    query=str(payload.get("query_understood", "")),
                    turn_number=int(payload.get("turn_number", 0) or 0),
                    opinion_id=str(payload.get("opinion_id", "")),
                    depth=str(payload.get("depth_used", "medium") or "medium"),
                )

                self.assertEqual(response.agent_id, agent_id)
                self.assertEqual(response.turn_number, payload["turn_number"])
                self.assertEqual(response.depth_used, payload["depth_used"])
                self.assertTrue(response.evidence)


if __name__ == "__main__":
    unittest.main()
