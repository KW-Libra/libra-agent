from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from libra_agent.libra_api import app


class LibraAgentApiTests(unittest.TestCase):
    def test_health(self) -> None:
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_judge_run_requires_knowledge_input(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/v1/judge-runs",
            json={
                "query": "포트폴리오 점검",
                "portfolio": {
                    "generated_at": "2026-04-25T00:00:00+09:00",
                    "holdings": [
                        {
                            "ticker": "005930",
                            "company_name": "삼성전자",
                            "weight": 0.5,
                        }
                    ],
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("knowledge_base", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
