from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import httpx

from libra_agent.libra.portfolio_sources.kis import (
    KISDomesticPortfolioClient,
    KISPortfolioBootstrapError,
    _config_from_args,
)


class LibraKISPortfolioTests(unittest.TestCase):
    def test_kis_config_loader_reads_flat_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "kis_devlp.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        'my_app: "real-app"',
                        'my_sec: "real-secret"',
                        'paper_app: "paper-app"',
                        'paper_sec: "paper-secret"',
                        'my_acct_stock: "12345678"',
                        'my_paper_stock: "87654321"',
                        'my_prod: "01" # 종합계좌',
                        'my_agent: "UnitTestAgent/1.0"',
                        'prod: "https://real.example"',
                        'vps: "https://demo.example"',
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                kis_env="real",
                kis_config=str(config_path),
                kis_app_key=None,
                kis_app_secret=None,
                kis_account_no=None,
                kis_product_code=None,
                kis_user_agent=None,
            )

            config = _config_from_args(args)

            self.assertEqual(config.app_key, "real-app")
            self.assertEqual(config.app_secret, "real-secret")
            self.assertEqual(config.account_no, "12345678")
            self.assertEqual(config.product_code, "01")
            self.assertEqual(config.base_url, "https://real.example")
            self.assertEqual(config.user_agent, "UnitTestAgent/1.0")

    def test_kis_client_builds_portfolio_snapshot(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/oauth2/tokenP":
                return httpx.Response(
                    200,
                    json={
                        "access_token": "test-token",
                    },
                )
            if request.url.path == "/uapi/domestic-stock/v1/trading/inquire-balance":
                self.assertEqual(request.headers["tr_id"], "TTTC8434R")
                self.assertEqual(request.headers["authorization"], "Bearer test-token")
                return httpx.Response(
                    200,
                    headers={"tr_cont": ""},
                    json={
                        "rt_cd": "0",
                        "output1": [
                            {
                                "pdno": "005930",
                                "prdt_name": "삼성전자",
                                "hldg_qty": "12",
                                "evlu_amt": "720000",
                                "prpr": "60000",
                            },
                            {
                                "pdno": "000660",
                                "prdt_name": "SK하이닉스",
                                "hldg_qty": "3",
                                "evlu_amt": "540000",
                                "prpr": "180000",
                            },
                            {
                                "pdno": "035420",
                                "prdt_name": "NAVER",
                                "hldg_qty": "0",
                                "evlu_amt": "0",
                                "prpr": "0",
                            },
                        ],
                        "output2": [
                            {
                                "dnca_tot_amt": "240000",
                                "tot_evlu_amt": "1500000",
                                "evlu_amt_smtl_amt": "1260000",
                            }
                        ],
                        "ctx_area_fk100": "",
                        "ctx_area_nk100": "",
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport, base_url="https://test.example") as http_client:
            args = argparse.Namespace(
                kis_env="real",
                kis_config="",
                kis_app_key="app-key",
                kis_app_secret="app-secret",
                kis_account_no="12345678",
                kis_product_code="01",
                kis_user_agent="UnitTestAgent/1.0",
            )
            config = _config_from_args(args)
            client = KISDomesticPortfolioClient(config)

            snapshot = client.fetch_snapshot(http_client=http_client)

        self.assertEqual(len(snapshot.holdings), 2)
        self.assertEqual(snapshot.holdings[0].ticker, "005930")
        self.assertAlmostEqual(snapshot.holdings[0].weight, 0.48, places=2)
        self.assertAlmostEqual(snapshot.holdings[1].weight, 0.36, places=2)
        self.assertAlmostEqual(snapshot.cash_weight, 0.16, places=2)
        self.assertEqual(snapshot.total_value_krw, 1500000.0)
        self.assertIn("005930.KS", snapshot.holdings[0].aliases)

    def test_kis_config_validation_rejects_missing_required_fields(self) -> None:
        args = argparse.Namespace(
            kis_env="real",
            kis_config="",
            kis_app_key="",
            kis_app_secret="",
            kis_account_no="",
            kis_product_code="",
            kis_user_agent="",
        )

        with self.assertRaises(KISPortfolioBootstrapError):
            _config_from_args(args)


if __name__ == "__main__":
    unittest.main()
