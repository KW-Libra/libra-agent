from __future__ import annotations

import json
import os
import unittest

import httpx

from libra_agent.gemini_client import GeminiChatClient, GeminiClientError


class GeminiChatClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {
            "LIBRA_GEMINI_FREE_TIER": os.environ.get("LIBRA_GEMINI_FREE_TIER"),
            "LIBRA_GEMINI_RETRY_ATTEMPTS": os.environ.get("LIBRA_GEMINI_RETRY_ATTEMPTS"),
            "LIBRA_GEMINI_RETRY_BASE_DELAY_SECONDS": os.environ.get("LIBRA_GEMINI_RETRY_BASE_DELAY_SECONDS"),
        }
        os.environ["LIBRA_GEMINI_FREE_TIER"] = "false"
        os.environ["LIBRA_GEMINI_RETRY_BASE_DELAY_SECONDS"] = "0"

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_chat_json_posts_generate_content_request_and_decodes_json(self) -> None:
        os.environ["LIBRA_GEMINI_RETRY_ATTEMPTS"] = "0"
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.url.path, "/v1beta/models/gemini-test:generateContent")
            self.assertEqual(request.url.params.get("key"), "test-key")
            body = json.loads(request.content.decode("utf-8"))
            self.assertEqual(body["system_instruction"]["parts"][0]["text"], "system")
            self.assertEqual(body["contents"][0]["parts"][0]["text"], "user")
            self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "결과입니다. {\"decision\":\"HOLD\",\"confidence\":0.72}"}
                                ]
                            }
                        }
                    ]
                },
            )

        client = GeminiChatClient(
            api_key="test-key",
            model="gemini-test",
            base_url="https://generativelanguage.test",
            transport=httpx.MockTransport(handler),
        )

        payload = client.chat_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload["decision"], "HOLD")
        self.assertEqual(payload["confidence"], 0.72)
        self.assertEqual(len(requests), 1)

    def test_ensure_available_checks_model_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1beta/models/gemini-test")
            self.assertEqual(request.url.params.get("key"), "test-key")
            return httpx.Response(200, json={"name": "models/gemini-test"})

        client = GeminiChatClient(
            api_key="test-key",
            model="gemini-test",
            base_url="https://generativelanguage.test",
            transport=httpx.MockTransport(handler),
        )

        client.ensure_available()

    def test_missing_api_key_fails_before_http_call(self) -> None:
        client = GeminiChatClient(api_key="", model="gemini-test")

        with self.assertRaises(GeminiClientError):
            client.chat_json(system_prompt="system", user_prompt="user")

    def test_retries_transient_http_status(self) -> None:
        os.environ["LIBRA_GEMINI_RETRY_ATTEMPTS"] = "1"
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503, json={"error": "temporarily unavailable"})
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]},
            )

        client = GeminiChatClient(
            api_key="secret-key",
            model="gemini-test",
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(
            client.chat_json(system_prompt="JSON only", user_prompt="{}", temperature=0),
            {"ok": True},
        )
        self.assertEqual(calls, 2)

    def test_redacts_api_key_in_http_errors(self) -> None:
        os.environ["LIBRA_GEMINI_RETRY_ATTEMPTS"] = "0"
        client = GeminiChatClient(
            api_key="secret-key",
            model="gemini-test",
            transport=httpx.MockTransport(lambda _: httpx.Response(403, json={"error": "forbidden"})),
        )

        with self.assertRaises(GeminiClientError) as raised:
            client.chat_json(system_prompt="JSON only", user_prompt="{}", temperature=0)

        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("<redacted>", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
