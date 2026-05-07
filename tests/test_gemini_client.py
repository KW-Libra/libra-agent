from __future__ import annotations

import json
import unittest

import httpx

from libra_agent.gemini_client import GeminiChatClient, GeminiClientError


class GeminiChatClientTests(unittest.TestCase):
    def test_chat_json_posts_generate_content_request_and_decodes_json(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
