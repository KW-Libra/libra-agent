from __future__ import annotations

import json
import unittest

import httpx

from libra_agent.anthropic_client import AnthropicChatClient, AnthropicClientError


class AnthropicChatClientTests(unittest.TestCase):
    def test_chat_json_posts_messages_request_and_decodes_text_json(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.url.path, "/v1/messages")
            self.assertEqual(request.headers["x-api-key"], "test-key")
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            body = json.loads(request.content.decode("utf-8"))
            self.assertEqual(body["model"], "claude-test")
            self.assertEqual(body["system"], "system")
            self.assertEqual(body["messages"], [{"role": "user", "content": "user"}])
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "text",
                            "text": '결과입니다. {"decision":"HOLD","confidence":0.72}',
                        }
                    ]
                },
            )

        client = AnthropicChatClient(
            api_key="test-key",
            model="claude-test",
            transport=httpx.MockTransport(handler),
        )

        payload = client.chat_json(system_prompt="system", user_prompt="user", temperature=0.0)

        self.assertEqual(payload, {"decision": "HOLD", "confidence": 0.72})
        self.assertEqual(len(requests), 1)

    def test_ensure_available_checks_models_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/models")
            return httpx.Response(200, json={"data": []})

        client = AnthropicChatClient(
            api_key="test-key",
            model="claude-test",
            transport=httpx.MockTransport(handler),
        )

        client.ensure_available()

    def test_chat_json_tool_forces_tool_choice_and_returns_tool_input(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = json.loads(request.content.decode("utf-8"))
            self.assertEqual(body["tool_choice"], {"type": "tool", "name": "submit_result"})
            self.assertEqual(body["tools"][0]["name"], "submit_result")
            self.assertEqual(body["tools"][0]["input_schema"]["required"], ["decision"])
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "submit_result",
                            "id": "toolu_test",
                            "input": {"decision": "HOLD"},
                        }
                    ]
                },
            )

        client = AnthropicChatClient(
            api_key="test-key",
            model="claude-test",
            transport=httpx.MockTransport(handler),
        )

        payload = client.chat_json_tool(
            system_prompt="system",
            user_prompt="user",
            tool_name="submit_result",
            tool_description="submit strict result",
            input_schema={
                "type": "object",
                "properties": {"decision": {"type": "string"}},
                "required": ["decision"],
            },
            temperature=0.0,
        )

        self.assertEqual(payload, {"decision": "HOLD"})
        self.assertEqual(len(requests), 1)

    def test_missing_api_key_fails_before_http_call(self) -> None:
        client = AnthropicChatClient(api_key="", model="claude-test")

        with self.assertRaises(AnthropicClientError):
            client.ensure_available()


if __name__ == "__main__":
    unittest.main()
