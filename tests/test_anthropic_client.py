from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

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

    def test_chat_json_appends_usage_log_record(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "model": "claude-response-model",
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                    },
                    "content": [{"type": "text", "text": '{"decision":"BUY"}'}],
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log = os.path.join(temp_dir, "usage.jsonl")
            client = AnthropicChatClient(
                api_key="test-key",
                model="claude-test",
                transport=httpx.MockTransport(handler),
            )

            with patch.dict(os.environ, {"LIBRA_LLM_USAGE_LOG": usage_log}):
                payload = client.chat_json(system_prompt="system", user_prompt="user")

            self.assertEqual(payload, {"decision": "BUY"})
            with open(usage_log, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 1)
            record = records[0]
            created_at = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
            self.assertEqual(created_at.tzinfo, UTC)
            self.assertEqual(record["provider"], "anthropic")
            self.assertEqual(record["model"], "claude-response-model")
            self.assertEqual(record["id"], "msg_test")
            self.assertEqual(record["stop_reason"], "end_turn")
            self.assertEqual(record["usage"]["input_tokens"], 11)
            self.assertEqual(record["usage"]["output_tokens"], 7)

    def test_chat_json_retries_malformed_json_response(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(
                    200,
                    json={
                        "content": [
                            {
                                "type": "text",
                                "text": '{"decision":"HOLD"',
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": '{"decision":"DEFER"}'}]},
            )

        client = AnthropicChatClient(
            api_key="test-key",
            model="claude-test",
            transport=httpx.MockTransport(handler),
        )

        with patch.dict(os.environ, {"LIBRA_ANTHROPIC_CHAT_JSON_ATTEMPTS": "2"}):
            payload = client.chat_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload, {"decision": "DEFER"})
        self.assertEqual(len(requests), 2)

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

    def test_chat_json_tool_appends_usage_log_record(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_tool_test",
                    "model": "claude-test",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 21, "output_tokens": 9},
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "submit_result",
                            "id": "toolu_test",
                            "input": {"decision": "SELL"},
                        }
                    ],
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log = os.path.join(temp_dir, "usage.jsonl")
            client = AnthropicChatClient(
                api_key="test-key",
                model="claude-test",
                transport=httpx.MockTransport(handler),
            )

            with patch.dict(os.environ, {"LIBRA_LLM_USAGE_LOG": usage_log}):
                payload = client.chat_json_tool(
                    system_prompt="system",
                    user_prompt="user",
                    tool_name="submit_result",
                    tool_description="submit strict result",
                    input_schema={"type": "object"},
                )

            self.assertEqual(payload, {"decision": "SELL"})
            with open(usage_log, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 1)
            record = records[0]
            datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
            self.assertEqual(record["provider"], "anthropic")
            self.assertEqual(record["model"], "claude-test")
            self.assertEqual(record["id"], "msg_tool_test")
            self.assertEqual(record["stop_reason"], "tool_use")
            self.assertEqual(record["usage"], {"input_tokens": 21, "output_tokens": 9})

    def test_usage_logging_failure_does_not_raise(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1},
                    "content": [{"type": "text", "text": '{"decision":"HOLD"}'}],
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = AnthropicChatClient(
                api_key="test-key",
                model="claude-test",
                transport=httpx.MockTransport(handler),
            )

            with patch.dict(os.environ, {"LIBRA_LLM_USAGE_LOG": temp_dir}):
                payload = client.chat_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload, {"decision": "HOLD"})

    def test_usage_logging_creates_parent_directory(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_nested",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "content": [{"type": "text", "text": '{"decision":"HOLD"}'}],
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log = os.path.join(temp_dir, "nested", "usage.jsonl")
            client = AnthropicChatClient(
                api_key="test-key",
                model="claude-test",
                transport=httpx.MockTransport(handler),
            )

            with patch.dict(os.environ, {"LIBRA_LLM_USAGE_LOG": usage_log}):
                payload = client.chat_json(system_prompt="system", user_prompt="user")

            self.assertEqual(payload, {"decision": "HOLD"})
            self.assertTrue(os.path.exists(usage_log))

    def test_missing_api_key_fails_before_http_call(self) -> None:
        client = AnthropicChatClient(api_key="", model="claude-test")

        with self.assertRaises(AnthropicClientError):
            client.ensure_available()

    def test_request_timeout_is_capped_for_external_llm_calls(self) -> None:
        self.addCleanup(os.environ.pop, "LIBRA_LLM_REQUEST_TIMEOUT_SECONDS", None)
        os.environ.pop("LIBRA_LLM_REQUEST_TIMEOUT_SECONDS", None)
        client = AnthropicChatClient(
            api_key="test-key",
            model="claude-test",
            timeout_seconds=180,
        )

        self.assertEqual(client._effective_timeout(), 45.0)

        os.environ["LIBRA_LLM_REQUEST_TIMEOUT_SECONDS"] = "12.5"
        self.assertEqual(client._effective_timeout(), 12.5)
        self.assertEqual(client._effective_timeout(5), 5.0)


if __name__ == "__main__":
    unittest.main()
