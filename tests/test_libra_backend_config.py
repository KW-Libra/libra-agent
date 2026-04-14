from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from libra_agent.libra.config import (
    LlamaCppBackendConfig,
    OllamaBackendConfig,
    backend_config_from_args,
)
from libra_agent.libra.llm_clients import create_chat_client
from libra_agent.llama_cpp_client import LlamaCppServerClient
from libra_agent.ollama_client import OllamaChatClient


class LibraBackendConfigTests(unittest.TestCase):
    def test_backend_config_from_args_builds_ollama_config(self) -> None:
        args = argparse.Namespace(
            backend="ollama",
            model="test-model",
            ollama_host="http://127.0.0.1:11434",
        )

        config = backend_config_from_args(args)

        self.assertIsInstance(config, OllamaBackendConfig)
        self.assertEqual(config.backend, "ollama")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.host, "http://127.0.0.1:11434")

    def test_backend_config_from_args_builds_llama_cpp_config(self) -> None:
        args = argparse.Namespace(
            backend="llama_cpp",
            llama_server_path="tools/llama.cpp/bin/llama-server.exe",
            llama_model_path="models/supergemma/model.gguf",
            llama_mmproj_path="",
            llama_alias="supergemma4-26b",
            llama_host="127.0.0.1",
            llama_port=8091,
            llama_ctx=4096,
            llama_gpu_layers="99",
            llama_no_launch=True,
        )

        config = backend_config_from_args(args)

        self.assertIsInstance(config, LlamaCppBackendConfig)
        self.assertEqual(config.backend, "llama_cpp")
        self.assertEqual(config.server_path, Path("tools/llama.cpp/bin/llama-server.exe"))
        self.assertEqual(config.model_path, Path("models/supergemma/model.gguf"))
        self.assertIsNone(config.mmproj_path)
        self.assertEqual(config.model, "supergemma4-26b")
        self.assertEqual(config.ctx_size, 4096)
        self.assertEqual(config.gpu_layers, "99")
        self.assertFalse(config.launch_server)

    def test_create_chat_client_returns_ollama_client(self) -> None:
        client = create_chat_client(
            OllamaBackendConfig(
                model="test-model",
                host="http://127.0.0.1:11434",
            )
        )

        self.assertIsInstance(client, OllamaChatClient)
        self.assertEqual(client.model, "test-model")

    def test_create_chat_client_returns_llama_cpp_client(self) -> None:
        client = create_chat_client(
            LlamaCppBackendConfig(
                server_path=Path("tools/llama.cpp/bin/llama-server.exe"),
                model_path=Path("models/supergemma/model.gguf"),
                mmproj_path=Path("models/supergemma/mmproj.gguf"),
                model_alias="supergemma4-26b",
                launch_server=False,
            )
        )

        self.assertIsInstance(client, LlamaCppServerClient)
        self.assertEqual(client.model, "supergemma4-26b")
        self.assertFalse(client.launch_server)


if __name__ == "__main__":
    unittest.main()
