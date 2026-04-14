from __future__ import annotations

from contextlib import ExitStack
from typing import Any

from libra_agent.llama_cpp_client import LlamaCppServerClient
from libra_agent.ollama_client import OllamaChatClient

from ..config import (
    LlamaCppBackendConfig,
    LibraBackendConfig,
    OllamaBackendConfig,
    backend_config_from_args,
)
from .base import ChatClientProtocol


def create_chat_client(config: LibraBackendConfig) -> Any:
    if isinstance(config, OllamaBackendConfig):
        return OllamaChatClient(
            model=config.model,
            host=config.host,
            timeout_seconds=config.timeout_seconds,
        )
    if isinstance(config, LlamaCppBackendConfig):
        return LlamaCppServerClient(
            server_path=config.server_path,
            model_path=config.model_path,
            mmproj_path=config.mmproj_path,
            model=config.model,
            host=config.host,
            port=config.port,
            ctx_size=config.ctx_size,
            gpu_layers=config.gpu_layers,
            timeout_seconds=config.timeout_seconds,
            startup_timeout_seconds=config.startup_timeout_seconds,
            launch_server=config.launch_server,
            extra_args=config.extra_args,
        )
    raise TypeError(f"Unsupported LIBRA backend config: {type(config)!r}")


def open_chat_client(config: LibraBackendConfig, *, stack: ExitStack) -> ChatClientProtocol:
    client = create_chat_client(config)
    if hasattr(client, "__enter__"):
        return stack.enter_context(client)
    return client


def open_chat_client_from_args(args: Any, *, stack: ExitStack) -> ChatClientProtocol:
    return open_chat_client(backend_config_from_args(args), stack=stack)
