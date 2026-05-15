from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BackendName = Literal["ollama", "llama_cpp", "anthropic", "gemini"]

DEFAULT_OLLAMA_MODEL = "dolphin-llama3:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_MAX_TOKENS = 4096
DEFAULT_LLAMA_SERVER_PATH = Path("tools") / "llama.cpp" / "b8783" / "bin" / "llama-server.exe"
DEFAULT_LLAMA_MODEL_PATH = (
    Path("models") / "supergemma4-26b" / "supergemma4-26b-abliterated-multimodal-Q4_K_M.gguf"
)
DEFAULT_LLAMA_MMPROJ_PATH = (
    Path("models") / "supergemma4-26b" / "mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf"
)
DEFAULT_LLAMA_HOST = "127.0.0.1"
DEFAULT_LLAMA_PORT = 8091
DEFAULT_LLAMA_CTX = 2048
DEFAULT_LLAMA_GPU_LAYERS = "auto"


@dataclass(slots=True, frozen=True)
class OllamaBackendConfig:
    backend: Literal["ollama"] = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    timeout_seconds: float = 180.0


@dataclass(slots=True, frozen=True)
class LlamaCppBackendConfig:
    backend: Literal["llama_cpp"] = "llama_cpp"
    server_path: Path = DEFAULT_LLAMA_SERVER_PATH
    model_path: Path = DEFAULT_LLAMA_MODEL_PATH
    mmproj_path: Path | None = DEFAULT_LLAMA_MMPROJ_PATH
    model_alias: str | None = None
    host: str = DEFAULT_LLAMA_HOST
    port: int = DEFAULT_LLAMA_PORT
    ctx_size: int = DEFAULT_LLAMA_CTX
    gpu_layers: str = DEFAULT_LLAMA_GPU_LAYERS
    timeout_seconds: float = 300.0
    startup_timeout_seconds: float = 600.0
    launch_server: bool = True
    extra_args: tuple[str, ...] = ()

    @property
    def model(self) -> str:
        return self.model_alias or self.model_path.stem


@dataclass(slots=True, frozen=True)
class AnthropicBackendConfig:
    backend: Literal["anthropic"] = "anthropic"
    api_key: str = ""
    model: str = DEFAULT_ANTHROPIC_MODEL
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS
    timeout_seconds: float = 180.0


@dataclass(slots=True, frozen=True)
class GeminiBackendConfig:
    backend: Literal["gemini"] = "gemini"
    api_key: str = ""
    model: str = DEFAULT_GEMINI_MODEL
    base_url: str = DEFAULT_GEMINI_BASE_URL
    max_tokens: int = DEFAULT_GEMINI_MAX_TOKENS
    timeout_seconds: float = 180.0


LibraBackendConfig = (
    OllamaBackendConfig | LlamaCppBackendConfig | AnthropicBackendConfig | GeminiBackendConfig
)


def add_backend_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_backend: BackendName = "ollama",
    backend_help: str = "LLM backend/provider",
) -> None:
    parser.add_argument(
        "--backend",
        default=default_backend,
        choices=("ollama", "llama_cpp", "anthropic", "gemini"),
        help=backend_help,
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
    parser.add_argument(
        "--anthropic-api-key",
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--anthropic-model",
        default=None,
        help="Claude model name. Defaults to ANTHROPIC_MODEL or LIBRA_ANTHROPIC_MODEL.",
    )
    parser.add_argument(
        "--anthropic-base-url",
        default=DEFAULT_ANTHROPIC_BASE_URL,
        help="Anthropic API base URL",
    )
    parser.add_argument(
        "--anthropic-version",
        default=DEFAULT_ANTHROPIC_VERSION,
        help="Anthropic API version header",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
        help="Maximum output tokens for Claude responses",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key. Defaults to GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name. Defaults to GEMINI_MODEL or LIBRA_GEMINI_MODEL.",
    )
    parser.add_argument(
        "--gemini-base-url",
        default=DEFAULT_GEMINI_BASE_URL,
        help="Gemini API base URL",
    )
    parser.add_argument(
        "--gemini-max-tokens",
        type=int,
        default=DEFAULT_GEMINI_MAX_TOKENS,
        help="Maximum output tokens for Gemini responses",
    )
    parser.add_argument(
        "--llama-server-path",
        default=str(DEFAULT_LLAMA_SERVER_PATH),
        help="Path to llama-server.exe",
    )
    parser.add_argument(
        "--llama-model-path",
        default=str(DEFAULT_LLAMA_MODEL_PATH),
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--llama-mmproj-path",
        default=str(DEFAULT_LLAMA_MMPROJ_PATH),
        help="Optional path to the multimodal projector GGUF",
    )
    parser.add_argument("--llama-alias", help="Model alias exposed by llama-server")
    parser.add_argument("--llama-host", default=DEFAULT_LLAMA_HOST, help="llama-server bind host")
    parser.add_argument(
        "--llama-port", type=int, default=DEFAULT_LLAMA_PORT, help="llama-server bind port"
    )
    parser.add_argument(
        "--llama-ctx", type=int, default=DEFAULT_LLAMA_CTX, help="llama.cpp context size"
    )
    parser.add_argument(
        "--llama-gpu-layers",
        default=DEFAULT_LLAMA_GPU_LAYERS,
        help="llama.cpp GPU layers setting",
    )
    parser.add_argument(
        "--llama-no-launch",
        action="store_true",
        help="Do not launch llama-server automatically; connect to an existing server instead",
    )


def backend_config_from_args(args: Any) -> LibraBackendConfig:
    backend = str(getattr(args, "backend", "ollama")).strip().lower()
    if backend == "ollama":
        return OllamaBackendConfig(
            model=str(getattr(args, "model", DEFAULT_OLLAMA_MODEL)),
            host=str(getattr(args, "ollama_host", DEFAULT_OLLAMA_HOST)),
        )
    if backend == "anthropic":
        api_key = str(getattr(args, "anthropic_api_key", "") or os.getenv("ANTHROPIC_API_KEY", ""))
        model = (
            getattr(args, "anthropic_model", None)
            or os.getenv("LIBRA_ANTHROPIC_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or DEFAULT_ANTHROPIC_MODEL
        )
        return AnthropicBackendConfig(
            api_key=api_key,
            model=str(model),
            base_url=str(getattr(args, "anthropic_base_url", DEFAULT_ANTHROPIC_BASE_URL)),
            anthropic_version=str(getattr(args, "anthropic_version", DEFAULT_ANTHROPIC_VERSION)),
            max_tokens=int(getattr(args, "anthropic_max_tokens", DEFAULT_ANTHROPIC_MAX_TOKENS)),
        )
    if backend == "gemini":
        model = (
            getattr(args, "gemini_model", None)
            or os.getenv("LIBRA_GEMINI_MODEL")
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_GEMINI_MODEL
        )
        return GeminiBackendConfig(
            api_key=str(getattr(args, "gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")),
            model=str(model),
            base_url=str(getattr(args, "gemini_base_url", DEFAULT_GEMINI_BASE_URL)),
            max_tokens=int(getattr(args, "gemini_max_tokens", DEFAULT_GEMINI_MAX_TOKENS)),
        )
    if backend == "llama_cpp":
        raw_mmproj = getattr(args, "llama_mmproj_path", None)
        mmproj = str(raw_mmproj).strip() if raw_mmproj is not None else ""
        model_alias = getattr(args, "llama_alias", None)
        return LlamaCppBackendConfig(
            server_path=Path(getattr(args, "llama_server_path", DEFAULT_LLAMA_SERVER_PATH)),
            model_path=Path(getattr(args, "llama_model_path", DEFAULT_LLAMA_MODEL_PATH)),
            mmproj_path=Path(mmproj) if mmproj else None,
            model_alias=str(model_alias).strip() if model_alias else None,
            host=str(getattr(args, "llama_host", DEFAULT_LLAMA_HOST)),
            port=int(getattr(args, "llama_port", DEFAULT_LLAMA_PORT)),
            ctx_size=int(getattr(args, "llama_ctx", DEFAULT_LLAMA_CTX)),
            gpu_layers=str(getattr(args, "llama_gpu_layers", DEFAULT_LLAMA_GPU_LAYERS)),
            launch_server=not bool(getattr(args, "llama_no_launch", False)),
        )
    raise ValueError(f"Unsupported LIBRA backend: {backend}")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def backend_config_from_env(*, default_backend: BackendName = "llama_cpp") -> LibraBackendConfig:
    backend = (
        (
            os.getenv("LIBRA_LLM_PROVIDER")
            or os.getenv("LIBRA_BACKEND")
            or os.getenv("LLM_PROVIDER")
            or default_backend
        )
        .strip()
        .lower()
    )
    if backend == "ollama":
        return OllamaBackendConfig(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            timeout_seconds=_env_float("LIBRA_LLM_TIMEOUT_SECONDS", 180.0),
        )
    if backend == "anthropic":
        return AnthropicBackendConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=(
                os.getenv("LIBRA_ANTHROPIC_MODEL")
                or os.getenv("ANTHROPIC_MODEL")
                or DEFAULT_ANTHROPIC_MODEL
            ),
            base_url=os.getenv("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
            anthropic_version=os.getenv("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
            max_tokens=_env_int("ANTHROPIC_MAX_TOKENS", DEFAULT_ANTHROPIC_MAX_TOKENS),
            timeout_seconds=_env_float("LIBRA_LLM_TIMEOUT_SECONDS", 180.0),
        )
    if backend == "gemini":
        return GeminiBackendConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=(
                os.getenv("LIBRA_GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            ),
            base_url=os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
            max_tokens=_env_int("GEMINI_MAX_TOKENS", DEFAULT_GEMINI_MAX_TOKENS),
            timeout_seconds=_env_float("LIBRA_LLM_TIMEOUT_SECONDS", 180.0),
        )
    if backend == "llama_cpp":
        return LlamaCppBackendConfig(
            server_path=Path(os.getenv("LIBRA_LLAMA_SERVER_PATH", str(DEFAULT_LLAMA_SERVER_PATH))),
            model_path=Path(os.getenv("LIBRA_LLAMA_MODEL_PATH", str(DEFAULT_LLAMA_MODEL_PATH))),
            mmproj_path=Path(os.getenv("LIBRA_LLAMA_MMPROJ_PATH", str(DEFAULT_LLAMA_MMPROJ_PATH)))
            if os.getenv("LIBRA_LLAMA_MMPROJ_PATH", str(DEFAULT_LLAMA_MMPROJ_PATH)).strip()
            else None,
            model_alias=os.getenv("LIBRA_LLAMA_ALIAS") or "supergemma4-26b",
            host=os.getenv("LIBRA_LLAMA_HOST", DEFAULT_LLAMA_HOST),
            port=_env_int("LIBRA_LLAMA_PORT", DEFAULT_LLAMA_PORT),
            ctx_size=_env_int("LIBRA_LLAMA_CTX", DEFAULT_LLAMA_CTX),
            gpu_layers=os.getenv("LIBRA_LLAMA_GPU_LAYERS", DEFAULT_LLAMA_GPU_LAYERS),
            timeout_seconds=_env_float("LIBRA_LLM_TIMEOUT_SECONDS", 300.0),
            launch_server=_env_bool("LIBRA_LLAMA_LAUNCH_SERVER", False),
        )
    raise ValueError(f"Unsupported LIBRA backend: {backend}")
