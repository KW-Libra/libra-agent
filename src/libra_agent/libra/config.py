from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


BackendName = Literal["ollama", "llama_cpp"]

DEFAULT_OLLAMA_MODEL = "dolphin-llama3:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
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


LibraBackendConfig = OllamaBackendConfig | LlamaCppBackendConfig


def add_backend_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_backend: BackendName = "ollama",
    backend_help: str = "LLM backend/provider",
) -> None:
    parser.add_argument(
        "--backend",
        default=default_backend,
        choices=("ollama", "llama_cpp"),
        help=backend_help,
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
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
    parser.add_argument("--llama-port", type=int, default=DEFAULT_LLAMA_PORT, help="llama-server bind port")
    parser.add_argument("--llama-ctx", type=int, default=DEFAULT_LLAMA_CTX, help="llama.cpp context size")
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
