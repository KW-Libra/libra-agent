from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .errors import ChatClientError


class LlamaCppServerClient:
    def __init__(
        self,
        *,
        server_path: str | Path,
        model_path: str | Path,
        model: str | None = None,
        mmproj_path: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8091,
        ctx_size: int = 2048,
        gpu_layers: str = "auto",
        timeout_seconds: float = 300.0,
        startup_timeout_seconds: float = 600.0,
        launch_server: bool = True,
        extra_args: Sequence[str] | None = None,
    ) -> None:
        self.server_path = Path(server_path).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.mmproj_path = Path(mmproj_path).expanduser().resolve() if mmproj_path else None
        self.model = model or self.model_path.stem
        self.host = host
        self.port = int(port)
        self.ctx_size = int(ctx_size)
        self.gpu_layers = str(gpu_layers)
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.launch_server = launch_server
        self.extra_args = tuple(extra_args or ())

        self._process: subprocess.Popen[str] | None = None
        self._stdout_lines: deque[str] = deque(maxlen=80)
        self._stderr_lines: deque[str] = deque(maxlen=160)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> LlamaCppServerClient:
        self.start_server()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_server()

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        self.ensure_available()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ChatClientError(f"Failed to call llama.cpp chat API at {self.base_url}: {exc}") from exc

        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ChatClientError("llama.cpp chat response did not contain choices.")
        message = choices[0].get("message", {}) if isinstance(choices[0], Mapping) else {}
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise ChatClientError("llama.cpp returned an empty response body.")
        return self._decode_json(content)

    def ensure_available(self) -> None:
        if self._is_healthy(timeout=min(self.timeout_seconds, 10.0)):
            return
        if not self.launch_server:
            raise ChatClientError(f"llama.cpp server is not reachable at {self.base_url}.")
        self.start_server()

    def start_server(self) -> None:
        if self._is_healthy(timeout=2.0):
            return
        if self._process is not None and self._process.poll() is None:
            self._wait_until_ready()
            return

        self._validate_paths()
        command = [
            str(self.server_path),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "-ngl",
            self.gpu_layers,
            "--alias",
            self.model,
            "--reasoning",
            "off",
            "--no-webui",
            "--jinja",
        ]
        if self.mmproj_path is not None:
            command.extend(["-mm", str(self.mmproj_path)])
        command.extend(self.extra_args)

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._stdout_lines.clear()
        self._stderr_lines.clear()
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.server_path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except OSError as exc:
            raise ChatClientError(f"Failed to launch llama.cpp server: {exc}") from exc

        self._stdout_thread = threading.Thread(
            target=self._drain_stream,
            args=(self._process.stdout, self._stdout_lines),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stream,
            args=(self._process.stderr, self._stderr_lines),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._wait_until_ready()

    def stop_server(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._process = None

    def _validate_paths(self) -> None:
        if not self.server_path.exists():
            raise ChatClientError(f"llama.cpp server executable does not exist: {self.server_path}")
        if not self.model_path.exists():
            raise ChatClientError(f"GGUF model file does not exist: {self.model_path}")
        if self.mmproj_path is not None and not self.mmproj_path.exists():
            raise ChatClientError(f"llama.cpp mmproj file does not exist: {self.mmproj_path}")

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._is_healthy(timeout=5.0):
                return
            if self._process is not None and self._process.poll() is not None:
                raise ChatClientError(
                    "llama.cpp server exited during startup with "
                    f"code {self._process.returncode}.\n{self._log_tail()}"
                )
            time.sleep(2)
        raise ChatClientError(
            f"Timed out waiting for llama.cpp server at {self.base_url}.\n{self._log_tail()}"
        )

    def _is_healthy(self, *, timeout: float) -> bool:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
        except httpx.HTTPError:
            return False
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return False
        return isinstance(payload, Mapping) and str(payload.get("status", "")).lower() == "ok"

    def _decode_json(self, raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = self._extract_first_object(raw_text)
        if not isinstance(payload, Mapping):
            raise ChatClientError("llama.cpp JSON response was not an object.")
        return dict(payload)

    def _extract_first_object(self, raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        if start < 0:
            raise ChatClientError("llama.cpp response did not contain JSON.")
        depth = 0
        for index in range(start, len(raw_text)):
            char = raw_text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw_text[start : index + 1]
                    try:
                        payload = json.loads(chunk)
                    except json.JSONDecodeError as exc:
                        raise ChatClientError(f"Failed to parse llama.cpp JSON chunk: {exc}") from exc
                    if not isinstance(payload, Mapping):
                        raise ChatClientError("llama.cpp JSON chunk was not an object.")
                    return dict(payload)
        raise ChatClientError("llama.cpp response contained an unterminated JSON object.")

    def _drain_stream(self, stream, buffer: deque[str]) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                buffer.append(line.rstrip())
        finally:
            stream.close()

    def _log_tail(self) -> str:
        stdout_tail = "\n".join(self._stdout_lines).strip()
        stderr_tail = "\n".join(self._stderr_lines).strip()
        parts: list[str] = []
        if stdout_tail:
            parts.append(f"[stdout]\n{stdout_tail}")
        if stderr_tail:
            parts.append(f"[stderr]\n{stderr_tail}")
        if not parts:
            return "No llama.cpp server logs were captured."
        return "\n\n".join(parts)
