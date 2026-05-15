"""
Ollama 로컬 LLM 클라이언트 — Phase 2 뉴스 파이프라인 2단계

Ollama REST API (http://ollama:11434) 를 통해 로컬 LLM을 호출합니다.
금융 텍스트 분석에는 llama3.2:3b 또는 mistral:7b-instruct를 권장합니다.

특징:
  - httpx 비동기 HTTP (이미 requirements.txt에 포함)
  - 타임아웃 30초 (로컬 LLM은 첫 토큰까지 시간이 걸릴 수 있음)
  - 스트리밍 비활성화 (stream=false — 전체 응답 반환)
  - Ollama 미사용 환경에서도 예외만 발생, 크래시 없음

환경변수:
  OLLAMA_BASE_URL  기본값: http://ollama:11434
  OLLAMA_MODEL     기본값: llama3.2:3b

사용:
    client = OllamaClient()
    text = await client.generate(prompt="Analyze: Fed raises rates by 50bp")
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Ollama 서비스에 연결할 수 없을 때 발생."""


class OllamaClient:
    """
    Ollama REST API 비동기 클라이언트.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout

    # ── 연결 확인 ─────────────────────────────────────────────

    async def is_available(self) -> bool:
        """Ollama 서버가 응답하는지 확인 (GET /api/tags)."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(self.base_url + "/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    # ── 텍스트 생성 ───────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
    ) -> str:
        """
        Ollama /api/generate 엔드포인트를 호출하여 텍스트 생성.

        Args:
            prompt:     사용자 입력 (뉴스 헤드라인 + 포트폴리오 컨텍스트)
            system:     시스템 프롬프트 (감성 분석 지시문)
            max_tokens: 최대 출력 토큰 수

        Returns:
            생성된 텍스트 (str)

        Raises:
            OllamaUnavailableError: Ollama 서버 연결 실패
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # 금융 분석은 낮은 온도 권장
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    self.base_url + "/api/generate",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                return data.get("response", "").strip()

        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                "Ollama 서버에 연결할 수 없습니다. "
                "docker compose --profile nlp up ollama 를 실행하세요."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaUnavailableError("Ollama 응답 타임아웃 (30s)") from e
        except httpx.HTTPStatusError as e:
            raise OllamaUnavailableError("Ollama HTTP 오류: " + str(e)) from e

    # ── 모델 확인 ─────────────────────────────────────────────

    async def ensure_model_pulled(self) -> bool:
        """
        설정된 모델이 로컬에 있는지 확인.
        없으면 pull 실패 안내 로그 후 False 반환 (자동 pull은 사용자 대역폭 고려 생략).
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self.base_url + "/api/tags")
                if r.status_code != 200:
                    return False
                models = [m["name"] for m in r.json().get("models", [])]
                if any(self.model in m for m in models):
                    return True
                logger.warning(
                    "[Ollama] 모델 '%s' 미설치. 컨테이너 내부에서 'ollama pull %s' 실행하세요.",
                    self.model,
                    self.model,
                )
                return False
        except Exception:
            return False


# ── 싱글턴 ────────────────────────────────────────────────────

_client: OllamaClient | None = None


def get_ollama() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
