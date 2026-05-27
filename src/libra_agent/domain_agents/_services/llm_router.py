"""
LLM Router — Claude + Gemini 멀티 LLM 라우터

설계 원칙:
  - Claude Sonnet  : 고위험 추론 (Compliance, Risk 최종 판단)
  - Claude Haiku   : 빠른 정형화 응답 (Tax 간단 체크)
  - Gemini Flash   : 고처리량 작업 (뉴스 감성, ESG 스크리닝, 매크로 요약)
  - Gemini Pro     : Claude 크로스 검증 (할루시네이션 감소)

라우팅 정책 (우선순위: DB 사용자 설정 > LLM_ROUTING_POLICY env):
  balanced  → 에이전트별 최적 모델 자동 배정 (기본값)
  claude    → Claude 전용
  gemini    → Gemini 전용

API 키 소스 (우선순위):
  1. 사용자별 DB 키 (llm_credentials 테이블) — 동적으로 주입
  2. 환경 변수 ANTHROPIC_API_KEY / GEMINI_API_KEY — 시스템 키

비용 비교 (2024 기준, 1M token):
  Claude Haiku    $0.25 input / $1.25 output
  Claude Sonnet   $3.00 input / $15.00 output
  Gemini Flash    $0.075 input / $0.30 output  ← 최저가, 고속
  Gemini Pro      $3.50 input / $10.50 output
"""

from __future__ import annotations

import logging
import os
import threading
import time
from enum import StrEnum
from typing import Any

from libra_agent.runtime.debate_events import (
    publish_llm_error,
    publish_llm_prompt,
    publish_llm_response,
)

logger = logging.getLogger(__name__)

# ── LLM 모델 열거형 ────────────────────────────────────────────


class LLMModel(StrEnum):
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET = "claude-sonnet-4-6"
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-1.5-pro"
    # 로컬 백엔드 (유료 API 우회용). 실제 모델명은 env 로 주입.
    LOCAL_OLLAMA = "local-ollama"
    LOCAL_LLAMA_CPP = "local-llama-cpp"


# 유료로 간주되는 모델 집합. LIBRA_LLM_BUDGET=free 일 때 차단/대체 대상.
# Gemini Flash 는 무료 티어가 있으므로 budget guard 대상에서 제외.
_PAID_MODELS: frozenset[LLMModel] = frozenset(
    {
        LLMModel.CLAUDE_HAIKU,
        LLMModel.CLAUDE_SONNET,
        LLMModel.GEMINI_PRO,
    }
)


def _is_local(model: "LLMModel") -> bool:
    return model in (LLMModel.LOCAL_OLLAMA, LLMModel.LOCAL_LLAMA_CPP)


def _budget_mode() -> str:
    """LIBRA_LLM_BUDGET = free | paid (default paid). 'free' 시 유료 모델 차단."""
    return os.environ.get("LIBRA_LLM_BUDGET", "paid").strip().lower()


def _local_backend() -> str | None:
    """LIBRA_LOCAL_LLM_BACKEND = ollama | llama_cpp. 미설정이면 None."""
    raw = os.environ.get("LIBRA_LOCAL_LLM_BACKEND")
    if not raw:
        return None
    val = raw.strip().lower()
    return val if val in {"ollama", "llama_cpp"} else None


# ── 에이전트별 기본 LLM 배정 ────────────────────────────────────

# balanced 정책에서 각 에이전트가 사용할 기본 모델
AGENT_MODEL_MAP: dict[str, LLMModel] = {
    "risk": LLMModel.CLAUDE_SONNET,  # 고위험 판단 — Claude
    "tax": LLMModel.CLAUDE_HAIKU,  # 정형적 계산 — 저렴한 Haiku
    "compliance": LLMModel.CLAUDE_SONNET,  # 규정 준수 — Claude (신뢰성 우선)
    "macro": LLMModel.GEMINI_FLASH,  # 뉴스/매크로 요약 — Gemini Flash (고속)
    "sentiment": LLMModel.GEMINI_FLASH,  # 감성 분석 — Gemini Flash (대량 처리)
    "execution": LLMModel.CLAUDE_HAIKU,  # 체결 계산 — 정형적, Haiku
    "esg": LLMModel.GEMINI_FLASH,  # ESG 스크리닝 — Gemini Flash
}

# 크로스 검증이 필요한 에이전트 (할루시네이션 고위험)
CROSS_VALIDATE_AGENTS: set[str] = {"macro", "sentiment"}


# ── LLM 라우터 클래스 ────────────────────────────────────────────


class LLMRouter:
    """
    에이전트 ID에 따라 최적 LLM을 선택하고 호출합니다.

    API 키 우선순위:
      1. 생성자에 직접 주입 (anthropic_key / gemini_key) — 사용자 DB 키
      2. 환경 변수 ANTHROPIC_API_KEY / GEMINI_API_KEY — 시스템 키

    사용법:
        # 사용자별 키 사용 (권장)
        keys = await load_llm_keys(user_id)
        if keys:
            router = LLMRouter(
                anthropic_key=keys.anthropic_key,
                gemini_key=keys.gemini_key,
                policy=keys.llm_policy,
            )

        # 환경 변수 시스템 키
        router = LLMRouter()
        result = router.ask(agent_id="macro", system="...", user="...")
    """

    _gemini_throttle_lock = threading.Lock()
    _last_gemini_call_at = 0.0

    def __init__(
        self,
        anthropic_key: str | None = None,
        gemini_key: str | None = None,
        policy: str | None = None,
    ) -> None:
        # API 키: 인자 우선, 없으면 환경 변수
        self._anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        self._gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY")

        # 라우팅 정책: 인자 우선, 없으면 환경 변수
        self._policy = (policy or os.environ.get("LLM_ROUTING_POLICY", "balanced")).lower()

        # 키 출처 로깅 (민감 정보는 마스킹)
        claude_src = (
            "user-db"
            if anthropic_key
            else ("env" if os.environ.get("ANTHROPIC_API_KEY") else "none")
        )
        gemini_src = (
            "user-db" if gemini_key else ("env" if os.environ.get("GEMINI_API_KEY") else "none")
        )
        logger.info(
            f"[LLMRouter] 초기화 — policy={self._policy}, claude={claude_src}, gemini={gemini_src}"
        )

        self._claude = self._init_claude()
        self._gemini = self._init_gemini()

    # ── 초기화 ─────────────────────────────────────────────────

    def _init_claude(self) -> Any | None:
        if not self._anthropic_key:
            logger.warning("[LLMRouter] Anthropic API 키 없음 — Claude 비활성")
            return None
        try:
            import anthropic

            return anthropic.Anthropic(api_key=self._anthropic_key)
        except ImportError:
            logger.error("[LLMRouter] anthropic 패키지 미설치. pip install anthropic")
            return None

    def _init_gemini(self) -> Any | None:
        if not self._gemini_key:
            logger.warning("[LLMRouter] Gemini API 키 없음 — Gemini 비활성")
            return None
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_key)
            return genai
        except ImportError:
            logger.error(
                "[LLMRouter] google-generativeai 패키지 미설치. pip install google-generativeai"
            )
            return None

    # ── 모델 선택 ───────────────────────────────────────────────

    def _select_model(self, agent_id: str) -> LLMModel:
        # 명시적 로컬 정책 — 무조건 로컬.
        if self._policy == "local":
            return self._local_model_or_raise()

        if self._policy == "claude":
            preferred = LLMModel.CLAUDE_SONNET
        elif self._policy == "gemini":
            preferred = LLMModel.GEMINI_FLASH
        else:
            preferred = AGENT_MODEL_MAP.get(agent_id, LLMModel.CLAUDE_HAIKU)

        # Budget guard: free 모드에서 유료 모델이 선택되면 대체.
        if _budget_mode() == "free" and preferred in _PAID_MODELS:
            backend = _local_backend()
            if backend is not None:
                replacement = self._local_model_or_raise()
                logger.info(
                    "[LLMRouter] budget=free → %s 차단, %s 로 대체 (agent=%s)",
                    preferred.value,
                    replacement.value,
                    agent_id,
                )
                return replacement
            # 로컬 백엔드도 없으면 무료 Gemini Flash 로 강등 (throttle 적용).
            logger.warning(
                "[LLMRouter] budget=free + 로컬 백엔드 미설정 — %s 호출을 Gemini Flash 무료 티어로 대체"
                " (agent=%s). 안정성 필요시 LIBRA_LOCAL_LLM_BACKEND=ollama 설정 권장.",
                preferred.value,
                agent_id,
            )
            return LLMModel.GEMINI_FLASH

        return preferred

    def _local_model_or_raise(self) -> LLMModel:
        backend = _local_backend()
        if backend == "ollama":
            return LLMModel.LOCAL_OLLAMA
        if backend == "llama_cpp":
            return LLMModel.LOCAL_LLAMA_CPP
        raise RuntimeError(
            "로컬 LLM 백엔드가 설정되지 않았습니다. "
            "LIBRA_LOCAL_LLM_BACKEND=ollama 또는 llama_cpp 를 설정하세요."
        )

    def model_name_for(self, agent_id: str) -> str:
        return self._model_name(self._select_model(agent_id))

    # ── 메인 호출 API ───────────────────────────────────────────

    def ask(
        self,
        agent_id: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cross_validate: bool = False,
    ) -> str:
        """
        동기 LLM 호출. 에이전트 ID에 따라 최적 모델 자동 선택.

        Args:
            agent_id:       에이전트 식별자 (라우팅 기준)
            system:         시스템 프롬프트
            user:           사용자 입력
            max_tokens:     최대 출력 토큰
            cross_validate: True이면 Gemini + Claude 모두 호출 후 합의 (할루시네이션 방지)
        """
        model = self._select_model(agent_id)
        model_name = self._model_name(model)
        logger.debug(f"[LLMRouter] {agent_id} → {model_name}")

        publish_llm_prompt(
            actor=agent_id,
            phase="domain_router_primary",
            model=model_name,
            system_prompt=system,
            user_prompt=user,
            temperature=None,
        )
        try:
            primary = self._call_model(model, system, user, max_tokens)
        except Exception as exc:
            publish_llm_error(
                actor=agent_id,
                phase="domain_router_primary",
                model=model_name,
                error=exc,
            )
            raise
        publish_llm_response(
            actor=agent_id,
            phase="domain_router_primary",
            model=model_name,
            output=primary,
        )

        # 크로스 검증: 주요 에이전트 or 명시 요청시 Gemini↔Claude 교차 확인
        if (
            cross_validate
            and agent_id in CROSS_VALIDATE_AGENTS
            and self._policy == "balanced"
            and self._claude
            and self._gemini
        ):
            secondary_model = (
                LLMModel.CLAUDE_HAIKU
                if model in (LLMModel.GEMINI_FLASH, LLMModel.GEMINI_PRO)
                else LLMModel.GEMINI_FLASH
            )
            secondary_model_name = self._model_name(secondary_model)
            publish_llm_prompt(
                actor=agent_id,
                phase="domain_router_cross_validate",
                model=secondary_model_name,
                system_prompt=system,
                user_prompt=user,
                temperature=None,
            )
            try:
                secondary = self._call_model(secondary_model, system, user, max_tokens)
            except Exception as exc:
                publish_llm_error(
                    actor=agent_id,
                    phase="domain_router_cross_validate",
                    model=secondary_model_name,
                    error=exc,
                )
                raise
            publish_llm_response(
                actor=agent_id,
                phase="domain_router_cross_validate",
                model=secondary_model_name,
                output=secondary,
            )
            return self._merge_responses(
                primary,
                secondary,
                agent_id,
                primary_label=self._model_family(model),
                secondary_label=self._model_family(secondary_model),
            )

        return primary

    # ── 모델별 호출 ─────────────────────────────────────────────

    def _call_model(
        self,
        model: LLMModel,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        model_name = self._model_name(model)
        try:
            if _is_local(model):
                return self._call_local(model, system, user, max_tokens)
            if model in (LLMModel.CLAUDE_HAIKU, LLMModel.CLAUDE_SONNET):
                return self._call_claude(model_name, system, user, max_tokens)
            else:
                return self._call_gemini(model_name, system, user, max_tokens)
        except Exception as e:
            logger.error(f"[LLMRouter] {model_name} 호출 실패: {e}")
            raise RuntimeError(f"{model_name} LLM 호출 실패: {e}") from e

    def _model_name(self, model: LLMModel) -> str:
        if model == LLMModel.GEMINI_FLASH:
            return (
                os.environ.get("LIBRA_DOMAIN_GEMINI_MODEL")
                or os.environ.get("LIBRA_GEMINI_MODEL")
                or os.environ.get("GEMINI_MODEL")
                or model.value
            )
        if model == LLMModel.GEMINI_PRO:
            return os.environ.get("LIBRA_DOMAIN_GEMINI_PRO_MODEL") or model.value
        if model == LLMModel.LOCAL_OLLAMA:
            return os.environ.get("LIBRA_OLLAMA_MODEL", "qwen2.5:7b-instruct")
        if model == LLMModel.LOCAL_LLAMA_CPP:
            return os.environ.get("LIBRA_LLAMA_CPP_MODEL_ALIAS", "local-llama-cpp")
        return model.value

    # ── 로컬 백엔드 호출 (텍스트 모드, JSON 강제 없음) ──────────
    # 기존 ollama_client / llama_cpp_client 는 JSON 강제이므로
    # 자유형 텍스트 응답이 필요한 도메인 에이전트용으로 별도 경로 사용.

    def _call_local(self, model: LLMModel, system: str, user: str, max_tokens: int) -> str:
        if model == LLMModel.LOCAL_OLLAMA:
            return self._call_local_ollama(system, user, max_tokens)
        return self._call_local_llama_cpp(system, user, max_tokens)

    def _call_local_ollama(self, system: str, user: str, max_tokens: int) -> str:
        import httpx

        host = os.environ.get("LIBRA_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        model_name = os.environ.get("LIBRA_OLLAMA_MODEL", "qwen2.5:7b-instruct")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{host}/api/chat", json=payload)
            resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama 응답이 비어있습니다.")
        return content

    def _call_local_llama_cpp(self, system: str, user: str, max_tokens: int) -> str:
        import httpx

        host = os.environ.get("LIBRA_LLAMA_CPP_HOST", "127.0.0.1")
        port = int(os.environ.get("LIBRA_LLAMA_CPP_PORT", "8091"))
        base_url = f"http://{host}:{port}"
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("llama.cpp 응답에 choices 가 없습니다.")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama.cpp 응답이 비어있습니다.")
        return content

    def _call_claude(self, model: str, system: str, user: str, max_tokens: int) -> str:
        if not self._claude:
            raise RuntimeError("Claude 클라이언트 없음")
        msg = self._claude.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text  # type: ignore[index]

    def _call_gemini(self, model: str, system: str, user: str, max_tokens: int) -> str:
        if not self._gemini:
            raise RuntimeError("Gemini 클라이언트 없음")
        self._throttle_gemini()
        genai_model = self._gemini.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config={"max_output_tokens": max_tokens},
        )
        response = genai_model.generate_content(user)
        return response.text

    def _throttle_gemini(self) -> None:
        delay_seconds = self._gemini_throttle_seconds()
        if delay_seconds <= 0:
            return
        with self._gemini_throttle_lock:
            now = time.monotonic()
            wait_seconds = delay_seconds - (now - self.__class__._last_gemini_call_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self.__class__._last_gemini_call_at = time.monotonic()

    def _gemini_throttle_seconds(self) -> float:
        raw = os.environ.get("LIBRA_GEMINI_THROTTLE_SECONDS")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except ValueError:
                return 0.0
        free_tier = os.environ.get("LIBRA_GEMINI_FREE_TIER", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if free_tier and self._policy == "gemini":
            return 13.0
        return 0.0

    def _model_family(self, model: LLMModel) -> str:
        if _is_local(model):
            return "Local"
        return "Gemini" if model in (LLMModel.GEMINI_FLASH, LLMModel.GEMINI_PRO) else "Claude"

    # ── 크로스 검증 응답 병합 ───────────────────────────────────

    def _merge_responses(
        self,
        primary: str,
        secondary: str,
        agent_id: str,
        *,
        primary_label: str,
        secondary_label: str,
    ) -> str:
        """
        두 LLM의 응답을 비교하여 합의 도출.
        내용이 유사하면 primary 반환, 크게 다르면 양쪽 포함 + [불일치 주의] 태그.
        """
        p_words = set(primary.lower().split())
        s_words = set(secondary.lower().split())
        if not p_words:
            return secondary

        overlap = len(p_words & s_words) / len(p_words)

        if overlap > 0.5:
            return primary
        else:
            logger.warning(f"[LLMRouter] {agent_id}: LLM 간 불일치 감지 (overlap={overlap:.1%})")
            return (
                f"[{primary_label} 판단] {primary}\n\n"
                f"[{secondary_label} 판단] {secondary}\n\n"
                f"⚠️ 두 모델 간 불일치 감지됨. 추가 검토 권장."
            )


# ── 팩토리: 사용자 DB 키 우선, env 시스템 키 ────────────────────


async def build_router_for_user(user_id: str) -> LLMRouter:
    """
    사용자 DB 키로 LLMRouter를 생성합니다.
    DB 키가 없거나 오류 시 환경 변수 시스템 키로 동작합니다.

    Args:
        user_id: Supabase auth.users.id

    Returns:
        LLMRouter — 사용자 키 또는 시스템 키로 구성된 라우터
    """
    try:
        from .llm_key_loader import load_llm_keys

        user_keys = await load_llm_keys(user_id)
        if user_keys:
            logger.info(
                f"[LLMRouter] user={user_id[:8]}...: DB 키 사용 (policy={user_keys.llm_policy})"
            )
            return LLMRouter(
                anthropic_key=user_keys.anthropic_key,
                gemini_key=user_keys.gemini_key,
                policy=user_keys.llm_policy,
            )
    except Exception as e:
        logger.warning(
            f"[LLMRouter] user={user_id[:8]}...: DB 키 로드 실패, env 시스템 키 사용 — {e}"
        )

    logger.info(f"[LLMRouter] user={user_id[:8]}...: 환경 변수 키 사용")
    return LLMRouter()


# ── 글로벌 싱글턴 (env 키 기반, 하위 호환) ────────────────────────

_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """
    환경 변수 기반 글로벌 라우터 (하위 호환용).
    사용자별 라우터가 필요하면 build_router_for_user()를 사용하세요.
    """
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
