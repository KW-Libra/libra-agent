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
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── LLM 모델 열거형 ────────────────────────────────────────────

class LLMModel(str, Enum):
    CLAUDE_HAIKU   = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET  = "claude-sonnet-4-6"
    GEMINI_FLASH   = "gemini-2.5-flash"
    GEMINI_PRO     = "gemini-1.5-pro"


# ── 에이전트별 기본 LLM 배정 ────────────────────────────────────

# balanced 정책에서 각 에이전트가 사용할 기본 모델
AGENT_MODEL_MAP: dict[str, LLMModel] = {
    "risk":        LLMModel.CLAUDE_SONNET,   # 고위험 판단 — Claude
    "tax":         LLMModel.CLAUDE_HAIKU,    # 정형적 계산 — 저렴한 Haiku
    "compliance":  LLMModel.CLAUDE_SONNET,   # 규정 준수 — Claude (신뢰성 우선)
    "macro":       LLMModel.GEMINI_FLASH,    # 뉴스/매크로 요약 — Gemini Flash (고속)
    "sentiment":   LLMModel.GEMINI_FLASH,    # 감성 분석 — Gemini Flash (대량 처리)
    "execution":   LLMModel.CLAUDE_HAIKU,    # 체결 계산 — 정형적, Haiku
    "esg":         LLMModel.GEMINI_FLASH,    # ESG 스크리닝 — Gemini Flash
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
        gemini_key:    str | None = None,
        policy:        str | None = None,
    ) -> None:
        # API 키: 인자 우선, 없으면 환경 변수
        self._anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        self._gemini_key    = gemini_key    or os.environ.get("GEMINI_API_KEY")

        # 라우팅 정책: 인자 우선, 없으면 환경 변수
        self._policy = (policy or os.environ.get("LLM_ROUTING_POLICY", "balanced")).lower()

        # 키 출처 로깅 (민감 정보는 마스킹)
        claude_src = "user-db" if anthropic_key else ("env" if os.environ.get("ANTHROPIC_API_KEY") else "none")
        gemini_src = "user-db" if gemini_key    else ("env" if os.environ.get("GEMINI_API_KEY")    else "none")
        logger.info(f"[LLMRouter] 초기화 — policy={self._policy}, claude={claude_src}, gemini={gemini_src}")

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
            logger.error("[LLMRouter] google-generativeai 패키지 미설치. pip install google-generativeai")
            return None

    # ── 모델 선택 ───────────────────────────────────────────────

    def _select_model(self, agent_id: str) -> LLMModel:
        if self._policy == "claude":
            return LLMModel.CLAUDE_SONNET
        if self._policy == "gemini":
            return LLMModel.GEMINI_FLASH

        # balanced: 에이전트별 최적 배정
        preferred = AGENT_MODEL_MAP.get(agent_id, LLMModel.CLAUDE_HAIKU)
        return preferred

    def model_name_for(self, agent_id: str) -> str:
        return self._select_model(agent_id).value

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
        logger.debug(f"[LLMRouter] {agent_id} → {model.value}")

        primary = self._call_model(model, system, user, max_tokens)

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
            secondary = self._call_model(secondary_model, system, user, max_tokens)
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
        try:
            if model in (LLMModel.CLAUDE_HAIKU, LLMModel.CLAUDE_SONNET):
                return self._call_claude(model.value, system, user, max_tokens)
            else:
                return self._call_gemini(model.value, system, user, max_tokens)
        except Exception as e:
            logger.error(f"[LLMRouter] {model.value} 호출 실패: {e}")
            raise RuntimeError(f"{model.value} LLM 호출 실패: {e}") from e

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
            logger.info(f"[LLMRouter] user={user_id[:8]}...: DB 키 사용 (policy={user_keys.llm_policy})")
            return LLMRouter(
                anthropic_key=user_keys.anthropic_key,
                gemini_key=user_keys.gemini_key,
                policy=user_keys.llm_policy,
            )
    except Exception as e:
        logger.warning(f"[LLMRouter] user={user_id[:8]}...: DB 키 로드 실패, env 시스템 키 사용 — {e}")

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
