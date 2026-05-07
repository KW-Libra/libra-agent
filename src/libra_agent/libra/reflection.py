"""Reflection generator for LIBRA decisions.

Adapted from TauricResearch/TradingAgents `tradingagents/graph/reflection.py`
(Apache License 2.0). See `NOTICE` and `docs/THIRD_PARTY_LICENSES.md` for the
full license text.

Modifications relative to the upstream `Reflector` class:

* Korean prompts tailored to LIBRA's decision context (KIS holdings,
  multi-agent debate, Judge-first orchestration).
* Uses LIBRA's :class:`ChatClientProtocol` (``chat_json``) instead of
  LangChain's ``invoke``; the model is instructed to wrap the reflection in a
  small JSON envelope so the existing JSON-only client interface keeps working.
* Accepts the Judge's signal score, the candidate rebalance plan, and the
  user's feedback so the reflection can comment on more than the trade
  outcome alone.
* Returns ``""`` when no client is available so the agent runtime can keep
  running on bare metric evaluation without raising.
"""

from __future__ import annotations

from typing import Mapping

from .llm_clients.base import ChatClientProtocol


_REFLECTION_SYSTEM_PROMPT = (
    "당신은 자신의 과거 판단을 결과가 나온 시점에 되돌아보는 한국 시장 트레이딩 분석가입니다.\n"
    "정확히 2~4문장의 평이한 산문으로 reflection을 작성하십시오 — 불릿, 헤더, 마크다운, 인용 부호 사용 금지.\n\n"
    "다음 순서로 다루십시오:\n"
    "  (1) 방향성 판단이 옳았는가? 실현 수익률 수치를 인용하십시오.\n"
    "  (2) 투자 논제 중 어느 부분이 맞고 어느 부분이 틀렸는가?\n"
    "  (3) 다음 유사한 분석에 적용할 구체적 교훈 한 가지.\n\n"
    "구체적이고 간결하게. 결과물은 결정 로그에 그대로 저장되어 미래 에이전트가 다시 읽으므로,\n"
    "모든 단어가 자기 자리를 지켜야 합니다.\n\n"
    "응답은 반드시 다음 JSON 한 객체로만 반환하십시오:\n"
    "  {\"reflection\": \"여기에 2~4문장 한국어 reflection\"}\n"
)


def _format_rebalance_plan(plan: Mapping[str, float]) -> str:
    if not plan:
        return "(없음)"
    return ", ".join(f"{ticker} {weight:+.2%}" for ticker, weight in plan.items())


def _build_user_prompt(
    *,
    decision: str,
    rebalance_plan: Mapping[str, float],
    signal_score: float,
    realized_return_pct: float,
    cost_pct: float,
    user_feedback: str | None,
) -> str:
    feedback_line = ""
    cleaned_feedback = (user_feedback or "").strip()
    if cleaned_feedback:
        feedback_line = f"사용자 피드백: {cleaned_feedback}\n"
    return (
        f"실현 수익률: {realized_return_pct:+.2f}%\n"
        f"비용: {cost_pct:+.2f}%\n"
        f"신호 점수 (Judge): {signal_score:+.3f}\n"
        f"{feedback_line}"
        f"\n결정: {decision}\n"
        f"리밸런싱 초안: {_format_rebalance_plan(rebalance_plan)}\n"
    )


def generate_reflection(
    *,
    client: ChatClientProtocol | None,
    decision: str,
    rebalance_plan: Mapping[str, float],
    signal_score: float,
    realized_return_pct: float,
    cost_pct: float,
    user_feedback: str | None,
) -> str:
    """Return a 2~4-sentence Korean reflection on the decision outcome.

    Returns an empty string when no LLM client is supplied so the caller can
    fall back to bare metric evaluation without raising.
    """
    if client is None:
        return ""
    user_prompt = _build_user_prompt(
        decision=decision,
        rebalance_plan=rebalance_plan,
        signal_score=signal_score,
        realized_return_pct=realized_return_pct,
        cost_pct=cost_pct,
        user_feedback=user_feedback,
    )
    response = client.chat_json(
        system_prompt=_REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    text = str(response.get("reflection") or "").strip()
    if text:
        return text
    # The model returned valid JSON but missed the expected key. Recover by
    # joining any string fields it produced rather than discarding the response.
    fallback_parts = [str(value).strip() for value in response.values() if isinstance(value, str)]
    return " ".join(part for part in fallback_parts if part)
