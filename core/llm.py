"""모든 LLM 호출의 단일 경유점 (CLAUDE.md 절대 규칙 4).

call_llm(state, messages) — state["llm_call_count"] 증가, assert <= 20,
temperature=0, 모델명은 .env의 LLM_MODEL.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_LLM_CALLS = 20  # 질의 1건당 LLM 호출 상한 (SPEC §1)
MAX_CLARIFY_CALLS = 13  # 명료화 노드 전용 상한 (SPEC §8 — 파이프라인 상한과 분리)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_llm(state: dict, messages: list, json_mode: bool = False) -> str:
    """state는 최소한 llm_call_count 키를 다룰 수 있는 dict.

    호출마다 llm_call_count +1, 상한 20 초과 시 assert 실패(불변식).
    """
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1
    assert state["llm_call_count"] <= MAX_LLM_CALLS, (
        f"llm_call_count={state['llm_call_count']} > {MAX_LLM_CALLS} (불변식 위반)"
    )
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content


def call_llm_clarify(counter: dict, messages: list, json_mode: bool = False,
                     temperature: float = 0.0, n: int = 1) -> list:
    """명료화 전용 경유점 (SPEC §8 확정 설계, Day 7 승인 C·D — 추가만).

    - 별도 카운터: counter["clarify_call_count"] += n, assert ≤ 13.
      기존 call_llm·llm_call_count 불변식은 무변경(파이프라인 상한 20과 분리).
    - temperature 인자 허용: 의도 샘플링(0.5)은 SPEC §8이 확정한 예외
      (CLAUDE.md temperature=0 조항은 답변 파이프라인 노드에 적용).
    - n>1이면 단일 API 요청의 n개 choices로 실행하되 호출 수는 n으로 계상
      (설계서 '호출 약 10회' 취지 보존, 지연 단축). 반환은 텍스트 목록(n개).
    """
    counter["clarify_call_count"] = counter.get("clarify_call_count", 0) + n
    assert counter["clarify_call_count"] <= MAX_CLARIFY_CALLS, (
        f"clarify_call_count={counter['clarify_call_count']} > {MAX_CLARIFY_CALLS}"
    )
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=temperature,
        n=n,
        messages=messages,
        **kwargs,
    )
    return [c.message.content for c in resp.choices]
