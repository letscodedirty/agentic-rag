"""모든 LLM 호출의 단일 경유점 (CLAUDE.md 절대 규칙 4).

call_llm(state, messages) — state["llm_call_count"] 증가, assert <= 20,
temperature=0, 모델명은 .env의 LLM_MODEL.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_LLM_CALLS = 20  # 질의 1건당 LLM 호출 상한 (SPEC §1)

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
