"""SPEC §2: AgentState 스키마 + make_initial_state. 필드 추가·삭제 금지.

기록용 필드(tried_queries, judge_history, evidence, sources)는 append만 —
노드는 기존 리스트에 원소를 더한 '새 리스트'를 반환한다 (덮어쓰기 금지).
retry_count는 Rewriter에서만 +1, hop전환에서만 0 리셋 (CLAUDE.md 절대 규칙 3).
"""
from typing import TypedDict


class AgentState(TypedDict):
    # 제어용
    judge_verdict: str
    judge_source: str        # "gatekeeper"/"llm_judge"
    relevance: str           # "high"/"low"
    sufficiency: str         # "high"/"low"
    hop_index: int
    retry_count: int
    exhausted: bool
    exhausted_reason: str    # "retry"/"hop"/"budget"/"extract"/""
    llm_call_count: int
    # 작업용 (덮어쓰기)
    query: str               # 원본, 불변
    plan: dict               # {query_type: single_hop|multi_hop,
                             #  hop_type: bridge|comparison|None,
                             #  search_queries: [...], reason: str}
    answer_strategy: str     # "정답형"/"탐색형"
    current_hop_query: str
    search_results: list     # [{id, title, text, distance}]
    top1_distance: float
    intermediate_answers: list
    judge_reason: str
    missing: str
    # 기록용 (append만)
    tried_queries: list
    judge_history: list      # [{hop, verdict, source, relevance, sufficiency, reason}]
    evidence: list           # [{"hop": n, "chunk_ids": [...]}] — hop전환·Generator 진입 시 박제
    sources: list
    # 출력
    answer: str


def make_initial_state(query: str) -> AgentState:
    """전 필드 빈 값, current_hop_query=query, tried_queries=[query]."""
    return AgentState(
        judge_verdict="",
        judge_source="",
        relevance="",
        sufficiency="",
        hop_index=0,
        retry_count=0,
        exhausted=False,
        exhausted_reason="",
        llm_call_count=0,
        query=query,
        plan={},
        answer_strategy="",
        current_hop_query=query,
        search_results=[],
        top1_distance=0.0,
        intermediate_answers=[],
        judge_reason="",
        missing="",
        tried_queries=[query],
        judge_history=[],
        evidence=[],
        sources=[],
        answer="",
    )
