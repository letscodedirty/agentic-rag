"""agents/v2 그래프 토폴로지 (SPEC §8, docs/V2_AGENT.md — Day 7).

baseline 골격(SPEC §1) + list 직행 경로:
Planner → [분기] → {list_lookup → Generator | 검색 → Judge → [분기] →
{hop전환→검색(추출 재실패 시 Generator) | Generator | Rewriter→검색}}

명료화 노드는 이번 범위 밖 — 그래프 진입 전 훅 자리만 예약(clarification_hook).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, StateGraph  # noqa: E402

from core.config import MAX_HOP, default_top_k  # noqa: E402
from core.state import AgentStateV2  # noqa: E402

NODE_NAMES = ["planner", "list_lookup", "search", "judge",
              "hop_transition", "rewriter", "generator"]

# 명료화 훅 (SPEC §8 확정 — agents/v2/clarify.py 구현):
# callable(question:str) -> None(통과) | {"proceed_query": q}(해석 확정 통과) |
# {"clarification": {...}}(조기 종료). 기본은 clarify.clarify_question이며,
# 이 변수로 대체 훅을 장착할 수 있다. run_agent(clarify=...)로 on/off.
clarification_hook = None


def has_next_hop(state: AgentStateV2) -> bool:
    plan = state.get("plan") or {}
    return plan.get("query_type") == "multi_hop" and state["hop_index"] < MAX_HOP - 1


def route_after_planner(state: AgentStateV2) -> str:
    """list는 Judge 없이 색인 조회 → Generator 직행 (SPEC §8 결정 1)."""
    plan = state.get("plan") or {}
    return "list" if plan.get("query_type") == "list" else "search"


def route_after_judge(state: AgentStateV2) -> str:
    if state["judge_verdict"] == "sufficient":
        return "hop" if has_next_hop(state) else "generate"
    if state["exhausted"]:
        return "generate"
    return "rewrite"


def route_after_hop_transition(state: AgentStateV2) -> str:
    return "generate" if state["exhausted_reason"] == "extract" else "search"


def build_graph(nodes: dict):
    g = StateGraph(AgentStateV2)
    for name in NODE_NAMES:
        g.add_node(name, nodes[name])
    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"list": "list_lookup", "search": "search"},
    )
    g.add_edge("list_lookup", "generator")
    g.add_edge("search", "judge")
    g.add_conditional_edges(
        "judge", route_after_judge,
        {"hop": "hop_transition", "generate": "generator", "rewrite": "rewriter"},
    )
    g.add_conditional_edges(
        "hop_transition", route_after_hop_transition,
        {"search": "search", "generate": "generator"},
    )
    g.add_edge("rewriter", "search")
    g.add_edge("generator", END)
    return g.compile()


_graph_cache = {}


def _empty_contract(question: str) -> dict:
    """명료화 조기 종료용 — 계약 키 빈 값 (승인 G, 상위집합 유지)."""
    return {
        "answer": "", "evidence": [], "sources": [], "plan": {}, "strategy": "",
        "judge_history": [], "intermediate_answers": [], "rewrite_history": [],
        "retry_total": 0, "hop_reached": 0, "exhausted": False,
        "exhausted_reason": "", "llm_calls": 0, "top1_distance": None,
        "structured_hits": {"infobox": [], "filmography": []},
        "list_summary": None, "clarification": None,
    }


def run_agent(question: str, top_k: int = None, clarify: bool = False,
              tau: float = None) -> dict:
    """하네스·백엔드 진입점 — 계약(질문 in → evidence 포함 out)은 baseline과 동일,
    v2 추가 정보(structured/list 요약, clarification)를 상위집합으로 포함.

    clarify=True면 그래프 진입 전 명료화 훅 실행 (τ 확정 전 기본 OFF — 승인 H).
    """
    import time

    from agents.v2.nodes import make_nodes
    from core.state import make_initial_state_v2

    if clarify:
        from agents.v2 import clarify as _clarify
        t0 = time.time()
        hook = clarification_hook or (lambda q: _clarify.clarify_question(q, tau=tau))
        res = hook(question)
        if res is not None:
            if "proceed_query" in res:  # 재질문 취소 — 확정 해석으로 계속
                question = res["proceed_query"]
            else:  # 재질문 조기 종료
                out = _empty_contract(question)
                out["clarification"] = res["clarification"]
                out["clarify_calls"] = res.get("clarify_calls", 0)
                out["elapsed_sec"] = round(time.time() - t0, 2)
                return out

    k = top_k if top_k is not None else default_top_k()
    if k not in _graph_cache:
        _graph_cache[k] = build_graph(make_nodes(k))
    t0 = time.time()
    final = _graph_cache[k].invoke(
        make_initial_state_v2(question), config={"recursion_limit": 50}
    )
    structured = final.get("structured_results") or {}
    list_results = final.get("list_results") or {}
    return {
        "answer": final["answer"],
        "evidence": final["evidence"],
        "sources": final["sources"],
        "plan": final["plan"],
        "strategy": final["answer_strategy"],
        "judge_history": final["judge_history"],
        "intermediate_answers": final["intermediate_answers"],
        "rewrite_history": final["tried_queries"][1:],
        "retry_total": max(0, len(final["tried_queries"]) - 1),
        "hop_reached": final["hop_index"],
        "exhausted": final["exhausted"],
        "exhausted_reason": final["exhausted_reason"],
        "llm_calls": final["llm_call_count"],
        "top1_distance": final["top1_distance"],
        "elapsed_sec": round(time.time() - t0, 2),
        # v2 상위집합 필드
        "structured_hits": {
            "infobox": [r["title"] for r in structured.get("infobox", [])],
            "filmography": [f["person"] for f in structured.get("filmography", [])],
        },
        "list_summary": ({"kind": list_results.get("kind"),
                          "key": list_results.get("key"),
                          "n_items": len(list_results.get("items", []))}
                         if list_results else None),
        "clarification": final.get("clarification") or None,
    }
