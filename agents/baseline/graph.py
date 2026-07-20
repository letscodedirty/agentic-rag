"""SPEC §1: baseline Agentic 그래프 토폴로지.

노드 6개: Planner → 검색 → Judge → [조건부 엣지] → {hop전환→검색 | Generator | Rewriter→검색}

조건부 엣지는 Judge 뒤 유일한 분기이며 state '읽기만' 하는 순수 함수다:
- ① sufficient & 다음 hop 있음 → hop전환
- ② sufficient & 마지막 hop     → Generator
- ③ insufficient & retry 남음   → Rewriter
- ④ insufficient & 한도 소진    → Generator  (exhausted 판정·기록은 Judge 단일 책임)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, StateGraph  # noqa: E402

from core.config import MAX_HOP  # noqa: E402
from core.state import AgentState  # noqa: E402

NODE_NAMES = ["planner", "search", "judge", "hop_transition", "rewriter", "generator"]


def has_next_hop(state: AgentState) -> bool:
    """다음 hop 존재 = 계획이 multi_hop이고 아직 마지막 hop이 아님 (읽기 전용)."""
    plan = state.get("plan") or {}
    return plan.get("query_type") == "multi_hop" and state["hop_index"] < MAX_HOP - 1


def route_after_judge(state: AgentState) -> str:
    """순수 함수 — state를 읽기만 한다. 쓰기는 어떤 노드의 몫."""
    if state["judge_verdict"] == "sufficient":
        return "hop" if has_next_hop(state) else "generate"
    if state["exhausted"]:
        return "generate"
    return "rewrite"


def build_graph(nodes: dict):
    """nodes: {노드명: 함수}. 가짜 노드(테스트)든 실제 노드든 같은 골격을 쓴다."""
    g = StateGraph(AgentState)
    for name in NODE_NAMES:
        g.add_node(name, nodes[name])
    g.set_entry_point("planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "judge")
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {"hop": "hop_transition", "generate": "generator", "rewrite": "rewriter"},
    )
    g.add_edge("hop_transition", "search")
    g.add_edge("rewriter", "search")
    g.add_edge("generator", END)
    return g.compile()
