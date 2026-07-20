"""Day 2: LLM 없는 가짜 노드로 그래프 골격의 4가지 흐름 검증 (PLAN Day 2).

시나리오: ① 정상(멀티홉) ② 재작성 루프 ③ 사후 재계획 ④ exhausted
실행: ./venv/bin/python agents/baseline/flow_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.baseline.graph import build_graph  # noqa: E402
from agents.baseline.nodes import mark_exhausted  # noqa: E402
from core.state import make_initial_state  # noqa: E402


def make_fake_nodes(plan: dict, judge_script: list, trace: list,
                    replan_on_rewrite: dict = None):
    """가짜 노드 세트. judge_script의 판정을 순서대로 소비한다."""

    def traced(name, fn):
        def wrapped(state):
            trace.append(name)
            return fn(state)
        return wrapped

    def planner(state):
        return {
            "plan": plan,
            "answer_strategy": "정답형",
            "current_hop_query": plan["search_queries"][0],
        }

    def search(state):
        hop = state["hop_index"]
        results = [{"id": f"doc_h{hop}", "title": f"doc_h{hop}",
                    "text": "가짜 본문", "distance": 0.4}]
        return {"search_results": results, "top1_distance": 0.4}

    idx = [0]

    def judge(state):
        script = judge_script[idx[0]]
        idx[0] += 1
        upd = {
            "judge_verdict": script["verdict"],
            "judge_source": script.get("source", "llm_judge"),
            "relevance": script.get("relevance", "high"),
            "sufficiency": script.get("sufficiency", "high"),
            "judge_reason": script.get("reason", "가짜 판정"),
            "missing": script.get("missing", ""),
        }
        upd.update(mark_exhausted(script["verdict"], state))  # 단일 책임 헬퍼
        entry = {"hop": state["hop_index"], "verdict": script["verdict"],
                 "source": upd["judge_source"], "relevance": upd["relevance"],
                 "sufficiency": upd["sufficiency"], "reason": upd["judge_reason"]}
        upd["judge_history"] = state["judge_history"] + [entry]
        return upd

    def hop_transition(state):
        chunk_ids = [r["id"] for r in state["search_results"]]
        nxt = state["plan"]["search_queries"][1].replace("{hop1}", "중간답")
        return {
            "hop_index": state["hop_index"] + 1,
            "retry_count": 0,  # 유일한 리셋 지점
            "evidence": state["evidence"] + [
                {"hop": state["hop_index"], "chunk_ids": chunk_ids}],
            "intermediate_answers": state["intermediate_answers"] + ["중간답"],
            "current_hop_query": nxt,
        }

    def rewriter(state):
        upd = {
            "current_hop_query": f"재작성 질의 #{state['retry_count'] + 1}",
            "tried_queries": state["tried_queries"] + [
                f"재작성 질의 #{state['retry_count'] + 1}"],
            "retry_count": state["retry_count"] + 1,  # 유일한 증가 지점
        }
        if replan_on_rewrite and state["retry_count"] == 0:
            upd["plan"] = {**state["plan"], **replan_on_rewrite}  # 사후 재계획
        return upd

    def generator(state):
        chunk_ids = [r["id"] for r in state["search_results"]]
        return {
            "answer": "[exhausted 답변]" if state["exhausted"] else "[정상 답변]",
            "evidence": state["evidence"] + [
                {"hop": state["hop_index"], "chunk_ids": chunk_ids}],
            "sources": state["sources"] + [
                {"hop": state["hop_index"], "titles": chunk_ids}],
        }

    fns = {"planner": planner, "search": search, "judge": judge,
           "hop_transition": hop_transition, "rewriter": rewriter,
           "generator": generator}
    return {k: traced(k, v) for k, v in fns.items()}


SINGLE_PLAN = {"query_type": "single_hop", "hop_type": None,
               "search_queries": ["단일 질의"], "reason": "테스트"}
MULTI_PLAN = {"query_type": "multi_hop", "hop_type": "bridge",
              "search_queries": ["1단계 질의", "{hop1} 2단계 질의"], "reason": "테스트"}

SCENARIOS = [
    {
        "name": "① 정상 (멀티홉 bridge: hop1 충분 → hop전환 → hop2 충분 → 생성)",
        "plan": MULTI_PLAN,
        "script": [{"verdict": "sufficient"}, {"verdict": "sufficient"}],
        "path": ["planner", "search", "judge", "hop_transition",
                 "search", "judge", "generator"],
        "checks": lambda s: [
            ("evidence 2개(hop0+최종)", len(s["evidence"]) == 2),
            ("hop_index=1", s["hop_index"] == 1),
            ("retry_count=0", s["retry_count"] == 0),
            ("exhausted=False", not s["exhausted"]),
            ("중간답 1개", len(s["intermediate_answers"]) == 1),
            ("정상 답변", s["answer"] == "[정상 답변]"),
        ],
    },
    {
        "name": "② 재작성 루프 (불충분 → Rewriter → 재검색 → 충분 → 생성)",
        "plan": SINGLE_PLAN,
        "script": [{"verdict": "insufficient", "relevance": "low",
                    "sufficiency": "low"}, {"verdict": "sufficient"}],
        "path": ["planner", "search", "judge", "rewriter",
                 "search", "judge", "generator"],
        "checks": lambda s: [
            ("retry_count=1", s["retry_count"] == 1),
            ("tried_queries 2개", len(s["tried_queries"]) == 2),
            ("evidence 1개(최종)", len(s["evidence"]) == 1),
            ("judge_history 2개", len(s["judge_history"]) == 2),
            ("exhausted=False", not s["exhausted"]),
        ],
    },
    {
        "name": "③ 사후 재계획 (single로 시작 → Rewriter가 multi로 갱신 → hop전환 경유)",
        "plan": SINGLE_PLAN,
        "replan": {"query_type": "multi_hop", "hop_type": "bridge",
                   "search_queries": ["재작성 질의 #1", "{hop1} 2단계 질의"]},
        "script": [{"verdict": "insufficient", "relevance": "high",
                    "sufficiency": "low", "missing": "2단계 정보"},
                   {"verdict": "sufficient"}, {"verdict": "sufficient"}],
        "path": ["planner", "search", "judge", "rewriter", "search", "judge",
                 "hop_transition", "search", "judge", "generator"],
        "checks": lambda s: [
            ("plan이 multi_hop으로 갱신", s["plan"]["query_type"] == "multi_hop"),
            ("evidence 2개", len(s["evidence"]) == 2),
            ("hop전환에서 retry 리셋", s["retry_count"] == 0),
            ("hop_index=1", s["hop_index"] == 1),
        ],
    },
    {
        "name": "④ exhausted (불충분 3연속 → retry 소진 → exhausted 생성)",
        "plan": SINGLE_PLAN,
        "script": [{"verdict": "insufficient", "relevance": "low", "sufficiency": "low"},
                   {"verdict": "insufficient", "relevance": "low", "sufficiency": "low"},
                   {"verdict": "insufficient", "relevance": "low", "sufficiency": "low"}],
        "path": ["planner", "search", "judge", "rewriter", "search", "judge",
                 "rewriter", "search", "judge", "generator"],
        "checks": lambda s: [
            ("exhausted=True", s["exhausted"]),
            ("reason='retry'", s["exhausted_reason"] == "retry"),
            ("retry_count=2 (MAX)", s["retry_count"] == 2),
            ("tried_queries 3개", len(s["tried_queries"]) == 3),
            ("exhausted 답변", s["answer"] == "[exhausted 답변]"),
        ],
    },
]


def main():
    all_ok = True
    for sc in SCENARIOS:
        trace = []
        nodes = make_fake_nodes(sc["plan"], sc["script"], trace,
                                replan_on_rewrite=sc.get("replan"))
        graph = build_graph(nodes)
        final = graph.invoke(make_initial_state("테스트 질문"),
                             config={"recursion_limit": 40})
        print(f"\n{sc['name']}")
        print(f"  경로: {' → '.join(trace)}")
        path_ok = trace == sc["path"]
        print(f"  [{'PASS' if path_ok else 'FAIL'}] 경로 일치")
        all_ok &= path_ok
        for label, ok in sc["checks"](final):
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            all_ok &= ok
        assert final["llm_call_count"] == 0, "가짜 노드인데 LLM 호출 발생"
    print(f"\n{'=' * 50}\n전체: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
