"""baseline 노드 구현 (SPEC §3).

Day 2 범위: 검색(LLM 0회), Planner(LLM 1회), Judge(문지기 + LLM 0~1회).
hop전환·Rewriter·Generator는 Day 3.

LLM 호출 카운트: 노드마다 state의 llm_call_count를 시드로 로컬 카운터를 만들어
call_llm에 넘기고, 누적값을 state 갱신으로 반환한다 (assert ≤ 20 불변식 유지).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.config import MAX_HOP, MAX_RETRY, default_top_k, gate_threshold  # noqa: E402
from core.llm import call_llm  # noqa: E402
from core.state import AgentState  # noqa: E402


# ---------- 검색 (LLM 0회) ----------

def search_node(state: AgentState) -> dict:
    """current_hop_query 임베딩 → top-k → search_results, top1_distance."""
    results, top1 = db.search(state["current_hop_query"], k=default_top_k())
    return {"search_results": results, "top1_distance": top1}


# ---------- Planner (LLM 1회) ----------

PLANNER_SYSTEM = (
    "너는 한국어 위키피디아 영화 도메인 RAG의 검색 계획 수립자다. "
    "반드시 JSON으로만 답하라."
)

PLANNER_USER_TMPL = """질문을 분석해 검색 계획을 세워라.

[질문] <<QUERY>>

판단 기준:
1. query_type: 문서 1개의 내용만으로 답할 수 있으면 "single_hop",
   서로 다른 문서를 단계적으로(또는 나란히) 봐야 하면 "multi_hop".
2. hop_type: multi_hop 중 "1단계에서 알아낸 엔티티로 2단계 문서를 찾아야 하는"
   연쇄형이면 "bridge", "두 대상을 나란히 비교"하는 형태면 "comparison".
   single_hop이면 따옴표 없는 JSON null 리터럴로 출력하라.
3. answer_strategy: 비교·선택을 요구하는 질문("vs", "어느 쪽", "누가 먼저",
   "더 ~한", "중에서" 등)이면 "탐색형", 그 외는 "정답형".
4. search_queries: 질문형 문장을 검색형 질의(핵심 개체·속성 중심)로 변환하라.
   - single_hop: [구체 질의] — 1개
   - bridge: [1단계 구체 질의, "{hop1} 포함 2단계 질의 템플릿"] — 2번째는
     1단계에서 얻을 답이 들어갈 자리에 {hop1} 플레이스홀더를 문자 그대로 남긴다
   - comparison: [대상1 구체 질의, 대상2 구체 질의] — 템플릿 불필요
5. reason: 판단 근거 한 문장.

예시:
- 질문 "영화 파묘는 어떤 장르의 영화인가?" →
  {"query_type": "single_hop", "hop_type": null,
   "search_queries": ["파묘 영화 장르"],
   "answer_strategy": "정답형", "reason": "파묘 문서 하나로 답할 수 있다"}
- 질문 "영화 러브픽션의 감독은 어떤 학교를 졸업했는가?" →
  {"query_type": "multi_hop", "hop_type": "bridge",
   "search_queries": ["러브픽션 감독", "{hop1} 출신 학교"],
   "answer_strategy": "정답형",
   "reason": "감독 이름을 먼저 알아낸 뒤 그 인물 문서에서 학교를 찾아야 한다"}
- 질문 "'박하사탕'과 '우리학교' 중 어느 영화가 먼저 개봉했나?" →
  {"query_type": "multi_hop", "hop_type": "comparison",
   "search_queries": ["박하사탕 개봉일", "우리학교 개봉일"],
   "answer_strategy": "탐색형", "reason": "두 작품의 개봉 시점을 나란히 비교한다"}

JSON: {"query_type": "single_hop|multi_hop", "hop_type": "bridge|comparison|null",
"search_queries": [...], "answer_strategy": "정답형|탐색형", "reason": "..."}"""


def _parse_plan(raw: str):
    """Planner 출력 검증. 실패 시 None. 문자열 "null"/"None"은 None으로 정규화."""
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    qt = out.get("query_type")
    ht = out.get("hop_type")
    if isinstance(ht, str) and ht.strip().lower() in ("null", "none", ""):
        ht = None
    sq = out.get("search_queries")
    strategy = out.get("answer_strategy")
    if qt not in ("single_hop", "multi_hop"):
        return None
    if not isinstance(sq, list) or not sq or not all(isinstance(s, str) for s in sq):
        return None
    if strategy not in ("정답형", "탐색형"):
        return None
    if qt == "single_hop":
        ht = None
    else:
        if ht not in ("bridge", "comparison"):
            return None
        if len(sq) < 2:
            return None
        if ht == "bridge" and "{hop1}" not in sq[1]:
            return None
    plan = {"query_type": qt, "hop_type": ht, "search_queries": sq,
            "reason": str(out.get("reason", ""))}
    return plan, strategy


def planner_node(state: AgentState) -> dict:
    counter = {"llm_call_count": state["llm_call_count"]}
    raw = call_llm(
        counter,
        [{"role": "system", "content": PLANNER_SYSTEM},
         {"role": "user", "content": PLANNER_USER_TMPL.replace("<<QUERY>>", state["query"])}],
        json_mode=True,
    )
    parsed = _parse_plan(raw)
    if parsed is None:  # 파싱 실패 → fallback (SPEC §3)
        plan = {"query_type": "single_hop", "hop_type": None,
                "search_queries": [state["query"]], "reason": "fallback"}
        strategy = "정답형"
    else:
        plan, strategy = parsed
    return {
        "plan": plan,
        "answer_strategy": strategy,
        "current_hop_query": plan["search_queries"][0],
        "llm_call_count": counter["llm_call_count"],
    }


# ---------- Judge (문지기 + LLM 0~1회, 설계 A) ----------

JUDGE_SYSTEM = (
    "너는 검색 결과가 질의에 답하기에 충분한지 판정하는 엄격한 심판이다. "
    "반드시 JSON으로만 답하라."
)

JUDGE_USER_TMPL = """검색 결과가 '현재 검색 질의'에 답하기에 충분한지 판정하라.

[현재 검색 질의] <<HOP_QUERY>>

[검색 결과 top-<<K>>]
<<DOCS>>

판정 기준:
1. relevance: 검색 결과가 현재 검색 질의의 대상·주제와 관련 있으면 "high",
   동떨어져 있으면 "low".
2. sufficiency: 검색 결과 본문만으로 '현재 검색 질의'에 완결된 답을 낼 수 있으면
   "high", 정보가 부족하면 "low". relevance가 "low"면 sufficiency는 반드시 "low"다.
   판정 대상은 오직 위의 '현재 검색 질의'다 — 현재 질의에 답할 수 있다면,
   그 너머의 추가 정보(이후 단계에서 찾을 정보)가 없다는 이유로
   "low"로 판정하지 마라.
3. verdict: relevance와 sufficiency가 모두 "high"면 "sufficient", 아니면 "insufficient".
4. reason: 판정 근거 한 문장.
5. missing: insufficient일 때 부족한 '구체적 개체명·속성명'(예: '전계수의 출신 학교')을
   짧은 구로. sufficient면 "".

JSON: {"verdict": "sufficient|insufficient", "relevance": "high|low",
"sufficiency": "high|low", "reason": "...", "missing": "..."}"""


def mark_exhausted(verdict: str, state: AgentState) -> dict:
    """exhausted 판정·기록은 Judge 단일 책임 (SPEC §3).

    insufficient이고 retry_count>=MAX_RETRY → "retry" / hop_index>=MAX_HOP → "hop".
    Judge 노드(실제·가짜 공용)만 이 헬퍼를 호출한다.
    """
    if verdict == "insufficient":
        if state["retry_count"] >= MAX_RETRY:
            return {"exhausted": True, "exhausted_reason": "retry"}
        if state["hop_index"] >= MAX_HOP:
            return {"exhausted": True, "exhausted_reason": "hop"}
    return {}


def _parse_judgement(raw: str):
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    v, rel, suf = out.get("verdict"), out.get("relevance"), out.get("sufficiency")
    if v not in ("sufficient", "insufficient"):
        return None
    if rel not in ("high", "low") or suf not in ("high", "low"):
        return None
    return {"verdict": v, "relevance": rel, "sufficiency": suf,
            "reason": str(out.get("reason", "")), "missing": str(out.get("missing", ""))}


def _finish_judge(state: AgentState, verdict: str, source: str, relevance: str,
                  sufficiency: str, reason: str, missing: str, llm_count: int) -> dict:
    upd = {
        "judge_verdict": verdict, "judge_source": source,
        "relevance": relevance, "sufficiency": sufficiency,
        "judge_reason": reason, "missing": missing,
        "llm_call_count": llm_count,
    }
    upd.update(mark_exhausted(verdict, state))
    upd["judge_history"] = state["judge_history"] + [{
        "hop": state["hop_index"], "verdict": verdict, "source": source,
        "relevance": relevance, "sufficiency": sufficiency, "reason": reason,
    }]
    return upd


def judge_node(state: AgentState) -> dict:
    gate = gate_threshold()
    # 문지기: LLM 없이 즉시 차단
    if state["top1_distance"] > gate:
        return _finish_judge(
            state, "insufficient", "gatekeeper", "low", "low",
            f"top1_distance {state['top1_distance']:.3f} > {gate} (문지기 차단)",
            "", state["llm_call_count"],
        )
    docs = "\n\n".join(
        f"[{r['title']}]\n{r['text']}" for r in state["search_results"]
    )
    user = (JUDGE_USER_TMPL
            .replace("<<HOP_QUERY>>", state["current_hop_query"])
            .replace("<<K>>", str(len(state["search_results"])))
            .replace("<<DOCS>>", docs))
    counter = {"llm_call_count": state["llm_call_count"]}
    out = None
    for _ in range(2):  # 파싱 실패 → 1회 재호출
        raw = call_llm(
            counter,
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True,
        )
        out = _parse_judgement(raw)
        if out is not None:
            break
    if out is None:  # 재실패 → sufficient 통과 + "parse_fail" 기록 (SPEC §3·§7)
        return _finish_judge(state, "sufficient", "llm_judge", "high", "high",
                             "parse_fail", "", counter["llm_call_count"])
    rel, suf = out["relevance"], out["sufficiency"]
    if rel == "low" and suf == "high":  # 모순 칸 → rel=high 교정 후 sufficient 전진
        rel = "high"
    verdict = "sufficient" if (rel == "high" and suf == "high") else "insufficient"
    missing = out["missing"] if verdict == "insufficient" else ""
    return _finish_judge(state, verdict, "llm_judge", rel, suf,
                         out["reason"], missing, counter["llm_call_count"])
