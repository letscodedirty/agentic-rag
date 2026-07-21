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


# ---------- hop전환 (LLM 0~1회) ----------

EXTRACT_SYSTEM = (
    "너는 문서에서 요구된 값만 정확히 추출하는 도구다. 반드시 JSON으로만 답하라."
)

EXTRACT_USER_TMPL = """아래 문서들에서 다음 질의의 답만 추출하라.

[질의] <<HOP_QUERY>>

[문서]
<<DOCS>>

규칙:
1. answer는 질의가 묻는 대상의 이름/값만 — 짧은 구(30자 이내). 설명·문장 금지.
2. 문서에 답이 없으면 answer를 빈 문자열 ""로 하라. 추측 금지.

JSON: {"answer": "..."}"""


def hop_transition_node(state: AgentState) -> dict:
    """bridge: 중간 답 추출 → {hop1} 치환. comparison: 추출 생략.

    공통 쓰기: hop_index+1, retry_count=0(유일 리셋), evidence·sources append.
    추출 재실패 시 공통 쓰기 생략 + exhausted=True/reason="extract" (SPEC §3).
    """
    plan = state["plan"]
    chunk_ids = [r["id"] for r in state["search_results"]]
    common = {
        "hop_index": state["hop_index"] + 1,
        "retry_count": 0,
        "evidence": state["evidence"] + [
            {"hop": state["hop_index"], "chunk_ids": chunk_ids}],
        "sources": state["sources"] + [
            {"hop": state["hop_index"], "titles": chunk_ids}],
    }
    if plan.get("hop_type") == "comparison":
        return {**common, "current_hop_query": plan["search_queries"][1]}

    # bridge: 중간 답 추출 (빈 답 → 1회 재시도)
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in state["search_results"])
    user = (EXTRACT_USER_TMPL
            .replace("<<HOP_QUERY>>", state["current_hop_query"])
            .replace("<<DOCS>>", docs))
    counter = {"llm_call_count": state["llm_call_count"]}
    answer = ""
    for _ in range(2):
        raw = call_llm(
            counter,
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True,
        )
        try:
            answer = (json.loads(raw).get("answer") or "").strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            answer = ""
        if answer:
            break
    if not answer:  # 재실패: 전환 미발생 — 공통 쓰기 생략 (승인 수정 1)
        return {"exhausted": True, "exhausted_reason": "extract",
                "llm_call_count": counter["llm_call_count"]}
    return {
        **common,
        "intermediate_answers": state["intermediate_answers"] + [answer],
        "current_hop_query": plan["search_queries"][1].replace("{hop1}", answer),
        "llm_call_count": counter["llm_call_count"],
    }


# ---------- Rewriter (LLM 1회, 3모드) ----------

REWRITER_SYSTEM = "너는 실패한 검색 질의를 개선하는 전문가다. 반드시 JSON으로만 답하라."

REWRITER_USER_TMPL = """검색이 실패했다. 아래 정보를 바탕으로 '다음 검색 질의'를 새로 작성하라.

[원본 질문] <<QUERY>>  ← 최종 목표(앵커). 이 의도에서 벗어나지 마라.
[현재 hop 질의] <<HOP_QUERY>>
[이미 시도한 질의] <<TRIED>>  ← 이것들과 반드시 달라야 한다.
[판정 사유] <<REASON>>
[부족한 정보] <<MISSING>>
[직전 검색 결과 발췌]
<<SNIPPETS>>

[재작성 지시]
<<MODE>>
<<LAST_CHANCE>>

<<JSON_SPEC>>"""

MODE_C = """검색 결과가 주제와 완전히 동떨어져 문지기에 차단됐다. 질의를 전면 재작성하라 —
다른 핵심 개체명, 다른 표현, 다른 관점을 탐색적으로 시도하라. 기존 질의의 단어를
그대로 재조합하는 수준은 금지."""

MODE_A = """검색 결과가 질의의 대상과 다른 주제를 가리켰다. 검색 방향을 전환하라 —
대상을 더 정확히 특정하는 표현(정확한 작품 제목, 인물 전체 이름, 구별 속성)으로
바꿔서 엉뚱한 문서가 잡히지 않게 하라."""

MODE_B = """검색 방향은 맞지만 정보가 부족하다. [부족한 정보]를 정면으로 겨냥하는 질의로
보강하라. 만약 부족한 정보가 현재 문서가 아니라 '다른 문서'(예: 언급된 인물·작품의
문서)에 있을 것으로 보이면 2단계 계획을 함께 제안하라 — replan의
hop2_query_template에는 1단계 답이 들어갈 자리에 {hop1}을 문자 그대로 남겨라.
한 문서로 해결될 문제면 replan은 null."""

JSON_SPEC_AC = 'JSON: {"new_query": "..."}'
JSON_SPEC_B = """JSON: {"new_query": "...", "replan": null 또는
{"hop_type": "bridge", "hop2_query_template": "{hop1} ..."}}"""

LAST_CHANCE = "이번이 마지막 기회다. 지금까지와 확연히 다른 각도로 과감하게 전환하라."


def _rewriter_mode(state: AgentState) -> str:
    if state["judge_source"] == "gatekeeper":
        return "C"
    if state["relevance"] == "low":
        return "A"
    return "B"


def rewriter_node(state: AgentState) -> dict:
    mode = _rewriter_mode(state)
    snippets = "\n".join(
        f"[{r['title']}] {r['text'][:120]}" for r in state["search_results"][:3]
    ) or "(없음)"
    user = (REWRITER_USER_TMPL
            .replace("<<QUERY>>", state["query"])
            .replace("<<HOP_QUERY>>", state["current_hop_query"])
            .replace("<<TRIED>>", json.dumps(state["tried_queries"], ensure_ascii=False))
            .replace("<<REASON>>", state["judge_reason"] or "(없음)")
            .replace("<<MISSING>>", state["missing"] or "(없음)")
            .replace("<<SNIPPETS>>", snippets)
            .replace("<<MODE>>", {"A": MODE_A, "B": MODE_B, "C": MODE_C}[mode])
            .replace("<<LAST_CHANCE>>",
                     LAST_CHANCE if state["retry_count"] + 1 >= MAX_RETRY else "")
            .replace("<<JSON_SPEC>>", JSON_SPEC_B if mode == "B" else JSON_SPEC_AC))
    counter = {"llm_call_count": state["llm_call_count"]}
    new_query, replan = "", None
    for attempt in range(2):  # 중복이면 1회 재요청 (SPEC §3)
        user_txt = user if attempt == 0 else (
            user + "\n\n주의: 직전 제안이 [이미 시도한 질의]와 중복이었다. "
                   "목록에 없는 질의를 제안하라.")
        raw = call_llm(
            counter,
            [{"role": "system", "content": REWRITER_SYSTEM},
             {"role": "user", "content": user_txt}],
            json_mode=True,
        )
        try:
            out = json.loads(raw)
            new_query = str(out.get("new_query") or "").strip()
            replan = out.get("replan") if mode == "B" else None
        except (json.JSONDecodeError, TypeError):
            new_query, replan = "", None
        if new_query and new_query not in state["tried_queries"]:
            break
    if not new_query:  # 파싱 재실패 안전판 — 원본 기반 변형으로 전진
        new_query = f"{state['query']} (재검색 {state['retry_count'] + 1})"
    upd = {
        "current_hop_query": new_query,
        "tried_queries": state["tried_queries"] + [new_query],
        "retry_count": state["retry_count"] + 1,  # 유일한 증가 지점
        "llm_call_count": counter["llm_call_count"],
    }
    # 사후 재계획 가드: 모드 B & single_hop & hop0 & 유효 템플릿일 때만 (SPEC §3)
    if (mode == "B" and isinstance(replan, dict)
            and state["plan"].get("query_type") == "single_hop"
            and state["hop_index"] == 0):
        tmpl = str(replan.get("hop2_query_template") or "")
        if replan.get("hop_type") == "bridge" and "{hop1}" in tmpl:
            upd["plan"] = {"query_type": "multi_hop", "hop_type": "bridge",
                           "search_queries": [new_query, tmpl],
                           "reason": "사후 재계획"}
    return upd


# ---------- Generator (LLM 1회, 2×2) ----------

GENERATOR_SYSTEM = "너는 제공된 문서만 근거로 답하는 한국어 QA 어시스턴트다."

GEN_INSTR = {
    ("정답형", False): "질문에 대한 정답을 첫 문장에서 명확히 제시하고, 근거를 1~2문장 덧붙여라.",
    ("탐색형", False): ("각 대상의 근거 값을 항목별로 나열한 뒤"
                     "(예: '- 박하사탕: 1999년 개봉'), "
                     "마지막 줄에 비교 결론을 한 문장으로 제시하라."),
    ("정답형", True): ("검색이 충분한 근거를 찾지 못한 채 종료됐다. 확정적인 답을 "
                    "단정하지 마라. 문서에서 확인되는 부분까지만 정리하고, "
                    "무엇을 확인할 수 없었는지 명시하라."),
    ("탐색형", True): ("검색이 충분한 근거를 찾지 못한 채 종료됐다. 확인된 대상의 "
                    "근거 값만 나열하고, 확인하지 못한 값을 명시하라. 단정적 비교 "
                    "결론 대신 제한적 결론(또는 '비교 불가')을 밝혀라."),
}

GEN_COMMON_RULES = """공통 규칙:
- 제공 문서에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 밝혀라.
- 답변 끝에 근거 문서를 (출처: title1, title2) 형식으로 표기하라."""


def fetch_chunks(chunk_ids: list) -> list:
    """comparison hop1 문서 재조회 — DB 접근은 검색 계층(core/db) 경유 (SPEC §3)."""
    return db.get_by_ids(chunk_ids)


def generator_node(state: AgentState) -> dict:
    plan = state["plan"] or {}
    docs_list = list(state["search_results"])
    # comparison: hop1 문서를 evidence의 chunk_ids로 재조회해 포함 (SPEC §3)
    if plan.get("hop_type") == "comparison" and state["evidence"]:
        hop1_ids = state["evidence"][0]["chunk_ids"][:3]
        have = {r["id"] for r in docs_list}
        docs_list = [c for c in fetch_chunks(hop1_ids) if c["id"] not in have] + docs_list
    strategy = state["answer_strategy"] if state["answer_strategy"] in ("정답형", "탐색형") else "정답형"
    instr = GEN_INSTR[(strategy, bool(state["exhausted"]))]
    inter = (f"\n[1단계에서 확인한 중간 답] {', '.join(state['intermediate_answers'])}"
             if state["intermediate_answers"] else "")
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in docs_list) or "(문서 없음)"
    user = (f"[질문] {state['query']}{inter}\n\n[문서]\n{docs}\n\n"
            f"[작성 지시]\n{instr}\n\n{GEN_COMMON_RULES}")
    counter = {"llm_call_count": state["llm_call_count"]}
    answer = call_llm(
        counter,
        [{"role": "system", "content": GENERATOR_SYSTEM},
         {"role": "user", "content": user}],
    )
    chunk_ids = [r["id"] for r in state["search_results"]]
    return {
        "answer": answer,
        "evidence": state["evidence"] + [
            {"hop": state["hop_index"], "chunk_ids": chunk_ids}],
        "sources": state["sources"] + [
            {"hop": state["hop_index"], "titles": chunk_ids}],
        "llm_call_count": counter["llm_call_count"],
    }


def make_nodes(top_k: int = None) -> dict:
    """실제 노드 세트. top_k 지정 시 검색 노드만 해당 k로 동작."""
    if top_k is None:
        search = search_node
    else:
        def search(state):
            results, top1 = db.search(state["current_hop_query"], k=top_k)
            return {"search_results": results, "top1_distance": top1}
    return {"planner": planner_node, "search": search, "judge": judge_node,
            "hop_transition": hop_transition_node, "rewriter": rewriter_node,
            "generator": generator_node}


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
