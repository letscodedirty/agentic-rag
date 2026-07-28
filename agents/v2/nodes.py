"""agents/v2 노드 구현 (SPEC §8, docs/V2_AGENT.md — Day 7).

baseline(agents/baseline/nodes.py) 이식 + 3층 병합 확장. LLM 프롬프트는 Day 7
사용자 승인본 전문. 불변식: retry_count는 Rewriter만 +1·hop전환만 0 리셋,
기록용 필드 append만, 모든 LLM 호출 call_llm 경유 (CLAUDE.md 절대 규칙).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.config import MAX_HOP, MAX_RETRY, default_top_k, gate_threshold  # noqa: E402
from core.llm import call_llm  # noqa: E402
from core.state import AgentStateV2  # noqa: E402

from agents.v2 import knowledge  # noqa: E402


# ---------- Planner (LLM 1회, 승인 프롬프트 — list·entities 확장) ----------

PLANNER_SYSTEM = (
    "너는 한국어 위키피디아 영화 도메인 RAG의 검색 계획 수립자다. "
    "반드시 JSON으로만 답하라."
)

PLANNER_USER_TMPL = """질문을 분석해 검색 계획을 세워라.

[질문] <<QUERY>>

판단 기준:
1. query_type: 문서 1개의 내용만으로 답할 수 있으면 "single_hop",
   서로 다른 문서를 단계적으로(또는 나란히) 봐야 하면 "multi_hop",
   특정 인물·작품·조건과 관련된 여러 항목을 빠짐없이 나열하라는 요구
   ("전부", "모두", "목록", "~한 영화들", "여러 개 추천해줘" 등)면 "list".
   단, 여러 항목 중 하나를 고르는 질문(가장 ~한, 첫 번째 등)은 list가 아니다.
2. hop_type: multi_hop 중 "1단계에서 알아낸 엔티티로 2단계 문서를 찾아야 하는"
   연쇄형이면 "bridge", "두 대상을 나란히 비교"하는 형태면 "comparison".
   single_hop·list면 따옴표 없는 JSON null 리터럴로 출력하라.
3. answer_strategy: list면 "목록형". 비교·선택을 요구하는 질문("vs", "어느 쪽",
   "누가 먼저", "더 ~한", "중에서" 등)이면 "탐색형", 그 외는 "정답형".
4. search_queries: 질문형 문장을 검색형 질의(핵심 개체·속성 중심)로 변환하라.
   - single_hop: [구체 질의] — 1개
   - bridge: [1단계 구체 질의, "{hop1} 포함 2단계 질의 템플릿"] — 2번째는
     1단계에서 얻을 답이 들어갈 자리에 {hop1} 플레이스홀더를 문자 그대로 남긴다
   - comparison: [대상1 구체 질의, 대상2 구체 질의] — 템플릿 불필요
   - list: [목록의 기준이 되는 핵심 키 1개] — 인물 이름 또는 분류성 키워드
5. entities: 질문에 등장한 개체명(영화 제목, 인물 이름 등 고유명사)을 질문의
   표기 그대로 배열로 추출하라. 없으면 빈 배열 [].
6. reason: 판단 근거 한 문장.

예시:
- 질문 "영화 파묘는 어떤 장르의 영화인가?" →
  {"query_type": "single_hop", "hop_type": null,
   "search_queries": ["파묘 영화 장르"], "entities": ["파묘"],
   "answer_strategy": "정답형", "reason": "파묘 문서 하나로 답할 수 있다"}
- 질문 "영화 러브픽션의 감독은 어떤 학교를 졸업했는가?" →
  {"query_type": "multi_hop", "hop_type": "bridge",
   "search_queries": ["러브픽션 감독", "{hop1} 출신 학교"],
   "entities": ["러브픽션"], "answer_strategy": "정답형",
   "reason": "감독 이름을 먼저 알아낸 뒤 그 인물 문서에서 학교를 찾아야 한다"}
- 질문 "'박하사탕'과 '우리학교' 중 어느 영화가 먼저 개봉했나?" →
  {"query_type": "multi_hop", "hop_type": "comparison",
   "search_queries": ["박하사탕 개봉일", "우리학교 개봉일"],
   "entities": ["박하사탕", "우리학교"], "answer_strategy": "탐색형",
   "reason": "두 작품의 개봉 시점을 나란히 비교한다"}
- 질문 "배우 유해진이 출연한 영화를 모두 알려줘" →
  {"query_type": "list", "hop_type": null,
   "search_queries": ["유해진"], "entities": ["유해진"],
   "answer_strategy": "목록형", "reason": "유해진의 출연작 전체 목록을 요구한다"}
- 질문 "한국 공포 영화 여러 편 추천해줘" →
  {"query_type": "list", "hop_type": null,
   "search_queries": ["공포 영화"], "entities": [],
   "answer_strategy": "목록형",
   "reason": "특정 개체가 아니라 분류(공포 영화)에 속한 여러 작품을 요구한다"}

JSON: {"query_type": "single_hop|multi_hop|list", "hop_type": "bridge|comparison|null",
"search_queries": [...], "entities": [...],
"answer_strategy": "정답형|탐색형|목록형", "reason": "..."}"""


def _parse_plan(raw: str):
    """Planner 출력 검증 (baseline 이식 + list·entities). 실패 시 None."""
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
    entities = out.get("entities")
    if qt not in ("single_hop", "multi_hop", "list"):
        return None
    if not isinstance(sq, list) or not sq or not all(isinstance(s, str) for s in sq):
        return None
    if strategy not in ("정답형", "탐색형", "목록형"):
        return None
    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
        entities = []
    if qt == "single_hop":
        ht = None
    elif qt == "list":
        ht = None
        strategy = "목록형"
    else:
        if ht not in ("bridge", "comparison"):
            return None
        if len(sq) < 2:
            return None
        if ht == "bridge" and "{hop1}" not in sq[1]:
            return None
    if qt != "list" and strategy == "목록형":
        return None
    plan = {"query_type": qt, "hop_type": ht, "search_queries": sq,
            "entities": entities, "reason": str(out.get("reason", ""))}
    return plan, strategy


def planner_node(state: AgentStateV2) -> dict:
    counter = {"llm_call_count": state["llm_call_count"]}
    raw = call_llm(
        counter,
        [{"role": "system", "content": PLANNER_SYSTEM},
         {"role": "user", "content": PLANNER_USER_TMPL.replace("<<QUERY>>", state["query"])}],
        json_mode=True,
    )
    parsed = _parse_plan(raw)
    if parsed is None:  # 파싱 실패 → fallback (SPEC §3과 동일 원칙)
        plan = {"query_type": "single_hop", "hop_type": None,
                "search_queries": [state["query"]], "entities": [],
                "reason": "fallback"}
        strategy = "정답형"
    else:
        plan, strategy = parsed
    return {
        "plan": plan,
        "answer_strategy": strategy,
        "current_hop_query": plan["search_queries"][0],
        "llm_call_count": counter["llm_call_count"],
    }


# 섹션성 어휘 — 명료화 사전 게이트와 검색 노드의 섹션 우선 조회가 공유
# (결함 2 수정 (2): 질의에 섹션 어휘 + entities 문서 해석 시 해당 섹션 청크를
# 거리 무관 포함. clarify.py가 이 목록을 import).
SECTION_WORDS = ("줄거리", "결말", "내용", "평가", "수상", "출연진", "캐스팅",
                 "관객 수", "관객수", "흥행", "리뷰", "명대사")
MAX_SECTION_CHUNKS_PER_DOC = 6  # 섹션 우선 조회 문서당 상한 (Judge·Generator 입력 규모)


# ---------- 검색 (LLM 0회 — 이중 갈래 병합 + 2·3층 키 조회 + 섹션 우선) ----------

def _entity_keys(state: AgentStateV2) -> list:
    """정형 키 후보 = plan.entities(1차) + 중간 답(2차 — hop2의 대상 개체)."""
    ents = list((state.get("plan") or {}).get("entities") or [])
    ents += [a for a in state.get("intermediate_answers", []) if a]
    return list(dict.fromkeys(e.strip() for e in ents if e and e.strip()))


def _structured_lookup(keys: list) -> dict:
    filmo = []
    for e in keys:
        ks, entries = knowledge.filmo_lookup(e)
        if entries:
            filmo.append({"person": e, "keys": ks, "entries": entries})
    return {"infobox": knowledge.infobox_lookup(keys), "filmography": filmo}


def make_search_node(top_k: int):
    def search_node(state: AgentStateV2) -> dict:
        """이중 갈래(전역 + title 필터) 병합, 거리 오름차순 (승인 규칙 2)."""
        q = state["current_hop_query"]
        global_res, _ = db.search_v2(q, k=top_k)
        for r in global_res:
            r["branch"] = "global"
        keys = _entity_keys(state)
        titles = []
        for e in keys:
            titles += knowledge.resolve_titles(e)
        titles = list(dict.fromkeys(titles))
        title_res, _ = db.search_v2_filtered(q, titles, k=top_k)
        merged = {r["id"]: r for r in global_res}
        for r in title_res:
            if r["id"] in merged:
                merged[r["id"]]["branch"] = "both"
            else:
                r["branch"] = "title"
                merged[r["id"]] = r
        results = sorted(merged.values(), key=lambda r: r["distance"])
        top1 = results[0]["distance"] if results else float("inf")

        # 섹션 우선 조회 (결함 2 수정 (2)): 질의에 섹션성 어휘 + entities가
        # 문서로 해석되면 해당 문서의 해당 섹션 청크 전부(분할 포함)를 거리
        # 무관 포함. 문서당 상한 6, 갈래 표시 "section", 병합 목록 끝에 추가
        # (top1_distance는 거리 갈래 기준 유지).
        sec_words = [w for w in SECTION_WORDS if w in q]
        if sec_words and titles:
            per_doc = {}
            for c in db.get_v2_by_titles(titles):
                if any(w in (c.get("section") or "") for w in sec_words):
                    per_doc.setdefault(c["title"], []).append(c)
            for t in sorted(per_doc):
                for c in sorted(per_doc[t], key=lambda x: x["id"])[
                        :MAX_SECTION_CHUNKS_PER_DOC]:
                    if c["id"] in merged:
                        merged[c["id"]]["branch"] += "+section"
                    else:
                        c["branch"] = "section"
                        c["distance"] = None
                        merged[c["id"]] = c
                        results.append(c)
        return {
            "search_results": results,
            "top1_distance": top1,
            "structured_results": _structured_lookup(keys),
        }
    return search_node


# ---------- list 경로 (LLM 0회 — SPEC §8 결정 1, 승인 규칙 5) ----------

def list_lookup_node(state: AgentStateV2) -> dict:
    """색인 조회 전용: 필모(별칭 보정) 우선 → 분류 색인(포함 최단 일치).

    양쪽 빈손이면 exhausted_reason="list_miss" (SPEC §8 결정 1 — list 경로는
    Judge를 타지 않으므로 소진 기록도 이 노드가 담당한다).
    """
    plan = state["plan"]
    keys = _entity_keys(state)
    for e in keys:
        ks, entries = knowledge.filmo_lookup(e)
        if entries:
            return {"list_results": {"kind": "filmography", "key": e,
                                     "matched_keys": ks, "items": entries}}
    cand = keys + list(plan.get("search_queries") or [])[:1]
    cat, titles = knowledge.category_lookup(cand)
    if cat:
        return {"list_results": {"kind": "category", "key": cat, "items": titles}}
    return {"exhausted": True, "exhausted_reason": "list_miss", "list_results": {}}


# Generator용 [정형 정보]에서 제외하는 비정보 필드 (승인 방침 2 — Judge용은
# 전 필드 유지). 근거: v2 인포박스 실측 필드 중 답변 본문에 무의미한
# 시각(그림·사진·포스터·로고)·표기 메타(name 등 중복 키) 필드.
NON_INFO_FIELDS = frozenset({
    "그림", "그림설명", "그림 설명", "그림크기", "그림 크기",
    "사진", "사진설명", "사진 설명", "사진크기", "사진 크기",
    "포스터", "로고", "이미지", "임베드", "name",
})


def _serialize_structured(structured: dict, exclude_fields: frozenset = None) -> str:
    """정형 조회 결과 → 프롬프트 직렬화.

    Judge·hop전환은 전 필드(exclude_fields=None), Generator는
    NON_INFO_FIELDS 제외 — 두 용도 분리 (승인 방침 2)."""
    lines = []
    for rec in (structured or {}).get("infobox", []):
        items = rec.get("fields", {}).items()
        if exclude_fields:
            items = [(k, v) for k, v in items if k not in exclude_fields]
        fields = " | ".join(f"{k}: {v}" for k, v in items)
        lines.append(f"인포박스 [{rec['title']}] ({rec['doc_type']}): {fields}")
    for fl in (structured or {}).get("filmography", []):
        works = ", ".join(e["작품"] + (f"({e['연도']})" if e.get("연도") else "")
                          for e in fl["entries"])
        lines.append(f"필모그래피 [{fl['person']}] {len(fl['entries'])}항목: {works}")
    return "\n".join(lines) or "(없음)"


# ---------- Judge (문지기 이원화 + LLM 0~1회 — 판정 규칙은 baseline 그대로) ----------

JUDGE_SYSTEM = (
    "너는 검색 결과가 질의에 답하기에 충분한지 판정하는 엄격한 심판이다. "
    "반드시 JSON으로만 답하라."
)

JUDGE_USER_TMPL = """검색 결과가 '현재 검색 질의'에 답하기에 충분한지 판정하라.

[현재 검색 질의] <<HOP_QUERY>>

[검색 결과 청크 <<K>>개 — 전문]
<<DOCS>>

아래는 질의의 개체명과 정확히 일치하는 문서에서 가져온 확정 정보다.
[정형 조회 결과]
<<STRUCTURED>>

판정 기준:
1. relevance: 검색 결과와 정형 조회 결과가 현재 검색 질의의 대상·주제와 관련
   있으면 "high", 동떨어져 있으면 "low".
2. sufficiency: 검색 결과 본문과 정형 조회 결과만으로 '현재 검색 질의'에 완결된
   답을 낼 수 있으면 "high", 정보가 부족하면 "low". relevance가 "low"면
   sufficiency는 반드시 "low"다.
   판정 대상은 오직 위의 '현재 검색 질의'다 — 현재 질의에 답할 수 있다면,
   그 너머의 추가 정보(이후 단계에서 찾을 정보)가 없다는 이유로
   "low"로 판정하지 마라.
3. verdict: relevance와 sufficiency가 모두 "high"면 "sufficient", 아니면 "insufficient".
4. reason: 판정 근거 한 문장.
5. missing: insufficient일 때 부족한 '구체적 개체명·속성명'(예: '전계수의 출신 학교')을
   짧은 구로. sufficient면 "".

JSON: {"verdict": "sufficient|insufficient", "relevance": "high|low",
"sufficiency": "high|low", "reason": "...", "missing": "..."}"""


def mark_exhausted(verdict: str, state: AgentStateV2) -> dict:
    """baseline 이식 — exhausted 판정·기록은 Judge 단일 책임 (SPEC §3)."""
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


def _finish_judge(state: AgentStateV2, verdict: str, source: str, relevance: str,
                  sufficiency: str, reason: str, missing: str, llm_count: int,
                  extra: dict = None) -> dict:
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
    if extra:
        upd.update(extra)
    return upd


def judge_node(state: AgentStateV2) -> dict:
    """문지기 이원화 (SPEC §8 결정 6): top1>GATE여도 2·3층 적중이면 hop 유지 +
    미달 1층 청크 병합 제외(search_results 비움). 양층 빈손일 때만 즉시 실패.
    LLM 판정 규칙·파싱 재실패 처리(sufficient 통과)는 baseline 그대로."""
    gate = gate_threshold()
    structured = state.get("structured_results") or {}
    structured_hit = bool(structured.get("infobox") or structured.get("filmography"))
    docs_list = state["search_results"]
    extra = None
    if state["top1_distance"] > gate:
        if not structured_hit:  # 양층 빈손 → 즉시 차단 (baseline 문지기와 동일)
            return _finish_judge(
                state, "insufficient", "gatekeeper", "low", "low",
                f"top1_distance {state['top1_distance']:.3f} > {gate} (문지기 차단)",
                "", state["llm_call_count"],
            )
        docs_list = []
        extra = {"search_results": []}  # 거리 미달 1층 청크 병합 제외
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in docs_list) or "(없음)"
    user = (JUDGE_USER_TMPL
            .replace("<<HOP_QUERY>>", state["current_hop_query"])
            .replace("<<K>>", str(len(docs_list)))
            .replace("<<DOCS>>", docs)
            .replace("<<STRUCTURED>>", _serialize_structured(structured)))
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
    if out is None:  # 재실패 → sufficient 통과 + "parse_fail" 기록
        return _finish_judge(state, "sufficient", "llm_judge", "high", "high",
                             "parse_fail", "", counter["llm_call_count"], extra)
    rel, suf = out["relevance"], out["sufficiency"]
    if rel == "low" and suf == "high":  # 모순 칸 → rel=high 교정 후 sufficient 전진
        rel = "high"
    verdict = "sufficient" if (rel == "high" and suf == "high") else "insufficient"
    missing = out["missing"] if verdict == "insufficient" else ""
    return _finish_judge(state, verdict, "llm_judge", rel, suf,
                         out["reason"], missing, counter["llm_call_count"], extra)


# ---------- hop전환 (2단 폴백: 인포박스 필드 → LLM 추출 — SPEC §8 결정 4) ----------

# 질의 표현 → 인포박스 필드 후보 사전 (승인 규칙 4 보강판, 20항목).
# 근거: v2 인포박스 최빈 필드(영화 정보: 감독·개봉일·장르·각본·제작·배급·원작·
# 음악·시간·관객수 / 인물 정보: 출생일·사망일·직업·활동기간·배우자·소속사·학력)와
# 자연어 질의 표현의 대응. 표현은 구체적(긴) 항목을 앞에 둬 먼저 매칭한다.
# 사전 미적중 시 필드명 문자 등장 규칙(최장 필드명 우선)으로 폴백.
FIELD_SYNONYMS = [
    ("언제 개봉", ["개봉일", "개봉"]),   # "언제 개봉했는가"
    ("상영 시간", ["시간"]),            # "상영 시간은 몇 분"
    ("출신 학", ["학력"]),              # "출신 학교/학과" — 학력 필드로 응답
    ("연출", ["감독"]),                # "연출한 사람" = 감독
    ("감독", ["감독"]),
    ("개봉", ["개봉일", "개봉"]),
    ("장르", ["장르"]),
    ("각본", ["각본"]),
    ("제작사", ["제작사", "제작"]),
    ("배급", ["배급", "배급사"]),
    ("원작", ["원작"]),
    ("음악", ["음악"]),
    ("관객", ["관객수", "관객 수"]),     # "관객 수/관객 동원"
    ("태어", ["출생일", "출생"]),        # "태어난/태어났" 공통 어간
    ("출생", ["출생일", "출생"]),
    ("사망", ["사망일", "사망"]),
    ("직업", ["직업"]),
    ("데뷔", ["데뷔일", "데뷔", "활동 기간", "활동기간"]),
    ("배우자", ["배우자"]),
    ("학력", ["학력"]),
]

MAX_FIELD_VALUE = 30  # 승인 규칙 4: 값 30자 초과는 부적합 → LLM 폴백


def _infobox_field_answer(state: AgentStateV2) -> str:
    """①단계 폴백: 현재 질의 ↔ 인포박스 필드 매칭 (LLM 0회).

    질의에 제목(괄호 제거)이 등장하는 레코드를 우선한다."""
    q = state["current_hop_query"]
    records = (state.get("structured_results") or {}).get("infobox", [])

    def rank(rec):
        base = knowledge._strip_paren(rec["title"])
        return 0 if base and base in q else 1

    for rec in sorted(records, key=rank):
        fields = rec.get("fields", {})
        for expr, cands in FIELD_SYNONYMS:  # ① 동의어 사전
            if expr in q:
                for f in cands:
                    v = str(fields.get(f) or "").strip()
                    if v and len(v) <= MAX_FIELD_VALUE:
                        return v
                break  # 표현 적중·필드 부재 → 문자 등장 규칙으로
        for f in sorted(fields, key=len, reverse=True):  # ② 필드명 문자 등장
            v = str(fields.get(f) or "").strip()
            if f in q and v and len(v) <= MAX_FIELD_VALUE:
                return v
    return ""


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


def _normalize_conversion_ids(ids: list) -> list:
    """정형·list 환산 id → 실존 1층 청크 id 정규화 (Day 8 검수 반영).

    근거: 필모 색인의 섹션 출처는 분할 접미사 없는 "인물::섹션"으로
    기록되므로, 섹션이 길어 분할된 경우 실제 1층 id와 어긋난다(검수 실측:
    "유해진::출연 작품" vs 실청크 ::1~4). 표성 섹션은 1층에서 제외돼 청크가
    아예 없을 수도 있다("봉준호::작품 목록"). 처리: 실존 id는 그대로,
    비실존 id는 접두 일치 분할 청크로 확장, 분할도 없으면 문서 서두로
    폴백. "::서두"는 전 문서 서두 1청크 보장(v2 전수 정책)이라 조회 없이
    통과 — 분류 환산(수백 건)에서 불필요한 DB 조회를 막는다.
    """
    need = [i for i in ids if not i.endswith("::서두")]
    by_title = {}
    if need:
        titles = sorted({i.split("::")[0] for i in need})
        for c in db.get_v2_by_titles(titles):
            by_title.setdefault(c["title"], set()).add(c["id"])
    out = []
    for i in ids:
        if i.endswith("::서두"):
            out.append(i)
            continue
        have = by_title.get(i.split("::")[0], set())
        if i in have:
            out.append(i)
            continue
        splits = sorted(
            (x for x in have if x.startswith(i + "::")),
            key=lambda x: int(x.rsplit("::", 1)[1]))
        out.extend(splits if splits else [i.split("::")[0] + "::서두"])
    return list(dict.fromkeys(out))


def _evidence_chunk_ids(state: AgentStateV2) -> list:
    """승인 규칙 2: 1층 병합 순서 그대로 + 정형 적중 환산 id를 뒤에 추가.

    환산분은 실존 청크로 정규화(검수 반영) — 1층 병합분은 DB에서 나온
    실존 id라 정규화 불요."""
    ids = [r["id"] for r in state["search_results"]]
    conv = _normalize_conversion_ids(
        knowledge.structured_chunk_ids(state.get("structured_results") or {}))
    for i in conv:
        if i not in ids:
            ids.append(i)
    return ids


def hop_transition_node(state: AgentStateV2) -> dict:
    """bridge: 인포박스 필드 직접 채용(LLM 0회) → 실패 시 병합 묶음 LLM 추출.
    comparison: 추출 생략. 공통 쓰기·재실패 처리(extract)는 baseline 그대로."""
    plan = state["plan"]
    chunk_ids = _evidence_chunk_ids(state)
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

    # bridge ①: 인포박스 필드 매칭 — LLM 0회 (SPEC §8 결정 4)
    answer = _infobox_field_answer(state)
    if answer:
        return {
            **common,
            "intermediate_answers": state["intermediate_answers"] + [answer],
            "current_hop_query": plan["search_queries"][1].replace("{hop1}", answer),
        }

    # bridge ②: 병합 묶음(청크 전문 + 정형 직렬화)에서 LLM 추출 (빈 답 → 1회 재시도)
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in state["search_results"])
    structured_txt = _serialize_structured(state.get("structured_results") or {})
    if structured_txt != "(없음)":
        docs = (docs + "\n\n[정형 조회 결과]\n" + structured_txt) if docs else structured_txt
    user = (EXTRACT_USER_TMPL
            .replace("<<HOP_QUERY>>", state["current_hop_query"])
            .replace("<<DOCS>>", docs or "(없음)"))
    counter = {"llm_call_count": state["llm_call_count"]}
    answer, over_limit = "", False
    for attempt in range(2):
        user_txt = user
        if attempt and over_limit:  # 30자 초과 재요청 — 재강조 (승인 방침 1)
            user_txt = (user + "\n\n주의: 직전 답이 30자를 초과했다. 대상의 "
                               "이름/값 하나만 30자 이내 짧은 구로 답하라.")
        raw = call_llm(
            counter,
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": user_txt}],
            json_mode=True,
        )
        try:
            answer = (json.loads(raw).get("answer") or "").strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            answer = ""
        if len(answer) > MAX_FIELD_VALUE:  # 초과=부적합(빈 답 동일) — 절단 금지
            over_limit = True
            answer = ""
        if answer:
            break
    if not answer:  # 재실패: 전환 미발생 — 공통 쓰기 생략
        return {"exhausted": True, "exhausted_reason": "extract",
                "llm_call_count": counter["llm_call_count"]}
    return {
        **common,
        "intermediate_answers": state["intermediate_answers"] + [answer],
        "current_hop_query": plan["search_queries"][1].replace("{hop1}", answer),
        "llm_call_count": counter["llm_call_count"],
    }


# ---------- Rewriter (baseline 3모드 이식 — 변경 금지 항목) ----------

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


def _rewriter_mode(state: AgentStateV2) -> str:
    if state["judge_source"] == "gatekeeper":
        return "C"
    if state["relevance"] == "low":
        return "A"
    return "B"


def rewriter_node(state: AgentStateV2) -> dict:
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
    for attempt in range(2):  # 중복이면 1회 재요청
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
    # 사후 재계획 가드: 모드 B & single_hop & hop0 & 유효 템플릿일 때만
    if (mode == "B" and isinstance(replan, dict)
            and state["plan"].get("query_type") == "single_hop"
            and state["hop_index"] == 0):
        tmpl = str(replan.get("hop2_query_template") or "")
        if replan.get("hop_type") == "bridge" and "{hop1}" in tmpl:
            upd["plan"] = {**state["plan"],
                           "query_type": "multi_hop", "hop_type": "bridge",
                           "search_queries": [new_query, tmpl],
                           "reason": "사후 재계획"}
    return upd


def list_chunk_ids(list_results: dict) -> list:
    """list 조회 적중 → 원본 청크 id 환산 (evidence 박제용, SPEC §8)."""
    if not list_results:
        return []
    ids = []
    if list_results["kind"] == "filmography":
        for e in list_results["items"]:
            src = e.get("출처", "")
            if src.endswith("::인포박스(역인덱스)"):
                ids.append(src.split("::")[0] + "::서두")
            elif src:
                ids.append(src)
    else:  # category: 수록 문서의 서두 청크로 환산
        ids = [f"{t}::서두" for t in list_results["items"]]
    return list(dict.fromkeys(ids))


# ---------- Generator (LLM 1회, 3×2 — 승인 프롬프트) ----------

GENERATOR_SYSTEM = "너는 제공된 문서만 근거로 답하는 한국어 QA 어시스턴트다."

GEN_INSTR = {
    ("정답형", False): (
        "질문에 대한 핵심 답을 첫 문장에서 명확히 제시하라. 이어서 제공 문서의 "
        "줄거리·평가·수상 등 관련 내용을 활용해 근거를 상세히 서술하고, "
        "[정형 정보]의 인포박스가 있으면 부가 정보(감독·개봉일·장르·출연 등)를 "
        "덧붙여라. 이때 인포박스 필드 나열을 복사하지 말고, 필요한 값만 골라 "
        "자연스러운 문장 한 줄로 녹여라. 관련 섹션 재료가 풍부하면 문단 수 제한 "
        "없이 충분히 상세하게 서술하라. 단 제공 문서 밖 서술 금지는 유지."),
    ("탐색형", False): (
        "각 대상의 근거 값을 항목별로 나열한 뒤(예: '- 박하사탕: 1999년 개봉'), "
        "마지막 줄에 비교 결론을 한 문장으로 제시하라. [정형 정보]의 인포박스가 "
        "있으면 각 항목에 관련 부가 정보를 짧게 덧붙이되, 인포박스 필드 나열을 "
        "복사하지 말고 필요한 값만 골라 자연스러운 문장 한 줄로 녹여라."),
    ("목록형", False): (
        "[목록 조회 결과]의 항목을 빠짐없이 나열하라('- 항목' 형식). 각 항목에 조회 "
        "결과나 [정형 정보]에서 확인되는 연도·장르 등을 괄호로 덧붙여라. 연도가 "
        "확인되지 않는 항목은 괄호 표기를 생략하라. 첫 문장에 총 항목 수를 밝히고, "
        "연도가 있는 항목은 연도 내림차순(최신 우선)으로 정렬하라. 항목이 50개를 "
        "초과하면 최신 순 50개만 나열하고 마지막에 '외 N편(총 M편)'으로 요약하라. "
        "첫 문장의 총 개수는 나열 개수가 아니라 전체 항목 수 M을 말하라"
        "(예: 총 108편)."),
    ("정답형", True): (
        "검색이 충분한 근거를 찾지 못한 채 종료됐다. 확정적인 답을 단정하지 마라. "
        "문서에서 확인되는 부분까지만 정리하고, 무엇을 확인할 수 없었는지 명시하라. "
        "[1단계에서 확인한 중간 답]이 있으면 그것까지는 확인된 사실로 출처와 함께 "
        "제시하라."),
    ("탐색형", True): (
        "검색이 충분한 근거를 찾지 못한 채 종료됐다. 확인된 대상의 근거 값만 나열하고, "
        "확인하지 못한 값을 명시하라. 단정적 비교 결론 대신 제한적 결론(또는 '비교 "
        "불가')을 밝혀라. [1단계에서 확인한 중간 답]이 있으면 그것까지는 확인된 "
        "사실로 출처와 함께 제시하라."),
    ("목록형", True): (
        "목록 조회가 결과를 찾지 못한 채 종료됐다. 어떤 키로 조회했고 무엇을 찾지 "
        "못했는지 명시하고, 제공 문서에서 확인되는 관련 정보가 있으면 그것만 제한적으로 "
        "정리하라. 완전한 목록인 것처럼 단정하지 마라."),
}

GEN_COMMON_RULES = """공통 규칙:
- 제공 문서에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 밝혀라.
- 답변 끝에 근거 문서를 (출처: title1, title2) 형식으로 표기하라."""


def fetch_chunks(chunk_ids: list) -> list:
    """앞 hop 문서 재조회 — DB 접근은 core/db 경유 (CLAUDE.md 계층 규칙)."""
    return db.get_v2_by_ids(chunk_ids)


def _serialize_list(list_results: dict) -> str:
    if not list_results:
        return "(조회 실패 — 결과 없음)"
    if list_results["kind"] == "filmography":
        lines = [f"[필모그래피 조회] 키: {list_results['key']} "
                 f"(매칭: {', '.join(list_results.get('matched_keys', []))}) — "
                 f"{len(list_results['items'])}항목"]
        # 연도 내림차순(최신 우선, 연도 미상은 뒤) — Day 7 (b) 승인. 코드 정렬로
        # LLM 정렬 부담을 없애고 50개 상한 절단이 최신작을 유지하게 한다.
        items = sorted(list_results["items"],
                       key=lambda e: -(e.get("연도") or 0))
        for e in items:
            y = f" ({e['연도']})" if e.get("연도") else ""
            lines.append(f"- {e['작품']}{y}")
    else:
        lines = [f"[분류 색인 조회] 분류: {list_results['key']} — "
                 f"{len(list_results['items'])}건"]
        lines += [f"- {t}" for t in list_results["items"]]
    return "\n".join(lines)


def generator_node(state: AgentStateV2) -> dict:
    plan = state["plan"] or {}
    docs_list = list(state["search_results"])
    # 앞 hop 문서 재조회: comparison(baseline 규칙) + exhausted 전 유형 (SPEC §8 결정 5)
    if (plan.get("hop_type") == "comparison" or state["exhausted"]) and state["evidence"]:
        prev_ids = state["evidence"][0]["chunk_ids"][:3]
        have = {r["id"] for r in docs_list}
        docs_list = [c for c in fetch_chunks(prev_ids) if c["id"] not in have] + docs_list
    strategy = (state["answer_strategy"]
                if state["answer_strategy"] in ("정답형", "탐색형", "목록형") else "정답형")
    instr = GEN_INSTR[(strategy, bool(state["exhausted"]))]
    inter = (f"\n[1단계에서 확인한 중간 답] {', '.join(state['intermediate_answers'])}"
             if state["intermediate_answers"] else "")
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in docs_list) or "(문서 없음)"
    structured_txt = _serialize_structured(state.get("structured_results") or {},
                                           exclude_fields=NON_INFO_FIELDS)
    list_block = (f"[목록 조회 결과]\n{_serialize_list(state.get('list_results') or {})}\n\n"
                  if strategy == "목록형" else "")
    user = (f"[질문] {state['query']}{inter}\n\n[문서]\n{docs}\n\n"
            f"[정형 정보]\n{structured_txt}\n\n{list_block}"
            f"[작성 지시]\n{instr}\n\n{GEN_COMMON_RULES}")
    counter = {"llm_call_count": state["llm_call_count"]}
    answer = call_llm(
        counter,
        [{"role": "system", "content": GENERATOR_SYSTEM},
         {"role": "user", "content": user}],
    )
    chunk_ids = _evidence_chunk_ids(state)
    for i in _normalize_conversion_ids(list_chunk_ids(state.get("list_results") or {})):
        if i not in chunk_ids:
            chunk_ids.append(i)
    return {
        "answer": answer,
        "evidence": state["evidence"] + [
            {"hop": state["hop_index"], "chunk_ids": chunk_ids}],
        "sources": state["sources"] + [
            {"hop": state["hop_index"], "titles": chunk_ids}],
        "llm_call_count": counter["llm_call_count"],
    }


def make_nodes(top_k: int = None) -> dict:
    """실제 노드 세트 (그래프 조립용)."""
    k = top_k if top_k is not None else default_top_k()
    return {"planner": planner_node, "search": make_search_node(k),
            "judge": judge_node, "hop_transition": hop_transition_node,
            "rewriter": rewriter_node, "generator": generator_node,
            "list_lookup": list_lookup_node}
