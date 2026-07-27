"""명료화 노드 (SPEC §8 확정 설계 + docs/재질문_워크플로우_최종설계서.md).

파이프라인: DB 메타 조회 → 사전 게이트 → 의도 샘플링(S=10, 온도 0.5) →
임베딩 동치(cos≥0.85)+DFS 클러스터 → 엔트로피 u(x) → OR 판정(u>τ ∨ DB≥2) →
선택지 조립(완전 질문 문장, 환각 검증, 1개면 취소) → clarification 기록.

- LLM은 core.llm.call_llm_clarify 경유 — 별도 카운터, 상한 13 (승인 C·D).
- 임베딩은 core.db.embed_texts 재사용 (text-embedding-3-small).
- 무상태: 선택지는 재구성 완료된 완전 질문 — 클릭 = 새 실행.
- τ(TAU)는 명확 질문 실측 후 사용자 확정으로 설정 (그 전에는 tau 명시 필수).
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.llm import call_llm_clarify  # noqa: E402

from agents.v2 import knowledge  # noqa: E402
from agents.v2.nodes import FIELD_SYNONYMS, SECTION_WORDS  # noqa: E402

# τ·동치 임계 (사용자 확정, 소표본 보정 한계 있음 — 명확 18·모호 4 문항 실측):
# - τ=1.05: 명확 18문항 u 실측 max 1.0297 + 여유 0.02.
# - SIM_THRESHOLD=0.85 유지: 0.75~0.80 완화 탐색 결과 모든 임계에서 분리
#   간격(모호 min − 명확 max)이 음수 — 임계를 낮추면 진짜 갈린 해석까지
#   합쳐져 신호가 소멸(0.85가 유일하게 다축 해석형 '학생 영화'를 분리).
# - 한계(후퇴 기록): 단일 해석·광범위형(추천/평가 등 — 해석은 하나, 범위만
#   넓음)은 해석-갈림(u) 신호의 원리상 미포착. 다축 해석형(예: "학생 영화",
#   실측 u=1.221)은 현 설정에서 포착 — SPEC §8 한계 문구 참조.
TAU = 1.05
SIM_THRESHOLD = 0.85
N_SAMPLES = 10        # 의도 샘플 수 S (설계서 초기값)
SAMPLE_TEMPERATURE = 0.5

FREE_INPUT_HINT = "다른 의도라면 질문을 직접 구체적으로 적어 주세요."

# 사전 게이트 속성 어휘 (승인 B): DB 단일 매칭 + 아래 속성 명시 → 즉시 통과.
# 구성 = 인포박스 질의 표현(FIELD_SYNONYMS의 표현부, hop전환과 동일 어휘) +
# 섹션성 어휘(nodes.SECTION_WORDS — 검색 노드의 섹션 우선 조회와 공유).
# 근거: 대상이 하나로 특정되고 묻는 속성까지 명시된 질문은 해석이 갈릴 여지가
# 없다 — 샘플링 비용 0으로 통과.
ATTRIBUTE_WORDS = tuple(expr for expr, _ in FIELD_SYNONYMS) + SECTION_WORDS

# 단일 음절 조사 (승인 A 보강): 2자 제목 base의 단어 경계 판정용.
# 근거: 《얼굴》(1999/2007) 같은 2자 동명작을 잡되, 일반 명사 오탐은
# "매칭 ≥2" OR 조건이 완충한다(오탐 1건 매칭은 게이트·판정에 무해).
_JOSA = set("은는이가을를의도와과만로에께랑나야")

# 지시어 규칙 게이트 (통제 검증 ② 해법 (가), 사용자 승인): 지시 대상 미해결
# 유형("그 영화 감독이 누구야?")은 해석이 하나뿐이라 의도 샘플링(u)으로 잡히지
# 않고 DB 매칭도 0이다 → 지시어 존재 & 매칭 0이면 샘플링 전에 발동(LLM 0회),
# 접지할 후보가 없으므로 choices=[]로 자유 입력만 유도. 목록은 소규모 유지 —
# 확장은 운영 관찰 후.
DEICTIC_TERMS = ("그 영화", "그 작품", "그 사람", "그 배우", "그 감독")


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


# ---------- 0단계: DB 메타 조회 (승인 A 보강판) ----------

_base_map = None


def _get_base_map() -> dict:
    """제목 base(괄호 한정어 제거) → [문서 제목들]. 2자 이상 base만 수록."""
    global _base_map
    if _base_map is None:
        _base_map = {}
        for t in knowledge.title_set():
            base = knowledge._strip_paren(t)
            if len(base) >= 2:
                _base_map.setdefault(base, []).append(t)
    return _base_map


def _base_in_question(base: str, q: str) -> bool:
    """base가 질문에 등장하는가. 2자 base는 단어 경계 확인(승인 A 보강):
    앞은 한글 음절이 아니어야 하고, 뒤는 한글 음절이 아니거나 조사여야 매칭."""
    if len(base) >= 3:
        return base in q
    start = q.find(base)
    while start != -1:
        end = start + len(base)
        before_ok = start == 0 or not _is_hangul(q[start - 1])
        after = q[end] if end < len(q) else ""
        after_ok = (not after) or (not _is_hangul(after)) or (after in _JOSA)
        if before_ok and after_ok:
            return True
        start = q.find(base, start + 1)
    return False


def db_match(question: str, cap: int = 5) -> list:
    """질문 내 제목 base 스캔 → 동명작·접두 확장 포함 매칭 문서 상위 cap.

    반환: [{"title", "base", "meta"}] — meta는 인포박스 연도·감독 등 표기.
    """
    bmap = _get_base_map()
    matched = [b for b in bmap if _base_in_question(b, question)]
    # 최장 일치 우선: 다른 매칭 base의 부분 문자열인 base 제거
    matched = [b for b in matched
               if not any(b != o and b in o for o in matched)]
    matched.sort(key=lambda b: (-len(b), b))
    docs, seen = [], set()
    for b in matched:
        expand = list(bmap[b])  # 동명작 (같은 base, 괄호 구분)
        for ob in sorted(bmap):  # 접두 확장: "베테랑" → "베테랑2"
            if ob != b and ob.startswith(b):
                expand += bmap[ob]
        for t in expand:
            if t not in seen:
                seen.add(t)
                docs.append((b, t))
            if len(docs) >= cap:
                break
        if len(docs) >= cap:
            break
    out = []
    for base, t in docs:
        rec = knowledge.infobox_by_title().get(t)
        year, who, rep, dt = _doc_year(t), "", "", (rec or {}).get("doc_type")
        if rec:
            f = rec.get("fields", {})
            who = str(f.get("감독") or f.get("직업") or "")[:30]
            rep = str(f.get("대표작") or "")[:30]
        # 연도 의미 명시: 인물은 "N년생"(출생), 그 외는 "N년"(개봉) — 표기가
        # 불명확하면 P2가 "N년 출연한"처럼 사실을 왜곡함 (통제 검증 ⑦ 실측)
        year_txt = ""
        if year:
            year_txt = f"{year}년생" if dt == "person" else f"{year}년"
        meta = [x for x in (year_txt, who, (f"대표작 {rep}" if rep else "")) if x]
        out.append({"title": t, "base": base, "year": year, "doc_type": dt,
                    "kind": {"person": "인물", "movie": "영화"}.get(dt, "문서"),
                    "meta": " · ".join(meta)})
    return out


_year_map = None


def _doc_year(title: str):
    """문서 연도: 인포박스(개봉일·개봉·출생일) → 분류 색인("YYYY년 영화"/
    "YYYY년 출생") 폴백. 동명작 구별의 표준 축."""
    rec = knowledge.infobox_by_title().get(title)
    if rec:
        f = rec.get("fields", {})
        ym = re.search(r"(19|20)\d{2}",
                       str(f.get("개봉일") or f.get("개봉") or f.get("출생일") or ""))
        if ym:
            return ym.group()
    global _year_map
    if _year_map is None:
        _year_map = {}
        for cat, titles in knowledge.category_index().items():
            cm = re.match(r"^((?:19|20)\d{2})년 (?:영화|출생)$", cat)
            if cm:
                for t in titles:
                    _year_map.setdefault(t, cm.group(1))
    return _year_map.get(title)


def reduce_by_year(question: str, matches: list) -> list:
    """(승인 해법 i) 질문에 연도가 명시되고 매칭 문서의 연도와 일치하면 그
    문서(들)로 축소 — 재구성 질문(P2가 연도 표기 강제)의 동명작 재발동 탈출구.

    잔여 한계(사용자 확인): 인물 동명이인은 재구성 질문에 연도(출생년)가
    실리지 않는 맥락이 많아 축소가 안 되고 재발동할 수 있다 — 해소는 향후
    과제(SPEC §8 한계 문구에 기록)."""
    years = {m.group() for m in re.finditer(r"(?:19|20)\d{2}", question)}
    if not years:
        return matches
    hit = [m for m in matches if m.get("year") in years]
    return hit if hit else matches


def has_duplicate(matches: list) -> bool:
    """조건 2(동명작): '같은 키워드(base)'로 매칭된 문서가 2건 이상인가.

    질문 전체 매칭 총수가 아니라 키워드별 카운트다 — "A와 B 중 누가…"처럼
    서로 다른 두 개체가 각 1건씩 매칭되는 명확 질문을 오발동시키지 않기 위함
    (τ 실측 중 발견: 총수 기준은 명확 18문항 중 11건 오발동). 설계서 0단계의
    의도("'베테랑'이 몇 건 매칭되는가")와 일치."""
    from collections import Counter
    per_base = Counter(m["base"] for m in matches)
    return any(v >= 2 for v in per_base.values())


def _has_attribute(question: str, matches: list = None) -> bool:
    """속성 어휘 검사. 매칭된 제목 base는 먼저 제거 — "극한직업"의 '직업'처럼
    제목 내부 문자열이 속성으로 오판되는 것을 방지."""
    q = question
    for m in matches or []:
        q = q.replace(m["base"], " ")
    return any(w in q for w in ATTRIBUTE_WORDS)


# ---------- 1단계: 의도 샘플링 (P1 승인 프롬프트) ----------

SAMPLE_SYSTEM = "너는 질문의 의도를 추론하는 분석가다."

SAMPLE_USER_TMPL = """아래 질문을 실제로 입력한 서로 다른 사용자 한 명을 상상하라. 그 사용자가
정확히 무엇을 원했는지, 구체적인 의도를 한국어 한 문장으로 써라.

[질문] <<QUERY>>

규칙:
- 의도 문장 하나만 출력하라. 설명·번호·따옴표 금지.
- 질문에 없는 구체 사실(작품명·연도 등)을 지어내지 말고, 해석의 방향만
  구체화하라."""


def sample_intents(question: str, counter: dict) -> list:
    outs = call_llm_clarify(
        counter,
        [{"role": "system", "content": SAMPLE_SYSTEM},
         {"role": "user", "content": SAMPLE_USER_TMPL.replace("<<QUERY>>", question)}],
        temperature=SAMPLE_TEMPERATURE,
        n=N_SAMPLES,
    )
    return [(o or "").strip().splitlines()[0].strip() for o in outs if (o or "").strip()]


# ---------- 2~4단계: 동치 그래프 → DFS 클러스터 → 엔트로피 ----------

def cluster_intents(intents: list):
    """임베딩 cos≥SIM_THRESHOLD 간선 → DFS 연결 요소. 반환: (클러스터 목록,
    대표 목록 — 임베딩 중심 최근접 샘플, 승인 I)."""
    embs = db.embed_texts(intents)
    import numpy as np
    E = np.array(embs, dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E.T
    n = len(intents)
    adj = [[j for j in range(n) if j != i and sim[i][j] >= SIM_THRESHOLD]
           for i in range(n)]
    seen, clusters = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:  # DFS
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        clusters.append(sorted(comp))
    reps = []
    for comp in clusters:
        center = E[comp].mean(axis=0)
        center /= max(1e-9, float((center ** 2).sum()) ** 0.5)
        best = max(comp, key=lambda k: float(E[k] @ center))
        reps.append(intents[best])
    return clusters, reps


def entropy(sizes: list) -> float:
    """클러스터 크기 비율 분포의 자연로그 엔트로피 (승인 E)."""
    total = sum(sizes)
    if total == 0:
        return 0.0
    return -sum((s / total) * math.log(s / total) for s in sizes if s)


# ---------- 6단계: 선택지 조립 (P2 승인 프롬프트) ----------

ASSEMBLE_SYSTEM = "너는 모호한 질문을 명확한 선택지로 바꾸는 도우미다. 반드시 JSON으로만 답하라."

ASSEMBLE_USER_TMPL = """사용자의 질문이 여러 해석으로 갈렸다. 아래 재료로 선택지형 재질문을 만들어라.

[원 질문] <<QUERY>>

[발동 사유] <<REASON>>  (db_duplicate=동명 문서 존재, intent_split=해석 갈림)

[해석 후보 — 의도 클러스터 대표]
<<INTENTS>>

[DB 동명 매칭 항목 — 실존 작품/인물. 연도·감독 등은 이 표기를 그대로 쓸 것]
<<DB_ITEMS>>

규칙:
1. category: '선택지들' 자체가 무엇인지를 묶는 한 단어 — 질문의 주제가
   아니라 선택지의 종류다(선택지가 인물들이면 "인물", 작품들이면 "작품",
   해석 기준들이면 "기준"). 묶기 어려우면 "것".
2. 소스 우선순위: 발동 사유가 db_duplicate면 선택지의 축은 '매칭 2건
   이상인 모호 키워드'의 각 매칭 항목이다 — 항목 하나가 선택지 하나가 된다.
   구별 표기는 항목의 유형([인물]/[영화])과 인포박스 정보(직업·출생·대표작·
   감독·연도)를 쓴다. 원 질문이 비교·목록 등 복합 구조면 그 구조를 보존한
   채 모호한 키워드 자리만 각 항목으로 치환한 완전 질문을 만들어라
   (예: "기생충 (영화, 2019)과 군체 중 어느 것이 먼저 개봉했나요?" /
   "기생충 VR(2021)과 군체 중 어느 것이 먼저 개봉했나요?").
   '단일 매칭' 표시가 붙은 항목(모호하지 않은 개체)은 선택지 축으로 삼지
   마라. 의도 클러스터는 보조 참고만 하라. intent_split 발동이면 기존대로
   클러스터 대표가 선택지 축이다.
3. choices의 각 question은 [원 질문]의 의도를 그 해석으로 확정했을 때의
   "재구성이 완료된 완전한 질문 한 문장"이다 — 사용자가 그대로 다시
   묻는다고 생각하고 자연스럽게 써라. 각 question은 그 질문 하나만 읽어도
   대상이 특정되도록 써라 — '그 영화', '그 사람' 같은 대명사·지시어 사용
   금지.
4. DB 항목 출신 선택지는 제공된 연도·감독 표기를 그대로 사용해 서로
   구별되게 하라. 제공되지 않은 작품·인물·연도를 지어내지 마라.
5. 같은 뜻의 선택지는 하나로 합쳐라. label은 핵심 구분점을 담은 짧은
   구다(예: "2015년 베테랑").
6. 선택지는 2~5개.
7. 괄호가 붙은 문서 제목은 자연스러운 표현으로 풀어 써라
   (예: 김성원 (희극인) → 희극인 김성원).
8. 각 question은 의문문으로 끝나야 한다(예: "~은 무엇인가요?").

예시:
- 원 질문 "김철수 배우 프로필 알려줘",
  DB 항목 [김철수 (희극인) (1970년 · 희극 배우), 김철수 (배우) (1955년 · 영화배우)] →
  {"category": "인물", "choices": [
   {"label": "1970년생 희극인 김철수",
    "question": "1970년생 희극인 김철수의 프로필을 알려 주시겠어요?"},
   {"label": "1955년생 영화배우 김철수",
    "question": "1955년생 영화배우 김철수의 프로필을 알려 주시겠어요?"}]}

JSON: {"category": "...", "choices": [{"label": "...", "question": "..."}]}"""


def _parse_assembled(raw: str):
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    cat = out.get("category")
    ch = out.get("choices")
    if not isinstance(cat, str) or not isinstance(ch, list):
        return None
    choices = []
    for c in ch:
        if (isinstance(c, dict) and isinstance(c.get("label"), str)
                and isinstance(c.get("question"), str)
                and c["label"].strip() and c["question"].strip()):
            choices.append({"label": c["label"].strip(),
                            "question": c["question"].strip()})
    if not choices:
        return None
    return cat.strip() or "것", choices


def assemble_choices(question: str, reps: list, matches: list, counter: dict,
                     reason: list):
    intents_txt = "\n".join(f"- {r}" for r in reps) or "(없음)"
    # 모호/단일 키워드 구분 표기 (결함 3 수정): 선택지 축은 동명 2건 이상
    # 키워드의 항목이어야 하며, 단일 매칭 개체가 축이 되면 모호 키워드가
    # 미해결로 남아 재실행에서 재발동 루프가 생긴다.
    from collections import Counter
    per_base = Counter(m["base"] for m in matches)
    lines = []
    for m in matches:
        n = per_base[m["base"]]
        tag = (f"모호 키워드 \"{m['base']}\" — 동명 {n}건" if n >= 2
               else "단일 매칭 — 선택지 축 금지")
        lines.append(f"- ({tag}) {m['title']} [{m.get('kind', '문서')}]"
                     + (f" ({m['meta']})" if m["meta"] else ""))
    db_txt = "\n".join(lines) or "(없음)"
    user = (ASSEMBLE_USER_TMPL
            .replace("<<QUERY>>", question)
            .replace("<<REASON>>", ", ".join(reason))
            .replace("<<INTENTS>>", intents_txt)
            .replace("<<DB_ITEMS>>", db_txt))
    for _ in range(2):  # 파싱 실패 → 1회 재시도
        raw = call_llm_clarify(
            counter,
            [{"role": "system", "content": ASSEMBLE_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True,
        )[0]
        parsed = _parse_assembled(raw)
        if parsed is not None:
            return parsed
    return None


def _grounded(choice_q: str, question: str) -> bool:
    """환각 검증 (승인 F): 원 질문·DB 제목 사전에 없는 새 작품 표기 도입 시 제거.

    검사 대상: 《…》 표기와 '…숫자' 시리즈형 토큰. 해석형 선택지는 통과.
    """
    bmap = _get_base_map()
    for m in re.findall(r"《([^》]+)》", choice_q):
        name = m.strip()
        if name and name not in question and name not in bmap:
            return False
    for tok in re.findall(r"[가-힣A-Za-z]+\d+", choice_q):
        if tok not in question and tok not in bmap:
            return False
    return True


# ---------- 통합: 명료화 판정 (그래프 진입 전 훅) ----------

def clarify_question(question: str, tau: float = None) -> dict:
    """반환: None=그대로 통과 / {"proceed_query": q}=재질문 취소·해석 확정 통과 /
    {"clarification": {...}, "clarify_calls": n}=재질문 조기 종료."""
    tau = TAU if tau is None else tau
    assert tau is not None, "τ 미확정 — 명확 질문 실측·사용자 확정 후 설정 (Day 7 (3))"
    counter = {"clarify_call_count": 0}

    matches = reduce_by_year(question, db_match(question))  # 연도 축소 (해법 i)
    if len(matches) == 1 and _has_attribute(question, matches):  # 사전 게이트 (LLM 0회)
        return None

    # 지시어 규칙 게이트 (해법 (가), 샘플링 전 — LLM 0회): 대상 미해결 유형은
    # 접지 후보가 없어 choices=[] — UI는 자유 입력 안내만 렌더링 (SPEC §6 메모)
    if not matches and any(t in question for t in DEICTIC_TERMS):
        clar = {
            "needed": True,
            "reason": ["unresolved_reference"],
            "u": None,
            "tau": tau,
            "db_matches": [],
            "category": "대상",
            "choices": [],
            "free_input_hint": FREE_INPUT_HINT,
        }
        return {"clarification": clar, "clarify_calls": 0}

    intents = sample_intents(question, counter)
    if not intents:
        return None  # 샘플 전무 → 보수적 통과
    clusters, reps = cluster_intents(intents)
    u = entropy([len(c) for c in clusters])

    reason = []
    if u > tau:
        reason.append("intent_split")
    if has_duplicate(matches):
        reason.append("db_duplicate")
    if not reason:
        return None

    assembled = assemble_choices(question, reps, matches, counter, reason)
    if assembled is None:  # 조립 파싱 재실패 → 보수적 통과 (오발동 방지)
        return None
    category, choices = assembled
    choices = [c for c in choices if _grounded(c["question"], question)]
    # category 후처리 (사용자 승인 (a)): P2가 원 질문 주제에 끌려 인물 선택지에
    # "작품"을 붙이는 사례(통제 검증 ⑦) 교정 — 선택지 재료인 DB 매칭 문서의
    # doc_type이 전원 person이면 "인물", 전원 movie면 "작품" 강제. 혼합·판별
    # 불가면 LLM 출력 유지.
    if "db_duplicate" in reason and matches:
        types = {(knowledge.infobox_by_title().get(m["title"]) or {}).get("doc_type")
                 for m in matches}
        if types == {"person"}:
            category = "인물"
        elif types == {"movie"}:
            category = "작품"
    if len(choices) == 1:  # 검증 후 1개 → 재질문 취소, 그 해석으로 계속
        return {"proceed_query": choices[0]["question"],
                "clarify_calls": counter["clarify_call_count"]}
    if not choices:
        return None

    clar = {
        "needed": True,
        "reason": reason,
        "u": round(u, 4),
        "tau": tau,
        "db_matches": [m["title"] for m in matches],
        "category": category,
        "choices": choices,
        "free_input_hint": FREE_INPUT_HINT,
    }
    return {"clarification": clar,
            "clarify_calls": counter["clarify_call_count"]}
