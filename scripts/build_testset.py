"""SPEC §4-2: "청크 선행 → 질문 생성" 테스트셋 3조합 × 50 = 150.

- single×정답형: 무작위 청크 1개 → 그 청크만으로 답할 질문 (정답은 본문에 그대로 등장)
- multi×정답형(bridge): 청크 A의 서두 링크로 연결된 문서 B → 2단 질문
  (hop1 답 = 연결 엔티티 B.title, gold_answer는 B 본문에 그대로 등장)
  B가 수집 분류 밖이면 서두를 추가 수집해 data/external_chunks.jsonl에 저장 (§4-3)
- multi×탐색형(comparison): 같은 범주(같은 하위 분류) 청크 쌍 → 비교 질문
  (근거 값 2개를 hop_answers에 기록, 각각 해당 본문에 그대로 등장)

출력: eval/testset.jsonl {question, combo, hop_type, answers, hop_answers, gold_answer}
      eval/testset_review.csv (검수용, X열에 표시 → --regen으로 해당 행만 재생성)
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collect_wiki  # noqa: E402  (같은 scripts/ 디렉토리)
from core.llm import call_llm  # noqa: E402

N_PER_COMBO = 50
MIN_INTRO_LEN = 200
MAX_ANSWER_LEN = 50
LINK_STOPLIST = {
    "대한민국", "영화", "한국", "한국어", "한국 영화", "서울", "미국", "일본",
    "중국", "영화 감독", "배우", "감독", "넷플릭스", "서울특별시", "드라마",
    # hop2 대상은 고유 개체 한정 — 일반명사 개념 문서 제외 (검수 지시 3)
    "소설", "연극", "망명", "데뷔", "뮤지컬", "각본", "원작", "리메이크",
    "독립 영화", "단편 영화", "장편 영화", "다큐멘터리", "애니메이션",
    "텔레비전 드라마", "웹툰", "영화화", "개봉", "촬영", "시나리오", "흥행",
    # 2차 검수 지시 (5): 언어·일반 개념 전면 금지
    "아이돌", "정치", "패션", "예명", "본명", "가수", "성우", "모델",
    "힙합", "발라드", "록 음악", "사운드트랙", "시트콤", "예능", "방송",
    "텔레비전", "라디오", "유튜브", "케이팝", "트로트", "래퍼", "아나운서",
}


def is_phrase(s: str, max_len: int = 30) -> bool:
    """gold_answer 정규화 검증: 서술형 문장이 아닌 핵심 구 (검수 지시 2·4)."""
    return (
        0 < len(s) <= max_len
        and not s.endswith(".")
        and not re.search(r"(하였다|했다|이다|였다|합니다|입니다|한다)$", s)
    )

TESTSET_PATH = ROOT / "eval" / "testset.jsonl"
REVIEW_CSV_PATH = ROOT / "eval" / "testset_review.csv"
EXTERNAL_PATH = ROOT / "data" / "external_chunks.jsonl"


def base_title(title: str) -> str:
    """"기생충 (2019년 영화)" → "기생충" (질문 포함 여부 검증용)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


PERSON_CAT = re.compile(r"배우|감독|평론가|영화인|작곡가|제작자|각본가|음악가")
FILM_CAT = re.compile(r"영화 작품|4DX 영화|배경으로 한 영화|영화$")


def entity_type(chunk: dict) -> str:
    """청크 개체 유형: "영화" / "인물" / "기타" (comparison 쌍은 같은 유형만)."""
    head = chunk["text"][:150]
    if re.search(r"(영화|다큐멘터리|애니메이션)(이다|다\.|이며|로,|인데)", head) or re.search(
        r"개봉[한된]", head
    ):
        return "영화"
    if re.search(r"\d{4}년[^)]{0,14}~", head) or re.search(
        r"(배우|감독|평론가|작곡가|각본가|제작자|영화인|가수)(이다|다\.|이며|로,)", head
    ):
        return "인물"
    cat = chunk.get("category") or ""
    if PERSON_CAT.search(cat):
        return "인물"
    if FILM_CAT.search(cat):
        return "영화"
    return "기타"


def comparable_axis(a: dict, b: dict):
    """같은 유형 + 비교 근거 값이 두 서두 모두에 실존하는 쌍만 채택.

    반환: 비교 축 힌트 문자열 또는 None(부적합 쌍).
    """
    ta, tb = entity_type(a), entity_type(b)
    if ta != tb or ta == "기타":
        return None
    if ta == "영화":
        if all("개봉" in c["text"] and re.search(r"\d{4}년", c["text"]) for c in (a, b)):
            return "개봉 연도 (두 문서 모두 명시됨)"
        return None
    if all("데뷔" in c["text"] and re.search(r"\d{4}년", c["text"]) for c in (a, b)):
        return "데뷔 연도 (두 문서 모두 명시됨)"
    if all(re.search(r"\d{4}년[^)]{0,14}~", c["text"]) for c in (a, b)):
        return "출생 연도 (두 문서 모두 명시됨)"
    return None


def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def llm_json(system: str, user: str) -> dict:
    state = {"llm_call_count": 0}  # 생성 1건당 독립 카운터
    raw = call_llm(
        state,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True,
    )
    return parse_json(raw)


# ---------- 조합별 생성기 (성공 시 row dict, 실패 시 None) ----------

SINGLE_SYS = (
    "너는 한국어 위키피디아 영화 문서 서두로 RAG 평가용 질문을 만드는 출제자다. "
    "반드시 JSON으로만 답하라."
)


def gen_single(chunk: dict):
    bt = base_title(chunk["title"])
    user = f"""아래 문서 서두만으로 답할 수 있는 자연스러운 한국어 질문 1개와 정답을 만들어라.

[문서 제목] {chunk['title']}
[서두] {chunk['text']}

규칙:
1. 질문에 '{bt}'를 반드시 명시하라 (대명사 금지).
2. answer는 서두 본문에 '토씨 하나 다르지 않게 그대로 등장'하는 핵심 구
   (인명·제목·연도·장소 등, 30자 이내). 서술형 문장·설명문 발췌 금지.
3. 제목 자체를 정답으로 삼지 마라. 감독·개봉일·수상·배우·소재 등 사실 위주.
4. 적절한 질문을 만들기 어려우면 {{"skip": true}}.

JSON: {{"question": "...", "answer": "..."}}"""
    out = llm_json(SINGLE_SYS, user)
    q, a = out.get("question", ""), out.get("answer", "")
    if out.get("skip") or not q or not a:
        return None
    if a not in chunk["text"] or not is_phrase(a) or bt not in q:
        return None
    return {
        "question": q,
        "combo": "single",
        "hop_type": None,
        "answers": [chunk["id"]],
        "hop_answers": {"1": {"title": chunk["id"], "answer": a}},
        "gold_answer": a,
    }


def gen_bridge(a: dict, b: dict):
    bt_a, bt_b = base_title(a["title"]), base_title(b["title"])
    user = f"""2단계(multi-hop) 한국어 질문 1개를 만들어라.

[1단계 문서 A] 제목: {a['title']}
서두: {a['text']}

[2단계 문서 B] 제목: {b['title']}
서두: {b['text']}

[연결 엔티티] {b['title']} (A의 서두에 등장하며 B 문서로 연결됨)

구조: 질문을 읽으면 먼저 A에서 '{bt_b}'를 알아내야 하고(1단계),
그 다음 B의 서두에서 최종 정답을 찾아야 한다(2단계).

규칙:
1. 질문에 '{bt_a}'를 명시하되, '{bt_b}'는 절대 언급하지 마라 (그게 1단계 답이므로).
2. 질문의 최종 답은 반드시 B의 서두에만 있는 사실이어야 한다.
   A의 서두만으로 답이 완결되는 질문은 금지.
3. gold_answer는 질문이 묻는 것과 정확히 일치하는 핵심 구(인명·제목·연도·장소 등,
   30자 이내)로, B의 서두 본문에 '그대로 등장'해야 한다.
   설명문 발췌·서술형 문장 금지.
4. gold_answer로 '{bt_b}' 자체를 쓰지 마라.
5. 연결 엔티티 '{b['title']}'가 인물/영화·드라마 등 작품/기관·단체/장소의
   고유명사가 아니면 {{"skip": true}}. 언어(영어·독일어 등)와 일반 개념
   (아이돌, 정치, 패션, 예명, 스포츠 종목, 기술 용어 등)은 전면 금지.
6. gold_answer는 질문의 의문사가 묻는 대상과 정확히 같은 유형이어야 한다
   (인물을 물으면 인물명, 연도를 물으면 연도).
7. 질문의 최종 답이 연결 엔티티 자신이 되는 형태는 절대 금지
   (예: "A의 감독은 누구인가?"의 답이 바로 B인 경우). 질문은 B를 특정한 뒤
   B의 '다른 속성'(수치·연도·B가 아닌 인물·장소·기관·작품)을 물어야 한다.
   B의 별칭·본명·외국어 제목을 gold_answer로 쓰는 것도 같은 위반이다.
8. 질문에는 대상을 유일하게 특정하는 한정어(연도, 공동 출연자, 장르 등)를
   반드시 포함하라. 한정어 없이 "출연한 드라마 중", "제작한 영화 중"처럼
   여러 후보가 남는 모호한 질문은 금지. 단, 한정어로 gold_answer와 겹치는
   표현을 쓰면 안 된다 (질문에 답이 노출되면 안 됨).
9. 질문의 전제가 논리적으로 모순되면 안 된다 (예: "X가 연출한 영화의 감독은
   누구인가?" 같은 자기모순, A의 사실과 B의 사실을 뒤섞은 잘못된 전제 금지).
10. 자연스러운 2단 질문이 어려운 조합이면 {{"skip": true}}.

JSON: {{"question": "...", "gold_answer": "..."}}"""
    out = llm_json(SINGLE_SYS, user)
    q, g = out.get("question", ""), out.get("gold_answer", "")
    if out.get("skip") or not q or not g:
        return None
    if g not in b["text"] or not is_phrase(g):
        return None
    if g in a["text"]:  # 최종 답이 A에도 있으면 hop2 전용 사실이 아님 (검수 지시 1)
        return None
    if bt_b in q or bt_a not in q:
        return None
    # 질문에 답 노출 금지 (3차 검수): gold 전체 또는 2자 이상 어절이 질문에 등장하면 폐기
    # (영화·드라마 같은 범용 명사 어절은 노출로 보지 않음)
    generic = {"영화", "드라마", "감독", "배우", "작품", "영화제", "방송", "애니메이션"}
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    if norm(g) in norm(q) or any(
        len(tok) >= 2 and tok not in generic and tok in q for tok in g.split()
    ):
        return None
    if not bridge_self_check(q, g, a, b["title"]):  # 자기 검증 (검수 지시 1·6)
        return None
    if not bridge_answer_check(q, g, a, b):  # 답변-대조 검증 (아래 docstring)
        return None
    return {
        "question": q,
        "combo": "bridge",
        "hop_type": "bridge",
        "answers": [a["id"], b["id"]],
        "hop_answers": {
            "1": {"title": a["id"], "answer": b["title"]},
            "2": {"title": b["id"], "answer": g},
        },
        "gold_answer": g,
    }


def bridge_self_check(question: str, gold: str, a: dict, b_title: str) -> bool:
    """생성 후 자기 검증 (검수 지시 1·6). 통과 시 True, 폐기 대상이면 False.

    1) 질문의 최종 답(동등 표현 포함)이 문서 A에 이미 등장하거나 A만으로
       답이 완결되면 폐기.
    2) gold가 질문의 의문사가 묻는 대상과 유형·의미가 일치하지 않으면 폐기.
    3) 질문의 올바른 최종 답이 연결 엔티티(B) 자신이면 사실상 1-hop이므로 폐기
       (gold를 본명·풀네임 등으로 비틀어 B제목-금지 규칙을 우회하는 실패 모드 차단).
    """
    user = f"""세 가지를 엄격히 판정하라.

[질문] {question}
[제시된 정답] {gold}
[문서 A] {a['text']}

1) answerable_from_A: 이 질문이 묻는 최종 답(또는 그와 동등한 표현)이 문서 A에
   이미 등장하는가, 혹은 문서 A만으로 완결된 답을 낼 수 있는가?
   (예: "어떤 작품인가?"의 답인 작품명이 A에 있으면 true)
2) gold_matches_question: 제시된 정답이 질문의 의문사가 묻는 대상과 정확히
   일치하는 유형인가? (인물을 물으면 인물명, 작품을 물으면 작품명,
   연도를 물으면 연도, 직업을 물으면 직업명이어야 true)
3) answer_is_entity: 이 질문에 대한 올바른 최종 답이 '{b_title}'인가?
   (본명·풀네임·다른 표기 등 같은 대상을 가리키는 표현이면 true)
4) unique_target: 질문의 한정 조건(연도·공동 출연자·장르 등)이 묻는 대상을
   후보 하나로 유일하게 특정하기에 충분한가? (여러 작품/인물이 남으면 false)

JSON: {{"answerable_from_A": true/false, "gold_matches_question": true/false,
"answer_is_entity": true/false, "unique_target": true/false}}"""
    out = llm_json(SINGLE_SYS, user)
    return (
        not out.get("answerable_from_A", True)
        and out.get("gold_matches_question", False)
        and not out.get("answer_is_entity", True)
        and out.get("unique_target", False)
    )


def bridge_answer_check(question: str, gold: str, a: dict, b: dict) -> bool:
    """답변-대조 검증: 독립 LLM이 A+B를 보고 실제로 답하게 한 뒤,

    - 그 답이 연결 엔티티(B) 자신이면 폐기 (사실상 1-hop 질문)
    - 그 답이 gold와 포함 관계로 정합하지 않으면 폐기 (질문-gold 불일치)
    yes/no 자기 검증보다 강함: 판정 LLM이 답을 '먼저 생성'하므로 우회가 어렵다.
    """
    user = f"""두 문서를 근거로 질문에 답하라. 답은 핵심 구 하나로만.

[질문] {question}

[문서1] {a['text']}

[문서2] {b['text']}

JSON: {{"answer": "..."}}"""
    out = llm_json(SINGLE_SYS, user)
    ans = (out.get("answer") or "").strip()
    if not ans:
        return False
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    na, ng, nb = norm(ans), norm(gold), norm(base_title(b["title"]))
    if nb and (nb in na or na in nb):
        return False
    return bool(ng) and (ng in na or na in ng)


def gen_comparison(a: dict, b: dict, axis_hint: str = None):
    bt_a, bt_b = base_title(a["title"]), base_title(b["title"])
    hint = (
        f"\n비교 축 제안: {axis_hint}. 이 축을 우선 사용하되, 더 자연스러운 비교 축이"
        " 두 문서 모두에서 확인되면 그것을 써도 된다.\n"
        if axis_hint
        else ""
    )
    user = f"""같은 범주의 두 문서 서두를 비교해야만 답할 수 있는 한국어 비교 질문 1개를 만들어라.
(예: 어느 쪽이 먼저 개봉/데뷔했나 같은 비교·선택형)
{hint}

[문서 1] 제목: {a['title']}
서두: {a['text']}

[문서 2] 제목: {b['title']}
서두: {b['text']}

규칙:
1. 질문에 '{bt_a}'와 '{bt_b}' 두 제목을 모두 명시하라.
2. value_1은 문서 1 본문에, value_2는 문서 2 본문에 각각 '그대로 등장'하는
   비교 근거 값(연도·날짜·이름 등, 40자 이내)이어야 한다.
3. gold_answer는 비교 결론을 핵심 구로만 쓰라: 질문이 고르라는 대상의
   이름·제목 또는 값 (예: "김희라"). 서술형 문장 금지.
4. 공정한 비교가 어려우면 {{"skip": true}}.

JSON: {{"question": "...", "value_1": "...", "value_2": "...", "gold_answer": "..."}}"""
    out = llm_json(SINGLE_SYS, user)
    q, v1, v2, g = (
        out.get("question", ""),
        out.get("value_1", ""),
        out.get("value_2", ""),
        out.get("gold_answer", ""),
    )
    if out.get("skip") or not q or not v1 or not v2 or not g:
        return None
    if v1 not in a["text"] or v2 not in b["text"]:
        return None
    if len(v1) > MAX_ANSWER_LEN or len(v2) > MAX_ANSWER_LEN:
        return None
    if not is_phrase(g, max_len=40):  # 결론도 핵심 구로 (검수 지시 4)
        return None
    if bt_a not in q or bt_b not in q:
        return None
    return {
        "question": q,
        "combo": "comparison",
        "hop_type": "comparison",
        "answers": [a["id"], b["id"]],
        "hop_answers": {
            "1": {"title": a["id"], "answer": v1},
            "2": {"title": b["id"], "answer": v2},
        },
        "gold_answer": g,
    }


# ---------- 재료 공급기 ----------

class Materials:
    def __init__(self, chunks: dict, externals: dict, rng: random.Random):
        self.chunks = chunks
        self.externals = externals  # bridge 외부 문서 캐시 {title: chunk}
        self.rng = rng
        self.used_single = set()
        self.used_pairs = set()
        self.hop1_count = {}  # hop1 문서 재사용 상한 2회 (2차 검수 지시 7)
        self.entity_type_cache = {}  # hop2 유형 LLM 판정 캐시 {title: type}

    def mark_used(self, row: dict):
        if row["combo"] == "single":
            self.used_single.add(row["answers"][0])
        else:
            self.used_pairs.add(tuple(sorted(row["answers"])))
            if row["combo"] == "bridge":
                a_id = row["answers"][0]
                self.hop1_count[a_id] = self.hop1_count.get(a_id, 0) + 1

    def hop2_entity_ok(self, chunk: dict) -> bool:
        """2차 검수 지시 (5): hop2는 인물/작품/기관·단체/장소 고유명사 문서만.

        언어·일반 개념 문서는 LLM 유형 분류로 배제 (제목별 캐시).
        """
        title = chunk["title"]
        if title not in self.entity_type_cache:
            user = f"""아래 위키 문서의 주제 유형을 하나로 분류하라.

[제목] {title}
[서두 일부] {chunk['text'][:400]}

유형: "인물" / "작품"(영화·드라마·소설책 등 특정 작품) / "기관단체" /
"장소" / "언어" / "일반개념"(직업·장르·개념·기술용어 등 고유명사가 아닌 것)

JSON: {{"type": "..."}}"""
            out = llm_json(SINGLE_SYS, user)
            self.entity_type_cache[title] = out.get("type", "일반개념")
        return self.entity_type_cache[title] in {"인물", "작품", "기관단체", "장소"}

    def get_chunk(self, cid: str):
        return self.chunks.get(cid) or self.externals.get(cid)

    def single_candidates(self):
        ids = [c for c in self.chunks if c not in self.used_single]
        self.rng.shuffle(ids)
        return ids

    def resolve_bridge_target(self, link: str):
        """링크 대상 → 청크 (수집분 or 외부 수집). 부적합 시 None."""
        if link in LINK_STOPLIST or len(link) < 2:
            return None
        # 연도·날짜·세기 문서는 bridge 엔티티로 부적합
        if re.fullmatch(r"\d+년(대)?|\d+월 \d+일|\d+세기", link):
            return None
        # 언어 문서(영어·독일어·프랑스어 등) 전면 금지 (2차 검수 지시 5)
        if re.fullmatch(r"[가-힣]{1,5}어", link) or link.endswith("언어"):
            return None
        chunk = self.chunks.get(link) or self.externals.get(link)
        if chunk is None:
            if "동음이의" in link:
                return None
            intro = collect_wiki.fetch_intros([link]).get(link, "")
            if len(intro) < MIN_INTRO_LEN:
                return None
            chunk = {"id": link, "title": link, "text": intro, "links": [], "category": None}
            self.externals[link] = chunk
        if not self.hop2_entity_ok(chunk):
            return None
        return chunk

    def bridge_pairs(self):
        a_ids = list(self.chunks)
        self.rng.shuffle(a_ids)
        for aid in a_ids:
            a = self.chunks[aid]
            links = [l for l in a["links"] if l not in LINK_STOPLIST]
            self.rng.shuffle(links)
            for link in links:
                if self.hop1_count.get(aid, 0) >= 2:  # hop1 최대 2회 (지시 7)
                    break
                if tuple(sorted((aid, link))) in self.used_pairs or link == aid:
                    continue
                yield a, link

    def comparison_pairs(self):
        by_cat = {}
        for c in self.chunks.values():
            by_cat.setdefault(c["category"], []).append(c["id"])
        cats = [k for k, v in by_cat.items() if len(v) >= 2]
        self.rng.shuffle(cats)
        used_in_comp = {t for p in self.used_pairs for t in p}
        for cat in cats:
            ids = [i for i in by_cat[cat] if i not in used_in_comp]
            self.rng.shuffle(ids)
            for i in range(0, len(ids) - 1, 2):
                pair = tuple(sorted((ids[i], ids[i + 1])))
                if pair in self.used_pairs:
                    continue
                a, b = self.chunks[ids[i]], self.chunks[ids[i + 1]]
                # 같은 개체 유형 + 비교 근거 값이 양쪽 서두에 실존하는 쌍만 채택
                axis = comparable_axis(a, b)
                if axis is None:
                    continue
                yield a, b, axis


def fill_combo(combo: str, need: int, mat: Materials, max_attempts: int = 400):
    """combo 유형 row를 need개 생성해 반환."""
    rows, attempts = [], 0
    if combo == "single":
        for cid in mat.single_candidates():
            if len(rows) >= need or attempts >= max_attempts:
                break
            attempts += 1
            row = gen_single(mat.chunks[cid])
            if row:
                rows.append(row)
                mat.mark_used(row)
    elif combo == "bridge":
        for a, link in mat.bridge_pairs():
            if len(rows) >= need or attempts >= max_attempts:
                break
            b = mat.resolve_bridge_target(link)
            if b is None:
                continue
            attempts += 1
            row = gen_bridge(a, b)
            if row:
                rows.append(row)
                mat.mark_used(row)
    elif combo == "comparison":
        for a, b, axis in mat.comparison_pairs():
            if len(rows) >= need or attempts >= max_attempts:
                break
            attempts += 1
            row = gen_comparison(a, b, axis)
            if row:
                rows.append(row)
                mat.mark_used(row)
    print(f"  {combo}: {len(rows)}/{need} 생성 (시도 {attempts}회)")
    return rows


# ---------- 입출력 ----------

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_review_csv(rows: list):
    with open(REVIEW_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["idx", "combo", "question", "gold_answer",
             "hop1_title", "hop1_answer", "hop2_title", "hop2_answer", "X"]
        )
        for i, r in enumerate(rows, 1):
            h1 = r["hop_answers"].get("1", {})
            h2 = r["hop_answers"].get("2", {})
            w.writerow(
                [i, r["combo"], r["question"], r["gold_answer"],
                 h1.get("title", ""), h1.get("answer", ""),
                 h2.get("title", ""), h2.get("answer", ""), ""]
            )


def read_marked_indices() -> list:
    with open(REVIEW_CSV_PATH, encoding="utf-8-sig") as f:
        return [
            int(row["idx"]) for row in csv.DictReader(f)
            if (row.get("X") or "").strip()
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="검수 CSV에서 X 표시된 행만 재생성")
    args = ap.parse_args()

    chunks = {c["id"]: c for c in load_jsonl(ROOT / "data" / "chunks.jsonl")}
    externals = {c["id"]: c for c in load_jsonl(EXTERNAL_PATH)}
    print(f"청크 {len(chunks)}개 로드 (외부 캐시 {len(externals)}개)")

    if args.regen:
        rows = load_jsonl(TESTSET_PATH)
        marked = read_marked_indices()
        if not marked:
            print("X 표시된 행이 없습니다. 종료.")
            return
        print(f"재생성 대상: {sorted(marked)}")
        mat = Materials(chunks, externals, random.Random(4242))
        for i, r in enumerate(rows, 1):
            if i not in marked:
                mat.mark_used(r)  # 유지되는 행의 재료는 재사용 금지
        for i in sorted(marked):
            combo = rows[i - 1]["combo"]
            new = fill_combo(combo, 1, mat)
            if new:
                rows[i - 1] = new[0]
                print(f"  idx {i} ({combo}) 재생성 완료")
            else:
                print(f"  idx {i} ({combo}) 재생성 실패 — 기존 행 유지, 수동 확인 필요")
    else:
        rng = random.Random(42)
        mat = Materials(chunks, externals, rng)
        rows = []
        for combo in ["single", "bridge", "comparison"]:
            print(f"[{combo}] 생성 시작")
            rows += fill_combo(combo, N_PER_COMBO, mat)

    save_jsonl(TESTSET_PATH, rows)
    save_jsonl(EXTERNAL_PATH, list(mat.externals.values()))
    write_review_csv(rows)

    from collections import Counter
    print("\n=== 테스트셋 요약 ===")
    print(Counter(r["combo"] for r in rows))
    print(f"외부(bridge) 추가 수집 문서: {len(mat.externals)}개 → {EXTERNAL_PATH}")
    print(f"검수 CSV: {REVIEW_CSV_PATH} (이상한 행 X열에 표시 → --regen)")


if __name__ == "__main__":
    main()
