"""v2 정형 지식층(2·3층) 로더·개체명 매칭 (SPEC §8, docs/V2_DESIGN.md).

data/v2/* 파일을 최초 1회 로드해 캐시한다. 1층(ChromaDB) 접근은 여기 없음 —
core/db 경유로 검색 노드에서만 (CLAUDE.md 계층 규칙).

별칭 매칭(승인 규칙 3): 정확 일치 → 괄호 한정어 제거("김윤석 (배우)"→"김윤석")
→ 괄호 부착("김윤석"→"김윤석 (…)" 시작, 최대 3건). 문서 제목 해석은 단계 적중
시 중단(한 개체=한 문서), 필모 키는 변형 전체 합집합(같은 인물의 항목이
"김윤석"(역인덱스)과 "김윤석 (배우)"(섹션) 두 키에 갈라져 있으므로 — Day 6
실측: 별칭 분리 1,277명).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "v2"

_infobox = None
_filmo = None
_cat_index = None
_titles = None


def infobox_by_title() -> dict:
    global _infobox
    if _infobox is None:
        _infobox = {}
        with open(DATA / "infobox.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _infobox[r["title"]] = r
    return _infobox


def filmography() -> dict:
    global _filmo
    if _filmo is None:
        with open(DATA / "filmography.json", encoding="utf-8") as f:
            _filmo = json.load(f)
    return _filmo


def category_index() -> dict:
    global _cat_index
    if _cat_index is None:
        with open(DATA / "category_index.json", encoding="utf-8") as f:
            _cat_index = json.load(f)
    return _cat_index


def title_set() -> set:
    """전 문서 제목 사전 — 인포박스 제목 ∪ 분류 색인 수록 제목 (사실상 전 문서)."""
    global _titles
    if _titles is None:
        _titles = set(infobox_by_title())
        for titles in category_index().values():
            _titles.update(titles)
    return _titles


def _strip_paren(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def resolve_titles(entity: str, max_attach: int = 3) -> list:
    """개체명 → 문서 제목 후보 (단계 적중 시 중단, 결정적 순서)."""
    entity = entity.strip()
    if not entity:
        return []
    titles = title_set()
    if entity in titles:
        return [entity]
    base = _strip_paren(entity)
    if base != entity and base in titles:
        return [base]
    pref = f"{entity} ("
    return sorted(t for t in titles if t.startswith(pref))[:max_attach]


def filmo_lookup(entity: str, max_attach: int = 3):
    """개체명 → (매칭 키 목록, 항목 합집합 — 작품명 기준 중복 제거)."""
    entity = entity.strip()
    filmo = filmography()
    keys = []
    if entity in filmo:
        keys.append(entity)
    base = _strip_paren(entity)
    if base != entity and base in filmo:
        keys.append(base)
    pref = f"{base if base else entity} ("
    keys += sorted(k for k in filmo if k.startswith(pref) and k not in keys)[:max_attach]
    seen, entries = set(), []
    for k in keys:
        for e in filmo[k]:
            if e["작품"] not in seen:
                seen.add(e["작품"])
                entries.append(e)
    return keys, entries


def infobox_lookup(entities: list) -> list:
    """개체명 목록 → 인포박스 레코드 목록 (제목 해석 경유, 중복 제거)."""
    out, seen = [], set()
    for e in entities:
        for t in resolve_titles(e):
            if t in seen:
                continue
            rec = infobox_by_title().get(t)
            if rec:
                seen.add(t)
                out.append(rec)
    return out


# 분류 토큰 매칭 표기 동의어 (Day 8 검수 반영, 1쌍 한정): 질문 어휘
# "한국"과 분류 표기 "대한민국(의)"은 문자열이 겹치지 않아('대한민국'은
# 대·한·민·국이라 '한국'이 연속 부분 문자열이 아님) 토큰 일치가 실패한다.
# 실측: "한국 좀비 영화"(토큰 한국·좀비·영화)가 '대한민국의 좀비 영화'(2일치)
# 대신 동점 최단인 '좀비 영화'(외국 영화 1건)로 매칭. 토큰 일치 판정에서만
# 한국↔대한민국을 동치로 본다.
_TOKEN_SYNONYM = {"한국": ("한국", "대한민국"), "대한민국": ("대한민국", "한국")}


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _variant_in(v: str, cat: str) -> bool:
    """토큰(변형) 일치에 단어 경계 적용 (Day 9 승인 A).

    근거: 부분 문자열 일치만으로는 '한국'이 '한국어 영화 작품'(7,997건 포괄
    분류)에 오탐돼, 동점 시 문서 수 우선 규칙과 결합하면 포괄 분류가 항상
    이긴다(Q32·33 잔존 결함). clarify._base_in_question과 동일 원리의 단어
    경계 — 일치 위치 뒤가 한글 음절이면 불일치, 조사 '의'만 허용
    ('한국의 영화'·'대한민국의 좀비 영화'는 유지, '한국어…'는 차단)."""
    start = cat.find(v)
    while start != -1:
        end = start + len(v)
        after = cat[end] if end < len(cat) else ""
        if (not after) or (not _is_hangul(after)) or after == "의":
            return True
        start = cat.find(v, start + 1)
    return False


def _tok_in(tok: str, cat: str) -> bool:
    return any(_variant_in(v, cat) for v in _TOKEN_SYNONYM.get(tok, (tok,)))


def category_lookup(keys: list):
    """list 경로(승인 규칙 5): 후보 문자열을 포함하는 분류명 중 최단 일치 1건.

    포함 실패 시 토큰 폴백(규칙 5 보정 — Day 7 재검증 ③에서 "2019년 개봉 영화"가
    분류명 "2019년 영화"에 부분 문자열로 안 걸리는 어순·수식어 불일치 발견):
    공백 토큰 2개 이상 일치하는 분류명을 정렬 키 (일치 토큰 수, 분류명
    커버리지, 수록 문서 수, 이름)로 선별. 토큰 일치는 한국↔대한민국 표기
    동의어(_tok_in, Day 8) + 단어 경계(_variant_in, Day 9 A) 적용.

    Day 9 동점 처리 3단 진화(승인 A2) — 근거:
    ① 동점 최단 우선: 2건짜리 잡분류 '한국의 영화'가 '2019년 영화' 217건을
       이김(본평가 Q32·33 실패) → 수록 문서 수 우선 도입.
    ② 문서 수 우선 단독: 부분 문자열 오탐('한국'⊂'한국어')으로 포괄 분류
       '한국어 영화 작품' 7,997건이 이김 → 단어 경계 규칙(_variant_in).
    ③ 경계 적용 후에도 정당한 일치인 '대한민국의 영화 작품' 6,768건이 동점
       문서 수로 이김(연도 토큰 무변별) → 분류명 커버리지(분류명 토큰 중
       질문 토큰 일치 비율) 축을 문서 수 앞에 삽입 — '작품'처럼 질문에 없는
       토큰을 가진 분류를 강등. 5케이스 실측 전부 의도 결과, Day 7 케이스
       무회귀 확인.
    """
    cats = category_index()
    best = None
    for key in keys:
        key = (key or "").strip()
        if not key:
            continue
        if key in cats:
            hits = [key]
        else:
            hits = sorted((c for c in cats if key in c), key=len)
            if not hits:
                toks = [t for t in key.split() if t]
                if len(toks) >= 2:
                    def _cover(c):
                        nts = [t for t in c.split() if t]
                        cov = sum(1 for nt in nts
                                  if any(_tok_in(q, nt) for q in toks))
                        return cov / len(nts) if nts else 0.0

                    scored = sorted(
                        (-sum(1 for t in toks if _tok_in(t, c)),
                         -_cover(c), -len(cats[c]), c)
                        for c in cats
                        if sum(1 for t in toks if _tok_in(t, c)) >= 2)
                    hits = [c for *_, c in scored]
        if hits and (best is None or len(hits[0]) < len(best)):
            best = hits[0]
    if best is None:
        return None, []
    return best, cats[best]


def structured_chunk_ids(structured: dict) -> list:
    """정형 적중 → 원본 청크 id 환산 (SPEC §8: evidence 박제용).

    인포박스: source_chunk(title::서두). 필모: 섹션 출처는 청크 id 그대로,
    역인덱스 출처("영화::인포박스(역인덱스)")는 해당 영화의 서두 청크로 환산.
    """
    ids = []
    for rec in (structured or {}).get("infobox", []):
        ids.append(rec.get("source_chunk") or f"{rec['title']}::서두")
    for fl in (structured or {}).get("filmography", []):
        for e in fl.get("entries", []):
            src = e.get("출처", "")
            if src.endswith("::인포박스(역인덱스)"):
                ids.append(src.split("::")[0] + "::서두")
            elif src:
                ids.append(src)
    return list(dict.fromkeys(ids))
