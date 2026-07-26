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


def category_lookup(keys: list):
    """list 경로(승인 규칙 5): 후보 문자열을 포함하는 분류명 중 최단 일치 1건.

    포함 실패 시 토큰 폴백(규칙 5 보정 — Day 7 재검증 ③에서 "2019년 개봉 영화"가
    분류명 "2019년 영화"에 부분 문자열로 안 걸리는 어순·수식어 불일치 발견):
    공백 토큰 2개 이상 일치하는 분류명 중 최다 일치·최단 우선.
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
                    scored = sorted(
                        (-sum(1 for t in toks if t in c), len(c), c)
                        for c in cats
                        if sum(1 for t in toks if t in c) >= 2)
                    hits = [c for _, _, c in scored]
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
