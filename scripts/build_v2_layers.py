"""v2 3층 데이터 구축 (V2_DESIGN.md): 스냅샷 → ①섹션 청크 ②인포박스 레코드 ③색인.

기존 코드·./db 무변경. 입력: data/v2/pages_snapshot.jsonl
출력: data/v2/{chunks.jsonl, infobox.jsonl, filmography.json, category_index.json,
      build_stats.json, sample_20.txt}

구현 재량 수치(보고 대상): 청크 허용 대역 100~1,200자(초과 시 문단 경계 분할
title::섹션::n, 미달 잔섹션은 직전 섹션에 병합 후에도 미달이면 제외),
참조성 섹션(각주·외부 링크 등) 제외, 서두 청크는 title::서두.

전수 정책(수집 정책 수정 반영): 2·3층은 전 문서에서 생성. 1층은 정리 후
서두<100자면 인포박스 핵심 필드(movie: 감독·개봉일·장르 / person: 출생·직업·
활동 기간)를 덧붙인 합성 서두(synthetic=True)로 만들고, 그래도 0청크인 문서는
제목 청크로 최소 1청크를 보장한다.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2"
MIN_CHUNK, MAX_CHUNK = 100, 1200
REF_SECTIONS = {"각주", "외부 링크", "같이 보기", "참고 문헌", "출처", "주해",
                "주석", "참고 자료", "관련 항목", "미주"}
FILMO_SECTION = re.compile(r"출연|작품|필모그래피|연기|드라마|영화|연극|뮤지컬|방송|공연")
FILMO_EXCLUDE = re.compile(r"외\s?활동")  # "연기 외 활동" 등 비출연 섹션
CAST_FIELDS = ("출연", "주연", "조연", "출연진", "출연자")
DATE_LINK = re.compile(r"^\d{4}(년(\s?\d{1,2}월(\s?\d{1,2}일)?)?)?$|^\d{1,2}월(\s?\d{1,2}일)?$|^\d{1,2}일$")
NONWORK_LINKS = {"KBS", "KBS1", "KBS2", "KBS 1TV", "KBS 2TV", "MBC", "MBC TV",
                 "SBS", "SBS TV", "EBS", "OCN", "tvN", "JTBC", "TV조선", "채널A",
                 "MBN", "넷플릭스", "Netflix", "쿠팡플레이", "티빙", "웨이브",
                 "디즈니+", "대한민국", "미국", "일본",
                 "문화방송", "한국방송공사", "서울방송", "한국교육방송공사",
                 "SBS (대한민국의 방송사)"}


# ---------- 위키텍스트 정리 ----------

def _strip_balanced(text: str, open_s: str, close_s: str, keep=None):
    """중첩 균형 블록 제거. keep(block)->str 지정 시 대체 텍스트 삽입."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith(open_s, i):
            depth, j = 1, i + len(open_s)
            while j < n and depth:
                if text.startswith(open_s, j):
                    depth += 1
                    j += len(open_s)
                elif text.startswith(close_s, j):
                    depth -= 1
                    j += len(close_s)
                else:
                    j += 1
            block = text[i:j]
            if keep:
                out.append(keep(block))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _file_link_keep(block: str) -> str:
    """[[파일:...|...|캡션]] → 캡션 텍스트만 보존 (이미지 설명 텍스트 유지)."""
    inner = block[2:-2]
    if not re.match(r"(파일|File|그림|Image):", inner, re.I):
        return block  # 파일 링크가 아니면 그대로 (후속 링크 처리로 넘어감)
    parts = re.split(r"\|(?![^\[]*\]\])", inner)
    caption = parts[-1].strip() if len(parts) > 2 else ""
    return f" {caption} " if len(caption) > 10 and "=" not in caption else " "


def clean_wikitext(w: str) -> str:
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)
    w = re.sub(r"<ref[^>/]*/>", "", w)
    w = re.sub(r"<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    w = _strip_balanced(w, "[[", "]]", keep=_file_link_keep)  # 파일 링크 캡션화
    w = _strip_balanced(w, "{{", "}}", keep=lambda b: " ")     # 템플릿 제거
    # 표 → 행 단위 텍스트
    lines, in_table = [], False
    for line in w.splitlines():
        s = line.strip()
        if s.startswith("{|"):
            in_table = True
            continue
        if in_table:
            if s.startswith("|}"):
                in_table = False
                continue
            if s.startswith("|-") or not s:
                continue
            cell = re.sub(r"^[|!]\+?", "", s)
            cell = re.sub(r"[|!]{2}", " · ", cell)
            cell = re.sub(r"^[^|\[]*\|(?!\|)", "", cell, count=1)  # 셀 속성 제거(링크 보호)
            cell = re.sub(r'(?:[a-zA-Z-]+=(?:"[^"]*"|[^\s|·]+)\s*)+\|(?!\|)', "", cell)
            if cell.strip():
                lines.append(cell.strip())
            continue
        lines.append(line)
    w = "\n".join(lines)
    w = re.sub(r"\[\[분류:[^\]]*\]\]", "", w)
    w = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", w)  # [[a|b]]→b
    w = re.sub(r"\[\[([^\]]*)\]\]", r"\1", w)            # [[a]]→a
    w = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", w)
    w = re.sub(r"\[https?://\S+\]", "", w)
    w = re.sub(r"'{2,}", "", w)
    w = re.sub(r"<br\s*/?>", "\n", w, flags=re.I)
    w = re.sub(r"<[^>]+>", "", w)
    w = re.sub(r"^[*#:;]+\s*", "- ", w, flags=re.M)
    w = re.sub(r"^=+\s*.*?\s*=+\s*$", "", w, flags=re.M)  # 남은 표제 제거
    w = re.sub(r"[ \t]+", " ", w)
    w = re.sub(r"\n{3,}", "\n\n", w)
    return w.strip()


# ---------- 인포박스 ----------

def extract_infobox(wikitext: str):
    m = re.search(r"\{\{[^{}\n|]*정보", wikitext)
    if not m:
        return None, None
    start = m.start()
    depth, j = 0, start
    while j < len(wikitext):
        if wikitext.startswith("{{", j):
            depth += 1
            j += 2
        elif wikitext.startswith("}}", j):
            depth -= 1
            j += 2
            if depth == 0:
                break
        else:
            j += 1
    block = wikitext[start:j]
    name = block[2:block.find("\n") if "\n" in block[:80] else 40].split("|")[0].strip()
    fields = {}
    depth = 0
    cur = []
    parts = []
    for ch_i, ch in enumerate(block[2:-2]):
        if block[2 + ch_i:2 + ch_i + 2] in ("{{", "[["):
            depth += 1
        elif block[ch_i:2 + ch_i] in ("}}", "]]"):
            depth = max(0, depth - 1)
        if ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = clean_wikitext(v).replace("\n", ", ").strip(" ,")
        if k and v:
            fields[k] = v
    return name, fields


def split_names(value: str) -> list:
    return [x.strip() for x in re.split(r"[,·/\n]| - ", value)
            if 1 < len(x.strip()) <= 20 and not x.strip().isdigit()]


def infobox_summary(dt: str, fields: dict) -> str:
    """합성 서두용 핵심 필드 요약 — movie: 감독·개봉일·장르 / person: 출생·직업·활동."""
    if not fields:
        return ""
    if dt == "movie":
        pick = [("감독", fields.get("감독")),
                ("개봉일", fields.get("개봉일") or fields.get("개봉")),
                ("장르", fields.get("장르"))]
    elif dt == "person":
        pick = [("출생", fields.get("출생일") or fields.get("출생")),
                ("직업", fields.get("직업")),
                ("활동 기간", fields.get("활동 기간") or fields.get("활동기간"))]
    else:
        pick = list(fields.items())[:3]
    return " · ".join(f"{k}: {str(v)[:80]}" for k, v in pick if v)


# ---------- doc_type ----------

def doc_type_of(categories: list) -> str:
    cats = " | ".join(categories)
    if re.search(r"\d{4}년 (출생|사망)|살아있는 사람", cats):
        return "person"
    if re.search(r"배우$|감독$|영화인$|성우$|가수$|평론가$|모델$", cats):
        return "person"
    if re.search(r"\d{4}년 영화|영화 작품|배경으로 한 영화|영화$", cats):
        return "movie"
    return "other"


# ---------- 섹션 청크 ----------

def doc_sections(rec: dict):
    """(섹션명, 원문 위키텍스트) 목록 — 최상위(level 2) 기준, 하위 섹션 병합.

    API byteoffset은 실측 결과 반환 wikitext의 '문자' 인덱스와 정확히 일치
    (섹션 표제 위치 전 표본 대조) → 문자 슬라이싱 사용."""
    w = rec["wikitext"]
    tops = [s for s in rec["sections"]
            if s.get("level") == "2" and isinstance(s.get("byteoffset"), int)]
    if not tops:
        return [("서두", w)]
    out = [("서두", w[:tops[0]["byteoffset"]])]
    for i, s in enumerate(tops):
        end = tops[i + 1]["byteoffset"] if i + 1 < len(tops) else len(w)
        out.append((s["line"].strip(), w[s["byteoffset"]:end]))
    return out


def _pack(units, sep: str) -> list:
    pieces, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) + len(sep) > MAX_CHUNK:
            pieces.append(buf)
            buf = u
        else:
            buf = f"{buf}{sep}{u}" if buf else u
    if buf:
        pieces.append(buf)
    return pieces


def split_text(text: str) -> list:
    """MAX_CHUNK 초과 시 문단(빈 줄) → 줄 → 고정 폭 순으로 분할."""
    if len(text) <= MAX_CHUNK:
        return [text]
    pieces = []
    for p in _pack(text.split("\n\n"), "\n\n"):
        if len(p) <= MAX_CHUNK:
            pieces.append(p)
            continue
        for q in _pack(p.split("\n"), "\n"):
            if len(q) <= MAX_CHUNK:
                pieces.append(q)
            else:  # 개행 없는 초장문 → 고정 폭
                pieces += [q[i:i + MAX_CHUNK] for i in range(0, len(q), MAX_CHUNK)]
    out = []  # MIN 미만 조각은 직전 조각에 병합 (위치 무관)
    for p in pieces:
        if out and len(p) < MIN_CHUNK:
            out[-1] += "\n" + p
        else:
            out.append(p)
    return out


def chunk_doc(rec: dict, dt: str, ib_fields):
    """전수 정책: 전 문서 최소 1청크. 서두<MIN_CHUNK면 인포박스 핵심 필드를
    덧붙인 합성 서두(synthetic=True)로 보강."""
    chunks, merged, dropped, n_split = [], 0, 0, 0
    cleaned = []
    for name, raw in doc_sections(rec):
        if name in REF_SECTIONS:
            continue
        text = clean_wikitext(raw)
        if not text:
            continue
        if len(text) < MIN_CHUNK and cleaned:  # 잔섹션 → 직전 섹션 병합
            cleaned[-1] = (cleaned[-1][0], cleaned[-1][1] + "\n" + text)
            merged += 1
            continue
        cleaned.append((name, text))

    # 합성 서두: 정리 후 서두가 MIN_CHUNK 미만이면 인포박스 핵심 필드 덧붙임
    intro_idx = next((k for k, (n, _) in enumerate(cleaned) if n == "서두"), None)
    intro_text = cleaned[intro_idx][1] if intro_idx is not None else ""
    short_intro = len(intro_text) < MIN_CHUNK
    synthetic = False
    if short_intro:
        summary = infobox_summary(dt, ib_fields or {})
        if summary:
            merged_intro = ((intro_text + "\n") if intro_text else "") + \
                f"{rec['title']} — {summary}"
            synthetic = True
            if intro_idx is not None:
                cleaned[intro_idx] = ("서두", merged_intro)
            else:
                cleaned.insert(0, ("서두", merged_intro))

    seen_names = {}  # 문서 내 동명 섹션 → "이름 (2)"로 구분 (청크 ID 유일성)
    renamed = []
    for name, text in cleaned:
        k = seen_names.get(name, 0) + 1
        seen_names[name] = k
        renamed.append((name if k == 1 else f"{name} ({k})", text))
    cleaned = renamed

    for name, text in cleaned:
        if len(text) < MIN_CHUNK and name != "서두":
            dropped += 1
            continue
        pieces = split_text(text)
        if len(pieces) > 1:
            n_split += 1
        for idx, piece in enumerate(pieces, 1):
            cid = (f"{rec['title']}::{name}" if len(pieces) == 1
                   else f"{rec['title']}::{name}::{idx}")
            c = {
                "id": cid, "title": rec["title"], "section": name,
                "text": piece.strip(), "doc_type": dt,
                "categories": rec["categories"],
            }
            if synthetic and name == "서두":
                c["synthetic"] = True
            chunks.append(c)

    fallback = False
    if not chunks:  # 서두·섹션·인포박스 전부 비어도 제목 청크로 1층 존재 보장
        fallback = True
        chunks.append({"id": f"{rec['title']}::서두", "title": rec["title"],
                       "section": "서두", "text": rec["title"], "doc_type": dt,
                       "categories": rec["categories"], "synthetic": True})
    return chunks, {"merged": merged, "dropped": dropped, "split": n_split,
                    "short_intro": short_intro, "synthetic": synthetic,
                    "fallback": fallback}


# ---------- 메인 ----------

def main():
    recs, seen_titles = [], set()
    with open(OUT / "pages_snapshot.jsonl", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:  # 수집 중 미완 라인 방어
                continue
            if r["title"] in seen_titles:
                continue
            seen_titles.add(r["title"])
            recs.append(r)
    print(f"스냅샷 {len(recs)}문서 로드")

    all_chunks, infoboxes = [], []
    filmo = {}
    cat_index = {}
    n_merged = n_dropped = n_split = n_ibox_present = n_ibox_ok = 0
    n_stub200 = n_short_intro = n_synth = n_fallback = 0
    dt_dist = Counter()

    for i, rec in enumerate(recs, 1):
        dt = doc_type_of(rec["categories"])
        dt_dist[dt] += 1
        if rec.get("intro_len", 0) < 200:
            n_stub200 += 1
        for c in rec["categories"]:
            cat_index.setdefault(c, []).append(rec["title"])

        name, fields = extract_infobox(rec["wikitext"])
        if name is not None:
            n_ibox_present += 1
            if fields:
                n_ibox_ok += 1
                ib = {"title": rec["title"], "doc_type": dt, "infobox": name,
                      "fields": fields, "source_chunk": f"{rec['title']}::서두"}
                infoboxes.append(ib)

        chunks, cinfo = chunk_doc(rec, dt, fields)
        all_chunks += chunks
        n_merged += cinfo["merged"]
        n_dropped += cinfo["dropped"]
        n_split += cinfo["split"]
        n_short_intro += cinfo["short_intro"]
        n_synth += cinfo["synthetic"]
        n_fallback += cinfo["fallback"]

        # 3층: 필모 1차 (인물 문서의 출연/작품 섹션)
        if dt == "person":
            for sname, raw in doc_sections(rec):
                if not FILMO_SECTION.search(sname) or FILMO_EXCLUDE.search(sname):
                    continue
                for line in raw.splitlines():
                    if not re.match(r"\s*[*#|]", line):
                        continue
                    for target in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", line):
                        t = target.strip()
                        if not t or t.startswith(("파일:", "분류:", "File:")):
                            continue
                        if DATE_LINK.match(t) or t in NONWORK_LINKS:  # 연도·방송사 링크 제외
                            continue
                        y = re.search(r"(19|20)\d{2}", line)
                        key = rec["title"]
                        lst = filmo.setdefault(key, [])
                        if not any(e["작품"] == t for e in lst):
                            lst.append({"작품": t, "연도": int(y.group()) if y else None,
                                        "출처": f"{rec['title']}::{sname}"})
        if i % 1000 == 0:
            print(f"  ... 파싱 {i}/{len(recs)}")

    # 3층: 역인덱스 보충 (영화 인포박스 출연 필드 → 배우별 출연작)
    n_reverse = 0
    persons_with_section = set(filmo)
    for ib in infoboxes:
        if ib["doc_type"] != "movie":
            continue
        year = None
        ym = re.search(r"(19|20)\d{2}", ib["fields"].get("개봉일", "") or
                       ib["fields"].get("개봉", ""))
        if ym:
            year = int(ym.group())
        for f in CAST_FIELDS:
            if f not in ib["fields"]:
                continue
            for name in split_names(ib["fields"][f]):
                lst = filmo.setdefault(name, [])
                if not any(e["작품"] == ib["title"] for e in lst):
                    lst.append({"작품": ib["title"], "연도": year,
                                "출처": f"{ib['title']}::인포박스(역인덱스)"})
                    n_reverse += 1

    # 저장
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(OUT / "infobox.jsonl", "w", encoding="utf-8") as f:
        for ib in infoboxes:
            f.write(json.dumps(ib, ensure_ascii=False) + "\n")
    with open(OUT / "filmography.json", "w", encoding="utf-8") as f:
        json.dump(filmo, f, ensure_ascii=False)
    with open(OUT / "category_index.json", "w", encoding="utf-8") as f:
        json.dump(cat_index, f, ensure_ascii=False)

    lens = sorted(len(c["text"]) for c in all_chunks)
    pct = lambda p: lens[min(len(lens) - 1, round(p / 100 * (len(lens) - 1)))]  # noqa: E731
    total_filmo = sum(len(v) for v in filmo.values())
    stats = {
        "docs": len(recs),
        "doc_type_dist": dict(dt_dist),
        "chunks": {"n": len(all_chunks),
                   "len": {"min": lens[0], "p25": pct(25), "p50": pct(50),
                           "p75": pct(75), "p90": pct(90), "max": lens[-1],
                           "mean": round(sum(lens) / len(lens))},
                   "per_doc_mean": round(len(all_chunks) / len(recs), 2)},
        "stub_docs": {"intro_lt200_collect": n_stub200,
                      "cleaned_intro_lt100": n_short_intro},
        "chunk_processing": {"merged_small_sections": n_merged,
                             "dropped_small_sections": n_dropped,
                             "split_long_sections": n_split,
                             "synthetic_intros": n_synth,
                             "fallback_title_only": n_fallback,
                             "docs_with_chunks": len({c["title"] for c in all_chunks})},
        "infobox": {"present": n_ibox_present, "parsed_ok": n_ibox_ok,
                    "present_rate": round(n_ibox_present / len(recs), 3),
                    "parse_success_rate": round(n_ibox_ok / max(1, n_ibox_present), 3),
                    "mean_fields": round(sum(len(x["fields"]) for x in infoboxes)
                                         / max(1, len(infoboxes)), 1)},
        "filmography": {"persons": len(filmo), "entries": total_filmo,
                        "reverse_entries": n_reverse,
                        "reverse_ratio": round(n_reverse / max(1, total_filmo), 3),
                        "persons_covered_only_by_reverse":
                            len(set(filmo) - persons_with_section)},
        "category_index": {"categories": len(cat_index)},
    }
    with open(OUT / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 무작위 20문서 샘플 (층별 산출물)
    rng = random.Random(42)
    sample = rng.sample(recs, 20)
    by_title_chunks = {}
    for c in all_chunks:
        by_title_chunks.setdefault(c["title"], []).append(c)
    ib_by_title = {x["title"]: x for x in infoboxes}
    lines = []
    for rec in sample:
        t = rec["title"]
        dt = doc_type_of(rec["categories"])
        cs = by_title_chunks.get(t, [])
        lines.append(f"■ {t} [{dt}] — 청크 {len(cs)}개")
        for c in cs[:6]:
            syn = " [합성]" if c.get("synthetic") else ""
            lines.append(f"   1층 {c['id']}{syn} ({len(c['text'])}자): {c['text'][:60]}…")
        ib = ib_by_title.get(t)
        if ib:
            preview = dict(list(ib["fields"].items())[:4])
            lines.append(f"   2층 인포박스[{ib['infobox']}] {len(ib['fields'])}필드: "
                         f"{json.dumps(preview, ensure_ascii=False)[:150]}")
        else:
            lines.append("   2층 인포박스: 없음/파싱 실패")
        if dt == "person":
            fl = filmo.get(t, [])
            src_r = sum(1 for e in fl if "역인덱스" in e["출처"])
            lines.append(f"   3층 필모 {len(fl)}항목 (역인덱스 보충 {src_r}): "
                         + ", ".join(e["작품"] for e in fl[:5]) + ("…" if len(fl) > 5 else ""))
    sample_txt = "\n".join(lines)
    with open(OUT / "sample_20.txt", "w", encoding="utf-8") as f:
        f.write(sample_txt)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n샘플 20문서: {OUT / 'sample_20.txt'}")


if __name__ == "__main__":
    main()
