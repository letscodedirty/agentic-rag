"""v2 수집 (V2_DESIGN.md, 전수 정책): 두 트리 합집합 전 문서 전문 스냅샷.

기존 코드·./db 무변경. 산출물: data/v2/pages_snapshot.jsonl (원문 스냅샷),
data/v2/collect_v2_stats.json, data/v2/intro_lens_cache.json (서두 길이 캐시).
재실행 시 이미 수집된 문서는 건너뜀(이어받기).

단계:
  A. "분류:대한민국의 영화" + "분류:대한민국의 배우" 트리(깊이 3) 목록 합집합
  B. 서두(extracts) 길이 조회 — intro_len 메타데이터·스텁 통계 기록용. 문서 제외 없음.
  C. 전 문서 전문: action=parse (wikitext + sections + categories) → 스냅샷.
     제외는 리다이렉트 중복·동음이의 문서뿐 (전수 정책).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_wiki  # noqa: E402  (api_get·category_members·fetch_intros 재사용)
from probe_v2 import bfs_titles  # noqa: E402

OUT_DIR = ROOT / "data" / "v2"
SNAPSHOT = OUT_DIR / "pages_snapshot.jsonl"
INTRO_CACHE = OUT_DIR / "intro_lens_cache.json"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[A] 분류 트리 합집합 나열", flush=True)
    movie = bfs_titles("분류:대한민국의 영화")
    actor = bfs_titles("분류:대한민국의 배우")
    titles = sorted(set(movie) | set(actor))
    tree_of = {t: ("both" if t in movie and t in actor
                   else "movie" if t in movie else "actor") for t in titles}
    print(f"[A] 완집합 {len(titles)}문서 (movie-only {sum(1 for v in tree_of.values() if v=='movie')}, "
          f"actor-only {sum(1 for v in tree_of.values() if v=='actor')}, "
          f"both {sum(1 for v in tree_of.values() if v=='both')})", flush=True)

    # [B] 서두 길이 조회 — 기록용(스텁 통계·intro_len 메타데이터). 필터 아님.
    intro_lens = {}
    if INTRO_CACHE.exists():
        with open(INTRO_CACHE, encoding="utf-8") as f:
            intro_lens = json.load(f)
    todo = [t for t in titles if t not in intro_lens]
    print(f"[B] 서두 길이 조회(기록용): 캐시 {len(intro_lens)}건, 신규 {len(todo)}건", flush=True)
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        got = collect_wiki.fetch_intros(batch)
        for t in batch:
            intro_lens[t] = len(got.get(t, ""))
        if (i // 20) % 100 == 0:
            print(f"[B] 진행 {min(i + 20, len(todo))}/{len(todo)}", flush=True)
            with open(INTRO_CACHE, "w", encoding="utf-8") as f:
                json.dump(intro_lens, f, ensure_ascii=False)
    with open(INTRO_CACHE, "w", encoding="utf-8") as f:
        json.dump(intro_lens, f, ensure_ascii=False)
    n_lt200 = sum(1 for t in titles if intro_lens.get(t, 0) < 200)
    n_lt100 = sum(1 for t in titles if intro_lens.get(t, 0) < 100)
    print(f"[B] 완료: 서두<200자 {n_lt200} / <100자 {n_lt100} (제외 없음 — 전수 수집)", flush=True)

    done = set()
    if SNAPSHOT.exists():
        with open(SNAPSHOT, encoding="utf-8") as f:
            done = {json.loads(l)["title"] for l in f if l.strip()}
        print(f"[C] 이어받기: 기존 스냅샷 {len(done)}문서 건너뜀", flush=True)
    n_prev = len(done)

    print(f"[C] 전문 수집(전수): 대상 {len(titles)}문서", flush=True)
    n_fail = n_dup = n_disambig = n_new = 0
    with open(SNAPSHOT, "a", encoding="utf-8") as out:
        for i, t in enumerate(titles, 1):
            if t in done:
                continue
            data = collect_wiki.api_get(
                {"action": "parse", "page": t,
                 "prop": "wikitext|sections|categories", "redirects": "1"})
            p = data.get("parse")
            if not p or not p.get("wikitext"):
                n_fail += 1
                continue
            resolved = p.get("title") or t
            if resolved in done:  # 리다이렉트가 이미 수집된 문서를 가리킴
                n_dup += 1
                continue
            cats = [c.get("category", "").replace("_", " ")
                    for c in p.get("categories", [])]
            if any("동음이의" in c for c in cats):
                n_disambig += 1
                continue
            rec = {
                "title": resolved,
                "tree": tree_of.get(t, "other"),
                "intro_len": intro_lens.get(t, 0),
                "wikitext": p["wikitext"],
                "sections": [{"line": s.get("line", ""), "level": s.get("level"),
                              "byteoffset": s.get("byteoffset")}
                             for s in p.get("sections", [])],
                "categories": cats,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(resolved)
            n_new += 1
            if n_new % 200 == 0:
                out.flush()
                print(f"[C] 진행 {i}/{len(titles)} (신규 {n_new}, 실패 {n_fail}, "
                      f"리다이렉트중복 {n_dup}, 동음이의 {n_disambig})", flush=True)
            time.sleep(0.03)
    stats = {
        "union_docs": len(titles),
        "snapshot_docs": n_prev + n_new,
        "snapshot_fail": n_fail,
        "skip_redirect_dup": n_dup,
        "skip_disambig": n_disambig,
        "intro_lt200": n_lt200,
        "intro_lt100": n_lt100,
        "tree_dist": {k: sum(1 for v in tree_of.values() if v == k)
                      for k in ("movie", "actor", "both")},
    }
    with open(OUT_DIR / "collect_v2_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[완료] 스냅샷 {SNAPSHOT} | {json.dumps(stats, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
