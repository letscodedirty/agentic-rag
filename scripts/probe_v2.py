"""v2 코퍼스 사전 탐침 (기존 코드·DB 무변경, 수집은 목록+샘플 100편만).

(1) "분류:대한민국의 배우" 트리(깊이 3) 문서 수
(2) 영화 트리(재나열)와의 title 합집합·교집합
(3) 무작위 100편(영화 50=기존 수집 청크 중 영화 유형, 배우 50=배우 트리)의
    본문(위키텍스트) 길이 분포, 줄거리/출연 작품 섹션 존재율, 인포박스 존재율

실행: ./venv/bin/python scripts/probe_v2.py
출력: 콘솔 보고 + data/probe_v2_stats.json
"""
import json
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_wiki  # noqa: E402  (api_get·category_members 재사용)

MAX_DEPTH = 3
PLOT_SECTIONS = ("줄거리", "시놉시스", "플롯", "스토리")


def bfs_titles(root_cat: str) -> dict:
    """분류 트리 BFS(깊이 3) — 문서 '목록'만 수집 (본문 미수집)."""
    page_cat = {}
    seen = {root_cat}
    queue = deque([(root_cat, 0)])
    n = 0
    while queue:
        cat, depth = queue.popleft()
        subcats, pages = collect_wiki.category_members(cat)
        n += 1
        for t in pages:
            if "동음이의" in t or t.endswith(" 목록"):
                continue
            page_cat.setdefault(t, cat)
        if depth < MAX_DEPTH:
            for sc in subcats:
                if sc not in seen:
                    seen.add(sc)
                    queue.append((sc, depth + 1))
        if n % 50 == 0:
            print(f"  ... [{root_cat}] 분류 {n}개, 문서 {len(page_cat)}개")
    print(f"  [{root_cat}] 완료: 분류 {n}개, 문서 {len(page_cat)}개")
    return page_cat


def probe_doc(title: str):
    """문서 1편: 섹션 목록 + 위키텍스트 (parse 1회)."""
    data = collect_wiki.api_get(
        {"action": "parse", "page": title, "prop": "sections|wikitext",
         "redirects": "1"}
    )
    p = data.get("parse")
    if not p:
        return None
    wikitext = (p.get("wikitext") or "")
    sections = [s.get("line", "") for s in p.get("sections", [])]
    return {
        "title": title,
        "len_wikitext": len(wikitext),
        "n_sections": len(sections),
        "has_plot": any(any(k in s for k in PLOT_SECTIONS) for s in sections),
        "has_filmo": any(("출연" in s) or ("작품" in s) or ("필모그래피" in s)
                         for s in sections),
        "has_infobox": bool(re.search(r"\{\{[^{}\n|]*정보", wikitext[:3000])),
    }


def dist(vals):
    if not vals:
        return {}
    v = sorted(vals)
    pct = lambda p: v[min(len(v) - 1, round(p / 100 * (len(v) - 1)))]  # noqa: E731
    return {"min": v[0], "p25": pct(25), "p50": pct(50), "p75": pct(75),
            "p90": pct(90), "max": v[-1], "mean": round(sum(v) / len(v))}


def main():
    rng = random.Random(42)

    print("[1/3] 배우 트리 나열 (깊이 3)")
    actor_pages = bfs_titles("분류:대한민국의 배우")

    print("[2/3] 영화 트리 재나열 (깊이 3, 합집합·교집합 계산용 — 목록만)")
    movie_pages = bfs_titles("분류:대한민국의 영화")

    a, m = set(actor_pages), set(movie_pages)
    overlap = {
        "actor_tree_docs": len(a),
        "movie_tree_docs": len(m),
        "union": len(a | m),
        "intersection": len(a & m),
        "actor_only": len(a - m),
        "movie_only": len(m - a),
    }

    print("[3/3] 샘플 100편 실측 (영화 50 = 기존 수집 청크 중 영화 유형 / 배우 50 = 배우 트리)")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_testset as bt
    chunks = [json.loads(l) for l in open(ROOT / "data" / "chunks.jsonl", encoding="utf-8")]
    film_titles = [c["id"] for c in chunks if bt.entity_type(c) == "영화"]
    film_sample = rng.sample(film_titles, 50)
    actor_sample = rng.sample(sorted(a), 50)

    film_stats, actor_stats = [], []
    for i, t in enumerate(film_sample + actor_sample, 1):
        d = probe_doc(t)
        if d:
            (film_stats if i <= 50 else actor_stats).append(d)
        if i % 20 == 0:
            print(f"  ... {i}/100")
        time.sleep(0.05)

    report = {
        "actor_tree": {"docs": len(a), "depth": MAX_DEPTH},
        "overlap": overlap,
        "sample": {
            "film": {
                "n": len(film_stats),
                "wikitext_len": dist([d["len_wikitext"] for d in film_stats]),
                "sections": dist([d["n_sections"] for d in film_stats]),
                "plot_rate": round(sum(d["has_plot"] for d in film_stats) / len(film_stats), 3),
                "infobox_rate": round(sum(d["has_infobox"] for d in film_stats) / len(film_stats), 3),
            },
            "actor": {
                "n": len(actor_stats),
                "wikitext_len": dist([d["len_wikitext"] for d in actor_stats]),
                "sections": dist([d["n_sections"] for d in actor_stats]),
                "filmo_rate": round(sum(d["has_filmo"] for d in actor_stats) / len(actor_stats), 3),
                "infobox_rate": round(sum(d["has_infobox"] for d in actor_stats) / len(actor_stats), 3),
            },
        },
    }
    out = ROOT / "data" / "probe_v2_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== 탐침 결과 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
