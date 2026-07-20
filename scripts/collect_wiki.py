"""SPEC §4-1: 한국어 위키피디아 "분류:대한민국의 영화"(+하위 분류) 수집.

- 문서 목록: 분류 트리 BFS (하위 분류 최대 깊이 3)
- 서두 = 문서 시작~첫 섹션 전 도입부 전체 (prop=extracts, exintro, 평문)
- 서두 200자 미만 제외 (제외 건수 보고)
- 서두의 하이퍼링크(bridge 재료): prop=links 중 "링크 대상 제목이 서두 평문에
  그대로 등장"하는 것만 저장 → hop1 답이 청크 A에 실존함을 보장 (무결성 체크와 정합)
- 출력: data/chunks.jsonl [{id, title, text, links, category}], data/collect_stats.json
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://ko.wikipedia.org/w/api.php"
ROOT_CATEGORY = "분류:대한민국의 영화"
MAX_DEPTH = 3
MIN_INTRO_LEN = 200

session = requests.Session()
session.headers["User-Agent"] = "agentic-rag-day1/0.1 (educational; fzs1357.oh@gmail.com)"


def api_get(params: dict) -> dict:
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data and data["error"].get("code") == "maxlag":
                time.sleep(2 * (attempt + 1))
                continue
            return data
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def category_members(cat: str):
    """분류 1개의 (하위분류 목록, 문서 목록)."""
    subcats, pages = [], []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": cat,
        "cmlimit": "500",
        "cmnamespace": "0|14",
    }
    cont = {}
    while True:
        data = api_get({**params, **cont})
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14:
                subcats.append(m["title"])
            elif m["ns"] == 0:
                pages.append(m["title"])
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.05)
    return subcats, pages


def collect_page_titles():
    """분류 트리 BFS → {title: 최초 발견 분류}."""
    page_cat = {}
    seen_cats = {ROOT_CATEGORY}
    queue = deque([(ROOT_CATEGORY, 0)])
    n_cats = 0
    while queue:
        cat, depth = queue.popleft()
        subcats, pages = category_members(cat)
        n_cats += 1
        for t in pages:
            if "동음이의" in t or t.endswith(" 목록"):
                continue
            page_cat.setdefault(t, cat)
        if depth < MAX_DEPTH:
            for sc in subcats:
                if sc not in seen_cats:
                    seen_cats.add(sc)
                    queue.append((sc, depth + 1))
        if n_cats % 50 == 0:
            print(f"  ... 분류 {n_cats}개 순회, 문서 {len(page_cat)}개 발견")
    print(f"분류 순회 완료: 분류 {n_cats}개, 문서 {len(page_cat)}개")
    return page_cat


def fetch_intros(titles: list) -> dict:
    """제목 목록 → {제목: 서두 평문}. 20개 배치, 리다이렉트 해소."""
    intros = {}
    for i in range(0, len(titles), 20):
        batch = titles[i : i + 20]
        cont = {}
        while True:
            data = api_get(
                {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": "1",
                    "explaintext": "1",
                    "exlimit": "max",
                    "redirects": "1",
                    "titles": "|".join(batch),
                    **cont,
                }
            )
            for p in data.get("query", {}).get("pages", []):
                text = (p.get("extract") or "").strip()
                if text:
                    intros[p["title"]] = text
            if "continue" not in data:
                break
            cont = data["continue"]
        if (i // 20) % 25 == 0:
            print(f"  ... 서두 추출 {min(i + 20, len(titles))}/{len(titles)}")
        time.sleep(0.05)
    return intros


def fetch_links(titles: list) -> dict:
    """제목 목록 → {제목: 문서 내 링크 대상 제목 목록(ns=0)}. 50개 배치."""
    links = {t: [] for t in titles}
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        cont = {}
        while True:
            data = api_get(
                {
                    "action": "query",
                    "prop": "links",
                    "plnamespace": "0",
                    "pllimit": "max",
                    "redirects": "1",
                    "titles": "|".join(batch),
                    **cont,
                }
            )
            for p in data.get("query", {}).get("pages", []):
                for l in p.get("links", []):
                    links.setdefault(p["title"], []).append(l["title"])
            if "continue" not in data:
                break
            cont = data["continue"]
            time.sleep(0.03)
        if (i // 50) % 10 == 0:
            print(f"  ... 링크 수집 {min(i + 50, len(titles))}/{len(titles)}")
    return links


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"[1/3] 분류 트리 순회: {ROOT_CATEGORY} (깊이 {MAX_DEPTH})")
    page_cat = collect_page_titles()
    titles = sorted(page_cat)

    print(f"[2/3] 서두 추출: {len(titles)}개 문서")
    intros = fetch_intros(titles)

    kept_titles = [t for t in titles if len(intros.get(t, "")) >= MIN_INTRO_LEN]
    n_no_intro = sum(1 for t in titles if t not in intros)
    n_short = sum(1 for t in titles if t in intros and len(intros[t]) < MIN_INTRO_LEN)

    print(f"[3/3] 링크 수집(서두 등장 링크만 보존): {len(kept_titles)}개 문서")
    all_links = fetch_links(kept_titles)

    chunks = []
    for t in kept_titles:
        text = intros[t]
        # 서두 평문에 링크 대상 제목이 그대로 등장하는 것만 bridge 재료로 보존
        intro_links = sorted(
            {l for l in all_links.get(t, []) if l != t and l in text}
        )
        chunks.append(
            {
                "id": t,
                "title": t,
                "text": text,
                "links": intro_links,
                "category": page_cat[t],
            }
        )

    with open(data_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lens = sorted(len(c["text"]) for c in chunks)
    n_with_links = sum(1 for c in chunks if c["links"])
    stats = {
        "root_category": ROOT_CATEGORY,
        "max_depth": MAX_DEPTH,
        "docs_found": len(titles),
        "docs_no_intro": n_no_intro,
        "docs_excluded_short(<200)": n_short,
        "chunks_kept": len(chunks),
        "intro_len": {
            "min": lens[0] if lens else 0,
            "p10": pct(lens, 10),
            "p25": pct(lens, 25),
            "p50": pct(lens, 50),
            "p75": pct(lens, 75),
            "p90": pct(lens, 90),
            "max": lens[-1] if lens else 0,
            "mean": round(sum(lens) / len(lens), 1) if lens else 0,
        },
        "chunks_with_intro_links": n_with_links,
        "avg_intro_links": round(
            sum(len(c["links"]) for c in chunks) / len(chunks), 2
        )
        if chunks
        else 0,
    }
    with open(data_dir / "collect_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 수집 결과 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
