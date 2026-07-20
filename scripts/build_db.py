"""SPEC §4-4: 수집 청크(+bridge 외부 문서 청크) → ChromaDB(./db).

- title 기준 중복 제거 (같은 title·같은 텍스트=1개, 다르면 긴 쪽 보존+경고)
- 문단=청크, id=title, 메타데이터 {title}
- 재구축 시 기존 컬렉션 삭제 후 생성, metadata={"hnsw:space": "cosine"} (추가 지시)
- 무결성 체크(유형별):
    single      = gold_answer가 정답 청크 텍스트에 존재
    bridge      = hop1 답이 hop1 청크에, gold_answer가 hop2 청크에 존재
    comparison  = 근거 값(hop_answers)이 각 청크에 존재 (gold_answer 자체는 검사 제외)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe(chunks: list) -> dict:
    by_title = {}
    for c in chunks:
        t = c["title"]
        if t not in by_title:
            by_title[t] = c
        elif c["text"] != by_title[t]["text"]:
            keep = c if len(c["text"]) > len(by_title[t]["text"]) else by_title[t]
            print(f"[경고] title 중복(텍스트 상이): {t!r} → 긴 쪽({len(keep['text'])}자) 보존")
            by_title[t] = keep
    return by_title


def integrity_check(by_title: dict, testset: list):
    fails = []
    counts = {}
    for i, r in enumerate(testset, 1):
        combo = r["combo"]
        counts.setdefault(combo, [0, 0])
        ok = True
        h1, h2 = r["hop_answers"].get("1"), r["hop_answers"].get("2")

        def in_chunk(title, needle):
            c = by_title.get(title)
            return c is not None and needle in c["text"]

        if combo == "single":
            ok = in_chunk(h1["title"], r["gold_answer"])
        elif combo == "bridge":
            ok = in_chunk(h1["title"], h1["answer"]) and in_chunk(h2["title"], r["gold_answer"])
        elif combo == "comparison":
            ok = in_chunk(h1["title"], h1["answer"]) and in_chunk(h2["title"], h2["answer"])
        counts[combo][0] += int(ok)
        counts[combo][1] += 1
        if not ok:
            fails.append((i, combo, r["question"][:40]))
    return counts, fails


def main():
    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl") + load_jsonl(
        ROOT / "data" / "external_chunks.jsonl"
    )
    if not chunks:
        sys.exit("data/chunks.jsonl 없음 — collect_wiki.py 먼저 실행")
    by_title = dedupe(chunks)
    titles = sorted(by_title)
    print(f"청크 {len(chunks)}개 로드 → 중복 제거 후 {len(titles)}개")

    coll = db.recreate_collection()  # 삭제 후 cosine으로 생성
    for i in range(0, len(titles), 100):
        batch = titles[i : i + 100]
        texts = [by_title[t]["text"] for t in batch]
        embs = db.embed_texts(texts)
        coll.add(
            ids=batch,
            embeddings=embs,
            documents=texts,
            metadatas=[{"title": t} for t in batch],
        )
        print(f"  ... 적재 {min(i + 100, len(titles))}/{len(titles)}")

    info = {"db_chunks": coll.count(), "space": (coll.metadata or {}).get("hnsw:space")}
    print(f"적재 완료: {info}")
    assert info["space"] == "cosine", "hnsw:space가 cosine이 아님"

    testset = load_jsonl(ROOT / "eval" / "testset.jsonl")
    if testset:
        counts, fails = integrity_check(by_title, testset)
        print("\n=== 무결성 체크 (유형별 통과/전체) ===")
        for combo, (ok, tot) in sorted(counts.items()):
            print(f"  {combo}: {ok}/{tot}")
        if fails:
            print("실패 행:")
            for i, combo, q in fails:
                print(f"  idx {i} [{combo}] {q}...")
            sys.exit(1)
    else:
        print("eval/testset.jsonl 없음 — 무결성 체크 생략")


if __name__ == "__main__":
    main()
