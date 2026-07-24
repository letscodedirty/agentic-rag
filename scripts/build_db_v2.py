"""v2 1층 청크 → ChromaDB(./db_v2) 적재 (Day 6).

기존 ./db 무변경. data/v2/chunks.jsonl(56,546청크)을 wiki_movies_v2 컬렉션에
cosine으로 임베딩·적재한다. 배치 500청크(임베딩은 core.db.embed_texts가 내부
100 단위), 일시 오류는 배치당 3회 재시도.

실행: ./venv/bin/python -u scripts/build_db_v2.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402

BATCH = 500


def main():
    chunks = [json.loads(l)
              for l in open(ROOT / "data" / "v2" / "chunks.jsonl", encoding="utf-8")]
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "청크 ID 중복"
    print(f"[적재] {len(chunks)}청크 → {db.V2_COLLECTION_NAME} (cosine)", flush=True)

    coll = db.recreate_v2_collection()
    assert (coll.metadata or {}).get("hnsw:space") == "cosine"

    t0 = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        for attempt in range(3):
            try:
                embs = db.embed_texts([c["text"] for c in batch])
                coll.add(
                    ids=[c["id"] for c in batch],
                    embeddings=embs,
                    documents=[c["text"] for c in batch],
                    metadatas=[{
                        "title": c["title"],
                        "doc_type": c["doc_type"],
                        "section": c["section"],
                        "categories": "|".join(c["categories"]),
                        "synthetic": bool(c.get("synthetic", False)),
                    } for c in batch],
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[재시도 {attempt + 1}] {i}~ 배치: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        done = min(i + BATCH, len(chunks))
        if (i // BATCH) % 10 == 0 or done == len(chunks):
            el = time.time() - t0
            print(f"[적재] {done}/{len(chunks)} ({el:.0f}s, "
                  f"{done / el:.0f}청크/s)", flush=True)

    n = coll.count()
    print(f"[완료] 컬렉션 청크 수 {n} (기대 {len(chunks)}) — "
          f"{'일치' if n == len(chunks) else '불일치!'}", flush=True)
    assert n == len(chunks)


if __name__ == "__main__":
    main()
