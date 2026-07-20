"""ChromaDB 접근 계층. PersistentClient(./db), collection="wiki_movies".

- 컬렉션 로드 직후 hnsw:space == "cosine" assert (Day 1 추가 지시)
- 임베딩은 .env의 EMBED_MODEL(OpenAI) 사용, distance = cosine ([0, 2])
"""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(ROOT / "db")
COLLECTION_NAME = "wiki_movies"

_openai = None
_collection = None


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def embed_texts(texts: list) -> list:
    """텍스트 목록 → 임베딩 목록 (EMBED_MODEL, 배치 100)."""
    model = os.environ["EMBED_MODEL"]
    out = []
    for i in range(0, len(texts), 100):
        batch = [t.replace("\n", " ") for t in texts[i : i + 100]]
        resp = _get_openai().embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=DB_PATH)


def recreate_collection():
    """재구축용: 기존 컬렉션 삭제 후 cosine으로 새로 생성 (build_db.py 전용)."""
    global _collection
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # 없으면 그냥 생성
    _collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        _collection = _client().get_collection(COLLECTION_NAME)
        # 로드 직후 검증 (Day 1 추가 지시): cosine이 아니면 즉시 중단
        space = (_collection.metadata or {}).get("hnsw:space")
        assert space == "cosine", f"hnsw:space={space!r} != 'cosine'"
    return _collection


def search(query: str, k: int = 5):
    """query 임베딩 → top-k. 반환: (search_results, top1_distance)

    search_results: [{id, title, text, distance}] (SPEC §2)
    """
    coll = get_collection()
    emb = embed_texts([query])[0]
    res = coll.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for id_, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append(
            {"id": id_, "title": meta["title"], "text": doc, "distance": dist}
        )
    top1 = results[0]["distance"] if results else float("inf")
    return results, top1


def db_info() -> dict:
    """/health용: 청크 수 + distance space."""
    coll = get_collection()
    return {
        "db_chunks": coll.count(),
        "space": (coll.metadata or {}).get("hnsw:space"),
    }
