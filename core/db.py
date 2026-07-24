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


def get_by_ids(ids: list) -> list:
    """chunk id 목록 → [{id, title, text}]. Generator의 comparison hop1 재조회용 (SPEC §3)."""
    coll = get_collection()
    res = coll.get(ids=ids, include=["documents", "metadatas"])
    return [
        {"id": i, "title": m["title"], "text": d}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def db_info() -> dict:
    """/health용: 청크 수 + distance space."""
    coll = get_collection()
    return {
        "db_chunks": coll.count(),
        "space": (coll.metadata or {}).get("hnsw:space"),
    }


# ---------- v2 (Day 6, 추가만 — 기존 함수 무변경) ----------

V2_DB_PATH = str(ROOT / "db_v2")
V2_COLLECTION_NAME = "wiki_movies_v2"

_v2_collection = None


def _v2_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=V2_DB_PATH)


def _v2_space(coll) -> str:
    """v2 space 판독: metadata 우선, 없으면 hnsw configuration (chromadb 1.x)."""
    space = (coll.metadata or {}).get("hnsw:space")
    if space is None:
        try:
            space = (coll.configuration_json or {}).get("hnsw", {}).get("space")
        except Exception:
            space = None
    return space


def recreate_v2_collection():
    """v2 재구축용: 기존 컬렉션 삭제 후 cosine으로 새로 생성 (build_db_v2.py 전용).

    56k 규모 ANN 재현율 확보를 위해 파라미터 상향 (실측 근거):
    기본값(efc100/M16/ef100) recall@10 74% → efc200/M32 + ef_search=2000에서
    100% (탐침 5쿼리, 웜업 후 35ms/쿼리). ef_search는 로드 시점에 읽히므로
    변경 시 프로세스 재시작 필요.
    """
    global _v2_collection
    client = _v2_client()
    try:
        client.delete_collection(V2_COLLECTION_NAME)
    except Exception:
        pass  # 없으면 그냥 생성
    from chromadb.api.collection_configuration import (
        CreateCollectionConfiguration, CreateHNSWConfiguration)
    _v2_collection = client.create_collection(
        V2_COLLECTION_NAME,
        configuration=CreateCollectionConfiguration(
            hnsw=CreateHNSWConfiguration(
                space="cosine", ef_construction=200,
                ef_search=2000, max_neighbors=32)),
    )
    assert _v2_space(_v2_collection) == "cosine"
    return _v2_collection


def get_v2_collection():
    global _v2_collection
    if _v2_collection is None:
        _v2_collection = _v2_client().get_collection(V2_COLLECTION_NAME)
        # 로드 직후 검증 (v1과 동일 규칙): cosine이 아니면 즉시 중단
        space = _v2_space(_v2_collection)
        assert space == "cosine", f"v2 hnsw:space={space!r} != 'cosine'"
    return _v2_collection


def search_v2(query: str, k: int = 5):
    """v2 1층(섹션 청크) 유사도 검색. 반환 형태는 search()와 동일 + metadata."""
    coll = get_v2_collection()
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
            {"id": id_, "title": meta["title"], "text": doc, "distance": dist,
             "doc_type": meta.get("doc_type"), "section": meta.get("section")}
        )
    top1 = results[0]["distance"] if results else float("inf")
    return results, top1


def get_v2_by_ids(ids: list) -> list:
    """v2 chunk id 목록 → [{id, title, text}]."""
    coll = get_v2_collection()
    res = coll.get(ids=ids, include=["documents", "metadatas"])
    return [
        {"id": i, "title": m["title"], "text": d}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def db_v2_info() -> dict:
    """v2 /health용: 청크 수 + distance space."""
    coll = get_v2_collection()
    return {
        "db_chunks": coll.count(),
        "space": _v2_space(coll),
    }
