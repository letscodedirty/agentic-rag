"""SPEC §6 backend (FastAPI). Day 1: /health만. /ask_naive·/ask는 Day 4.

계층 규칙: 접수-위임-반환만 (로직 금지).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402

from core import db  # noqa: E402

app = FastAPI(title="agentic-rag")


@app.get("/health")
def health():
    info = db.db_info()  # {"db_chunks": n, "space": "cosine"} — space 포함 (추가 지시)
    return {"status": "ok", **info}
