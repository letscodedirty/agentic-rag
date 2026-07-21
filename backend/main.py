"""SPEC §6 backend (FastAPI). 계층 규칙: 접수-위임-반환만, 로직 금지.

- GET  /health     → {status, db_chunks, space}  (space 포함은 Day 1 추가 지시)
- POST /ask_naive  {question, top_k} → agents/naive 위임
- POST /ask        {question, top_k} → agents/baseline 위임 (max_retry 서버 고정 2 —
                   요청 파라미터로 받지 않음, core/config.MAX_RETRY 상수 사용)

응답에는 §6 계약 필드 외에 evidence를 포함한다 — 하네스 계약(CLAUDE.md:
"evidence 포맷 포함 결과 out")에 따라 run_eval --http 채점에 필요.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.baseline.graph import run_agent  # noqa: E402
from agents.naive.pipeline import run_naive  # noqa: E402
from core import db  # noqa: E402

app = FastAPI(title="agentic-rag", description="한국어 위키 영화 도메인 Agentic RAG")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="질문 (한국어)")
    top_k: int = Field(5, ge=1, le=20, description="검색 top-k")


@app.get("/health")
def health():
    info = db.db_info()  # {"db_chunks": n, "space": "cosine"}
    return {"status": "ok", **info}


@app.post("/ask_naive")
def ask_naive(req: AskRequest):
    try:
        return run_naive(req.question, top_k=req.top_k)
    except Exception as e:  # 위임 실패는 그대로 500으로 반환
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return run_agent(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
