"""SPEC §3 agents/naive: 1-pass RAG 비교 기준선 (day 1 완성 후 동결).

임베딩 → top-k 검색 → 생성 1회. hop/retry 없음. LLM 1회.
evidence 포맷 [{"hop": 0, "chunk_ids": [...]}] — 하네스 계약 준수.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.llm import call_llm  # noqa: E402

SYSTEM_PROMPT = (
    "너는 제공된 문서만 근거로 답하는 한국어 QA 어시스턴트다. "
    "제공 문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고 답하라. "
    "답변 끝에 근거 문서의 title을 (출처: ...) 형식으로 표기하라."
)


def run_naive(question: str, top_k: int = 5) -> dict:
    t0 = time.time()
    state = {"llm_call_count": 0}
    results, top1 = db.search(question, k=top_k)
    docs = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in results)
    answer = call_llm(
        state,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[질문]\n{question}\n\n[문서]\n{docs}"},
        ],
    )
    chunk_ids = [r["id"] for r in results]
    return {
        "answer": answer,
        "evidence": [{"hop": 0, "chunk_ids": chunk_ids}],
        "sources": [{"hop": 0, "titles": chunk_ids}],
        "llm_calls": state["llm_call_count"],
        "top1_distance": top1,
        "elapsed_sec": round(time.time() - t0, 2),
    }
