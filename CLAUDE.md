# CLAUDE.md — 프로젝트 헌법

HotpotQA 기반 Agentic RAG 시스템. 1-pass RAG 대비 검색·답변 품질 개선을 **수치로 증명**하는 것이 목표.
상세 설계는 `docs/SPEC.md`, 일정과 완료 기준은 `docs/PLAN.md` — **이 세 문서가 단일 진실 원천이다.**

## 절대 규칙

1. **SPEC.md와 충돌하는 구현 판단이 필요하면 임의로 결정하지 말고 멈추고 보고하라.**
   "더 좋아 보이는" 구조 변경, 필드 추가/삭제, 모델 변경 전부 포함.
2. **평가 먼저**: 각 단계 완료 시 `python eval/run_eval.py`를 실행하고 결과를
   `eval/results/`에 저장 후 git commit. 완료 기준은 PLAN.md의 해당 Day 참조.
3. **retry_count는 Rewriter에서만 +1, hop전환에서만 0 리셋.** 다른 곳에서 건드리지 않는다.
4. **모든 LLM 호출은 `core/llm.py`의 `call_llm()` 단일 경유** — llm_call_count 증가,
   assert ≤ 20(불변식), temperature=0, 모델명은 .env의 LLM_MODEL.
5. **기록용 state 필드(tried_queries, judge_history, evidence, sources)는 append만, 덮어쓰기 금지.**
6. **계층 규칙**: frontend는 HTTP로 backend만 호출. backend는 접수-위임-반환만(로직 금지).
   agent는 HTTP/화면을 모른다. ChromaDB 접근은 agent의 검색 노드를 통해서만.
7. API 키·주소는 .env로만. 코드에 하드코딩 금지. .env는 .gitignore에 포함.

## 기술 스택 (변경 금지)

Python / LangGraph / ChromaDB(PersistentClient, ./db) / FastAPI+uvicorn / Streamlit
LLM: gpt-4o-mini (.env LLM_MODEL) / 임베딩: text-embedding-3-small (.env EMBED_MODEL)
temperature=0 전 노드 고정.

## 디렉토리 구조

```
CLAUDE.md  docs/{SPEC.md, PLAN.md}  .env(.gitignore)
core/       # 공유: llm.py(call_llm), db.py(ChromaDB 접근), state.py(AgentState, make_initial_state)
agents/
  naive/    # 1-pass 직선 파이프라인 (비교 기준선 — day 1 완성 후 동결)
  baseline/ # 기본 Agentic 그래프 (day 5 튜닝 완료 후 git tag v1-baseline로 동결)
  improved/ # 구조 수정판 (day 6~7, 내용은 day 5 결과 보고 결정)
scripts/    # build_dataset.py, derive_singlehop.py, build_db.py
eval/       # run_eval.py, testset.jsonl, results/
backend/    # main.py (FastAPI)
frontend/   # app.py (Streamlit)
```

## 운영 리듬

- 지시 형식: "docs/PLAN.md의 Day N을 수행. SPEC.md와 이 문서를 준수."
- 하네스와 에이전트의 계약: 어떤 에이전트든 "질문 in → evidence 포맷
  [{"hop": n, "chunk_ids": [...]}] 포함 결과 out"만 지키면 run_eval.py로 채점 가능.
- git tag: day 5 종료 시 v1-baseline, day 8 종료 시 v2-improved.
