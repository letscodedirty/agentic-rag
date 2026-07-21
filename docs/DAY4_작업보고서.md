# Day 4 작업 보고서 — 통합 (backend API + Streamlit 화면)

> 2026-07-21 · 근거 문서: SPEC.md §6, PLAN.md Day 4, RUNBOOK Day 4

## 요약

FastAPI backend(`/health` → `/ask_naive` → `/ask`)와 Streamlit frontend(탭 2개)를
완성해 **브라우저에서 같은 질문의 naive vs agentic 비교 화면 동작을 확인**했다
(Day 4 완료 기준 충족, 사용자 브라우저 검증 완료). 계층 규칙 준수: backend는
접수-위임-반환만, frontend는 HTTP로 backend만 호출.

---

## 1. backend/main.py (FastAPI)

| 엔드포인트 | 계약 | 실측 검증 |
|---|---|---|
| `GET /health` | {status, db_chunks, **space**} | `{"status":"ok","db_chunks":1185,"space":"cosine"}` |
| `POST /ask_naive` | {question, top_k} → answer, sources, llm_calls, elapsed_sec | 사바하 개봉일 정답, LLM 1회, 5.1초 |
| `POST /ask` | {question, top_k} → answer, strategy, plan, rewrite_history, judge_history(2×2 값), intermediate_answers, retry_total, hop_reached, exhausted(+reason), llm_calls, sources, elapsed_sec | bridge 질문 완주(중간답 전계수 → 서강대학교), LLM 5회, 8.4초 |

- `localhost:8000/docs`(Swagger)에서 3개 경로 모두 노출·테스트 가능 (HTTP 200 확인).
- **계층 규칙**: pydantic 요청 검증 후 `run_naive`/`run_agent`에 위임, 결과 그대로 반환.
  로직 없음. 예외는 HTTP 500으로 전달만.
- `max_retry`는 요청 파라미터로 받지 않음 — 서버 상수(core/config.MAX_RETRY=2) 고정 (§6).
- **계약 외 추가 필드**: 응답에 `evidence` 포함 — CLAUDE.md 하네스 계약("evidence
  포맷 포함 결과 out")상 `run_eval --http` 채점에 필수이기 때문.

## 2. frontend/app.py (Streamlit, 탭 2개)

**탭 ① Agentic 단독**
- 시스템 선택기(현재 `baseline`만 — day 7 후 improved 추가 예정), 질문 입력,
  top_k 슬라이더(1~10, 기본 5)
- 답변 + 전략 뱃지(정답형/탐색형 색상 구분) + hop별 출처
- expander 4종: ① 계획(plan) ② hop별 판정 표(verdict·source·relevance·sufficiency·
  reason) ③ 재작성 이력 ④ 중간 답·통계(LLM 호출/재작성/도달 hop/소요 시간)
- exhausted 경고 박스: 사유 표시 + 제한적 답변 안내

**탭 ② 비교**
- 입력 하나 → `/ask_naive`·`/ask` 호출 → naive | agentic 좌우 배치
  (agentic 쪽은 단독 탭과 동일한 상세 표시)

**공통**
- 시작 시 `/health` 사전 확인: 실패하면 uvicorn 기동 명령이 담긴 안내 문구 표시 후
  중단, 성공하면 청크 수·space 캡션 표시
- `session_state`로 마지막 결과 유지 (rerun에도 화면 보존)
- BACKEND_URL은 .env에서만 읽음 (하드코딩 금지 규칙)

## 3. 기동 방법

```bash
# 터미널 1 — backend
./venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
# 터미널 2 — frontend
./venv/bin/streamlit run frontend/app.py   # → http://localhost:8501
```

## 4. 검증

- backend: 서버 기동 후 3개 엔드포인트 실호출 + `/docs` HTTP 200 확인
- frontend: 문법 검사 + 헤드리스 기동(HTTP 200) 확인
- **사용자 브라우저 검증 완료**: 질문 입력 → 답변·expander 4종 확인, 비교 탭에서
  naive vs agentic 좌우 비교 동작 확인 (Day 4 완료 기준)

## 5. 다음 단계 (Day 5 예고)

서브셋 48(조합별 16 층화)로 k∈{3, 5, 10} 회전 평가 → best k를 150 전체로 확정 측정
→ config/baseline.yaml 고정 → `git tag v1-baseline`. GATE_THRESHOLD는 0.70 고정
(튜닝 비대상)이므로 그리드는 k 하나다. 청크 모드(문단 vs 문장) 실험은 미실시
상태라 포함 여부를 Day 5 시작 시 결정.
