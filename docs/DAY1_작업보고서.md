# Day 1 작업 보고서 — 기반 구축 + 1-pass Baseline (한국어 위키 영화 도메인)

> 2026-07-20 ~ 07-21 · 커밋 `df2967a` · 근거 문서: SPEC.md §4(개정)·§5, PLAN.md Day 1

## 요약

한국어 위키피디아 영화 도메인에서 **수집 → 테스트셋 150문항 → cosine 벡터 DB →
평가 하네스 → 1-pass baseline 평가**까지 Day 1 전 과정을 완료했다.
테스트셋은 사용자 검수 3라운드를 거쳐 확정했고, naive 1-pass의 성능을
**전체 Hit Rate 0.560 / MRR 0.621**로 박제했다. single 질문은 0.92로 잘 찾지만
multi-hop(bridge·comparison)은 0.38에 그친다 — 이 격차가 Day 2~3 Agentic RAG가
개선해야 할 목표 지점이다.

---

## 1. 환경·기반 셋업

| 항목 | 내용 |
|---|---|
| 가상환경 | `venv/` (Python 3.12) + `requirements.txt` (openai, chromadb, langgraph, fastapi, uvicorn, streamlit, requests, python-dotenv) |
| 비밀 관리 | `.env`에 OPENAI_API_KEY, LLM_MODEL(gpt-4o-mini), EMBED_MODEL(text-embedding-3-small), BACKEND_URL. `.gitignore`에 `.env`, `db/`, `venv/` 등록 → git에 절대 올라가지 않음 |
| 디렉토리 | CLAUDE.md 구조 그대로: `core/` `agents/naive/` `scripts/` `eval/` `backend/` `data/`(수집 산출물) |

## 2. core 공유 모듈

### core/llm.py — 모든 LLM 호출의 단일 경유점
- `call_llm(state, messages, json_mode)` 하나로만 OpenAI를 호출한다.
- 호출마다 `state["llm_call_count"]`를 +1 하고 **assert ≤ 20**(불변식), temperature=0,
  모델명은 .env의 LLM_MODEL만 사용 — CLAUDE.md 절대 규칙 4를 코드로 강제.

### core/db.py — ChromaDB 접근 계층
- `PersistentClient(./db)`, 컬렉션 `wiki_movies`.
- **컬렉션 로드 직후 `hnsw:space == "cosine"`을 assert** (Day 1 추가 지시).
  cosine이 아니면 그 자리에서 즉시 중단된다.
- `recreate_collection()`: 재구축 시 기존 컬렉션을 **삭제 후** cosine으로 새로 생성.
- `search(query, k)`: 질의 임베딩 → top-k → `[{id, title, text, distance}]` + top1_distance.
- `db_info()`: `/health`용 `{db_chunks, space}`.

## 3. 데이터 수집 — scripts/collect_wiki.py

**방식**: 위키 API로 "분류:대한민국의 영화" 분류 트리를 깊이 3까지 BFS →
각 문서의 **서두**(문서 시작~첫 섹션 전)를 평문으로 추출 → 서두 200자 미만 제외.
bridge 재료로 **서두 평문에 실제 등장하는 하이퍼링크만** 저장한다
(→ "hop1 답이 청크 A에 실존"하는 무결성 요건과 정합).

**결과** (`data/chunks.jsonl`, `data/collect_stats.json`):

| 항목 | 값 |
|---|---|
| 발견 문서 | 16,148개 |
| 서두 200자 미만 제외 | 14,973개 (92.7% — 한국어 위키 영화 문서 다수가 1~2문장 스텁) |
| **최종 청크** | **1,153개** (영화 ~454 / 인물 ~554 / 기타 ~145) |
| 서두 길이 | 중앙값 319자, 평균 398자, 최대 3,608자 |
| 서두 등장 링크 | 청크당 평균 10.3개 (98.6%가 1개 이상 보유) |

## 4. 테스트셋 생성 — scripts/build_testset.py

**설계 원칙**: "청크 선행 → 질문 생성"(SPEC §4-2). 정답 청크를 먼저 뽑고 그 내용으로
질문을 만들기 때문에 라벨 매핑이 필요 없고, 정답·근거 값이 청크 원문에
**그대로 존재**하는지 코드로 기계 검증한다.

### 조합별 생성 방식 (3조합 × 50 = 150)

| 조합 | 방식 |
|---|---|
| single×정답형 | 무작위 청크 1개 → 그 청크만으로 답할 질문. 정답은 본문에 그대로 등장하는 핵심 구 |
| multi×정답형(bridge) | 청크 A의 서두 링크로 연결된 문서 B → 2단 질문. hop1 답 = 연결 엔티티(B 제목), gold = B 본문의 핵심 구. **B가 수집 분류 밖이면 서두를 추가 수집해 DB에 포함** (SPEC §4-3, 최종 32개) |
| multi×탐색형(comparison) | **같은 개체 유형**(영화-영화 / 인물-인물)이면서 **비교 근거 값(개봉·데뷔·출생 연도 등)이 양쪽 서두에 실존하는** 같은 하위 분류 쌍 → 비교 질문. 근거 값 2개를 hop_answers에 기록 |

### bridge 품질 파이프라인 (검수 3라운드를 거친 최종 형태)

생성 프롬프트 규칙 10종: A 명시·B 언급 금지 / 최종 답은 B에만 있는 사실 /
gold는 의문사와 유형이 일치하는 30자 이내 핵심 구(서술형 금지) / hop2는 인물·작품·
기관·장소 고유명사만(언어·일반 개념 전면 금지) / **답이 연결 엔티티 자신이 되는 질문
금지** / **유일 특정 한정어 필수**(연도·공동 출연자·장르 등, 단 gold와 겹치면 금지) /
전제 모순 금지.

코드 검증: hop1 답∈A, gold∈B, **gold∉A**, 핵심 구 형식, 질문에 gold 어절 노출 금지,
같은 hop1 문서 최대 2회, **hop2 문서 전면 중복 금지**.

LLM 3중 검증(모두 통과해야 채택):
1. **자기 검증 4판정** — A만으로 답 완결 여부 / 의문사-gold 유형 일치 /
   답이 B 자신인지 / 한정어가 대상을 유일하게 특정하는지
2. **hop2 유형 분류** — B 문서를 인물/작품/기관단체/장소/언어/일반개념으로 분류해
   앞 4개만 통과 (제목별 캐시)
3. **답변-대조 검증** — 별도 LLM이 A+B를 보고 실제로 답을 생성 → 그 답이
   B 자신이면 폐기, gold와 포함 관계로 정합하지 않으면 폐기

### 검수 워크플로

`eval/testset_review.csv`(엑셀 호환) 출력 → 사용자가 이상한 행 X 표시 →
해당 행만 재생성. 3라운드 검수를 거쳐 **150행 확정** (사용자 "검수 통과" 승인).
- 알려진 예외: idx 82·94는 hop2 중복이지만 대체 공급 부족(시도 100회 상한)으로
  사용자 허용 하에 유지. "신과함께-죄와 벌" 청크는 위키 원문 훼손이 발견되어
  hop1 사용을 차단.

### 라벨 스키마 (`eval/testset.jsonl`)

```json
{"question": "...", "combo": "single|bridge|comparison",
 "hop_type": null|"bridge"|"comparison",
 "answers": ["정답 청크 title", ...],
 "hop_answers": {"1": {"title": "hop1 청크", "answer": "hop1 답"},
                 "2": {"title": "hop2 청크", "answer": "근거 값"}},
 "gold_answer": "최종 정답"}
```

## 5. DB 구축 — scripts/build_db.py

- 수집 청크 + bridge 외부 문서 → title 기준 중복 제거(다르면 긴 쪽 보존+경고)
- 기존 컬렉션 **삭제 후** `metadata={"hnsw:space": "cosine"}`로 생성 (distance ∈ [0,2])
- 문단=청크, id=title, 메타데이터 {title}, 임베딩 배치 적재
- **유형별 무결성 체크**: single=gold∈정답 청크 / bridge=hop1 답∈hop1 청크 &
  gold∈hop2 청크 / comparison=근거 값∈각 청크
- **최종: 1,185청크 적재, 무결성 150/150 전 유형 통과 (실패 0)**

## 6. 평가 하네스 — eval/run_eval.py

- 하네스-에이전트 계약: 어떤 시스템이든 `evidence: [{"hop": n, "chunk_ids": [...]}]`
  포맷만 지키면 채점 가능 (naive든 agentic이든 동일 채점기).
- 채점: 라벨의 각 hop 정답 title이 evidence 목록 어디든 등장하면 hit,
  목록 내 순위로 RR. 질문 Hit = 모든 hop hit, 질문 MRR = hop RR 평균.
  전체·조합별·hop별로 집계.
- 옵션: `--system {naive|baseline|improved} --subset {48|150} --tag --top-k [--http]`
  (서브셋 48 = 조합별 16 층화, --http 시 /health 가드).
- 부가 지표: llm_calls 평균, top1_distance 분포, (agentic용) retry율·exhausted
  비율·Planner 분류 정확도 훅.
- 결과는 설정 스냅샷·질문별 상세와 함께 `eval/results/{일시}_{tag}.json` 저장.

## 7. agents/naive — 1-pass Baseline (동결 대상)

질문 임베딩 → top-k(5) 검색 → 생성 1회 (LLM 1회, hop/retry 없음).
"제공 문서만 근거, 없으면 '문서에서 확인할 수 없습니다', 출처 title 표기" 프롬프트.
evidence는 `[{"hop": 0, "chunk_ids": [...]}]`로 기록해 동일 채점기 사용.

## 8. backend/main.py — /health

`GET /health → {"status": "ok", "db_chunks": 1185, "space": "cosine"}`
(space 포함은 Day 1 추가 지시. /ask_naive·/ask는 Day 4에 추가 예정)

## 9. 확정 평가 결과 (naive, 150 전체)

| | n | Hit Rate | MRR |
|---|---|---|---|
| **전체** | 150 | **0.560** | **0.621** |
| single×정답형 | 50 | 0.92 | 0.84 |
| multi×정답형(bridge) | 50 | 0.38 | 0.50 |
| multi×탐색형(comparison) | 50 | 0.38 | 0.52 |
| hop1 기준 | 150 | 0.913 | 0.812 |
| hop2 기준 | 100 | 0.41 | 0.223 |

**해석**: 1-pass는 질문과 직접 닿는 hop1 청크는 91% 찾아내지만, 질문에 명시되지
않은 hop2 청크는 41%밖에 못 가져온다(순위도 낮음, MRR 0.22). multi-hop 조합의
Hit 0.38이 구조적 한계를 수치로 보여주며, 이것이 Agentic(계획→검색→판정→재작성/
hop전환)이 공략할 지점이다.

**PLAN Day 1 판정 기준 점검**: "전 조합 Hit Rate < 0.4"에 미해당(single 0.92)
→ 임베딩 한국어 성능 문제 없음, Day 5 임베딩 비교 실험 불필요.

## 10. GATE_THRESHOLD 후보 (보너스 과제)

cosine DB 구축 **후** 150개 질문의 top1_distance 분포를 측정:

min 0.238 · p50 0.489 · p75 0.545 · **p90 0.590** · **p95 0.617** · p99 0.690 · max 0.693

정상 질문의 95%가 0.62 이하 → 이보다 먼 검색은 "동떨어진 결과"로 판단할 근거.
**후보 확정: 0.60 / 0.62 / 0.65** — Day 2 Judge 문지기 초기값 및 Day 5 튜닝 그리드로 사용.

## 11. 산출물

| 경로 | 내용 |
|---|---|
| `core/llm.py`, `core/db.py` | 공유 모듈 |
| `scripts/collect_wiki.py` → `data/chunks.jsonl` | 수집 1,153청크 + 통계 |
| `scripts/build_testset.py` → `eval/testset.jsonl` | 검수 확정 150문항 (+검수 CSV, --regen) |
| `scripts/build_db.py` → `db/` | cosine 1,185청크 + 무결성 체크 |
| `eval/run_eval.py` → `eval/results/20260721_024656_naive_day1_final.json` | 하네스 + 확정 결과 |
| `agents/naive/pipeline.py`, `backend/main.py` | 1-pass baseline, /health |

## 12. 다음 단계 (Day 2 예고)

`core/state.py`(AgentState) + LangGraph 그래프 골격을 만들고, LLM 없는 가짜 노드로
4가지 흐름(정상/재작성 루프/사후 재계획/exhausted)을 먼저 검증한 뒤 Planner·검색·
Judge를 LLM화한다. RUNBOOK에 따라 **각 노드의 프롬프트 전문은 코드 작성 전에
사용자 승인**을 받는다. GATE_THRESHOLD 초기값은 위 후보 중에서 결정.
