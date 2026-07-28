# Agentic RAG — 한국어 위키 영화 도메인

한국어 위키백과 영화·인물 문서(19,639건)를 코퍼스로, 1-pass RAG 대비 Agentic RAG(LangGraph 다중 노드 + 정형 지식층 + 재질문 명료화)의 검색·답변 품질 개선을 **수치로 증명**하는 프로젝트입니다. 세 시스템(naive 1-pass / baseline Agentic / v2 구조 개선판)을 동일 코퍼스·동일 테스트셋(50문항, 유형 혼합)에서 Hit Rate·MRR로 비교합니다. 모호 문항은 하네스의 오라클 루프가 사용자 대리로 되묻기에 응답합니다.

**최종 결과** (50문항, k=10, v2 코퍼스, 정답=청크 집합·하나라도 적중 시 hit):

| 시스템 | 전체 Hit / MRR | 단일(5) | 연쇄(10) | 비교(10) | 목록(12) | 모호(13) |
|---|---|---|---|---|---|---|
| naive (1-pass) | 0.26 / 0.151 | 0.20 | 0.00 | 0.40 | 0.33 | 0.31 |
| baseline (Agentic) | 0.52 / 0.361 | 0.40 | 0.60 | 1.00 | 0.33 | 0.31 |
| **v2 (개선판)** | **0.88 / 0.715** | **1.00** | **0.70** | **1.00** | **1.00** | **0.77** |

(유형 칸은 Hit. 상세 수치·문항별 기록은 `eval/results/*.json`)

## 아키텍처

```
질문 ─→ [명료화 훅(v2 전용, 발동 시 되묻기 카드로 조기 종료)]
      └→ Planner ──(list)──→ List 조회(필모·분류 색인, LLM 0회) ──→ Generator
             │
             └─(그 외)─→ 검색(이중 갈래: 전역 top-k + 개체 title 필터
                          + 섹션 우선 + 인포박스·필모 정형 조회)
                              └→ Judge ──충분──→ Generator
                                   ├─(bridge 다음 hop)→ hop 전환(인포박스 폴백
                                   │        LLM 0회 → LLM 추출) → 검색 → Judge …
                                   └─불충분──→ Rewriter(재작성·재계획) → 검색
                                              (hop별 재시도 ≤ 2, 소진 시 정직 실패)
```

데이터는 3층입니다. **1층** 섹션 청크 56,546개(모든 문서 ≥1청크, 100자 미만 서두는 인포박스 핵심 필드 합성 서두)를 ChromaDB(`./db_v2`, cosine, HNSW efc200/M32/ef_search2000)에 임베딩 적재하고, **2층** 인포박스 정형 레코드 18,576건(문서의 94.6%), **3층** 필모그래피 색인(인물 14,047명 · 작품 항목 313,040건 — 배우 문서 필모 섹션 + 영화 인포박스 출연 필드 역인덱스)과 분류 색인(11,820개 분류)을 JSON으로 둡니다. 정형 조회 적중은 원본 청크 id로 환산해 evidence에 선두 배치합니다(정확 일치 우선).

## 설치·실행

API 키 없이 결과만 보려면 [비용 고지](#비용-고지) 절을 참조하세요.

```bash
git clone https://github.com/letscodedirty/agentic-rag.git && cd agentic-rag
python3 -m venv venv && source venv/bin/activate   # Python 3.12 기준
pip install -r requirements.txt
```

`.env` 파일을 저장소 루트에 작성합니다:

```ini
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
BACKEND_URL=http://localhost:8000
```

**DB 복원** — 대용량 원본은 gzip으로 커밋돼 있습니다. 압축 해제 후 무결성을 확인하고 임베딩 적재를 실행합니다(임베딩 API 비용 약 $0.6, 약 10~20분):

```bash
gunzip -k data/v2/chunks.jsonl.gz          # 1층 청크 원본 복원
sha256sum data/v2/chunks.jsonl             # 기대값: 8251cdf4b2016550a2258253adafd4a2e65835a6825c4838668d09e5d2e4fb0d
python scripts/build_db_v2.py              # ./db_v2 생성 (56,546청크 임베딩)
```

(`data/v2/pages_snapshot.jsonl.gz`는 수집 원문 스냅샷 — 평가 재현에는 불필요, 해시는 `data/v2/README.md` 참조)

**(선택) 서버 데모** — 웹 UI의 naive·baseline 탭과 `/health`는 v1 DB(`./db`)를 사용하므로 데모를 띄우려면 v1 DB도 구축합니다(1,185청크, 비용 미미):

```bash
python scripts/build_db.py                 # ./db 생성 (v1)
uvicorn backend.main:app --port 8000       # FastAPI 백엔드
streamlit run frontend/app.py              # 웹 UI (탭: v2/baseline/naive)
```

## 평가 재현

세 시스템 모두 v2 코퍼스를 검색하도록 주입해 실행합니다(`--corpus v2`). v2는 명료화 ON + 오라클 루프(모호 문항은 intended_query를 아는 오라클 LLM이 되묻기에 응답 후 재실행)입니다.

```bash
python eval/run_eval.py --system naive    --subset v2 --corpus v2 --top-k 10
python eval/run_eval.py --system baseline --subset v2 --corpus v2 --top-k 10
python eval/run_eval.py --system v2       --subset v2 --corpus v2 --top-k 10
```

결과는 `eval/results/<타임스탬프>_<태그>.json`에 저장됩니다(전체·유형별 지표, 명료화 발동·오라클 기록, 문항별 채점). 3종 합계 LLM 비용 약 $1, v2는 오라클 루프 포함 약 10분 소요됩니다. 테스트셋은 `eval/testset_v2.jsonl`(50문항, 라벨=정답 청크 집합, 모호 문항은 intended_query 포함)입니다.

## 한계

명료화는 다축 해석형(동명작·동명이인 등 해석이 갈리는 질문)을 포착하도록 설계돼, 해석은 하나이고 범위만 넓은 광범위형("재밌는 영화 추천해줘")은 원리상 미포착입니다. 비교형 명확 질문에서 의도 표현 변주로 엔트로피가 부풀어 오발동할 수 있고(실측 1건), 인물 동명이인은 재구성 질문에 연도가 실리지 않아 재발동할 수 있습니다. 최상급 시간 질문("제일 최근작")은 문서 서술 인용에 의존해 코드 집계(argmax) 미도입 상태이며, 평가의 오라클은 사용자 대리로서 불완전합니다(50문항 중 오선택 1건 관측). τ=1.05·동치 임계 0.85는 소표본(명확 18문항) 실측 기반입니다. 상세는 `docs/SPEC.md` §8 한계 문구 참조.

## 저장소 구조

```
CLAUDE.md            프로젝트 헌법(절대 규칙)
docs/                SPEC.md(설계)·PLAN.md(일정)·일자별 작업보고서
core/                공유 계층: llm.py(call_llm 단일 경유)·db.py(ChromaDB)·state.py
agents/naive/        1-pass 직선 파이프라인 (비교 기준선, 동결)
agents/baseline/     기본 Agentic 그래프 (v1-baseline 태그로 동결)
agents/v2/           구조 개선판: nodes(7노드)·knowledge(2·3층)·clarify(명료화)
scripts/             수집·3층 산출·DB 적재 스크립트
data/, data/v2/      수집 스냅샷·3층 데이터(대용량은 gzip)
eval/                run_eval.py(하네스+오라클 루프)·testset_v2.jsonl·results/
backend/             FastAPI (접수-위임-반환)
frontend/            Streamlit 3탭 UI (되묻기 카드 포함)
config/              baseline.yaml (top_k, gate_threshold)
```

## 비용 고지

실행에는 OpenAI API 키가 필요합니다. 예상 비용: **DB 구축 약 $0.6**(임베딩 1회), **평가 3종 약 $1**(gpt-4o-mini). 키 없이도 확인할 수 있는 것: `eval/results/`의 전체 평가 결과 JSON(시스템별 지표·문항별 채점·명료화 기록 전부 포함), `docs/`의 설계서(SPEC)·일정(PLAN)·일자별 작업보고서, `data/v2/build_stats.json` 등 데이터 통계.
