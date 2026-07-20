# SPEC.md — Agentic RAG 설계서 (확정본)

목표: HotpotQA에서 1-pass RAG 대비 Agentic RAG의 검색·답변 품질 개선을 유형별 Hit Rate·MRR로 정량 증명.
데모: 질의 → 계획·재작성 이력·hop별 판정·반복 횟수·출처가 보이는 화면.

## 1. 그래프 토폴로지 (agents/baseline)

노드 6개: Planner → 검색 → Judge → [조건부 엣지] → {hop전환→검색 | Generator | Rewriter→검색}

조건부 엣지 (Judge 뒤 유일한 분기, 순수 함수 — state 읽기만):
- ① verdict=sufficient & 다음 hop 있음 → hop전환
- ② verdict=sufficient & 마지막 hop → Generator
- ③ verdict=insufficient & retry 남음 → Rewriter
- ④ verdict=insufficient & 한도 소진(exhausted) → Generator

상수: MAX_HOP=2, MAX_RETRY=2(hop별 리셋), LLM 호출 상한 20(assert), top-k 초기값 5(튜닝 대상).

## 2. state 스키마 (core/state.py)

```python
class AgentState(TypedDict):
    # 제어용
    judge_verdict: str; judge_source: str        # "gatekeeper"/"llm_judge"
    relevance: str; sufficiency: str             # "high"/"low"
    hop_index: int; retry_count: int
    exhausted: bool; exhausted_reason: str       # "retry"/"hop"/"budget"/"extract"/""
    llm_call_count: int
    # 작업용 (덮어쓰기)
    query: str                                   # 원본, 불변
    plan: dict    # {query_type: single_hop|multi_hop, hop_type: bridge|comparison|None,
                  #  search_queries: [...], reason: str}
    answer_strategy: str                         # "정답형"/"탐색형"
    current_hop_query: str
    search_results: list                         # [{id, title, text, distance}]
    top1_distance: float
    intermediate_answers: list
    judge_reason: str; missing: str
    # 기록용 (append만)
    tried_queries: list
    judge_history: list   # [{hop, verdict, source, relevance, sufficiency, reason}]
    evidence: list        # [{"hop": n, "chunk_ids": [...]}] — hop전환·Generator 진입 시 박제. 채점 대상.
    sources: list
    # 출력
    answer: str
```
make_initial_state(query): 전 필드 빈 값, current_hop_query=query, tried_queries=[query].

## 3. 노드별 명세

### Planner (LLM 1회)
- 읽기 query → 쓰기 plan, answer_strategy, current_hop_query(=search_queries[0])
- 출력 JSON: query_type, hop_type, search_queries(질문형→검색형 변환), answer_strategy, reason
- search_queries: bridge=[구체 질의, "{hop1} 포함 템플릿"] / comparison=[구체, 구체](템플릿 불필요)
- 전략 규칙: 비교·선택("vs","which","first","more" 등)=탐색형, 그 외=정답형
- 파싱 실패 → fallback {single_hop, None, [원본], "fallback"}, 정답형

### 검색 (LLM 0회, core/db.py 경유)
- current_hop_query 임베딩 → top-k → search_results, top1_distance

### Judge (LLM 0~1회, 설계 A: 문지기 내장)
- 문지기: top1_distance > GATE_THRESHOLD(day 1에 분포 측정 후 설정) →
  즉시 {insufficient, gatekeeper, relevance="low", sufficiency="low", 기계 문구 reason}
- 통과 시 LLM 판정. **기준점 = current_hop_query** (마지막 hop도 동일).
  출력: verdict, relevance, sufficiency, reason, missing
- 모순 칸 처리: 프롬프트에 "rel low면 suf 반드시 low" 명시 + 검증에서
  (rel low & suf high) 나오면 rel=high 교정 후 sufficient 전진
- **exhausted 판정·기록은 Judge 단일 책임**: insufficient이고
  (retry_count>=MAX_RETRY → "retry" / hop_index>=MAX_HOP → "hop") 시 exhausted=True 기록
- judge_history append. 파싱 실패 → 1회 재호출 → 재실패 시 sufficient 통과 + "parse_fail" 기록

### hop전환 (LLM 0~1회)
- bridge: LLM으로 중간 답 추출("문서에서 {hop질의}의 답만") → intermediate_answers append
  → search_queries[1]의 {hop1} 치환 → current_hop_query
- comparison: 추출 생략, search_queries[1] 그대로 current_hop_query
- 공통: hop_index+1, retry_count=0, evidence append({"hop": 이전hop, "chunk_ids": ...})
- 추출이 빈 문자열 → 1회 재시도 → 재실패 시 exhausted_reason="extract"로 Generator행

### Rewriter (LLM 1회)
- 모드: judge_source=gatekeeper → C 탐색적 전면 수정 / relevance=low → A 방향 전환 /
  rel high & suf low → B 겨냥 보강(missing 활용, **사후 재계획**: plan을 multi_hop으로 갱신 가능)
- 차기 retry가 마지막(=MAX_RETRY)이면 "과감한 전환" 지시 추가 (시도별 차등화)
- 프롬프트 입력: 원본 query(앵커), tried_queries(중복 방지), judge_reason, missing, 결과 발췌
- 쓰기: current_hop_query, tried_queries append, retry_count+1 (유일 증가 지점)
- 새 질의가 tried_queries에 있으면 1회 재요청

### Generator (LLM 1회)
- 프롬프트 2×2: (정답형/탐색형) × (정상/exhausted). 탐색형=근거 값 나열 후 비교 결론.
  공통: "제공 문서만 근거, 없으면 '문서에서 확인할 수 없습니다'" + title 출처 표기
- evidence append(최종 search_results), sources 기록, answer 작성

### agents/naive (1-pass)
- 임베딩 → top-k 검색 → 생성 1회. hop/retry 없음. LLM 1회.
- 결과에 동일 evidence 포맷([{"hop":0, "chunk_ids":[...]}]) 기록 → 같은 채점기 사용.

## 4. 데이터·테스트셋 (scripts/)

1) HotpotQA dev distractor 다운로드
2) build_dataset.py: bridge 50 + comparison 50 선별
   (answer 존재, supporting_facts 정확히 2문단, 문단 길이 정상)
3) derive_singlehop.py: 같은 bridge 50의 hop1 문단으로 LLM 질문화 → 검수 CSV 출력
   → 사용자가 이상한 행 X 표시 → X만 재생성. (multi와 같은 원천 재활용 — 발표 시 비독립성 명시)
4) build_db.py: 150개 질문 context 문단 전부 → title 기준 중복 제거(같은 title=같은 텍스트,
   다르면 긴 쪽 보존+경고) → 문단=청크, id=title, 메타데이터 {title} → ChromaDB(./db)
   컬렉션 생성 시 metadata={"hnsw:space": "cosine"} 지정 (distance = cosine, 범위 [0,2])
   → 무결성 체크: 모든 supporting_facts의 (title, sent_idx)가 해당 청크 텍스트에 존재 확인
5) 라벨: eval/testset.jsonl — {question, combo, hop_type, answers(title set),
   hop_answers(hop별), gold_answer}. supporting_facts title = 청크 id 직접 매칭.
- 테스트셋 3조합 × 50 = 150: multi×정답형(bridge) / multi×탐색형(comparison) / single×정답형(파생)
- 청크 id 규칙: 문단 모드 title / 문장 모드(튜닝 실험 시) title::sent_idx.
  라벨 변환 함수는 chunk_mode 파라미터로 두 모드 지원.

## 5. 평가 하네스 (eval/run_eval.py)

- 옵션: --system {naive|baseline|improved} --subset {48|150} --tag NAME [--http]
- 서브셋 48 = 조합별 16개 층화 추출(튜닝용). 확정 측정은 150.
- 채점: final evidence의 hop별 chunk_ids vs hop_answers → 전체·조합별·hop별 Hit Rate, MRR
- 부가: llm_call_count 평균, retry 발생률, exhausted 비율(reason별),
  Planner 분류 정확도(hop_type·answer_strategy vs 라벨)
- 기록: eval/results/{일시}_{tag}.json (설정 스냅샷 포함) → git commit
- --http 시 /health 가드

## 6. API·화면 계약

- GET /health → {status, db_chunks} / POST /ask_naive {question, top_k} /
  POST /ask {question, top_k} (max_retry 서버 고정 2)
- /ask 응답: answer, strategy, plan, rewrite_history, judge_history(2×2 값 포함),
  intermediate_answers, retry_total, hop_reached, exhausted(+reason), llm_calls,
  sources[{hop, titles}], elapsed_sec
- /ask_naive 응답: answer, sources, llm_calls, elapsed_sec
- Streamlit 탭 2개: ① Agentic 단독(입력+top_k 슬라이더 → 답변+전략 뱃지+출처,
  expander: 계획/hop별 판정 표/재작성 이력/중간 답/통계, exhausted 경고 박스)
  ② 비교(같은 질문 → naive|agentic 좌우). 공통: /health 사전 확인, session_state 유지.

## 7. 확정 결정 요약 (근거는 노션 페이지 참조)

LLM: gpt-4o-mini(전 노드 동일) / 임베딩: text-embedding-3-small / distance: cosine /
한 리포 + agents/{naive, baseline, improved} + 공유 core/ + git tag 박제 /
temperature=0 / k 초기 5 / 판정 유효 3칸 / 파싱 재실패=sufficient 통과 /
추출 재실패=exhausted("extract") / re-ranking은 day 6~7 개선 1순위 후보 /
보존 아이디어: 청크별 판정·k분리·중간답 하이브리드·모델 차등화.
