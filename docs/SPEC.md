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
- 예외 엣지: hop전환 뒤 exhausted_reason=='extract'면 Generator, 아니면 검색
  (판정 기반 분기가 아닌 실패 처리 — Judge 뒤 유일 분기 조항은 판정 분기에 한함)

상수: MAX_HOP=2, MAX_RETRY=2(hop별 리셋), LLM 호출 상한 20(assert), top-k 초기값 5(튜닝 대상) GATE_THRESHOLD=0.70 고정(day 1 정상 질의 분포의 관측 최대 0.693 초과 기준, 튜닝 비대상).

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
- **exhausted 판정·기록은 Judge 단일 책임(예외: 'extract'만 hop전환이 기록)**: insufficient이고
  (retry_count>=MAX_RETRY → "retry" / hop_index>=MAX_HOP → "hop") 시 exhausted=True 기록
- judge_history append. 파싱 실패 → 1회 재호출 → 재실패 시 sufficient 통과 + "parse_fail" 기록

### hop전환 (LLM 0~1회)
- bridge: LLM으로 중간 답 추출("문서에서 {hop질의}의 답만") → intermediate_answers append
  → search_queries[1]의 {hop1} 치환 → current_hop_query
- comparison: 추출 생략, search_queries[1] 그대로 current_hop_query
- 공통: hop_index+1, retry_count=0, evidence append({"hop": 이전hop, "chunk_ids": ...})
- 추출이 빈 문자열 → 1회 재시도 → 재실패 시 공통 쓰기(hop_index+1·retry 리셋·
  evidence·sources append) 생략, exhausted=True + exhausted_reason="extract"로 Generator행

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
- comparison일 때 evidence의 hop1 chunk_ids를 fetch_chunks(id 조회, core/db 경유)로
  재조회해 문서에 포함

### agents/naive (1-pass)
- 임베딩 → top-k 검색 → 생성 1회. hop/retry 없음. LLM 1회.
- 결과에 동일 evidence 포맷([{"hop":0, "chunk_ids":[...]}]) 기록 → 같은 채점기 사용.

## 4. 데이터·테스트셋 (scripts/) — 한국어 위키 영화 도메인 (개정)

언어: 데이터·질문·정답 전부 한국어. 원천: 한국어 위키피디아.

1) collect_wiki.py: 분류 "분류:대한민국의 영화"(+하위 분류)에서 문서 목록 수집(위키 API)
   → 각 문서의 서두 추출. 서두 = 문서 시작~첫 섹션 제목 전 도입부 전체를 청크 1개로.
   서두 200자 미만 문서 제외. 서두의 하이퍼링크 목록도 함께 저장(bridge 재료).
2) build_testset.py: "청크 선행 → 질문 생성"(LLM, 한국어). 정답 청크 id가 생성
   입력이므로 라벨 매핑 불필요 — answers/hop_answers/gold_answer를 생성 시 직접 기록.
   - single×정답형 50: 무작위 청크 1개 → 그 청크만으로 답할 질문
   - multi×정답형(bridge) 50: 청크 A의 하이퍼링크로 연결된 문서 B의 서두 청크 쌍
     → 2단 질문 (hop1 답 = A→B 연결 엔티티)
   - multi×탐색형(comparison) 50: 같은 범주 청크 쌍 → 비교 질문 (근거 값을 hop_answers에 기록)
   - 전 조합 검수 CSV 출력 → 사용자가 이상한 행 X 표시 → X만 재생성
3) bridge 외부 문서 포함 규칙: 채택된 bridge 쌍의 문서 B가 수집 분류 밖이면
   그 서두 청크도 DB에 포함 (hop2 정답 청크의 실존 보장).
4) build_db.py: 수집 청크 전부(+bridge 외부 문서 청크) → title 기준 중복 제거(같은
   title=같은 텍스트, 다르면 긴 쪽 보존+경고) → 문단=청크, id=title, 메타데이터 {title}
   → ChromaDB(./db), 컬렉션 생성 시 metadata={"hnsw:space": "cosine"} (distance=[0,2])
   → 무결성 체크(유형별):
     single = gold_answer가 정답 청크 텍스트에 존재
     bridge = hop1 답이 hop1 청크에, gold_answer가 hop2 청크에 존재
     comparison = 근거 값(hop_answers)이 각 청크에 존재 (gold_answer 자체는 검사 제외)
5) 라벨: eval/testset.jsonl — {question, combo, hop_type, answers(title set),
   hop_answers(hop별), gold_answer}
- 테스트셋 3조합 × 50 = 150: multi×정답형(bridge) / multi×탐색형(comparison) / single×정답형
- 청크 id 규칙 유지: 문단 모드 title / 문장 모드(튜닝 실험 시) title::sent_idx.
  라벨 변환 함수는 chunk_mode 파라미터로 두 모드 지원.
- 삭제: derive_singlehop.py (생성 방식에 흡수). Planner 분류 정확도 대조는 유지
  (hop_type·answer_strategy 라벨이 생성 시 기록되므로).

## 5. 평가 하네스 (eval/run_eval.py)

- 옵션: --system {naive|baseline|improved} --subset {dev|150} --tag NAME [--http]
- 튜닝은 독립 dev셋 38(eval/devset.jsonl, single 16/bridge 6/comparison 16)에서만
  수행 — 본 테스트셋과 동일 생성 파이프라인, 기존 DB 청크 한정, 기존 150과 질문
  중복 금지. 기존 150(eval/testset.jsonl)은 확정 측정 전용(데이터 누수 방지).
  bridge는 in-DB 후보 고갈로 6행(재사용 1 포함 — 질문·정답 상이 강제, provenance
  기록) — k 튜닝에서 방향 참고용이며 주 신호는 single·comparison 32문항 + 전체 Hit.
  best k 동률 규칙: bridge 차이 1~2건은 동률 취급, 동률이면 llm_calls 적은 쪽.
- 시스템 간 비교표는 동일 top_k로 측정(v1 이후 k=10 통일).
  day 1 naive@k5는 초기 기준선으로 보존.
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
- Streamlit 탭 2개: ① Agentic 단독(시스템 선택기: baseline|improved — day 7 전에는
  baseline만, day 7 후 기본값 improved. 입력+top_k 슬라이더 → 선택 시스템의
  답변+전략 뱃지+출처, expander: 계획/hop별 판정 표/재작성 이력/중간 답/통계
  + improved 고유 정보 패널 추가 가능, exhausted 경고 박스)
  ② 비교(같은 질문 → naive|agentic 좌우). 공통: /health 사전 확인, session_state 유지.
  day 7 v2 완성 후: POST /ask_v2 추가(응답 계약 /ask의 상위집합 — clarification·
  list 필드 추가, 기존 계약 무변경), 단독 탭 선택기(baseline|v2, 기본 v2),
  비교 탭을 naive|baseline|v2 3열로 확장(입력·실행은 단일). 명료화 응답 시
  화면은 답변 대신 예시 버튼들을 렌더링(클릭=입력창 채움→재제출).

## 7. 확정 결정 요약 (근거는 노션 페이지 참조)

LLM: gpt-4o-mini(전 노드 동일) / 임베딩: text-embedding-3-small / distance: cosine /
한 리포 + agents/{naive, baseline, improved} + 공유 core/ + git tag 박제 /
temperature=0 / k 초기 5 / 판정 유효 3칸 / 파싱 재실패=sufficient 통과 /
추출 재실패=exhausted("extract") / day 6~9는 v2 트랙으로 확정(사용자 결정, 기존 improved 우선순위 대체):
v2 데이터 구축 → agents/v2 신설 → 명료화 → v2 테스트셋 → 3시스템 비교.
re-ranking·본문 청크 등 기존 후보는 v2 설계에 흡수 또는 보존 아이디어 유지.

## 8. v2 트랙 (상세: docs/V2_DESIGN.md=데이터, docs/V2_AGENT.md=에이전트)

- 데이터: 영화+배우 분류 트리 합집합 19,639 문서 전문 → 3층 저장.
  ① 섹션 청크(id=title::섹션, 긴 섹션 문단 분할·잔섹션 병합. 문서 제외 없이
  전 문서 수록 — 청크 하한 100자는 청크 생성 기준으로만 적용, 100자 미만
  서두는 인포박스 핵심 필드 합성 서두로 수록, 메타데이터 {title, doc_type,
  section, categories}) ② 인포박스 정형 레코드
  ③ 필모그래피·분류 색인(배우 섹션 + 영화 인포박스 출연진 역인덱스, 출처 기록).
  ./db_v2 별도(./db 무변경), 원문 스냅샷 git 추적.
- agents/v2 (baseline·naive 동결, core는 추가만): 확정 결정 6건 —
  (1) query_type에 list 신설(여러 항목 요구 유형): Judge 없이 색인 조회→Generator
  직행, 조회 빈손이면 exhausted_reason="list_miss".
  (2) 개체명 인식 = Planner의 plan.entities 1차 + 전 문서 제목 사전 매칭 보정.
  (3) Judge 입력 = 1층 청크 전문 전량 + structured_results (발췌 금지). LLM은
  gpt-4o-mini 유지.
  (4) hop전환 추출 2단 폴백: 질의↔인포박스 필드 매칭 시 값 직접 채용(LLM 0회)
  → 실패 시 병합 묶음에서 LLM 추출. 최종 답변 문장화는 항상 Generator.
  (5) Generator 3×2 = (정답형/탐색형/목록형)×(정상/exhausted), 정상 칸은 구성
  지시형(핵심 답→근거 상세→인포박스 부가→출처), exhausted 칸은 앞 hop 문서
  fetch_chunks 재조회 포함 + 확인분 상세 + 한계 명시. 문서 밖 서술 금지 유지.
  (6) k=10, GATE_THRESHOLD=0.70 이월(추후 튜닝 시 갱신). 문지기 이원화:
  1층 거리 기준 hop 생사 판정하되 2·3층 적중 시 hop 유지 + 미달 1층 청크는
  병합 제외, 양층 빈손일 때만 즉시 실패.
- state 추가 필드(추가만): plan.entities, structured_results, list_results,
  clarification. evidence에는 2·3층 적중분도 원본 청크 id로 환산해 박제.
- 명료화(구현은 agents/v2 완성 후 상세 확정): 무상태 2회 실행 — Planner 앞
  노드가 애매성 사유 분류(보수적 발동), DB 실존 문서 접지 예시 생성,
  clarification 기록 후 조기 종료. 재입력은 새 실행.
- core 공유 규칙: core/state.py·db.py 등은 새 함수·새 필드 추가만 허용,
  기존 함수·필드의 동작·의미 변경 금지. v2 전용 로직·프롬프트는 agents/v2에.
- 평가: v2 테스트셋(기존 3유형+list 등 확장, 라벨=정답 청크 집합 — 하나라도
  적중이면 hit). 3시스템 전부 v2 1층을 검색하도록 DB 경로 config 주입(코드
  동결 유지). v1 결과(v1-baseline)는 서두 코퍼스 기준 기록으로 보존.