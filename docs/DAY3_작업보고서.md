# Day 3 작업 보고서 — Agentic baseline 완성 (hop전환·Rewriter·Generator) + 150 평가

> 2026-07-21 · 커밋 `f50b74a` · 근거 문서: SPEC.md §1·§3(개정 포함), PLAN.md Day 3

## 요약

남은 3개 노드(hop전환·Rewriter·Generator)를 프롬프트 전문 승인 방식으로 구현해
**baseline Agentic 그래프를 완성**했고, 통제 케이스 6종 검증을 모두 통과시킨 뒤
150 전체 평가를 수행했다. 결과는 **전체 Hit 0.560 → 0.820, MRR 0.621 → 0.812**로
naive 대비 **전 조합 개선** — Day 3 완료 기준 충족. 특히 1-pass의 구조적 약점이던
hop2 검색이 0.41 → 0.79로 뛰었다. 최대 잔여 약점은 bridge(0.56)로, Day 5~6의
1순위 공략 지점이다.

---

## 1. SPEC 개정 (구현 전 확정한 판단 2건)

구현 중 SPEC 조항 간 긴장을 발견해 멈추고 보고 → 사용자 승인으로 SPEC에 반영했다.

1. **예외 엣지**: §1의 "hop전환→검색" 고정 엣지와 §3의 "추출 재실패 시 Generator행"이
   충돌 → hop전환 뒤 조건부 엣지 추가(`exhausted_reason=='extract'`면 Generator,
   아니면 검색). 판정 기반 분기가 아닌 실패 처리이므로 "Judge 뒤 유일 분기" 조항과
   구분된다. 이 엣지도 state 읽기만 하는 순수 함수다.
2. **comparison의 hop1 문서 확보**: 탐색형 답변은 양쪽 근거 값이 필요한데
   `search_results`는 hop마다 덮어써서 Generator 시점엔 hop2 문서만 남는다 →
   Generator가 evidence에 박제된 hop1 chunk_ids를 `fetch_chunks()`(core/db 경유,
   검색 계층 한정)로 재조회해 문서에 포함.

추가 반영: 추출 재실패 시 **공통 쓰기(hop_index+1, evidence·sources append) 생략** —
전환이 일어나지 않았으므로 evidence에는 Generator가 박제하는 hop0 항목만 정확히
1개 남는다. exhausted 기록의 단일 책임(Judge)에 "extract만 hop전환이 기록" 예외 명시.

## 2. 구현된 3개 노드 (최종 형태)

### hop전환 (LLM 0~1회)
- **bridge**: hop1 검색 결과에서 중간 답만 추출(승인 프롬프트: "이름/값만, 30자 이내,
  없으면 빈 문자열") → `intermediate_answers` append → `search_queries[1]`의 `{hop1}`
  치환 → hop_index+1, **retry_count=0(유일 리셋)**, evidence·sources append.
- **comparison**: 추출 생략, `search_queries[1]` 그대로 (LLM 0회).
- 빈 답 → 1회 재시도 → 재실패 시 공통 쓰기 생략 + `exhausted=True/reason="extract"`
  → 예외 엣지로 Generator행.

### Rewriter (LLM 1회, 3모드)
모드 선택은 코드가 한다:
| 모드 | 조건 | 지시 요지 |
|---|---|---|
| C 탐색적 전면 수정 | judge_source=gatekeeper | 다른 개체명·표현·관점으로 전면 재작성 |
| A 방향 전환 | relevance=low | 대상을 정확히 특정하는 표현으로 교정 |
| B 겨냥 보강 | rel high & suf low | missing을 정면 겨냥 + **사후 재계획**(부족 정보가 다른 문서에 있으면 `{hop1}` 템플릿과 함께 multi_hop 전환 제안) |

- 입력: 원본 질문(앵커), tried_queries(중복 방지), judge_reason, missing, 결과 발췌.
- 쓰기: current_hop_query, tried_queries append, **retry_count+1(유일 증가 지점)**.
- 차기 retry가 마지막이면 "과감한 전환" 지시 추가. 새 질의가 tried_queries와 중복이면
  1회 재요청. 사후 재계획은 모드 B & single_hop & hop0 & 유효 `{hop1}` 템플릿일 때만
  코드 가드로 반영.

### Generator (LLM 1회, 2×2)
(정답형/탐색형) × (정상/exhausted) 4칸 지시. 공통: 제공 문서만 근거, 없으면
"문서에서 확인할 수 없습니다", (출처: title) 표기. 탐색형은 근거 값 나열 후 비교 결론,
exhausted 칸은 확정 답 대신 확인된 것/못 한 것을 구분해 한계를 밝힌다.
쓰기: evidence append(최종 search_results), sources append, answer.

## 3. 통제 케이스 검증 6종 (전부 통과)

| # | 케이스 | 결과 |
|---|---|---|
| ① | Rewriter 모드 분기 3종 | 프롬프트 스파이로 모드별 지시 삽입 확인, 모드 취지대로 질의 변화 |
| ② | 모드 B 사후 재계획 | 엔티티 기지(旣知)면 직접 질의(옳음), 미상이면 replan 발생 — `{hop1}` 템플릿 보존 |
| ③ | 중복 재요청 | 중복 시 정확히 2회 호출 + 중복 경고 프롬프트 + 최종 미중복 |
| ④ | 추출 실패 경로 | 2회 시도 → exhausted="extract" → 공통 쓰기 생략 → Generator행, evidence hop0 정확히 1개·중복 없음 |
| ⑤ | Generator 2×2 | 4칸 모두 형식 준수 (탐색형: 값 나열→결론, exhausted: "비교 불가" 등 한계 명시) |
| ⑥ | bridge 완주 스모크 | 러브픽션 감독 학교 질문: 중간답 '전계수' → 최종 '서강대학교'(gold 일치), evidence 2-hop, LLM 5회, 10.2초 |

## 4. 150 전체 평가 결과 (핵심)

결과 파일: `eval/results/20260721_112613_baseline_day3.json`

### 4-1. naive 대비 비교표

| | Hit Rate (naive→baseline) | MRR (naive→baseline) |
|---|---|---|
| **전체** | 0.560 → **0.820** (+0.260) | 0.621 → **0.812** (+0.192) |
| single | 0.920 → 0.980 (+0.060) | 0.840 → 0.921 (+0.081) |
| bridge | 0.380 → 0.560 (+0.180) | 0.502 → 0.625 (+0.123) |
| comparison | 0.380 → **0.920** (+0.540) | 0.520 → 0.891 (+0.371) |
| hop1 분해 | 0.913 → 0.907 (−0.007) | 0.812 → 0.857 (+0.044) |
| hop2 분해 | 0.410 → **0.790** (+0.380) | 0.223 → 0.691 (+0.467) |

**읽는 법**: 1-pass는 질문에 직접 닿는 hop1 문서는 잘 찾지만(0.91) 두 번째 문서를
못 가져오는(0.41) 구조적 한계가 있었다. Agentic은 계획 분해 + hop전환 + 재작성으로
hop2를 0.79까지 끌어올렸다. comparison이 +0.54로 최대 수혜 — 두 대상을 질의 2개로
분리한 효과가 그대로 수치에 나타났다. bridge는 +0.18 개선됐지만 0.56로 최약.

### 4-2. Planner 분류 정확도
- **hop_type(+query_type) 89.3%** (134/150), **answer_strategy 98.7%** (148/150)
- 오분류 패턴: 16건 중 15건이 **single 인물 질문을 bridge로 과분해**
  ("어떤 대학 졸업?", "어떤 드라마로 데뷔?" — 한 문서로 답할 수 있는데 2단으로 계획).
  과분해가 곧 오답은 아니지만(hop1에서 답을 찾으면 정답 처리) 역전 사례의 원인이 됨.
- 전략 오분류 2건: "중에서/누구" 표현이 탐색형으로 오인 (#31, #88).

### 4-3. 부가 지표
- llm_calls **평균 5.19 / 최대 13** — 상한 20 대비 여유 7. assert 발동 0건.
- retry 발생률 **32.7%** (49개 질문, 총 87회 재작성) — Rewriter가 실질 작동.
- exhausted **19.3%** (retry 28건, extract 1건) — 재작성 2회로도 못 찾은 질문들.
- 평균 소요 9.4초/질문 (최대 119초 1건: 재작성 루프 + API 지연 중첩).

### 4-4. 실패 분석 (Day 5~6 재료)
- **오답 분포**: single 1 / **bridge 22** / comparison 4.
- **bridge 오답의 지배적 패턴은 hop2 miss**: 중간 답 추출까지는 성공했는데 hop2
  질의("{중간답} 속성")가 정답 청크를 검색에서 못 잡는 경우가 대부분. re-ranking
  (Day 6 1순위 후보)이 정확히 이 지점을 겨냥한다.
- **역전 사례 5건** (naive 정답 → baseline 오답: #61, #72, #77, #86, #130):
  Planner가 질문을 분해하면서 hop1 질의가, 질문 전문을 통째로 검색하던 naive라면
  잡았을 문서를 놓친 경우. "분해 이득 vs 원문 검색 이득"의 트레이드오프 사례.
- **앵커 이탈 0건**: Rewriter 발동 49건 전수 검사(한국어 조사 제거 정규화 후 토큰
  대조) 결과 재작성 질의가 원본 질문의 대상을 잃은 사례 없음. Day 2에서 인위적
  시나리오로 관찰됐던 모드 A 이탈은 실전에서 재현되지 않았다.

## 5. 산출물

| 경로 | 내용 |
|---|---|
| `docs/SPEC.md` | 개정 4곳 (예외 엣지, extract 예외, 공통 쓰기 생략, fetch_chunks) |
| `agents/baseline/nodes.py` | hop전환·Rewriter·Generator 추가 (전 6노드 완성) |
| `agents/baseline/graph.py` | 예외 엣지(순수 함수) + `run_agent()` 진입점 |
| `core/db.py` | `get_by_ids()` (comparison hop1 재조회용) |
| `eval/run_eval.py` | 질문별 elapsed·rewrite_history 기록 추가 |
| `eval/results/20260721_112613_baseline_day3.json` | 150 평가 결과 (질문별 상세 포함) |

## 6. 다음 단계 (Day 4 예고)

backend/main.py에 `/ask_naive` → `/ask` 추가(접수-위임-반환만), frontend/app.py에
Streamlit 탭 2개(Agentic 단독 + naive 비교). 완료 기준: 브라우저에서 같은 질문의
naive vs agentic 비교 화면 동작.
