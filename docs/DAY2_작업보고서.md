# Day 2 작업 보고서 — Agentic 그래프 골격 + 앞 3노드 (Planner·검색·Judge)

> 2026-07-21 · 커밋 `6c3f967` · 근거 문서: SPEC.md §1~§3, PLAN.md Day 2, RUNBOOK Day 2

## 요약

LangGraph 기반 Agentic 그래프의 **골격(노드 6개 + 조건부 엣지 4분기)**을 만들고,
LLM 없는 가짜 노드로 **4가지 흐름 시나리오를 먼저 검증**한 뒤, 앞 3개 노드
(Planner·검색·Judge)를 실제로 구현했다. 프롬프트는 RUNBOOK 절차대로 전문 승인을
받았고, 통합 검증에서 **Judge의 "원본 질문 유출" 결함을 발견·수정**(수정본 재승인)
해 4개 검증 케이스를 모두 통과했다. GATE_THRESHOLD는 방침 변경에 따라
**0.70 고정(튜닝 비대상)**으로 반영했다.

---

## 1. GATE_THRESHOLD 방침 변경 반영

- 기존: Day 1에서 후보 0.60/0.62/0.65를 뽑아 Day 5에서 튜닝할 예정이었다.
- 변경(SPEC §1·PLAN Day 5 개정): **0.70 고정, 튜닝 비대상.**
  근거는 Day 1 정상 질의 150개의 top1_distance **관측 최대 0.693을 초과**하는 값 —
  정상 질문은 절대 차단되지 않으면서, 그보다 먼 검색만 걸러낸다.
- 구현: [config/baseline.yaml](config/baseline.yaml)에 `gate_threshold: 0.70` 고정,
  [core/config.py](../core/config.py) 로더 제공. Day 5 튜닝 그리드에는 top_k만 남음.

## 2. core/state.py — AgentState (SPEC §2 그대로)

- 필드 4그룹: **제어용**(judge_verdict, hop_index, retry_count, exhausted,
  llm_call_count 등) / **작업용**(query, plan, current_hop_query, search_results 등,
  덮어쓰기 허용) / **기록용**(tried_queries, judge_history, evidence, sources —
  **append만, 덮어쓰기 금지**) / **출력**(answer).
- `make_initial_state(query)`: 전 필드 빈 값 + `current_hop_query=query`,
  `tried_queries=[query]`.
- append 규칙 구현 방식: 노드가 `기존 리스트 + [새 원소]`의 **새 리스트를 반환**
  (LangGraph 상태 병합과 충돌 없이 append-only 보장).

## 3. 그래프 골격 — agents/baseline/graph.py

```
Planner → 검색 → Judge → [조건부 엣지]
                            ├ ① sufficient & 다음 hop 있음 → hop전환 → 검색(재진입)
                            ├ ② sufficient & 마지막 hop    → Generator → END
                            ├ ③ insufficient & retry 남음  → Rewriter → 검색(재진입)
                            └ ④ insufficient & 한도 소진   → Generator → END
```

- 분기는 Judge 뒤 **단 한 곳**, `route_after_judge(state)`는 state를 **읽기만 하는
  순수 함수**다 (exhausted 같은 쓰기는 전부 Judge 노드의 몫 — 역할 분리).
- `build_graph(nodes: dict)` 구조로 만들어 **같은 골격에 가짜 노드든 실제 노드든**
  끼울 수 있다 → 골격 검증과 노드 구현을 분리.
- "다음 hop 있음" = `plan.query_type == "multi_hop"이고 hop_index < MAX_HOP-1`.

## 4. 가짜 노드 4시나리오 테스트 — agents/baseline/flow_test.py

LLM 호출 없이(비용 0) 판정 대본을 소비하는 가짜 Judge로 흐름만 검증했다.
단, **exhausted 판정만은 실제와 같은 `mark_exhausted` 헬퍼**를 쓰게 해
"exhausted는 Judge 단일 책임" 규칙이 가짜/실제에서 동일하게 작동함을 보장했다.

| 시나리오 | 경로 | 핵심 검증 |
|---|---|---|
| ① 정상 멀티홉 | planner→search→judge→**hop전환**→search→judge→generator | evidence 2개, hop_index=1, retry=0, 중간답 1개 |
| ② 재작성 루프 | …judge→**rewriter**→search→judge→generator | retry=1, tried_queries 2개 |
| ③ 사후 재계획 | single로 시작→rewriter가 plan을 multi로 갱신→**hop전환 경유** | plan 갱신 확인, hop전환의 retry 리셋 확인 |
| ④ exhausted | 불충분 3연속→rewriter×2→judge가 exhausted 기록→generator | exhausted=True, reason="retry", retry=2(MAX) |

**결과: 4시나리오 경로·상태 검증 전부 PASS.** retry_count "Rewriter에서만 +1,
hop전환에서만 0 리셋" 불변식(CLAUDE.md 규칙 3)도 여기서 검증됐다.

## 5. 검색 노드 (LLM 0회)

`current_hop_query` 임베딩 → ChromaDB top-k(config, 초기 5) →
`search_results`, `top1_distance` 기록. core/db.py 경유 (계층 규칙 준수).

## 6. Planner (LLM 1회)

- 출력: `plan{query_type, hop_type, search_queries, reason}` + `answer_strategy`,
  그리고 `current_hop_query = search_queries[0]`.
- 프롬프트(승인본): 판단 기준 5개 + **3조합 미니 예시** 포함. bridge의
  search_queries[1]은 `{hop1}` 플레이스홀더를 **문자 그대로** 남긴 템플릿.
- 승인 시 반영한 수정 3건: ① single이면 hop_type을 따옴표 없는 JSON `null`
  리터럴로 출력하도록 명시 + 코드에서 문자열 "null"/"None"도 None으로 정규화
  ② (Judge) missing 지시 교체 ③ 3조합 예시 추가.
- 코드 검증: query_type/answer_strategy 값 검사, bridge면 질의 2개+`{hop1}` 필수.
  **파싱·검증 실패 → fallback** `{single_hop, null, [원본 질문], "fallback"}` + 정답형.

## 7. Judge (문지기 + LLM 0~1회, 설계 A)

1. **문지기(LLM 아님)**: `top1_distance > 0.70`이면 즉시
   `{insufficient, gatekeeper, rel=low, suf=low}` + 기계 문구 reason. LLM 비용 0.
2. **LLM 판정**: 기준점은 **오직 current_hop_query** (마지막 hop도 동일).
   3칸(verdict/relevance/sufficiency) + reason + missing
   (missing은 "부족한 구체적 개체명·속성명"을 짧은 구로 — 예: '전계수의 출신 학교').
3. **모순 칸 교정(코드)**: rel low & suf high → rel=high로 교정 후 sufficient 전진.
4. **파싱 실패 → 1회 재호출 → 재실패 시 sufficient 통과** + judge_history에
   "parse_fail" 기록 (그래프가 멈추지 않게 하는 안전판).
5. **exhausted 판정·기록은 Judge 단일 책임**: insufficient이고 retry 소진("retry")
   또는 hop 소진("hop")이면 exhausted=True 기록. 분기 함수는 이를 읽기만 한다.

### 프롬프트 승인 과정에서 잡은 결함 (유출 테스트)

검증 케이스 ①(아래)에서 승인 초안의 `[원본 질문] 참고용` 줄이 **유출 통로**로
확인됐다 — Judge가 hop1 판정인데 원본 질문의 최종 답("출신 학교")까지 요구해
insufficient를 냈다. SPEC §3의 기준점 규칙에 따라 **원본 질문 줄을 제거**하고
"현재 질의에 답할 수 있다면 그 너머의 정보 부재를 이유로 low 판정 금지" 문장을
추가했다(수정본 재승인 완료). 수정 후 동일 케이스가 sufficient로 정상 판정됐다.

## 8. 통합 검증 4케이스 (전부 통과)

| 케이스 | 조건 | 결과 |
|---|---|---|
| ① 원본 질문 유출 | bridge 질문의 hop1 판정, 검색 결과에 최종 답 없음(통제 조건) | 수정본으로 **sufficient** — "감독이 전계수임을 명확히 언급... 충분" |
| ② 문지기 발화 | DB 무관 질문(슈뢰딩거 방정식), top1=0.754 | **gatekeeper 즉시 차단, LLM 0회** |
| ③ 전략 분류 | 예시와 다른 질문 2개 | 정답형(single/null 정규화 확인)·탐색형(comparison) 정확 분류, bridge `{hop1}` 템플릿 유지 |
| ④ 파싱 실패 | call_llm 몽키패치로 비JSON 강제 | Planner→fallback 계획, Judge→정확히 2회 호출 후 sufficient+parse_fail 기록 |

## 9. 구현 세부 (알아두면 좋은 것)

- **llm_call_count 카운터 패턴**: LangGraph 노드는 state를 직접 변경하지 않으므로,
  각 노드가 state 값을 시드로 로컬 카운터를 만들어 `call_llm`에 넘기고 누적값을
  상태 갱신으로 반환한다. assert ≤ 20 불변식이 노드 경계를 넘어 유지된다.
- 가짜/실제 노드가 같은 `build_graph`를 쓰므로 Day 3에서 hop전환·Rewriter·
  Generator를 구현하면 그 자리에 끼우기만 하면 된다.

## 10. 산출물

| 경로 | 내용 |
|---|---|
| `core/state.py` | AgentState + make_initial_state (SPEC §2) |
| `core/config.py`, `config/baseline.yaml` | gate_threshold 0.70 고정, top_k 5 |
| `agents/baseline/graph.py` | 그래프 골격 + 순수 함수 분기 |
| `agents/baseline/nodes.py` | 검색·Planner·Judge(문지기) + mark_exhausted |
| `agents/baseline/flow_test.py` | 가짜 노드 4시나리오 테스트 (재실행 가능) |
| `eval/run_eval.py` | 결과 스냅샷의 gate_threshold를 config 참조로 변경 |

## 11. 다음 단계 (Day 3 예고)

hop전환(bridge 중간 답 추출·`{hop1}` 치환, comparison은 추출 생략)·Rewriter
(3모드: gatekeeper→C 전면 수정 / rel low→A 방향 전환 / suf low→B 겨냥 보강+사후
재계획)·Generator(정답형/탐색형 × 정상/exhausted 2×2)를 구현해 multi-hop 경로를
완성한다. **세 노드의 프롬프트 전문은 코드 작성 전 승인** 절차를 따른다.
완료 기준: 150 전체 평가에서 naive 대비 조합별 개선 + Planner 분류 정확도 산출.
