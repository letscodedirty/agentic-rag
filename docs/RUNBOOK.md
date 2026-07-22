# RUNBOOK.md — Day별 실행 가이드 (순서대로 따라하기)

표기: [나] = 내가 직접 / [Code] = Claude Code에 입력 / [웹] = 웹브라우저 클로드 새 채팅(SPEC.md 첨부)
모든 Day 공통 예외: Code가 SPEC과 다른 방향으로 가면 → "SPEC.md §N과 다르다. 맞춰라."
에러가 3회 이상 반복되면 → [웹] 새 채팅에 SPEC.md + 에러 전문 첨부하고 원인 분석.

GitHub 규칙 (원격: https://github.com/letscodedirty/agentic-rag):
- 마일스톤 push 3회는 필수 — [나]가 직접 수행:
  ① Day 3 완료(baseline 코드+평가 통과) → `git push`
  ② Day 5 완료(최적 하이퍼파라미터 확정, tag v1-baseline) → `git push && git push --tags`
  ③ Day 8 완료(improved 확정 측정, tag v2-improved) → `git push && git push --tags`
- 그 외 Day의 commit 후 push는 선택(백업 겸 권장).
- push 전 `git status`에 .env가 절대 보이면 안 됨(보이면 중단 후 .gitignore 확인).

---

## Day 0.5 — 최초 준비 (10분)
1. [나] platform.openai.com에서 API 키 발급 + 크레딧 충전($10 권장)
2. [나] WSL 터미널:
   mkdir agentic-rag && cd agentic-rag && mkdir docs
   (CLAUDE.md → 루트 / SPEC.md, PLAN.md, RUNBOOK.md → docs/)
   git init && git add . && git commit -m "day 0" && code .
3. [나] VS Code 터미널에서 `claude` 실행

## Day 1 — 기반 + 1-pass
1. [나] .env 파일 생성해 API 키 입력 (형식은 SPEC §7 / CLAUDE.md 참조)
2. [Code] "docs/PLAN.md의 Day 1을 수행해. docs/SPEC.md와 CLAUDE.md를 준수해.
추가 지시: core/db.py에서 컬렉션 로드 직후 hnsw:space가 "cosine"인지
assert로 검증하고, build_db.py는 재구축 시 기존 컬렉션을 삭제 후 생성하라.
/health 응답에 space 값도 포함하라. GATE_THRESHOLD 분포 측정은 반드시
cosine DB 구축 완료 후에 수행하라."
3. [나·낮] Code가 검수 CSV(single 파생 50개)를 만들면 → 열어서 훑고 이상한 행에 X 표시
4. [Code] "검수 완료. X 표시된 행만 재생성해."
5. [Code·저녁] "run_eval.py로 naive를 150 전체 평가하고 eval/results/에 저장.
   PLAN.md Day 1 완료 기준을 결과 숫자로 확인하고, 충족했으면 git commit.
   추가로 top1_distance 분포 요약을 보여줘." → [나] 분포 보고 GATE_THRESHOLD 후보 메모
- 미달 시: 원인 명확 → [Code]에서 수정 / 불명확 → [웹] 결과 JSON 첨부 분석

## Day 2 — Agentic 골격 + 앞 3노드
1. [Code] "PLAN.md Day 2 수행. SPEC·CLAUDE.md 준수.
   ★각 노드의 LLM 프롬프트는 코드 작성 전에 전문을 먼저 보여주고 내 승인을 받아."
2. [나·낮] 프롬프트 전문(Planner, Judge) 검토 — SPEC §3의 규칙(출력 JSON 필드,
   전략 규칙, 모순 칸 문구)이 다 들어갔는지 확인 후 승인
3. [Code·저녁] "가짜 노드 4가지 흐름 시나리오(정상/재작성/사후 재계획/exhausted)
   테스트 결과를 보여줘. 통과했으면 git commit."

## Day 3 — Agentic 완성
1. [Code] "PLAN.md Day 3 수행. hop전환·Rewriter·Generator의 LLM 프롬프트도
   전문 승인 방식으로." → [나] Rewriter 3모드·Generator 2×2 프롬프트 검토
2. [Code·저녁] "baseline을 150 전체 평가. naive 결과와 조합별 비교표 +
   Planner 분류 정확도를 보여줘. Day 3 완료 기준 확인 후 commit."
3. [나] 비교표 확인 — 개선이 안 보이는 조합이 있으면 → [웹] 결과 첨부 분석
4. [나] ★마일스톤 ①: 완료 기준 충족 확인 후 `git push` (baseline 코드 GitHub 업로드)

## Day 4 — 통합 (완성품)
1. [Code] "PLAN.md Day 4 수행. /health → /ask_naive → /ask 순서로 만들고
   각각 localhost:8000/docs에서 검증 가능하게."
2. [나·낮] 브라우저에서 localhost:8000/docs 열어 /ask 직접 테스트 1회
3. [Code] "이제 frontend/app.py. SPEC §6 화면 계약대로 탭 2개."
4. [나·저녁] 터미널 2개(uvicorn / streamlit) 띄우고 브라우저에서:
   질문 입력 → 답변·expander 확인, 비교 탭에서 naive vs agentic 나란히 확인
5. [Code] "동작 확인 완료. git commit."

## Day 5 — 튜닝 (누수 방지 절차)
1. [Code] "PLAN.md Day 5 수행. 먼저 독립 dev셋 48(조합별 16)을 기존 생성 규칙
   전부 적용해 생성하라. 기존 DB 청크 한정, 기존 150과 질문 중복 금지.
   검수 CSV를 출력하고 멈춰라."
2. [나] 검수(48행, X 표시 → 재생성 반복) → 통과 시 "검수 통과. devset.jsonl 확정."
3. [Code] "dev셋으로 baseline k=5 평가(기준점) → k=3, k=10 평가 → 3개 결과 표."
4. [나] best k 결정(Hit 우선, 동률 시 llm_calls 적은 쪽) → [Code] "k=_ 확정.
   150 전체로 개선 측정 1회 실행하고 결과 저장."
5. [Code] "Planner 과분해 교정: 프롬프트에 넣을 창작 예시를 먼저 보여주고 승인
   받아라(기존 150 질문 문구 금지). 승인 후 dev셋 재평가 — 분류 정확도와 반대
   방향 오분류를 보고하라."
6. [나] 개선 확인 → [Code] "150 확정 측정 → config 고정 → commit, git tag v1-baseline."
7. [나] ★마일스톤 ②: `git push && git push --tags`

## Day 6 — 구조 수정 결정 + 착수
1. [웹·아침] 새 채팅에 SPEC.md + Day5 결과 JSON 첨부:
   "확정 설계서와 baseline 평가 결과다. 약점을 분석하고 improved 수정안을 논의하자.
   "1순위 후보 re-ranking, 2순위 본문 청크 추가(중복 최소화, 채점 라벨 확장 과제
포함), 3순위 청크별 판정/k분리/중간답 하이브리드/모델 차등화. 내가 강하게
추천하는 아이디어가 있으면 우선순위를 높여 논의하라."
2. [나] 논의 후 수정안 확정 → 그 채팅에서 "SPEC.md에 추가할 §8(improved 설계) 문구를 써줘"
3. [나] SPEC.md에 §8 붙여넣기 → git commit
4. [Code] "SPEC.md에 §8이 추가됐다. PLAN.md Day 6~7 수행. agents/improved에 구현,
   baseline은 건드리지 마. 새 LLM 프롬프트는 승인 방식."

## Day 7 — 구조 수정 완성
1. [Code] "improved 완성 후 150 전체 평가. baseline과 비교표 보여줘.
   (re-ranking이면 MRR 변화 중심으로.) 완료 기준 확인 후 commit."
2. [Code] "backend에 /ask_improved 추가(응답 계약 /ask와 동일). frontend:
   단독 탭에 시스템 선택기(baseline|improved, 기본값 improved) 추가, 비교 탭을
   naive|baseline|improved 3열로 확장. 브라우저에서 단독 탭 improved 동작 +
   같은 질문 3-way 동작 확인."
3. [나] 비교표 확인 — 개선이 없으면 → [웹] Day 6 채팅 이어서 원인 분석·조정 논의

## Day 8 — 최종 측정
1. [Code] "improved 세부 튜닝(서브셋) → 최적값으로 150 확정 측정 →
   naive/baseline/improved × 조합별 Hit Rate·MRR 최종 비교표 생성.
   PLAN.md Day 8의 요구사항 체크리스트도 점검해줘."
2. [나] 표·체크리스트 확인 → [Code] "commit, git tag v2-improved."
3. [나] ★마일스톤 ③: `git push && git push --tags` (improved 최종 버전 업로드)
4. [나] 데모 화면 스크린샷 몇 장 찍어두기 (Day 11 PPT 재료)

## Day 9~10 — 버퍼
- 지연분 소화. 남으면 [Code]에 하나씩: "넛지형 시연 질문 5개 추가" /
  "화면에 개선 기능 ON/OFF 토글" / "스트리밍" 등 (전부 선택)
- 전부 여유면 [나] 데모 시나리오 대본 작성: 어떤 질문을 어떤 순서로 시연할지
  (실패→개선이 극적인 질문을 평가 결과에서 골라두기)

## Day 11 — 발표
1. [웹] 새 채팅에 SPEC.md + 최종 비교표 JSON + 스크린샷 첨부:
   "이 프로젝트의 발표 PPT를 만들어줘. 구성: 문제→구조(그래프)→
   1-pass 대비 개선 표→데모 안내→한계와 확장."
2. [나] PPT 받아 다듬기
3. [나] 데모 리허설 1회: 터미널 2개 기동 → /health 확인 → 대본대로 시연
   (리허설 중 발견된 문제만 [Code]로 즉시 수정)
