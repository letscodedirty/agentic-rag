# PLAN.md — 11일 실행 계획

각 Day의 지시: "docs/PLAN.md의 Day N 수행. SPEC.md·CLAUDE.md 준수."
매일 저녁: 평가 실행 → eval/results/ 저장 → git commit. 완료 기준 미달 시 사용자와 논의.

## Day 1 — 기반 + 1-pass baseline
- 셋업: 가상환경, .env(OPENAI_API_KEY, LLM_MODEL, EMBED_MODEL, BACKEND_URL), 디렉토리 구조
- scripts/ 3종 실행: 데이터 선별(100) → single 파생(50, 검수 CSV) → 통합 DB 구축(+무결성 체크)
- eval/testset.jsonl 생성, run_eval.py 완성 (하네스 먼저!)
- agents/naive 완성 → **완료 기준: 1-pass의 조합별 Hit Rate·MRR 박제 (150 전체)**
- 보너스: top1_distance 분포 출력 → GATE_THRESHOLD 후보값 기록
- (여유 시) 청크 단위 실험: 문단 vs 문장 모드 Hit Rate 비교

## Day 2~3 — baseline Agentic
- Day 2: core/state.py + 그래프 골격(LLM 없는 가짜 노드로 4가지 흐름 시나리오 검증:
  정상/재작성 루프/사후 재계획/exhausted) → Planner·검색·Judge LLM화
- Day 3: hop전환·Rewriter(3모드)·Generator(2×2 프롬프트) 완성, multi-hop 경로 통과
- **완료 기준: 150 전체 평가에서 1-pass 대비 조합별 개선 확인 + Planner 분류 정확도 산출**

## Day 4 — 통합 (Agentic 완성품)
- backend/main.py: /health → /ask_naive → /ask (docs로 검증)
- frontend/app.py: 탭 2개 (단독 + 비교)
- **완료 기준: 브라우저에서 같은 질문의 naive vs agentic 비교 화면 동작**

## Day 5 — 하이퍼파라미터 튜닝
- 서브셋 48로 회전: k∈{3,5,10}, GATE_THRESHOLD 후보, (청크 모드 미결 시 포함)
- best 조합을 150 전체로 확정 측정 → config/baseline.yaml 고정
- **완료 기준: best params 표 + `git tag v1-baseline`**

## Day 6~7 — 구조 수정 (agents/improved)
- day 5 결과의 약점 분석 → 수정안 선택 (1순위 re-ranking, 2순위 보존 아이디어:
  청크별 판정/k분리/중간답 하이브리드/모델 차등화, 3순위 새 구조 — 사용자와 논의 후 확정)
- agents/improved에 구현 (baseline 동결 유지, core 공유)
- **완료 기준: improved의 150 전체 평가 완료 (re-ranking이면 MRR 개선 확인)**

## Day 8 — 최종 측정
- improved 세부 튜닝 → 확정 측정
- **완료 기준: {naive, baseline, improved} × 조합별 Hit Rate·MRR 최종 비교표 +
  요구사항 체크리스트 점검(계획/정규화=분해/검토/재정규화·재검색, 데모 표시 항목) +
  `git tag v2-improved`**

## Day 9~10 — 버퍼
- 지연 흡수. 여유 시: 넛지형 시연 질문 추가, 스트리밍, 화면 다듬기, 추가 실험

## Day 11 — 발표
- PPT: 문제→구조(그래프 그림)→핵심 표(유형별 개선)→데모 시나리오
- **데모 리허설 필수** (시작 전 /health 확인 루틴 포함)
- **완료 기준: 발표 자료 + 리허설 1회 완료**
