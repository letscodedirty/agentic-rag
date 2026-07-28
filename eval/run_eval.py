"""SPEC §5: 평가 하네스.

사용: python eval/run_eval.py --system {naive|baseline|improved} [--subset {48|150}]
      [--tag NAME] [--top-k 5] [--http]

하네스-에이전트 계약: "질문 in → evidence 포맷 [{"hop": n, "chunk_ids": [...]}] 포함
결과 out"만 지키면 채점 가능 (CLAUDE.md).

채점: 각 라벨 hop의 정답 청크 title이 evidence의 chunk_ids 목록들 중 어디든
등장하면 hit, 그 목록 내 1-based 순위로 RR 계산 (여러 목록에 있으면 best).
질문 단위 Hit = 모든 hop hit, 질문 MRR = hop RR 평균 → 전체·조합별·hop별 집계.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from core.config import gate_threshold  # noqa: E402

load_dotenv()

# 라벨 combo → Planner 정답 (분류 정확도 대조용, SPEC §4)
COMBO_LABELS = {
    "single": {"query_type": "single_hop", "hop_type": None, "answer_strategy": "정답형"},
    "bridge": {"query_type": "multi_hop", "hop_type": "bridge", "answer_strategy": "정답형"},
    "comparison": {"query_type": "multi_hop", "hop_type": "comparison", "answer_strategy": "탐색형"},
}


def load_testset(subset: str) -> list:
    """"dev" → 독립 dev셋(튜닝 전용) / "150" → 확정 측정용 본 테스트셋 (SPEC §5)
    / "v2" → Day 8 v2 통합 테스트셋(라벨=정답 청크 집합, SPEC §8)."""
    name = {"dev": "devset.jsonl", "v2": "testset_v2.jsonl"}.get(
        subset, "testset.jsonl")
    path = ROOT / "eval" / name
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_runner(system: str, top_k: int, use_http: bool):
    if use_http:
        import requests

        base = os.environ["BACKEND_URL"].rstrip("/")
        h = requests.get(f"{base}/health", timeout=10).json()  # /health 가드
        assert h.get("status") == "ok", f"/health 실패: {h}"
        assert h.get("space") == "cosine", f"/health space != cosine: {h}"
        endpoint = "/ask_naive" if system == "naive" else "/ask"

        def run(question):
            r = requests.post(
                f"{base}{endpoint}", json={"question": question, "top_k": top_k},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()

        return run

    if system == "v2":
        # Day 8 오라클 하네스(승인 설계 c): v2는 프로세스 내 실행 전용.
        # clarify 인자는 오라클 루프(1차 ON → 재실행 OFF)가 제어한다.
        assert not use_http, "--system v2는 --http 미지원 (오라클 루프는 프로세스 내)"
        from agents.v2.graph import run_agent

        return lambda q, clarify=False: run_agent(q, top_k=top_k, clarify=clarify)
    if system == "naive":
        from agents.naive.pipeline import run_naive

        return lambda q: run_naive(q, top_k=top_k)
    if system == "baseline":
        from agents.baseline.graph import run_agent

        return lambda q: run_agent(q, top_k=top_k)
    if system == "improved":
        from agents.improved.graph import run_agent

        return lambda q: run_agent(q, top_k=top_k)
    raise ValueError(system)


def score_question(label: dict, evidence: list):
    """hop별 hit/RR. evidence: [{"hop": n, "chunk_ids": [...]}]"""
    per_hop = {}
    for hop, ha in label["hop_answers"].items():
        gold = ha["title"]
        best_rank = None
        for ev in evidence or []:
            ids = ev.get("chunk_ids", [])
            if gold in ids:
                rank = ids.index(gold) + 1
                if best_rank is None or rank < best_rank:
                    best_rank = rank
        per_hop[hop] = {
            "hit": best_rank is not None,
            "rr": (1.0 / best_rank) if best_rank else 0.0,
        }
    hits = [v["hit"] for v in per_hop.values()]
    rrs = [v["rr"] for v in per_hop.values()]
    return {
        "per_hop": per_hop,
        "hit": all(hits),
        "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
    }


def score_question_v2(label: dict, evidence: list):
    """v2 테스트셋 채점 (SPEC §8: 라벨=정답 청크 집합, 하나라도 적중 hit).

    기존 score_question의 RR 로직(목록 내 1-based 순위, 여러 목록이면 best)을
    집합 라벨로 이식 — 기존 함수는 hop별 단일 gold 스키마라 무변경 유지
    (승인 설계 b). 목록마다 gold와 처음 만나는 위치가 그 목록의 순위."""
    gold = set(label.get("gold_chunk_ids") or [])
    best_rank = None
    for ev in evidence or []:
        for pos, cid in enumerate(ev.get("chunk_ids", []), start=1):
            if cid in gold:
                if best_rank is None or pos < best_rank:
                    best_rank = pos
                break
    return {
        "hit": best_rank is not None,
        "mrr": (1.0 / best_rank) if best_rank else 0.0,
        "best_rank": best_rank,
    }


def aggregate_v2(rows: list):
    """v2 테스트셋 집계: overall + 유형(qtype)별 Hit·MRR (승인 설계 b)."""
    def agg(sub):
        n = len(sub)
        return {
            "n": n,
            "hit_rate": round(sum(r["score"]["hit"] for r in sub) / n, 4) if n else None,
            "mrr": round(sum(r["score"]["mrr"] for r in sub) / n, 4) if n else None,
        }

    out = {"overall": agg(rows)}
    for t in ["single", "bridge", "comparison", "list", "ambiguous"]:
        out[t] = agg([r for r in rows if r.get("qtype") == t])
    return out


# ---------- 오라클 루프 (Day 8, SPEC §8 평가 문단 — 승인 프롬프트·설계 a) ----------

ORACLE_SYSTEM = "너는 RAG 평가 하네스의 오라클(사용자 대리)이다. 반드시 JSON으로만 답하라."

ORACLE_USER_TMPL = """시스템이 모호한 질문에 되묻기 선택지를 제시했다. 너는 사용자의 실제 의도를
알고 있는 오라클로서, 의도와 일치하는 선택지를 고른다.

[원 질문] <<QUESTION>>
[사용자의 실제 의도] <<INTENDED>>
[선택지]
<<CHOICES>>

규칙:
1. 실제 의도와 같은 대상·같은 요구를 가리키는 선택지가 있으면 그 번호를
   choice_index로 반환하라.
2. 일치하는 선택지가 없으면 choice_index를 null로 반환하라.
3. 한쪽만 특정하는 정보로만 답하라 — 선택지 번호 외에 어떤 추가 정보·설명·
   질문 재작성도 제공하지 마라.
4. 의도가 확실히 일치할 때만 선택하라. 애매하면 null.

JSON: {"choice_index": <번호 또는 null>}"""


def oracle_decide(question: str, intended, clarification: dict):
    """오라클 응답 결정 (승인 설계 a — LLM은 선택지 대조가 필요할 때만 1회).

    반환: (재실행 질문, mode, choice_index, 오라클 LLM 호출 수)
    mode: "choice"(의도 일치 선택지 클릭) / "free_input"(intended 자유 입력)
          / "original"(비모호 오발동 — 원 질문 그대로, 추가 정보 금지).
    나머지 분기는 규칙상 반환값이 결정돼 있어 LLM 미경유(CLAM 교훈 —
    왜곡 표면 최소화). 호출은 call_llm 경유(CLAUDE.md 규칙 4), temperature=0.
    """
    choices = clarification.get("choices") or []
    if not intended:
        return question, "original", None, 0
    if not choices:
        return intended, "free_input", None, 0
    from core.llm import call_llm

    lines = "\n".join(f"{i}. {c.get('label', '')} — {c.get('question', '')}"
                      for i, c in enumerate(choices))
    user = (ORACLE_USER_TMPL.replace("<<QUESTION>>", question)
            .replace("<<INTENDED>>", intended)
            .replace("<<CHOICES>>", lines))
    counter = {"llm_call_count": 0}  # 문항별 독립 카운터
    raw = call_llm(counter, [
        {"role": "system", "content": ORACLE_SYSTEM},
        {"role": "user", "content": user},
    ], json_mode=True)
    try:
        idx = json.loads(raw).get("choice_index")
    except (json.JSONDecodeError, AttributeError):
        idx = None
    if isinstance(idx, int) and 0 <= idx < len(choices) and choices[idx].get("question"):
        return choices[idx]["question"], "choice", idx, 1
    return intended, "free_input", None, 1


def run_v2_item(item: dict, runner, system: str) -> dict:
    """v2 서브셋 문항 1건 실행·채점 (오라클 루프 포함) → 결과 행.

    v2 시스템: 1차 clarify=True → 발동 시 오라클 결정 질문으로 clarify=False
    재실행, 재실행 결과가 채점 대상(SPEC §8 — 무한 재발동 방지).
    naive·baseline: 오라클 없이 원 질문 1회.
    """
    q = item["question"]
    intended = item.get("intended_query")
    oracle = {"activated": False, "reason": None, "category": None,
              "mode": None, "choice_index": None, "rerun_question": None,
              "rerun": False, "oracle_llm_calls": 0}
    clarify_calls = 0
    llm_calls = 0
    try:
        if system == "v2":
            res = runner(q, clarify=True)
            clarify_calls = res.get("clarify_calls") or 0
            llm_calls += res.get("llm_calls") or 0
            cl = res.get("clarification")
            if cl:
                rq, mode, idx, oc = oracle_decide(q, intended, cl)
                oracle.update(activated=True, reason=cl.get("reason"),
                              category=cl.get("category"), mode=mode,
                              choice_index=idx, rerun_question=rq,
                              rerun=True, oracle_llm_calls=oc)
                res = runner(rq, clarify=False)
                llm_calls += res.get("llm_calls") or 0
        else:
            res = runner(q)
            llm_calls = res.get("llm_calls") or 0
    except Exception as e:  # 실패 문항은 0점 처리하되 기록 (기존 규약)
        print(f"  [오류] {item.get('qid')}: {e}")
        res = {"evidence": [], "answer": f"[error] {e}"}
    score = score_question_v2(item, res.get("evidence", []))
    return {
        "qid": item.get("qid"),
        "question": q,
        "qtype": item.get("qtype"),
        "subtype": item.get("subtype"),
        "intended_query": intended,
        "answer": res.get("answer", ""),
        "score": score,
        "oracle": oracle,
        "clarify_calls": clarify_calls,
        "llm_calls": llm_calls,
        "top1_distance": res.get("top1_distance"),
        "plan": res.get("plan"),
        "retry_total": res.get("retry_total"),
        "exhausted": res.get("exhausted"),
        "exhausted_reason": res.get("exhausted_reason"),
        "elapsed_sec": res.get("elapsed_sec"),
    }


def clarification_metrics(rows: list):
    """부가 지표(지시 3): 모호 발동률·비모호 오발동률·오라클 호출 수."""
    amb = [r for r in rows if r.get("intended_query")]
    non = [r for r in rows if not r.get("intended_query")]

    def act(sub):
        return sum(1 for r in sub if r["oracle"]["activated"])

    return {
        "ambiguous_n": len(amb),
        "ambiguous_activated": act(amb),
        "ambiguous_activation_rate": round(act(amb) / len(amb), 4) if amb else None,
        "non_ambiguous_n": len(non),
        "false_activated": act(non),
        "false_activation_rate": round(act(non) / len(non), 4) if non else None,
        "oracle_llm_calls": sum(r["oracle"]["oracle_llm_calls"] for r in rows),
        "clarify_calls_total": sum(r.get("clarify_calls") or 0 for r in rows),
    }


def aggregate(rows: list):
    def agg(sub):
        n = len(sub)
        return {
            "n": n,
            "hit_rate": round(sum(r["score"]["hit"] for r in sub) / n, 4) if n else None,
            "mrr": round(sum(r["score"]["mrr"] for r in sub) / n, 4) if n else None,
        }

    out = {"overall": agg(rows)}
    for combo in ["single", "bridge", "comparison"]:
        out[combo] = agg([r for r in rows if r["combo"] == combo])
    # hop별 (라벨 hop 기준)
    for hop in ["1", "2"]:
        sub = [r for r in rows if hop in r["score"]["per_hop"]]
        n = len(sub)
        out[f"hop{hop}"] = {
            "n": n,
            "hit_rate": round(
                sum(r["score"]["per_hop"][hop]["hit"] for r in sub) / n, 4
            ) if n else None,
            "mrr": round(
                sum(r["score"]["per_hop"][hop]["rr"] for r in sub) / n, 4
            ) if n else None,
        }
    return out


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 4)


def planner_accuracy(rows: list):
    """plan 필드가 있는 시스템(baseline/improved)만 산출."""
    with_plan = [r for r in rows if isinstance(r.get("plan"), dict)]
    if not with_plan:
        return None
    ht = sum(
        1 for r in with_plan
        if r["plan"].get("hop_type") == COMBO_LABELS[r["combo"]]["hop_type"]
        and r["plan"].get("query_type") == COMBO_LABELS[r["combo"]]["query_type"]
    )
    st = sum(
        1 for r in with_plan
        if r.get("answer_strategy") == COMBO_LABELS[r["combo"]]["answer_strategy"]
    )
    n = len(with_plan)
    return {"n": n, "hop_type_acc": round(ht / n, 4), "strategy_acc": round(st / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["naive", "baseline", "improved", "v2"])
    ap.add_argument("--subset", default="150", choices=["dev", "150", "v2"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--corpus", default="v1", choices=["v1", "v2"],
                    help="검색 코퍼스 주입 (SPEC §8: 3시스템 동일 v2 1층 비교. "
                         "--http 모드에서는 미지원 — 서버 측 코퍼스를 따름)")
    args = ap.parse_args()
    if args.subset == "v2":
        # Day 8 오라클 하네스(승인 설계 c): 3시스템 동일 v2 코퍼스 대조가
        # 전제 — v1 코퍼스·HTTP 조합은 의미가 어긋나므로 차단.
        assert args.corpus == "v2", "--subset v2는 --corpus v2 전제 (SPEC §8)"
        assert not args.http, "--subset v2는 --http 미지원 (오라클 루프)"
    if args.corpus == "v2":
        assert not args.http, "--corpus v2는 --http와 병용 불가 (프로세스 내 주입)"
        from core import db as _db

        _db.set_search_collection("v2")
    tag = args.tag or (f"{args.system}_k{args.top_k}_s{args.subset}"
                       + ("_cv2" if args.corpus == "v2" else ""))

    testset = load_testset(args.subset)
    runner = get_runner(args.system, args.top_k, args.http)
    print(f"평가 시작: system={args.system} n={len(testset)} top_k={args.top_k}")

    rows = []
    t0 = time.time()
    if args.subset == "v2":
        # v2 서브셋 전용 루프 (Day 8 오라클 하네스 — 기존 dev/150 루프 무변경)
        for i, item in enumerate(testset, 1):
            rows.append(run_v2_item(item, runner, args.system))
            if i % 10 == 0:
                print(f"  ... {i}/{len(testset)} ({time.time() - t0:.0f}s)")
        metrics = aggregate_v2(rows)
        if args.system == "v2":
            metrics["clarification"] = clarification_metrics(rows)
    else:
        for i, item in enumerate(testset, 1):
            try:
                res = runner(item["question"])
            except Exception as e:  # 실패 질문은 0점 처리하되 기록
                print(f"  [오류] Q{i}: {e}")
                res = {"evidence": [], "answer": f"[error] {e}"}
            score = score_question(item, res.get("evidence", []))
            rows.append(
                {
                    "question": item["question"],
                    "combo": item["combo"],
                    "gold_answer": item["gold_answer"],
                    "answer": res.get("answer", ""),
                    "score": score,
                    "llm_calls": res.get("llm_calls"),
                    "top1_distance": res.get("top1_distance"),
                    "plan": res.get("plan"),
                    "answer_strategy": res.get("strategy") or res.get("answer_strategy"),
                    "retry_total": res.get("retry_total"),
                    "exhausted": res.get("exhausted"),
                    "exhausted_reason": res.get("exhausted_reason"),
                    "rewrite_history": res.get("rewrite_history"),
                    "elapsed_sec": res.get("elapsed_sec"),
                }
            )
            if i % 25 == 0:
                print(f"  ... {i}/{len(testset)} ({time.time() - t0:.0f}s)")
        metrics = aggregate(rows)
    llm_calls = [r["llm_calls"] for r in rows if r["llm_calls"] is not None]
    metrics["llm_calls_avg"] = round(sum(llm_calls) / len(llm_calls), 2) if llm_calls else None
    retries = [r["retry_total"] for r in rows if r["retry_total"] is not None]
    metrics["retry_rate"] = (
        round(sum(1 for x in retries if x > 0) / len(retries), 4) if retries else None
    )
    exh = [r for r in rows if r["exhausted"] is not None]
    if exh:
        from collections import Counter

        metrics["exhausted_rate"] = round(
            sum(1 for r in exh if r["exhausted"]) / len(exh), 4
        )
        metrics["exhausted_reasons"] = dict(
            Counter(r["exhausted_reason"] for r in exh if r["exhausted"])
        )
    # v2 서브셋은 combo 라벨이 없어 Planner 대조 미산출 (유형 지표는 aggregate_v2)
    metrics["planner_accuracy"] = (
        None if args.subset == "v2" else planner_accuracy(rows))

    # top1_distance 분포 → GATE_THRESHOLD 후보 (cosine DB 기준)
    dists = sorted(r["top1_distance"] for r in rows if r["top1_distance"] is not None)
    if dists:
        metrics["top1_distance"] = {
            "n": len(dists),
            "min": round(dists[0], 4), "p25": pct(dists, 25), "p50": pct(dists, 50),
            "p75": pct(dists, 75), "p90": pct(dists, 90), "p95": pct(dists, 95),
            "p99": pct(dists, 99), "max": round(dists[-1], 4),
        }

    result = {
        "config": {
            "system": args.system, "subset": args.subset, "top_k": args.top_k,
            "http": args.http, "corpus": args.corpus, "tag": tag,
            # 승인 규칙 2 개정(Day 8): 정형 환산 id를 유사도 병합 앞에 배치
            "evidence_order": "structured_first",
            "llm_model": os.environ.get("LLM_MODEL"),
            "embed_model": os.environ.get("EMBED_MODEL"),
            "gate_threshold": gate_threshold(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "metrics": metrics,
        "per_question": rows,
    }
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 결과 ({tag}) ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_question"},
                     ensure_ascii=False, indent=2))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
