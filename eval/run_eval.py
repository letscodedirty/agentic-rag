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
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# 라벨 combo → Planner 정답 (분류 정확도 대조용, SPEC §4)
COMBO_LABELS = {
    "single": {"query_type": "single_hop", "hop_type": None, "answer_strategy": "정답형"},
    "bridge": {"query_type": "multi_hop", "hop_type": "bridge", "answer_strategy": "정답형"},
    "comparison": {"query_type": "multi_hop", "hop_type": "comparison", "answer_strategy": "탐색형"},
}


def load_testset(subset: int) -> list:
    path = ROOT / "eval" / "testset.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if subset and subset < len(rows):
        # 서브셋 48 = 조합별 16개 층화 추출 (고정 시드)
        per_combo = subset // 3
        rng = random.Random(42)
        picked = []
        for combo in ["single", "bridge", "comparison"]:
            group = [r for r in rows if r["combo"] == combo]
            picked += rng.sample(group, min(per_combo, len(group)))
        rows = picked
    return rows


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
    ap.add_argument("--system", required=True, choices=["naive", "baseline", "improved"])
    ap.add_argument("--subset", type=int, default=150, choices=[48, 150])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--http", action="store_true")
    args = ap.parse_args()
    tag = args.tag or f"{args.system}_k{args.top_k}_s{args.subset}"

    testset = load_testset(args.subset)
    runner = get_runner(args.system, args.top_k, args.http)
    print(f"평가 시작: system={args.system} n={len(testset)} top_k={args.top_k}")

    rows = []
    t0 = time.time()
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
    metrics["planner_accuracy"] = planner_accuracy(rows)

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
            "http": args.http, "tag": tag,
            "llm_model": os.environ.get("LLM_MODEL"),
            "embed_model": os.environ.get("EMBED_MODEL"),
            "gate_threshold": os.environ.get("GATE_THRESHOLD"),
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
