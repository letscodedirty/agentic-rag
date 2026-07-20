"""config/baseline.yaml 로더 + SPEC §1 고정 상수.

GATE_THRESHOLD는 0.70 고정(튜닝 비대상) — 값은 config 파일에 두되
Day 5 튜닝 그리드에서는 제외한다 (SPEC §1, PLAN Day 5).
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# SPEC §1 상수 (변경 금지)
MAX_HOP = 2
MAX_RETRY = 2  # hop별 리셋

_cfg = None


def load_config() -> dict:
    global _cfg
    if _cfg is None:
        with open(ROOT / "config" / "baseline.yaml", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def gate_threshold() -> float:
    return float(load_config()["gate_threshold"])


def default_top_k() -> int:
    return int(load_config()["top_k"])
