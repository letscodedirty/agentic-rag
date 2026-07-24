# data/v2 — v2 3층 데이터 (Day 6)

대용량 원본 2종은 GitHub 파일 한도(100MB) 때문에 gzip으로 커밋한다. 복원:

```bash
gunzip -k data/v2/pages_snapshot.jsonl.gz data/v2/chunks.jsonl.gz
```

무결성 (sha256, 원본 기준):

- `pages_snapshot.jsonl` dbd3f9a3089ef07047d83a7e8ac9237cbbb0c1cffd17a59ffc5c2b5871f9da85
- `chunks.jsonl` 8251cdf4b2016550a2258253adafd4a2e65835a6825c4838668d09e5d2e4fb0d

재생성 경로: `scripts/collect_v2.py`(수집 스냅샷) → `scripts/build_v2_layers.py`(3층 산출)
→ `scripts/build_db_v2.py`(./db_v2 적재, chunks.jsonl 필요 — 위 gunzip 선행).
