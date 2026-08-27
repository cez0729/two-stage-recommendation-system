# Production-Style Serving Demo

这是一个基于冻结离线模型的 production-style demo，不代表线上业务系统或真实商业收益。

## Architecture

```text
user_id -> history -> Two-Tower/FAISS(100) + ItemCF(100) + Popularity(20)
        -> RRF dedupe/backfill(200) -> 16 offline features -> LambdaRank -> top-k
        -> Redis rec:{model_version}:{user_id}:{k} (TTL 300s) -> FastAPI
        -> request_id/model_version -> SQLite feedback events -> JSONL/CSV export

FastAPI -> Prometheus -> Grafana
```

## Start locally

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH="src"
$env:REDIS_URL="redis://127.0.0.1:6379/0"  # optional; failures bypass cache
python scripts/build_serving_artifacts.py
python -m uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000
```

## API

```powershell
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/A0266076X6KPZ6CCHGVS?k=10"
curl -X POST http://127.0.0.1:8000/events -H "Content-Type: application/json" -d '{"user_id":"u","item_id":"i","event_type":"click","timestamp":"2026-08-27T12:00:00Z","request_id":"demo","model_version":"v1","rank_position":1,"simulated":true}'
curl http://127.0.0.1:8000/metrics
curl "http://127.0.0.1:8000/metrics?format=prometheus"
```

`k` is restricted to 1..50. Known users use the complete pipeline; unknown users use deterministic Popularity fallback. Recommendations are unique and exclude the configured history. Real serving responses include `request_id`, `model_version`, and `fallback`. Models and indexes load once at application startup.

Feedback is persisted at `data/feedback/events.db` by default. In Compose it is mounted as the `feedback_data` volume. Export it with:

```powershell
python scripts/export_feedback.py --output runs/feedback.jsonl
python scripts/export_feedback.py --output runs/feedback.csv --format csv
```

Every demo event must keep `simulated=true`; exported simulated events must not be mixed into the frozen benchmark as online evidence.

## Observability

`/metrics` returns the legacy JSON view by default. Prometheus scrapes `/metrics?format=prometheus` and Grafana is provisioned by Compose. The exported metrics cover request count, latency quantiles, errors, cache hits/misses, fallback count, recommendation count, and model version.

## Reproducible versions

Preview the existing training/evaluation chain without running it:

```powershell
python scripts/retrain.py --config configs/retrain.yaml --dry-run
```

Run the chain with `python scripts/retrain.py --config configs/retrain.yaml`. It reuses the existing split, ItemCF, Two-Tower, FAISS, LambdaRank, frozen evaluation, and manifest commands. Each completed run writes `artifacts/versions/<version>/manifest.json`. Promotion is gated by explicit metric thresholds and 20-user parity:

```powershell
python scripts/promote_model.py --version <version>
```

Serving reads `artifacts/versions/current.json` when present; without it, `configs/serving.yaml` remains the source of truth.

## Evidence

- Two-Tower test Recall@100: 9.51%; Popularity: 6.41%; ItemCF: 15.70%.
- Candidate Recall@200: 21.00% on test.
- LambdaRank test NDCG@10: 2.99%, versus RRF 1.78%.
- FAISS IndexFlatIP exact-match against NumPy: 100%; single-query P95: 2.15 ms.
- Serving benchmark after warmup: P50 175.19 ms, P95 191.49 ms, 5.68 QPS.
- Serving parity: 20/20 fixed users exactly match the saved offline ranker path.

## Commands

```powershell
python scripts/check_serving_parity.py --users 20
python scripts/benchmark_api.py --requests 100
bash scripts/docker_smoke.sh
python -m pytest -q
docker compose up -d --build
```

The current machine has no Docker CLI, so container startup is explicitly blocked here and must be run on a host with Docker installed. The measured API P95 is above the 100 ms reference because each request performs Python-side ItemCF and feature construction; this is reported rather than hidden by reducing candidate quality.

For one Ubuntu VM deployment, see `deploy/README.md` and `deploy/bootstrap.sh`. GitHub Actions runs unit/API tests, lint, an artifact-aware parity smoke, and a Docker build; it does not retrain the full model.

## Limitations

The data is public Amazon Reviews and evaluation is offline. There are no exposure logs, online feedback, A/B tests, CTR, conversion, or GMV claims. Product metadata price/brand/category fields are incomplete in the formal subset; no constant-filled content features are used.
