# Production-Style Serving Demo

这是一个基于冻结离线模型的 production-style demo，不代表线上业务系统或真实商业收益。

## Architecture

```text
user_id -> history -> Two-Tower/FAISS(100) + ItemCF(100) + Popularity(20) + Content(100)
        -> RRF dedupe/backfill(200) -> model-declared 16/18 features -> LambdaRank -> top-k
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

The content-aware model is available as an isolated serving configuration:
`configs/serving_content.yaml`. It loads the TF-IDF content index and
`ranker_phase6_content.txt` as `recsys_phase6_content_v1`; the original
`recsys_baseline_v1` snapshot remains untouched for rollback.

Feedback is persisted at `data/feedback/events.db` by default. In Compose it is mounted as the `feedback_data` volume. Export it with:

```powershell
python scripts/export_feedback.py --output runs/feedback.jsonl
python scripts/export_feedback.py --output runs/feedback.csv --format csv
```

Every demo event must keep `simulated=true`; exported simulated events must not be mixed into the frozen benchmark as online evidence.

## Observability

`/metrics` returns the JSON view by default. Prometheus scrapes `/metrics?format=prometheus` and Grafana is provisioned by Compose. The exported metrics cover request count, end-to-end latency quantiles, per-stage latency quantiles, candidate count, errors, cache hits/misses, fallback count, recommendation count, and model version. Per-stage traces include history preparation, Two-Tower encoding, FAISS search, ItemCF, Popularity, candidate merge, feature construction, LambdaRank prediction, and result assembly.

Profile the frozen pipeline directly (without HTTP or Redis) to identify the actual bottleneck:

```powershell
python scripts/profile_serving_pipeline.py --users 20 --repeats 5 --warmup 10
```

The command preserves the configured 200-candidate budget and writes reproducible P50/P95/P99 evidence to `results/video_games_2018/serving_stage_profile.json`.

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
  This is the frozen baseline V1 result. The later content ablation regenerates
  retrieval channels under a shared `retrieval_k=200` protocol; its matched
  three-channel baseline is 20.65%, so the four-channel 24.20% result is
  compared against 20.65%, not 21.00%.
- LambdaRank test NDCG@10: 2.99%, versus RRF 1.78%.
- FAISS IndexFlatIP exact-match against NumPy: 100%; single-query P95: 2.15 ms.
- Earlier sequential serving benchmark after warmup: P50 26.59 ms, P95 30.48 ms,
  37.24 requests/s. This is a sequential microbenchmark, not capacity.
- Direct 20-user pipeline profile (100 requests): P95 31.69 ms; feature construction is
  the largest stage at 57.64% of mean latency, while FAISS search is 2.63%.
- Closed-loop concurrency test on one local Uvicorn process and no Redis: about 11–15 QPS
  across concurrency 1–50; P95 grows from 76.58 ms at concurrency 1 to 8,303.33 ms at
  concurrency 50, with zero request errors. The service is functionally stable but
  CPU-bound and not horizontally validated.
- Serving parity: 20/20 fixed users exactly match the saved offline ranker path
  for both the baseline and content-aware configuration.
- Content-aware validation: four-channel ranker NDCG@10 is 0.03848 versus 0.03190
  for retrieval order, with all 18 features reloaded identically. The frozen
  three-channel baseline ranker uses 16 features.

## Commands

```powershell
python scripts/check_serving_parity.py --config configs/serving_content.yaml --users 20
python scripts/profile_serving_pipeline.py --users 20 --repeats 5
python scripts/benchmark_api.py --requests 100
python scripts/load_test_api.py
bash scripts/docker_smoke.sh
python -m pytest -q
docker compose up -d --build
```

The content-aware Compose stack was validated on an AWS Ubuntu host port 8001 while the baseline remained healthy at port 8000. A single request is below the 100 ms reference in the local profile, but P95 breaches that reference under concurrency. Latency remains environment-dependent, so the sequential benchmark, stage profile, and concurrency curve are retained together instead of making a production SLA or capacity claim.

For one Ubuntu VM deployment, see `deploy/README.md` and `deploy/bootstrap.sh`. GitHub Actions runs unit/API tests, lint, and a Docker build; artifact-aware parity is run after restoring the release bundle. CI does not retrain the full model.

## Limitations

The data is public Amazon Reviews and evaluation is offline. There are no exposure logs, online feedback, A/B tests, CTR, conversion, or GMV claims. The content retrieval pipeline recovered high-coverage title, brand, and fine-category metadata for content retrieval, but price and aggregate-rating fields remain excluded. Static metadata availability time cannot be independently verified.
