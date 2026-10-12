# Leakage-Aware Two-Stage Recommendation System

English | [中文](README_ZH.md)

[![CI](https://github.com/cez0729/two-stage-recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/cez0729/two-stage-recommendation-system/actions/workflows/ci.yml)
[Runtime release](https://github.com/cez0729/two-stage-recommendation-system/releases/tag/recsys_phase6_content_v1)

An end-to-end recommendation research prototype built on 197,597 temporally
split Amazon Video Games interactions. It combines collaborative, neural, and
content retrieval with LambdaRank, then serves the frozen pipeline through a
versioned FastAPI/Redis/Docker stack.

The project focuses on reproducible offline evaluation, leakage-aware feature construction, cold-start retrieval, and consistency between offline ranking and online serving.

## Highlights

| Result | Evidence |
| --- | ---: |
| Evaluated users / items | 19,621 / 14,391 |
| Four-channel test Candidate Recall@200 | 24.20% |
| Matched three-channel baseline (content ablation) | 20.65% |
| Strict-cold test Recall@100 | Content 14.29%, Two-Tower 0% |
| Strict-cold test users | 721 |
| Content-aware validation NDCG@10| LambdaRank 3.85%, RRF order 3.19% |
| Offline/serving Top-10 parity | 20/20 users |
| Automated checks | Unit/API tests, Ruff, and Docker build in CI |

The fixed candidate budget remains 200. The content gain therefore comes from
better candidate composition rather than retrieving more items. The strict-cold
comparison is supported by a 95% bootstrap interval and an exact McNemar test.

## Questions investigated

- Why did the first Two-Tower model retrieve fewer relevant items than Popularity?
- Can item metadata recover targets with no retrieval-training interactions?
- Do retrieval gains hold at the same final candidate budget and under global time cutoffs?
- Does the deployed ranking path reproduce offline recommendations?

See the [research notes and evidence map](reports/research_notes_en.md) for the
failed baseline, controlled comparisons, method references, and limitations.

## Extension evidence

The [extension study](reports/extension_study_en.md) adds a second category,
an ordered-history encoder comparison, and serving/index scaling measurements.

| New experiment | Result and boundary |
| --- | --- |
| Musical Instruments, 9,249 test users | Matched Candidate Recall@200: 17.29% → 19.68%; paired 95% gain interval +1.92 to +2.87 points |
| Fresh mean vs GRU history encoder | Video Games Recall@100: 9.39% → 10.19%; coverage@10 falls from 73.40% to 48.67%; one seed |
| 1 vs 4 serving processes, concurrency 20 | Median 19.04 → 35.11 QPS; 3,600/3,600 requests succeed; summed RSS increases about 4x |

Final ranking gains on the second category remain statistically inconclusive.
The million-vector test measures synthetic index cost only. Released serving
models and the original results above remain unchanged.

## Architecture

```mermaid
flowchart LR
    U[User history] --> TT[Two-Tower + FAISS]
    U --> CF[ItemCF]
    U --> POP[Popularity]
    U --> CT[TF-IDF content]
    TT --> RRF[RRF fusion and deduplication]
    CF --> RRF
    POP --> RRF
    CT --> RRF
    RRF --> C[200 candidates]
    C --> F[Model-declared 16 or 18 features]
    F --> L[LightGBM LambdaRank]
    L --> API[FastAPI Top-K API]
    API --> REDIS[Redis cache]
    API --> OBS[Prometheus and Grafana]
    API --> FB[SQLite feedback log]
```

The Content-aware serving version is `recsys_phase6_content_v1`. The original
`recsys_baseline_v1` remains an immutable rollback snapshot.

The frozen three-channel baseline uses 16 ranking features; the content-aware
ranker adds `content_score` and `content_rank` for a total of 18. The 20.65%
three-channel result is the matched baseline for the content ablation, while
the frozen baseline V1 test result is reported separately as 21.00%.

## Research Design

- Per-user temporal splits separate retrieval training, rank training,
  validation, and test interactions.
- Retrieval metrics use the complete training catalog rather than sampled
  negatives.
- TF-IDF vocabulary is fitted only on retrieval-training catalog items, while
  the searchable matrix includes metadata-visible cold items.
- A second evaluation uses pre-registered global cutoffs at 2016-01-01 and
  2017-01-01.
- A controlled three-seed ablation isolates logQ sampling correction. It
  improves accuracy consistently but reduces catalog coverage and increases
  popularity bias; both sides of the trade-off are reported.
- Offline and serving code paths are checked for exact Top-10 parity.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/recsys/data` | Downloading, cleaning, metadata audit, temporal splits |
| `src/recsys/retrieval` | Popularity, ItemCF, Two-Tower, FAISS, TF-IDF content |
| `src/recsys/ranking` | Candidate fusion, feature generation, LambdaRank |
| `src/recsys/serving` | API, cache, feedback, metrics, version resolution |
| `configs` | Reproducible experiment and serving configuration |
| `tests` | Unit, smoke, API, and leakage checks |
| `results` | Compact machine-readable research evidence |
| `reports` | Full Chinese research report and supporting analysis |
| `monitoring` | Prometheus and Grafana configuration |
| `deploy` | Ubuntu VM deployment instructions |

## Quick Verification

Python 3.11 or newer is required.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

The tests use synthetic fixtures and do not require the Amazon dataset or model
artifacts.

## Run the Frozen Demo

Large data and model binaries are deliberately excluded from Git history. A
runtime bundle can be attached to the GitHub Release for
`recsys_phase6_content_v1`. Extract that archive into the repository root, then:

```bash
docker compose -p recsys-content -f docker-compose.content.yml up -d --build
curl http://127.0.0.1:8001/health
curl "http://127.0.0.1:8001/recommend/A0266076X6KPZ6CCHGVS?k=10"
curl "http://127.0.0.1:8001/recommend/unknown-user?k=5"
```

Expected health version: `recsys_phase6_content_v1`. Unknown users use a
deterministic popularity fallback. `k` is restricted to 1 through 50.

Build the release bundle from a fully trained local workspace with:

```bash
python scripts/package_github_release.py
```

## Data

The formal experiment uses the public UCSD McAuley Lab Amazon Reviews 2018
Video Games 5-core dataset:

- [Interactions](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz)
- [Product metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz)

Ratings of at least 3 are treated as positive implicit-feedback proxies. They
are not impressions, clicks, or purchases.

## Engineering Validation

- `recsys_phase6_content_v1` contains checksumed Two-Tower, FAISS, TF-IDF, and
  LambdaRank artifacts.
- The Content-aware Docker stack was validated on an AWS Ubuntu host at port 8001
  while the baseline remained healthy at port 8000.
- Known-user, unknown-user fallback, invalid-parameter, metrics, Redis, and
  health paths were exercised successfully.
- A local one-process load test saturated around 11-15 QPS; P95 latency rose to
  8.30 seconds at concurrency 50. This is reported as a limitation, not an SLA.
- A candidate-budget sweep measured sequential mean latency of 22.30/22.56/22.91/25.14 ms
  at budgets 50/100/200/400. At the current catalog size, the 200-candidate budget is a
  reasonable accuracy/latency operating point; the sweep is host-specific, not a capacity claim.

## Limitations

- Public review data has no exposure logs, online CTR/CVR, purchases, or A/B
  tests, so no commercial lift is claimed.
- Static metadata availability time cannot be independently reconstructed.
- The content channel is interpretable TF-IDF plus score-level RRF, not a jointly
  trained hybrid item tower.
- SQLite and anonymous Grafana access are appropriate for a demo, not a
  multi-tenant production platform.

See [the full Chinese research report](reports/final_research_report_zh.md) and
[the serving guide](README_SERVING.md) for implementation and evaluation detail.
For a candid comparison with stronger research and industry recommender
projects, see the [reviewer audit](reports/reviewer_audit_en.md).
