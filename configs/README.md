# Experiment and serving configurations

These files separate model selection, frozen test evaluation, retrieval
ablations, and serving. Paths are relative to the repository root.

| Configuration | Purpose | Entry point |
| --- | --- | --- |
| `data.yaml` | Formal Amazon Video Games download, cleaning, and temporal split | `python -m recsys.data.download --config configs/data.yaml`, followed by `recsys.data.clean` and `recsys.data.split` |
| `two_tower.yaml` | Initial neural retrieval experiment | `python -m recsys.retrieval.train --config configs/two_tower.yaml` |
| `two_tower_v2_convergence.yaml` | Extended-training investigation | `python -m recsys.retrieval.train --config configs/two_tower_v2_convergence.yaml` |
| `two_tower_v3_logq.yaml` | Sampling-corrected training with validation selection | `python -m recsys.retrieval.train --config configs/two_tower_v3_logq.yaml` |
| `two_tower_v3_final.yaml` | Frozen neural test evaluation | `python -m recsys.retrieval.train --config configs/two_tower_v3_final.yaml` |
| `faiss.yaml` | Exact inner-product index and NumPy parity benchmark | `python -m recsys.retrieval.faiss_index --config configs/faiss.yaml` |
| `ranker.yaml` | Three-channel baseline, 16 ranking features | `python -m recsys.ranking.pipeline --config configs/ranker.yaml` |
| `ranker_content.yaml` | Content-aware ranking, 18 ranking features | `python -m recsys.ranking.pipeline --config configs/ranker_content.yaml` |
| `content_retrieval.yaml` | Matched channel ablation and global-cutoff parameters | `python scripts/run_content_retrieval.py` |
| `serving.yaml` | Baseline serving, with optional promoted-version pointer | Default API configuration |
| `serving_content.yaml` | Frozen content-serving version | `docker compose -p recsys-content -f docker-compose.content.yml up -d --build` |

Training and evaluation commands can write artifacts and result files. They
are experiment entry points, not prerequisites for reading the published
results or running unit tests. For the frozen demo, restore the
[release bundle](../artifacts/README.md) and use the serving configuration.

## Candidate budgets

The frozen ranker baseline requests Two-Tower 100, ItemCF 100, and Popularity
20, then fuses and backfills to 200 candidates. The content-aware ranker adds
100 content candidates but keeps the final pool at 200.

The separate retrieval ablation uses `retrieval_k=200` for channel generation
and `report_k=100` for reported Top-100 retrieval metrics. Its matched
three-channel Candidate Recall@200 is 20.65%; the four-channel result is
24.20%. The frozen baseline's 21.00% is a different configuration.

## Evaluation boundaries

`rank_train` trains LambdaRank; validation chooses early stopping. Final
test evaluation loads the selected model. Content vocabulary is fitted only
on the retrieval-training catalog. Global-cutoff checks use separately fitted
methods rather than a checkpoint exposed to later interactions.

Historical `phase6` output paths and version names are retained to keep existing
metrics and released artifacts traceable. See the
[research evidence map](../reports/research_notes_en.md) for interpretation.
