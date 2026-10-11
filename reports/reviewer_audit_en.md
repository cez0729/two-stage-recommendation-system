# Reviewer audit: what is strong and what is still missing

This audit compares the repository with the evidence normally expected from a
strong research or ML engineering recommender project. It is an assessment of
the current artifact, not a claim of a public leaderboard ranking.

## Evidence already present

| Area | Evidence in this repository | Assessment |
| --- | --- | --- |
| Evaluation discipline | Per-user chronological splits, full-catalog retrieval, validation-only selection, and separately refit global cutoffs | Strong research practice for an offline public-data study |
| Failure analysis | The first Two-Tower run reached 3.02% test Recall@100, below Popularity at 6.41%; the saved report checks engineering causes before discussing sampling bias | Stronger than reporting only the final model |
| Controlled intervention | Three logQ seeds report both accuracy gains and lower coverage/higher popularity bias | Good ablation discipline; the trade-off is explicit |
| Cold start | TF-IDF metadata retrieval reaches 14.29% Recall@100 on 721 strict-cold users while Two-Tower is 0% | Clear problem-method-result connection |
| Ranking | LambdaRank raises test NDCG@10 from 1.78% to 2.99% on the frozen three-channel candidate set | Validates the second stage without claiming it fixes retrieval misses |
| Systems correctness | FAISS–NumPy exact parity, 20/20 offline–serving Top-10 parity for baseline and content versions, feature-schema compatibility tests | Strong reproducibility and integration evidence |
| Complexity | Candidate-budget sweep, stage profile, and concurrency curve are saved with host and scope limitations | Useful engineering evidence, not a production capacity claim |

## Gaps relative to stronger recommender projects

### 1. No online outcome evidence

The data has ratings but no impressions, exposure logs, clicks, purchases, or
randomized traffic. The project cannot establish CTR, conversion, revenue, or
causal business impact. This is the largest gap compared with an industry
recommender project. The correct response is to state the limitation and show
the offline protocol clearly, rather than infer online gains from recall.

### 2. The model is a strong classical/research prototype, not a current SOTA stack

The final system combines Two-Tower retrieval, ItemCF, TF-IDF, RRF, and
LambdaRank. It does not include a sequential Transformer, contrastive item
content encoder, multi-task objective, calibrated propensity model, or learned
fusion network. This is acceptable for an MLDS or Computational DS project;
it limits claims about novelty. The defensible contribution is the diagnosis,
controlled evaluation, cold-start path, and reproducible serving implementation.

### 3. Generalization is tested within one domain

The formal results use Amazon Video Games. The repository has a Beauty data
configuration, but the headline results are not a multi-domain benchmark. A
stronger paper-style study would repeat the frozen protocol on at least one
second category or a public benchmark such as MovieLens and report whether the
direction of the ablation survives.

### 4. Scale is modest

The catalog has about 14K items, so exact `IndexFlatIP` retrieval is a sensible
choice. It does not demonstrate billion-scale ANN engineering, distributed
training, sharded feature storage, or horizontal serving. The saved load test
shows a single local process saturating around 11–15 QPS and P95 reaching about
8.30 seconds at concurrency 50. These results identify a bottleneck; they do not
represent a production capacity target.

### 5. Reproducibility depends on the release bundle

The source tree, configs, tests, compact results, and release checksum are
public. The raw Amazon data and large model/index files are intentionally not in
Git. A fresh clone therefore needs the release ZIP for the frozen demo, and a
full retraining run needs the public raw data download. This is normal for a
data-heavy project, but it should be stated when sharing the link.

### 6. Monitoring is a demo, not a complete ML lifecycle

FastAPI, Redis, Prometheus, Grafana, feedback logging, and model-version
resolution are implemented. There is no real traffic, automatic retraining,
feature-drift alert, data-quality alert, privacy review, or fairness analysis.
These are appropriate next steps for deployment work, not missing evidence for
the current offline research claim.

## Highest-value next steps

1. Add a second public dataset and reuse the same frozen protocol. This gives
   the strongest improvement in external validity.
2. Add a sequential baseline such as a GRU or Transformer only if the goal is to
   compare model classes. Keep the existing baselines and report compute cost.
3. Add calibration and ranking confidence analysis if the intended application
   needs uncertainty-aware decisions.
4. For deployment work, profile multiple worker processes and a larger synthetic
   catalog; label the result as a scaling experiment.

The first step is the best research extension. A more complex model on the same
single dataset would add less credibility than a second-domain replication.

## Reproduction checks run for this audit

The current workspace passed:

```text
python -m pytest -q                         31 passed
python scripts/check_serving_parity.py ...  baseline 20/20
python scripts/check_serving_parity.py ...  content  20/20
```

The candidate-budget evidence is saved in
`results/video_games_2018/phase6/candidate_budget_benchmark.json`.
