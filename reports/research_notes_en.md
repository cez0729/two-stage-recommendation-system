# Research notes: retrieval failures, cold start, and reproducibility

This is a retrospective summary of the saved experiments, prepared in October
2026. It is not a backdated development log. Numbers below link to reports,
configurations, or machine-readable outputs already present in the repository.
Dates in older reports describe those experiments, not the date of this summary.

## Problem and evaluation

The study asks whether a two-stage recommender can recover relevant items from
a sparse interaction catalog, including items with no training interactions.
The dataset contains 197,597 processed Amazon Video Games interactions,
19,621 evaluated users, and 14,391 items. The collaborative retrieval-training
catalog contains 13,923 items.

Per-user chronological splits separate retrieval training, ranking training,
validation, and test. Evaluation retrieves from the full catalog rather than
using 99 sampled negatives. Ratings of at least 3 are positive-feedback proxies;
they do not represent exposure, purchase, or click events.

## Decisions and evidence

| Question | Observation | Decision / evidence |
| --- | --- | --- |
| Was the neural model a better starting point? | First Two-Tower test Recall@100 was 3.02%, below Popularity at 6.41% and ItemCF at 15.70%. | Keep the baselines and investigate the failure. The [original failure report](two_tower_failure_analysis.md) records the July 23, 2026 run and its training curve. |
| Would more training solve it? | Extended training reached only 4.88% validation Recall@100. | Investigate sampling rather than attribute all underperformance to epoch count. See [the research report](final_research_report_zh.md). |
| Does logQ help under a controlled comparison? | Across seeds 42, 2026, and 3407, mean validation Recall@100 changed from 4.64% to 11.47%. Coverage@10 fell from 99.10% to 68.31%. | Report accuracy and diversity together. See [three-seed results](../results/video_games_2018/phase6/logq_ablation_summary.json). |
| Should neural retrieval replace ItemCF? | The selected corrected Two-Tower reached 9.51% test Recall@100, still below ItemCF at 15.70%. | Keep complementary retrieval channels instead of claiming neural dominance. See [final neural metrics](../results/video_games_2018/two_tower_v3_final_metrics.json). |
| Can content help without increasing the final pool? | Matched three-channel Candidate Recall@200 was 20.65%; four channels reached 24.20% with the same 200-candidate pool. | Use TF-IDF metadata retrieval and RRF fusion. See [content experiment](../results/video_games_2018/phase6/content_experiment.json). |
| Can metadata cover unseen items? | On 721 strict-cold test users, content Recall@100 was 14.29%; Two-Tower was 0%. | Treat metadata as a separate cold-start path. The difference has a 95% bootstrap interval of 11.79–16.92 percentage points. |
| Does the conclusion depend on per-user time splits? | A separate global-cutoff experiment checks content and collaborative baselines. | Refit eligible methods before the cutoff; do not reuse a neural checkpoint trained on later interactions. See [global temporal results](../results/video_games_2018/phase6/global_temporal_experiment.json). |
| Are retrieval and serving implementations consistent? | FAISS Top-100 matches NumPy for 100 users; saved offline and serving Top-10 match for 20 fixed users per version. | Use exact IndexFlatIP at this catalog size and retain parity checks. See [FAISS benchmark](../results/video_games_2018/faiss_benchmark.json) and [parity script](../scripts/check_serving_parity.py). |
| Does fast sequential serving imply high capacity? | One local process saturates around 11–15 QPS; concurrency 50 raises P95 to 8.30 seconds. | Report the CPU feature-construction bottleneck. See [load-test evidence](../results/video_games_2018/phase6/load_test.json). |
| How does the final candidate budget affect cost? | Sequential mean latency was 22.30/22.56/22.91/25.14 ms at budgets 50/100/200/400. | Keep 200 as the current accuracy/latency operating point and retain the sweep as host-specific evidence. See [budget sweep](../results/video_games_2018/phase6/candidate_budget_benchmark.json). |

## Comparisons that must stay separate

The frozen three-channel baseline uses Two-Tower 100, ItemCF 100, and
Popularity 20 before fusion into 200 candidates. Its test Candidate Recall@200
is 21.00%. The content ablation regenerates each channel with a shared
`retrieval_k=200` protocol. Its direct comparison is **20.65% versus 24.20%**,
not 21.00% versus 24.20%.

The frozen baseline ranker uses 16 features. The content-aware ranker adds
`content_score` and `content_rank`, making 18. Evaluation and serving select
columns using the feature names saved inside the LightGBM model.

The three-seed logQ values are validation means. The 9.51% neural retrieval
value is a selected model's test result. They describe different evaluations.

## Method references

- Covington, Adams, and Sargin (2016), [Deep Neural Networks for YouTube Recommendations](https://doi.org/10.1145/2959100.2959190): a reference for retrieval followed by ranking.
- Yi et al. (2019), [Sampling-bias-corrected neural modeling for large corpus item recommendations](https://doi.org/10.1145/3298689.3346996): a reference for correcting sampled-softmax bias.

Two-Tower, logQ correction, TF-IDF, RRF, and LambdaRank are established methods.
The contribution here is their implementation and controlled evaluation on this
dataset, including failure diagnosis and the measured trade-offs.

## What the evidence does not establish

There are no exposure logs or online A/B tests. Recall gains do not establish
CTR, conversion, revenue, or causal business impact. Static metadata availability
cannot be reconstructed independently. The content channel is separately fitted
TF-IDF, not a jointly trained hybrid item tower. Local load tests are not a
production SLA.

For complete analysis, see [the Chinese research report](final_research_report_zh.md).
For a runnable frozen model, see [the release restoration guide](../artifacts/README.md).
