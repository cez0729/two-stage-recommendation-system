# Cross-category replication, sequence encoding, and serving scaling

This extension asks whether the original findings transfer to a second Amazon
category, whether ordered histories improve neural retrieval, and how the frozen
service behaves with more processes. It does not replace the Video Games release.

## Why these experiments

Relative to mature recommendation research toolkits, the original project had
three gaps: evidence from one category, one neural history encoder, and limited
serving scaling measurements. These experiments address those gaps without
claiming that results from different public projects are numerically comparable.

- [RecBole](https://github.com/RUCAIBox/RecBole) supports multiple datasets and
  recommendation model families. We borrow the principle of a shared evaluation
  protocol; we do not claim a RecBole implementation or benchmark reproduction.
- [GRU4Rec](https://github.com/hidasib/GRU4Rec) models ordered session histories.
  Our experiment changes the history encoder inside the existing Two-Tower. It
  retains user ID embeddings, the existing item tower, and the existing loss.
  This is not an exact GRU4Rec reproduction, and these are user histories rather
  than a published session benchmark.
- [FAISS](https://github.com/facebookresearch/faiss) separates exact and
  approximate vector search, with speed, accuracy, and memory trade-offs. Our
  larger-catalog test measures exact `IndexFlatIP` cost on synthetic vectors.
- [Uvicorn deployment](https://www.uvicorn.org/deployment/) supports multiple
  workers. A process count is an experimental variable, not a guarantee of
  proportional throughput gains.

## Second dataset and unchanged evaluation

Source: [Amazon Reviews 2018, UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
The category is Musical Instruments, using the published 5-core reviews and
category metadata. Download URLs and processing settings are in
[`data_musical_instruments.yaml`](../configs/data_musical_instruments.yaml).
This is cross-category replication from the same source and year, not a second
independent data collection process.

The first alternative inspected was Amazon All Beauty 2023. After applying the
existing strict split, it had only 198 users and 280 items. A 200-item candidate
budget would cover most of that catalog, so it was rejected before model
comparison. Musical Instruments was selected for a meaningful catalog size.

The cleaned Musical Instruments interactions number 175,848 (21,363 users and
8,627 mapped items). Strict timestamp handling and user eligibility retain 9,249
users and 85,389 interactions spanning 8,575 observed items: 57,642
retrieval-training rows plus 9,249 rows each
for rank training, validation, and test. Those numbers describe different stages
of processing, not contradictory dataset sizes. In particular, 12,114 users are
excluded for insufficient distinct timestamps; this limits conclusions to the
eligible population. Metadata matches 99.93% of mapped
items. The collaborative retrieval-training catalog contains 8,169 items.

The replication retains rating >= 3, user/item 5-core filtering, strict per-user
time order, one selected interaction per target timestamp, full-catalog retrieval,
seen-item filtering, and 200 final candidates. Histories grow by split, but model
fitting remains on the permitted earlier split. TF-IDF vocabulary is fitted only
on retrieval-training items; the content matrix also contains metadata-visible
cold items. Metadata visibility is an assumption, not a historical availability
guarantee. Neural hyperparameters, early stopping, ranker settings, and RRF are
copied from the original protocol, without tuning them on the new test set.

Popularity, ItemCF, Two-Tower, TF-IDF, pairwise channel fusions, three-channel and
four-channel retrieval, and matched LambdaRank pipelines are evaluated. Cold and
tail subgroup results and paired confidence intervals use the existing code.
Global-cutoff dates are reused from Video Games; their original percentile
interpretation does not apply automatically to this category.

### Replication results

| Musical Instruments test endpoint | Result |
| --- | ---: |
| Matched three-channel Candidate Recall@200 | 17.29% |
| Four-channel Candidate Recall@200 | 19.68% |
| Paired difference, four minus three | +2.39 percentage points |
| 95% paired bootstrap interval for the difference | +1.92 to +2.87 points |
| Exact McNemar p, candidate comparison | 8.33e-23 |
| Strict-cold content Recall@100 (476 users) | 10.08% (48 hits) |
| Strict-cold Two-Tower Recall@100 | 0% |
| Cold-target paired difference interval | +7.35 to +12.82 points |
| Popularity / ItemCF / Two-Tower Recall@100 | 10.95% / 6.83% / 10.00% |
| Two-Tower + content RRF Recall@100 | 12.29% |

The main content-channel finding transfers to this category at the same final
candidate budget. It does not establish that neural retrieval always beats
simple methods: Two-Tower loses to Popularity here. The magnitude of the content
gain differs from Video Games. These are separate category-specific results,
not a combined score. The paired candidate comparison has 369 four-channel-only
hits and 148 three-channel-only hits.

The FAISS check covers 100 queries: Top-100 sets agree 100%, serialized reload
is identical, but ordered rows agree 98%. This is not perfect ordering parity;
two rows differ in order despite identical neighbors. The index contains 8,169
items. Timing collected while other experiments ran is not used as a capacity
claim. Evidence:
[`content_experiment.json`](../results/musical_instruments_2018/phase6/content_experiment.json)
and [`faiss_benchmark.json`](../results/musical_instruments_2018/faiss_benchmark.json).

The separately refitted global-time robustness check also supports the content
direction. At the 2017 final cutoff, 5,246 eligible queries give Candidate
Recall@200 of 15.15% for Popularity + ItemCF versus 17.04% after adding content;
the paired gain is +1.89 points (95% interval +1.11 to +2.67). Content recalls
32/280 strict-cold targets (11.43%), versus 0 for ItemCF. The Two-Tower checkpoint
is excluded from this check because it was trained under the per-user split.
This robustness check is not a repetition of the identical four-channel model.
See [`global_temporal_experiment.json`](../results/musical_instruments_2018/phase6/global_temporal_experiment.json).

### Complete second-stage replication

The ranking pipeline retains its separate channel request sizes (Two-Tower 100,
ItemCF 100, Popularity 20, and content 100 when enabled), followed by the same
200-candidate pool. Its candidate figures therefore differ from the matched
retrieval ablation above, which requests 200 from each channel.

| Musical Instruments frozen test ranking endpoint | Three channels | Four channels |
| --- | ---: | ---: |
| Candidate Recall@200 in the ranker pipeline | 16.27% | 18.41% |
| RRF-order NDCG@10 | 0.73% | 0.91% |
| LambdaRank NDCG@10 | 1.51% | 1.67% |
| LambdaRank Recall@10 | 2.93% | 3.18% |

Both trained rankers improve on their own retrieval order. However, the paired
four-channel-minus-three-channel NDCG@10 difference is only +0.15 points with
a 95% interval of -0.04 to +0.35 points. The Recall@10 difference likewise has
an interval crossing zero (-0.09 to +0.61 points). This does **not** confirm
an end-to-end ranking improvement from adding content in this category.
The candidate-retrieval finding is stronger than the final Top-10 finding.
All 9,249 users enter this comparison; retrieval misses count as zero. The
reconstructed user-level means are checked against the frozen metric files.
Evidence: [`paired_ranker_comparison.json`](../results/musical_instruments_2018/paired_ranker_comparison.json).

## Sequence model comparison

The controlled mean-versus-GRU comparison uses Video Games, seed 42, 64-dimensional
embeddings, hidden dimension 128, the most recent 20 historical items, batch size
512, learning rate 0.001, dropout 0.1, logQ weight 1, maximum 30 epochs, and
validation Recall@100 early stopping with patience 7. Both models are newly
trained in the same environment. Each frozen checkpoint is evaluated once by
the same exact retrieval code. Test queries include preceding validation history,
as in the original protocol.

The primary endpoint is paired test Recall@100 (GRU minus mean), with 2,000 user
bootstrap resamples and an exact McNemar test on discordant hits. Recall@10 and
NDCG@10 are exploratory endpoints. User-level uncertainty does not establish
stability across random seeds. GRU adds parameters; this is not a capacity-matched
architecture experiment. The existing Video Games test set was already explored
in earlier work. Concurrent training makes recorded epoch times unsuitable for
fair training-speed comparisons. The experiment does not promote a new serving
checkpoint or claim end-to-end GRU ranking gains.

### Sequence results

| Video Games frozen test endpoint | Fresh mean | GRU |
| --- | ---: | ---: |
| Recall@100 | 9.39% | 10.19% |
| Recall@10 | 1.80% | 2.13% |
| NDCG@10 | 0.85% | 1.07% |
| Catalog coverage@10 | 73.40% | 48.67% |
| Popularity bias ratio | 2.25 | 2.85 |
| Parameters | 3,577,984 | 3,602,944 |
| Selected epoch / epochs run | 29 / 30 | 15 / 22 |

The primary gain is +0.80 percentage points (8.46% relative), with a 95% paired
interval of +0.34 to +1.23 points and exact McNemar p=0.000502. GRU alone hits
1,071 users; mean alone hits 915. This supports an accuracy gain for this seed,
but coverage falls and popularity bias rises. GRU also remains below the existing
ItemCF test Recall@100 of 15.70%. It is therefore useful comparative evidence,
not a reason to replace the released pipeline. The fresh mean result is distinct
from the earlier frozen Video Games neural result; it is the controlled
contemporaneous baseline for this comparison.

Evidence: [`paired_comparison.json`](../results/video_games_2018/sequence/paired_comparison.json),
the isolated final metrics in the same directory, and the GRU validation training
trace at `results/video_games_2018/tuning/two_tower_gru_logq_metrics.json`.

## Engineering experiment design

Real HTTP testing uses the frozen content-aware Video Games serving version,
1/2/4 Uvicorn processes, no Redis, one native numerical thread per process, 100
known users, and concurrency 1/5/10/20. Each combination has three repetitions of
100 requests. Successes, errors, latency percentiles, throughput, and summed RSS
are retained. Summed process RSS can double-count shared pages; it is not unique
physical memory. Tests are local and closed-loop, with a fixed workload order.
The client shares the server host. These are measurements on one machine, not
production capacity, an SLA, or a distributed horizontal-scaling experiment.

The catalog experiment separately generates normalized seeded 64-dimensional
float32 vectors at catalog sizes 13,923, 100,000, and 1,000,000. It measures
index-only build time, serialized index size, and 50 warmed single-query Top-100
searches with one FAISS thread. It excludes data preparation, encoder latency,
filtering, RRF, ranking, and network cost. It does not simulate a million-item
behavior dataset, evaluate accuracy, or measure peak process memory. Increasing
the real recommendation catalog would require a separate dataset and evaluation.

### Scaling results

Host: Windows, 20 logical CPUs, 15.73 GiB physical RAM, Python 3.13.14,
PyTorch 2.12.1+cpu, FAISS 1.14.3. The 3,600 measured HTTP requests all succeed.
The table reports the median of three run-level measurements, not pooled
request percentiles or a statistical confidence interval.

| Processes, concurrency 20 | Median QPS | Median run P95 (ms) | Median summed RSS (MiB) |
| --- | ---: | ---: | ---: |
| 1 | 19.04 | 1,400.81 | 844.84 |
| 2 | 30.79 | 900.78 | 1,696.62 |
| 4 | 35.11 | 885.97 | 3,341.40 |

Four processes provide 1.84x throughput and a 36.75% lower P95 than one at
concurrency 20, meeting the declared directional criterion. This is far from
linear 4x scaling and costs about 3.96x summed RSS. Four-process QPS at this
load ranges from 28.11 to 36.71 across the three runs, so the point estimate is
not a guaranteed capacity. At concurrency 1, four processes are slower
(14.16 versus 21.84 median QPS). At concurrency 10, four processes have higher
P95 than two (539.62 versus 508.88 ms). More workers are not universally better.
Two processes give a useful memory/throughput compromise in this short local
study; no production worker count is prescribed and deployment defaults stay
unchanged. Independent ordered sweeps may also reflect host scheduling and
thermal variation.

| Synthetic catalog | Index search P50 (ms) | Index search P95 (ms) | Serialized index (MiB) |
| --- | ---: | ---: | ---: |
| 13,923 | 0.48 | 0.62 | 3.40 |
| 100,000 | 6.22 | 10.74 | 24.41 |
| 1,000,000 | 102.75 | 127.29 | 244.14 |

Exact search remains executable at one million vectors on this host, but its
latency grows substantially. A future large real catalog would justify an
accuracy/latency comparison with approximate search; this study does not provide
that comparison. Index sizes exclude the separately retained input vectors and
temporary allocations. Evidence:
[`scaling.json`](../results/video_games_2018/scaling/scaling.json).

## Acceptance and remaining work

The cross-category candidate/content finding passes the positive paired-interval
check, including the separate global-time robustness check. GRU passes the
single-seed retrieval-accuracy comparison with a substantial coverage trade-off.
The serving sweep passes the directional concurrency-20 criterion and records
the memory cost. The end-to-end four-versus-three-channel ranking comparison is
inconclusive and is not advertised as a proven improvement.

Remaining gaps are independent data sources, multi-seed sequence comparisons,
capacity-matched encoders, stronger sequential baselines, a validated final
ranking gain, approximate-index accuracy experiments, longer randomized serving
tests, and online outcome evidence. For an application portfolio, the contribution
is controlled investigation and reproducible evidence, not a SOTA claim.

## Reproduce

Install the project with `pip install -e ".[dev]"`. Fetch and prepare the new
category with the standard data entrypoints and `configs/data_musical_instruments.yaml`.
Run these from the repository root:

```bash
python -m recsys.data.download --config configs/data_musical_instruments.yaml
python -m recsys.data.clean --config configs/data_musical_instruments.yaml
python -m recsys.data.split --config configs/data_musical_instruments.yaml
python -m recsys.retrieval.popularity --config configs/replication/popularity.yaml
python -m recsys.retrieval.itemcf --config configs/replication/itemcf.yaml
python scripts/audit_content_metadata.py --config configs/replication/content_retrieval.yaml
python -m recsys.retrieval.train --config configs/replication/two_tower_v3_logq.yaml
python -m recsys.retrieval.train --config configs/replication/two_tower_v3_final.yaml
python -m recsys.retrieval.faiss_index --config configs/replication/faiss.yaml
python scripts/run_content_retrieval.py --config configs/replication/content_retrieval.yaml
python -m recsys.ranking.pipeline --config configs/replication/ranker.yaml
python -m recsys.ranking.evaluate --config configs/replication/ranker.yaml
python -m recsys.ranking.pipeline --config configs/replication/ranker_content.yaml
python -m recsys.ranking.evaluate --config configs/replication/ranker_content.yaml
python scripts/compare_replication_rankers.py
python scripts/run_global_cutoff_experiment.py --config configs/replication/content_retrieval.yaml
python -m recsys.retrieval.train --config configs/sequence/mean.yaml
python -m recsys.retrieval.train --config configs/sequence/gru.yaml
python -m recsys.retrieval.train --config configs/sequence/mean_final.yaml
python -m recsys.retrieval.train --config configs/sequence/gru_final.yaml
python scripts/compare_sequence_encoders.py
python scripts/benchmark_scaling.py
python -m ruff check .
python -m pytest
```

Run the scaling script after training and other benchmarks finish. It requires
the original local content-serving artifacts; these are intentionally not
committed. Raw data, checkpoints, indices, and per-user exports remain ignored.
New compact evidence is stored under `results/musical_instruments_2018`,
`results/video_games_2018/sequence`, and `results/video_games_2018/scaling`.

The research settings and outputs are isolated. Existing default constructors
still use the mean encoder, and old checkpoints load without GRU parameters.
No cloud infrastructure, historical release, or old model snapshot is changed.
