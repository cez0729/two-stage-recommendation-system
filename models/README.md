# Frozen models

Training writes checkpoints and LightGBM models here. They are excluded from
Git; the [runtime release](https://github.com/cez0729/two-stage-recommendation-system/releases/tag/recsys_phase6_content_v1)
distributes the frozen content-serving artifacts under `artifacts/versions/`.

| Model | Role | Compatibility |
| --- | --- | --- |
| `two_tower_v3_logq.pt` | User/item embeddings for collaborative retrieval | Load checkpoint model arguments and weights together. |
| `ranker.txt` | Frozen three-channel LambdaRank baseline | 16 model-declared features; no content score or rank. |
| `ranker_phase6_content.txt` | Content-aware LambdaRank | 18 model-declared features, including content score and rank. |

Evaluation and serving read feature names from the LightGBM artifact. Feature
count alone is insufficient: column names and order must also match.
Training chooses its schema from the ranker configuration.

The content-serving release does not include `ranker.txt` or the baseline
rollback snapshot. Historical names such as `recsys_phase6_content_v1` identify
published artifacts and are retained for provenance.

See [artifact restoration](../artifacts/README.md),
[experiment configuration](../configs/README.md), and
[research evidence](../reports/research_notes_en.md).
