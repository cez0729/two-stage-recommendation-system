# Runtime artifacts and restoration

This directory holds generated mappings, manifests, feature schemas, and
retrieval indexes. Large binaries and candidate-feature Parquet files are
excluded from Git. Compact experiment results live in `results/`.

## Restore the frozen content-serving version

Download both attachments from the
[content-serving release](https://github.com/cez0729/two-stage-recommendation-system/releases/tag/recsys_phase6_content_v1):

- `recsys_phase6_content_v1_runtime.zip`
- `recsys_phase6_content_v1_runtime.zip.sha256`

Verify the ZIP before extraction. If both downloads are in the current directory:

```bash
sha256sum -c recsys_phase6_content_v1_runtime.zip.sha256
```

PowerShell alternative:

```powershell
$archive = 'recsys_phase6_content_v1_runtime.zip'
$expected = ((Get-Content "$archive.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash $archive -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw 'Runtime archive checksum mismatch' }
```

The published ZIP's SHA-256 is:

```text
c70a4d14e5b728c7e1d484d7c2ed4d0864adcf9689f27d9cb86f749b4eb369b4
```

Extract the ZIP into the repository root, preserving its internal paths.
It provides the content-serving version manifest, Two-Tower checkpoint,
FAISS index and item IDs, TF-IDF artifacts, the content-aware ranker, and
three processed data tables. The ZIP has 11 files, about 22.5 MB compressed.

It does not include the raw Amazon downloads, baseline model snapshot,
saved offline candidate-feature tables, or a real feedback database.
Unit/API tests use synthetic fixtures and do not need the bundle.

Start the restored demo from the repository root:

```bash
docker compose -p recsys-content -f docker-compose.content.yml up -d --build
curl http://127.0.0.1:8001/health
curl "http://127.0.0.1:8001/recommend/A0266076X6KPZ6CCHGVS?k=10"
```

The health response should report `recsys_phase6_content_v1`.
See [the serving guide](../README_SERVING.md) for API and monitoring details.

## Full offline parity checks

The parity script also needs saved offline candidate features. These are not
included in the runtime bundle. In a workspace with the corresponding offline
features and models restored, run:

```bash
python scripts/check_serving_parity.py --config configs/serving_content.yaml --users 20
```

Do not treat a missing candidate-feature file in a fresh clone as a model failure.
Recreating the offline experiment requires the data and training/evaluation
commands in the [project README](../README.md), not just the serving ZIP.
