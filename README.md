# 两阶段电商推荐系统

本项目将实现一套可复现的两阶段推荐系统：先从商品库召回候选商品，再使用排序模型生成最终推荐结果。

当前状态：**正式数据、按用户时间切分、统一评估器、多路召回、FAISS 和 LightGBM LambdaRank 均已实现并完成最终测试**。修复批内负样本采样偏差后，双塔测试 `Recall@100=9.51%`，超过 Popularity 的 `6.41%`；多路候选的 `Recall@200=21.00%`，LambdaRank 测试 `NDCG@10=2.99%`，高于同候选集 RRF 排序的 `1.78%`。完整解释见 `reports/final_research_report_zh.md`。

正式实验使用 19,621 位可评估用户、14,391 个商品和 197,597 条严格时间切分交互。早期 `All_Beauty 2023` 结果仅作为小数据管道验证保留。

## 当前目录说明

- `configs/`：保存可读、可修改的运行参数。
- `src/recsys/`：保存真正可复用的 Python 代码。
- `tests/`：保存自动检查程序，用小样例证明基础代码行为正确。
- `data/`：保存原始数据、中间数据和清洗后的数据；实际数据不提交 Git。
- `artifacts/`：保存 ID 映射、字段定义、FAISS 索引等由程序生成的配套产物。
- `models/`：保存训练好的双塔和精排模型。
- `runs/`：保存每次训练的配置、日志、指标和环境信息。
- `results/`：保存可公开的小型 JSON/CSV 评估结果。
- `reports/`：保存技术报告和图表。

## 安装与检查

```powershell
python -m pip install -e ".[dev]"
python -c "import recsys; print(recsys.__version__)"
python -m ruff check .
python -m pytest
```

完整需求见 `two_stage_recommender_codex_spec_zh.md`。

## 数据流程

```powershell
python -m recsys.data.download --config configs/data.yaml
python -m recsys.data.clean --config configs/data.yaml
python -m recsys.data.split --config configs/data.yaml
```

程序会生成：

- `artifacts/raw_manifest.json`：原始文件来源、字段、大小和哈希。
- `artifacts/data_manifest.json`：每一步过滤后的规模和处理后文件哈希。
- `artifacts/split_stats.json`：四个时间切分的行数和防泄露检查结果。
- `data/processed/interactions.parquet`：带有 `split` 标签的最终交互表。
- `data/processed/items.parquet`、`users.parquet`：商品表和用户统计表。

## 运行召回基线

```powershell
python -m recsys.retrieval.popularity --config configs/popularity.yaml
python -m recsys.retrieval.itemcf --config configs/itemcf.yaml
python -m recsys.evaluation.compare_baselines --results-dir results/video_games_2018/baselines
```

正式实验解释见 `reports/video_games_baseline_evaluation.md`，机器可读结果见
`results/video_games_2018/baselines/retrieval_comparison.csv`。早期小数据结果见
`reports/baseline_evaluation.md`。

## 双塔实验

```powershell
python -m recsys.retrieval.train --config configs/two_tower_v3_logq.yaml
python -m recsys.retrieval.train --config configs/two_tower_v3_final.yaml
```

第一条命令只使用验证集选择模型；第二条命令加载冻结 checkpoint 并执行一次最终测试。调参过程不会反复读取测试集。

## FAISS 与精排

```powershell
python -m recsys.retrieval.faiss_index --config configs/faiss.yaml
python -m recsys.ranking.pipeline --config configs/ranker.yaml
python -m recsys.ranking.evaluate --config configs/ranker.yaml
```

FAISS 命令会验证真实用户 Top-100 与 NumPy 精确点积一致。精排训练只使用 `rank_train`，验证集用于早停；最终评估命令只加载冻结模型并读取测试集。

主要机器可读结果：

- `results/video_games_2018/two_tower_v3_final_metrics.json`
- `results/video_games_2018/faiss_benchmark.json`
- `results/video_games_2018/ranking_validation_metrics.json`
- `results/video_games_2018/final_test_metrics.json`
