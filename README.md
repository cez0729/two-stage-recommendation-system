# 两阶段电商推荐系统

[English](README_EN.md) | 中文

本项目实现了一套两阶段推荐系统，基于 Amazon Reviews 2018 Video Games 数据集完成离线训练与评估。

候选召回由 Two-Tower + FAISS、ItemCF、Popularity 和 TF-IDF Content Retrieval 组成，并通过 RRF 融合候选；精排阶段使用 LightGBM LambdaRank。

## 主要结果

| 实验                         |     结果 |
| -------------------------- | -----: |
| 四通道 Candidate Recall@200   | 24.20% |
| 三通道 Candidate Recall@200   | 20.65% |
| 严格冷启动 Content Recall@100   | 14.29% |
| 严格冷启动 Two-Tower Recall@100 |  0.00% |

正式实验包含 19,621 位可评估用户、14,391 个商品和 197,597 条时间切分后的交互记录。详细实验、消融和稳健性分析见 `reports/final_research_report_zh.md`。


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

## Phase 6：内容召回与稳健性

```powershell
python scripts/freeze_phase6_baseline.py
python scripts/audit_phase6_metadata.py
python scripts/run_phase6_content.py
python scripts/run_phase6_global_cutoff.py
python scripts/run_phase6_logq_ablation.py
python scripts/load_test_api.py
```

Phase 6 保持每位用户 200 个候选不变。内容通道只使用 `title`、规范化 `brand` 和细粒度类目路径，TF-IDF 词表只在召回训练期可见商品上拟合；`price`、`avg_rating`、`rating_number` 仍因不可用而排除。当前“混合”是可解释的分数级 RRF 融合，不是训练过的 Hybrid Item Tower。

关键结果：

- 元数据审计结论为 GO：商品标题/品牌/细类目覆盖率分别为 99.90%/98.99%/99.05%。
- 测试四通道 Candidate Recall@200 为 24.20%，原三通道为 20.65%。
- 严格冷启动测试用户中，内容 Recall@100 为 14.29%，Two-Tower 为 0%。
- 2017 全局 cutoff 的最终留出集上，内容召回相对协同候选的改进方向仍成立。
- 三随机种子中 logQ 的 Recall@100 均高于 raw，但 Coverage@10 明显下降且热门偏置上升；报告将其作为精度与多样性的权衡，而不是单向胜利。
- 单 Uvicorn 进程吞吐在约 11–15 QPS 饱和；并发 50 时 P95 约 8.30 秒，因此早期顺序请求的 37.24 QPS 不再作为容量结论。

机器可读证据位于 `results/video_games_2018/phase6/`，Baseline V1 哈希清单位于 `artifacts/video_games_2018/phase6/baseline_v1_manifest.json`。
