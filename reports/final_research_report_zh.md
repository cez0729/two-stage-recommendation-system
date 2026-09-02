# 两阶段推荐系统最终研究报告

## 1. 一句话结论

项目实现了从数据处理、时间切分、多路候选召回、Two-Tower、FAISS 到 LambdaRank 精排和在线 serving 的完整两阶段推荐流水线，并通过冷启动实验、全局时间稳健性实验、多随机种子消融和并发压测进行验证。所有结果均来自公开数据上的离线实验和工程测试，不代表真实线上 CTR、CVR、GMV 或其他商业指标。

## 2. 数据与评估口径

- 数据：Amazon Reviews 2018 `Video_Games` 5-core。
- 最终交互：197,597 条；可评估用户：19,621；商品：14,391。
- 召回训练目录：13,923 个商品。
- 每位用户按不同日期排序，最后三次行为依次作为 `rank_train`、`validation`、`test`。
- 所有主召回指标均为完整商品目录检索，不使用“1 个正样本 + 99 个采样负样本”的宽松口径。
- 调参只看验证集；测试集仅由冻结的双塔和 ranker 各执行一次。

主实验的“时间切分”是按用户的时间顺序。Phase 6 另外预注册 `2016-01-01` 验证 cutoff 和 `2017-01-01` 最终 cutoff，完成了 global temporal robustness check。该实验不复用原双塔 checkpoint，因为旧 checkpoint 见过 cutoff 之后的数据；它只比较可在新时间窗口中重新拟合的 Content、Popularity、ItemCF 及其融合。

## 3. 关键问题与修复

第一版双塔测试 `Recall@100=3.02%`，低于 Popularity 的 `6.41%`。工程检查证明 checkpoint、向量、已见过滤和全目录检索没有错误；继续训练到 30 轮后，验证集也只有 `4.88%`。

真正的问题是批内负样本采样偏差。热门商品更频繁出现在 batch 中，因此也更频繁被当成负例；未校正模型推荐商品的平均热度只有真实目标的 39%。加入文献常用的 logQ 校正后，模型学习时会扣除商品被抽中的对数概率，避免把“常被抽到”误认为“更应该被压低”。

Phase 6 用 `42/2026/3407` 三个随机种子做了受控消融：两组使用完全相同的数据、结构、30 个最大 epoch、优化器、调度器和 batch size，唯一机制差异是 `sampling_correction_weight`。logQ 的验证 Recall@100 在三个种子上都高于 raw，均值从 `4.64% ± 0.74%` 提升到 `11.47% ± 0.46%`；NDCG@10 和 Recall@10 也三组同向。但 Coverage@10 从 `99.10%` 降到 `68.31%`，热门偏置比从 `0.40` 升到 `2.09`。因此 logQ 是明显的准确率修复，同时带来覆盖和热门偏置代价。

修复后双塔：

| 切分 | Recall@10 | Recall@100 | NDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| 验证 | 2.42% | 11.48% | 1.13% | 74.01% |
| 测试 | 1.81% | 9.51% | 0.88% | 74.31% |

测试对比：Popularity `Recall@100=6.41%`，ItemCF `15.70%`。双塔已通过最低门槛，但仍不应宣称取代 ItemCF；正式架构采用多路召回。

## 4. FAISS 结果

使用 `IndexFlatIP` 对 13,923 个 L2 归一化的 64 维商品向量建立精确索引。

- 100 位真实用户的 FAISS Top-100 与 NumPy 全量点积逐行完全一致。
- 保存后重新加载的推荐结果完全一致。
- 索引大小：3,564,333 字节，约 3.40 MiB。
- 单用户 P50/P95：0.74/2.15 ms。
- 128 用户批量 P95：168.11 ms，即摊销 1.31 ms/用户。

当前目录只有约 1.4 万商品，精确索引已经足够。此规模下改用 IVF/HNSW 不会带来有说服力的业务收益，反而增加近似误差；把它列为可选扩展比强行使用更合理。

## 5. 多路候选与 LambdaRank

每个查询融合：Two-Tower 100、ItemCF 100、Popularity 20，经 RRF 去重并补齐到 200。`rank_train` 中目标真实进入候选的查询才用于 LambdaRank；验证和测试不强塞正样本。

使用 16 个特征：双塔/ItemCF/热门分数与排名、来源数、RRF、用户历史长度和评分统计、商品训练期热度与评分统计等。原始商品表的 `price`、`avg_rating`、`rating_number` 在正式子集中 100% 为空，因此被删除，没有用常数填充制造“特征数量”。两个可能包含跨用户未来信号的相对商品时间特征也在最终模型中删除。

参数：LambdaRank，学习率 0.05，最多 500 棵树，31 叶，`min_child_samples=30`，行/列采样 0.9，L2 正则 1.0；验证 NDCG@10 早停，最佳迭代 83。

| 指标 | 验证 RRF | 验证 Ranker | 测试 RRF | 测试 Ranker |
| --- | ---: | ---: | ---: | ---: |
| Candidate Recall@200 | 23.37% | 23.37% | 21.00% | 21.00% |
| Recall@10 | 4.46% | 6.95% | 3.60% | 5.66% |
| NDCG@10 | 2.29% | 3.62% | 1.78% | 2.99% |
| Recall@20 | 7.02% | 10.39% | 5.55% | 8.80% |

测试 NDCG@10 相对提升约 67.7%，Recall@10 相对提升约 57.2%。Ranker 不能提高 Candidate Recall，因为没有进入候选的商品无法靠排序找回来。

## 6. 统计与分组结论

基于 19,621 位相同测试用户做成对 bootstrap：

- Recall@10 绝对提升 2.06 个百分点，95% CI `[1.77, 2.34]`。
- NDCG@10 绝对提升 1.21 个百分点，95% CI `[1.04, 1.37]`。
- Ranker 新增命中 623 人、损失 219 人，McNemar 精确检验 `p=1.28e-45`。

按用户历史长度分组后，短、中、长历史用户的 Ranker Recall@10 分别为 5.57%、5.79%、5.64%，提升并非只来自重度用户。

在 Phase 6 之前，主要短板在商品侧：

- 721 个测试目标从未出现在训练目录，候选召回必然为 0。
- 尾部商品 Candidate Recall@200 约 10.35%，头部商品约 36.85%。
- 头部商品 Ranker Recall@10 为 12.52%，尾部只有 1.08%。

Phase 6 已落实这一方向，结果见第 11–13 节。它解决了“协同模型对训练期从未出现商品必然为 0”的结构性问题，但没有消除所有长尾差距。

## 7. 通俗解释

可以把系统想成两次考试：

1. 召回先从 1.4 万件商品里选 200 件“可能合适的”。当前每 100 个真实目标，大约 21 个能进入这 200 件。
2. 精排再把这 200 件重新排队。原来的简单规则只能让约 3.6 个目标进入前 10，学习排序后变成约 5.7 个。

第一关漏掉的商品，第二关无法救回来。所以现在最值得提升的是候选召回，尤其是没有历史或很少出现的商品，而不是继续微调已经有效的排序器。

## 8. Engineering Takeaways

本项目的主要工程与实验结论包括：

1. Full-catalog evaluation 暴露了 sampled-negative evaluation 可能掩盖的召回问题。
2. logQ correction 显著改善了 Two-Tower 的召回准确率，但同时降低 Coverage，并增加热门商品偏置。
3. LambdaRank 能有效改善已召回候选的 Top-K 排序，但无法修复候选阶段遗漏的目标商品。
4. Content Retrieval 为协同模型无法覆盖的冷启动商品提供了互补召回能力。
5. 离线与 serving 路径的一致性测试能够降低训练、评估和在线推理实现不一致的风险。
6. 并发压测表明当前单进程 serving 的主要扩展瓶颈在 CPU 特征构造，而不是 FAISS 检索。


## 9. 关键产物

- 双塔最终指标：`results/video_games_2018/two_tower_v3_final_metrics.json`
- FAISS 基准：`results/video_games_2018/faiss_benchmark.json`
- 精排验证：`results/video_games_2018/ranking_validation_metrics.json`
- 最终测试：`results/video_games_2018/final_test_metrics.json`
- 分组分析：`results/video_games_2018/final_group_analysis.csv`
- 双塔 checkpoint：`models/video_games_2018/two_tower_v3_logq.pt`
- LambdaRank 模型：`models/video_games_2018/ranker.txt`
- Phase 6 元数据审计：`results/video_games_2018/phase6/metadata_audit.json`
- 内容召回主实验：`results/video_games_2018/phase6/content_experiment.json`
- 全局时间稳健性：`results/video_games_2018/phase6/global_temporal_experiment.json`
- 三随机种子消融：`results/video_games_2018/phase6/logq_ablation_summary.json`
- 并发压测：`results/video_games_2018/phase6/load_test.json`

## 10. Serving 阶段

按照下一阶段规格，系统已封装为 production-style demo service：

- `ServingPipeline` 一次加载双塔、FAISS、ItemCF、Popularity、Content 和 LambdaRank，在线流程与离线流程一致。
- Phase 6 已形成独立版本 `recsys_phase6_content_v1`，已知用户走 `100 + 100 + 20 + 100 -> RRF -> 200 -> 18 特征 -> LambdaRank`；未知用户稳定回退 Popularity。
- 20 个固定用户的 serving top-10 与保存的离线 ranker 路径 `20/20` 一致。
- FastAPI 提供 `/health`、`/recommend/{user_id}`、`/metrics`；`k` 限制在 1 到 50。
- Redis key 为 `rec:v1:{user_id}:{k}`，TTL 300 秒；Redis 连接失败设置 50 ms 超时并自动 bypass。
- 早期顺序请求微基准为 P50 `26.59 ms`、P95 `30.48 ms`、37.24 请求/秒；它只描述串行单请求路径，不能作为系统容量。
- 20 位已知用户共 100 次直接流水线测量的 P95 为 `31.69 ms`，每次候选数均为 `200`。
- Phase 6 已在 AWS Docker 主机的独立 8001 测试栈完成验收：API 与 Redis 均 healthy，已知用户、未知用户 fallback、非法参数和指标接口均通过；云端旧 8000 基线保持运行，可随时回滚。

分阶段 profiling 证明特征构造占平均总耗时 `57.64%`，LambdaRank 预测占 `16.91%`，FAISS 只占 `2.63%`。Phase 6 的闭环并发压测显示，单 Uvicorn 进程在并发 1–50 时吞吐约为 11–15 QPS；并发 1 的 P95 为 `76.58 ms`，并发 5/10/20/50 分别升至 `518.70/1013.48/2142.06/8303.33 ms`，请求错误均为 0。系统功能稳定，但单进程 CPU 扩展性差；这些本地数字不构成生产 SLA。

## 11. Phase 6 元数据审计与 Go/No-Go

内容召回只有在元数据足够完整时才值得开展，因此先执行元数据质量门禁，而不是直接堆模型。Baseline V1 的 16 个关键数据、配置、模型、索引和结果资产均记录 SHA-256，避免新实验覆盖旧结论。

| 审计对象 | 覆盖率 | 结论 |
| --- | ---: | --- |
| 原始元数据与正式目录匹配 | 99.90% | 可用 |
| Title | 99.90% | 可用 |
| Brand | 98.99% | 可用 |
| 细粒度类目 | 99.05% | 可用 |
| 严格冷商品 Title | 100.00% | 可用 |
| 严格冷商品 Brand | 99.03% | 可用 |
| 严格冷商品细类目 | 97.78% | 可用 |

审计结论为 **GO**。内容模型只使用 title、规范化 brand 和细粒度 category path；`price`、`avg_rating`、`rating_number` 仍不可靠，因此继续排除。该决策避免用常数填充或不可验证字段制造虚假复杂度。

## 12. 内容召回设计

内容基线使用 TF-IDF：最大 50,000 个特征、1–2 gram、`min_df=2`，用户表示由最近最多 50 个历史商品内容向量聚合得到。词表只在 retrieval-train 可见商品的元数据上拟合，但检索矩阵覆盖所有元数据可见目录商品，因此能召回训练期没有交互的新商品。

每条通道和融合结果都严格返回 200 个候选。融合使用 score-level RRF，当前不能称为“训练过的 Hybrid Item Tower”，因为没有联合训练内容塔和协同塔。

| 指标 | 验证 | 测试 |
| --- | ---: | ---: |
| Content Recall@100 | 13.89% | 12.76% |
| Two-Tower Recall@100 | 11.48% | 9.51% |
| Two-Tower + Content RRF Recall@100 | 16.17% | 14.22% |
| 原三通道 Candidate Recall@200 | 23.45% | 20.65% |
| 四通道 Candidate Recall@200 | 27.25% | 24.20% |
| 原三通道 Overall Recall@100 | 17.37% | 14.76% |
| 四通道 Overall Recall@100 | 20.43% | 17.59% |

测试集中，四通道相对三通道新增命中 1,079 位用户，同时损失 382 位用户；Candidate Recall@200 净增 3.55 个百分点。候选预算未增加，所以收益来自候选构成变化，而不是多取候选。

## 13. 严格冷启动与长尾结论

“严格冷启动”指测试目标商品在 retrieval-train 中完全没有交互。协同双塔没有可学习的商品交互表示，所以该组理论和实测命中均为 0；内容模型可以依据标题、品牌和类目建立表示。

- 测试严格冷用户 721 位，Content 命中 103 位，Recall@100=`14.29%`；Two-Tower 为 `0%`。
- 冷启动差值 95% CI 为 `[11.79%, 16.92%]`，McNemar 精确检验 `p≈1.97e-31`。
- 验证严格冷 Recall@100 为 `12.71%` 对 `0%`，差值区间同样不跨 0。
- 测试尾部商品上，Two-Tower + Content RRF Recall@100=`8.18%`，Two-Tower=`0.29%`。

通俗地说，协同模型只能根据“以前谁看过它”理解商品；新商品没有这类记录。内容召回则能根据“它叫什么、是什么品牌、属于哪一类”寻找相似商品，因此补上了原系统无法处理的一块。

## 14. 全局时间稳健性

主实验按每位用户自己的时间顺序切分，Phase 6 又增加统一日期切分：`2016-01-01` 到 `2017-01-01` 为验证窗口，`2017-01-01` 之后为最终留出。日期在读取结果前写入配置，没有根据指标事后调整。

最终留出包含 6,102 位可评估用户和 1,227 位严格冷用户：

| 指标 | 结果 |
| --- | ---: |
| Content Overall Recall@100 | 11.39% |
| ItemCF Overall Recall@100 | 8.73% |
| Content 严格冷 Recall@100 | 15.73% |
| ItemCF 严格冷 Recall@100 | 0.00% |
| Content+协同 Candidate Recall@200 | 16.49% |
| 仅协同 Candidate Recall@200 | 11.73% |
| 融合 Tail Recall@100 | 6.51% |
| 仅协同 Tail Recall@100 | 4.24% |

这说明内容召回改善冷启动/长尾的方向在更严格的统一时间协议下仍然成立。限制是 Amazon 静态元数据快照没有字段级发布时间，无法独立证明每段文本在当时已经上线；报告因此只称其为稳健性证据，而不称为完全无偏的历史回放。

## 15. 研究结论分层

**离线研究证据**：内容召回显著改善严格冷商品，四通道在固定候选预算下提高 Candidate Recall；logQ 的准确率收益在三个随机种子上重复出现，但伴随覆盖率下降和热门偏置上升。

**工程证据**：模型/索引/配置有哈希清单；Phase 6 离线/Serving 固定用户 parity 为 20/20；新 ranker 使用 18 个特征且重载预测一致；API 在本地并发压测中无错误，但单进程约 11–15 QPS 即饱和；AWS Docker 独立测试栈已实际验收，旧基线仍可回滚。

**不能声称的业务证据**：没有真实 impression/click/purchase 日志，没有线上对照实验，不能声称 CTR、CVR、GMV 提升，也不能用本地 P95 宣称生产 SLA。

## 16. 最终判断

项目最有价值的部分不是模型数量，而是两条完整研究链：第一条从双塔异常低分定位到 in-batch sampling bias，用三随机种子证明 logQ 的准确率收益并揭示多样性代价；第二条从冷启动失败出发，经过元数据 Go/No-Go、无泄露内容基线、固定候选预算、分组显著性和全局 cutoff 稳健性，证明内容通道能修补协同系统的结构性盲区。

面试时可用一句话概括：

> 在 19.8 万条公开 Amazon 交互上构建两阶段推荐系统，完成完整目录评估、采样偏差诊断、FAISS 与 LambdaRank；进一步用高覆盖元数据加入内容召回，使固定 200 候选的测试 Candidate Recall@200 从 20.65% 提升到 24.20%，并把 721 位严格冷用户的 Recall@100 从协同模型的 0% 提升到 14.29%，同时用全局时间切分、三随机种子和并发压测公开验证边界。
