# Two-Stage Recommendation System：技术设计与实验说明

本文档记录项目的核心技术设计、数据处理流程、离线实验、在线服务架构以及主要工程决策。

项目基于 Amazon Reviews 2018 Video Games 数据集，实现从数据处理、候选召回、Learning-to-Rank 精排到在线 serving 的完整两阶段推荐流水线。文档中的实验结果均来自离线评估或工程测试，不代表真实线上业务指标。

---

## 1. 项目一句话介绍

这是一个可复现的两阶段电商推荐系统：

1. **召回（retrieval）**：从约 1.4 万个商品中快速找出约 200 个可能相关的候选。
2. **排序（ranking）**：用 LambdaRank 学习排序模型，把 200 个候选重新排序，返回前 `k` 个。
3. **在线服务（serving）**：用 FastAPI 提供接口，Redis 缓存结果，SQLite 记录反馈，Prometheus/Grafana 提供监控。

Baseline V1 的 Serving 包括三条召回通道：

- Two-Tower + FAISS：学习用户向量和商品向量的语义匹配。
- ItemCF：根据用户历史商品寻找相似商品。
- Popularity：根据训练期交互次数生成热门商品兜底。

Phase 6 的研究在固定 200 候选预算下增加第四条 Content 通道：使用标题、品牌和细粒度类目构造 TF-IDF 商品表示。新 ranker 已重新训练，18 特征 Serving 已完成 20/20 parity，并在云端独立 8001 Docker 测试栈通过验收；8000 Baseline V1 仍保留为回滚版本。

本项目定位为可复现的离线推荐系统研究原型和 production-style demo。系统使用公开评论数据进行训练与评估，不包含真实曝光日志、线上点击率、转化率、GMV 或 A/B 测试，因此项目结果仅用于评价模型和系统本身的离线性能。


---

## 2. 项目要解决的问题

电商平台有大量商品和用户行为。如果每次请求都给用户遍历、打分全部商品，成本会随着商品数增加。工业推荐通常拆成两阶段：

```text
全部商品（约 14K）
        |
        | 召回：速度优先，缩小搜索空间
        v
候选商品（约 200）
        |
        | 排序：精度优先，使用更多特征
        v
最终推荐（Top-k）
```

本项目的研究问题不是简单调用一个模型，而是回答以下问题：

- 如何从公开评论数据构造合理的推荐任务？
- 如何避免把未来行为泄露给训练或评估？
- 传统协同过滤、热门推荐和神经召回各自有什么优缺点？
- 如何验证 FAISS 近似/向量检索没有破坏正确性？
- 如何把多路召回交给学习排序模型？
- 如何让离线评估路径和线上服务路径完全一致？
- 如何面对冷启动、长尾、缺失商品元数据和 Redis 故障？
- 如何将模型、索引、配置、指标和版本一起管理？

---

## 3. 项目当前状态和主要证据

截至本文生成时，离线研究资产已经生成，云端实例也曾成功启动完整服务。当前最重要的可验证事实如下：

| 项目 | 当前结果 |
| --- | ---: |
| 正式交互数量 | 197,597 |
| 可评估测试用户 | 19,621 |
| 训练商品目录 | 13,923 |
| 正式商品总数 | 14,391 |
| Two-Tower 测试 Recall@100 | 9.51% |
| ItemCF 测试 Recall@100 | 15.70% |
| Popularity 测试 Recall@100 | 6.41% |
| 多路候选 Candidate Recall@200 | 21.00% |
| LambdaRank 测试 Recall@10 | 5.66% |
| LambdaRank 测试 NDCG@10 | 2.99% |
| 同候选集 RRF 测试 NDCG@10 | 1.78% |
| LambdaRank 相对 RRF 的 NDCG@10 提升 | 约 67.7% |
| FAISS Top-100 与 NumPy 全量结果一致率 | 100%（100 位用户） |
| FAISS 单查询 P95 | 2.15 ms |
| 离线/线上固定用户 Top-10 一致率 | 20/20 |
| 本地自动化测试 | 30 项通过 |
| Ruff 静态检查 | 通过 |
| 云端 `/health` | `HTTP 200`，模型已加载 |
| 元数据质量门禁 | GO；Title/Brand/细类目覆盖率 99.90%/98.99%/99.05% |
| 四通道测试 Candidate Recall@200 | 24.20%（原三通道 20.65%） |
| 严格冷启动测试 Recall@100 | Content 14.29%；Two-Tower 0%（721 用户） |
| 全局 2017 cutoff Candidate Recall@200 | Content+协同 16.49%；仅协同 11.73% |
| 三种子 logQ 验证 Recall@100 | 11.47% ± 0.46%（raw 4.64% ± 0.74%） |
| 单进程并发容量观测 | 约 11–15 QPS；并发 50 的 P95 约 8.30 秒 |

这些结果说明系统的工程闭环成立：数据、训练、评估、索引、服务和监控能够连接起来。它们不等同于商业收益。

---

## 4. 目录结构：每个文件夹和文件是做什么的

```text
推荐系统/
├─ src/recsys/                 # 可复用的核心 Python 包
│  ├─ data/                    # 下载、清洗、字段校验、时间切分
│  ├─ retrieval/               # Popularity、ItemCF、Two-Tower、FAISS
│  ├─ ranking/                 # 候选特征、LambdaRank、评估
│  ├─ evaluation/              # 统一指标和基线比较
│  ├─ serving/                 # FastAPI、在线 pipeline、Redis、反馈、版本
│  └─ utils/                   # IO、随机种子和通用辅助函数
├─ configs/                    # 所有实验配置，避免把参数硬编码在代码中
├─ data/                       # 原始、中间、处理后数据；通常不提交 Git
├─ artifacts/                  # ID 映射、FAISS 索引、schema、版本 manifest
├─ models/                     # PyTorch 双塔 checkpoint 和 LightGBM ranker
├─ results/                    # 可公开的 JSON/CSV/Parquet 结果
├─ runs/                       # 每次运行的日志、配置、环境和中间产物
├─ reports/                    # 研究报告、基线报告和图表
├─ tests/                      # 小样例单元测试、API 测试和算法 smoke test
├─ monitoring/                 # Prometheus 配置和 Grafana dashboard
├─ deploy/                     # VM、Docker、CloudShell 部署脚本和说明
├─ Dockerfile                 # API 镜像构建文件
├─ docker-compose.yml         # API、Redis、Prometheus、Grafana 编排
├─ pyproject.toml             # Python 依赖、打包和 Ruff/Pytest 配置
├─ README.md                  # 项目快速入口
└─ README_SERVING.md          # 在线服务、接口和监控说明
```

### 4.1 `src/recsys/data/`

- `download.py`：下载评论和商品元数据，记录 URL、文件大小、哈希和字段探测结果。
- `clean.py`：去重、过滤评分、过滤低频用户/商品，建立标准化表。
- `split.py`：按用户行为时间切分训练、验证和测试。
- `schemas.py`：字段结构和类型约束。
- `metadata.py`：解析、规范化并审计 title、brand 和细粒度类目，生成内容召回商品表。

### 4.2 `src/recsys/retrieval/`

- `popularity.py`：训练期商品热度基线。
- `itemcf.py`：基于商品共现的 Item-based Collaborative Filtering。
- `two_tower.py`：用户塔、商品塔及向量编码逻辑。
- `dataset.py`：双塔训练样本和用户历史构造。
- `train.py`：双塔训练、验证、checkpoint 和指标输出。
- `faiss_index.py`：用商品向量建立 `IndexFlatIP`，并做精确性基准。
- `candidate_filter.py`：过滤已看商品、去重和候选约束。
- `content.py`：TF-IDF 内容表示、用户历史聚合和内容相似度召回。

### 4.3 `src/recsys/ranking/`

- `pipeline.py`：生成多路候选、RRF 融合、构建精排特征和训练 LambdaRank。
- `evaluate.py`：加载冻结 ranker，在验证/测试集上评估。

### 4.4 `src/recsys/serving/`

- `api.py`：FastAPI 路由：`/health`、`/recommend/{user_id}`、`/events`、`/metrics`。
- `pipeline.py`：启动时一次性加载离线模型、索引和商品表；线上推荐路径。
- `cache.py`：Redis 读写，TTL 300 秒，Redis 故障时自动旁路。
- `events.py`：SQLite 反馈事件表和写入逻辑。
- `versioning.py`：读取当前 promoted model pointer。
- `artifacts.py`：检查/构建 serving artifact manifest。

---

## 5. 数据来源和字段含义

### 5.1 正式数据来源

正式实验使用 UCSD McAuley Lab 发布的 Amazon Reviews 2018 Video Games 5-core 数据：

- 评论/交互：
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz`
- 商品元数据：
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz`

配置位置：`configs/data.yaml`。

早期还保留了 `All_Beauty 2023` 数据管道，用于小数据开发验证，但它不是正式主结果来源，不应和 Video Games 的结果混为一谈。

### 5.2 原始评论字段

| 原始字段 | 处理后概念 | 含义 |
| --- | --- | --- |
| `reviewerID` | `user_id` | 用户的稳定标识 |
| `asin` | `item_id` | Amazon 商品标识 |
| `unixReviewTime` | `timestamp` | Unix 秒级时间戳，转换为时间 |
| `overall` | `rating` | 用户给出的评分，通常为 1–5 |

配置中 `min_rating: 3.0`，表示只保留评分大于等于 3 的交互，把它作为“用户对商品有正向兴趣”的隐式反馈近似。它不是点击或购买，所以不能直接解释为真实转化。

### 5.3 原始商品字段

| 原始字段 | 处理后概念 | 含义 |
| --- | --- | --- |
| `asin` | `item_id` | 商品 ID，与评论表连接 |
| `title` | `title` | 商品标题；Phase 6 用于内容召回 |
| `brand` | `brand` | 规范化品牌；Phase 6 用于内容召回 |
| `category` | `fine_category_path` | 多级细粒度类目路径；Phase 6 用于内容召回 |
| `main_cat` | `category` | 商品主类目；用于审计和展示 |
| `price` | `price` | 商品价格；正式子集中缺失比例过高，没有强行填常数使用 |
| `avg_rating` | `avg_rating` | 元数据平均评分；正式源字段为空 |
| `rating_number` | `rating_number` | 元数据评分数；正式源字段为空 |

### 5.4 每个数据目录的意义

- `data/raw/`：从外部 URL 下载的原始压缩文件和断点分片。它是数据来源的原始证据，不应手工改动。
- `data/interim/`：解析或初步清洗后的临时表。
- `data/processed/video_games_2018/interactions_clean.parquet`：去重、评分和频次过滤后的交互。
- `data/processed/video_games_2018/interactions.parquet`：带 `split` 标签的最终交互表。
- `data/processed/video_games_2018/items.parquet`：商品索引、商品 ID、标题及商品属性。
- `data/processed/video_games_2018/users.parquet`：用户索引及统计信息。
- `artifacts/video_games_2018/mappings/`：原始字符串 ID 与连续整数 `user_idx/item_idx` 的映射。
- `artifacts/video_games_2018/data_manifest.json`：每一步数据规模、哈希和处理参数。
- `artifacts/video_games_2018/split_stats.json`：切分行数、用户数和防泄露检查。

### 5.5 为什么要把字符串 ID 映射成整数

神经网络 embedding 和 FAISS 索引不能直接高效使用任意字符串。系统把：

```text
Amazon user_id  -> user_idx: 0, 1, 2, ...
Amazon item_id  -> item_idx: 0, 1, 2, ...
```

模型内部使用整数索引；API 返回时再查回原始 `item_id` 和 `title`。映射文件是模型可复现和线上正确返回商品的关键。

---

## 6. 数据清洗和时间切分

### 6.1 清洗规则

正式配置使用：

- 项目随机种子：`42`。
- 用户-商品去重：同一用户同一商品保留一条有效交互。
- 最低评分：`3.0`。
- 用户最少交互数：`5`。
- 商品最少交互数：`5`。
- 最高交互上限：`1,500,000`，用于防止异常输入无限膨胀。

低频用户被过滤，是因为没有足够历史构造序列和评估目标；低频商品被过滤，是为了让 ItemCF 和双塔训练有基本统计稳定性。过滤的代价是系统并不覆盖整个 Amazon 商品世界。

### 6.2 用户级时间切分

对每一个用户，按时间从早到晚排序，最后几次行为分配为：

```text
较早历史      -> retrieval_train
倒数第 3 次   -> rank_train
倒数第 2 次   -> validation
最后一次      -> test
```

具体实现由 `configs/data.yaml` 的 `same_timestamp_policy: select_one_per_target_timestamp` 控制：当同一用户的多个事件时间戳相同时，目标时间戳只选择一个样本，避免把无法辨识的顺序当成真实顺序。

这不是全局时间窗口切分，而是“每个用户自己的时间顺序切分”。优点是每个用户都能产生预测目标；限制是它不完全模拟一个全球统一日期上线的场景。Phase 6 已追加预注册的 `2016-01-01` 验证 cutoff 和 `2017-01-01` 最终 cutoff，内容召回改善冷启动和候选召回的方向在统一日期协议下仍然成立。静态元数据的历史可用时间仍无法独立验证。

### 6.3 防止数据泄露

训练、验证和测试之间的核心原则：预测某个目标时，只允许使用这个目标之前的行为。

- Popularity 只在 `retrieval_train` 统计。
- ItemCF 只在 `retrieval_train` 构建邻居。
- 双塔训练只使用训练交互，验证用于选择 checkpoint，测试只在最终冻结模型上跑一次。
- 精排模型只用 `rank_train` 训练，validation 做早停，test 只用于最终评估。
- 精排特征中的商品热度、评分统计均来自训练期。
- 发现商品相对时间特征可能跨用户携带未来信息后，最终模型将其移除。
- 原始元数据中缺失的 price/avg_rating/rating_number 没有用常数填充制造假信号。

---

## 7. 评价任务、目标和指标

### 7.1 评价任务

对于每个可评估用户，系统知道用户过去的历史，隐藏最后一次正向交互作为目标商品。模型必须从完整商品目录或候选集中把目标找出来。

### 7.2 `Recall@K`

因为每个用户测试目标只有一个商品，所以：

```text
Recall@K = 目标商品出现在前 K 个推荐中的用户数 / 总用户数
```

例如测试 Recall@10 = 5.66%，表示每 100 个测试用户中，约 5.66 个用户的隐藏目标出现在前 10 个结果中。

### 7.3 `Candidate Recall@200`

它只问“目标有没有进入 200 个候选”，不问最终排序是否把它放前面：

```text
Candidate Recall@200 = 目标进入候选池的用户数 / 总用户数
```

它是召回阶段的上限。候选池里没有的商品，排序模型不可能救回来。

### 7.4 `NDCG@K`

NDCG 同时考虑是否命中以及命中位置。越靠前，折损越小。它比 Recall 更能反映“推荐是否排得靠前”。本项目的 NDCG 数值以 0–1 小数表示，例如 `0.0299` 就是约 `2.99%`。

### 7.5 其他指标

- `MRR@K`：第一个命中的倒数排名，越靠前越高。
- `Coverage@K`：推荐结果覆盖多少不同商品，反映目录探索能力。
- `P50/P95 latency`：请求延迟中位数/95 分位数。
- `QPS`：每秒可处理的请求数。
- `popularity_bias_ratio`：推荐商品平均热度与真实目标平均热度的比值，观察是否过度偏头部。
- Bootstrap 置信区间：按用户重采样，衡量结果不确定性。
- McNemar 检验：对同一批用户比较两个排序器命中/未命中差异。

正式主评估使用完整商品目录检索，而不是“1 个正样本 + 99 个随机负样本”。完整目录更严格，也更接近真实召回问题，但数值通常比 sampled-negative 论文结果低，不能直接横向比较。

---

## 8. 第一条召回通道：Popularity

### 8.1 做法

在 `retrieval_train` 中统计每个商品被交互的次数，按次数降序生成一个全局排行榜。给用户推荐时，跳过用户已经看过的商品。

### 8.2 作用

- 提供最简单、最快的 baseline。
- 给完全未知用户提供冷启动结果。
- 当其他召回器返回数量不足时回填候选。
- 作为复杂模型是否真的有价值的最低比较标准。

### 8.3 结果和局限

测试 `Recall@100=6.41%`，`Coverage@10=0.14%`。它速度快，但几乎总是推荐同一小批头部商品，个性化和目录探索能力差。

面试中应说：Popularity 不是“没用的模型”，而是任何推荐系统都应该保留的稳定兜底和可解释基线。

---

## 9. 第二条召回通道：ItemCF

### 9.1 直观解释

如果大量用户同时看过商品 A 和商品 B，那么 A 与 B 可能相似。对当前用户历史中的每个商品，找它的共现邻居，再按相似度和历史新鲜度累积得分。

### 9.2 工程实现

- 只用 `retrieval_train` 计算共现。
- 每个商品最多保留 `top_neighbors=200` 个邻居。
- 用户最多使用最近 `max_history=50` 个历史商品。
- 使用 `recency_decay=0.9`，越近的历史权重越大。
- 候选不足时由 Popularity 回填。

正式 ItemCF 产物约有 1,243,706 条稀疏邻接边，覆盖约 13,870 个有邻居的商品。

### 9.3 结果

- 测试 Recall@100：`15.70%`，高于 Popularity 的 `6.41%`。
- 测试 Recall@10：`3.57%`。
- 测试 Coverage@10：`97.25%`，明显比 Popularity 更能探索目录。
- P95 约 `19.76 ms`。

### 9.4 解释

在这个数据集上，ItemCF 比简单热门排序有效很多，但它依赖用户历史和商品共现：新用户、新商品、交互稀疏商品的效果会变差。

---

## 10. 第三条召回通道：Two-Tower

### 10.1 为什么用双塔

双塔把用户和商品分别编码到同一个 64 维向量空间：

```text
用户塔：user_id + 历史商品序列 -> 用户向量 u
商品塔：item_id -> 商品向量 v_i
匹配分数：u · v_i
```

商品向量可以离线预计算。在线只需算一次用户向量，再到向量索引中搜索，不必逐个运行复杂模型。

### 10.2 当前模型参数

- embedding dimension：64。
- hidden dimension：128。
- dropout：0.1。
- 用户历史最大长度：20。
- temperature：0.07。
- batch size：512。
- learning rate：0.001。
- weight decay：`1e-5`。
- 最多 30 epochs。
- early stopping patience：7。
- CPU 训练和评估。
- 随机种子：42。

配置文件：`configs/two_tower_v3_logq.yaml` 和 `configs/two_tower_v3_final.yaml`。

### 10.3 关键失败和修复：批内负样本偏差

早期双塔测试 Recall@100 只有约 `3.02%`，低于 Popularity 的 `6.41%`。检查确认 checkpoint、向量生成、已看过滤和全目录评估没有错误。

原因是批内负样本采样偏差：热门商品更容易出现在 batch 中，也更容易被当成负例。模型可能错误地学习“被频繁抽到的商品应该被压低”。未校正时，模型推荐商品的平均热度只有真实目标的约 39%。

修复采用 logQ 校正：训练时依据商品被抽样的概率，对 logits 进行采样概率修正，使模型不把“经常被负采样抽到”误认为“不相关”。这是本项目最有研究价值的诊断之一，因为它展示了从异常结果、排查、文献方法到重新评估的完整过程。

### 10.4 修复后结果

| 切分 | Recall@10 | Recall@100 | NDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| validation | 2.42% | 11.48% | 1.13% | 74.01% |
| test | 1.81% | 9.51% | 0.88% | 74.31% |

测试上双塔高于 Popularity，但低于 ItemCF（9.51% vs 15.70%）。因此正式方案没有让双塔取代 ItemCF，而是把它作为多路候选来源，利用神经模型的互补性。

### 10.5 必须诚实说明的限制

双塔的 `Recall@100=9.51%` 是在完整 13,923 商品目录上的严格评估，不能与只使用少量负样本的论文数字直接比较。它目前不是该数据上的最强单独召回器，但作为多通道系统的一部分是合理的。

---

## 11. FAISS 向量索引

### 11.1 建索引

商品向量归一化后使用内积相似度，建立：

```text
IndexFlatIP
```

商品数约 13,923，向量维度 64，索引约 3.40 MiB。

### 11.2 为什么当前不用 IVF/HNSW

在 1.4 万商品规模下，精确 `IndexFlatIP` 已经很快，而且没有近似误差。此时强行换 IVF/HNSW 不会产生有说服力的收益，反而增加调参和误差解释成本。将近似索引列为未来扩展更科学。

### 11.3 正确性和性能证据

- 100 位真实用户：FAISS Top-100 与 NumPy 全量点积逐行完全一致。
- 保存并重新加载索引后结果仍完全一致。
- 单查询 P50/P95：`0.74/2.15 ms`。
- batch size 128 的每用户摊销 P95：约 `1.31 ms`。

这说明 FAISS 层没有悄悄改变离线模型的排名结果；线上较高延迟来自 Python 侧 ItemCF 和特征构造，不是向量检索本身。

---

## 12. 多路候选融合：RRF

### 12.1 候选来源

每个已知用户获取：

- Two-Tower/FAISS：100 个。
- ItemCF：100 个。
- Popularity：20 个。

然后去重并用 Reciprocal Rank Fusion（RRF）融合，最终候选池目标大小为 200，RRF 常数为 60。

### 12.2 RRF 直观公式

某个商品在来源列表中排名为 `r`，其 RRF 贡献近似为：

```text
1 / (C + r)
```

商品出现在多个来源时，各来源贡献相加；`source_count` 记录它被多少来源推荐。这样不需要强行把不同模型的原始分数校准到同一个尺度。

### 12.3 回填

如果三路去重后少于 200，继续从 Popularity 取未出现商品回填。所有候选必须：

- 不重复；
- 不在用户历史中；
- 属于可服务商品目录；
- 有完整精排特征。

### 12.4 候选召回结果

测试 Candidate Recall@200 为 `21.00%`。这表示约每 100 个测试目标，有 21 个进入 200 候选。剩余目标没有进入候选，后面的 LambdaRank 无法挽救。

---

## 13. LambdaRank 精排

### 13.1 为什么需要精排

召回器关注“不要漏掉可能相关商品”，可以使用不同方法并行扩大覆盖；精排器关注“候选里谁应该更靠前”，可以综合更多信号。

### 13.2 训练协议

- `rank_train`：训练特征和标签。
- `validation`：早停和模型选择。
- `test`：最终一次冻结评估。
- 只评估目标已经进入候选池的排序效果，同时保留完整的 Candidate Recall 作为上限。

主要 LightGBM 参数：

- objective：LambdaRank。
- n_estimators：最多 500。
- learning_rate：0.05。
- num_leaves：31。
- min_child_samples：30。
- row/column sampling：0.9/0.9。
- L2 regularization：1.0。
- early stopping rounds：30。
- 最佳迭代：83。

### 13.3 当前 16 个特征

| 特征 | 含义 |
| --- | --- |
| `two_tower_score` | 双塔向量匹配分数 |
| `two_tower_rank` | 商品在双塔列表中的名次 |
| `itemcf_score` | ItemCF 累积相似度 |
| `itemcf_rank` | 商品在 ItemCF 列表中的名次 |
| `popularity_score` | 商品训练期热度的 `log1p` 变换前/基础分数 |
| `popularity_rank` | 热门排行榜名次 |
| `source_count` | 商品出现在多少个召回来源 |
| `rrf_score` | RRF 融合得分 |
| `best_source_rank` | 商品在各来源中最好的名次 |
| `user_history_length` | 用户历史交互数量 |
| `user_mean_rating` | 用户历史平均评分 |
| `user_rating_std` | 用户历史评分标准差 |
| `days_since_last_action` | 从最后一次历史行为到预测参考时间的天数 |
| `item_popularity_log` | 商品热度的对数变换 |
| `item_mean_rating` | 商品训练期平均评分 |
| `item_rating_std` | 商品训练期评分标准差 |

原始商品 `price`、`avg_rating`、`rating_number` 在正式子集中缺失，不作为精排特征。两个可能携带跨用户未来信息的相对商品时间特征也从最终模型删除。

### 13.4 排序结果

| 指标 | 测试 RRF 顺序 | 测试 LambdaRank |
| --- | ---: | ---: |
| Candidate Recall@200 | 21.00% | 21.00% |
| Recall@10 | 3.60% | 5.66% |
| Recall@20 | 5.55% | 8.80% |
| NDCG@10 | 1.78% | 2.99% |
| MRR@10 | 1.24% | 2.19% |

排序器不能提高 Candidate Recall，因为它不能创造召回阶段没有的商品；它的价值体现在候选内部的顺序优化。

相对提升：

- Recall@10 从 3.60% 到 5.66%，相对约提升 57.2%。
- NDCG@10 从 1.78% 到 2.99%，相对约提升 67.7%。
- NDCG@10 绝对提升约 1.21 个百分点。

### 13.5 特征重要性

验证结果中较重要的特征包括：`itemcf_score`、`itemcf_rank`、`popularity_rank`、`popularity_score`、`rrf_score`、`days_since_last_action`、`item_mean_rating`、`two_tower_score`。这说明当前数据上 ItemCF 和热度信号仍然很强，双塔提供互补但不是唯一主导信号。

---

## 14. 统计和分组分析

在同一批 19,621 个测试用户上做成对 bootstrap：

- Recall@10 绝对提升 `2.06` 个百分点，95% CI `[1.77, 2.34]`。
- NDCG@10 绝对提升 `1.21` 个百分点，95% CI `[1.04, 1.37]`。
- LambdaRank 新增命中 623 人，损失 219 人。
- McNemar 精确检验 `p=1.28e-45`。

按历史长度分组，短、中、长历史用户的 Ranker Recall@10 分别约为 `5.57%`、`5.79%`、`5.64%`，提升并非只由重度用户贡献。

### 14.1 当前最大短板：长尾和冷商品

- 721 个测试目标从未出现在训练商品目录，候选召回必然为 0。
- 尾部商品 Candidate Recall@200 约 `10.35%`，头部约 `36.85%`。
- 头部商品 Ranker Recall@10 约 `12.52%`，尾部约 `1.08%`。

因此当时确定的下一项高价值研究不是继续盲目堆树或调学习率，而是：

1. 使用完整标题、品牌和细分类目等内容特征；
2. 使用 content-based 或 hybrid cold-item retrieval；
3. 对长尾目标做独立采样和分组评估；
4. 设计新商品冷启动实验。

以上四项已经在 Phase 6 中落实。下面记录具体设计和结论，供导师、面试官或其他 AI 继续追问。

### 14.2 Phase 6 元数据门禁

先冻结 Baseline V1：16 个关键数据、配置、模型、索引和结果文件记录 SHA-256。然后审计 Amazon 原始元数据与正式商品目录的连接质量：

| 字段 | 全目录覆盖率 | 721 个严格冷测试目标覆盖率 | 是否使用 |
| --- | ---: | ---: | --- |
| Title | 99.90% | 100.00% | 是 |
| Brand | 98.99% | 99.03% | 是 |
| 细粒度类目 | 99.05% | 97.78% | 是 |
| Price | 不满足可靠性要求 | 不满足可靠性要求 | 否 |
| Avg rating/rating number | 不满足可靠性要求 | 不满足可靠性要求 | 否 |

结论为 GO。这里的严谨点在于：模型方向由数据门禁决定；如果 title/brand/category 仍严重缺失，就应停止内容模型，而不是用空值或常数制造特征。

### 14.3 TF-IDF 内容召回

内容文档由 title、规范化 brand 和细粒度 category path 拼接；最大词表 50,000，1–2 gram，`min_df=2`。词表只在 retrieval-train 可见商品上拟合，避免用验证/测试商品统计反向影响特征空间；检索矩阵覆盖所有元数据可见商品，因此严格冷商品仍能被返回。用户向量由最近最多 50 个历史商品内容向量聚合。

所有通道和融合都严格返回 200 个候选，不通过扩大候选数量获得收益。当前 hybrid 是 score-level RRF，不是联合训练的 Hybrid Item Tower。

| 指标 | 验证 | 测试 |
| --- | ---: | ---: |
| Content Recall@100 | 13.89% | 12.76% |
| Two-Tower Recall@100 | 11.48% | 9.51% |
| Two-Tower + Content Recall@100 | 16.17% | 14.22% |
| 原三通道 Candidate Recall@200 | 23.45% | 20.65% |
| 四通道 Candidate Recall@200 | 27.25% | 24.20% |
| 原三通道 Overall Recall@100 | 17.37% | 14.76% |
| 四通道 Overall Recall@100 | 20.43% | 17.59% |

测试四通道相对原三通道新增命中 1,079 位用户、损失 382 位，净提升 3.55 个百分点。严格冷测试目标中 Content 命中 103/721，Recall@100 为 14.29%，Two-Tower 为 0%；差值 95% CI `[11.79%, 16.92%]`，McNemar `p≈1.97e-31`。尾部目标的 Two-Tower+Content Recall@100 为 8.18%，Two-Tower 仅 0.29%。

### 14.4 全局时间 cutoff 稳健性

配置在看结果前固定 `2016-01-01` 验证 cutoff 和 `2017-01-01` 最终 cutoff。旧 Two-Tower checkpoint 因见过更晚数据而被排除，防止时间泄露。2017 之后最终留出包含 6,102 位用户，其中 1,227 位严格冷用户：

- Content Overall Recall@100：11.39%；ItemCF：8.73%。
- Content 严格冷 Recall@100：15.73%；ItemCF：0%。
- Content+协同 Candidate Recall@200：16.49%；仅协同：11.73%。
- 融合 Tail Recall@100：6.51%；仅协同：4.24%。

方向在统一时间协议下仍成立。必须保留的限制是：静态 Amazon 元数据快照没有字段级上线时间，无法独立证明某段 title/brand/category 在历史时点已经可见。

### 14.5 三随机种子 logQ 消融

使用随机种子 `42/2026/3407`，raw 和 logQ 除采样校正权重外保持所有因素一致。三组 Recall@10、Recall@100、NDCG@10 都同向提升：

| 指标 | Raw 均值 ± 样本标准差 | logQ 均值 ± 样本标准差 | 解释 |
| --- | ---: | ---: | --- |
| Recall@100 | 4.64% ± 0.74% | 11.47% ± 0.46% | 三种子均提升 |
| Recall@10 | 0.64% ± 0.09% | 2.35% ± 0.08% | 三种子均提升 |
| NDCG@10 | 0.30% ± 0.04% | 1.14% ± 0.03% | 三种子均提升 |
| Coverage@10 | 99.10% ± 1.07% | 68.31% ± 7.33% | 明显下降 |
| Popularity bias ratio | 0.40 ± 0.02 | 2.09 ± 0.18 | 明显更偏热门 |

正确结论是“logQ 稳定修复准确率，但牺牲覆盖并增加热门偏置”，不是“logQ 在所有维度全面更优”。

---

## 15. 在线服务架构

```text
HTTP user_id
    |
    v
FastAPI /recommend/{user_id}
    |
    +--> Redis: rec:{model_version}:{user_id}:{k}, TTL 300s
    |
    +--> 已知用户：
    |      history
    |       -> Two-Tower/FAISS(100)
    |       -> ItemCF(100)
    |       -> Popularity(20)
    |       -> RRF + 去重 + 回填(200)
    |       -> 16 features
    |       -> LambdaRank
    |
    +--> 未知用户：Popularity fallback
    |
    +--> request_id/model_version/fallback
            |
            +--> SQLite feedback events
            +--> Prometheus metrics -> Grafana
```

### 15.1 启动时加载

`ServingPipeline` 在应用 lifespan 启动阶段一次加载：

- 交互表和商品表；
- ItemCF 邻居；
- Popularity 排行；
- FAISS 索引和 item IDs；
- PyTorch 双塔 checkpoint；
- LightGBM ranker；
- 当前版本 pointer。

避免每个请求重复加载模型，是在线系统的基本工程要求。

### 15.2 已知用户和未知用户

- 已知用户：使用完整两阶段 pipeline，返回 `fallback: false`。
- 未知用户：没有历史，返回确定性的 Popularity 结果，返回 `fallback: true`。

两者都必须去重，且不推荐用户历史中的商品。`k` 只允许 1–50。

### 15.3 Redis 策略

缓存 key：

```text
rec:{model_version}:{user_id}:{k}
```

TTL：300 秒。版本进入 key，避免模型升级后旧结果继续污染新模型。Redis 连接失败时短超时旁路，不让缓存故障拖垮推荐接口。

### 15.4 SQLite 反馈

`POST /events` 记录：

- `event_id`
- `user_id`
- `item_id`
- `event_type`: impression/click/purchase
- `timestamp`
- `request_id`
- `model_version`
- `rank_position`
- `simulated`

当前 demo 事件必须保持 `simulated=true`。导出的模拟反馈不能混入冻结离线 benchmark，更不能被描述为真实线上证据。

---

## 16. API 接口说明

### 16.1 健康检查

```http
GET /health
```

示例响应：

```json
{
  "status": "ok",
  "version": "recsys_baseline_v1",
  "model_loaded": true,
  "catalog_size": 13923,
  "redis_available": true,
  "feedback_store": true
}
```

字段含义：

- `status`：服务是否能响应。
- `version`：当前服务模型版本。
- `model_loaded`：双塔、FAISS 和 ranker 是否加载。
- `catalog_size`：FAISS/服务商品目录大小。
- `redis_available`：Redis ping 是否成功。
- `feedback_store`：SQLite 反馈存储是否初始化。

### 16.2 推荐

```http
GET /recommend/{user_id}?k=10
```

响应核心字段：

```json
{
  "user_id": "A0266076X6KPZ6CCHGVS",
  "k": 10,
  "recommendations": [
    {
      "item_idx": 123,
      "item_id": "B000...",
      "title": "商品标题",
      "score": 0.42,
      "source_count": 2
    }
  ],
  "request_id": "uuid",
  "model_version": "recsys_baseline_v1",
  "fallback": false
}
```

### 16.3 反馈

```http
POST /events
Content-Type: application/json
```

请求示例：

```json
{
  "user_id": "A0266076X6KPZ6CCHGVS",
  "item_id": "test-item",
  "event_type": "click",
  "timestamp": "2026-08-28T12:00:00Z",
  "request_id": "demo-001",
  "model_version": "recsys_baseline_v1",
  "rank_position": 1,
  "simulated": true
}
```

### 16.4 监控

```http
GET /metrics
GET /metrics?format=prometheus
```

覆盖：请求数、错误数、缓存命中/未命中、fallback 数、推荐数量、延迟 P50/P95、模型版本。

---

## 17. 本地运行和复现实验

### 17.1 安装依赖

在项目根目录运行：

```powershell
python -m pip install -e ".[dev]"
python -c "import recsys; print(recsys.__version__)"
python -m ruff check .
python -m pytest -q
```

### 17.2 从原始数据重新生成

```powershell
python -m recsys.data.download --config configs/data.yaml
python -m recsys.data.clean --config configs/data.yaml
python -m recsys.data.split --config configs/data.yaml
```

### 17.3 运行基线

```powershell
python -m recsys.retrieval.popularity --config configs/popularity.yaml
python -m recsys.retrieval.itemcf --config configs/itemcf.yaml
python -m recsys.evaluation.compare_baselines --results-dir results/video_games_2018/baselines
```

### 17.4 训练和评估双塔

```powershell
python -m recsys.retrieval.train --config configs/two_tower_v3_logq.yaml
python -m recsys.retrieval.train --config configs/two_tower_v3_final.yaml
```

第一条配置用于验证集选择，第二条加载冻结 checkpoint 做最终测试。不要为了调参反复读取测试集。

### 17.5 建 FAISS 和训练精排

```powershell
python -m recsys.retrieval.faiss_index --config configs/faiss.yaml
python -m recsys.ranking.pipeline --config configs/ranker.yaml
python -m recsys.ranking.evaluate --config configs/ranker.yaml
```

### 17.6 在线服务本地启动

```powershell
$env:PYTHONPATH="src"
$env:REDIS_URL="redis://127.0.0.1:6379/0"
python scripts/build_serving_artifacts.py
python -m uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000
```

如果没有 Redis，服务也能通过旁路逻辑运行；生产式 Compose 则默认启动 Redis。

---

## 18. Docker Compose 和 AWS Lightsail 部署

### 18.1 服务编排

`docker-compose.yml` 定义四个服务：

- `redis:7-alpine`：缓存。
- `api`：Python 3.12 slim + FastAPI + 模型。
- `prom/prometheus:v2.54.1`：抓取 API 指标。
- `grafana/grafana:11.2.0`：展示 dashboard。

API 对外暴露 `8000`。Prometheus `9090` 和 Grafana `3000` 在最新本地配置中只绑定 `127.0.0.1`，不应直接开放公网。

API 镜像额外安装 Debian/Ubuntu 系统库 `libgomp1`，因为 LightGBM 依赖 `libgomp.so.1`。API 的处理后数据以只读方式挂载到 `/app/data/processed`；反馈数据库使用独立可写 Docker volume。不能把整个 `/app/data` 设为只读后再在子目录挂载可写 volume，否则会出现 Docker mountpoint read-only 错误。

### 18.2 Lightsail 实例

当前实例信息（用于本次展示环境）：

- 名称：`recsys-demo`
- 区域：Seoul，`ap-northeast-2a`
- 规格：4 GB RAM、2 vCPU、80 GB SSD
- 网络：Dual-stack
- 静态 IPv4：通过环境变量或部署平台配置，不写入公开仓库
- 系统：Ubuntu

Lightsail 4GB 套餐标价约 24 美元/月，免费账户的 Credits 可抵扣，但它不是永久零价套餐。停止状态仍可能继续计费，展示结束要删除实例及无用快照/未绑定 IP。

### 18.3 防火墙边界

最终应只保留：

```text
TCP 8000  Anywhere IPv4   # 推荐 API
TCP 22    按需开放        # SSH 管理，最好限制来源 IP
```

Prometheus 9090、Grafana 3000 不直接开放公网。为绕过本地网络限制，部署过程曾临时使用 TCP 443 作为 SSH 入口；部署完成后必须删除 443 规则并关闭服务器对应监听。

### 18.4 云端验收命令

在 Lightsail 浏览器 SSH 终端：

```bash
cd /opt/recsys
sudo docker compose ps
curl -i http://127.0.0.1:8000/health
curl -sS "http://127.0.0.1:8000/recommend/A0266076X6KPZ6CCHGVS?k=10"
curl -sS "http://127.0.0.1:8000/recommend/test_unknown_user?k=5"
curl -sS "http://127.0.0.1:8000/metrics?format=prometheus"
```

浏览器地址：

- `http://<VM_PUBLIC_IP>:8000/docs`
- `http://<VM_PUBLIC_IP>:8000/health`
- `http://<VM_PUBLIC_IP>:8000/recommend/A0266076X6KPZ6CCHGVS?k=10`

云端终端曾返回：API/Redis healthy、`model_loaded=true`、`catalog_size=13923`、`redis_available=true`、`feedback_store=true`，并返回 `HTTP/1.1 200 OK`。

---

## 19. 模型版本和可复现性

### 19.1 当前版本

```text
recsys_baseline_v1
```

pointer：`artifacts/versions/current.json`。manifest：`artifacts/versions/recsys_baseline_v1/manifest.json`。

manifest 记录：

- 创建时间；
- 数据 fingerprint；
- Git commit；
- Python/依赖版本；
- 双塔、FAISS、ranker 文件路径和 SHA256；
- Candidate Recall、Two-Tower Recall、LambdaRank NDCG；
- promotion gate 是否通过。

### 19.2 重训练链

```powershell
python scripts/retrain.py --config configs/retrain.yaml --dry-run
python scripts/retrain.py --config configs/retrain.yaml
python scripts/promote_model.py --version <version>
```

promotion gate 检查最低指标和 20 个固定用户的 serving parity。只有通过后才更新 `current.json`。

### 19.3 为什么需要 parity

离线评估可能直接调用 Python 函数，线上则经过配置解析、版本 pointer、Docker volume 和 API。20/20 parity 证明固定用户的最终 Top-10 在离线保存路径和线上 ServingPipeline 中完全一致，降低“离线好看、线上变了”的风险。

---

## 20. 测试和验收清单

### 20.1 单元和集成测试

```powershell
python -m pytest -q
python -m ruff check .
```

当前结果：30 项测试通过，Ruff 通过。测试覆盖：

- 配置解析；
- 数据清洗和切分；
- Popularity 和 ItemCF；
- 候选过滤；
- 双塔 smoke；
- FAISS 建索引和检索；
- 评价指标；
- 精排特征；
- API 健康、推荐和反馈行为。

### 20.2 Serving parity 和性能

```powershell
python scripts/check_serving_parity.py --users 20
python scripts/profile_serving_pipeline.py --users 20 --repeats 5 --warmup 10
python scripts/benchmark_api.py --requests 100
python scripts/load_test_api.py
bash scripts/docker_smoke.sh
```

2026-09-01 同机证据：20/20 parity；候选池每次保持 200；顺序请求微基准约 P50 26.59ms、P95 30.48ms、37.24 请求/秒，但它不代表容量。闭环并发测试在单 Uvicorn 进程、无 Redis 的条件下得到约 11–15 QPS；并发 1/5/10/20/50 的 P95 分别为 76.58/518.70/1013.48/2142.06/8303.33ms，错误均为 0。分阶段 profiling 显示特征构造占平均总耗时 57.64%，LambdaRank 预测占 16.91%，FAISS 仅占 2.63%。结论是功能路径稳定，但单进程 CPU 扩展性差；这些仍是本地离线 demo 数字，不是生产 SLA。

### 20.3 线上人工验收

必须至少测试四类请求：

1. 已知用户：`fallback=false`，返回去重的个性化结果。
2. 未知用户：`fallback=true`，返回热门商品。
3. 非法 `k`：1–50 之外应返回参数错误。
4. Redis 停止：推荐仍应可用，只是没有缓存。

---

## 21. 常见问题和解决过程

### 21.1 API 容器启动失败：只读挂载冲突

症状：Docker 报错无法在 `/app/data/feedback` 创建 mountpoint，提示 `read-only file system`。

原因：父目录 `/app/data` 被整体以只读挂载，子目录又要挂载可写 feedback volume。

修复：只读挂载 `./data/processed:/app/data/processed:ro`，把 `/app/data/feedback` 留给独立 volume。

### 21.2 API 容器启动失败：`libgomp.so.1` 不存在

症状：LightGBM 导入时报：

```text
OSError: libgomp.so.1: cannot open shared object file
```

原因：`python:3.12-slim` 没有 GNU OpenMP 运行库。

修复：Dockerfile 安装 `libgomp1`，重新构建 API image。

### 21.3 浏览器终端粘贴出现 `^[[200~`

原因：直接 Ctrl+V 粘贴多行 shell 命令时，浏览器终端的 bracketed-paste 控制字符被当作普通输入，命令被拆散。

解决：先 `Ctrl+C`，使用 Lightsail 右下角 `Paste into terminal`，每次粘贴一整行；不要把带反斜杠换行的命令拆开输入。

### 21.4 SSH 22 端口超时

原因可能是本机网络限制，而不是实例不运行。浏览器内置 SSH 能打开说明实例和密钥正常。

部署时可临时让 sshd 监听 443，完成后必须：

1. 删除 Lightsail TCP 443 规则；
2. 删除 `/etc/ssh/sshd_config.d/99-recsys-upload.conf`；
3. 终止 443 sshd 监听；
4. 保留必要的 22 管理规则。

---

## 22. 研究结论：应该如何向别人解释

### 22.1 30 秒版本

> 我做了一个可复现的两阶段电商推荐系统。数据来自 Amazon Reviews Video Games 5-core；Baseline V1 包含 ItemCF、Popularity、logQ Two-Tower、FAISS 和 LambdaRank。完整目录测试中 LambdaRank NDCG@10 比同候选 RRF 提升约 67.7%。随后我针对 721 个严格冷商品目标加入 title/brand/category 内容召回，在候选数仍为 200 的情况下把测试 Candidate Recall@200 从 20.65% 提升到 24.20%，冷启动 Recall@100 从协同模型的 0% 提升到 14.29%，并用全局 cutoff 和三随机种子验证稳健性。服务侧完成 20/20 parity 和并发压测；由于没有真实线上流量，我只把它称为研究原型，不声称商业转化提升。

### 22.2 2 分钟版本结构

1. **问题**：全量商品排序太慢，所以拆召回和精排。
2. **数据**：公开 Amazon Reviews，评分大于等于 3 作为正向兴趣，用户级时间切分。
3. **基线**：Popularity 速度快但头部偏置；ItemCF 提升个性化和覆盖。
4. **研究诊断**：双塔初始结果低于 Popularity，定位到批内负采样偏差，加入 logQ 修复。
5. **系统设计**：三路召回 -> RRF 200 -> LambdaRank 16 特征 -> Top-k。
6. **证据**：完整目录 Recall、NDCG、bootstrap、FAISS exact match、parity、API smoke。
7. **研究深化**：元数据门禁通过后加入内容召回，严格冷 Recall@100 达到 14.29%，并通过全局 cutoff 验证方向。
8. **诚实限制**：logQ 准确率提升伴随覆盖率下降；单进程并发扩展差；没有线上 CTR/A/B。

### 22.3 推荐简历表述

英文：

> Built a leakage-aware two-stage recommender on 197K temporally split interactions; diagnosed in-batch sampling bias with a three-seed logQ ablation, verified exact FAISS retrieval, and added metadata-based content retrieval that improved fixed-budget Candidate Recall@200 from 20.65% to 24.20% while reaching 14.29% Recall@100 on 721 strict-cold users.

中文：

> 基于 19.8 万条 Amazon Reviews 交互构建防泄露两阶段推荐系统；用三随机种子验证 logQ 对批内采样偏差的修复，并加入元数据内容召回，在固定 200 候选下将 Candidate Recall@200 从 20.65% 提升到 24.20%，对 721 位严格冷用户实现 14.29% Recall@100。

不要写：

- “上线后 CTR 提升 xx%”；
- “转化率提升 xx%”；
- “达到生产环境 SLA”；
- “双塔超过 ItemCF”；
- “免费运行半年”作为永久云资源承诺。

---

## 23. 当前局限、风险和下一阶段

### 23.1 数据局限

- 评论不等于曝光、点击或购买；评分存在选择偏差。
- 主实验是用户级时间切分；Phase 6 已增加全局日期切分，但静态元数据历史可用时间不可独立验证。
- Title、brand、细分类目已进入 Phase 6 内容召回，并已接入独立的 `recsys_phase6_content_v1` Serving 快照；它仍不是联合训练的 Hybrid Item Tower。
- 价格和元数据评分字段缺失，不能分析价格敏感度。

### 23.2 模型局限

- 双塔仍弱于 ItemCF 单独召回，且 logQ 提升准确率的同时降低覆盖、增加热门偏置。
- 内容召回显著改善严格冷和长尾，但绝对 Recall 仍有限。
- Phase 6 内容通道是 TF-IDF 与 score-level RRF，不是训练过的 Hybrid Item Tower。
- 单进程吞吐约 11–15 QPS，并发升高时尾延迟快速恶化。

### 23.3 生产化局限

- 没有真实用户身份、权限、HTTPS、域名和 API 鉴权。
- SQLite 只适合 demo/单机，不适合高并发生产反馈写入。
- Grafana 当前匿名 Viewer 仅适合展示，不适合真实业务环境。
- 没有 A/B 测试、灰度发布、自动回滚和在线特征平台。
- AWS Lightsail 是短期展示环境，不是高可用架构。

### 23.4 高价值下一步

1. 在验证集上学习四通道融合权重，并重新训练使用内容特征的 ranker；测试集继续保持冻结。
2. 增加多个全局时间窗口或 rolling evaluation，并获取可追溯的元数据时间戳。
3. 把 ItemCF/特征构造预计算或批量化，验证多 worker 和水平扩展。
4. 用真实曝光日志记录 impression/click/purchase，建立在线指标和 A/B 协议。
5. 增加 novelty、diversity、calibration 和长尾公平性约束，处理 logQ 的准确率/覆盖权衡。
6. 将 SQLite 替换为生产数据库，加入鉴权、TLS、限流和密钥管理。

---

## 24. 关键产物索引

### 研究报告

- `reports/final_research_report_zh.md`：最终研究报告。
- `reports/video_games_baseline_evaluation.md`：正式召回基线报告。
- `reports/baseline_evaluation.md`：早期小数据基线记录。

### 机器可读指标

- `results/video_games_2018/two_tower_v3_final_metrics.json`
- `results/video_games_2018/faiss_benchmark.json`
- `results/video_games_2018/ranking_validation_metrics.json`
- `results/video_games_2018/final_test_metrics.json`
- `results/video_games_2018/final_group_analysis.csv`
- `results/video_games_2018/baselines/retrieval_comparison.csv`

### 模型和索引

- `models/video_games_2018/two_tower_v3_logq.pt`
- `models/video_games_2018/ranker.txt`
- `artifacts/video_games_2018/faiss/item_flat.index`
- `artifacts/video_games_2018/faiss/faiss_item_ids.npy`
- `artifacts/versions/recsys_baseline_v1/manifest.json`
- `artifacts/versions/current.json`

### 服务和部署

- `README_SERVING.md`
- `Dockerfile`
- `docker-compose.yml`
- `deploy/bootstrap.sh`
- `deploy/CLOUD_VM_GUIDE.md`
- `deploy/cloudshell_deploy.sh`
- `scripts/docker_smoke.sh`
- `scripts/check_serving_parity.py`
- `scripts/benchmark_api.py`
- `scripts/export_feedback.py`

---

## 25. 交给其他 AI 时的阅读顺序

如果另一个 AI 要理解本项目，建议让它按以下顺序阅读：

1. 本文件 `PROJECT_MASTER_GUIDE_ZH.md`，掌握全貌和口径。
2. `README.md`，确认快速运行入口。
3. `README_SERVING.md`，理解在线服务接口。
4. `reports/final_research_report_zh.md`，核对实验结论。
5. `configs/data.yaml`、`configs/serving.yaml`、`configs/ranker.yaml`，核对参数。
6. `src/recsys/data/clean.py` 和 `split.py`，核对数据处理。
7. `src/recsys/retrieval/`，核对召回实现。
8. `src/recsys/ranking/pipeline.py`，核对候选融合和精排。
9. `src/recsys/serving/pipeline.py` 和 `api.py`，核对线上路径。
10. `tests/`、`results/`、`artifacts/versions/recsys_baseline_v1/manifest.json`，核对证据链。

### 交给 AI 的建议提示词

```text
请先阅读 PROJECT_MASTER_GUIDE_ZH.md，再阅读 README.md、README_SERVING.md、
reports/final_research_report_zh.md、configs/ 和 src/recsys/。回答时必须区分：
1）离线实验结论；2）云端 demo 已验证的工程事实；3）尚未拥有证据的商业假设。
不要把模拟反馈、离线 Recall 或 NDCG 写成真实线上 CTR/转化率。
如果发现文档数字和代码结果冲突，请优先检查 results/ JSON、manifest 和测试，
指出冲突而不是猜测。
```

---

## 26. 最终判断

这个项目的含金量来自完整而诚实的工程和研究闭环，而不是单个模型名字：

```text
公开数据
  -> 可审计清洗
  -> 用户级时间切分
  -> 基线比较
  -> 失败诊断和 logQ 修复
  -> 双塔/ItemCF/Popularity 多路召回
  -> FAISS 正确性验证
  -> RRF 候选融合
  -> LambdaRank 精排
  -> bootstrap 和分组分析
  -> 版本化模型资产
  -> FastAPI/Redis/Prometheus/Grafana
  -> Docker/AWS 可运行 demo
```

最准确的描述是：**一个有真实公开数据、严格离线评估、可解释实验结论、可重复版本管理、并成功封装为在线服务的推荐系统研究原型。**

它已经足够用于研究生申请和实习展示；面试时重点讲清楚数据切分、双塔失败原因、候选召回上限、长尾短板和证据边界，会比单纯强调“用了深度学习和 FAISS”更有说服力。
