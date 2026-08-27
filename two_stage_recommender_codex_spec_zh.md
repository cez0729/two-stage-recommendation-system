# 两阶段电商推荐系统

> 面向 Codex 的完整工程规格、实施计划与验收标准（旗舰个人项目）

文档版本：1.0

目标完成窗口：约 20 个学习日 × 每天 2 小时（约 40 小时）

默认技术路线：Python + PyTorch + FAISS + LightGBM + DuckDB + FastAPI + Streamlit

定位：研究生申请作品集 + 推荐算法/机器学习实习求职旗舰项目

> **核心目标：** 构建一个可复现的两阶段推荐系统：先使用热门度、ItemCF 和双塔模型进行候选召回，再使用 LightGBM LambdaRank 精排，最终通过 FastAPI 提供推荐服务，并用严格的时间切分、Recall@K、NDCG@K、覆盖率和延迟指标完成评估。

本文件既是项目设计文档，也是 Codex 的任务合同。Codex 应严格按章节中的范围、文件路径、接口和验收标准工作；任何超出范围的设计都应先作为“可选扩展”提出，不应直接加入主分支。


---

## 文档目录

- 1. 如何把本文件交给 Codex 使用

- 2. 项目摘要与最终交付物

- 3. 范围、非目标与完成定义

- 4. 技术路线和总体架构

- 5. 数据源、子集选择与数据合同

- 6. 严格的时间切分与防泄露规则

- 7. 离线评估指标与统一评估器

- 8. 召回基线：Popularity 与 ItemCF

- 9. 双塔召回模型

- 10. FAISS 候选检索

- 11. 精排训练数据生成

- 12. LightGBM LambdaRank 精排

- 13. 冷启动、长尾与分组评估

- 14. 在线推理服务与 API 合同

- 15. Streamlit 演示界面

- 16. 实验追踪、复现与结果产物

- 17. 测试、代码质量与失败处理

- 18. 仓库结构和命令规范

- 19. 20 天开发计划

- 20. Codex 工作包与可直接复制的提示词

- 21. 最终验收清单

- 22. 简历证据链与项目展示

- 附录 A-F：配置、数据表、接口模型、伪代码、报告结构、参考资料


---

## 1. 如何把本文件交给 Codex 使用

### 1.1 人与 Codex 的职责边界

| 参与者 | 主要职责 | 不得做的事 |

| --- | --- | --- |

| 你（项目负责人） | 选择数据类别；确认范围；运行命令；检查结果；解释实验；决定是否合并代码。 | 不能只接受“代码能运行”而不检查数据泄露、指标定义和实验合理性。 |

| Codex（开发助手） | 创建目录和模块；实现函数；编写测试；修复错误；补全文档；根据验收标准自检。 | 不得虚构实验结果；不得修改数据定义；不得无理由更换技术栈；不得把核心逻辑只放在 Notebook。 |

| 训练环境 | 执行数据处理、训练、索引构建和服务启动。 | 不得依赖未记录的手工步骤或仅存在于某次交互式会话中的状态。 |



### 1.2 推荐的协作流程

- 每次只让 Codex 完成一个工作包；不要一次要求“完成整个推荐系统”。

- 在开始任务前，让 Codex先阅读本规格中对应章节、当前仓库树、已有测试和最近错误日志。

- 要求 Codex先给出“拟修改文件清单”和“实现计划”，再修改代码。

- 每个工作包完成后，必须运行规定的单元测试和最小烟雾测试，并返回实际命令和输出摘要。

- 你检查代码、结果文件和 README 后再进入下一工作包。

- 任何指标都必须由脚本生成并保存到 results/，不得手工输入到 README。

### 1.3 Codex 全局行为约束

```text
You are implementing a scoped machine-learning portfolio project.
Follow these rules for every task:
1. Read SPEC.md and the current repository before editing.
2. Do not expand scope unless the change is explicitly marked optional.
3. Put reusable logic under src/, not only in notebooks.
4. Use type hints, docstrings, deterministic seeds, pathlib, and structured logging.
5. All commands must be rerunnable and idempotent where practical.
6. Never fabricate data, metrics, benchmark results, or successful test output.
7. After changes, run the requested tests and report exact commands and failures.
8. Do not commit raw dataset files, model binaries, or secrets.
9. Preserve time-ordering and prevent target leakage.
10. Prefer a simple correct implementation over a complex unverified one.
```

> **建议：** 将本文件的 Markdown 版本复制为仓库根目录的 SPEC.md。Codex 更容易检索 Markdown；DOCX 版本适合阅读、打印和提交给导师。

## 2. 项目摘要与最终交付物

### 2.1 一句话项目定义

基于 Amazon Reviews 2023 的用户—商品隐式反馈，构建一个两阶段个性化推荐系统：使用 Popularity、ItemCF 和 PyTorch 双塔模型生成候选商品，使用 FAISS 执行向量检索，再通过 LightGBM LambdaRank 对候选集进行精排，并用 FastAPI 和 Streamlit 提供可演示的推荐服务。

### 2.2 最终使用场景

- 已知用户推荐：输入历史用户 ID，返回 Top-K 个未交互商品及分数。

- 新用户推荐：输入若干最近喜欢的商品或类别，生成临时用户表示并返回推荐。

- 相似商品：输入商品 ID，返回向量空间内的相似商品。

- 离线评估：对验证集和测试集计算召回、排序、多样性、覆盖率和延迟。

- 作品集展示：README 中包含架构图、实验结果、错误分析、运行命令和演示视频。

### 2.3 最终交付物

| 交付物 | 最低要求 | 验收证据 |

| --- | --- | --- |

| 代码仓库 | 模块化 src/；测试；配置；命令行入口；无原始数据。 | pytest 通过；README 可按命令复现最小流程。 |

| 处理后数据 | Parquet/DuckDB 产物；数据统计；时间切分清单。 | artifacts/data_manifest.json 与 split_stats.json。 |

| 召回模型 | Popularity、ItemCF、Two-Tower 三类。 | results/retrieval_metrics.json 和对比表。 |

| 向量索引 | FAISS 索引、商品映射和延迟基准。 | artifacts/faiss.index；results/faiss_benchmark.json。 |

| 精排模型 | LightGBM LambdaRank；至少 10 个特征。 | models/ranker.txt；results/ranking_metrics.json。 |

| API | health、recommend、similar-items、metrics。 | OpenAPI 文档；API 单元测试；演示截图。 |

| 演示界面 | 用户推荐、相似商品、项目指标页面。 | Streamlit 可本地运行；3 分钟录屏。 |

| 技术报告 | 6-10 页英文报告。 | reports/technical_report.pdf 或 .md。 |

| 简历证据 | 3 条真实量化 bullet。 | README 结果与简历数字完全一致。 |



## 3. 范围、非目标与完成定义

### 3.1 必须完成（MVP）

- 使用一个 Amazon Reviews 2023 商品类别；处理后交互规模建议 50 万至 150 万。

- 每个用户至少 5 次有效交互；每个商品至少 5 次有效交互。

- 严格按用户时间顺序生成 retrieval_train、rank_train、validation、test。

- 实现 Popularity、ItemCF、Two-Tower 三种召回方法。

- Two-Tower 使用 in-batch negatives；可选增加热门负样本。

- FAISS 对所有候选商品向量建立索引并返回 Top-N。

- 从召回候选中构建 LightGBM LambdaRank 的精排数据。

- 计算 Recall@20/50/100、NDCG@10、MRR@10、Catalog Coverage 和 P95 延迟。

- 完成至少 3 组消融或对比实验。

- 提供 FastAPI、Streamlit、Dockerfile、README、技术报告和演示视频。

### 3.2 可选扩展

- 标题文本 TF-IDF + TruncatedSVD 向量，或小型预训练文本向量。

- 多路召回融合：Popularity + ItemCF + Two-Tower 的归一化分数融合。

- 近似索引 HNSW/IVF 与精确 IndexFlatIP 的效果—延迟对比。

- 额外排序模型：Logistic Regression 或 LightGBM 二分类器。

- MLflow 实验界面和模型注册。

- GitHub Actions 自动运行 lint 和 pytest。

### 3.3 暂不做

- Kafka、Flink、Spark Streaming 等实时流系统。

- Kubernetes、多节点训练、云端大规模部署。

- 复杂序列模型（SASRec、BERT4Rec）作为主模型。

- 完整 Electronics 全量数据训练。

- 在线 A/B 测试或声称真实业务收益。

- 把大语言模型聊天界面作为项目核心。

> **范围控制：** 如果进度落后，按“删除复杂前端 → 删除可选文本特征 → 缩小数据规模 → 减少额外实验”的顺序删减。不得删除时间切分、基线、统一评估、README 和真实结果。

### 3.4 完成定义（Definition of Done）

只有同时满足下列条件，项目才可以在简历上标为 Completed：

- 从空环境按 README 命令能够完成最小数据处理、训练、评估和 API 启动。

- 测试集结果由脚本自动生成，且结果文件包含时间、配置哈希和 Git commit。

- 结果表至少包括三种召回模型和“召回后 vs 精排后”的比较。

- README 明确数据切分、负样本、排除已见商品、候选规模和指标定义。

- API 不返回用户历史中已经交互的商品，除非请求明确允许。

- 项目限制写清楚：公开数据、离线评估、非真实线上部署。

- 简历中的所有数字均能在 results/ 或报告中找到。

## 4. 技术路线和总体架构

### 4.1 默认技术栈

| 层 | 工具 | 用途 | 选择理由 |

| --- | --- | --- | --- |

| 语言与环境 | Python 3.11；uv 或 pip | 依赖、CLI、训练和服务。 | 生态成熟；与实习岗位匹配。 |

| 数据存储 | Parquet + DuckDB | 列式存储、SQL 清洗、统计。 | 轻量；无需启动数据库服务器。 |

| 模型 | PyTorch | 双塔召回模型。 | 便于展示深度学习与自定义训练。 |

| 近邻检索 | FAISS | 商品向量索引和 Top-N 检索。 | 适合稠密向量相似搜索。 |

| 排序 | LightGBM LGBMRanker | LambdaRank 精排。 | 训练快；适合结构化排序特征。 |

| API | FastAPI + Pydantic | 推理接口与自动文档。 | 易测试、类型清晰、可容器化。 |

| Demo | Streamlit | 交互式展示。 | 开发快；适合作品集演示。 |

| 质量 | pytest + ruff | 测试和代码规范。 | 控制工程质量而不过度复杂。 |

| 实验 | JSON/CSV；可选 MLflow | 记录配置、参数和指标。 | 保证可复现。 |



### 4.2 系统架构

```text
Amazon Reviews 2023 (reviews + metadata)
                   |
                   v
        DuckDB / Parquet preprocessing
                   |
        +----------+-----------+
        |                      |
        v                      v
  retrieval_train         item/user features
        |                      |
        +----------+-----------+
                   v
      Popularity / ItemCF / Two-Tower
                   |
                   v
          item embeddings + FAISS index
                   |
       retrieve Top 200 candidate items
                   |
                   v
       ranking feature generation
                   |
                   v
        LightGBM LambdaRank model
                   |
                   v
              Top-K results
                   |
        +----------+-----------+
        v                      v
     FastAPI               Streamlit Demo
```

### 4.3 离线与在线一致性原则

- 用户历史特征必须接受 cutoff_timestamp 参数；验证/测试时只能读取截止时间之前的行为。

- 训练、评估和 API 共用同一个 FeatureBuilder，不允许各写一套计算逻辑。

- 商品 ID、用户 ID 映射保存在 artifacts/mappings/，训练和服务读取相同文件。

- 候选过滤逻辑（排除已见商品、无效商品）由单一 CandidateFilter 实现。

- 模型输入特征顺序保存在 feature_schema.json，服务启动时校验。

## 5. 数据源、子集选择与数据合同

### 5.1 默认数据类别

默认选择 Amazon Reviews 2023 的 All_Beauty 类别。原因是规模通常足够训练和展示推荐系统，同时比 Electronics 更适合在个人电脑上完成。若过滤后不足 50 万条交互，可改用 Video_Games 或选择更大类别后抽样。

| 选项 | 适用情况 | 建议 |

| --- | --- | --- |

| All_Beauty | 普通笔记本；希望快速完成。 | 默认。 |

| Video_Games | 内存与磁盘较充足；希望商品语义更清晰。 | 备选。 |

| Electronics 子集 | 高配置机器；愿意花更多时间做抽样。 | 仅在流程稳定后使用。 |



### 5.2 原始字段的最小需求

| 实体 | 字段 | 处理 |

| --- | --- | --- |

| Review | user_id、parent_asin/item_id、timestamp、rating | user_id/item_id 转字符串；timestamp 转 UTC；rating 保留。 |

| Review 可选 | verified_purchase、helpful_vote、text | 作为排序特征或分析，不作为第一版必需输入。 |

| Metadata | parent_asin/item_id、title、main_category、categories | 清洗空值；主类别规范化。 |

| Metadata 可选 | price、average_rating、rating_number | 转换数值；缺失时增加 missing flag。 |



### 5.3 隐式反馈定义

第一版把每条有效评论视为一次正向交互，不直接把 rating 当作监督标签。可选规则：保留 rating >= 3 的交互；如果保留所有评分，必须在报告中说明“评论行为本身表示用户关注，评分仅作为附加特征”。

> **推荐选择：** 为减少歧义，MVP 使用 rating >= 3 的评论作为正反馈，并在数据统计中报告过滤前后规模。不要把 1-2 星商品当作正样本。

### 5.4 数据过滤规则

```python
# 推荐默认值
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
MIN_RATING = 3.0
MAX_INTERACTIONS = 1_500_000
RANDOM_SEED = 42

# 过滤顺序
1. 删除缺失 user_id、item_id、timestamp 的记录
2. 去除完全重复的 user-item-timestamp 记录
3. 保留 rating >= MIN_RATING
4. 迭代执行 k-core 过滤，直到用户和商品均满足最小交互数
5. 若仍超过 MAX_INTERACTIONS，按时间分层或按用户抽样，不随机打乱时间
6. 对 user_id 与 item_id 建立连续整数映射
```

### 5.5 处理后数据表

| 文件/表 | 主键或粒度 | 字段 |

| --- | --- | --- |

| interactions.parquet | 一行一次用户—商品交互 | user_idx, item_idx, user_id, item_id, timestamp, rating, split |

| items.parquet | 一行一个商品 | item_idx, item_id, title, category, price, avg_rating, rating_count |

| users.parquet | 一行一个用户 | user_idx, total_interactions, first_ts, last_ts |

| item_stats.parquet | 商品在 retrieval_train 上的统计 | popularity, mean_rating, recency, category_popularity |

| user_stats.parquet | 用户在 cutoff 之前的统计模板 | history_length, mean_rating, top_category, category_entropy |

| data_manifest.json | 一次数据构建 | 源类别、过滤参数、记录数、时间范围、文件哈希 |



### 5.6 数据产物目录

```text
data/
├── raw/                 # gitignored；只放下载后的原始文件
├── interim/             # gitignored；清洗中间文件
└── processed/           # gitignored；最终 Parquet/DuckDB

artifacts/
├── data_manifest.json
├── split_stats.json
├── mappings/
│   ├── user_mapping.parquet
│   └── item_mapping.parquet
└── schemas/
    ├── interaction_schema.json
    └── feature_schema.json
```

## 6. 严格的时间切分与防泄露规则

### 6.1 推荐的四段式按用户切分

对每个用户按 timestamp 升序排序。使用最后三次交互分别构造 rank_train、validation 和 test，其余交互用于 retrieval_train。要求用户至少 5 次交互，确保 retrieval_train 至少有两次历史行为。

```text
For each user u with interactions [i1, i2, ..., in] in time order:
retrieval_train = [i1, ..., i(n-3)]
rank_train      = i(n-2)
validation      = i(n-1)
test            = in

User history for each query:
- rank_train query history: retrieval_train
- validation query history: retrieval_train + rank_train
- test query history: retrieval_train + rank_train + validation
```

### 6.2 为什么需要 rank_train

精排模型需要独立的训练查询。若直接用 validation 或 test 生成精排训练标签，会造成测试信息泄露。rank_train 为每个用户提供一个正目标，候选由只在 retrieval_train 上训练的召回模型生成。

### 6.3 全局时间检查

- 每个用户的 retrieval_train 最大时间必须小于 rank_train 时间。

- rank_train 时间必须小于 validation，validation 必须小于 test。

- 所有统计特征必须明确使用哪个 split 和哪个 cutoff。

- 商品流行度只能在对应查询时间之前计算；MVP 可统一使用 retrieval_train 统计，并在报告中说明。

- 文本、类别和固定商品元数据可以用于所有阶段；未来评论数量、未来平均评分不能使用。

### 6.4 数据泄露自动测试

```python
def test_split_order_per_user(split_df):
    # For every user:
    # max(retrieval_train.timestamp) < rank_train.timestamp
    # rank_train.timestamp < validation.timestamp < test.timestamp
    ...

def test_no_target_in_history(query_examples):
    # The positive target item must not appear in the history used to build the query.
    ...

def test_features_respect_cutoff(feature_rows):
    # Each dynamic feature stores feature_max_timestamp <= query_timestamp.
    ...
```

## 7. 离线评估指标与统一评估器

### 7.1 核心评估协议

- 每个用户在 validation/test 各有一个目标商品。

- 召回阶段从全商品库或 FAISS 全索引返回 Top-K，排除用户历史商品。

- 不得只从 100 个随机负样本中评估并把结果称为“全量 Recall”。

- 若调试阶段使用 sampled evaluation，结果文件必须含 evaluation_mode="sampled"。

- 最终报告使用 full-catalog evaluation。

### 7.2 指标定义

| 指标 | 定义 | 用途 |

| --- | --- | --- |

| Recall@K / HitRate@K | 目标商品是否出现在 Top-K；单目标时两者相同。 | 评价召回模型能否找回目标。 |

| MRR@K | 目标排名倒数；未命中为 0。 | 奖励更靠前的命中。 |

| NDCG@K | 单目标时为 1/log2(rank+1)。 | 精排主指标。 |

| Catalog Coverage@K | 所有用户推荐结果中出现的唯一商品数 / 可推荐商品数。 | 防止只推头部商品。 |

| Popularity Bias | 推荐商品平均流行度与测试目标流行度比较。 | 观察头部偏置。 |

| P50/P95 latency | 单用户请求端到端耗时分位数。 | 工程指标。 |

| Candidate Recall@200 | 目标是否被候选集包含。 | 决定精排上限。 |



### 7.3 统一评估结果格式

```json
{
  "run_id": "retrieval_twotower_v3",
  "git_commit": "<sha>",
  "data_manifest_sha256": "<hash>",
  "split": "test",
  "evaluation_mode": "full_catalog",
  "num_users": 12345,
  "metrics": {
    "recall@20": 0.0000,
    "recall@50": 0.0000,
    "recall@100": 0.0000,
    "mrr@10": 0.0000,
    "ndcg@10": 0.0000,
    "coverage@10": 0.0000,
    "p50_latency_ms": 0.0,
    "p95_latency_ms": 0.0
  }
}
```

### 7.4 最低结果表

| 模型 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | Coverage@10 | P95 ms |

| --- | --- | --- | --- | --- | --- | --- |

| Popularity | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

| ItemCF | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

| Two-Tower exact | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

| Two-Tower + FAISS | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

| Two-Tower + Ranker | — | — | Candidate Recall | 待运行 | 待运行 | 待运行 |



## 8. 召回基线：Popularity 与 ItemCF

### 8.1 Popularity Baseline

按 retrieval_train 中商品交互次数排序。对每个用户过滤已见商品后返回 Top-K。可选加入按类别热门推荐，但全局热门必须作为最简单基线。

```text
score_popularity(item) = log(1 + interaction_count(item))

recommend(user, k):
    seen = history[user]
    return first k items from global_popularity_order not in seen
```

### 8.2 ItemCF Baseline

使用隐式反馈构造 item-item 相似度。为了控制内存，不建立完整稠密矩阵；按用户历史累积共现，并只保留每个商品最高的邻居。推荐使用 cosine-like normalization。

```text
cooccur(i, j) += 1 / log(1 + len(user_history))

similarity(i, j) = cooccur(i, j) / sqrt(count(i) * count(j))

user_score(candidate j) = sum_{i in recent_history(u)}
                          recency_weight(i) * similarity(i, j)
```

### 8.3 ItemCF 实现约束

- 默认只使用每个用户最近 50 个历史商品，避免长序列产生 O(n²) 爆炸。

- 每个商品只保存 Top 200 相似邻居。

- 相似度构建只读取 retrieval_train。

- 推荐时过滤已见商品，并记录不足 K 个结果的用户数。

- 保存 item_neighbors.parquet，字段为 item_idx、neighbor_idx、similarity、rank。

### 8.4 基线验收

- Popularity 对任意有效用户返回不重复且未见过的商品。

- ItemCF 对至少 95% 测试用户返回不少于 K 个候选；不足时用 Popularity 回填。

- 评估器对一个手工构造小数据集得到可人工验证的 Recall/NDCG。

- 基线结果写入 results/baselines/。

## 9. 双塔召回模型

### 9.1 模型目标

学习用户表示 q(u) 和商品表示 c(i)，使正交互的点积高于批内其他商品。训练完成后离线计算所有商品向量，用 FAISS 检索与用户向量最相似的商品。

### 9.2 MVP 输入特征

| 塔 | 必需特征 | 可选特征 |

| --- | --- | --- |

| 用户塔 | user_idx；最近 N 个 item_idx 的池化；历史长度；类别偏好向量。 | 平均评分；活跃度桶；时间间隔特征。 |

| 商品塔 | item_idx；category_idx。 | 价格桶；平均评分桶；标题 TF-IDF/SVD 向量。 |



### 9.3 推荐架构

```text
UserTower:
  user_id_embedding        -> 64
  mean(recent_item_embs)   -> 64
  category_profile         -> C or embedded 32
  numeric_features         -> normalized
  concat -> Linear(160, 128) -> ReLU -> Dropout
         -> Linear(128, 64) -> L2 normalize

ItemTower:
  item_id_embedding        -> 64
  category_embedding       -> 16
  optional numeric/text    -> 16-64
  concat -> Linear(..., 128) -> ReLU -> Dropout
         -> Linear(128, 64) -> L2 normalize

score(u, i) = dot(user_vector, item_vector) / temperature
```

### 9.4 训练样本

retrieval_train 中每条交互是一个正样本。用户表示只能使用该交互发生之前的历史；为了在 40 小时范围内完成，MVP 可使用“用户全部 retrieval_train 历史聚合”作为静态表示，但必须在报告中说明该近似。更严谨的版本按交互时间生成前缀历史。

> **推荐实现顺序：** 先实现 user_id + item_id + category 的静态双塔并跑通；再加入最近历史池化。不要一开始同时加入文本、复杂序列编码和多任务损失。

### 9.5 损失与负样本

```python
# Batch size B, user embeddings U[B,d], positive item embeddings V[B,d]
logits = U @ V.T / temperature
labels = arange(B)
loss = cross_entropy(logits, labels)

# Batch 中其他 B-1 个正商品自动作为当前用户的负样本。
# 可选：为每个用户额外采样 M 个热门商品或模型高分商品作为 hard negatives。
```

### 9.6 训练配置建议

| 参数 | 默认值 | 搜索范围 |

| --- | --- | --- |

| embedding_dim | 64 | 32, 64, 128 |

| batch_size | 1024（内存不足用 256/512） | 256-2048 |

| learning_rate | 1e-3 | 3e-4, 1e-3, 3e-3 |

| weight_decay | 1e-5 | 0, 1e-6, 1e-5 |

| temperature | 0.07 | 0.05, 0.07, 0.1 |

| epochs | 10，早停 | 5-30 |

| history_length | 20 | 10, 20, 50 |

| seed | 42 | 固定 |



### 9.7 训练时记录

- 每个 epoch 的 train loss、validation Recall@50 和 Recall@100。

- 参数量、训练耗时、峰值内存或显存。

- 最佳 checkpoint、optimizer state、配置和映射版本。

- 随机种子、Python/PyTorch 版本、设备信息。

- 训练异常：NaN、全零向量、Embedding 范数分布。

### 9.8 双塔验收标准

- 一个 1,000 行的 toy 数据集可在 CPU 上 2 分钟内完成训练。

- 训练损失在前几个 epoch 有明显下降，且无 NaN。

- 所有导出商品向量为有限值，L2 范数接近 1。

- 双塔 Recall@100 至少不低于 Popularity；若低于，必须完成诊断报告。

- checkpoint 能重新加载并产生一致的用户/商品向量。

## 10. FAISS 候选检索

### 10.1 索引策略

商品数量在数十万以内时，先使用 IndexFlatIP 作为精确向量检索；因为双塔向量已经 L2 归一化，内积等价于余弦相似度排序。完成正确性后，可选增加 HNSW 或 IVF 索引做延迟对比。

| 索引 | 是否必需 | 用途 |

| --- | --- | --- |

| IndexFlatIP | 必需 | 精确基准；验证向量检索正确。 |

| IndexHNSWFlat | 可选 | 展示近似检索的延迟—召回权衡。 |

| IndexIVFFlat | 可选 | 商品量较大时实验。 |



### 10.2 索引产物

```text
artifacts/faiss/
├── item_flat.index
├── item_hnsw.index          # optional
├── faiss_item_ids.npy       # row -> item_idx
├── index_metadata.json      # dim, metric, item count, model checkpoint hash
└── benchmark.json
```

### 10.3 检索流程

```text
1. Build user vector from user history.
2. Search FAISS for top (K + buffer) items, e.g. 300.
3. Map FAISS row ids back to item_idx.
4. Remove seen items, invalid items, and duplicates.
5. If fewer than requested candidates remain, backfill from ItemCF then Popularity.
6. Return candidate item, retrieval score, retrieval source and retrieval rank.
```

### 10.4 正确性和性能测试

- 随机抽取 100 个用户，将 IndexFlatIP 结果与 NumPy 全量点积 Top-K 比较，必须完全一致或仅有浮点并列差异。

- 记录单用户和批量 128 用户的 P50/P95 检索耗时。

- 记录索引文件大小和构建时间。

- 若使用近似索引，报告 ANN Recall@100：近似结果与精确结果的重合率。

## 11. 精排训练数据生成

### 11.1 每个查询的定义

一个 ranking query 对应“某用户在某个 cutoff 时刻需要推荐”。每个 query 包含固定或上限为 200 个候选商品。rank_train 的目标商品是正标签 1，其余候选为 0。

### 11.2 候选来源

| 来源 | 候选数建议 | 目的 |

| --- | --- | --- |

| Two-Tower FAISS | 150 | 主要个性化候选。 |

| ItemCF | 40 | 补充局部相似和共现候选。 |

| Popularity | 20 | 覆盖冷用户和召回不足。 |

| 去重后截断 | 200 | 控制精排计算量。 |



### 11.3 训练样本生成规则

- 如果目标商品未被召回，仍记录该 query 的 candidate_recall=0，但该 query 无正样本，不能用于 LambdaRank 训练；应单独统计。

- 用于 ranker 训练的 query 必须至少包含一个正样本和一个负样本。

- 验证/测试必须保留所有 query，用于报告候选召回上限。

- 不能把目标商品强行加入 validation/test 候选并计算最终指标；可额外做 oracle reranking 分析，但必须明确标为 oracle。

- 训练时可把正目标加入候选以保证 ranker 有训练信号，但必须保存 was_retrieved 字段，报告真实候选召回。

### 11.4 精排特征

| 特征组 | 特征示例 | 是否必需 |

| --- | --- | --- |

| 召回 | two_tower_score, itemcf_score, popularity_score, retrieval_rank, source_count | 必需 |

| 用户 | history_length, mean_rating, category_entropy, days_since_last_action | 必需 |

| 商品 | popularity, avg_rating, rating_count, category_popularity, item_age_days | 必需 |

| 交叉 | user_category_affinity, cosine(user,item), seen_same_category_count | 必需 |

| 文本/价格 | title_similarity, price_distance, price_missing | 可选 |



### 11.5 精排数据格式

```text
query_id,user_idx,item_idx,label,two_tower_score,itemcf_score,
popularity_score,retrieval_rank,user_history_length,item_popularity,
user_category_affinity,item_avg_rating,...

# 必须按 query_id 连续排序，并生成 group_sizes：
# group_sizes = [number_of_candidates_for_query_1, ...]
```

## 12. LightGBM LambdaRank 精排

### 12.1 模型配置

```python
LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_at=[5, 10, 20],
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
)
```

### 12.2 训练与早停

- 训练集为 rank_train queries；验证集为 validation queries。

- 使用 query group sizes；不能把候选行随机打散后按普通分类器训练。

- 以 validation NDCG@10 早停。

- 测试集只在最终模型冻结后评估一次；调参不得使用测试集。

- 保存特征名称、顺序、重要性、最佳迭代和模型文本文件。

### 12.3 最低对比实验

| 实验 | 目的 | 预期输出 |

| --- | --- | --- |

| Retrieval score 排序 | 无学习精排基线。 | NDCG@10, MRR@10。 |

| LightGBM binary classifier（可选） | 比较 pointwise 与 listwise。 | 同一候选集指标。 |

| LightGBM LambdaRank | 最终精排模型。 | 主结果。 |

| 去掉用户—类别交叉特征 | 消融。 | 指标变化。 |

| 去掉 ItemCF 分数 | 消融。 | 指标变化。 |

| 候选数 100/200/300 | 效果—延迟权衡。 | NDCG 与延迟。 |



### 12.4 精排验收

- 验证集 NDCG@10 不低于仅按 Two-Tower score 排序。若没有提升，必须输出特征分布和错误分析。

- 训练/验证 query 数、正样本率和候选召回率写入结果。

- 特征重要性前 20 项可视化。

- 模型重新加载后，对固定样例输出相同排序。

- 测试指标写入 results/final_test_metrics.json，并标记为 final。

## 13. 冷启动、长尾与分组评估

### 13.1 用户分组

| 用户组 | 定义示例 | 分析 |

| --- | --- | --- |

| Cold / Low-history | 历史 2-5 次 | 双塔是否依赖 user_id；热门回填效果。 |

| Medium | 历史 6-20 次 | 主要人群。 |

| Heavy | 历史 >20 次 | 个性化和历史池化是否改善。 |



### 13.2 商品分组

| 商品组 | 定义 | 指标 |

| --- | --- | --- |

| Head | retrieval_train 流行度前 20% | Recall、曝光占比。 |

| Mid | 中间 60% | Recall、Coverage。 |

| Tail | 后 20% 或低频商品 | Recall、长尾曝光。 |

| Cold item | rank/validation/test 出现，但 retrieval_train 很少或无交互 | 仅在元数据特征可用时分析。 |



### 13.3 必须回答的问题

- Two-Tower 相比 ItemCF 的提升主要来自哪类用户？

- 精排是否进一步加剧热门商品偏置？

- Coverage 提升是否以明显降低 NDCG 为代价？

- 新用户输入偏好类别后，结果是否符合输入而不是只返回全局热门？

- 错误案例中，目标商品未被召回还是被错误排序？

## 14. 在线推理服务与 API 合同

### 14.1 服务启动时加载

- item metadata 和 ID 映射。

- 用户历史或可查询的历史存储。

- Two-Tower checkpoint。

- FAISS 索引。

- ItemCF 邻居和 Popularity 列表。

- LightGBM ranker 和 feature_schema.json。

- 最终指标摘要。

### 14.2 API 端点

| 方法与路径 | 请求 | 响应 |

| --- | --- | --- |

| GET /health | 无 | status、model_version、index_size。 |

| GET /recommend/{user_id} | k、candidate_size、exclude_seen | 推荐商品列表、各阶段分数、延迟。 |

| POST /recommend | recent_item_ids、preferred_categories、k | 新用户或临时画像推荐。 |

| GET /similar-items/{item_id} | k | 相似商品和向量相似度。 |

| GET /metrics | 无 | 离线测试指标和版本信息。 |



### 14.3 请求与响应示例

```json
POST /recommend
{
  "recent_item_ids": ["B000...", "B001..."],
  "preferred_categories": ["Skin Care"],
  "k": 10,
  "exclude_item_ids": []
}

Response:
{
  "request_id": "uuid",
  "model_version": "twotower_v3_ranker_v2",
  "recommendations": [
    {
      "item_id": "B009...",
      "title": "...",
      "category": "Skin Care",
      "rank": 1,
      "ranking_score": 1.234,
      "retrieval_score": 0.812,
      "retrieval_sources": ["two_tower", "itemcf"]
    }
  ],
  "latency_ms": 43.8
}
```

### 14.4 API 错误约定

| 情况 | HTTP | 响应 |

| --- | --- | --- |

| 用户不存在且未提供临时历史 | 404 | USER_NOT_FOUND。 |

| 商品不存在 | 404 | ITEM_NOT_FOUND。 |

| k 越界 | 422 | VALIDATION_ERROR；允许 1-100。 |

| 模型文件缺失 | 503 | MODEL_NOT_READY。 |

| 内部推理异常 | 500 | INFERENCE_ERROR；日志含 request_id，不回传堆栈。 |



### 14.5 性能目标

- 本地 CPU 单用户 Top-10：P95 目标 < 200 ms。

- 批量 32 用户离线推理：报告总耗时和每用户平均耗时。

- 服务启动失败时应明确指出缺失 artifact，而不是静默使用随机模型。

- API 测试使用小型固定 artifact，不依赖外网和完整数据。

## 15. Streamlit 演示界面

### 15.1 页面

| 页面 | 控件 | 展示 |

| --- | --- | --- |

| User Recommendations | 用户 ID、K、排除已见开关 | 商品卡片、分数、来源、延迟。 |

| New User | 最近商品多选、类别多选 | 临时画像推荐和解释。 |

| Similar Items | 商品 ID | 相似商品列表和相似度。 |

| Model Metrics | 模型选择、split | 结果表、消融表、Coverage 和延迟。 |

| Project Info | 无 | 架构、数据规模、限制、GitHub 链接。 |



### 15.2 界面限制

- 前端不直接加载模型；只调用 FastAPI。

- 不把复杂推荐解释包装为“模型知道用户为什么喜欢”；只展示可观察的召回来源和主要特征。

- 演示时允许固定 5-10 个示例用户，避免让观看者猜用户 ID。

- 服务不可用时显示明确错误，而不是空白页面。

## 16. 实验追踪、复现与结果产物

### 16.1 每次运行必须记录

- run_id、开始/结束时间、Git commit、命令。

- 配置文件内容和配置哈希。

- 数据 manifest 哈希。

- 依赖版本和设备信息。

- 训练/验证/测试指标。

- 模型、图表和错误样例文件路径。

### 16.2 推荐目录

```text
runs/
└── 2026-08-01_twotower_v3/
    ├── config.yaml
    ├── environment.json
    ├── metrics.json
    ├── train.log
    ├── checkpoint.pt
    └── figures/

results/
├── retrieval_comparison.csv
├── ranking_comparison.csv
├── ablation_results.csv
├── group_metrics.csv
├── latency_benchmark.csv
└── final_test_metrics.json
```

### 16.3 防止结果造假或混淆

- README 的表格由 scripts/update_readme_results.py 从结果 CSV 生成或人工复制后核对。

- 所有指标注明 split 和 evaluation_mode。

- “模拟数据结果”与“真实子集结果”分开。

- 测试集只保留一组 final 结果；调参过程只用 validation。

- 失败实验也保留配置和简短结论，便于技术报告讨论。

## 17. 测试、代码质量与失败处理

### 17.1 测试层级

| 层级 | 示例 | 运行频率 |

| --- | --- | --- |

| 单元测试 | ID 映射、时间切分、指标、过滤已见、特征计算。 | 每次修改。 |

| 组件测试 | toy 数据上的双塔训练、FAISS 索引、ranker fit。 | 每个工作包。 |

| API 测试 | health、recommend、错误码、响应 schema。 | 服务修改后。 |

| 烟雾测试 | 小数据端到端：preprocess → train → evaluate → serve。 | 合并里程碑前。 |

| 完整运行 | 真实子集训练和测试评估。 | 模型定稿时。 |



### 17.2 必需测试文件

```text
tests/
├── test_data_cleaning.py
├── test_split.py
├── test_metrics.py
├── test_candidate_filter.py
├── test_popularity.py
├── test_itemcf.py
├── test_two_tower_smoke.py
├── test_faiss_index.py
├── test_ranking_features.py
├── test_ranker_smoke.py
└── test_api.py
```

### 17.3 代码规范

- 核心函数有类型标注和简短 docstring。

- 使用 pathlib；禁止到处拼接字符串路径。

- 随机数通过统一 seed_everything() 设置。

- 模块通过 logging 输出，不在核心代码中使用大量 print。

- 配置来自 YAML 或 dataclass，不在脚本内散落魔法数字。

- Notebook 仅用于 EDA 和可视化；模型逻辑从 src/ 导入。

- ruff check . 和 pytest -q 必须通过。

### 17.4 失败处理原则

| 失败 | 处理 |

| --- | --- |

| 内存不足 | 缩小列、分块处理、降低交互上限、减小 batch。 |

| 双塔不如热门度 | 检查切分、负样本、用户表示；报告失败，不伪造提升。 |

| 目标候选召回率低 | 增加候选数、融合 ItemCF/Popularity、检查已见过滤。 |

| 精排无提升 | 核对 query group；检查特征泄露/常数；比较 retrieval score 基线。 |

| API 过慢 | 缓存用户向量、批量特征、减少候选数；先测各阶段耗时。 |

| 数据字段变化 | 下载器做 schema validation，并在错误中列出缺失字段。 |



## 18. 仓库结构和命令规范

### 18.1 完整仓库结构

```text
recommender-system/
├── README.md
├── SPEC.md
├── pyproject.toml
├── uv.lock                    # or requirements.lock
├── Makefile
├── .env.example
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── popularity.yaml
│   ├── itemcf.yaml
│   ├── two_tower.yaml
│   ├── ranker.yaml
│   └── service.yaml
├── data/                      # raw/interim/processed are gitignored
├── artifacts/                 # large artifacts are gitignored
├── models/                    # model binaries are gitignored
├── runs/                      # experiment runs are gitignored
├── results/                   # small JSON/CSV results may be committed
├── reports/
│   ├── technical_report.md
│   └── figures/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_error_analysis.ipynb
├── src/recsys/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── data/
│   │   ├── download.py
│   │   ├── clean.py
│   │   ├── split.py
│   │   ├── features.py
│   │   └── schemas.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── group_analysis.py
│   ├── retrieval/
│   │   ├── popularity.py
│   │   ├── itemcf.py
│   │   ├── dataset.py
│   │   ├── two_tower.py
│   │   ├── train.py
│   │   └── faiss_index.py
│   ├── ranking/
│   │   ├── candidates.py
│   │   ├── features.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── serving/
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   └── schemas.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── timing.py
├── dashboard/
│   └── streamlit_app.py
├── scripts/
│   ├── run_smoke_pipeline.sh
│   ├── benchmark_api.py
│   └── update_readme_results.py
└── tests/
```

### 18.2 CLI 命令

```bash
# Environment
make install
make lint
make test

# Data
python -m recsys.data.download --config configs/data.yaml
python -m recsys.data.clean --config configs/data.yaml
python -m recsys.data.split --config configs/data.yaml

# Baselines
python -m recsys.retrieval.popularity --config configs/popularity.yaml
python -m recsys.retrieval.itemcf --config configs/itemcf.yaml

# Two-tower and index
python -m recsys.retrieval.train --config configs/two_tower.yaml
python -m recsys.retrieval.faiss_index --config configs/two_tower.yaml

# Ranking
python -m recsys.ranking.candidates --config configs/ranker.yaml
python -m recsys.ranking.train --config configs/ranker.yaml

# Evaluation
python -m recsys.evaluation.evaluator --model final --split test

# Serving
uvicorn recsys.serving.app:app --host 0.0.0.0 --port 8000
streamlit run dashboard/streamlit_app.py
```

### 18.3 Makefile 目标

| 目标 | 内容 |

| --- | --- |

| make install | 创建环境并安装开发依赖。 |

| make lint | ruff check + formatting check。 |

| make test | pytest -q。 |

| make smoke | 在 tests/fixtures/small_data 上运行端到端流程。 |

| make api | 启动 FastAPI。 |

| make demo | 启动 Streamlit。 |

| make clean-artifacts | 删除可重建 artifact，不删除原始数据。 |



## 19. 20 天开发计划（每天约 2 小时）

| 天 | 目标 | 必须产物 | 退出条件 |

| --- | --- | --- | --- |

| 1 | 初始化仓库与环境 | pyproject、目录、Makefile、SPEC、基础测试。 | make lint/test 可运行。 |

| 2 | 下载与 schema 审计 | download.py、raw manifest。 | 能读取一小批 review/meta。 |

| 3 | 清洗与 k-core | clean.py、统计 JSON。 | 过滤后规模合理且可复现。 |

| 4 | 时间切分 | split.py、split_stats。 | 泄露测试通过。 |

| 5 | 统一指标 | metrics.py、手工测试。 | toy 数据指标正确。 |

| 6 | Popularity | 模型、结果。 | full-catalog Recall 可运行。 |

| 7 | ItemCF 构建 | 邻居表。 | 小数据正确，真实数据不爆内存。 |

| 8 | ItemCF 推荐与评估 | 候选和结果。 | 对 95% 用户返回足够候选。 |

| 9 | Two-Tower 数据集 | Dataset、batch、history features。 | 单 batch 检查通过。 |

| 10 | Two-Tower 模型 | 模型和 smoke train。 | toy 数据 loss 下降。 |

| 11 | 真实训练 | checkpoint、validation 指标。 | 有可加载最佳模型。 |

| 12 | 调参与一次消融 | 对比表。 | 选择最终双塔配置。 |

| 13 | FAISS 索引 | index、映射、benchmark。 | 与精确点积一致。 |

| 14 | 候选融合 | Top-200 候选。 | candidate Recall 记录。 |

| 15 | 精排特征 | rank_train parquet。 | 无 NaN/常数异常；groups 正确。 |

| 16 | LambdaRank | ranker、validation NDCG。 | 不低于 retrieval 排序或有诊断。 |

| 17 | 测试集和分组分析 | final 指标、长尾表。 | 测试集只运行最终模型。 |

| 18 | FastAPI | 端点和测试。 | 本地请求返回正确结果。 |

| 19 | Streamlit 和 Docker | Demo、Dockerfile。 | 录屏流程可运行。 |

| 20 | README、报告、简历 | 完整文档和结果表。 | 证据链一致；项目可发布。 |



### 19.1 每日两小时模板

- 前 10 分钟：阅读昨天日志，确定唯一的当天退出条件。

- 第一个 50 分钟：实现核心功能，不调整文档样式和前端。

- 休息 10 分钟。

- 第二个 40 分钟：运行测试/实验、修复阻塞错误。

- 最后 10 分钟：Git commit、记录结果、写明明天第一步。

## 20. Codex 工作包与可直接复制的提示词

以下工作包按顺序执行。每个提示词使用前，把 SPEC.md 放在仓库根目录，并确保 Codex 能读取当前文件树。

### WP-00 仓库脚手架

目标：建立可安装的 Python 包、质量工具和空模块。

文件：pyproject.toml、Makefile、src/recsys、tests、configs、.gitignore。

验收：`python -c "import recsys"`、`ruff check .`、`pytest -q` 通过。

可复制给 Codex 的提示词：

```text
Read SPEC.md sections 1, 17, and 18. Inspect the current repository. Create only the project scaffold for WP-00. Use a src-layout Python package named recsys, Python 3.11, pytest, ruff, PyYAML, pandas, pyarrow, duckdb, pydantic and typer. Add a Makefile with install/lint/test/smoke placeholders. Add one import smoke test. Do not implement recommender logic yet. Before editing, list files you will create. After editing, run the acceptance commands and report exact output.
```

### WP-01 数据下载与审计

目标：下载/读取指定 Amazon Reviews 2023 类别并验证字段。

文件：data/download.py、schemas.py、configs/data.yaml、测试。

验收：支持 limit 参数；生成 raw_manifest.json；缺字段时明确报错。

可复制给 Codex 的提示词：

```text
Read SPEC.md sections 5 and 18. Implement WP-01 only. Create a configurable downloader/reader for the selected Amazon Reviews 2023 category. It must support a small-row limit for tests and must not commit data. Validate required review and metadata columns, normalize column names, and write artifacts/raw_manifest.json with row counts, fields, timestamps and source configuration. Use dependency injection or local fixtures so unit tests do not access the network. Run tests and show the generated manifest from a fixture.
```

### WP-02 清洗、k-core 与映射

目标：完成去重、评分过滤、迭代 k-core、抽样和连续 ID。

验收：同配置重复运行输出一致；data_manifest 有哈希和统计。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 5. Implement WP-02 only. Add deterministic cleaning for reviews and metadata: required-null filtering, duplicate removal, rating threshold, iterative user/item k-core, optional maximum interaction cap that preserves chronology, and contiguous user/item mappings. Write processed Parquet files plus artifacts/data_manifest.json. Add tests for iterative k-core and deterministic mappings. Do not implement splitting yet.
```

### WP-03 四段式时间切分

目标：生成 retrieval_train、rank_train、validation、test。

验收：每用户严格时间递增；动态特征不使用未来。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 6. Implement the per-user four-way chronological split exactly as specified. Users must have at least five interactions. Persist split labels and split statistics. Add tests that verify strict timestamp ordering, one rank/validation/test target per user, and that each query history excludes its target. Fail loudly for invalid users instead of silently creating leakage.
```

### WP-04 指标与全量评估器

目标：实现 Recall、MRR、NDCG、Coverage 和延迟汇总。

验收：手工 toy case 数值正确；full/sample 模式显式区分。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 7. Implement a model-agnostic evaluator with Recall@K, MRR@K, NDCG@K, catalog coverage, popularity bias, and latency quantiles. Support full_catalog and sampled modes, but never label sampled results as full. Write deterministic hand-calculated unit tests. Define a JSON result schema and a helper to persist run metadata.
```

### WP-05 Popularity 与 ItemCF

目标：实现两个基线和已见过滤/回填。

验收：真实小样本结果；邻居表；候选不足统计。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 8. Implement PopularityRecommender and ItemCFRecommender behind a common protocol. ItemCF must use weighted co-occurrence, normalized similarity, maximum history length and top neighbors per item. Add a CandidateFilter that removes seen/invalid/duplicate items and backfills from popularity. Persist item neighbors. Evaluate both baselines with the shared evaluator. Do not implement neural models.
```

### WP-06 双塔数据与模型

目标：PyTorch 双塔；静态 user/item/category + 历史池化。

验收：toy 训练 loss 下降；checkpoint 重载一致。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 9. Implement a scoped PyTorch two-tower model. Start with user id, item id, item category, user recent-item mean pooling, and small numeric features. Use in-batch softmax negatives, configurable temperature, L2-normalized outputs, deterministic seeds, checkpoint save/load, and early stopping on validation Recall@100. Add a CPU smoke test using a tiny fixture; do not add transformers or text encoders.
```

### WP-07 训练与实验记录

目标：真实子集训练，保存配置、指标、环境。

验收：run 目录完整；最佳模型可导出所有商品向量。

可复制给 Codex 的提示词：

```text
Read SPEC.md sections 9 and 16. Implement the training CLI and run artifact management. Each run must save config, git commit if available, environment versions, logs, epoch metrics, best checkpoint and validation retrieval metrics. Add an item-vector export command. Never fabricate metrics when a run fails; persist a failed status and error summary.
```

### WP-08 FAISS 索引

目标：IndexFlatIP；可选 HNSW。

验收：100 个用户与 NumPy 精确 Top-K 一致；保存 benchmark。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 10. Implement FAISS item indexing from exported normalized item embeddings. Required: IndexFlatIP, id mapping, metadata validation, save/load, single and batch search, seen-item filtering integration, and a correctness test against NumPy brute-force top-k. Add a benchmark command that reports build time, index size, P50 and P95 latency. HNSW may be optional and isolated.
```

### WP-09 精排候选与特征

目标：生成 rank_train/validation/test 的 query-group 数据。

验收：组连续；训练有正负样本；候选召回单独报告。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 11. Build candidate generation and ranking feature pipelines. Merge Two-Tower, ItemCF and Popularity candidates, preserve source scores/ranks, remove seen items, compute required user/item/cross features at the correct cutoff, and write Parquet sorted by query_id plus group size arrays. Report real candidate recall. For training only, allow adding the positive if missing but record was_retrieved=false. Do not force positives into validation or test final evaluation.
```

### WP-10 LambdaRank 精排

目标：训练 LGBMRanker，验证 NDCG，保存模型和特征。

验收：与 retrieval-score 排序比较；特征重要性。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 12. Implement LightGBM LGBMRanker with lambdarank and query groups. Use rank_train for fitting, validation for early stopping/model selection, and test only in the final command. Compare against sorting by retrieval score. Save model, feature schema, best iteration, feature importance and metrics. Add smoke tests for group alignment and deterministic prediction.
```

### WP-11 API 与服务加载

目标：实现五个端点和明确错误。

验收：TestClient 测试；不会返回已见商品；模型缺失时 503。

可复制给 Codex 的提示词：

```text
Read SPEC.md section 14. Implement a FastAPI service that loads artifacts through a single dependency container. Add /health, GET /recommend/{user_id}, POST /recommend, /similar-items/{item_id}, and /metrics with Pydantic schemas and documented errors. Use small fixture artifacts in tests. Ensure recommendations are unique and exclude seen items. Do not make network calls or train models inside request handlers.
```

### WP-12 Demo、Docker 与文档

目标：Streamlit 调 API；容器化；README/报告。

验收：本地完整演示；所有数字有结果文件。

可复制给 Codex 的提示词：

```text
Read SPEC.md sections 15, 18, 21 and 22. Implement a minimal Streamlit client that calls the FastAPI endpoints, a Dockerfile and optional docker-compose for API + dashboard. Finish README with problem, architecture, data, split, models, metrics, commands, results, limitations and demo instructions. Do not invent result values: read them from results files or leave explicit placeholders. Add a release checklist.
```

## 21. 最终验收清单

### 21.1 数据与切分

- ☐ 原始数据未提交 Git；下载/读取方式写入 README。

- ☐ data_manifest 包含类别、过滤参数、规模、时间范围和哈希。

- ☐ 每个用户四段式时间顺序测试通过。

- ☐ 验证/测试目标未出现在对应历史中。

- ☐ 用户/商品映射在训练、评估和服务一致。

### 21.2 模型与评估

- ☐ Popularity、ItemCF、Two-Tower 均有 full-catalog 结果。

- ☐ FAISS 精确索引与 NumPy Top-K 一致。

- ☐ Candidate Recall@200 单独报告。

- ☐ Ranker 与 retrieval-score 排序进行比较。

- ☐ 至少 3 组消融/对比实验。

- ☐ 冷/中/重用户和头/中/尾商品分组指标。

- ☐ 测试集只用于最终冻结模型。

### 21.3 工程与展示

- ☐ ruff 和 pytest 全部通过。

- ☐ make smoke 在小数据上端到端通过。

- ☐ API 五个端点有自动测试。

- ☐ 推荐结果唯一且默认排除已见商品。

- ☐ Docker 镜像可构建；服务可启动。

- ☐ Streamlit 可演示示例用户。

- ☐ README 和英文报告完成。

- ☐ 3 分钟演示视频完成。

- ☐ 简历数字与 results/ 完全一致。

## 22. 简历证据链与项目展示

### 22.1 项目标题

```text
Two-Stage E-commerce Recommendation System | Python, PyTorch, FAISS, LightGBM, FastAPI
GitHub | Technical Report | 3-min Demo
```

### 22.2 简历 bullet 模板

```text
• Built a two-stage recommendation system on [X] million user-item interactions,
  combining popularity, ItemCF and two-tower retrieval with LambdaRank reranking.

• Improved full-catalog Recall@100 from [A] to [B] over the strongest baseline and
  increased NDCG@10 by [C%] through feature-based reranking and candidate fusion.

• Served FAISS-based retrieval and personalized ranking through FastAPI with
  [Y] ms P95 latency, supported by reproducible data, training and evaluation pipelines.
```

### 22.3 README 首屏必须出现

- 一句话项目摘要。

- 系统架构图。

- 数据规模和时间切分。

- 主结果表。

- 运行命令。

- Demo GIF 或截图。

- 项目限制。

### 22.4 面试准备问题

- 为什么推荐系统需要召回和精排两阶段？

- 为什么不能随机切分用户行为？

- In-batch negatives 有什么偏差？热门负样本有什么作用？

- Recall@100 和 NDCG@10 分别衡量什么？

- 为什么目标未被召回时，精排无法解决？

- ItemCF 与 Two-Tower 各自更适合什么用户？

- 为什么使用 LambdaRank，而不是普通二分类？

- 如何处理新用户、新商品和长尾商品？

- FAISS 精确索引和近似索引的权衡是什么？

- 离线指标为什么不等同于线上业务收益？


---

## 附录 A：配置文件示例

```yaml
# configs/data.yaml
project_seed: 42
dataset:
  name: amazon_reviews_2023
  category: All_Beauty
  min_rating: 3.0
  min_user_interactions: 5
  min_item_interactions: 5
  max_interactions: 1500000
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  artifacts_dir: artifacts

# configs/two_tower.yaml
seed: 42
data_manifest: artifacts/data_manifest.json
model:
  embedding_dim: 64
  user_id_dim: 64
  item_id_dim: 64
  category_dim: 16
  hidden_dims: [128, 64]
  dropout: 0.1
  history_length: 20
  temperature: 0.07
training:
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 0.00001
  max_epochs: 10
  early_stopping_patience: 2
  num_workers: 4
  device: auto
evaluation:
  ks: [20, 50, 100]
  candidate_size: 200

# configs/ranker.yaml
seed: 42
candidate_sources:
  two_tower: 150
  itemcf: 40
  popularity: 20
max_candidates: 200
model:
  objective: lambdarank
  metric: ndcg
  ndcg_at: [5, 10, 20]
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 30
  subsample: 0.9
  colsample_bytree: 0.9
  reg_lambda: 1.0
```

## 附录 B：DuckDB 数据表与 SQL 示例

```sql
CREATE TABLE interactions AS
SELECT
    CAST(user_id AS VARCHAR) AS user_id,
    CAST(parent_asin AS VARCHAR) AS item_id,
    to_timestamp(timestamp / 1000.0) AS event_ts,
    CAST(rating AS DOUBLE) AS rating
FROM read_json_auto('data/raw/reviews.jsonl.gz')
WHERE user_id IS NOT NULL
  AND parent_asin IS NOT NULL
  AND timestamp IS NOT NULL
  AND rating >= 3.0;

CREATE TABLE item_metadata AS
SELECT
    CAST(parent_asin AS VARCHAR) AS item_id,
    COALESCE(title, '') AS title,
    COALESCE(main_category, 'Unknown') AS category,
    TRY_CAST(price AS DOUBLE) AS price,
    TRY_CAST(average_rating AS DOUBLE) AS avg_rating,
    TRY_CAST(rating_number AS BIGINT) AS rating_count
FROM read_json_auto('data/raw/meta.jsonl.gz');
```

## 附录 C：Pydantic 接口模型示例

```python
from pydantic import BaseModel, Field

class NewUserRecommendRequest(BaseModel):
    recent_item_ids: list[str] = Field(default_factory=list, max_length=50)
    preferred_categories: list[str] = Field(default_factory=list, max_length=20)
    exclude_item_ids: list[str] = Field(default_factory=list, max_length=200)
    k: int = Field(default=10, ge=1, le=100)

class RecommendationItem(BaseModel):
    item_id: str
    title: str
    category: str | None = None
    rank: int
    ranking_score: float
    retrieval_score: float | None = None
    retrieval_sources: list[str]

class RecommendationResponse(BaseModel):
    request_id: str
    model_version: str
    recommendations: list[RecommendationItem]
    latency_ms: float
```

## 附录 D：关键指标伪代码

```python
def recall_at_k(ranked_items, target_item, k):
    return float(target_item in ranked_items[:k])

def reciprocal_rank_at_k(ranked_items, target_item, k):
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == target_item:
            return 1.0 / rank
    return 0.0

def ndcg_at_k_single_target(ranked_items, target_item, k):
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == target_item:
            return 1.0 / log2(rank + 1)
    return 0.0

def catalog_coverage(all_recommendations, eligible_items):
    unique_recommended = set(chain.from_iterable(all_recommendations))
    return len(unique_recommended) / len(eligible_items)
```

## 附录 E：英文技术报告结构

- 1. Abstract：问题、系统、数据和最主要结果。

- 2. Problem Definition：隐式反馈和两阶段系统。

- 3. Dataset and Temporal Split：过滤、四段切分、防泄露。

- 4. Retrieval Baselines：Popularity、ItemCF。

- 5. Two-Tower Retrieval：特征、损失、负样本。

- 6. Approximate Nearest Neighbor Search：FAISS 和延迟。

- 7. Learning-to-Rank：候选、特征、LambdaRank。

- 8. Experiments：主结果、消融、分组分析。

- 9. Error Analysis：未召回 vs 排序错误。

- 10. Serving：API、延迟和可复现性。

- 11. Limitations and Ethics：公开评论数据、离线指标、热门偏置。

- 12. Conclusion。

## 附录 F：参考资料

- Amazon Reviews 2023 数据集主页: https://amazon-reviews-2023.github.io/main.html

- Amazon Reviews 2023 Hugging Face 数据卡: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

- TensorFlow Recommenders：两阶段推荐概念参考: https://www.tensorflow.org/recommenders

- TensorFlow Recommenders Retrieval Task: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval

- FAISS Getting Started: https://github.com/facebookresearch/faiss/wiki/Getting-started

- LightGBM LGBMRanker: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html

- LightGBM LambdaRank 参数: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html

- FastAPI Docker 部署: https://fastapi.tiangolo.com/deployment/docker/

- MLflow Tracking（可选）: https://mlflow.org/docs/latest/ml/tracking/

> **最后提醒：** 这份规格的价值不在于把所有功能都做满，而在于让数据切分、评估和工程产物形成完整证据链。先保证正确、可复现和可解释，再做可选扩展。
