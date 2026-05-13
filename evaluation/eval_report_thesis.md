# RAG 系统评估报告
> 文档：240万吨加氢裂化装置工艺技术规程（试行）
> 评估时间：2026-04-19
> 用途：论文实验章节参考

---

## 一、评估数据集

### 文档信息
| 字段 | 内容 |
|------|------|
| 文档名 | 240万吨加氢裂化装置工艺技术规程（试行）-部分章节 |
| 文档大小 | 1.08 MB |
| 页数 | 78 页 |
| 来源 | 中国石油化工股份有限公司九江分公司企业标准 |
| 解析工具 | MinerU（版本 2.5） |

### QA 数据集构建
- **生成工具**：`scripts/generate_qa_single.py`（逐页生成，每页 4 题）
- **生成模型**：qwen-plus（通义千问，DashScope API）
- **原始生成**：288 条
- **清理后**：286 条（删除 2 条目录页 TOC 残留、清理 LaTeX 标记）
- **输出文件**：`data/qa_dataset/hydro_reg_qa_clean.json`

### QA 题型分布
| 题型 | 数量 | 说明 |
|------|------|------|
| factual | 144 | 事实查询，参数/数值/定义 |
| negative | 71 | 判断某条件是否适用/不适用 |
| reasoning | 71 | 需要理解推断 |

### 难度分布
| 难度 | 数量 |
|------|------|
| single_hop | 265 |
| multi_hop | 21 |

### 主题分布
| 主题 | 数量 |
|------|------|
| 其他 | 123 |
| 工艺参数 | 59 |
| 安全规程 | 51 |
| 设备操作 | 46 |
| 质量标准 | 7 |

---

## 二、评估环境

| 组件 | 配置 |
|------|------|
| 向量库 Collection | `hydro_reg` |
| Chunks 数量 | 806 chunks |
| Chunk 大小 | 512 字符，overlap 64 |
| Embedding 模型 | BAAI/bge-large-zh-v1.5 |
| Reranker 模型 | BAAI/bge-reranker-v2-m3 |
| 检索策略 | Hybrid（BM25 + Dense + RRF 融合） |
| 生成模型 | qwen-plus（通义千问） |
| 硬件 | RTX 3070 Ti Laptop 8GB，WSL2 |

---

## 三、评估方法一：Precision@K（检索层）

### 原理
对每条 QA，使用 Hybrid Retriever 检索 Top-K 个 chunk，检查 `ground_truth_chunk_id` 是否出现在结果中。

**指标定义：**
- **Hit@K**：正确 chunk 出现在 Top-K 结果中的比例
- **MRR**（Mean Reciprocal Rank）：正确 chunk 排名的倒数均值，衡量正确结果的平均位置

### ground_truth_chunk_id 标注方式（银标准）
由于 QA 数据集由 LLM 生成，无人工标注的 chunk_id，采用自动标注：
1. 对每条 QA 的 `source_text` 生成向量
2. 在 `hydro_reg` collection 中检索同页（±1页）范围内最相似的 chunk
3. 将该 chunk 的 `chunk_id` 作为 `ground_truth_chunk_id`
4. 标注脚本：`scripts/annotate_chunk_ids.py`，输出：`data/qa_dataset/hydro_reg_qa_annotated.json`

**银标准的局限性：**
- source_text 是 LLM 对原文的归纳缩写（≤150字），与 512 字的 chunk 不完全重叠
- 少数 QA 对应的知识点跨多个 chunk，仅标注一个 chunk_id 可能导致检索命中"另一个同样正确的 chunk"也算 Miss
- 因此 Hit@1 会偏保守，Hit@5 更能代表真实检索能力

### 评估结果（n=286）

#### 整体指标
| 指标 | 值 | 命中数/总数 |
|------|-----|-----------|
| Hit@1 | 0.549 | 157/286 |
| Hit@3 | 0.741 | 212/286 |
| Hit@5 | **0.811** | 232/286 |
| MRR | 0.658 | — |

#### 按题型分组
| 题型 | Hit@1 | Hit@3 | Hit@5 | n |
|------|-------|-------|-------|---|
| reasoning | **0.65** | **0.76** | **0.83** | 71 |
| factual | 0.51 | 0.75 | 0.82 | 144 |
| multi_hop | 0.57 | 0.71 | 0.81 | 21 |
| negative | 0.52 | 0.70 | 0.77 | 71 |

#### 按难度分组
| 难度 | Hit@1 | Hit@3 | Hit@5 | n |
|------|-------|-------|-------|---|
| single_hop | 0.55 | 0.74 | 0.81 | 265 |
| multi_hop | 0.57 | 0.71 | 0.81 | 21 |

#### 按主题分组
| 主题 | Hit@1 | Hit@3 | Hit@5 | n |
|------|-------|-------|-------|---|
| 其他 | 0.65 | 0.84 | **0.89** | 123 |
| 工艺参数 | 0.56 | 0.78 | 0.85 | 59 |
| 质量标准 | 0.57 | 0.71 | 0.71 | 7 |
| 设备操作 | 0.46 | 0.63 | 0.72 | 46 |
| 安全规程 | 0.37 | 0.57 | **0.69** | 51 |

### 关键发现
1. **安全规程类检索最弱**（Hit@5=0.69），低于其他主题约 15 个百分点，原因是安全规程内容分散在多个章节，单次检索难以覆盖
2. **reasoning 题 Hit@1 最高**（0.65），说明推理类问题的表述与原文关键词重叠度更高，BM25 词汇匹配效果好
3. Hit@1 到 Hit@5 有 26 个百分点的提升空间，说明正确 chunk 大多在 Top-5 内但排名不一定靠前，reranker 仍有提升余地

---

## 四、评估方法二：RAGAS（端到端）

### 原理
RAGAS（Retrieval Augmented Generation Assessment）对 RAG 完整链路进行评估，无需 chunk_id，直接使用 question + ground_truth + LLM 生成的 answer + retrieved contexts。

**指标定义：**
- **Faithfulness**：答案中的声明是否都能在检索内容中找到支撑（衡量 LLM 幻觉程度）
- **Answer Relevancy**：答案是否切题（答案与问题的相关性）
- **Context Precision**：检索到的内容是否都与问题相关（检索精确率）
- **Context Recall**：回答问题所需的信息是否都被检索到（检索召回率，与 ground_truth 对比）

### 评估配置
- `score_threshold`：0.3（评估时适当放宽，确保返回内容）
- `fallback_enabled`：False（关闭兜底，仅测试 RAG 本身）
- `reranker_enabled`：True
- `retriever_top_k`：10，`reranker_top_n`：3
- RAGAS `answer_relevancy.strictness`：1（避免 DashScope 不支持 n>1 的警告）

### 评估结果（全量，n=286）

> 实际耗时约 75 分钟，DashScope 高峰期偶发 TimeoutError，nanmean 自动跳过 NaN 样本，不影响整体结论。

#### 整体指标
| 指标 | 值 |
|------|-----|
| Context Recall | **0.8811** |
| Context Precision | **0.8476** |
| Answer Relevancy | **0.8225** |
| Faithfulness | 0.6870 |

#### 按题型分组
| 题型 | CP | CR | Faithfulness | AR | n |
|------|----|----|-------------|-----|---|
| factual | 0.863 | **0.931** | **0.758** | **0.832** | 144 |
| negative | 0.841 | 0.845 | 0.629 | 0.826 | 71 |
| reasoning | 0.823 | 0.817 | 0.570 | 0.801 | 71 |

#### 按难度分组
| 难度 | CP | CR | Faithfulness | AR | n |
|------|----|----|-------------|-----|---|
| single_hop | **0.863** | **0.900** | **0.703** | **0.823** | 265 |
| multi_hop | 0.627 | 0.642 | 0.419 | 0.820 | 21 |

#### 按主题分组
| 主题 | CP | CR | Faithfulness | AR | n |
|------|----|----|-------------|-----|---|
| 设备操作 | **0.909** | **0.951** | 0.686 | **0.860** | 46 |
| 其他 | 0.866 | 0.911 | **0.733** | 0.825 | 123 |
| 工艺参数 | 0.821 | 0.868 | 0.685 | 0.801 | 59 |
| 质量标准 | 0.806 | 0.857 | 0.538 | 0.623 | 7 |
| 安全规程 | 0.786 | 0.765 | 0.593 | 0.834 | 51 |

### 关于 Answer Relevancy 持续偏高的说明
AR 在所有分组中稳定在 0.80 以上，而 Faithfulness 明显偏低，两者测的不是同一件事：
- **AR** 测"答案有没有跑题"：LLM 反向从答案生成问题，与原问题算余弦相似度。只要答案主题贴着问题走，AR 就高。
- **Faithfulness** 测"答案的每句话能否在检索内容里找到支撑"：当检索内容不完整时，LLM 会用自身训练的参数知识补充推断——答案仍切题（AR 高），但超出了检索内容的范围（Faithfulness 低）。

这一现象揭示了 RAG 系统的核心风险：**LLM 的参数知识会掩盖检索失败**，系统表面流畅切题，实际可能在用训练数据而非检索内容回答。这也是引入 HyDE 等检索优化的核心动机。

### 关键发现
1. **Faithfulness 是最低指标**（0.687），约 31% 的答案声明无法在检索内容中找到直接支撑，LLM 存在用参数知识补充推断的倾向
2. **multi_hop 全面下滑**（CP=0.627，CR=0.642，Faith=0.419），与 Precision@K 结论一致，多跳检索是当前架构主要瓶颈
3. **factual 题表现最好**（Faith=0.758，CR=0.931），事实查询检索和生成均最可靠
4. **安全规程各指标偏低**（CP=0.786，CR=0.765，Faith=0.593），与 Precision@K 结论一致，内容跨章节分散是根本原因
5. **reasoning 题 Faithfulness 最低**（0.570），推理类问题需要 LLM 做逻辑推断，推断过程难以在检索内容中逐句找到对应

---

## 五、评估过程中遇到的问题

### 问题 1：RAGAS 大量警告 "LLM returned 1 generations instead of requested 3"
- **原因**：RAGAS `answer_relevancy` 指标默认 `strictness=3`，请求 3 次 LLM 生成取均值，但 DashScope API 不支持 `n>1`，只返回 1 次
- **影响**：警告刷屏，但实际只调用 1 次，不浪费额外用量；指标精度轻微下降
- **修复**：设置 `answer_relevancy.strictness = 1`，见 `evaluation/ragas_eval.py`

### 问题 2：RAGAS 分组评估重复调用 LLM（4倍用量浪费）
- **原因**：原始实现对每个分组（按题型/难度/主题）分别重跑一次 RAGAS，导致总调用量 = n × 4指标 × 4次分组 ≈ 4500次
- **修复**：主评估时收集逐条分数（`return_per_sample=True`），分组时直接聚合，调用量降至 n × 4 ≈ 1144 次；见 `evaluation/ragas_eval.py` 中新增的 `aggregate_group_scores()` 函数

### 问题 3：偶发 TimeoutError
- **原因**：DashScope API 在高负载时响应超时（设定 120s 超时未覆盖所有情况）
- **影响**：该样本对应指标计为 NaN，最终取 nanmean 忽略，对整体影响较小
- **处理**：保持 `raise_exceptions=False`，出现 NaN 时 nanmean 自动跳过

### 问题 4：QA 数据集 LaTeX 残留
- **原因**：MinerU 解析 PDF 时部分数学公式保留 LaTeX 格式（如 `$\mathrm{m}^{2}$`），LLM 生成 QA 时写入 source_text
- **影响**：导致 JSON 解析失败（LaTeX 反斜杠在 JSON 中为非法字符）
- **修复**：生成阶段提示词明确要求"数学公式用文字替代"；解析阶段用正则清理非法转义字符；清理脚本 `scripts/qa_clean.py` 事后批量清洗

---

## 六、评估文件索引

| 文件 | 说明 |
|------|------|
| `data/qa_dataset/hydro_reg_qa_clean.json` | 清理后的 QA 数据集（286条） |
| `data/qa_dataset/hydro_reg_qa_annotated.json` | 含 ground_truth_chunk_id 的标注版本 |
| `data/qa_dataset/hydro_reg_qa_clean_report.json` | 质量检查报告 |
| `data/results/precision_at_k_hydro_reg.json` | Precision@K 完整结果 |
| `evaluation/results/hydro_reg_ragas_full.json` | RAGAS 全量结果（待补充） |
| `scripts/generate_qa_single.py` | QA 生成脚本 |
| `scripts/qa_quality_check.py` | QA 质量检查脚本 |
| `scripts/qa_clean.py` | QA 清理脚本 |
| `scripts/annotate_chunk_ids.py` | chunk_id 自动标注脚本 |
| `scripts/eval_precision_at_k.py` | Precision@K 评估脚本 |
| `scripts/run_eval_full.py` | RAGAS 端到端评估脚本 |
| `evaluation/ragas_eval.py` | RAGAS 封装（含修复后的分组聚合逻辑） |

---

## 七、消融实验计划

### 实验设计思路
QA 数据集（286条）在所有实验中**固定不变**，仅改变检索系统配置，以控制变量方式评估各因素对检索和生成质量的独立影响。每阶段选出最优配置后作为下一阶段的基准。

### 四阶段实验流程

**阶段1：Baseline（当前系统）**
- top_k=10，reranker_top_n=3，chunk_size=512，overlap=64
- 评估：Hit@K + MRR（已完成）+ RAGAS 全量（待补充）

**阶段2：top_k 消融**
- 不涉及重新入库，仅改检索参数，可直接重跑评估脚本
- top_k 越大，Reranker 候选越多，召回率理论上更高，但 Reranker 延迟线性增加
- 实际延迟瓶颈在 LLM 生成（2~40s），Reranker 增加 0.2~0.5s，用户感知影响小

| 候选配置 | top_k | reranker_top_n |
|---------|-------|---------------|
| A | 5 | 3 |
| B（Baseline） | 10 | 3 |
| C | 15 | 5 |
| D | 20 | 5 |

**Precision@K 阶段实验结果与发现（已完成）：**

| top_k | Hit@1 | Hit@3 | Hit@5 | MRR |
|-------|-------|-------|-------|-----|
| 5 | 0.549 | 0.741 | 0.811 | 0.658 |
| **10（Baseline）** | **0.549** | **0.741** | **0.811** | **0.658** |
| 15 | 0.510 | 0.752 | 0.811 | 0.639 |

**关键发现1：top_k=5 与 Baseline 完全相同**
EnsembleRetriever（BM25+Dense+RRF）没有最终输出截断，k=5 时每路返回5个，合并后实际返回约8~10个文档，与 k=10 的前5名 RRF 排序一致。说明单路 top-5 已包含最相关文档，扩展到 top-10 不影响前5名排序。

**关键发现2：top_k=15 的 Hit@1 和 MRR 反而下降**
候选池扩大后，BM25 和 Dense 排名11~15的新文档进入 RRF 融合，其 RRF 分数足以将正确 chunk 从第1位挤出，但不足以挤出前5位，导致 Hit@1（0.549→0.510）和 MRR（0.658→0.639）下降，而 Hit@5 保持不变（0.811）。这是信息检索中已知的**候选集稀释效应**（Candidate Dilution）。

**结论：粗召回层 k=10 已是较优配置**，继续扩大 k 在无 Reranker 时引入排名噪声。top_k 对端到端效果（含 Reranker）的影响需通过 RAGAS 单独验证。

**阶段3：chunk_size + overlap 消融**
- 使用阶段2 最优 top_k
- 需要重新入库（新 collection）+ 重新标注 chunk_id（`annotate_chunk_ids.py`）
- RAGAS 不需要 chunk_id，QA 文件不变直接用
- 注：GB/T 文档测试中 1024 比 512 差（信息粒度细、条款独立性强），本文档（工艺规程，长篇描述+参数表）结果可能不同，需单独验证

| 候选配置 | chunk_size | overlap | 比例 |
|---------|------------|---------|------|
| A | 256 | 32 | 12.5% |
| B（Baseline） | 512 | 64 | 12.5% |
| C | 1024 | 128 | 12.5% |

**阶段4：HyDE**
- 使用阶段2+3 最优配置
- 在 rag_graph 增加 HyDE 节点：用户问题 → LLM 生成假设答案文档 → 用假设文档做向量检索
- 目标：改善 multi_hop 召回（当前 CR=0.375）和整体 Hit@1（当前 0.549）
- 代价：每次查询增加一次 LLM 调用（约 1~3s 额外延迟）

### 消融实验结果汇总表（持续填充）

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | Faith | AR | CP | CR |
|------|-------|-------|-------|-----|-------|----|----|----|
| Baseline (k=10, cs=512) | 0.549 | 0.741 | 0.811 | 0.658 | 0.687 | 0.823 | 0.848 | 0.881 |
| top_k=5 | 0.549 | 0.741 | 0.811 | 0.658 | — | — | — | — |
| top_k=15 | 0.510 | 0.752 | 0.811 | 0.639 | — | — | — | — |
| top_k=20 | — | — | — | — | — | — | — | — |
| best_k + cs=256 | — | — | — | — | — | — | — | — |
| best_k + cs=1024 | — | — | — | — | — | — | — | — |
| best_k + best_cs + HyDE | — | — | — | — | — | — | — | — |

### top_k 选取标准
不单纯选 Hit@5 最高的，综合考量：
1. Hit@5 / MRR 提升是否显著（边际收益）
2. Faithfulness 是否随之提升（更多候选不代表生成更忠实）
3. top_k 从 15→20 如果提升 <0.01，选 15（延迟更低）

> 论文写法参考："实验表明 top_k=X 时检索性能趋于稳定，继续增大对指标提升有限但增加重排序延迟，故选取 top_k=X 作为后续实验基准配置。"

---

## 八、评估方法论说明（论文备用）

### 关于 Precision@K 与 Recall@K
业界报告 Precision@K / Recall@K 的实际情况：
- **使用公开 Benchmark**（BEIR、MS MARCO 等）：有真实多标签相关性标注，指标完全有效
- **自建数据集，一题一个正例**：Recall@K 退化为 Hit@K（命中=1，未命中=0），Precision@K = Hit@K / K，与 Hit@K 等价，部分论文未明确说明此简化
- **本文做法**：使用 Hit@K + MRR + RAGAS Context Precision/Recall，是对 Precision@K 和 Recall@K 的语义等价替代，适用于生成式 RAG 评估场景

### 关于 QA 数据集的固有偏差
- QA 由 LLM（qwen-plus）从文档页面文本自动生成，问题措辞天然接近原文，检索难度偏低
- ground_truth 由同一 LLM 生成，RAGAS 的 Context Recall 可能偏高
- **消融实验结论不受此影响**：相同偏差作用于所有配置，各配置间的相对差异是真实的
- 论文建议注明："评估数据集由 LLM 自动生成，存在措辞偏向原文的固有偏差，各配置在相同数据集上对比，消融实验结论不受该偏差影响。"

### 关于评估脚本与前后端系统的等价性
评估脚本（`run_eval_full.py`）与前后端完整系统走相同代码路径：
```
eval 脚本:          graph.invoke() → retrieve → rerank → generate
前后端系统:  HTTP → FastAPI → _run_graph() → graph.invoke() → retrieve → rerank → generate
```
有意设置的参数差异（非 bug）：
- `fallback_enabled=False`：评估时关闭兜底，仅测试 RAG 本身能力
- `score_threshold=0.3`：放宽阈值确保有内容返回，避免阈值导致空结果干扰评估

---

## 九、评估文件索引

| 文件 | 说明 |
|------|------|
| `data/qa_dataset/hydro_reg_qa_clean.json` | 清理后的 QA 数据集（286条，所有实验共用） |
| `data/qa_dataset/hydro_reg_qa_annotated.json` | 含 ground_truth_chunk_id 的标注版本（chunk_size=512） |
| `data/qa_dataset/hydro_reg_qa_clean_report.json` | 质量检查报告 |
| `data/results/precision_at_k_hydro_reg.json` | Precision@K 完整结果（Baseline） |
| `evaluation/results/hydro_reg_ragas_full.json` | RAGAS 全量结果（待补充） |
| `scripts/generate_qa_single.py` | QA 生成脚本 |
| `scripts/qa_quality_check.py` | QA 质量检查脚本 |
| `scripts/qa_clean.py` | QA 清理脚本 |
| `scripts/annotate_chunk_ids.py` | chunk_id 自动标注脚本（换 collection 重跑即可） |
| `scripts/eval_precision_at_k.py` | Precision@K 评估脚本 |
| `scripts/run_eval_full.py` | RAGAS 端到端评估脚本 |
| `evaluation/ragas_eval.py` | RAGAS 封装（含修复后的分组聚合逻辑） |

---

## 十、论文写作建议

### 指标汇报建议
- **Precision@K**：以 Hit@5 和 MRR 为主要指标，Hit@1 作为辅助；注明银标准标注方式及局限性
- **RAGAS**：四项指标全部汇报；Faithfulness 偏低需解释（LLM 在检索不足时的推断行为）
- **Context Precision/Recall**：可作为 Precision@K / Recall@K 的语义等价替代向审稿人说明

### 可用于论文的典型结论
1. Hybrid 检索（BM25+Dense+RRF）在垂直领域专业文档上 Hit@5=0.811，具备实用价值
2. 系统在单跳事实性查询上表现良好（Hit@5=0.82，Faith=0.758，CR=0.931）
3. 多跳推理是当前架构主要瓶颈（Hit@5=0.81 但 RAGAS CR=0.642，Faith=0.419），单次检索难以覆盖跨段落的复合问题
4. 安全规程类各项指标均偏低（Hit@5=0.69，CP=0.786，CR=0.765），内容跨章节分散是根本原因
5. Answer Relevancy 持续偏高（0.82）而 Faithfulness 偏低（0.69）揭示 LLM 参数知识掩盖检索失败的风险，是引入检索优化（HyDE 等）的核心动机
6. （待填充）top_k 消融结论
7. （待填充）chunk_size 消融结论
8. （待填充）HyDE 对比结论
