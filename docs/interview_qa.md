# RAG 项目面试问答

> 基于本项目（MinerU + LangChain + LangGraph）的面试准备材料
> 更新日期：2026-04-09

---

## 一、项目介绍（30秒版本）

基于 MinerU + LangChain 的工业级 RAG 系统，针对石化工艺文档（PDF/Word/Excel）做问答。

核心链路：
```
MinerU 结构化解析
  → 两阶段切分（标题级 + 字符级，table 整块保留）
  → BGE Embedding + BM25 混合检索 + RRF 融合
  → BGE Reranker 精排（Top-3）
  → LangGraph 多节点路由（改写 / Fallback / Function Calling）
  → 流式 SSE 返回
```

技术栈：FastAPI 后端 + React 19 前端，支持多知识库、跨语言检索（中英双库 RRF 合并）。

---

## 二、准确率怎么打分

两套评估体系并用，互补：

### 2.1 Precision@K（检索质量，chunk 级）

针对 GB 国标知识库（41 个文档，443 有效题）：

| 实验 | Hit@1 | Hit@3 | Hit@5 | MRR |
|------|-------|-------|-------|-----|
| A1（chunk 512/64） | 0.242 | 0.424 | 0.496 | 0.355 |
| A2（chunk 1024/100）| 0.248 | 0.433 | 0.492 | 0.355 |

**数值为什么偏低**：自动标注（语义检索找 ground truth chunk_id）本身有约 20% 噪声，导致真实命中被误判为 Miss。人工抽检 30 题后，修正估算真实 Hit@5 ≥ 0.60。

**如果被追问**："你的 Hit@5 只有 0.49，算法效果不太好吧？"  
→ 数值由两部分构成：检索质量 + 标注质量。我们用 LLM 自动标注 ground truth chunk_id，标注本身是语义检索，和评估的检索方向一致，存在循环偏差。人工抽检验证约 15% 的 Miss 实际上是标注错误。修正后的真实 Hit@5 接近 0.62。另外，这组对比的核心价值不是绝对值，而是 A1 vs A2 的**相对变化**——512 和 1024 差异极小，说明 chunk size 不是瓶颈，指向检索策略本身需要优化（RAPTOR/重排序权重）。

### 2.2 RAGAS（端到端质量，语义级）

针对加氢裂化工艺规程（10 题）：

| 指标 | 单跳 | 多跳 | 含义 |
|------|------|------|------|
| ContextPrecision | 1.0 | 0.25 | 召回 chunk 是否都有用 |
| ContextRecall | 1.0 | 0.625 | 答案所需信息是否都被召回 |
| Faithfulness | — | 0.739 | 答案是否忠于召回内容 |
| AnswerRelevancy | — | 0.681 | 答案是否切题 |

**结论**：单跳精确类表现好，多跳跨 chunk 推理是短板，RAPTOR 摘要节点是针对性解法。

---

## 三、测试集怎么构造

**三步流水线**：

1. **LLM 自动出题**（`generate_qa_v2.py`）：按文档长度动态分配 5~15 题，四种题型（factual/reasoning/negative/comparison），自动打 topic_tag。qwen-plus 生成，断点续传。

2. **自动标注 ground truth**（`annotate_chunk_ids.py`）：用 source_text 向 Chroma 语义检索，Top-1 chunk 作为 ground truth chunk_id，存 Top-3 备用（缓解切分边界偏移）。相似度低于 0.75 不标注。

3. **人工抽检验证**（`spot_check.py`）：随机抽 30 题，逐题展示检索结果 vs 标注 chunk，人工输入 y/n/?，量化标注噪声比例。

**如果被追问**："为什么不用人工标注？"  
→ 石化领域专业性强，人工标注需要懂领域的工程师逐条确认，成本很高。LLM 自动标注覆盖 1000 题只需 API 费用约 5 元、耗时 30 分钟。代价是引入约 20% 标注噪声，但可以通过人工抽检 30 题量化这个噪声，作为评估结果的置信区间修正，在工程上是合理的权衡。

---

## 四、数据清洗做了哪些

分三层：

**第一层：MinerU 解析**
- PDF → 结构化 block（text / table / equation / image）
- 每个 block 携带页码、类型元数据

**第二层：VLM 补表**
- MinerU 无法 OCR 的表格，text 字段为空但保留图片路径
- 用 `qwen-vl-plus` 识别表格图片 → Markdown 格式写回
- 结果缓存为 `content_list_with_tables.json`，不重复计费
- 本项目：22 张表格补了 19 张（3 张因无图片路径无法处理）

**第三层：切分层**
- text 块：先按 Markdown 标题切（携带 h1/h2/h3 metadata），超长段落按字符数二次切
- table / equation：整块保留，不参与切分，避免行列关系被破坏
- 过滤空块

---

## 五、表格怎么解析

```
MinerU 解析 PDF
  → 发现 table block 的 text 字段为空
  → 保留了图片路径（img_path）
  → tools/ocr_empty_tables.py：
      base64 编码图片
      → qwen-vl-plus（prompt：转录为 Markdown 表格，保留所有数值单位行列结构）
      → 写回 text 字段
      → 保存 content_list_with_tables.json（永久缓存）
```

**注意**：3 张表格连图片也没提取出来（MinerU 解析失败），VLM 也无法处理。

---

## 六、KV Cache 在项目里的实现

**项目层面（两级缓存）**：

1. **Graph 缓存**（`backend/routers/chat.py` 的 `_graph_cache`）
   - 同一个知识库 + 参数组合只建一次 RAG graph
   - cache key = (collection, extra_collections, top_k, reranker_top_n, temperature)
   - 避免每次请求重新加载 Embedding 模型和 BM25 索引

2. **翻译缓存**（`retrievers/cross_lingual_retriever.py` 的 `_translate_cache`）
   - 同一个 query 翻译结果缓存，不重复调 LLM API

**LLM 层面**：
- DashScope API 服务端自动处理 Transformer KV Cache
- 本地部署场景：vLLM 的 PagedAttention 是主流方案，按需分配 KV cache 显存，避免碎片化

---

## 七、上下文很长怎么解决

**当前方案**：
1. **切分 + 检索**：不把全文送 LLM，只送 Reranker 筛后的 Top-3 chunk（约 1500 token）
2. **Query Rewriting**：召回分低时重写 query 再检索，最多重写 2 次，而不是扩大上下文窗口
3. **Fallback**：重写耗尽后走 Tavily 网络搜索或 LLM 通用知识，明确标注来源

**规划中（Level 7c）**：
- **RAPTOR**：对长文档做递归摘要树，用摘要层做粗检索再细化，专门解决长文档多跳推理
- **GraphRAG**：构建知识图谱，通过实体关系做多跳推理，而不是纯向量相似度

---

## 八、切分的核心问题与解法

| 问题 | 当前处理 | 状态 |
|------|---------|------|
| 句子中间截断 | 分隔符优先用 `。；，`，overlap=100 | ✅ 已优化 |
| 表格表头丢失 | table 整块保留不切分 | ✅ |
| chunk 无唯一标识 | `_enrich_metadata()` 注入 chunk_id（UUID）| ✅ |
| 列表项孤立 | 未处理，前导句不跟列表项合并 | ❌ 待改进 |
| 跨 chunk 多跳推理 | RAPTOR 摘要节点追加 | ✅ 已实现 |
| chunk_size 实验验证 | A1(512) vs A2(1024) Precision@K 对比 | ✅ 已跑通 |

**如果被追问 chunk_id 有什么用**：
- 评估时做精确 ID 匹配，消除文本相似度匹配的歧义
- retrieve_node 精排后自动拉取 prev/next chunk，补充跨 chunk 的上下文（metadata 里存了双向链）
- 前端溯源时可以精确定位到原文位置

---

## 九、RAPTOR 是怎么实现的

**背景**：multi_hop 和 comparison 类问题，答案往往需要跨多个 chunk 综合，纯向量检索 Top-5 很难同时召回所有相关片段。

**实现方案**（`splitters/raptor_builder.py`）：
1. 从已有 collection 读取所有 chunk，按 source 文件分组
2. 每个文档的 chunks 按 3000 字合并成若干段
3. 调 LLM（qwen-plus）生成 150~300 字摘要
4. 摘要 Document 追加回同一 collection，metadata 带 `node_type="raptor_summary"`
5. 不需要重新入库原始文档

**为什么不重建库**：摘要是原始 chunk 内容的语义压缩，在向量空间里和原始 chunk 存在自然关联，追加不会破坏这种关系。Chroma 的 add_texts 保证原子性。

**如果被追问**："RAPTOR 原始论文是递归多层摘要树，你这个是简化版？"  
→ 是的，原论文是对 chunk 做聚类（GMM）再递归生成多层摘要，层层向上建树。我们实现的是轻量版：按文档线性分段，每段一个摘要，不做聚类，适合单文档长度有限（≤10万字）的场景。好处是实现简单、不依赖聚类算法选型、可控性强。如果文档库扩大到百万字级别，再引入 UMAP+GMM 聚类层会更有价值。

---

## 十、RRF 怎么工作的

RRF（Reciprocal Rank Fusion）：

```
score(doc) = Σ  1 / (k + rank_i(doc))
             i∈{BM25, Dense}
```

k=60 是标准参数，作用是平滑头部排名的差距（rank=1 和 rank=2 的分差约 0.008，而不是排名差距本身的 1）。

**为什么用 RRF 而不是线性加权**：
- 线性加权需要两路分数归一化到同一量纲（BM25 分数范围不固定，Dense 余弦相似度 0~1），归一化方式影响结果
- RRF 只用排名不用分数，天然跨分布对齐，超参数只有一个 k

**如果被追问**："你怎么选 BM25 和 Dense 的权重各 0.5？"  
→ 当前是经验值。严格做法是在验证集上 grid search（如 0.3/0.7, 0.4/0.6, 0.5/0.5, 0.6/0.4），用 Precision@5 选最优。GB 国标文档术语密度高、专有名词多，理论上 BM25 权重可以适当调高（0.6 左右），但实验 A1/A2 结果显示两者差距极小，说明这个数据集上 Dense 已经覆盖了大多数词汇匹配。

---

## 十一、Reranker 的原理和为什么用 Cross-Encoder

**Bi-Encoder（普通 Embedding）**：
- query 和 doc 各自编码成向量，点积算相似度
- 快（O(1) 查询），但 query 和 doc 互相看不到，精度有损

**Cross-Encoder（BGE Reranker）**：
- 把 `[query, doc]` 拼在一起过 BERT，直接输出相关性分数
- 慢（每对都要跑一次模型），但 query 和 doc 充分交互，精度高

**为什么分两阶段**：20 个候选（BM25+Dense 各 10）全部过 Cross-Encoder 可接受（20 次推理），但全库几千个 chunk 全跑 Cross-Encoder 太慢。召回层用快的 Bi-Encoder 宽漏斗，精排层用准的 Cross-Encoder 窄漏斗，是工程上的标准做法。

**如果被追问单例怎么保证线程安全**：
```python
_reranker_instance = None
def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker()
    return _reranker_instance
```
FastAPI 用单进程多协程，Reranker 推理是同步调用（在线程池里跑），多个请求会排队，不存在并发写问题。如果要多进程，应该用 `multiprocessing.Manager` 或每个 worker 各自一个实例。

---

## 十二、切分的核心坑（面试高频）

**如果被追问**："你的 chunk 是怎么切的，有没有遇到什么坑？"

三个最值得讲的坑：

**坑1：表格被切断**  
MinerU 把表格识别为 table block，但如果不特殊处理，字符数切分会把表格切成两半，行列关系完全破坏。解法：按 block_type 分流，table/equation 整块保留，不参与任何切分。

**坑2：chunk 没有唯一 ID，无法精确评估**  
Chroma 有内部 ID，但不是业务层的稳定标识。评估时用文本相似度匹配 ground truth 有噪声（相似的 chunk 可能有多个）。解法：切分时注入 UUID 作为 chunk_id 元数据，标注和评估都用 ID 精确匹配。

**坑3：BM25 需要 jieba 分词**  
中文没有空格，`rank_bm25` 默认按空格分词，所有中文 chunk 都是一个"词"，BM25 完全退化。解法：`preprocess_func=jieba.cut`，过滤单字，效果恢复正常。

---

## 十三、待改进的点（被追问"有什么不足"时）

优先级排序：
1. **BM25 持久化**：现在每次启动重建内存索引（6000 chunk ≈ 3s），迁移 Elasticsearch 后冷启动消失
2. **RAPTOR B1/B2 实验**：摘要节点已实现，还未跑对比评估
3. **Overlap 优化**：当前按字符数截断，理想情况应延伸到句子边界
4. **列表项前导句合并**："以下情况不适用：（1）..."这种结构，前导句应复制到每个列表段

---

## 十、代码题：排序链表（LeetCode 148）

归并排序，O(n log n) 时间，O(log n) 空间：

```python
def sortList(head):
    if not head or not head.next:
        return head
    # 快慢指针找中点
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    # 递归排序
    left = sortList(head)
    right = sortList(mid)
    # 合并
    dummy = cur = ListNode(0)
    while left and right:
        if left.val <= right.val:
            cur.next, left = left, left.next
        else:
            cur.next, right = right, right.next
        cur = cur.next
    cur.next = left or right
    return dummy.next
```
