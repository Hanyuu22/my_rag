# RAG Project — 版本变更记录

> 基于 MinerU + LangChain 1.0 的工业级 RAG 系统
> 环境：`mineru_2.5` | 路径：`~/rag_project/`

---

## V1.0 — 基础 RAG 系统

**发布日期**：2026-03-19
**对应阶段**：Level 0 ~ Level 3

### 核心目标
从零建立可运行的 RAG 链路：文档解析 → 切分 → 向量检索 → 生成，支持石化工艺手册类文档。

---

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 全局配置 | `config.py` | API Key、模型路径、检索参数集中管理 |
| MinerU 加载器 | `loaders/mineru_loader.py` | 读 `content_list.json`，每个 block → Document，携带 page/block_type |
| Excel 加载器 | `loaders/excel_loader.py` | xlsx 按行转 Document，支持 row_mode/sheet_mode |
| Word 加载器 | `loaders/word_loader.py` | `.doc/.docx` → Windows Word COM → PDF → MinerU（WSL 专用路径转换） |
| Markdown 切分器 | `splitters/markdown_splitter.py` | 两阶段：标题层级切分 + 字符数二次切（≤512），table/equation 整块保留 |
| RAG 链 | `chains/rag_chain.py` | BGE Embedding 单例 + Chroma 持久化 + LCEL 链（答案+来源同时返回） |
| Hybrid 检索器 | `retrievers/hybrid_retriever.py` | BM25（jieba 中文分词）+ Chroma Dense → EnsembleRetriever（RRF 权重融合） |
| BGE Reranker | `retrievers/reranker.py` | bge-reranker-v2-m3，Top-20 候选 → 精排 Top-3 |
| LangGraph 图 | `graphs/rag_graph.py` | 5节点：router / retrieve / evaluate / rewrite / generate / direct |
| VLM 表格 OCR | `tools/ocr_empty_tables.py` | qwen-vl-plus 补全 MinerU 未提取文字的空表格，输出 Markdown 格式 |

---

### 关键设计

**检索三板斧**
```
BM25（jieba 关键词）
        ↓
Dense（bge-large-zh-v1.5 语义）   → EnsembleRetriever（RRF 融合）→ BGE Reranker → Top-3
```

**LangGraph 流程**
```
问题 → [router]
         ├─ 闲聊/问候 → [direct] → 直接回答
         └─ 需要检索 → [retrieve] → [evaluate]
                                        ├─ top1_score ≥ 0.5 → [generate]
                                        └─ 不足 → [rewrite] → 重检索（最多2次）
```

---

### 指标结果（加氢裂化工艺规程，78页）

| 指标 | Level 1（纯 Dense） | Level 2（Hybrid+Reranker） | Level 3（+LangGraph+VLM OCR） |
|------|-------------------|--------------------------|------------------------------|
| Faithfulness | 0.743 | 0.780 | **0.858** |
| AnswerRelevancy | 0.807 | **0.912** | 0.911 |
| ContextPrecision | 0.122 | **0.500** | 0.500 |
| ContextRecall | 0.667 | 0.583 | **0.750** |
| Q3 表格 Top1 score | 0.244 | 0.244 | **0.849**（+0.605）|
| 平均 Top1 score | ~0.735 | ~0.735 | **0.952** |
| 路由准确率 | N/A | N/A | **100%**（5/5）|
| 平均全链路耗时 | ~15s | ~12s（稳态） | **9.18s** |
| 知识库 chunks | 814 | 814 | **1632**（VLM OCR 补充后）|

**关键发现**
- BM25 需要 jieba 分词，否则退化为纯 Dense（中文无空格，默认分词失效）
- 表格在 MinerU 中被识别为图片，text 字段为空；VLM OCR 后 Q3 从 0.244 → 0.849
- ContextPrecision：Reranker 将 Level 1 的 0.122 提升 4 倍至 0.500

---

## V2.0 — 工程化 + 评估体系

**发布日期**：2026-03-19 ~ 2026-03-20
**对应阶段**：Level 4 ~ Level 6 + 工程优化

### 核心目标
搭建前后端完整服务，引入 RAGAS 四指标自动评估，新增 Fallback 兜底路由，完成工程健壮性改造。

---

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| RAGAS 评估 | `evaluation/ragas_eval.py` | Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall |
| 自动出题 | `evaluation/generate_testset.py` | RAGAS TestsetGenerator，中文 persona（石化工艺工程师/操作员）|
| 批量评估 | `evaluation/run_generate.py` + `run_ragas_eval.py` | 断点续跑，分类汇总 |
| FastAPI 后端 | `backend/` | SSE 流式问答，upload / chat / collections 三个 router |
| React 前端 | `frontend/` | React 19 + Vite + TypeScript，三页面（问答/知识库/设置）|
| Fallback 节点 | `graphs/rag_graph.py` | Tavily 网络搜索（方案B）+ LLM 兜底（方案A），前端标注来源类型 |
| 预热机制 | `backend/main.py` | FastAPI lifespan + `run_in_executor`，后台异步预热，不阻塞服务就绪 |
| 专用入库脚本 | `data/preprocess_investment_db.py` | 93 列 xlsx 专用清洗：筛列 + 拆 chunk + 实体 header |

---

### 主要改进

**检索层**
- Router/Rewrite Prompt 从石化硬编码改为**通用领域感知**，支持多知识库切换
- Reranker 改为**全局单例**（`get_reranker()`），N 个知识库只加载一次（节省 ~5s + 显存）
- `build_vectorstore` 先用 chromadb 检查 collection 是否存在，再决定是否加载 Embedding

**前端功能**
- 流式问答（SSE），Markdown 全渲染（表格/代码/LaTeX 公式 KaTeX）
- 来源溯源（页码 + 置信分，可展开）
- 对话历史按知识库独立持久化（localStorage），切换不互染（修复竞态 Bug）
- Pipeline 控制面板：Reranker 开关 / Fallback 开关 / fallback 方法选择
- 文件上传（PDF/Word/Excel），拖拽 + 命名弹窗 + 进度条实时轮询

**工程健壮性**
- Fallback 标注：🌐 蓝色（网络搜索）/ ⚠️ 黄色（LLM 兜底）
- 冷启动从 ~45s 消除（lifespan warmup）
- Excel 入库 Bug 修复（`source_name` 多余参数）

---

### RAGAS 评估结果（10 道自动生成中文题）

| 指标 | 总体 | 单跳（6道）| 多跳抽象（2道）| 多跳精确（2道）|
|------|------|-----------|--------------|--------------|
| Faithfulness | 0.739 | 0.750 | 0.604 | **0.938** |
| AnswerRelevancy | 0.681 | 0.610 | 0.527 | **0.904** |
| ContextPrecision | **0.850** | **1.000** | 0.250 | **1.000** |
| ContextRecall | **0.875** | **1.000** | 0.625 | **1.000** |

**关键发现**
- 单跳题 Precision/Recall 均为 1.0，检索精准
- 多跳抽象型 ContextPrecision=0.25，跨章节抽象推理对 chunk 截断敏感（已知局限）
- AnswerRelevancy 总体 0.681 偏低，与 RAGAS 自动生成 reference 质量有关

---

### 知识库状态

| 库名 | Chunks | 来源 |
|------|--------|------|
| `hydro_manual` | 1,632 | 加氢裂化工艺规程（含 VLM OCR 表格）|
| `investment_db` | 7,440 | 投资机构&项目数据库 xlsx（专用清洗）|

---

## V3.0 — 高级检索能力 + 多知识库 + 工程加固

**发布日期**：2026-04-08
**对应阶段**：Level 7a / 7b / 7d + 工程优化2 + 工程优化3

### 核心目标
引入 Function Calling、MCP、跨语言检索、多知识库并行检索，完成 Redis 持久化改造，增强 chunk 元数据关联性。

---

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| Function Calling | `tools/` + `graphs/rag_graph.py` | `bind_tools` 工具路由，DCS 时序数据 Pandas 分析工具 |
| MCP Server | `mcp_server/server.py` | FastMCP，将 RAG 问答暴露为标准 MCP 工具接口 |
| 跨语言检索器 | `retrievers/cross_lingual_retriever.py` | 中英互译 + 双库并行检索，Redis 翻译缓存（TTL 7天）|
| 多库检索器 | `retrievers/multi_collection_retriever.py` | 并行跨 collection 检索，返回 `(retriever, all_chunks)` |
| Redis 客户端 | `backend/redis_client.py` | Redis 单例 + 任务状态封装（task_set/update/get，TTL 24h）|
| chunk 元数据增强 | `splitters/markdown_splitter.py` | `_enrich_metadata()`：chunk_id / section_path / prev_next_id |
| 邻居上下文扩展 | `graphs/rag_graph.py` | Reranker 后自动拉取 prev/next chunk，补充跨 chunk 推理上下文 |
| 批量入库脚本 | `scripts/batch_ingest_raw.py` | 扫描目录批量入库，断点续传，`--prefix` 文件名过滤 |
| 测试集生成 | `scripts/generate_qa_v2.py` | 动态出题：按文档长度分配 5~15 题，目标 2000 条，topic_tag 自动标注 |
| 分类评估 | `scripts/run_eval_full.py` | RAGAS 按题型/难度/主题分组输出 |

---

### 主要改进

**检索能力**
- **per-collection Embedding 注册表**：`data/chroma_db/embedding_registry.json`，不同库可用不同 Embedding 模型（中文用 bge-large-zh-v1.5，混合语言用 bge-m3）
- **跨语言检索**：用户中文提问自动翻译并在英文库检索，结果 RRF 融合；翻译结果 Redis 缓存复用
- **多库并行**：前端勾选附加知识库，后端并行检索多个 collection，chunk_lookup 传入 graph 支持邻居扩展
- **chunk 双向链**：每个 chunk 带 `prev_chunk_id` / `next_chunk_id`，Reranker 后自动拉取相邻 chunk 补充上下文

**工程加固**
- **Redis 持久化**：任务状态从内存 dict 迁移到 Redis Hash，服务重启不丢失入库进度
- **LRU graph 缓存**：`_graph_cache` 改为 OrderedDict（上限 10），修复 tuple key bug
- **文件校验**：上传文件大小上限 300MB + 空文件校验
- **LLM 流式透传**：节点通过 `state.get("token_callback")` 判断是否流式，有则 `llm.stream()`
- **聚合查询检测**：识别跨文档统计类问题，自动路由到 direct 节点
- **start.sh**：自动启动 Redis（WSL 不自动启动服务）

**前端**
- `CollectionList`：每行 checkbox 勾选附加检索库
- `ChatPage`：`extraCollections` 状态提升，多库勾选持久化
- `ChatWindow`：`extra_collections` 传入请求体，`crossLingualEnabled` 持久化到 localStorage

---

### 知识库状态（2026-04-08）

| 库名 | Chunks | 来源 | Embedding |
|------|--------|------|-----------|
| `hydro_manual` | 1,632 | 加氢裂化工艺规程 | bge-large-zh-v1.5 |
| `investment_db` | 7,440 | 投资机构&项目数据库 xlsx | bge-large-zh-v1.5 |
| `energy_zh` | 12,508 | CNPC/NEA 等中文能源报告（12份）| bge-m3 |
| `energy_en` | 5,961 | BP/EIA/Shell 等英文能源报告（17份）| bge-m3 |
| `rag_docs` | 916 | 其他文档 | bge-large-zh-v1.5 |
| `gb_standards` | 433+（入库中）| 200+ GB 国标石油化工 PDF | bge-large-zh-v1.5 |

---

### 架构演进对比

| 维度 | V1 | V2 | V3 |
|------|----|----|-----|
| 检索策略 | BM25+Dense+RRF+Reranker | 同左，通用化 | +跨语言+多库并行+邻居扩展 |
| 知识库数量 | 1 | 2 | **6**（含入库中）|
| 工具能力 | 无 | Fallback（Tavily/LLM）| +Function Calling + MCP |
| 持久化 | Chroma（向量）| Chroma + localStorage | +Redis（任务状态+翻译缓存）|
| chunk 元数据 | page/block_type | 同左 | +chunk_id/section_path/prev_next |
| 入库方式 | 手动单文件 | 手动+前端上传 | +批量断点续传脚本 |
| LLM 透传 | 批量 | 批量 | **流式 SSE 透传** |
| 评估规模 | 5道主观题 | 10道 RAGAS 自动题 | 规划 2000 条分类测试集 |

---

## 下一步计划（V4.0 候选）

| 功能 | 说明 | 优先级 |
|------|------|--------|
| GB 标准入库完成 | 200+ PDF 跑完，`gb_standards` 预计 8k~15k chunks | 🔄 进行中 |
| 2000条测试集 | `generate_qa_v2.py` 生成，`run_eval_full.py` 分类评估 | 🔲 |
| Level 7c — GraphRAG | 知识图谱辅助检索，实体关系抽取 | 🔲 |
| Level 7c — RAPTOR | 多层摘要树，增强长文档多跳理解 | 🔲 |
| Elasticsearch 集成 | BM25 持久化 + IK 分词，替换 rank_bm25（待 GB 入库完评估必要性）| 🔲 |
