# RAG Project 进度

> 基于 MinerU + LangChain 1.0 的工业级 RAG 系统
> 环境：`mineru_2.5` conda env | 路径：`~/rag_project/`

---

## 一、当前状态

| 阶段 | 模块 | 文件 | 状态 |
|------|------|------|------|
| Level 0 | 全局配置 | `config.py` | ✅ |
| Level 1 | MinerU 文档加载 | `loaders/mineru_loader.py` | ✅ |
| Level 1 | 文本切分 | `splitters/markdown_splitter.py` | ✅ |
| Level 1 | Excel 加载 | `loaders/excel_loader.py` | ✅ |
| Level 1 | Word 文档加载 | `loaders/word_loader.py` | ✅ |
| Level 1 | Embedding + Chroma + LCEL 链 | `chains/rag_chain.py` | ✅ |
| Level 2 | Hybrid Retriever（BM25 + Dense + RRF） | `retrievers/hybrid_retriever.py` | ✅ |
| Level 2 | BGE Reranker 精排 | `retrievers/reranker.py` | ✅ |
| Level 3 | LangGraph 多步 RAG | `graphs/rag_graph.py` | ✅ |
| Level 4 | RAGAS 评估 | `evaluation/ragas_eval.py` | ✅ |
| Level 5 | 前后端（FastAPI + React） | `backend/` + `frontend/` | ✅ |
| Level 6 | RAGAS 大规模测试 + Fallback 路由 + Pipeline 控制 | `evaluation/` + `graphs/` + `frontend/` | ✅ |
| 工程优化 | 多知识库通用化 + Reranker 单例 + 非阻塞预热 + 一键启动 | 多文件 | ✅ |
| Level 7a | Function Calling（bind_tools 工具路由 + DCS 数据分析） | `tools/` + `graphs/rag_graph.py` | ✅ |
| Level 7b | MCP Server（FastMCP，暴露 RAG 为标准工具接口） | `mcp_server/server.py` | ✅ |

| Level 7d | 多语言检索实验 + CrossLingualRetriever + CrossCollectionRetriever 分库路由 | `retrievers/cross_lingual_retriever.py` + `tests/` | ✅ |
| 工程优化2 | per-collection Embedding 注册表 + 多文件上传 + LLM 流式透传 + 聚合查询检测 | `chains/rag_chain.py` + `graphs/rag_graph.py` + `backend/routers/` + `frontend/` | ✅ |

**下一步**：Level 7c — ① GraphRAG（知识图谱辅助检索）；② RAPTOR（多层摘要树，增强长文档理解）

---

## 二、项目结构

```
~/rag_project/
├── PROGRESS.md                        ← 本文件
├── RUN_LOG.md                         ← 运行问题记录
├── config.py                          ✅ 全局配置
├── loaders/
│   ├── mineru_loader.py               ✅ MinerU content_list 加载器
│   ├── excel_loader.py                ✅ xlsx 按行加载器
│   └── word_loader.py                 ✅ .doc/.docx → PDF → MinerU 加载器
├── splitters/
│   └── markdown_splitter.py           ✅ 两阶段切分（标题 + 字符数）
├── chains/
│   └── rag_chain.py                   ✅ Embedding 单例 + Chroma 建库 + LCEL RAG 链
├── retrievers/
│   ├── hybrid_retriever.py            ✅ BM25 + Dense + RRF 融合
│   └── reranker.py                    ✅ BGE Reranker 精排（bge-reranker-v2-m3）
├── start.sh                           ✅ 一键启动脚本（自动检查端口/激活环境/前后端联动）
├── graphs/
│   └── rag_graph.py                   ✅ 6节点图（+fallback节点，通用化 Prompt 支持多知识库）
├── evaluation/
│   ├── ragas_eval.py                  ✅ RAGAS 四指标评估（Level 4）
│   ├── generate_testset.py            ✅ RAGAS TestsetGenerator 自动出题
│   ├── run_generate.py                ✅ 批量跑 RAG 获取草稿答案
│   ├── run_ragas_eval.py              ✅ 批量 RAGAS 评分 + 分类汇总
│   ├── golden_set_auto.json           ✅ 10道中文自动生成测试题
│   ├── draft_auto.json                ✅ 10道题的 RAG 草稿答案
│   └── results/
│       ├── eval_20260319_223534.json  ✅ 完整评估报告
│       └── latest.json               ✅ 最新评估结果（软链接）
├── tests/
│   ├── test_level1.py                 ✅ Level 1 集成测试
│   ├── test_level2.py                 ✅ Level 2 集成测试（带指标）
│   ├── test_level3.py                 ✅ Level 3 集成测试
│   └── check_embedding.py             ✅ Embedding 环境验证
└── data/
    ├── raw/                           原始文档（.doc/.pdf/.xlsx）
    ├── mineru_output/                 MinerU 解析结果（content_list.json 等）
    ├── preprocess_investment_db.py    ✅ 投资数据库 xlsx 专用清洗入库脚本
    └── chroma_db/                     Chroma 向量库持久化
        ├── hydro_manual/              加氢裂化工艺规程（~1632 chunks）
        └── investment_db/             投资机构&项目数据库（7440 chunks）
```

---

## 三、系统设计

### 3.1 完整数据链路

```
原始文档（PDF / Word / xlsx）
   │
   ▼
[文档加载器]
   ├─ MinerULoader    → content_list.json → List[Document]（带 page / block_type）
   ├─ WordLoader      → Word COM → PDF → MinerU → List[Document]
   └─ ExcelLoader     → 每行一个 Document
   │
   ▼
[MarkdownSplitter]
   ├─ 按 Markdown 标题层级切（携带章节 metadata）
   ├─ 超长段落按字符数二次切（≤512）
   └─ table / equation 整块保留，不切分
   │
   ├──────────────────────────────┐
   ▼                              ▼
[BGE Embedding]              [BM25 索引]
[Chroma 向量库]               rank_bm25
   │                              │
   └──────── 查询时 RRF 融合 ─────┘
                   │
                   ▼
          [BGE Reranker]  精排 → Top-3
                   │
                   ▼
          [LCEL Chain]  Prompt → LLM → Answer（带页码溯源）
```

LangGraph 在链路之上加流程调度：
```
问题 → [路由] → 需要检索？
                  ├─ 否 → 直接回答
                  └─ 是 → 检索 → [评估] 结果充分？
                                    ├─ 是 → 生成答案
                                    └─ 否 → 改写查询，重检索（最多 2 次）
                                              └─ 耗尽后 → [Fallback]
                                                            ├─ 方案B: Tavily 网络搜索（标注来源链接）
                                                            └─ 方案A: LLM 自身知识（标注免责声明）
```

### 3.2 模型分工

| 角色 | 模型 | 运行方式 |
|------|------|----------|
| PDF 解析 | MinerU pipeline | 本地，一次性离线 |
| Embedding | bge-large-zh-v1.5 | 本地 GPU（CUDA） |
| Reranker | bge-reranker-v2-m3 | 本地 GPU |
| 生成 LLM | qwen-plus | DashScope API |
| 评估 LLM | qwen-plus | DashScope API（复用 key） |
| 网络搜索 | Tavily API | 云端 API（Fallback 兜底路由，方案B） |

> 生成层用 qwen-plus 而非 claude：RAG 的 token 消耗大头在 context，qwen-plus 更便宜且质量够用。需要更高质量时改 `config.py` 的 `LLM_MODEL` 一行即可。

### 3.3 多知识库设计

通过 Chroma 的 `collection_name` 区分不同来源：

| 数据类型 | Loader | 切分策略 | collection 示例 |
|---------|--------|---------|----------------|
| PDF / Word | MinerULoader / WordLoader | 标题 + 字符数 | `tech_docs` |
| xlsx（结构化表格）| ExcelLoader | 按行，不二次切 | `business_data` |

> xlsx 的 DCS 时序数据（数值型）不适合直接进向量库，留给 Level 3 的 Tool Agent 用 Pandas 分析。

### 3.4 工程设计原则

- **关注点分离**：Loader / Splitter / Retriever 各做一件事，独立可测可替换
- **结构优先于文本**：用 `content_list.json` 而非裸 markdown，保留 block_type + page_idx
- **检索增强三板斧**：BM25（关键词）+ Dense（语义）→ RRF → Reranker，覆盖两类 Query
- **评估驱动迭代**：每级改动后跑 RAGAS，用 Faithfulness / Answer Relevancy / Context Precision / Context Recall 量化效果

---

## 四、各级进度详情

### ✅ Level 0 — 全局配置
**文件**：`config.py`
API Key、模型路径、Chroma 目录、检索参数（Top-K、chunk_size 等）集中管理。

---

### ✅ Level 1 — 基础 RAG 链

**Step 1 — 文档加载器**
- `MinerULoader`：读 `_content_list.json`，每个 block → Document，metadata 含 source / page / block_type
- `ExcelLoader`：xlsx 按行转 Document，支持 row_mode / sheet_mode
- `WordLoader`：.doc/.docx → Windows Word COM → PDF → MinerU → Documents（WSL 专用）

**Step 2 — 文本切分器**
- 两阶段：MarkdownHeaderTextSplitter（标题层级）→ RecursiveCharacterTextSplitter（≤512 字符）
- table / equation 整块保留

**Step 3 — Embedding + 向量库 + LCEL 链**
- BGE bge-large-zh-v1.5 本地加载，单例复用（避免重复加载）
- Chroma 持久化到 `data/chroma_db/`，支持多 collection
- LCEL 链：`RunnableParallel(answer=chain, source_docs=retriever)`，答案 + 来源同时返回

**测试结果**（加氢裂化工艺规程，78页）：
- MinerU 解析：381 公式 + 22 表格，约 2 分钟
- 问答质量主观评估：反应条件、开工准备 ✅ 优秀；操作参数范围 ⚠️ 良好（诚实说明表格数据未检索到，无幻觉）

---

### ✅ Level 2 — 检索增强

**目标**：用 Hybrid Retriever + Reranker 替换 Level 1 的纯 Dense 检索，提升召回率和精度。

- `retrievers/hybrid_retriever.py`：BM25（jieba 中文分词）+ Chroma Dense → EnsembleRetriever（RRF 权重融合）
- `retrievers/reranker.py`：FlagEmbedding BGE Reranker（bge-reranker-v2-m3），输入 Top-20，输出精排 Top-3

**测试结果**（同一份加氢裂化工艺规程，78页，814 chunks）：

| 问题 | Top1 score | 评估 |
|------|-----------|------|
| 加氢裂化装置的主要反应条件 | **0.995** | ✅ 极高，3条来源全部高度相关（0.995/0.987/0.986） |
| 装置开工前需要哪些准备工作 | **0.965** | ✅ 高，前两条来源精准（0.965/0.964），第3条一般（0.617） |
| 反应器的操作温度和压力范围 | **0.244** | ⚠️ 低，原文表格（表9-1～9-4）数据未被提取为可检索文本 |

**平均性能（稳态，第2次运行）**：
- 平均全链路：~12s（检索 2.2s + 精排 1~2s + 生成 9s）
- 首次运行因 CUDA kernel 编译，第1个问题精排耗时约 74~105s，属正常冷启动

**结论**：Q1/Q2 质量已达上限；Q3 低分是源文档表格解析局限，不是检索问题。

**代码优化**：`build_vectorstore` 改为先用 chromadb 检查向量库是否存在，再决定是否加载 Embedding，避免无意义等待。

---

### ✅ Level 3 — LangGraph 多步 RAG

- `graphs/rag_graph.py`：5节点图（router / retrieve / rewrite / generate / direct）
- **router**：LLM 分类，判断是否需要检索知识库
- **retrieve**：调用 Level 2 的 Hybrid + Reranker
- **evaluate**（条件边）：top1_score ≥ 0.5 → generate；否则 → rewrite（最多 2 次）
- **rewrite**：石化领域感知的 LLM 查询改写，换专业术语角度重检索
- **generate / direct**：带溯源生成 或 直接回答
- `tools/ocr_empty_tables.py`：qwen-vl-plus 补充 MinerU 未提取文字的表格（19/19 成功，结果缓存）
- `loaders/word_loader.py`：集成 VLM OCR，`use_vlm_ocr=True` 触发，结果自动缓存复用

**测试结果**（加氢裂化工艺规程，1632 chunks（含 VLM 表格 OCR 补充））：

| 问题 | Top1 score | 改写次数 | 评估 |
|------|-----------|---------|------|
| 加氢裂化装置的主要反应条件 | **0.995** | 0 | ✅ 无需改写，直接命中 |
| 装置开工前需要哪些准备工作 | **0.965** | 0 | ✅ 无需改写，直接命中 |
| 反应器的操作温度和压力范围 | **0.849** | 0 | ✅ 表格 OCR 后命中表9-2/9-4，从 0.244 大幅提升 |
| 你好，帮我介绍一下你自己 | N/A | 0 | ✅ 正确路由到 direct，不检索知识库 |
| 冷氢的作用是什么 | **0.997** | 1 | ✅ 改写后命中，改写 query 更聚焦石化术语 |

**整体指标**：路由准确率 100%（5/5）｜ 平均 Top1 置信分 0.952 ｜ 平均全链路 9.18s

---

### ✅ Level 4 — RAGAS 评估

四个指标：Faithfulness / Answer Relevancy / ContextPrecision / ContextRecall

**评估结果**（加氢裂化工艺规程，4道问题，818 chunks）：

| 指标 | Level 1 | Level 2 | Level 3 | 趋势 |
|------|---------|---------|---------|------|
| Faithfulness | 0.743 | 0.780 | **0.858** | ↑ 表格 OCR 让答案更有据可查 |
| AnswerRelevancy | 0.807 | **0.912** | 0.911 | ↑ Hybrid 检索大幅提升切题度 |
| ContextPrecision | 0.122 | **0.500** | 0.500 | ↑↑↑ Reranker 过滤噪音文档 |
| ContextRecall | 0.667 | 0.583 | **0.750** | L2 略降（Top-3 截断），L3 靠表格 OCR 回升 |

**结论**：每一级改动都有明确数字支撑，均值综合提升约 15~30%。

---

### ✅ Level 5 — 前后端（FastAPI + React）

**后端**（`backend/`，FastAPI + uvicorn）：
- `backend/main.py`：FastAPI 入口，CORS、路由挂载，支持 `--reload` 热重载
- `backend/routers/upload.py`：多格式上传（PDF / Word / Excel），异步任务队列，进度轮询 `/api/tasks/{id}`
- `backend/routers/chat.py`：SSE 流式问答，接收前端动态参数（Top-K / Reranker Top-N / Temperature / score_threshold / max_retry），按参数组合缓存 RAG graph
- `backend/routers/collections.py`：知识库列表查询、删除

**前端**（`frontend/`，React 19 + Vite + TypeScript）：
- 三页面导航（左侧 72px icon 栏）：问答 / 知识库 / 设置
- **问答页**：左侧知识库选择（230px）+ 右侧聊天（最大宽 820px，避免过宽）；AI 回复完整 Markdown 渲染（react-markdown + remark-gfm + KaTeX，含表格/代码块/加粗/列表/LaTeX 公式）；对话历史按知识库独立存入 localStorage，切换知识库自动加载对应历史，支持单条删除和一键清空
- **知识库页**：拖拽上传（进度条实时轮询）+ 知识库列表管理
- **设置页**：双列全页面，检索/生成参数滑块，头像选择器（支持自定义图片）；所有设置 localStorage 持久化，刷新不丢失
- 主题：清新浅绿白，SVG 叶片花纹背景，白色磨砂玻璃卡片
- 自定义头像：羽入（Hanyuu from Higurashi）作为 Bot 默认头像；右代宫战人作为 User 默认头像

**启动方式**：
```bash
# 一键启动（推荐）
bash ~/rag_project/start.sh

# 手动启动
conda activate mineru_2.5 && cd ~/rag_project
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
cd ~/rag_project/frontend && npm run dev   # http://localhost:3000
```

---

### ✅ Level 6 — 大规模测试 + Fallback 路由 + Pipeline 控制

**Step 1 — RAGAS TestsetGenerator 自动出题**
- `evaluation/generate_testset.py`：调用 RAGAS `TestsetGenerator` 从文档 chunks 自动生成测试题
  - KG 构建：SummaryExtractor → NERExtractor → EmbeddingExtractor → ThemesExtractor
  - 查询分布：60% SingleHopSpecific + 20% MultiHopAbstract + 20% MultiHopSpecific
  - 配置中文 persona（石化工艺工程师、装置操作员）+ `llm_context` 引导中文出题
- 输出 10 道中文题目，手动翻译修正后保存为 `evaluation/golden_set_auto.json`

**Step 2 — 批量评估 Pipeline**
- `evaluation/run_generate.py`：按问题列表跑 RAG graph，收集答案 + 检索文档 + 分数 + 路由，输出 `draft_auto.json`；支持断点续跑（跳过已答）
- `evaluation/run_ragas_eval.py`：加载草稿，按有/无 reference 分批评估，输出 per-question 分数 + 分类汇总到 `evaluation/results/`

**RAGAS 评估结果**（10道题，2026-03-19）：

| 指标 | 总体 | 单跳 | 多跳抽象 | 多跳精确 |
|------|------|------|---------|---------|
| Faithfulness | 0.739 | 0.750 | 0.604 | **0.938** |
| AnswerRelevancy | 0.681 | 0.610 | 0.527 | **0.904** |
| ContextPrecision | **0.850** | **1.000** | 0.250 | **1.000** |
| ContextRecall | **0.875** | **1.000** | 0.625 | **1.000** |

> 单跳精确型 context 指标完美（1.0），多跳抽象型 ContextPrecision 偏低（0.25）为预期内，跨段推理对 Top-K 截断敏感。

**Step 3 — Fallback 路由**
- `graphs/rag_graph.py`：
  - `RAGState` 新增字段：`fallback_type / web_sources / reranker_enabled / fallback_enabled / fallback_method`
  - `evaluate_decision`：重试耗尽后检查 `fallback_enabled`，返回 `"fallback"` 而非强制 `"generate"`
  - `retrieve_node`：`reranker_enabled=False` 时跳过精排，scores 统一设为 1.0
  - `fallback_node`：优先走 Tavily 网络搜索（方案B），失败或 method=llm 时降级到 LLM 自身知识（方案A）
- `config.py`：新增 `TAVILY_API_KEY`
- `backend/routers/chat.py`：
  - `ChatRequest` 新增 `reranker_enabled / fallback_enabled / fallback_method`
  - SSE 新增 `{"type": "fallback", "content": {"type": "llm"|"web", "web_sources": [...]}}` 事件
- `frontend/src/components/ChatWindow.tsx`：
  - 接收 fallback SSE 事件，渲染来源标注气泡
  - `web` fallback：蓝色 🌐 + 可点击来源链接
  - `llm` fallback：黄色 ⚠️ + 免责文字（"以下内容来自 LLM 通用知识，非本知识库"）

**Step 4 — Pipeline 控制面板**
- `frontend/src/hooks/useSettings.ts`：新增 `rerankerEnabled / fallbackEnabled / fallbackMethod` 字段，localStorage 持久化
- `frontend/src/pages/SettingsPage.tsx`：
  - 新增"流程控制"卡片（位于参数滑块上方）
  - `PipelineToggle` 组件：自定义开关，label + hint + enabled + comingSoon 占位
  - 可用开关：BGE Reranker 精排 / Fallback 兜底路由（含方法选择器：自动/LLM知识/网络搜索）
  - 占位（开发中）：知识图谱检索 / RAPTOR 递归摘要 / 多路召回融合

**Step 5 — 冷启动消除**
- `backend/main.py`：FastAPI `lifespan` 中异步启动 `_warmup()` 后台任务
  - 枚举所有 Chroma collection，预建 RAG graph 并存入 `_graph_cache`
  - 不阻塞服务就绪，第一个请求到达时模型已加载完毕
  - 冷启动延迟从 ~45s（Embedding 31s + Jieba 4s + Reranker 10s）消除

---

### ✅ 工程优化（2026-03-20）

**Bug 修复**
- `backend/services/pipeline.py`：删除 `ExcelLoader(source_name=...)` 多余参数（会导致 TypeError，xlsx 上传必现）

**多知识库通用化**
- `graphs/rag_graph.py`：Router/Rewrite/Direct Prompt 全部去除石化领域硬编码，改为通用逻辑
  - Router：遇到模糊情况优先 retrieve，只有纯闲聊/问候才 direct
  - Rewrite：通用策略（提取实体、具体化、同义替换），适配任意领域知识库

**Reranker 单例化**
- `retrievers/reranker.py`：新增 `get_reranker()` 全局单例工厂函数
- `backend/routers/chat.py`：改用 `get_reranker()` 替代每次 `BGEReranker()`
- 效果：多个知识库共享同一个 Reranker 实例，预热时从加载 3 次缩减为 1 次

**非阻塞预热**
- `backend/main.py`：`_warmup()` 中的 `_build_graph` 改用 `loop.run_in_executor()` 在线程池执行
- 效果：服务启动后立即响应 `/api/collections` 等请求，前端可进入页面，预热在后台静默完成

**投资数据库专用入库**
- `data/preprocess_investment_db.py`：针对 93 列投资 xlsx 的专用清洗脚本
  - Sheet1（投资方 1815 行）+ Sheet2（项目 1386 行）→ 7440 个 chunk
  - 每行拆成结构化 chunk（身份+关键字段）+ 长文本 chunk（简介/纪要，≤600 字）
  - 每个 chunk 开头带实体 header，确保检索到任意片段都知道来源
  - 跳过隐私字段（手机/微信/邮箱）和低价值字段（时间戳/状态标记）
  - 平均 chunk 长度 196 字，最长 639 字，全部在 BGE 512 token 有效范围内

**一键启动脚本**
- `start.sh`：自动检查端口占用 → 激活 conda → 后台启动后端 → 等待就绪 → 前台启动前端
  - Ctrl+C 联动停止前后端
  - 后端日志写入 `/tmp/rag_backend.log`

---

### ✅ Level 7a — Function Calling（2026-03-26）

**新增文件**：

| 文件 | 说明 |
|------|------|
| `tools/tool_definitions.py` | 3 个 LangChain StructuredTool：search_kb / analyze_process_data / web_search |
| `tools/data_analyzer.py` | DCS 时序数据 Pandas 分析（9985行×34列，5分钟采样） |

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphs/rag_graph.py` | RAGState 新增 tools_enabled/tool_name/tool_args/tool_result；router 双模式；新增 tool_executor_node 和 "tool" 路由分支 |
| `backend/routers/chat.py` | ChatRequest 新增 tools_enabled；SSE 新增 tool_call 事件 |

**核心设计**：

```
tools_enabled=False（默认）：原有字符串路由，向后兼容
tools_enabled=True：
  问题 → llm.bind_tools([search_kb, analyze_data, web_search, direct_answer])
              ├─ search_knowledge_base → retrieve_node（复用 Hybrid+Reranker）
              ├─ analyze_process_data → tool_executor_node（Pandas 分析 DCS）
              ├─ web_search           → tool_executor_node（Tavily 主动调用）
              └─ direct_answer        → direct_node
```

**data_analyzer 能力**：
- 数据源：`加氢裂化装置数据集-汇总20230308.xlsx`（2023-01-30 ~ 2023-03-05）
- 支持：温度/压力/流量的均值、极值、时段筛选、趋势判断
- 实现：LLM 生成 pandas 代码 → exec → 失败降级 LLM 描述

---

### ✅ Level 7b — MCP Server（2026-03-26）

**新增文件**：`mcp_server/server.py`（基于 FastMCP，mcp==1.26.0）

**暴露的工具**：

| 工具 | 功能 |
|------|------|
| `list_collections` | 列出所有知识库名称及文档片段数 |
| `search_knowledge_base` | 向量检索指定知识库，返回 Top-K 片段 |
| `analyze_process_data` | Pandas 分析 DCS 时序数据 |

**启动方式**：
```bash
conda activate mineru_2.5
python ~/rag_project/mcp_server/server.py   # stdio 模式（Claude Desktop）
```

**Claude Desktop 配置**（`~/.config/claude/claude_desktop_config.json`）：
```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "/path/to/conda/envs/mineru_2.5/bin/python",
      "args": ["/home/hanyuu/rag_project/mcp_server/server.py"]
    }
  }
}
```

---

### ✅ Level 7d — 多语言检索实验 + CrossLingualRetriever（2026-03-30）

**背景**：为后续引入英文文档做技术选型，系统评估两个模型和三种策略。

**实验数据（合成）**：6条中文 + 6条英文加氢裂化工艺文档片段，8条跨语言查询。

**三方案对比结果**：

| 场景 | 方案1 bge-zh 单语 | 方案2 bge-m3 单语 | 方案3 m3+双语 query |
|------|-----------------|-----------------|---------------------|
| A: ZH→ZH | 2/2 | 2/2 | 2/2 |
| B: EN→EN | 2/2 | 2/2 | 2/2 |
| C: EN→ZH | 0/2 ❌ | 0/2 ❌ | **1/2 ✅** |
| D: ZH→EN | 0/2 ❌ | 0/2 ❌ | 0/2 ❌ |
| **总命中率** | 50% | 50% | **62.5%** |

**核心结论**：
1. **换 bge-m3** 不能单独解决跨语言命中，但跨语言文档的 gt_score 显著提升（C场景: 0.40→0.57）
2. **双语 query 策略**（CrossLingualRetriever）对 EN→ZH 方向有效（+12pp）
3. **ZH→EN 方向**仍需分库路由解决——中文 query 在混合库中语言偏置更严重

**新增文件**：
- `retrievers/cross_lingual_retriever.py`：语言检测 + LLM 翻译 + 双路 RRF 融合
- `tests/test_multilingual_embedding.py`：bge-zh vs bge-m3 基础对比
- `tests/test_cross_lingual.py`：三方案完整对比实验

**接入方式**：`ChatRequest.cross_lingual_enabled=True` 启用，`retrieve_node` 自动使用 `CrossLingualRetriever`。

**四方案完整对比**（含方案4分库路由，测试文件：`tests/test_cross_collection.py`，2026-03-30）：

| 场景 | 方案3 m3+双语 top1 | 方案4 分库路由 top1 | 方案4 top3 | 方案4 top5 | partner_rank |
|------|-----------------|-------------------|-----------|-----------|-------------|
| ZH→ZH | 2/2 | 2/2 | 2/2 | 2/2 | N/A |
| EN→EN | 2/2 | 2/2 | 2/2 | 2/2 | N/A |
| EN→ZH | 1/2 | 0/2 | **2/2** ✅ | 2/2 | **1.0** |
| ZH→EN | 0/2 | 0/2 | **2/2** ✅ | 2/2 | **1.0** |
| **总体** | 62.5% | 50% | **100%** | 100% | - |

**方案4 top-1 为何偏低**：小规模合成数据导致 RRF tie（primary/partner 各自 top-1 分相同=0.01639），GT 总排第2。生产环境大量文档不会出现完美 tie，partner_rank=1.0 已证明翻译 query 精准命中。

**核心结论**（方案4 vs 方案3）：
- 方案3 对 ZH→EN 的 top-3 覆盖仅 1/2；方案4 达到 **2/2**，根本修复
- partner_rank=1.0 证明分库策略完全消除了语言偏置：翻译 query 在配对库里精准命中
- 实际生产中 CrossCollectionRetriever → top-20 → BGE Reranker，GT 必然被选出

**迁移路径（建议）**：
```
当前：bge-large-zh-v1.5（纯中文）
↓ 加入英文文档时：换 config.py EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
↓ 重新入库（bge-m3 已下载）
↓ 建配对 collection（命名约定：foo ↔ foo_en）
↓ 启用 cross_lingual_enabled=True → 自动走 CrossCollectionRetriever
↓ 找不到配对库自动降级为 CrossLingualRetriever（双语 query 单库）
```

---

### 🔲 Level 7c — 高级检索增强（待开发）

**目标一：GraphRAG（知识图谱检索）**
- 从文档中抽取实体关系，构建图结构，检索时利用实体关联扩展召回
- 对多跳推理类问题（如"催化剂活性下降时应如何调整操作参数"）效果显著
- RAGAS 多跳抽象型 ContextPrecision 当前 0.25，GraphRAG 目标提升至 0.6+

**目标二：RAPTOR（递归摘要树）**
- 对长文档分层聚类 + 摘要，形成多粒度索引树
- 全局性问题（如"整个开工方案的核心步骤"）可在摘要层命中，减少对 chunk 级精确匹配的依赖

---

## 五、技术参考

### 5.1 环境依赖

| 包 | 版本 |
|----|------|
| langchain | 1.2.10 |
| langchain-community | 0.4.1 |
| langchain-chroma | 1.1.0 |
| langchain-huggingface | 1.2.1 |
| langchain-openai | 1.1.11 |
| langchain-text-splitters | 1.1.1 |
| chromadb | 1.5.5 |
| rank-bm25 | 0.2.2 |
| sentence-transformers | 5.3.0 |
| FlagEmbedding | 1.3.5 |
| ragas | 0.4.3 |
| mineru | 2.6.4 |

### 5.2 LCEL 关键模式

```python
# 带溯源的基础链
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    ),
    source_docs=retriever,
)
result = rag_chain.invoke("问题")
# result["answer"] / result["source_docs"]

# 插入自定义步骤
from langchain_core.runnables import RunnableLambda
rerank_step = RunnableLambda(lambda docs: bge_rerank(docs))
full_chain = ensemble_retriever | rerank_step | format_docs | prompt | llm | StrOutputParser()
```

### 5.3 解析器选型

| 维度 | MinerU 2.6 | PaddleOCR-VL | DeepSeek-OCR |
|------|-----------|--------------|--------------|
| 表格处理 | ✅ Markdown，结构完整 | ⚠️ 丢失行列关系 | ✅ 能理解但不稳定 |
| 公式处理 | ✅ LaTeX | ❌ 当普通文字 | ⚠️ 部分支持 |
| 输出格式 | ✅ content_list.json（带元数据） | ❌ 纯文字 | ❌ 纯文字 |
| RAG 友好度 | ✅ 天然适合 | ⚠️ 需后处理 | ⚠️ 需后处理 |

### 5.4 与 RAGFlow 的关系

| | 本项目 | RAGFlow |
|---|---|---|
| 目标 | 理解每个模块的工程决策 | 开箱即用的生产服务 |
| 价值 | 面试能讲清楚链路 | 快速部署验证 |

---

## 六、知识库现状（2026-04-16）

| 库名 | Chunks | 来源 | Embedding 模型 | 说明 |
|------|--------|------|----------------|------|
| `hydro_manual` | 1,632 | 加氢裂化工艺规程（78页） | bge-large-zh-v1.5 | 含 VLM OCR 补表，论文核心实验数据集 |
| `investment_db` | 7,440 | 投资机构&项目 xlsx | bge-large-zh-v1.5 | 专用清洗脚本，93列→结构化 chunks |
| `energy_zh` | 12,508 | CNPC/NEA 等中文能源报告（12份） | bge-m3 | 跨语言检索配对库 |
| `energy_en` | 5,961 | BP/EIA/Shell 等英文能源报告（17份） | bge-m3 | 与 energy_zh 配对 |
| `rag_docs` | 916 | 其他文档 | bge-large-zh-v1.5 | — |
| `gb_standards_512` | ~6,000 | GB 国标 41份（实验用） | bge-large-zh-v1.5 | chunk_size=512/overlap=64，V4 A1 组 |
| `gb_standards_1024` | ~3,200 | GB 国标 41份（实验用） | bge-large-zh-v1.5 | chunk_size=1024/overlap=100，V4 A2 组 |
| `gb_standards` | 433+（入库中） | GB 国标 200+ PDF | bge-large-zh-v1.5 | 完整库，入库完成后作为 V4 主实验库 |

---

## 七、V4 实验计划（GB 入库完成后执行）

### 7.1 前置条件

- [ ] `gb_standards` 完整入库（目标 8k~15k chunks）
- [ ] 2000 条分类测试集生成 + ground truth 标注
- [ ] RAPTOR 实现（`splitters/raptor_builder.py`，~150 行）

### 7.2 实验分组（8组对比）

| 组别 | chunk_size | overlap | RAPTOR | ES | 说明 |
|------|-----------|---------|--------|----|------|
| A1 | 512 | 64 | ❌ | ❌ | Baseline，`gb_standards_512` 已有数据 |
| A2 | 1024 | 100 | ❌ | ❌ | 需新建 `gb_standards_1024` collection |
| B1 | 512 | 64 | ✅ | ❌ | 在 A1 基础上追加摘要节点 |
| B2 | 1024 | 100 | ✅ | ❌ | 在 A2 基础上追加摘要节点 |
| C1 | 512 | 64 | ❌ | ✅ | 仅在 gb_standards 最终 chunks > 40k 时执行 |
| C2 | 1024 | 100 | ❌ | ✅ | 同上 |
| D1 | 512 | 64 | ✅ | ✅ | 同上 |
| D2 | 1024 | 100 | ✅ | ✅ | 同上 |

> ES 必要性判断：完整入库后，若 chunks ≤ 40k，跳过 C/D 组，只做 A/B 组。

### 7.3 执行顺序

```
P0 完成 GB 完整入库（断点续传）
  ↓
P1 生成 2000 条测试集（generate_qa_v2.py + annotate_chunk_ids.py）
  题型分布：单跳 60% / 多跳 25% / 跨文档汇总 15%
  ↓
P2 实现 RAPTOR（splitters/raptor_builder.py）
  不重跑 MinerU，在现有 chunks 上聚类 + LLM 摘要 + 追加写回
  ↓
P3 运行 8 组实验（scripts/run_eval_full.py，串行）
  A1 → A2（需重新入库）→ B1 → B2 → [视情况 C/D]
  ↓
P4 分析结果，重点关注多跳抽象题 ContextPrecision 变化
```

### 7.4 评估脚本

```bash
# 生成测试集
python scripts/generate_qa_v2.py \
  --input_dir data/raw/gb_standards/mineru_output \
  --target 2000 \
  --output data/qa_dataset/gb_qa_2000.json

# 标注 ground truth
python scripts/annotate_chunk_ids.py \
  --input data/qa_dataset/gb_qa_2000.json \
  --collection gb_standards

# 批量评估（每组）
python scripts/run_eval_full.py \
  --collection gb_standards \
  --input data/qa_dataset/gb_qa_2000.json \
  --tag A1_512_baseline
```

---

## 八、后续 Roadmap

按优先级排列，详细技术方案见各节。

| 优先级 | 模块 | 说明 | 依赖 |
|--------|------|------|------|
| P0 | GB 入库完成 | 主线阻塞项 | — |
| P1 | 测试集生成 + 标注 | 依赖 GB 入库 | P0 |
| P2 | RAPTOR 实现 | 论文亮点，解决多跳精度问题 | 独立可开始 |
| P3 | V4 实验 8 组对比 | 核心方法论验证 | P1 + P2 |
| P4a | Redis 查询缓存 | 同问题二次命中 3s → <100ms | — |
| P4b | 前端 FC 开关 | SettingsPage 补 Function Calling toggle | — |
| P4c | MCP HTTP/SSE 模式 | 当前仅 stdio | — |
| P5 | Level 7c GraphRAG | 知识图谱检索，独立模块 | 独立可开始 |

> 毕业论文策略详见 [docs/thesis_strategy.md](docs/thesis_strategy.md)
