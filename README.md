# RAG Project

基于 MinerU + LangChain 的工业级检索增强生成系统，以加氢裂化工艺规程为核心实验数据集。

## 实验规模

本项目包含两个层次的系统性评估：

**小规模消融实验**（加氢裂化工艺规程，~100 页）
- 目标：在受控规模下对比单一变量（chunk_size / embedding 模型）的影响
- 知识库规模：98～361 chunks，约 2～4 万 token
- 评估方式：Hit@K（无 LLM 成本）+ RAGAS 四指标
- 典型对比：chunk_size = 256 / 512 / 1024 token，bge-m3 vs qwen3-embedding

**大规模总体测试**（GB/T 国标文档库，200+ PDF）
- 目标：验证系统在真实生产规模下的检索与生成能力
- 知识库规模：目标 8k～15k chunks，数十万 token
- 评估方式：2000 条分类测试集（单跳 / 多跳 / 跨文档汇总），RAGAS 全量评估
- 典型对比：Baseline（纯 Hybrid）/ RAPTOR 摘要树 / GraphRAG 知识图谱

## 技术栈

| 层 | 技术 |
|----|------|
| 文档解析 | MinerU 2.6（PDF/Word/Excel，保留表格+公式结构） |
| Embedding | BAAI/bge-m3（本地 GPU，8192 token，中英双语） |
| 向量库 | Chroma（持久化，多 collection） |
| 检索 | BM25 + Dense → RRF 融合 → BGE Reranker 精排 |
| 流程编排 | LangGraph（7节点：路由/检索/评估/改写/生成/工具/Fallback） |
| 生成 LLM | qwen-plus（DashScope API） |
| 后端 | FastAPI + SSE 流式输出 |
| 前端 | React 19 + TypeScript + Vite |
| 评估 | RAGAS（四指标）+ Hit@K（Precision@1/3/5/MRR） |

## 项目结构

```
rag_project/
├── config.py                   全局配置（模型/路径/参数）
├── loaders/                    文档加载器（MinerU/Excel/Word）
├── splitters/                  Markdown 两阶段 token 切分
├── chains/                     Embedding 单例 + Chroma + LCEL 链
├── retrievers/                 Hybrid（BM25+Dense+RRF）+ Reranker + 跨语言检索
├── graphs/                     LangGraph 多步 RAG（含 Function Calling）
├── tools/                      工具定义 + DCS 时序数据分析
├── mcp_server/                 FastMCP Server（标准工具接口）
├── backend/                    FastAPI 后端（上传/问答/知识库管理）
├── frontend/                   React 前端（对话/知识库/设置）
├── scripts/                    评估/入库/QA生成工具脚本
├── evaluation/                 RAGAS 评估脚本
├── data/
│   ├── chroma_db/              向量库持久化
│   └── qa_dataset/             评估数据集
├── start.sh                    一键启动（后端+前端）
└── PROGRESS.md                 详细进度与实验记录
```

## 快速启动

```bash
conda activate mineru_2.5
bash ~/rag_project/start.sh
# 后端: http://localhost:8000
# 前端: http://localhost:5173
```

## 知识库（Collections）

| 库名 | Chunks | 说明 |
|------|--------|------|
| `hydro_crack` | 178 | 加氢裂化工艺规程，chunk_size=512 token（主实验库） |
| `hydro_crack_256` | 361 | 同源，chunk_size=256 token（消融对比组） |
| `hydro_crack_1024` | 98 | 同源，chunk_size=1024 token（消融对比组） |
| `energy_zh` / `energy_en` | 12508 / 5961 | 中英能源报告，跨语言检索配对库 |
| `investment_db` | 7440 | 投资机构&项目数据库 |

## 系统架构

### LangGraph 流程图（10节点）

```
START
  │
  ▼
[condense]  ← 对话历史压缩：将追问改写为独立完整问题（无历史时零开销跳过）
  │
  ▼
[router]    ← 双模式路由
  │   tools_enabled=False（默认）：LLM 字符串分类
  │     ├─ 聚合/计数类问题（正则匹配）→ "direct"（RAG无法全库统计）
  │     ├─ 闲聊/问候 → "direct"
  │     └─ 知识查询 → "retrieve"
  │   tools_enabled=True：bind_tools 模式，LLM 主动选择工具
  │     ├─ search_knowledge_base → "retrieve"
  │     ├─ analyze_process_data → "tool"（DCS时序分析）
  │     ├─ web_search → "tool"（Tavily主动搜索）
  │     └─ direct_answer → "direct"
  │
  ├─ "direct" ──────────────────────────→ [direct] → END
  │                                         带对话历史的直接回答
  │
  ├─ "tool" ────────────────────────────→ [tool_executor] → END
  │                                         Pandas DCS分析 / Tavily网络搜索
  │
  └─ "retrieve" ──→ [decompose]  ← Query 拆解节点（多跳复杂问题检测）
                       │   LLM 判断问题是否复杂（需跨实体/多跳推理）
                       │   is_complex=False → "retrieve"（原样进入正常检索）
                       │   is_complex=True  → 拆成子问题分别检索 → RRF合并 → Reranker精排
                       │                      → "generate"（跳过retrieve，直接生成）
                       │
                       └─ "retrieve" ──→ [retrieve]  ← 混合检索节点
                                            │   BM25（jieba分词）+ Dense（bge-m3）→ RRF融合
                                            │   可选：cross_lingual_enabled=True
                                            │     → 语言检测 → LLM翻译 → 双路检索 → RRF
                                            │     → 找到配对库（foo↔foo_en）→ CrossCollectionRetriever
                                            │   可选：reranker_enabled=True
                                            │     → BGE Reranker（bge-reranker-v2-m3）精排 Top-N
                                            │   可选：parent_child_enabled=True
                                            │     → child chunk 命中 → 替换为 parent chunk（更完整上下文）
                                            │   邻居扩展：prev/next_chunk_id 自动追加相邻 chunk
                                            │
                                            ▼
                                       [evaluate_decision]  ← 条件边
                                            │   top1_score ≥ threshold（默认0.5）→ "generate"
                                            │   top1_score < threshold 且 retry < max（默认2次）→ "rewrite"
                                            │   重试耗尽 + fallback_enabled=True → "fallback"
                                            │   重试耗尽 + fallback_enabled=False → "generate"（强制）
                                            │
                                   ┌────────┼────────────────────┐
                                   ▼        ▼                    ▼
                              [rewrite] [generate]          [fallback]
                                 │      带对话历史              优先：Tavily网络搜索
                                 │      + 页码溯源生成          失败降级：LLM自身知识
                                 │      → END                  标注 fallback_type=web/llm
                                 │                             → END
                                 └──────→ [retrieve]（循环，最多max次）
```

### RAGState 关键字段

| 字段 | 说明 |
|------|------|
| `score_threshold` | 检索置信度阈值（默认0.5），低于此值触发改写 |
| `max_retry_limit` | 最大改写重试次数（默认2） |
| `reranker_enabled` | 是否启用 BGE Reranker 精排 |
| `fallback_enabled` | 是否启用 Fallback 路由（重试耗尽时） |
| `fallback_method` | fallback 策略：`auto`/`web`/`llm` |
| `tools_enabled` | 是否启用 bind_tools 路由（默认 False） |
| `cross_lingual_enabled` | 是否启用跨语言双路检索 |
| `decompose_enabled` | 是否启用 Query 拆解（默认 True） |
| `parent_child_enabled` | 是否启用父子 chunk 替换 |
| `token_callback` | 流式输出回调（None 时退化为批量模式） |

---

## 核心实验

### 阶段一：chunk_size 消融实验（固定 bge-m3，变 chunk_size）

目标：找到最优切分粒度，隔离 Reranker 变量，纯评估检索质量。

```bash
# Hit@K 评估（无 LLM 调用，零成本）
python scripts/eval_precision_at_k.py \
    --input data/qa_dataset/hydro_crack_qa_annotated.json \
    --collection hydro_crack --top_k 10        # 512 token

python scripts/eval_precision_at_k.py \
    --input data/qa_dataset/hydro_crack_256_annotated.json \
    --collection hydro_crack_256 --top_k 10    # 256 token

python scripts/eval_precision_at_k.py \
    --input data/qa_dataset/hydro_crack_1024_annotated.json \
    --collection hydro_crack_1024 --top_k 10   # 1024 token

# RAGAS 端到端评估（含 LLM 成本，--no_reranker 隔离检索变量）
python scripts/run_ragas_ablation.py --collection hydro_crack      --max 30 --no_reranker
python scripts/run_ragas_ablation.py --collection hydro_crack_256  --max 30 --no_reranker
python scripts/run_ragas_ablation.py --collection hydro_crack_1024 --max 30 --no_reranker
```

评估指标：Hit@1 / Hit@3 / Hit@5 / MRR（检索质量）+ Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall（生成质量）

### 阶段二：Embedding 模型对比（固定最优 chunk_size，变 Embedding）

目标：对比 bge-m3 与 Qwen3-Embedding-0.6B 在专业工业文档上的检索效果。

```bash
# 用最优 chunk_size 建 Qwen3 库（待 ModelScope 下载完成后执行）
python scripts/ingest_ablation.py --collection hydro_crack_qwen3 --chunk_size 512

# RAGAS 评估（开启 Reranker，对比端到端生成质量）
python scripts/run_ragas_ablation.py --collection hydro_crack       # bge-m3
python scripts/run_ragas_ablation.py --collection hydro_crack_qwen3 # qwen3-embedding
```

### 阶段三：GB/T 大规模测试（待 GB 完整入库后执行）

目标：验证系统在 8k～15k chunks 规模下，RAPTOR 摘要树和 GraphRAG 知识图谱对多跳问题的提升效果。

```bash
# 2000 条分类测试集（单跳60% / 多跳25% / 跨文档汇总15%）
python scripts/generate_qa_v2.py --collection gb_standards --target 2000

# 对比：Baseline / +RAPTOR / +GraphRAG
python scripts/run_eval_full.py --collection gb_standards --tag baseline
python scripts/run_eval_full.py --collection gb_standards --raptor_enabled --tag raptor
```

## 作者

朱添乐（Hanyuu）× Claude Sonnet 4.6
