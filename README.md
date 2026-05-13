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

## 核心实验

**chunk_size 消融实验**（256 / 512 / 1024 token，bge-m3 嵌入固定）

```bash
# Hit@K 评估（无 LLM 成本）
python scripts/eval_precision_at_k.py \
    --input data/qa_dataset/hydro_crack_qa_annotated.json \
    --collection hydro_crack --top_k 10

# RAGAS 端到端评估
python scripts/run_ragas_ablation.py --collection hydro_crack --max 30
```

## 系统架构

```
问题输入
  │
  ▼
[LangGraph 路由节点]
  ├─ 闲聊 → 直接回答
  ├─ 工具调用 → Function Calling（DCS数据分析/网络搜索）
  └─ 知识库查询
        │
        ▼
   BM25 + Dense → RRF 融合 → BGE Reranker
        │
        ▼
   [评估节点] 置信度充分？
        ├─ 是 → 生成答案（含页码溯源）
        └─ 否 → 改写查询重检索（最多2次）
                    └─ 耗尽 → Fallback（Tavily网搜 / LLM通用知识）
```

## 作者

朱添乐（Hanyuu）× Claude Sonnet 4.6
