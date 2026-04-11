# rag_project — Claude 工作指南

## 环境
- conda 环境：`mineru_2.5`（所有 Python 命令假设已激活）
- 启动服务：`bash ~/rag_project/start.sh`
- 后端：FastAPI，port 8000
- 前端：React + Vite，port 5173
- 工作目录：`~/rag_project/`

## 项目架构

```
loaders/        文档加载（MinerU PDF / Excel / Word）
splitters/      Markdown 两阶段切分（标题 + 字符数）
chains/         rag_chain.py — Embedding 多实例缓存 + Chroma + LCEL
retrievers/     hybrid_retriever（BM25+Dense+RRF）+ reranker + cross_lingual
graphs/         rag_graph.py — LangGraph 多步 RAG（7节点）
tools/          Function Calling 工具定义 + DCS 数据分析
mcp_server/     FastMCP server（暴露 RAG 为标准工具接口）
backend/
  routers/      upload.py / chat.py / collections.py
  services/     pipeline.py — 文件处理流水线
frontend/
  src/
    components/ ChatWindow / CollectionList / UploadZone
    pages/      ChatPage / KnowledgePage / SettingsPage
evaluation/     RAGAS 评估脚本
tests/          单元/集成测试
data/
  chroma_db/    向量数据库（不要手动修改）
    embedding_registry.json  ← 各 collection 的 embedding 模型映射
```

## 关键约定

**Embedding**
- 不要硬编码模型名，一律通过 `get_collection_embedding_model(collection_name)` 获取
- 注册表：`data/chroma_db/embedding_registry.json`
- 中文文档用 `BAAI/bge-large-zh-v1.5`，中英混合用 `BAAI/bge-m3`

**SSE 事件类型（chat.py → 前端）**
- `status` — 初始化进度提示（仅在流式 token 到达前显示）
- `answer_token` — 流式 token（逐字追加）
- `answer` — 完整答案（非流式客户端兜底）
- `sources` — 来源引用列表
- `fallback` — fallback 类型标注（llm / web）
- `tool_call` — Function Calling 工具调用信息
- `done` — 结束标志
- `error` — 错误信息

**LangGraph 节点**
- `router` → `retrieve` → `evaluate` → `generate` / `rewrite` / `fallback`
- `router` → `direct`（闲聊 / 聚合查询）
- `router` → `tool_executor`（Function Calling）
- 流式透传：节点通过 `state.get("token_callback")` 判断是否流式，有则 `llm.stream()`

**RAGState 关键字段**
- `token_callback: Any` — 流式回调，None 时退化为批量
- `collection_name: str` — 当前知识库名（用于查注册表 / 分库路由）
- `cross_lingual_enabled: bool` — 开启跨语言检索

## 禁止事项
- 不要修改 `data/chroma_db/` 下的任何文件（含 SQLite）
- 不要全局 pip install，必须在 mineru_2.5 环境下安装
- 不要修改 `config.py` 里的 API Key（从环境变量读取）
- 不要在节点函数里直接 `import` 重量级模块（Embedding/Reranker），用已有的单例函数

## 修改后注意事项
- 改 `graphs/rag_graph.py` → 需重启后端（graph 有缓存）
- 改 `backend/routers/` → 需重启 uvicorn
- 改 `frontend/` → Vite HMR 自动热更新，无需重启
- 改 `chains/rag_chain.py` 的 Embedding → 清理 `_graph_cache` 或重启后端

## 代码规范
- 注释用中文，变量名用英文
- 新增后端功能 → 更新 `RUN_LOG.md`
- 重要架构变更 → 更新 `PROGRESS.md`
