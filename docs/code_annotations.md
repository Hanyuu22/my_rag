# RAG 项目核心代码注解

> 覆盖所有核心模块，每段附带功能概述和关键行注解。
> 更新日期：2026-04-09

---

## 目录

1. [config.py — 全局配置](#1-configpy)
2. [loaders/ — 文档加载层](#2-loaders)
3. [splitters/ — 文本切分层](#3-splitters)
4. [chains/rag_chain.py — Embedding + 向量库 + LCEL 链](#4-chainsrag_chainpy)
5. [retrievers/ — 检索层](#5-retrievers)
6. [graphs/rag_graph.py — LangGraph 多步 RAG](#6-graphsrag_graphpy)
7. [backend/ — FastAPI 服务层](#7-backend)
8. [tools/ — Function Calling 工具](#8-tools)
9. [mcp_server/ — MCP 协议接口](#9-mcp_server)
10. [评估流水线 scripts/](#10-评估流水线-scripts)
11. [RAPTOR — 摘要节点增强](#11-raptor--摘要节点增强)

---

## 整体架构一览

```
用户问题
    │
    ▼
[FastAPI backend]          接收 HTTP 请求，管理 graph 缓存，SSE 流式返回
    │    ├── Redis          任务状态持久化 + 翻译缓存（跨进程共享，重启不丢）
    │    ├── MySQL          知识库注册表 + 文档入库记录（替代 JSON 文件）
    │    ├── MinIO          原始文件对象存储（PDF/Word/Excel 永久保留）
    │    └── Elasticsearch  全文检索索引（替代内存 BM25，持久化，IK 中文分词）
    │
    ▼
[LangGraph rag_graph]      路由 → 检索 → 评估 → 生成/改写/fallback
    │
    ├─ router_node          判断走检索 / 直接回答 / 工具调用
    ├─ retrieve_node        调用 Hybrid Retriever + Reranker
    ├─ evaluate_node        判断检索质量是否达标
    ├─ rewrite_node         改写 query 重试（最多2次）
    ├─ generate_node        基于召回文档生成答案
    ├─ direct_node          闲聊/聚合类直接回答
    ├─ fallback_node        Tavily 网络搜索 或 LLM 自身知识
    └─ tool_executor_node   执行 Function Calling 工具
         │
         ▼
    [Hybrid Retriever]     ES全文检索 + Dense（BGE Embedding）→ RRF 融合
         │
         ▼
    [BGE Reranker]         Cross-Encoder 精排，Top-K → Top-3
         │
         ▼
    [Chroma 向量库]        持久化向量存储，每个知识库独立 collection
```

**存储层分工一览**

| 存什么 | 用什么 | 为什么不用别的 |
|--------|--------|--------------|
| 原始文件（PDF/Word/Excel）| MinIO | 对象存储天然适合二进制大文件，本地磁盘绑定单机 |
| MinerU 解析输出（JSON/图片）| MinIO | 与原始文件配套，方便重新入库 |
| 知识库→模型映射 | MySQL collections 表 | 原来是 JSON 文件，并发写有损坏风险 |
| 文档入库记录 | MySQL documents 表 | 记录哪些文件在哪个库，支持追溯和重处理 |
| 上传任务状态 | Redis Hash + TTL | 重启不丢，24h 自动过期，无需手动清理 |
| 翻译缓存 | Redis String + TTL | 跨进程共享，7天 TTL，不重复计费 |
| 全文检索（BM25等价）| Elasticsearch | 持久化索引，冷启动不重建，IK 中文分词 |
| 向量检索 | Chroma | 本地持久化，轻量，适合单机部署 |

---

## 1. config.py

**做了什么**：全局配置中心，所有模块从这里读取参数。包括 LLM API、Embedding 模型、Reranker 模型、Chroma 路径、检索参数、切分参数。设计原则：任何参数只在这一个地方修改，不硬编码在各业务文件里。

```python
# ── LLM ───────────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-...")  # 优先读环境变量
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DashScope 实现了 OpenAI 兼容接口，直接用 ChatOpenAI 就能调用通义千问

LLM_MODEL = "qwen-plus"       # 生成用；换模型只改这一行
LLM_TEMPERATURE = 0.1         # 低温 = 保守/确定性强，适合知识问答

# ── Embedding ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 中文 MTEB 榜单强模型，本地运行
EMBEDDING_DEVICE = "cuda"                          # GPU 加速；无 GPU 改 "cpu"

# ── Reranker ──────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 中英双语 Cross-Encoder
RERANKER_TOP_N = 3                                 # 精排后只保留 3 条送 LLM

# ── 向量数据库 ────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.expanduser("~/rag_project/data/chroma_db")
# 所有 collection 共用同一个目录，按 collection_name 区分

# ── 检索参数 ──────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = 10   # BM25 和 Dense 各召回 10 条，RRF 融合后再精排
RERANKER_TOP_N = 3     # 精排后保留 3 条

# ── 切分参数 ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 1024      # 二次切分的最大字符数（从 512 升级，覆盖更多上下文）
CHUNK_OVERLAP = 100    # 相邻 chunk 的重叠字符数（从 64 升级，减少边界截断）
```

---

## 2. loaders/

### 2.1 mineru_loader.py

**做了什么**：读取 MinerU 解析输出的 `_content_list.json`，将每个 block 转成 LangChain `Document` 对象，携带页码和 block 类型元数据。MinerU 已经完成了 PDF 的版面分析和 OCR，这里只是"搬运"结果。

```python
class MinerULoader(BaseLoader):
    def __init__(self, content_list_path, source_name="", include_types=None):
        self.include_types = include_types or ["text", "table", "equation"]
        # 默认只加载这三类，image 块没有文字内容所以排除

    def lazy_load(self) -> Iterator[Document]:
        with open(self.path) as f:
            content_list = json.load(f)  # MinerU 输出的结构化 block 列表

        for block in content_list:
            block_type = block.get("type", "")
            if block_type not in self.include_types:
                continue  # 跳过 image 等不含文字的块

            if block_type == "table":
                caption = " ".join(block.get("table_caption", []))
                body = block.get("text", "")          # MinerU 输出的 Markdown 表格
                text = f"{caption}\n{body}".strip()   # caption + 表格内容合并
            else:
                text = block.get("text", "").strip()

            if not text:
                continue  # 空块（如未 OCR 的图片表格）跳过

            yield Document(
                page_content=text,
                metadata={
                    "source": self.source_name,
                    "page": block.get("page_idx", -1),   # 原始页码，用于溯源
                    "block_type": block_type,             # 区分文字/表格/公式
                }
            )
```

### 2.2 excel_loader.py

**做了什么**：用 pandas 读取 xlsx，按行生成 Document。每行格式为 `列名：值` 的文本，适合投资数据库这类结构化知识库。支持 row_mode（每行一 Doc）和 sheet_mode（整表一 Doc）。

```python
class ExcelLoader(BaseLoader):
    def lazy_load(self) -> Iterator[Document]:
        xl = pd.ExcelFile(self.path)
        for sheet in sheets:
            df = xl.parse(sheet).fillna("")  # NaN 统一转空字符串

            if self.row_mode:
                for idx, row in df.iterrows():
                    # 把一行的列名和值拼成自然语言格式
                    text = "\n".join(
                        f"{col}：{row[col]}"
                        for col in text_cols if str(row[col]).strip()
                    )
                    # metadata 保留 sheet 名和行号，便于溯源
                    yield Document(page_content=text,
                                   metadata={"source": ..., "sheet": sheet, "row_index": idx})
            else:
                # 整个 sheet 转 Markdown 表格，适合小表全量检索
                text = df[text_cols].to_markdown(index=False)
                yield Document(page_content=text, metadata={"sheet": sheet})
```

### 2.3 word_loader.py

**做了什么**：处理 `.doc/.docx` 文件的加载。因为 `.doc` 是旧版 OLE 二进制格式，python-docx 不支持，所以通过 PowerShell 调用 Windows Word COM 对象将其转成 PDF，再走 MinerU pipeline。整个流程：`.doc` → PDF（Word COM）→ MinerU 解析 → VLM 补表 → Documents。

```python
def doc_to_pdf(doc_path, output_pdf=None):
    # 把 WSL Linux 路径转为 Windows UNC 路径（\\wsl.localhost\Ubuntu-22.04\...）
    win_doc = _wsl_to_win_unc(str(doc_path))
    win_pdf = _wsl_to_win_unc(str(output_pdf))

    ps_script = f"""
$word = New-Object -ComObject Word.Application   # 打开 Word
$doc = $word.Documents.Open('{win_doc}')
$doc.SaveAs([ref]'{win_pdf}', [ref]17)           # 17 = wdFormatPDF
$doc.Close(); $word.Quit()
"""
    result = subprocess.run(["powershell.exe", "-Command", ps_script], ...)
    # 从 WSL 直接调用 Windows 的 powershell.exe，跨系统边界

class WordLoader:
    def load(self):
        # Step 1: .doc → PDF（复用已有 PDF 跳过转换）
        # Step 2: PDF → MinerU content_list.json（复用已有解析跳过 MinerU）
        # Step 3: 检查是否有 content_list_with_tables.json（VLM 增强版）
        #         有则优先用，无则根据 use_vlm_ocr 决定是否调用 VLM
        # Step 4: MinerULoader 读取最终 content_list → Documents
```

---

## 3. splitters/markdown_splitter.py

**做了什么**：对 MinerULoader 输出的 Documents 做两阶段切分。第一阶段按 Markdown 标题层级切（让每个 chunk 知道自己属于哪个章节），第二阶段对超长段落按字符数二次切。表格和公式整块保留不切，避免行列结构被破坏。

```python
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

def split_documents(docs):
    # 按 block_type 分流
    text_docs    = [d for d in docs if d.metadata.get("block_type") == "text"]
    preserve_docs = [d for d in docs if d.metadata.get("block_type") in ("table","equation")]
    # table 和 equation 整块保留，不参与任何切分

    # ── Step 1：标题级切分 ────────────────────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # 保留标题行，chunk 里能看到自己所属章节
    )
    header_chunks = []
    for doc in text_docs:
        splits = header_splitter.split_text(doc.page_content)
        for chunk in splits:
            # 合并原始 metadata（source/page）和标题 metadata（h1/h2/h3）
            merged_meta = {**doc.metadata, **chunk.metadata}
            header_chunks.append(Document(page_content=chunk.page_content,
                                          metadata=merged_meta))

    # ── Step 2：字符数二次切分（超长段落） ───────────────────────────────
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # 1024（已从 512 升级，对比实验 A1 vs A2）
        chunk_overlap=CHUNK_OVERLAP, # 100（已从 64 升级）
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        # 优先级：段落 > 句子 > 标点 > 空格 > 强制截断
        # 确保不在句子中间切断
    )
    final_text_chunks = char_splitter.split_documents(header_chunks)

    # ── Step 3：合并（text chunks + 不切的 table/equation） ─────────────
    all_chunks = final_text_chunks + preserve_docs
    all_chunks = [c for c in all_chunks if c.page_content.strip()]

    # ── Step 4：元数据增强 ──────────────────────────────────────────────
    return _enrich_metadata(all_chunks)
```

**`_enrich_metadata()` — chunk 元数据注入**

为每个 chunk 注入四个字段，是评估和上下文扩展的基础：

```python
def _enrich_metadata(chunks):
    import uuid
    for chunk in chunks:
        # chunk_id：全局唯一 UUID，Precision@K 评估时的精确匹配 key
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        # section_path：从 h1/h2/h3/h4 标题元数据拼接，前端溯源用
        parts = [chunk.metadata.get(h,"") for h in ["h1","h2","h3","h4"]]
        chunk.metadata["section_path"] = " / ".join(p for p in parts if p)

    # 同一 source 内建双向链（按入库顺序）
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata.get("source","")].append(chunk)
    for src_chunks in by_source.values():
        for i, chunk in enumerate(src_chunks):
            chunk.metadata["prev_chunk_id"] = src_chunks[i-1].metadata["chunk_id"] if i > 0 else ""
            chunk.metadata["next_chunk_id"] = src_chunks[i+1].metadata["chunk_id"] if i < len(src_chunks)-1 else ""
    return chunks

# chunk_id 的意义：
# 旧库（gb_standards）无此字段，Precision@K 只能用文本相似度匹配（有噪声）
# 新库（gb_standards_512 / gb_standards_1024）有 chunk_id，评估 ID 精确匹配
# prev/next_chunk_id：retrieve_node 精排后自动拉取相邻 chunk，补充跨 chunk 上下文
```

**已知局限**：
- 列表项前导句未处理（"以下情况不适用：（1）..."会被切断）
- overlap 未对齐句子边界

---

## 4. chains/rag_chain.py

**做了什么**：三件事合一：① Embedding 模型的多实例缓存管理；② Chroma 向量库的建库/加载；③ 基础 LCEL RAG 链的构建。其中 Embedding 注册表解决了"不同知识库用不同模型"的问题。

```python
# ── Embedding 多模型缓存 ──────────────────────────────────────────────────
_embeddings_cache: dict = {}   # {model_name: HuggingFaceEmbeddings 实例}

def get_embeddings(model_name=None):
    name = model_name or EMBEDDING_MODEL_NAME
    if name in _embeddings_cache:
        return _embeddings_cache[name]   # 单例：同一模型只加载一次

    instance = HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": EMBEDDING_DEVICE, "local_files_only": True},
        # local_files_only=True：离线环境不触发网络请求
        encode_kwargs={"normalize_embeddings": True},
        # normalize=True：向量归一化，余弦相似度等价于点积，加速计算
    )
    _embeddings_cache[name] = instance
    return instance


# ── Embedding 注册表 ──────────────────────────────────────────────────────
# 解决问题：不同 collection 可能用不同 Embedding 模型
# 存储：MySQL collections 表（原来是 data/chroma_db/embedding_registry.json）
# 迁移原因：JSON 文件在并发写入时有损坏风险，MySQL 有事务保护

# MySQL collections 表结构：
# CREATE TABLE collections (
#     name       VARCHAR(128) PRIMARY KEY,
#     emb_model  VARCHAR(128) NOT NULL,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

def get_collection_embedding_model(collection_name):
    # SELECT emb_model FROM collections WHERE name = ?
    result = db.execute("SELECT emb_model FROM collections WHERE name=?", (collection_name,))
    return result.scalar() or EMBEDDING_MODEL_NAME
    # 未记录的库返回全局默认值，向后兼容旧库

def save_collection_embedding_model(collection_name, model_name):
    # INSERT INTO collections ... ON DUPLICATE KEY UPDATE
    db.execute("INSERT INTO collections (name, emb_model) VALUES (?,?) "
               "ON DUPLICATE KEY UPDATE emb_model=?",
               (collection_name, model_name, model_name))
    # 入库时自动记录，查询时自动读取，无需手动维护


# ── 向量库建库/加载 ───────────────────────────────────────────────────────
def build_vectorstore(chunks, collection_name, force_rebuild=False, embedding_model=None):
    model = embedding_model or get_collection_embedding_model(collection_name)

    if not force_rebuild:
        count = _check_vectorstore_count(collection_name)
        # _check_vectorstore_count：直接用 chromadb 查文档数，不加载 Embedding
        # 避免"先加载 71s Embedding 再发现库已存在"的浪费
        if count > 0:
            embeddings = get_embeddings(model)
            return Chroma(collection_name=collection_name, ...)  # 复用已有库

    # force_rebuild=True 时：追加新 chunks 到已有 collection
    # 用于多文件上传场景，不覆盖原有数据
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(model),
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,  # 自动持久化到磁盘
    )
    save_collection_embedding_model(collection_name, model)  # 记录模型到注册表
    return vectorstore


# ── LCEL 基础 RAG 链 ──────────────────────────────────────────────────────
# 注：生产环境主要用 LangGraph（rag_graph.py），这个链用于测试和简单场景

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "只使用上下文中的信息作答，不要编造内容\n"
     "如果上下文没有相关信息，直接说'根据现有资料无法回答'\n"
     "上下文：\n{context}"),
    ("human", "{question}"),
])

def build_rag_chain(collection_name):
    retriever = load_vectorstore(collection_name).as_retriever(...)

    # LCEL 链的核心结构：parallel 同时取答案和来源文档
    rag_chain_with_source = RunnableParallel(
        answer=(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT | llm | StrOutputParser()
            # retriever → format_docs：文档列表转带页码的字符串
            # RunnablePassthrough：原样传递 question
            # 两者并行计算，填入 prompt 模板
        ),
        source_docs=retriever,  # 同时返回原始文档，用于前端显示来源
    )
    return rag_chain_with_source
```

---

## 5. retrievers/

### 5.1 hybrid_retriever.py

**做了什么**：将全文检索（关键词）和 Dense（向量语义）两路检索结果用 RRF 算法融合，取长补短。全文检索负责精确的专有名词/数字召回，Dense 负责同义词/语义相近的召回。

**演进历史**：最初用 `rank_bm25` 库做内存 BM25，每次后端启动都要重建索引（1600 chunks 约 3~5 秒），迁移到 Elasticsearch 后索引持久化，冷启动消失。

```python
# ── ES 全文检索（替代原来的内存 BM25） ────────────────────────────────────
# 原来：BM25Retriever.from_documents(chunks, preprocess_func=jieba_tokenize)
#   问题：每次启动要重建，服务重启后第一次请求有明显冷启动延迟
# 现在：ES 持久化索引，入库时写入，检索时直接查，无冷启动

class ESRetriever:
    def invoke(self, query: str) -> list[Document]:
        resp = self.es.search(
            index=self.collection_name,
            body={
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "analyzer": "ik_max_word",  # IK 中文分词，比 jieba 更准
                        }
                    }
                },
                "size": self.k,
            }
        )
        return [
            Document(
                page_content=hit["_source"]["content"],
                metadata=hit["_source"]["metadata"],
            )
            for hit in resp["hits"]["hits"]
        ]

def build_hybrid_retriever(collection_name, vectorstore, es_weight=0.5, dense_weight=0.5, k=10):
    es_retriever = ESRetriever(collection_name=collection_name, k=k)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # EnsembleRetriever 内部自动做 RRF（Reciprocal Rank Fusion）
    # RRF 公式：score = Σ 1/(k+rank)，k=60 是标准参数
    # 两路各取 Top-10 → RRF 融合 → 返回 Top-10（去重）
    return EnsembleRetriever(
        retrievers=[es_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )

# ── 入库时同步写 ES（pipeline.py 调用） ───────────────────────────────────
def index_chunks_to_es(chunks: list[Document], collection_name: str):
    """向量入 Chroma 的同时，文本写入 ES 索引，两库保持同步"""
    actions = [
        {
            "_index": collection_name,
            "_source": {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        }
        for chunk in chunks
    ]
    bulk(es_client, actions)   # ES bulk API，批量写入效率高
```

### 5.2 reranker.py

**做了什么**：BGE Reranker 精排，用 Cross-Encoder 对每个 (query, doc) 对直接打一个精确相关性分数，把最相关的文档推到最前。全局单例避免重复加载 2.27GB 模型。

```python
_reranker_instance = None

def get_reranker(top_n=RERANKER_TOP_N):
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker(top_n=top_n)
    _reranker_instance.top_n = top_n  # 允许运行时动态调整 top_n
    return _reranker_instance
    # 单例模式：bge-reranker-v2-m3 有 2.27GB，只在第一次请求时加载

class BGEReranker:
    def rerank(self, query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        # Cross-Encoder：把 query 和每个文档拼在一起过模型
        # 比 Bi-Encoder（分别编码再点积）更慢但更准确
        scores = self.model.compute_score(pairs, normalize=True)
        # normalize=True：输出归一化到 0~1，便于设置阈值

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return scored[:self.top_n]  # 只返回 Top-N（默认3条）
```

### 5.3 cross_lingual_retriever.py

**做了什么**：两个类解决跨语言检索问题。`CrossLingualRetriever` 在同一库内用双语 query 检索（原始+翻译后各搜一次，RRF 合并）。`CrossCollectionRetriever` 在配对的中英文分库间路由（如 energy_zh ↔ energy_en）。`find_partner_collection` 按命名约定自动配对。

```python
def detect_language(text):
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_alpha = len(re.findall(r'[\u4e00-\u9fff\w]', text))
    return "zh" if (zh_chars / total_alpha) > 0.3 else "en"
    # 中文字符占比 > 30% 判定为中文

# 翻译缓存：Redis String，key=translate:{lang}:{query}，TTL 7天
# 优势：跨进程共享（多 worker 共用同一缓存），重启不丢
# 降级：Redis 不可用时自动退化为模块级内存 dict
_mem_translate_cache = {}   # Redis 不可用时的内存兜底

def find_partner_collection(collection_name):
    # 命名约定：energy_zh ↔ energy_en，hydro_manual ↔ hydro_manual_en
    if collection_name.endswith("_en"):
        partner = collection_name[:-3] + "_zh"   # energy_en → energy_zh
        if partner not in existing: partner = collection_name[:-3]
    elif collection_name.endswith("_zh"):
        partner = collection_name[:-3] + "_en"   # energy_zh → energy_en
    else:
        partner = collection_name + "_en"         # hydro_manual → hydro_manual_en
    return partner if partner in existing else None

class CrossCollectionRetriever:
    def invoke(self, query):
        lang = detect_language(query)
        docs_primary = self.primary_retriever.invoke(query)          # 原始库
        translated = translate_query(query, lang, self._llm)         # LLM 翻译
        docs_partner = self.partner_retriever.invoke(translated)      # 配对库
        return _rrf_merge([docs_primary, docs_partner])[:self.top_k]
        # 中文 query 找中文内容 + 翻译成英文找英文内容，RRF 合并
```

### 5.4 multi_collection_retriever.py

**做了什么**：并行搜索多个 collection，RRF 合并结果。用 ThreadPoolExecutor 并行执行各库的 Hybrid Retrieval，从串行（N × 单库耗时）变为并行（最慢那个库的耗时）。

```python
class MultiCollectionRetriever:
    def invoke(self, query):
        results = []
        # ThreadPoolExecutor：多库并行搜索
        # 改进前：串行 for 循环，2库约 2.3s
        # 改进后：并行，约 1.5s（取最慢那个）
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as executor:
            futures = {executor.submit(r.invoke, query): r for r in self.retrievers}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"某个子检索器失败，跳过：{e}")
        return _rrf_merge(results, top_n=self.top_k)
        # 多路结果再做一次 RRF，消除不同库之间的分数偏置
```

---

## 6. graphs/rag_graph.py

**做了什么**：整个系统的控制核心，用 LangGraph 实现多步 RAG 状态机。定义了 8 个节点的流转逻辑，包括智能路由、检索质量评估、query 改写重试、流式输出、fallback 降级、Function Calling 工具调用。

### RAGState — 全局状态定义

```python
class RAGState(TypedDict):
    question: str           # 原始问题，全程不变
    query: str              # 当前检索用的 query（改写后会更新）
    docs: List[Document]    # Reranker 精排后的文档
    scores: List[float]     # 精排分数
    answer: str             # 最终答案
    retry: int              # 已改写次数
    route: str              # "retrieve" | "direct" | "tool"
    fallback_type: str      # "" | "llm" | "web"
    # 可由前端覆盖的运行时参数
    score_threshold: float  # 低于此值触发改写（默认 0.5）
    max_retry_limit: int    # 最多改写次数（默认 2）
    reranker_enabled: bool
    fallback_enabled: bool
    fallback_method: str    # "auto" | "llm" | "web"
    # Function Calling
    tools_enabled: bool     # True 时 router 用 bind_tools 模式
    tool_name: str
    tool_args: dict
    # 跨语言检索
    cross_lingual_enabled: bool
    collection_name: str    # 用于查找配对库
    # 流式输出
    token_callback: Any     # 非 None 时逐 token 回调，None 时批量模式
```

### router_node — 路由节点

```python
_AGGREGATION_RE = re.compile(
    r'(一共|总共|共有?|总计)[多少几个家条份]|[多少几](个|家|只|条|份)...',
    re.IGNORECASE
)
# 聚合类问题正则：检测"共有多少""列出所有"等 → 直接回答
# 原因：RAG 只看 top-k，无法全库计数，直接告知用户

def router_node(state, llm):
    if state.get("tools_enabled", False):
        return _router_with_tools(state, llm)  # bind_tools 模式
    else:
        return _router_string(state, llm)       # 字符串分类模式（默认）

def _router_with_tools(state, llm):
    # LLM 选择：search_knowledge_base / analyze_process_data / web_search / direct_answer
    llm_with_tools = llm.bind_tools(ALL_TOOLS + [direct_answer])
    msg = llm_with_tools.invoke(ROUTER_TOOL_PROMPT.format_messages(...))
    tc = msg.tool_calls[0]
    # search_knowledge_base → route="retrieve"（走正常检索流程）
    # analyze_process_data / web_search → route="tool"（走 tool_executor）
    # direct_answer → route="direct"
```

### retrieve_node — 检索节点

```python
def retrieve_node(state, hybrid_retriever, reranker, llm):
    if state.get("cross_lingual_enabled", False):
        # 已是多库检索器 → CrossLingualRetriever（双 query，避免重复搜索）
        # 单库 → CrossCollectionRetriever（找配对库）→ 找不到降级双 query
        cross_ret = CrossLingualRetriever(hybrid_retriever, llm)
        raw_docs = cross_ret.invoke(query)
    else:
        raw_docs = hybrid_retriever.invoke(query)  # 普通混合检索

    if state.get("reranker_enabled", True):
        scored = reranker.rerank(query, raw_docs)  # Cross-Encoder 精排
        docs = [doc for _, doc in scored]
        scores = [round(s, 3) for s, _ in scored]
    # ...
```

### evaluate_decision — 检索质量评估（条件边）

```python
def evaluate_decision(state):
    top_score = scores[0] if scores else 0.0
    threshold = state.get("score_threshold") or SCORE_THRESHOLD  # 默认 0.5

    if top_score >= threshold:
        return "generate"    # 质量达标 → 直接生成
    if retry < max_r:
        return "rewrite"     # 还有重试次数 → 改写 query
    if state.get("fallback_enabled", True):
        return "fallback"    # 重试耗尽 → fallback
    return "generate"        # fallback 关闭 → 强制生成（可能质量差）
```

### _stream_or_invoke — 流式/批量统一封装

```python
def _stream_or_invoke(prompt, llm, inputs, callback):
    if callback is None:
        return (prompt | llm | StrOutputParser()).invoke(inputs)  # 批量模式
    full = ""
    for chunk in (prompt | llm).stream(inputs):
        token = chunk.content
        if token:
            callback(token)   # 逐 token 回调（通过 asyncio.Queue 传给 SSE）
            full += token
    return full
# callback 由 chat.py 的 SSE 处理器注入，None 时退化为普通调用
```

### fallback_node — 降级节点

```python
def fallback_node(state, llm):
    # 优先走 Tavily 网络搜索（method="auto" 或 "web"）
    if method in ("auto", "web"):
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(query=question, max_results=3)
            if results:
                # 把搜索结果作为 context，让 LLM 基于网络内容回答
                return {"answer": ..., "fallback_type": "web", "web_sources": [...]}
        except Exception:
            pass  # Tavily 失败 → 降级到 LLM 自身知识

    # 方案 A：LLM 自身通用知识（明确标注，不混淆为知识库答案）
    answer = _stream_or_invoke(FALLBACK_PROMPT, ...)
    return {"answer": answer, "fallback_type": "llm"}
```

### build_rag_graph — 图构建

```python
def build_rag_graph(hybrid_retriever, reranker, llm):
    # partial 把外部依赖绑定进节点函数，避免全局变量
    _retrieve = partial(retrieve_node, hybrid_retriever=hybrid_retriever,
                        reranker=reranker, llm=_llm)
    # ...

    builder = StateGraph(RAGState)
    builder.add_node("router",        _router)
    builder.add_node("retrieve",      _retrieve)
    # ... 8 个节点

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "direct": "direct", "tool": "tool_executor"},
    )
    builder.add_conditional_edges(
        "retrieve", evaluate_decision,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    builder.add_edge("rewrite", "retrieve")  # 改写后重新检索（形成循环）
    # ...
    return builder.compile()
```

---

## 7. backend/

### 7.1 main.py — FastAPI 入口

**做了什么**：FastAPI 应用入口，配置 CORS，注册三个路由，实现非阻塞启动预热。预热在 `lifespan` 里用 `asyncio.create_task` 后台运行，不阻塞服务就绪，服务启动后即可接请求。

```python
async def _warmup():
    # 启动后后台加载所有 collection 的 RAG graph
    # run_in_executor：在线程池执行（建 BM25/加载 Embedding 是 CPU 密集型）
    # 不阻塞事件循环，服务启动期间仍可响应 /api/health 等请求
    for name in col_names:
        _graph_cache[key] = await loop.run_in_executor(None, _build_graph, name, ...)

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_warmup())  # 后台任务，不 await
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)  # 允许前端跨域
app.include_router(upload.router, prefix="/api")     # /api/upload
app.include_router(chat.router, prefix="/api")       # /api/chat
app.include_router(collections.router, prefix="/api") # /api/collections
```

### 7.2 routers/chat.py — 聊天路由

**做了什么**：核心 SSE 流式问答接口。维护 graph 缓存（避免重复建 BM25/加载 Embedding），把 LangGraph 的同步流式回调桥接到 FastAPI 的异步 SSE，通过 `asyncio.Queue` 在后台线程和主事件循环之间传递 token。

```python
# Graph 缓存：LRU OrderedDict，上限 10 个，防止显存/内存无限增长
# 缓存活在进程内存（不放 Redis）：graph 含 Embedding/BM25 对象，不可序列化
# key = (collection, extra_collections_tuple, top_k, reranker_top_n, temperature)
_graph_cache: OrderedDict = OrderedDict()
_MAX_GRAPH_CACHE = 10

def _build_graph(collection_name, extra_collections=None, top_k=10, ...):
    if extra_collections:
        # 多库模式：MultiCollectionRetriever（并行搜多个库）
        hybrid_retriever = build_multi_collection_retriever(
            [collection_name] + list(extra_collections), top_k=top_k
        )
    else:
        # 单库模式：正常 Hybrid Retriever
        vectorstore = Chroma(collection_name=collection_name, ...)
        hybrid_retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)

    reranker = get_reranker(top_n=reranker_top_n)  # 全局单例
    return build_rag_graph(hybrid_retriever, reranker, llm)

def _get_or_build_graph(req):
    # cache_key 含 collection + extra_collections + 检索参数
    # 参数变了（如 top_k 从 10 改到 5）会建新 graph
    cache_key = (req.collection, tuple(sorted(req.extra_collections)),
                 req.retriever_top_k, req.reranker_top_n, req.temperature)
    if cache_key not in _graph_cache:
        _graph_cache[cache_key] = _build_graph(...)
    return _graph_cache[cache_key]

@router.post("/chat")
async def chat(req: ChatRequest):
    graph = _get_or_build_graph(req)
    queue = asyncio.Queue()  # 后台线程 → 主事件循环的 token 传递通道

    def token_callback(token: str):
        # 在后台线程里，把 token 放进 queue
        asyncio.run_coroutine_threadsafe(queue.put(token), loop)

    async def run_graph():
        # 在线程池运行 LangGraph（同步），完成后发送结束信号
        await loop.run_in_executor(None, lambda: graph.invoke({
            "question": req.question,
            "token_callback": token_callback,  # 注入流式回调
            "cross_lingual_enabled": req.cross_lingual_enabled,
            ...
        }))
        await queue.put(None)  # None = 结束信号

    async def event_stream():
        asyncio.create_task(run_graph())   # 后台跑 graph
        # 先发 status（等第一个 token 前的状态提示）
        yield f'data: {json.dumps({"type":"status","content":"检索中..."})}\n\n'
        while True:
            token = await queue.get()
            if token is None: break        # 收到结束信号
            yield f'data: {json.dumps({"type":"answer_token","content":token})}\n\n'
        # 发来源文档、fallback 标注、done 信号
        yield f'data: {json.dumps({"type":"done"})}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**SSE 事件类型**：

| 事件 | 触发时机 | 前端处理 |
|------|---------|---------|
| `status` | 第一个 token 前 | 显示状态提示（流式 token 来了后隐藏）|
| `answer_token` | LLM 每输出一个 token | 逐字追加到气泡 |
| `answer` | 非流式兜底 | 整段替换 |
| `sources` | graph 执行完 | 显示来源引用 |
| `fallback` | 走了 fallback | 显示 web/llm 标注 |
| `done` | 结束 | 设 loading=false |
| `error` | 异常 | 显示错误信息 |

### 7.3 routers/upload.py — 上传路由

**做了什么**：接收前端上传的文件，上传至 MinIO 对象存储，调用 `pipeline.process_file` 处理（ES + Chroma + MySQL 三写），任务状态持久化至 Redis。

```python
@router.post("/upload")
async def upload_file(file: UploadFile, collection_name: str = Form(...)):
    # ── 1. 文件校验 ───────────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".doc", ".docx", ".xlsx", ".xls"):
        raise HTTPException(400, f"不支持的文件格式：{suffix}")
    # 大小校验：保存后检查，超 300MB 删除并返回 413
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"文件过大：{file_size//1024//1024}MB")

    # ── 2. 上传到 MinIO（永久保留原始文件） ──────────────────────────────
    minio_key = f"{collection_name}/{task_id}/{file.filename}"
    minio_client.put_object("rag-docs", minio_key, file.file, length=file_size)
    # 好处：pipeline 失败可从 MinIO 重新处理，不需用户重新上传

    # ── 3. 写 Redis：任务状态（重启后前端仍可查进度） ─────────────────────
    task_set(task_id, {
        "status": "pending", "step": "等待处理...", "percent": 0,
        "filename": file.filename, "collection": collection_name,
        "minio_key": minio_key,   # 记录 MinIO 路径，方便重处理
        "error": None, "result": None,
    })
    # Redis TTL = 24h，任务自动过期，不需要手动清理

    # ── 4. 后台线程处理 ──────────────────────────────────────────────────
    threading.Thread(target=_run_pipeline, args=(task_id, minio_key, ...), daemon=True).start()
    return {"task_id": task_id}

# pipeline 完成后同步写 MySQL documents 表（文档追溯）：
# INSERT INTO documents (collection, filename, minio_key, chunks, use_vlm)
# VALUES (?, ?, ?, ?, ?)
```

**任务状态查询（GET /api/tasks/{task_id}）**：从 Redis 读取，重启后仍可返回正确进度。原来是内存 dict，重启后返回 404。

### 7.4 routers/collections.py — 知识库管理

**做了什么**：列出所有知识库（带 chunk 数和 embedding 模型信息），以及删除指定知识库（级联清理 Chroma + MySQL + ES）。

```python
@router.get("/collections")
async def list_collections():
    # 从 MySQL 读取 embedding 模型（原来是 embedding_registry.json）
    cols = chromadb_client.list_collections()
    return {"collections": [
        {"name": c.name, "count": c.count(),
         "embedding_model": db.get_collection_model(c.name)}
        for c in cols
    ]}

@router.delete("/collections/{name}")
async def delete_collection(name):
    chromadb_client.delete_collection(name)   # 从 Chroma 删除向量
    es_client.indices.delete(index=name)       # 从 ES 删除全文索引
    db.execute("DELETE FROM collections WHERE name=?", (name,))    # MySQL
    db.execute("DELETE FROM documents WHERE collection=?", (name,)) # 级联删文档记录
    # 四处同步删除，保持数据一致性
    # 原来只删 Chroma，embedding_registry.json 残留是已知问题
```

### 7.5 backend/redis_client.py — Redis 封装

**做了什么**：Redis 单例连接 + 任务状态的三个操作函数（set/update/get）。核心设计：dict/list 字段自动 JSON 序列化，percent 自动恢复为 int，TTL 24h 自动过期。

```python
def task_set(task_id: str, data: dict) -> None:
    """创建任务，写入 Redis Hash，设置 24h TTL"""
    r = get_redis()
    flat = {k: json.dumps(v) if isinstance(v, (dict,list)) else (v or "")
            for k, v in data.items()}
    r.hset(f"task:{task_id}", mapping=flat)
    r.expire(f"task:{task_id}", 86400)   # 24h 后自动过期，无需手动清理

def task_get(task_id: str) -> dict | None:
    """读取任务，自动反序列化 JSON 字段和 int 类型"""
    data = get_redis().hgetall(f"task:{task_id}")
    if not data: return None
    result = {}
    for k, v in data.items():
        try: result[k] = json.loads(v)    # dict/list 字段反序列化
        except: result[k] = v or None
    if "percent" in result:
        result["percent"] = int(result["percent"])  # 恢复 int 类型
    return result
```

### 7.6 backend/db.py — MySQL 封装

**做了什么**：SQLAlchemy 连接池 + 两张表的 CRUD 操作。collections 表替代 embedding_registry.json，documents 表记录每次入库的文档信息（原来完全缺失）。

```python
# 表结构
# collections: name(PK) | emb_model | created_at
# documents:   id | collection | filename | minio_key | chunks | use_vlm | ingested_at

def record_document(collection: str, filename: str, minio_key: str,
                    chunks: int, use_vlm: bool):
    """入库完成后写一条 documents 记录，支持后续追溯和重处理"""
    db.execute(
        "INSERT INTO documents (collection, filename, minio_key, chunks, use_vlm) "
        "VALUES (?, ?, ?, ?, ?)",
        (collection, filename, minio_key, chunks, use_vlm)
    )

def list_documents(collection: str) -> list[dict]:
    """查询某个知识库包含哪些文件（前端'文档列表'功能的数据源）"""
    return db.execute(
        "SELECT filename, chunks, use_vlm, ingested_at FROM documents "
        "WHERE collection=? ORDER BY ingested_at DESC", (collection,)
    ).fetchall()
```

### 7.7 backend/storage.py — MinIO 封装

**做了什么**：MinIO 客户端单例 + 上传/下载/删除操作封装。原始文件上传后永久保留，出问题可以重新入库而不需用户重新上传。

```python
BUCKET = "rag-docs"

def upload_file(local_path: str, object_key: str) -> str:
    """上传文件到 MinIO，返回 object key（存入 MySQL documents 表）"""
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        get_minio().put_object(BUCKET, object_key, f, length=size)
    return object_key

def download_file(object_key: str, local_path: str):
    """从 MinIO 下载文件到本地临时路径（pipeline 重处理时用）"""
    get_minio().fget_object(BUCKET, object_key, local_path)

def delete_collection_files(collection: str):
    """删除知识库时清理 MinIO 中对应的所有原始文件"""
    objects = get_minio().list_objects(BUCKET, prefix=f"{collection}/", recursive=True)
    for obj in objects:
        get_minio().remove_object(BUCKET, obj.object_name)
```

### 7.5 services/pipeline.py — 文件处理流水线

**做了什么**：统一的文件入库流水线，根据文件类型（PDF/Word/Excel）自动分发处理，通过 `progress_callback` 实时汇报进度。含一个优化：`energy` 类知识库（无数学公式）自动禁用 MinerU 的 MFD+MFR 模块，避免图表误判为公式。

```python
def process_file(file_path, collection_name, use_vlm_ocr=True, progress_callback=None, embedding_model=None):
    lang = "en" if collection_name.endswith("_en") else "ch"
    # 根据库名后缀自动设置 OCR 语言

    if suffix == ".pdf":
        return _process_pdf(path, collection_name, ...)
    elif suffix in (".doc", ".docx"):
        return _process_word(...)  # Word → PDF → MinerU
    elif suffix in (".xlsx", ".xls"):
        return _process_excel(...)  # 直接 pandas 读取

def _process_pdf(path, collection_name, ...):
    # 优化：energy 库禁用 MFD+MFR（公式检测），加速解析且避免误判
    has_formula = not collection_name.startswith("energy")
    do_parse(
        output_dir=..., pdf_bytes_list=[read_fn(path)],
        p_lang_list=[lang], backend="pipeline",
        # MFD/MFR 控制参数
    )
    # 检查 content_list_with_tables.json（VLM 增强版）是否存在
    enhanced_path = ... / "content_list_with_tables.json"
    if enhanced_path.exists():
        content_list_path = enhanced_path   # 优先用 VLM 增强版
    elif use_vlm_ocr:
        run_ocr(content_list_path, enhanced_path)  # 首次处理调 VLM

    docs = MinerULoader(content_list_path).load()
    chunks = split_documents(docs)
    build_vectorstore(chunks, collection_name, force_rebuild=True, embedding_model=...)
    return {"chunks": len(chunks), "collection": collection_name}
```

---

## 8. tools/

### 8.1 tool_definitions.py

**做了什么**：定义三个 LangChain StructuredTool，供 LangGraph router 的 `bind_tools` 路由使用，也供 MCP Server 直接调用。工具有明确的 pydantic Schema，LLM 调用时能准确提取参数。

```python
# 三个工具的分工：
# search_knowledge_base → 向量检索本地知识库（专业文档问答）
# analyze_process_data  → pandas 分析 DCS 时序数据（数值统计）
# web_search            → Tavily 搜索互联网（知识库未覆盖时）

class SearchKBInput(BaseModel):
    query: str      = Field(description="检索查询语句，尽量具体")
    collection: str = Field(description="知识库名称")

TOOL_SEARCH_KB = StructuredTool.from_function(
    func=_search_kb_func,
    name="search_knowledge_base",
    description="在本地知识库检索专业文档。遇到专业性问题优先选此工具。",
    args_schema=SearchKBInput,
)

ALL_TOOLS = [TOOL_SEARCH_KB, TOOL_ANALYZE_DATA, TOOL_WEB_SEARCH]
# bind_tools 时传入整个列表，LLM 自行选择
```

### 8.2 ocr_empty_tables.py

**做了什么**：对 MinerU 未能 OCR 的表格（`text` 字段为空但有 `img_path`），调用 qwen-vl-plus 识别图片内容，输出 Markdown 表格格式写回。结果永久缓存为 `content_list_with_tables.json`，不重复计费。

```python
def ocr_table_image(img_path, page):
    b64 = image_to_base64(img_path)  # 图片转 base64
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}"}},
                {"type": "text",
                 "text": "请将表格完整转录为 Markdown 表格格式，保留所有数值、单位、行列结构。只输出 Markdown 表格。"}
            ]
        }]
    )
    return response.choices[0].message.content.strip()

def run_ocr(content_list_path, output_path):
    empty_tables = [b for b in blocks if b.get("type")=="table"
                    and not b.get("text","").strip()
                    and b.get("img_path")]  # 有图片才能 VLM 处理
    for block in empty_tables:
        text = ocr_table_image(img_path, page)
        block["text"] = text          # 写回 text 字段
        time.sleep(REQUEST_INTERVAL)  # 避免触发 API 限流
    with open(output_path, "w") as f:
        json.dump(blocks, f)          # 保存增强版 content_list
```

---

## 9. mcp_server/server.py

**做了什么**：用 FastMCP 把本项目的 RAG 系统暴露为 MCP 协议工具，供 Claude Desktop 等 MCP 客户端直接调用。通过 stdio 传输（`mcp.run(transport="stdio")`），Claude Desktop 启动时自动加载，像内置工具一样使用知识库。

```python
mcp = FastMCP(
    "rag-knowledge-base",
    instructions="这是一个工业文档 RAG 知识库系统。使用 list_collections 获取知识库列表，再用 search_knowledge_base 检索。",
)

@mcp.tool()
def list_collections() -> str:
    """列出所有可用知识库及文档片段数"""
    # 直接查 Chroma，返回格式化字符串给 MCP 客户端

@mcp.tool()
def search_knowledge_base(query: str, collection: str, top_k: int = 5) -> str:
    """向量检索指定知识库"""
    from tools.tool_definitions import _search_kb_func
    return _search_kb_func(query=query, collection=collection)
    # 复用 tool_definitions 里的实现，保持一致

@mcp.tool()
def analyze_process_data(query: str) -> str:
    """DCS 数据 pandas 分析"""
    from tools.data_analyzer import analyze_process_data as _analyze
    return _analyze(query)

if __name__ == "__main__":
    mcp.run(transport="stdio")  # stdio 模式：标准输入输出，Claude Desktop 用这个
    # 改为 mcp.run(transport="sse") 可切换到 HTTP/SSE 模式供其他客户端使用
```

---

## 数据流完整示例

用户问："加氢精制的脱硫反应原理是什么？"

```
1. 前端 POST /api/chat
   {"question": "加氢精制的脱硫反应原理是什么？", "collection": "hydro_manual"}

2. chat.py _get_or_build_graph → 命中缓存，直接返回已有 graph

3. LangGraph 开始执行：
   router_node → LLM 判断 → "retrieve"（需要查知识库）

4. retrieve_node：
   BM25（jieba）：["加氢精制", "脱硫", "反应原理"] → 召回 10 条
   Dense（bge-large-zh）：语义向量 → 召回 10 条
   RRF 融合 → 20 条去重排序
   BGE Reranker → 精排保留 Top-3

5. evaluate_decision：
   top_score = 0.87 ≥ 0.5 → "generate"

6. generate_node：
   context = format_docs(docs)   # 3条文档拼成带页码的字符串
   LLM 流式生成 → token_callback 逐 token 推给 queue

7. SSE event_stream 消费 queue：
   → "data: {type: answer_token, content: '脱'}"
   → "data: {type: answer_token, content: '硫'}"
   → ...
   → "data: {type: sources, content: [{page:6, score:0.87, ...}]}"
   → "data: {type: done}"

8. 前端 ChatWindow 渲染：
   逐字追加到气泡，显示来源引用，清除 loading 状态
```

---

## 10. 评估流水线 scripts/

### 整体思路

RAG 系统的检索质量评估面临一个核心难题：**自动标注的 ground truth 本身有噪声**。我们用三层递进的方式解决这个问题：

```
LLM 自动出题（generate_qa_v2.py）
    ↓
语义检索自动标注 chunk_id（annotate_chunk_ids.py）   ← 有标注噪声
    ↓
Precision@K 评估（eval_precision_at_k.py）            ← 受噪声低估
    ↓
人工抽检修正（spot_check.py）                         ← 量化低估程度
    ↓
RAGAS 端到端评估（eval_ragas_gb.py）                  ← 不依赖 chunk_id，更鲁棒
```

### annotate_chunk_ids.py — 自动标注 ground truth

```python
# 原理：用 QA 的 source_text 向 Chroma 做语义检索，Top-1 chunk 作为 ground truth
# 问题：source_text 来自 mineru_output 原始文本，chunk 经 markdown_splitter 切分后文本略有差异
#       导致约 20% 的标注可能偏移（标注到次优 chunk）

MIN_SCORE = 0.75   # 相似度低于此值不标注（视为无法匹配）

query_emb = embeddings.embed_query(source_text)
results = collection.query(query_embeddings=[query_emb], n_results=TOP_K, ...)
scores = [max(0.0, 1 - d / 2) for d in distances]  # L2 距离 → 相似度

# 优先用 metadata 里的 chunk_id（新库有），否则用 Chroma 内部 ID（旧库兜底）
qa["ground_truth_chunk_id"]  = metas[0].get("chunk_id") or ids[0]
qa["ground_truth_chunk_ids"] = [m.get("chunk_id") or i for m, i in zip(metas[:3], ids[:3])]
# 存 Top-3 备用：评估时任意一个命中即算 hit，缓解切分边界 off-by-one 问题
```

### eval_precision_at_k.py — Precision@K 评估

```python
# 核心指标：Hit@K（是否在 Top-K 结果里命中 ground truth）
# MRR：Mean Reciprocal Rank，衡量命中位置（1/rank 的均值）

gt_ids = set(qa.get("ground_truth_chunk_ids", [gt_id]))   # 用 Top-3 标注集合
retrieved_ids = [d.metadata.get("chunk_id","") for d in docs]

rank = None
for pos, cid in enumerate(retrieved_ids, 1):
    if cid and cid in gt_ids:
        rank = pos
        break

# 实验结果（gb_standards_512 vs gb_standards_1024，443 道有效题）
# A1（512/64）：Hit@1=0.242  Hit@3=0.424  Hit@5=0.496  MRR=0.355
# A2（1024/100）：Hit@1=0.248  Hit@3=0.433  Hit@5=0.492  MRR=0.355
# → 绝对值偏低，原因：~20% 标注噪声导致系统性低估（人工抽检估算真实 Hit@5 ≈ 0.60+）
# → 512 vs 1024 差异极小，两者 BM25+Dense+RRF 检索能力相当
```

### spot_check.py — 人工抽检验证

```python
# 目的：量化"标注噪声导致 Precision@K 被低估"的程度
# 输出：终端逐题展示"问题 + 标注 chunk + 检索 Top-K"，人工输入 y/n/?

# 交互逻辑：
# y → 检索结果包含正确答案，但标注指向了错误 chunk（标注噪声）
# n → 检索确实没找对（真实 Miss）
# ? → 无法判断（如标注本身是乱码、题目有歧义）

# 统计：
# 自动命中（ID 匹配）：auto_hit / total
# 人工修正后命中：(auto_hit + manual_y) / total  ← 更接近真实 Hit@K
# 标注噪声估算：manual_y / total
```

### eval_ragas_gb.py — RAGAS 端到端评估

```python
# 与 Precision@K 的核心区别：不需要 chunk_id，用 LLM 语义判断
# Context Recall：ground_truth 答案文本能否从检索结果中推导出来
# Context Precision：检索到的 chunk 是否都有用（噪声少的精度高）
# Faithfulness：生成答案是否忠于检索内容（不幻觉）
# Answer Relevancy：答案是否切题

# 两阶段，支持断点续传：
# 1. generate_answers()：逐题检索 + LLM 生成，存 draft_{collection}.json
# 2. run_ragas()：批量送 RAGAS，结果存 ragas_{collection}.json

# 运行：
# python scripts/eval_ragas_gb.py --collection gb_standards_512 --limit 50  # 快速验证
# python scripts/eval_ragas_gb.py --collection gb_standards_512             # 全量
# python scripts/eval_ragas_gb.py --ragas-only --draft data/results/draft_gb_standards_512.json
```

---

## 11. RAPTOR — 摘要节点增强

**做了什么**：在已有 collection 基础上，为每个文档生成 1~N 个段落级摘要节点追加入库，不需要重新入库原始文档。摘要节点在向量检索时与普通 chunk 一起参与，对 multi_hop 和 comparison 类问题有额外帮助（摘要覆盖范围更广，能帮助检索器找到相关区域再由原始 chunk 细化）。

```python
# 文件：splitters/raptor_builder.py

def build_raptor_nodes(collection_name, max_chunk_chars=3000, request_interval=1.0):
    # 1. 读取 collection 所有 chunks，按 source 文件分组
    # 2. 跳过已有 raptor_summary 节点的文件（断点续传）
    # 3. 每个文件的 chunks 按 max_chunk_chars 合并成若干段
    # 4. 每段调 LLM 生成 150~300 字摘要
    # 5. 摘要 Document 追加回同一 collection（vectorstore.add_texts）

# 摘要节点 metadata 标识：
# node_type   = "raptor_summary"   ← 区分于普通 chunk
# chunk_id    = "raptor_{source}_{seg_i}"
# seg_index   = 段编号
# orig_chunks = 该文档原始 chunk 数

# 评估时的行为：
# - 摘要节点参与向量检索，增加宽泛问题的召回面
# - Precision@K 匹配基于 chunk_id，摘要节点不会被误算为命中
# - B1（512+RAPTOR）vs A1（512）= RAPTOR 对检索质量的净增益

# 运行：
# python splitters/raptor_builder.py --collection gb_standards_512   → B1 实验库
# python splitters/raptor_builder.py --collection gb_standards_1024  → B2 实验库
```

**为什么不重新入库**：RAPTOR 的摘要是对已有 chunk 内容的压缩，向量空间里摘要向量和原始 chunk 向量本来就存在语义关联，追加而非重建不会破坏这种关系。Chroma 的 `add_texts` 保证追加原子性。

**实验对比设计**：

| 实验 | Collection | 说明 |
|------|-----------|------|
| A1 | gb_standards_512 | 512/64，无 RAPTOR（基线）|
| A2 | gb_standards_1024 | 1024/100，无 RAPTOR |
| B1 | gb_standards_512 + RAPTOR | 512/64 + 摘要节点 |
| B2 | gb_standards_1024 + RAPTOR | 1024/100 + 摘要节点 |

---

## 待改进计划

| 优先级 | 改动 | 位置 | 状态 |
|--------|------|------|------|
| 高 | RAPTOR 摘要节点追加 | `splitters/raptor_builder.py` | ✅ 已实现 |
| 高 | Precision@K 评估流水线 | `scripts/eval_precision_at_k.py` | ✅ 已实现 |
| 高 | RAGAS 端到端评估 | `scripts/eval_ragas_gb.py` | ✅ 已实现 |
| 中 | B1/B2 RAPTOR 实验跑通 | 运行 raptor_builder + eval | 🔲 待执行 |
| 中 | Overlap 改为句子边界对齐 | `markdown_splitter.py` | 🔲 |
| 低 | 列表项合并前导句 | `markdown_splitter.py` | 🔲 |
| 低 | GraphRAG（知识图谱）| 新增 `graphs/graph_rag.py` | 🔲 |
