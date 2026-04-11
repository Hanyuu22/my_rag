"""
Level 7d — Cross-Lingual Retriever

两个类解决不同层面的跨语言检索问题：

CrossLingualRetriever（单库双语）：
  适合混合语言文档在同一 collection 的场景。
  原始 query + 翻译 query 各检索一次，RRF 合并。
  实验结论：对 EN→ZH 有效（+12pp），对 ZH→EN 改善有限。

CrossCollectionRetriever（分库路由）：
  适合中英文档分别建库的场景（根本解法）。
  命名约定：hydro_manual（ZH）↔ hydro_manual_en（EN）
  strategy：
    1. 检测 query 语言
    2. 原始 query 搜原始 collection
    3. 翻译 query 搜配对 collection（自动查找）
    4. RRF 跨库融合，彻底消除语言偏置

与现有架构的集成：
  - retrieve_node：cross_lingual_enabled=True 时自动选用合适的 Retriever
  - ChatRequest / RAGState 新增 cross_lingual_enabled / translated_query
  - 对外接口统一（.invoke(query) → List[Document]）
"""

import re
import sys
from typing import List

sys.path.insert(0, "/home/hanyuu/rag_project")

from langchain_core.documents import Document


# ── 语言检测 ───────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    简单语言检测：统计中文字符比例
    返回 "zh" 或 "en"
    """
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_alpha = len(re.findall(r'[\u4e00-\u9fff\w]', text))
    if total_alpha == 0:
        return "en"
    return "zh" if (zh_chars / total_alpha) > 0.3 else "en"


# ── 查询翻译 ───────────────────────────────────────────────────────────────

_mem_translate_cache: dict[str, str] = {}  # Redis 不可用时的内存兜底

TRANSLATE_TTL = 3600 * 24 * 7  # 翻译缓存 7 天


def translate_query(query: str, src_lang: str, llm=None) -> str:
    """
    将 query 翻译到另一种语言，结果缓存避免重复调用。
    src_lang: "zh" → 翻译成英文；"en" → 翻译成中文
    缓存优先用 Redis（key: translate:{src_lang}:{query}），不可用时降级内存 dict。
    """
    cache_key = f"translate:{src_lang}:{query}"

    # 读缓存
    try:
        from backend.redis_client import get_redis
        r = get_redis()
        cached = r.get(cache_key)
        if cached:
            return cached
        use_redis = True
    except Exception:
        use_redis = False
        if query in _mem_translate_cache:
            return _mem_translate_cache[query]

    _llm = llm
    if _llm is None:
        from chains.rag_chain import get_llm
        _llm = get_llm()

    target = "英文" if src_lang == "zh" else "中文"
    prompt = (
        f"请将以下检索查询翻译成{target}，只输出翻译结果，不要任何解释或前缀。\n\n"
        f"查询：{query}"
    )
    from langchain_core.messages import HumanMessage
    result = _llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # 写缓存
    if use_redis:
        try:
            r.set(cache_key, result, ex=TRANSLATE_TTL)
        except Exception:
            pass
    else:
        _mem_translate_cache[query] = result

    return result


# ── RRF 融合 ───────────────────────────────────────────────────────────────

def _rrf_merge(lists: list[list[Document]], k: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion：合并多路检索结果
    k=60 是标准 RRF 参数，较小的 k 让头部排名影响更大
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content[:100]  # 用内容前缀作为去重 key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


# ── CrossLingualRetriever ──────────────────────────────────────────────────

class CrossLingualRetriever:
    """
    双语检索器：原始 query + 翻译 query 各检索一次，RRF 合并。

    用法：
        retriever = CrossLingualRetriever(hybrid_retriever, top_k=10)
        docs = retriever.invoke("What is the reactor temperature range?")
        # 内部自动翻译成中文也检索一遍，合并后返回

    接口与 LangChain Retriever 兼容（支持 .invoke()）。
    """

    def __init__(self, hybrid_retriever, top_k: int = 10, llm=None):
        self.hybrid_retriever = hybrid_retriever
        self.top_k = top_k
        self._llm = llm

    def invoke(self, query: str) -> list[Document]:
        lang = detect_language(query)

        # 原始语言检索
        docs_original = self.hybrid_retriever.invoke(query)

        # 翻译语言检索
        try:
            translated = translate_query(query, lang, self._llm)
            docs_translated = self.hybrid_retriever.invoke(translated)
        except Exception:
            # 翻译失败降级为单路检索
            return docs_original[:self.top_k]

        # RRF 融合
        merged = _rrf_merge([docs_original, docs_translated])
        return merged[:self.top_k]

    def get_translated_query(self, query: str) -> tuple[str, str]:
        """返回 (原始query, 翻译query)，供调试或前端显示"""
        lang = detect_language(query)
        translated = translate_query(query, lang, self._llm)
        return query, translated


# ── 分库配对工具 ────────────────────────────────────────────────────────────

def find_partner_collection(collection_name: str) -> str | None:
    """
    按命名约定查找配对 collection，不存在则返回 None。

    约定：
      hydro_manual     ↔  hydro_manual_en
      investment_db    ↔  investment_db_en
      foo_en           ↔  foo
      （即：无 _en 后缀的视为 ZH 库，_en 后缀视为 EN 库）
    """
    import chromadb
    from config import CHROMA_PERSIST_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    existing = {c.name for c in client.list_collections()}

    if collection_name.endswith("_en"):
        # foo_en → 先找 foo_zh，再找 foo
        partner = collection_name[:-3] + "_zh"
        if partner not in existing:
            partner = collection_name[:-3]
    elif collection_name.endswith("_zh"):
        # foo_zh → 先找 foo_en
        partner = collection_name[:-3] + "_en"
    else:
        partner = collection_name + "_en"       # foo → foo_en

    return partner if partner in existing else None


def build_partner_retriever(partner_collection: str, top_k: int, embeddings=None):
    """给配对 collection 建一个轻量级 Dense-only retriever（不需要 BM25）"""
    from langchain_chroma import Chroma
    from chains.rag_chain import get_embeddings
    from config import CHROMA_PERSIST_DIR

    _emb = embeddings or get_embeddings()
    vs = Chroma(
        collection_name=partner_collection,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=_emb,
    )
    return vs.as_retriever(search_kwargs={"k": top_k})


# ── CrossCollectionRetriever ───────────────────────────────────────────────

class CrossCollectionRetriever:
    """
    分库跨语言检索器：在两个配对 collection（ZH + EN）之间做跨语言检索。

    使用场景：
      - ZH query 查 EN collection（中文问题，英文文档）
      - EN query 查 ZH collection（英文问题，中文文档）
      - 查询语言与文档语言不同时的根本解法

    工作流程：
      1. 检测 query 语言
      2. 原始 query → 搜原始 collection（hybrid）
      3. 翻译 query → 搜配对 collection（dense-only，轻量）
      4. RRF 跨库融合，彻底消除单库语言偏置

    用法：
        # 自动查找配对库
        retriever = CrossCollectionRetriever.from_collection(
            "hydro_manual", hybrid_retriever, top_k=10, llm=llm
        )
        docs = retriever.invoke("What is reactor temperature range?")
        # 同时搜 hydro_manual（ZH）和 hydro_manual_en（EN）
    """

    def __init__(self, primary_retriever, partner_retriever, top_k: int = 10, llm=None):
        self.primary_retriever = primary_retriever    # hybrid，搜原始库
        self.partner_retriever = partner_retriever    # dense-only，搜配对库
        self.top_k = top_k
        self._llm = llm

    @classmethod
    def from_collection(cls, collection_name: str, hybrid_retriever,
                        top_k: int = 10, llm=None, embeddings=None):
        """
        工厂方法：自动查找配对 collection，构建 CrossCollectionRetriever。
        找不到配对库则返回 None，调用方降级到 CrossLingualRetriever。
        """
        partner = find_partner_collection(collection_name)
        if partner is None:
            return None
        partner_ret = build_partner_retriever(partner, top_k, embeddings)
        return cls(hybrid_retriever, partner_ret, top_k, llm)

    def invoke(self, query: str) -> list[Document]:
        lang = detect_language(query)

        # 原始 query 搜原始 collection
        docs_primary = self.primary_retriever.invoke(query)

        # 翻译 query 搜配对 collection
        try:
            translated = translate_query(query, lang, self._llm)
            docs_partner = self.partner_retriever.invoke(translated)
        except Exception:
            return docs_primary[:self.top_k]

        # RRF 跨库融合（两路结果语言不同，不会互相竞争）
        merged = _rrf_merge([docs_primary, docs_partner])
        return merged[:self.top_k]
