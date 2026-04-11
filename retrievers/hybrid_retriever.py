"""
Level 2 — Hybrid Retriever

BM25/ES（关键词）+ Chroma Dense（语义）→ EnsembleRetriever（RRF 融合）

为什么要混合？
- 纯 Dense：语义相近的能找到，但专有名词/型号/参数数字容易漏
- 纯 BM25：关键词精确，但同义词/换种说法就找不到
- 混合 + RRF：互补，召回率显著提升

ES vs 内存 BM25：
- 内存 BM25：每次服务重启需重建索引（6000 chunk ≈ 3~5s 冷启动）
- ES：索引持久化，冷启动消失；IK 中文分词比 jieba 更精准
- ES 不可用时自动降级到内存 BM25，无感知切换

用法：
    # 优先用 ES（ES 不可用时自动降级 BM25）
    retriever = build_hybrid_retriever(chunks, vectorstore, collection_name="gb_standards_512")

    # 强制用内存 BM25（如评估脚本临时使用）
    retriever = build_hybrid_retriever(chunks, vectorstore)
"""

import logging
from typing import List, Optional

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from config import RETRIEVER_TOP_K

logger = logging.getLogger(__name__)


def _jieba_tokenize(text: str) -> List[str]:
    """jieba 分词，过滤单字和纯空白，提升 BM25 对中文的召回质量"""
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def _build_bm25(chunks: List[Document], k: int) -> BM25Retriever:
    """内存 BM25（fallback）"""
    r = BM25Retriever.from_documents(chunks, preprocess_func=_jieba_tokenize)
    r.k = k
    return r


def build_hybrid_retriever(
    chunks: List[Document],
    vectorstore: Chroma,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    k: int = RETRIEVER_TOP_K,
    collection_name: Optional[str] = None,   # 传入则尝试用 ES
) -> EnsembleRetriever:
    """
    构建关键词 + Dense 混合检索器。

    collection_name 传入时优先用 ES 检索（持久化，无冷启动）；
    ES 不可用或 collection_name 为 None 时降级到内存 BM25。

    Args:
        chunks:          BM25 降级时需要的文档列表（ES 模式下不用）
        vectorstore:     已建好的 Chroma 向量库
        bm25_weight:     关键词路权重（RRF）
        dense_weight:    Dense 路权重（RRF）
        k:               每路各取 Top-K
        collection_name: 指定则尝试 ES，None 则直接用 BM25
    """
    # ── 关键词检索路：ES 优先，降级 BM25 ─────────────────────────────────
    keyword_retriever = None

    if collection_name:
        try:
            from retrievers.es_retriever import ESRetriever, es_available
            if es_available():
                keyword_retriever = ESRetriever(collection_name, k=k)
                logger.info(f"Hybrid Retriever：ES 模式（{collection_name}）")
            else:
                logger.info("ES 不可用，降级到内存 BM25")
        except Exception as e:
            logger.warning(f"ES 初始化失败（{e}），降级到内存 BM25")

    if keyword_retriever is None:
        keyword_retriever = _build_bm25(chunks, k)
        logger.info("Hybrid Retriever：内存 BM25 模式")

    # ── Dense 检索路 ──────────────────────────────────────────────────────
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # ── RRF 融合 ─────────────────────────────────────────────────────────
    return EnsembleRetriever(
        retrievers=[keyword_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
    )
