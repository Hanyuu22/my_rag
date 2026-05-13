"""
Level 2 — Hybrid Retriever

BM25/ES（关键词）+ Chroma Dense（语义）→ EnsembleRetriever（RRF 融合）

为什么要混合？
- 纯 Dense：语义相近的能找到，但专有名词/型号/参数数字容易漏
- 纯 BM25：关键词精确，但同义词/换种说法就找不到
- 混合 + RRF：互补，召回率显著提升

BM25 冷启动优化（pickle 缓存）：
- 问题：BM25 是纯确定性算法，每次重建结果完全一样，但 jieba 分词耗时
- 解法：首次构建后 pickle 序列化到磁盘，重启直接加载（< 0.5s）
- 失效 key：collection_name + chunk 数量（chunk 数变化时自动重建）
- 缓存位置：data/bm25_cache/{collection_name}_{n_chunks}.pkl

ES vs 内存 BM25（当 ES 不可用时 pickle BM25 作为降级）：
- ES：持久化索引，冷启动消失，IK 中文分词，支持分布式
- pickle BM25：无额外服务依赖，单机部署首选

用法：
    # 优先用 ES，ES 不可用时走 pickle BM25 缓存
    retriever = build_hybrid_retriever(chunks, vectorstore, collection_name="gb_standards_512")

    # 强制用内存 BM25（不走缓存，如临时脚本）
    retriever = build_hybrid_retriever(chunks, vectorstore)
"""

import logging
import pickle
from pathlib import Path
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

_BM25_CACHE_DIR = Path("/home/hanyuu/rag_project/data/bm25_cache")


def _jieba_tokenize(text: str) -> List[str]:
    """jieba 分词，过滤单字和纯空白，提升 BM25 对中文的召回质量"""
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def _build_bm25(chunks: List[Document], k: int,
                collection_name: Optional[str] = None) -> BM25Retriever:
    """
    构建 BM25 检索器，collection_name 传入时启用 pickle 缓存。

    缓存 key：{collection_name}_{len(chunks)}.pkl
    失效条件：chunk 数量变化（新文档入库 / 删除文档）
    """
    if collection_name:
        _BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _BM25_CACHE_DIR / f"{collection_name}_{len(chunks)}.pkl"

        if cache_file.exists():
            try:
                r = pickle.loads(cache_file.read_bytes())
                r.k = k
                logger.info(f"BM25 从缓存加载：{cache_file.name}")
                return r
            except Exception as e:
                logger.warning(f"BM25 缓存读取失败，重建：{e}")

        logger.info(f"BM25 构建中（{len(chunks)} chunks）...")
        r = BM25Retriever.from_documents(chunks, preprocess_func=_jieba_tokenize)
        r.k = k

        try:
            cache_file.write_bytes(pickle.dumps(r))
            # 删除同一 collection 的旧版本缓存（chunk 数不同的文件）
            for old in _BM25_CACHE_DIR.glob(f"{collection_name}_*.pkl"):
                if old != cache_file:
                    old.unlink(missing_ok=True)
            logger.info(f"BM25 缓存已保存：{cache_file.name}")
        except Exception as e:
            logger.warning(f"BM25 缓存保存失败（不影响使用）：{e}")

        return r

    # 无 collection_name：直接构建，不缓存（评估脚本等临时场景）
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
        keyword_retriever = _build_bm25(chunks, k, collection_name=collection_name)
        logger.info("Hybrid Retriever：BM25 模式（pickle 缓存）")

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
