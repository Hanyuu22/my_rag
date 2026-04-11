"""
MultiCollectionRetriever — 跨多个 collection 并行检索，RRF 合并结果

流程：
    对每个 collection 各跑一次 Hybrid Retrieval（BM25 + Dense + RRF）
    → 多路结果再做一次 RRF 合并
    → 返回统一的 top-k 候选，送 Reranker 精排

用法：
    retriever = build_multi_collection_retriever(
        ["energy_zh", "energy_en"], top_k=10
    )
    docs = retriever.invoke("BP 2024年能源展望主要结论")
"""

import sys
from typing import List, Dict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import RETRIEVER_TOP_K


def _rrf_merge(results: List[List[Document]], k: int = 60, top_n: int = None) -> List[Document]:
    """
    Reciprocal Rank Fusion：合并多路检索结果。

    Args:
        results:  每路检索返回的文档列表
        k:        RRF 平滑参数（默认 60）
        top_n:    返回前 N 条，None 则返回全部

    Returns:
        按 RRF 分数降序排列的文档列表（去重）
    """
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for ranked_docs in results:
        for rank, doc in enumerate(ranked_docs):
            # 用 page_content 前 200 字作为去重 key（避免同一文档被不同库重复计分）
            key = doc.page_content[:200]
            scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    if top_n:
        sorted_keys = sorted_keys[:top_n]
    return [doc_map[k] for k in sorted_keys]


class MultiCollectionRetriever:
    """
    并行查询多个 collection，RRF 合并结果。
    接口与 EnsembleRetriever 一致，支持 .invoke(query)。
    """

    def __init__(self, retrievers: list, top_k: int = RETRIEVER_TOP_K):
        self.retrievers = retrievers
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        results = []
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as executor:
            futures = {executor.submit(r.invoke, query): r for r in self.retrievers}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[MultiCollectionRetriever] 某个子检索器失败，跳过：{e}")
        if not results:
            return []
        return _rrf_merge(results, top_n=self.top_k)


def build_multi_collection_retriever(
    collection_names: List[str],
    top_k: int = RETRIEVER_TOP_K,
) -> MultiCollectionRetriever:
    """
    为多个 collection 构建并行混合检索器。

    Args:
        collection_names: collection 名称列表
        top_k:            每个子检索器各取多少条，最终合并后也取 top_k

    Returns:
        MultiCollectionRetriever 实例
    """
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from retrievers.hybrid_retriever import build_hybrid_retriever
    from config import CHROMA_PERSIST_DIR

    retrievers = []
    all_chunks: List[Document] = []

    for name in collection_names:
        try:
            # 加载向量库
            model = get_collection_embedding_model(name)
            vectorstore = Chroma(
                collection_name=name,
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=get_embeddings(model),
            )
            # 还原 chunks 用于 BM25
            raw = vectorstore._collection.get(include=["documents", "metadatas"])
            chunks = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(raw["documents"], raw["metadatas"])
            ]
            if not chunks:
                print(f"[MultiCollectionRetriever] 警告：{name} 为空，跳过")
                continue

            retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)
            retrievers.append(retriever)
            all_chunks.extend(chunks)
            print(f"[MultiCollectionRetriever] 加载 {name}：{len(chunks)} chunks")
        except Exception as e:
            print(f"[MultiCollectionRetriever] 加载 {name} 失败，跳过：{e}")

    return MultiCollectionRetriever(retrievers, top_k=top_k), all_chunks
