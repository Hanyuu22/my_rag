"""
Level 2 — BGE Reranker 精排

初检（BM25 + Dense）召回 Top-K 之后，用 Cross-Encoder 对每个 (query, doc) 对
直接打分，把真正相关的推到最前。

Bi-Encoder（Embedding）vs Cross-Encoder（Reranker）：
- Bi-Encoder：query 和 doc 分别编码成向量，点积算相似度，快但粗糙
- Cross-Encoder：query 和 doc 拼在一起过模型，打一个精确相关性分数，慢但准

所以通常两段式：Bi-Encoder 粗检 Top-20 → Cross-Encoder 精排 → Top-3

用法：
    reranker = BGEReranker()
    ranked_docs = reranker.rerank(query, docs)
    # ranked_docs: List of (score, Document)，已按分数降序
"""

import time
from typing import List, Tuple

from langchain_core.documents import Document

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from config import RERANKER_MODEL_NAME, RERANKER_TOP_N


# 全局单例：模型只加载一次，所有 collection 共享
_reranker_instance: "BGEReranker | None" = None

def get_reranker(top_n: int = RERANKER_TOP_N) -> "BGEReranker":
    """获取 Reranker 单例，top_n 以第一次调用时的值为准，之后可通过实例动态设置。"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker(top_n=top_n)
    _reranker_instance.top_n = top_n   # 允许调用方动态调整 top_n
    return _reranker_instance


class BGEReranker:
    """
    BGE Reranker 封装（基于 FlagEmbedding）。

    Args:
        model_name: reranker 模型名，默认读 config
        top_n: 精排后保留几个，默认读 config
        use_fp16: 用半精度推理，省显存，精度基本无损
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        top_n: int = RERANKER_TOP_N,
        use_fp16: bool = True,
    ):
        from FlagEmbedding import FlagReranker

        print(f"加载 Reranker 模型（{model_name}）...")
        t0 = time.time()
        self.model = FlagReranker(model_name, use_fp16=use_fp16)
        self.top_n = top_n
        print(f"  Reranker 加载完成  耗时: {time.time()-t0:.1f}s")

    def rerank(
        self,
        query: str,
        docs: List[Document],
    ) -> List[Tuple[float, Document]]:
        """
        对 docs 按与 query 的相关性重新排序。

        Returns:
            List of (score, Document)，按 score 降序，只返回 top_n 个
        """
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.compute_score(pairs, normalize=True)  # 归一化到 0~1

        # 打包、排序、截断
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return scored[:self.top_n]

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """只返回排序后的 Document 列表（不带分数）"""
        return [doc for _, doc in self.rerank(query, docs)]
