"""
RAPTOR Retriever — Recursive Abstractive Processing for Tree-Organized Retrieval

构建两层树：
  Level 0 (叶节点)：原始 chunks（已在 Chroma，直接用 base_retriever 检索）
  Level 1 (根节点)：KMeans 聚类后用 LLM 生成的摘要节点

查询时：搜索原始 chunks + 摘要节点，合并返回 top-k。
摘要节点缓存到 data/raptor_cache/{collection}.json，chunk 数量不变时复用。
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, "/home/hanyuu/rag_project")

from langchain_core.documents import Document

CACHE_DIR = Path("/home/hanyuu/rag_project/data/raptor_cache")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class RaptorRetriever:
    """
    用法与标准 LangChain retriever 一致：retriever.invoke(query) -> List[Document]

    Args:
        collection_name: Chroma collection 名称（用于磁盘缓存 key）
        chunks:          该 collection 的全量 Document 列表
        base_retriever:  已构建好的 HybridRetriever（负责检索原始 chunks）
        embeddings:      LangChain embedding 实例（与 collection 同型）
        llm:             LangChain LLM 实例（用于生成聚类摘要）
        top_k:           最终返回文档数
        n_clusters:      聚类数，None 时自动推断（chunks 数 // 8，最少 2，最多 10）
    """

    def __init__(
        self,
        collection_name: str,
        chunks: List[Document],
        base_retriever,
        embeddings,
        llm,
        top_k: int = 10,
        n_clusters: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.chunks = chunks
        self.base_retriever = base_retriever
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.summary_docs: List[Document] = []
        self._summary_embs: Optional[np.ndarray] = None
        self._load_or_build()

    # ── 缓存管理 ───────────────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{self.collection_name}_raptor.json"

    def _load_or_build(self):
        cache = self._cache_path()
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("n_chunks") == len(self.chunks):
                    self.summary_docs = [
                        Document(page_content=d["content"], metadata=d["metadata"])
                        for d in data["summaries"]
                    ]
                    print(f"[RAPTOR] 从缓存加载 {len(self.summary_docs)} 个摘要节点")
                    return
            except Exception as e:
                print(f"[RAPTOR] 缓存失效，重新构建: {e}")
        self._build_tree()

    # ── 树构建 ─────────────────────────────────────────────────────────────

    def _build_tree(self):
        from sklearn.cluster import KMeans
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        if not self.chunks:
            return

        n = self.n_clusters or max(2, min(10, len(self.chunks) // 8))
        n = min(n, len(self.chunks))
        print(f"[RAPTOR] 构建树：{len(self.chunks)} chunks → {n} 聚类")

        texts = [c.page_content[:512] for c in self.chunks]
        print("[RAPTOR] 嵌入 chunks...")
        embs = np.array(self.embeddings.embed_documents(texts))

        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(embs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业文档摘要助手。将以下文档片段整合成一段简洁摘要（150字以内），保留关键信息和专业术语。"),
            ("human", "{context}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        self.summary_docs = []
        for cid in range(n):
            idx = [i for i, l in enumerate(labels) if l == cid]
            cluster_chunks = [self.chunks[i] for i in idx]
            combined = "\n\n".join(c.page_content[:300] for c in cluster_chunks[:8])
            try:
                summary = chain.invoke({"context": combined})
            except Exception as e:
                summary = combined[:200]
                print(f"[RAPTOR] 簇{cid} 摘要失败，截断代替: {e}")

            self.summary_docs.append(Document(
                page_content=summary,
                metadata={
                    "type": "raptor_summary",
                    "cluster_id": cid,
                    "source": f"[RAPTOR 聚类 {cid}，{len(idx)} 个 chunk]",
                    "page": "",
                    "chunk_id": f"raptor_summary_{cid}",
                }
            ))
            print(f"[RAPTOR] 簇{cid} ✓ ({len(idx)} chunks)")

        self._cache_path().write_text(json.dumps({
            "n_chunks": len(self.chunks),
            "summaries": [{"content": d.page_content, "metadata": d.metadata}
                          for d in self.summary_docs],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RAPTOR] 完成，{len(self.summary_docs)} 个摘要节点已缓存")

    # ── 检索 ───────────────────────────────────────────────────────────────

    def invoke(self, query: str) -> List[Document]:
        """搜索原始 chunks（via base_retriever）+ 摘要节点，合并去重返回 top-k。"""
        original_docs = self.base_retriever.invoke(query)

        top_summaries: List[Document] = []
        if self.summary_docs:
            if self._summary_embs is None:
                self._summary_embs = np.array(
                    self.embeddings.embed_documents([d.page_content for d in self.summary_docs])
                )
            q_emb = np.array(self.embeddings.embed_query(query))
            scored = sorted(
                zip(self._summary_embs, self.summary_docs),
                key=lambda x: -_cosine_sim(q_emb, x[0]),
            )
            top_summaries = [d for _, d in scored[:3]]

        seen, result = set(), []
        for doc in top_summaries + original_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)

        return result[:self.top_k]
