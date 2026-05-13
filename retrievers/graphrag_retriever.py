"""
GraphRAG 检索器（LightRAG 风格）

融合 Local Search + Global Search：
  Local：  从查询实体出发，找实体邻居 chunk（精准局部检索）
  Global： 从社区摘要出发，找相关社区对应的 chunk（宏观全局检索）

社区检测：对实体嵌入做 KMeans，同一聚类视为一个"社区"。
社区摘要：每个社区的代表 chunk 用 LLM 摘要。

缓存策略：
  data/graphrag_cache/{collection}_entities.json   实体图
  data/graphrag_cache/{collection}_communities.json 社区摘要
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional

sys.path.insert(0, "/home/hanyuu/rag_project")

import numpy as np
import networkx as nx
from langchain_core.documents import Document

CACHE_DIR = Path("/home/hanyuu/rag_project/data/graphrag_cache")

_ENTITY_PROMPT = """\
从以下文本提取5个以内关键实体（专业术语/设备/物质/组织），\
用逗号分隔，只输出实体名：
{text}"""

_COMMUNITY_PROMPT = """\
你是专业摘要助手。以下是同一主题的多个文档片段，请生成一段简洁的社区摘要（150字以内），\
描述这组文档的核心主题：
{context}"""


def _parse_entities(raw: str) -> List[str]:
    return [e.strip() for e in re.split(r"[,，、\n]", raw)
            if e.strip() and len(e.strip()) >= 2]


class GraphRAGRetriever:
    """
    GraphRAG 风格检索器：Local + Global 双路搜索。

    Args:
        collection_name:  Chroma collection 名称（缓存 key）
        chunks:           全量 Document 列表
        base_retriever:   底层 HybridRetriever（Local 路径用）
        embeddings:       LangChain embedding 实例
        llm:              LangChain LLM（实体抽取 + 社区摘要）
        top_k:            返回文档数
        n_communities:    社区数，None 时自动推断
        local_weight:     Local 结果权重（0~1），Global = 1 - local_weight
    """

    def __init__(
        self,
        collection_name: str,
        chunks: List[Document],
        base_retriever,
        embeddings,
        llm,
        top_k: int = 10,
        n_communities: Optional[int] = None,
        local_weight: float = 0.6,
    ):
        self.collection_name = collection_name
        self.chunks = chunks
        self.base_retriever = base_retriever
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.n_communities = n_communities
        self.local_weight = local_weight

        self.graph = nx.Graph()
        self.chunk_id_to_doc: Dict[str, Document] = {
            c.metadata.get("chunk_id", f"__idx_{i}"): c
            for i, c in enumerate(chunks)
        }
        self.community_docs: List[Document] = []
        self._community_embs: Optional[np.ndarray] = None

        self._load_or_build()

    # ── 缓存 ───────────────────────────────────────────────────────────────

    def _entity_cache(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{self.collection_name}_entities.json"

    def _community_cache(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{self.collection_name}_communities.json"

    def _load_or_build(self):
        ec, cc = self._entity_cache(), self._community_cache()
        loaded_entities = False
        if ec.exists():
            try:
                data = json.loads(ec.read_text(encoding="utf-8"))
                if data.get("n_chunks") == len(self.chunks):
                    for e in data["edges"]:
                        self.graph.add_edge(e["chunk_id"], e["entity"])
                    loaded_entities = True
                    print(f"[GraphRAG] 从缓存加载实体图："
                          f"{self.graph.number_of_nodes()} 节点")
            except Exception as e:
                print(f"[GraphRAG] 实体缓存失效: {e}")

        if not loaded_entities:
            self._build_entity_graph()

        if cc.exists():
            try:
                data = json.loads(cc.read_text(encoding="utf-8"))
                if data.get("n_chunks") == len(self.chunks):
                    self.community_docs = [
                        Document(page_content=d["content"], metadata=d["metadata"])
                        for d in data["communities"]
                    ]
                    print(f"[GraphRAG] 从缓存加载 {len(self.community_docs)} 个社区摘要")
                    return
            except Exception as e:
                print(f"[GraphRAG] 社区缓存失效: {e}")

        self._build_communities()

    # ── 图构建 ─────────────────────────────────────────────────────────────

    def _build_entity_graph(self):
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = PromptTemplate.from_template(_ENTITY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        print(f"[GraphRAG] 构建实体图：{len(self.chunks)} chunks")
        edges = []
        for i, chunk in enumerate(self.chunks):
            cid = chunk.metadata.get("chunk_id", f"__idx_{i}")
            try:
                raw = chain.invoke({"text": chunk.page_content[:400]})
                for entity in _parse_entities(raw):
                    self.graph.add_edge(cid, entity)
                    edges.append({"chunk_id": cid, "entity": entity})
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"[GraphRAG] 实体抽取 {i+1}/{len(self.chunks)}")

        self._entity_cache().write_text(json.dumps(
            {"n_chunks": len(self.chunks), "edges": edges},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[GraphRAG] 实体图完成：{self.graph.number_of_edges()} 边")

    def _build_communities(self):
        from sklearn.cluster import KMeans
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        if not self.chunks:
            return

        n = self.n_communities or max(2, min(8, len(self.chunks) // 10))
        n = min(n, len(self.chunks))
        print(f"[GraphRAG] 构建 {n} 个社区...")

        texts = [c.page_content[:512] for c in self.chunks]
        embs = np.array(self.embeddings.embed_documents(texts))
        labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(embs)

        prompt = PromptTemplate.from_template(_COMMUNITY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        # chunk_id → community_id 映射（用于 local search 的社区归属）
        self.chunk_community: Dict[str, int] = {}
        for i, chunk in enumerate(self.chunks):
            cid = chunk.metadata.get("chunk_id", f"__idx_{i}")
            self.chunk_community[cid] = int(labels[i])

        self.community_docs = []
        for cid in range(n):
            idx = [i for i, l in enumerate(labels) if l == cid]
            cluster_chunks = [self.chunks[i] for i in idx]
            combined = "\n\n".join(c.page_content[:250] for c in cluster_chunks[:6])
            try:
                summary = chain.invoke({"context": combined})
            except Exception as e:
                summary = combined[:200]
                print(f"[GraphRAG] 社区{cid} 摘要失败: {e}")

            self.community_docs.append(Document(
                page_content=summary,
                metadata={
                    "type": "graphrag_community",
                    "community_id": cid,
                    "source": f"[GraphRAG 社区 {cid}，{len(idx)} 个 chunk]",
                    "page": "",
                    "chunk_id": f"graphrag_community_{cid}",
                    "member_chunk_ids": json.dumps([
                        self.chunks[i].metadata.get("chunk_id", f"__idx_{i}")
                        for i in idx
                    ]),
                }
            ))
            print(f"[GraphRAG] 社区{cid} ✓ ({len(idx)} chunks)")

        self._community_cache().write_text(json.dumps({
            "n_chunks": len(self.chunks),
            "communities": [{"content": d.page_content, "metadata": d.metadata}
                            for d in self.community_docs],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[GraphRAG] {len(self.community_docs)} 个社区摘要已缓存")

    # ── 检索 ───────────────────────────────────────────────────────────────

    def _local_search(self, query: str) -> List[Document]:
        """Local: 实体图遍历 + base_retriever。"""
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = PromptTemplate.from_template(_ENTITY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({"text": query})
            query_entities = _parse_entities(raw)
        except Exception:
            query_entities = []

        expanded: Set[str] = set()
        for entity in query_entities:
            if entity in self.graph:
                for neighbor in self.graph.neighbors(entity):
                    if neighbor in self.chunk_id_to_doc:
                        expanded.add(neighbor)

        base_docs = self.base_retriever.invoke(query)
        seen = {d.metadata.get("chunk_id") for d in base_docs}
        result = list(base_docs)
        for cid in expanded:
            if cid not in seen:
                result.append(self.chunk_id_to_doc[cid])
                seen.add(cid)
        return result

    def _global_search(self, query: str) -> List[Document]:
        """Global: 找最相关社区，返回该社区的代表 chunk。"""
        if not self.community_docs:
            return []
        if self._community_embs is None:
            self._community_embs = np.array(
                self.embeddings.embed_documents([d.page_content for d in self.community_docs])
            )
        q_emb = np.array(self.embeddings.embed_query(query))
        sims = [
            float(np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e) + 1e-8))
            for e in self._community_embs
        ]
        top2_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:2]

        result: List[Document] = []
        for idx in top2_idx:
            result.append(self.community_docs[idx])
            member_ids = json.loads(
                self.community_docs[idx].metadata.get("member_chunk_ids", "[]")
            )
            for cid in member_ids[:3]:
                if cid in self.chunk_id_to_doc:
                    result.append(self.chunk_id_to_doc[cid])
        return result

    def invoke(self, query: str) -> List[Document]:
        """Local + Global 双路搜索，按权重合并返回 top-k。"""
        local_docs = self._local_search(query)
        global_docs = self._global_search(query)

        local_k = max(1, round(self.top_k * self.local_weight))
        global_k = self.top_k - local_k

        seen, result = set(), []
        for doc in local_docs[:local_k]:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)
        for doc in global_docs[:global_k]:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)

        return result[:self.top_k]
