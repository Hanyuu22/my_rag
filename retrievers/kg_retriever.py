"""
知识图谱检索器 (KG Retriever)

构建实体-chunk 二部图：
  节点：chunk_id（文档节点）+ 实体名（实体节点）
  边：  chunk 包含该实体

检索流程：
  1. base_retriever 检索初始 docs
  2. 从查询中提取关键实体（jieba TF-IDF，可选加载领域词典）
  3. 图遍历：找到与查询实体共现的其他 chunk
  4. 合并去重，返回 top-k

实体抽取结果缓存到 data/kg_cache/{collection}_kg.json。
领域词典路径：data/kg_cache/{collection}_userdict.txt（由 scripts/build_kg_userdict.py 生成）
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, "/home/hanyuu/rag_project")

import jieba
import jieba.analyse
import networkx as nx
from langchain_core.documents import Document

CACHE_DIR = Path("/home/hanyuu/rag_project/data/kg_cache")


def _load_userdict(collection: str):
    """若存在领域词典则加载，让 jieba 识别专业术语。"""
    path = CACHE_DIR / f"{collection}_userdict.txt"
    if path.exists():
        jieba.load_userdict(str(path))
        print(f"[KG] 已加载领域词典：{path.name}")


def _extract_entities_local(text: str, topk: int = 8) -> List[str]:
    """jieba TF-IDF 关键词提取，过滤过短词。"""
    words = jieba.analyse.extract_tags(text[:400], topK=topk)
    return [w for w in words if len(w) >= 2]


class KGRetriever:
    """
    知识图谱增强检索器。

    Args:
        collection_name: 用于缓存 key
        chunks:          全量 Document 列表
        base_retriever:  底层 HybridRetriever
        llm:             保留参数（兼容调用方，已不使用）
        top_k:           返回文档数
    """

    def __init__(
        self,
        collection_name: str,
        chunks: List[Document],
        base_retriever,
        llm=None,
        top_k: int = 10,
        batch_size: int = 20,
    ):
        self.collection_name = collection_name
        self.chunks = chunks
        self.base_retriever = base_retriever
        self.top_k = top_k

        self.graph = nx.Graph()
        self.chunk_id_to_doc: Dict[str, Document] = {
            c.metadata.get("chunk_id", f"__idx_{i}"): c
            for i, c in enumerate(chunks)
        }
        _load_userdict(collection_name)
        self._load_or_build()

    # ── 缓存管理 ───────────────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{self.collection_name}_kg.json"

    def _load_or_build(self):
        cache = self._cache_path()
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("n_chunks") == len(self.chunks):
                    self._load_graph(data["edges"])
                    print(f"[KG] 从缓存加载图：{self.graph.number_of_nodes()} 节点，"
                          f"{self.graph.number_of_edges()} 边")
                    return
            except Exception as e:
                print(f"[KG] 缓存失效，重建: {e}")
        self._build_graph()

    def _load_graph(self, edges: List[dict]):
        for e in edges:
            self.graph.add_edge(e["chunk_id"], e["entity"],
                                weight=e.get("weight", 1))

    # ── 图构建 ─────────────────────────────────────────────────────────────

    def _build_graph(self):
        print(f"[KG] 构建图（jieba TF-IDF）：{len(self.chunks)} chunks")
        edges: List[dict] = []

        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.metadata.get("chunk_id", f"__idx_{i}")
            entities = _extract_entities_local(chunk.page_content)
            for entity in entities:
                self.graph.add_edge(chunk_id, entity, weight=1)
                edges.append({"chunk_id": chunk_id, "entity": entity, "weight": 1})

        if i % 100 == 99:
            print(f"[KG] 进度 {i + 1}/{len(self.chunks)}")

        self._cache_path().write_text(json.dumps({
            "n_chunks": len(self.chunks),
            "edges": edges,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[KG] 完成：{self.graph.number_of_nodes()} 节点，{self.graph.number_of_edges()} 边")

    # ── 检索 ───────────────────────────────────────────────────────────────

    def _extract_query_entities(self, query: str) -> List[str]:
        return _extract_entities_local(query, topk=6)

    def invoke(self, query: str) -> List[Document]:
        """base 检索 + 图邻居扩展，返回 top-k。"""
        base_docs = self.base_retriever.invoke(query)
        query_entities = self._extract_query_entities(query)

        expanded_ids: Set[str] = set()

        # 从查询实体出发，找包含这些实体的 chunk
        for entity in query_entities:
            if entity in self.graph:
                for neighbor in self.graph.neighbors(entity):
                    if neighbor in self.chunk_id_to_doc:
                        expanded_ids.add(neighbor)

        # 已检索 chunk 的图邻居 chunk（一跳）
        for doc in base_docs:
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self.graph:
                for entity_node in self.graph.neighbors(cid):
                    for chunk_neighbor in self.graph.neighbors(entity_node):
                        if chunk_neighbor in self.chunk_id_to_doc:
                            expanded_ids.add(chunk_neighbor)

        seen = {d.metadata.get("chunk_id") for d in base_docs}
        result = list(base_docs)
        for cid in expanded_ids:
            if cid not in seen:
                result.append(self.chunk_id_to_doc[cid])
                seen.add(cid)

        return result[:self.top_k]
