"""
Elasticsearch 检索器

替代内存 BM25（rank_bm25），解决冷启动重建问题。
- ES 未运行时自动降级到 BM25（不影响现有流程）
- 入库时调用 index_chunks() 同步写 ES
- 检索时直接查 ES，无需每次重建索引

IK 分词：需安装 analysis-ik 插件
  若未安装，自动回退到 standard analyzer（中文效果差但不报错）

用法：
    from retrievers.es_retriever import ESRetriever, index_chunks_to_es, es_available

    if es_available():
        retriever = ESRetriever(collection_name, k=10)
    else:
        # 降级到 BM25
        ...
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ES_HOST = "http://localhost:9200"
ES_INDEX_PREFIX = "rag_"   # ES index 名 = rag_{collection_name}


# ── 连接单例 ──────────────────────────────────────────────────────────────

_es_client = None

def get_es():
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        _es_client = Elasticsearch(ES_HOST, request_timeout=10)
    return _es_client


def es_available() -> bool:
    """检查 ES 是否可用（连接+ping），不抛异常"""
    try:
        return get_es().ping()
    except Exception:
        return False


# ── IK 分词检测 ────────────────────────────────────────────────────────────

def _ik_available() -> bool:
    try:
        es = get_es()
        result = es.indices.analyze(body={"analyzer": "ik_max_word", "text": "测试"})
        return bool(result.get("tokens"))
    except Exception:
        return False


def _text_analyzer() -> str:
    """返回可用的中文 analyzer"""
    if _ik_available():
        return "ik_max_word"
    logger.warning("IK 分词插件未安装，回退到 standard analyzer（中文效果差）")
    return "standard"


# ── 索引管理 ──────────────────────────────────────────────────────────────

def _index_name(collection_name: str) -> str:
    return f"{ES_INDEX_PREFIX}{collection_name}"


def ensure_index(collection_name: str):
    """创建 ES 索引（若不存在）"""
    es   = get_es()
    name = _index_name(collection_name)
    if es.indices.exists(index=name):
        return

    analyzer = _text_analyzer()
    es.indices.create(
        index=name,
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "chinese": {
                            "type": analyzer if analyzer != "standard" else "standard"
                        }
                    }
                },
                "number_of_shards": 1,
                "number_of_replicas": 0,   # 单机不需要副本
            },
            "mappings": {
                "properties": {
                    "content":    {"type": "text", "analyzer": analyzer},
                    "chunk_id":   {"type": "keyword"},
                    "source":     {"type": "keyword"},
                    "page":       {"type": "integer"},
                    "node_type":  {"type": "keyword"},
                }
            }
        }
    )
    logger.info(f"ES 索引已创建：{name}（analyzer={analyzer}）")


def index_chunks_to_es(chunks: List[Document], collection_name: str):
    """
    将 chunks 批量写入 ES（入库时调用，和 Chroma 同步）。
    已有同 chunk_id 的文档会被覆盖（upsert by chunk_id）。
    """
    if not es_available():
        logger.warning("ES 不可用，跳过 ES 索引")
        return

    ensure_index(collection_name)
    es   = get_es()
    name = _index_name(collection_name)

    actions = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        doc  = {
            "content":   chunk.page_content,
            "chunk_id":  meta.get("chunk_id", ""),
            "source":    meta.get("source", meta.get("file_name", "")),
            "page":      meta.get("page", -1),
            "node_type": meta.get("node_type", "chunk"),
        }
        # 用 chunk_id 作为 ES 文档 ID（有 chunk_id 时），方便幂等写入
        doc_id = meta.get("chunk_id") or None
        actions.append((doc_id, doc))

    # 分批写入（避免单次请求过大）
    batch_size = 200
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        body  = []
        for doc_id, doc in batch:
            meta_action = {"index": {"_index": name}}
            if doc_id:
                meta_action["index"]["_id"] = doc_id
            body.append(meta_action)
            body.append(doc)
        es.bulk(body=body)

    logger.info(f"ES 索引完成：{name} ← {len(chunks)} 条")


def delete_index(collection_name: str):
    """删除 collection 对应的 ES 索引（删库时调用）"""
    try:
        get_es().indices.delete(index=_index_name(collection_name), ignore_unavailable=True)
    except Exception as e:
        logger.warning(f"删除 ES 索引失败：{e}")


def get_index_count(collection_name: str) -> int:
    """查询 ES 索引的文档数"""
    try:
        name = _index_name(collection_name)
        res  = get_es().count(index=name)
        return res.get("count", 0)
    except Exception:
        return 0


# ── 检索器 ────────────────────────────────────────────────────────────────

class ESRetriever:
    """
    封装 ES match query，接口与 BM25Retriever 兼容（都有 .invoke(query)）。

    analyzer 优先用 ik_max_word（需安装 IK 插件），
    未安装时用 standard（效果差但不报错）。
    """

    def __init__(self, collection_name: str, k: int = 10):
        self.index = _index_name(collection_name)
        self.k     = k
        self._analyzer = _text_analyzer()

    def invoke(self, query: str) -> List[Document]:
        try:
            es   = get_es()
            resp = es.search(
                index=self.index,
                body={
                    "query": {
                        "match": {
                            "content": {
                                "query":    query,
                                "analyzer": self._analyzer,
                                "operator": "or",   # 任意词命中即可
                            }
                        }
                    },
                    "size": self.k,
                    "_source": True,
                }
            )
            docs = []
            for hit in resp["hits"]["hits"]:
                src  = hit["_source"]
                meta = {
                    "chunk_id":  src.get("chunk_id", ""),
                    "source":    src.get("source", ""),
                    "page":      src.get("page", -1),
                    "node_type": src.get("node_type", "chunk"),
                    "_es_score": hit["_score"],
                }
                docs.append(Document(page_content=src["content"], metadata=meta))
            return docs
        except Exception as e:
            logger.warning(f"ES 检索失败，返回空列表：{e}")
            return []

    # 兼容 LangChain retriever 接口
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
