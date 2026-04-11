"""
问答路由（SSE 流式输出）

POST /api/chat   发送问题，SSE 流式返回路由状态 + 答案 + 来源
"""

import sys
import json
import asyncio
from collections import OrderedDict
sys.path.insert(0, "/home/hanyuu/rag_project")

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# 组件单例缓存（LRU，最多保留 10 个 graph 防止内存无限增长）
_MAX_GRAPH_CACHE = 10
_graph_cache: OrderedDict = OrderedDict()
_graph_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    question: str
    collection: str
    extra_collections: list = []    # 辅助库列表，为空时走单库逻辑
    # 检索/生成参数
    retriever_top_k: int = 10
    reranker_top_n: int = 3
    score_threshold: float = 0.5
    max_retry: int = 2
    temperature: float = 0.1
    # 流程控制
    reranker_enabled: bool = True
    fallback_enabled: bool = True
    fallback_method: str = 'auto'   # 'auto' | 'llm' | 'web'
    tools_enabled: bool = False     # 是否启用 Function Calling 工具路由
    cross_lingual_enabled: bool = False  # 是否启用双语 query 跨语言检索


def _build_graph(collection_name: str, extra_collections: list = None,
                 top_k: int = 10, reranker_top_n: int = 3, temperature: float = 0.1):
    """构建并缓存 RAG graph，支持单库和多库两种模式"""
    from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from retrievers.reranker import get_reranker
    from graphs.rag_graph import build_rag_graph
    from langchain_openai import ChatOpenAI

    # 多库模式
    if extra_collections:
        from retrievers.multi_collection_retriever import build_multi_collection_retriever
        all_collections = [collection_name] + list(extra_collections)
        print(f"[multi-collection] 加载库：{all_collections}")
        hybrid_retriever, chunks = build_multi_collection_retriever(all_collections, top_k=top_k)
    else:
        # 单库模式（原有逻辑）
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from retrievers.hybrid_retriever import build_hybrid_retriever

        emb_model = get_collection_embedding_model(collection_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=get_embeddings(emb_model),
        )
        results = vectorstore._collection.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        hybrid_retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)

    # chunk_lookup：{chunk_id → Document}，用于 retrieve_node 邻居扩展
    # 多库模式下 chunks 来自多个库，合并后统一建 lookup
    chunk_lookup = {
        c.metadata["chunk_id"]: c
        for c in chunks
        if c.metadata.get("chunk_id")
    }

    reranker = get_reranker(top_n=reranker_top_n)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=temperature,
                     api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    return build_rag_graph(hybrid_retriever, reranker, llm, chunk_lookup=chunk_lookup)


def _get_or_build_graph(req: ChatRequest):
    # extra_collections 也纳入 cache key，避免单/多库命中同一缓存
    cache_key = (
        req.collection,
        tuple(sorted(req.extra_collections)),
        req.retriever_top_k,
        req.reranker_top_n,
        req.temperature,
    )
    if cache_key not in _graph_cache:
        # LRU 驱逐：超出上限时弹出最早未用的条目
        if len(_graph_cache) >= _MAX_GRAPH_CACHE:
            evicted_key, _ = _graph_cache.popitem(last=False)
            print(f"[graph_cache] 缓存已满，驱逐：{evicted_key}")
        _graph_cache[cache_key] = _build_graph(
            req.collection, req.extra_collections,
            req.retriever_top_k, req.reranker_top_n, req.temperature,
        )
    else:
        # 命中缓存：移到末尾（最近使用）
        _graph_cache.move_to_end(cache_key)
    return _graph_cache[cache_key]


def _invalidate_graph(collection_name: str):
    """知识库更新后清除包含该 collection 的所有缓存条目"""
    keys_to_drop = [k for k in _graph_cache if k[0] == collection_name]
    for k in keys_to_drop:
        _graph_cache.pop(k, None)


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    SSE 流式问答。

    事件格式：
      data: {"type": "status", "content": "路由中..."}
      data: {"type": "answer", "content": "答案文本"}
      data: {"type": "sources", "content": [...]}
      data: {"type": "done"}
      data: {"type": "error", "content": "错误信息"}
    """

    async def event_stream():
        def send(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        try:
            # 检查 collection 是否存在
            import chromadb
            from config import CHROMA_PERSIST_DIR
            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            try:
                col = client.get_collection(req.collection)
                if col.count() == 0:
                    yield send({"type": "error", "content": f"知识库 '{req.collection}' 为空"})
                    return
            except Exception:
                yield send({"type": "error", "content": f"知识库 '{req.collection}' 不存在"})
                return

            yield send({"type": "status", "content": "正在初始化..."})

            # ── 流式透传：asyncio.Queue 桥接后台线程与 SSE ────────────────
            loop = asyncio.get_event_loop()
            token_queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            def token_callback(token: str):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)

            def _run_and_signal():
                r = _run_graph(req, token_callback)
                loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)
                return r

            graph_future = loop.run_in_executor(None, _run_and_signal)

            # 消费 token 直到哨兵
            while True:
                item = await token_queue.get()
                if item is _SENTINEL:
                    break
                yield send({"type": "answer_token", "content": item})

            result = await graph_future

            if "error" in result:
                yield send({"type": "error", "content": result["error"]})
                return

            yield send({"type": "status", "content": f"路由：{result['route']}"})
            # answer_token 已流式发出；这里补发完整答案供不支持流式的客户端
            yield send({"type": "answer", "content": result["answer"]})

            # Function Calling 工具调用信息（供前端渲染气泡）
            if result.get("tool_name") and result["tool_name"] not in ("search_knowledge_base", "direct_answer"):
                yield send({"type": "tool_call", "content": {
                    "tool": result["tool_name"],
                    "args": result.get("tool_args", {}),
                }})

            if result.get("fallback_type"):
                yield send({"type": "fallback", "content": {
                    "type": result["fallback_type"],
                    "web_sources": result.get("web_sources", []),
                }})

            if result.get("sources"):
                yield send({"type": "sources", "content": result["sources"]})

            yield send({"type": "done"})

        except Exception as e:
            yield send({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _run_graph(req: ChatRequest, token_callback=None) -> dict:
    try:
        graph = _get_or_build_graph(req)
        result = graph.invoke({
            "question": req.question,
            "score_threshold": req.score_threshold,
            "max_retry_limit": req.max_retry,
            "reranker_enabled": req.reranker_enabled,
            "fallback_enabled": req.fallback_enabled,
            "fallback_method": req.fallback_method,
            "tools_enabled": req.tools_enabled,
            "cross_lingual_enabled": req.cross_lingual_enabled,
            "collection_name": req.collection,
            "token_callback": token_callback,
        })

        answer = result.get("answer", "")
        route = result.get("route", "retrieve")
        fallback_type = result.get("fallback_type", "")
        web_sources   = result.get("web_sources", [])
        tool_name     = result.get("tool_name", "")
        tool_args     = result.get("tool_args", {})
        docs = result.get("docs", [])
        scores = result.get("scores", [])

        sources = []
        for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
            sources.append({
                "index": i + 1,
                "score": score,
                "page": doc.metadata.get("page", ""),
                "content": doc.page_content[:150],
            })

        return {
            "answer": answer,
            "route": route,
            "fallback_type": fallback_type,
            "web_sources": web_sources,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "sources": sources,
        }
    except Exception as e:
        return {"error": str(e)}
