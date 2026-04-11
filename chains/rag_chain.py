"""
Level 1 基础 RAG 链

包含：
1. BGE Embedding 初始化
2. Chroma 向量库建库 / 加载
3. LCEL 基础 RAG 链（带溯源）

用法：
    # 首次建库
    build_vectorstore(chunks)

    # 问答
    chain = build_rag_chain()
    result = chain.invoke("你的问题")
    print(result["answer"])
    print(result["source_docs"])
"""

import threading
import itertools
import time
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_TOP_K,
)


# ── Embedding ─────────────────────────────────────────────────────────────

def _spinner(msg: str, stop_event: threading.Event):
    for ch in itertools.cycle("|/-\\"):
        if stop_event.is_set():
            break
        print(f"\r  {msg} {ch}", end="", flush=True)
        time.sleep(0.1)
    print(f"\r  {msg} 完成    ", flush=True)


# 支持多模型并存：{model_name: HuggingFaceEmbeddings}
_embeddings_cache: dict = {}


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """初始化 BGE Embedding 模型（本地），按模型名多单例复用"""
    name = model_name or EMBEDDING_MODEL_NAME
    if name in _embeddings_cache:
        return _embeddings_cache[name]

    short = name.split("/")[-1]
    print(f"加载 Embedding 模型 {short}...")
    stop = threading.Event()
    t = threading.Thread(target=_spinner, args=(f"{short} 加载中", stop))
    t.start()
    t0 = time.time()
    instance = HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": EMBEDDING_DEVICE, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    stop.set()
    t.join()
    print(f"  device: {instance._client.device}  耗时: {time.time()-t0:.1f}s")
    _embeddings_cache[name] = instance
    return instance


# ── Embedding Registry（每个 collection 用了哪个 embedding 模型） ──────────

import json as _json

_REGISTRY_PATH = str(CHROMA_PERSIST_DIR).rstrip("/") + "/embedding_registry.json"


def _load_registry() -> dict:
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except FileNotFoundError:
        return {}


def _save_registry(registry: dict):
    import os
    os.makedirs(str(CHROMA_PERSIST_DIR), exist_ok=True)
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        _json.dump(registry, f, ensure_ascii=False, indent=2)


def get_collection_embedding_model(collection_name: str) -> str:
    """返回该 collection 使用的 embedding 模型名，未记录则返回全局默认值"""
    return _load_registry().get(collection_name, EMBEDDING_MODEL_NAME)


def save_collection_embedding_model(collection_name: str, model_name: str):
    """记录该 collection 使用的 embedding 模型名"""
    registry = _load_registry()
    registry[collection_name] = model_name
    _save_registry(registry)


# ── 向量库 ────────────────────────────────────────────────────────────────

def _check_vectorstore_count(collection_name: str) -> int:
    """不加载 Embedding，直接用 chromadb 检查 collection 的文档数"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        col = client.get_collection(collection_name)
        return col.count()
    except Exception:
        return 0


def build_vectorstore(
    chunks: List[Document],
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_rebuild: bool = False,
    embedding_model: str = None,
) -> Chroma:
    """
    从 chunk 列表建库并持久化。
    - force_rebuild=False：collection 已有数据时直接加载（跳过 embedding）
    - force_rebuild=True：追加新 chunks 到已有 collection（多文件入库）
    - embedding_model：指定模型，None 则用 registry 记录值或全局默认
    """
    # 确定本次使用的模型
    model = embedding_model or get_collection_embedding_model(collection_name)

    if not force_rebuild:
        count = _check_vectorstore_count(collection_name)
        if count > 0:
            print(f"向量库已存在（{count} 条），加载 Embedding 供查询使用...")
            embeddings = get_embeddings(model)
            existing = Chroma(
                collection_name=collection_name,
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
            print(f"复用已有向量库（{count} 条向量，collection='{collection_name}'）")
            return existing

    embeddings = get_embeddings(model)

    print(f"向量化并追加入库（{len(chunks)} 个 chunk → '{collection_name}'）...")
    stop = threading.Event()
    t = threading.Thread(target=_spinner, args=("Chroma 建库中", stop))
    t.start()
    t0 = time.time()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    stop.set()
    t.join()
    print(f"  向量库建完：{vectorstore._collection.count()} 条向量  耗时: {time.time()-t0:.1f}s")

    # 记录该 collection 使用的模型（首次写入或与现有一致）
    save_collection_embedding_model(collection_name, model)
    return vectorstore


def load_vectorstore(
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """加载已有向量库（不重新 embedding）"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


# ── LLM ───────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """初始化 LLM（DashScope OpenAI-compatible 接口）"""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


# ── Prompt ────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个专业的知识库问答助手。"
        "请根据下方检索到的上下文回答用户问题。\n"
        "要求：\n"
        "- 只使用上下文中的信息作答，不要编造内容\n"
        "- 如果上下文中没有相关信息，直接说'根据现有资料无法回答'\n"
        "- 回答简洁、准确，必要时分点说明\n\n"
        "上下文：\n{context}"
    )),
    ("human", "{question}"),
])


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """把检索到的 Document 列表拼成字符串，带来源标注"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "")
        page_str = f" p.{page}" if page != "" else ""
        parts.append(f"[{i}] 来源：{source}{page_str}\n{doc.page_content}")
    return "\n\n".join(parts)


# ── LCEL 链 ───────────────────────────────────────────────────────────────

def build_rag_chain(collection_name: str = CHROMA_COLLECTION_NAME):
    """
    构建基础 RAG 链，返回带溯源的结果。

    输入：问题字符串
    输出：{"answer": str, "source_docs": List[Document]}

    链结构：
        retriever ──► format_docs ──┐
                                    ├──► prompt ──► llm ──► parser → answer
        RunnablePassthrough ────────┘ (question)
    """
    vectorstore = load_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )
    llm = get_llm()

    # 核心 LCEL 链
    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # 并行：同时拿答案和原始来源文档
    rag_chain_with_source = RunnableParallel(
        answer=answer_chain,
        source_docs=retriever,
    )

    return rag_chain_with_source
