"""
Level 7 — LangGraph 多步 RAG（含 Function Calling）

节点：
  router        → LLM 路由：tools_enabled=True 时用 bind_tools 选工具，否则字符串分类
  retrieve      → Hybrid + Reranker 检索（Level 2 组件）
  evaluate      → 评估检索质量，决定生成 / 改写 / fallback
  rewrite       → LLM 改写查询（换角度重试，最多 2 次）
  generate      → LLM 生成最终答案（带页码溯源）
  direct        → 无需检索，直接由 LLM 回答
  tool_executor → 执行 analyze_process_data / web_search 工具调用
  fallback      → 重试耗尽后降级（Tavily 网络搜索 或 LLM 自身知识）

流程图（tools_enabled=True）：
  START
    │
    ▼
  [router/bind_tools] ── "retrieve" ──→ [retrieve] ──→ [evaluate]
     │                                                       │
     ├─ "direct"  ──→ [direct]           top1≥0.5 → [generate]
     └─ "tool"    ──→ [tool_executor]    不足 → [rewrite] ──→ [retrieve]
                           │             耗尽 → [fallback]
                           ▼
                          END
"""

import sys
import json
sys.path.insert(0, "/home/hanyuu/rag_project")

import re
from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from chains.rag_chain import format_docs, get_llm, RAG_PROMPT

# ── 配置 ──────────────────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.5   # top1_score 低于此值认为检索不充分
MAX_RETRY = 2           # 最多改写查询并重试几次


# ── State ─────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str           # 原始问题，全程不变
    query: str              # 当前检索用的 query（可能被改写）
    history: List[dict]     # 对话历史 [{"role": "user"|"assistant", "content": "..."}]
    docs: List[Document]    # 精排后的 Top-N 文档
    scores: List[float]     # 精排分数列表
    answer: str             # 最终答案
    retry: int              # 已重试次数
    route: str              # "retrieve" | "direct" | "tool"
    fallback_type: str      # "" | "llm" | "web"（非空表示走了 fallback 路由）
    web_sources: List[dict] # Tavily 搜索结果列表（title/url/content）
    # 可由调用方覆盖的运行时参数（不传则使用模块级默认值）
    score_threshold: float  # top1_score 低于此值触发改写
    max_retry_limit: int    # 最多改写几次
    reranker_enabled: bool  # 是否启用 BGE Reranker
    fallback_enabled: bool  # 是否启用 Fallback 路由
    fallback_method: str    # "auto" | "llm" | "web"
    # Function Calling 相关（tools_enabled=True 时由 router 填充）
    tools_enabled: bool     # 是否启用 bind_tools 路由（默认 False，保持向后兼容）
    tool_name: str          # 工具名称（analyze_process_data / web_search）
    tool_args: dict         # 工具参数
    tool_result: str        # 工具执行结果
    # 跨语言检索相关（cross_lingual_enabled=True 时启用）
    cross_lingual_enabled: bool  # 是否启用跨语言检索
    translated_query: str        # 翻译后的 query（供前端显示）
    collection_name: str         # 当前检索的 collection 名（用于查找配对库）
    # 高级检索模式（互斥，优先级：graphrag > kg > raptor > hybrid）
    raptor_enabled: bool         # RAPTOR 层级摘要树检索
    kg_enabled: bool             # 知识图谱实体扩展检索
    graphrag_enabled: bool       # GraphRAG Local+Global 双路检索
    # Query 拆解（多跳复杂问题）
    decompose_enabled: bool      # 是否启用 query 拆解（默认 True）
    sub_queries: List[str]       # 拆解后的子问题列表（供前端展示）
    # 父子检索（仅对有 parent_chunk_id 元数据的库有效，旧库自动跳过）
    parent_child_enabled: bool
    # 流式输出回调（token_callback(token: str) → None，None 时退化为批量模式）
    token_callback: Any


# ── Prompt ────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个问题路由分类器。你的任务是判断用户的问题是否需要查询知识库。\n\n"
     "判断规则（遇到模糊情况，优先选 retrieve）：\n\n"
     "输出 retrieve 的情况：\n"
     "- 询问特定人物、机构、公司、项目的具体信息（如：XX是谁、XX的情况怎么样）\n"
     "- 询问具体数据、参数、金额、时间、地点等事实性内容\n"
     "- 询问某个领域的专业知识、规程、操作步骤、技术细节\n"
     "- 询问文件、报告、记录中的内容\n"
     "- 任何需要查阅资料才能准确回答的问题\n\n"
     "输出 direct 的情况（必须非常明确才选）：\n"
     "- 纯粹的问候或闲聊（你好、谢谢、今天天气怎么样）\n"
     "- 让你介绍你自己\n"
     "- 纯数学计算（1+1等于几）\n\n"
     "只输出一个词：retrieve 或 direct，不要任何解释。"),
    ("human", "{question}"),
])

# bind_tools 模式下的路由 Prompt（让 LLM 选择工具）
ROUTER_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个智能助手。根据用户问题，调用最合适的工具来回答。\n\n"
     "工具选择指南：\n"
     "- search_knowledge_base：查询专业知识库（工艺规程、操作步骤、投资机构/项目信息等）\n"
     "- analyze_process_data：对装置 DCS 时序数据做数值统计分析（均值、极值、趋势等）\n"
     "- web_search：搜索知识库未覆盖的外部信息或最新动态\n"
     "- direct_answer：直接回答问候、闲聊、通用常识问题，无需任何工具\n\n"
     "遇到专业性或查询类问题优先选 search_knowledge_base；"
     "涉及具体数值计算或趋势分析才选 analyze_process_data。"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史和用户最新问题，将最新问题改写为一个无需上下文就能独立理解的问题。\n"
     "如果最新问题本身已经完整清晰（不依赖历史），直接原文返回，不做任何修改。\n"
     "只输出改写后的问题，不要任何解释或前缀。"),
    ("human", "对话历史：\n{chat_history}\n\n最新问题：{question}"),
])

GENERATE_WITH_HISTORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个专业的知识库问答助手。"
        "请根据下方检索到的上下文回答用户问题。\n"
        "要求：\n"
        "- 只使用上下文中的信息作答，不要编造内容\n"
        "- 如果上下文中没有相关信息，直接说'根据现有资料无法回答'\n"
        "- 回答简洁、准确，必要时分点说明\n\n"
        "【对话历史】\n{chat_history}\n\n"
        "【参考文档】\n{context}"
    )),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个检索查询优化专家。当前检索质量不足，需要改写查询来提升召回效果。\n\n"
     "改写策略：\n"
     "1. 提取问题的核心实体和关键词（人名、机构名、项目名、专业术语等），"
     "用这些关键词重新组织查询\n"
     "2. 将模糊的问题具体化（如把'情况怎么样'改为询问具体的数据、状态、特征）\n"
     "3. 尝试同义替换或扩展（如缩写↔全称、简称↔正式名称）\n"
     "4. 如果原问题涉及多个方面，拆分为最核心的单一查询\n"
     "5. 保持查询语言与原始问题一致（中文问题用中文改写）\n\n"
     "只输出改写后的查询语句，不要任何解释或前缀。"),
    ("human",
     "原始问题：{question}\n"
     "上一次查询：{query}\n"
     "检索最高相关分：{top_score:.3f}（阈值 0.5，未达标）\n\n"
     "请改写查询："),
])

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个智能助手，请直接、简洁地回答用户的问题。"
     "如果是问候就友好回应，如果是通用问题就给出准确答案。\n\n"
     "注意：本系统基于 RAG（检索增强生成），每次只检索少量相关文档片段（Top-K）。"
     "如果用户询问「共有多少条记录」「列出所有XX」等需要遍历全库的问题，"
     "请明确告知系统无法统计全量数据，只能回答关于具体内容的问题。"),
    ("human", "{question}"),
])

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业助手。用户的问题在当前知识库中未能检索到足够相关的内容。\n"
     "请根据你自身的通用知识尽力回答，回答时不要提及'知识库'或'检索'等字眼，"
     "直接给出你所了解的信息即可。"),
    ("human", "{question}"),
])


# ── 节点函数 ───────────────────────────────────────────────────────────────

def _format_history(history: List[dict], max_turns: int = 3) -> str:
    """将历史消息格式化为文本，最多保留最近 max_turns 轮（2条/轮）"""
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{role}：{msg.get('content', '')}")
    return "\n".join(lines)


def condense_node(state: RAGState, llm=None) -> dict:
    """
    问题压缩节点：将追问改写为独立的完整问题，改善检索质量。
    无对话历史时直接跳过（零开销）。
    """
    history = state.get("history") or []
    if not history:
        return {}
    _llm = llm or get_llm()
    chain = CONDENSE_PROMPT | _llm | StrOutputParser()
    condensed = chain.invoke({
        "chat_history": _format_history(history),
        "question": state["question"],
    }).strip()
    # 只更新 query（用于检索），question 保持原始问题不变
    return {"query": condensed}


def router_node(state: RAGState, llm=None) -> dict:
    """
    路由节点：
      tools_enabled=True  → bind_tools 模式，LLM 选择工具（search_kb / analyze / web / direct）
      tools_enabled=False → 字符串分类模式（兼容旧行为）
    """
    _llm = llm or get_llm()

    if state.get("tools_enabled", False):
        return _router_with_tools(state, _llm)
    else:
        return _router_string(state, _llm)


_AGGREGATION_RE = re.compile(
    r'(一共|总共|共有?|总计)[多少几个家条份]'
    r'|[多少几](个|家|只|条|份|种)'
    r'|列出所有|列举所有|所有的?\S+有哪些'
    r'|统计.{0,10}(数量|个数|多少|几个|几家)',
    re.IGNORECASE
)


def _router_string(state: RAGState, llm) -> dict:
    """原有字符串分类路由（兼容模式）"""
    question = state["question"]
    # 聚合/计数类问题 → 直接回答（RAG 只看 top-k，无法全库计数）
    if _AGGREGATION_RE.search(question):
        return {"route": "direct", "query": question, "tool_name": "", "tool_args": {}}
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()
    route = "retrieve" if "retrieve" in result else "direct"
    return {"route": route, "query": question, "tool_name": "", "tool_args": {}}


def _router_with_tools(state: RAGState, llm) -> dict:
    """bind_tools 路由：LLM 选择工具并提取参数"""
    from tools.tool_definitions import ALL_TOOLS
    from pydantic import BaseModel, Field
    from langchain_core.tools import tool as lc_tool

    class _DirectInput(BaseModel):
        reason: str = Field(default="闲聊或通用问题")

    @lc_tool(args_schema=_DirectInput)
    def direct_answer(reason: str) -> str:
        """直接回答问候、闲聊、通用常识，无需检索任何工具"""
        return reason

    llm_with_tools = llm.bind_tools(ALL_TOOLS + [direct_answer])
    msg = llm_with_tools.invoke(
        ROUTER_TOOL_PROMPT.format_messages(question=state["question"])
    )

    if not getattr(msg, "tool_calls", None):
        # LLM 未调用工具 → 默认检索
        return {"route": "retrieve", "query": state["question"], "tool_name": "", "tool_args": {}}

    tc = msg.tool_calls[0]
    tool_name: str = tc["name"]
    tool_args: dict = tc.get("args", {})

    if tool_name == "search_knowledge_base":
        return {
            "route": "retrieve",
            "query": tool_args.get("query", state["question"]),
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
    elif tool_name == "direct_answer":
        return {"route": "direct", "query": state["question"], "tool_name": tool_name, "tool_args": tool_args}
    else:
        # analyze_process_data / web_search → 走 tool_executor
        return {
            "route": "tool",
            "query": tool_args.get("query", state["question"]),
            "tool_name": tool_name,
            "tool_args": tool_args,
        }


def retrieve_node(state: RAGState, hybrid_retriever=None, reranker=None, llm=None, chunk_lookup: dict = None) -> dict:
    """
    调用 Hybrid Retriever，可选：
      - BGE Reranker 精排
      - 跨语言双语 query 检索（cross_lingual_enabled=True）
    """
    query = state["query"]
    translated_query = ""

    if state.get("cross_lingual_enabled", False):
        from retrievers.cross_lingual_retriever import (
            CrossLingualRetriever, CrossCollectionRetriever,
            detect_language, translate_query,
        )
        from retrievers.multi_collection_retriever import MultiCollectionRetriever
        _llm = llm or get_llm()
        collection_name = state.get("collection_name", "")

        # 已经是多库检索器时，直接双 query（避免重复搜索配对库）
        if isinstance(hybrid_retriever, MultiCollectionRetriever):
            cross_ret = CrossLingualRetriever(hybrid_retriever, llm=_llm)
        elif collection_name:
            # 单库时优先尝试分库路由（找配对库），找不到则降级双 query
            cross_ret = CrossCollectionRetriever.from_collection(
                collection_name, hybrid_retriever, top_k=10, llm=_llm
            )
            if cross_ret is None:
                cross_ret = CrossLingualRetriever(hybrid_retriever, llm=_llm)
        else:
            cross_ret = CrossLingualRetriever(hybrid_retriever, llm=_llm)

        raw_docs = cross_ret.invoke(query)

        # 记录翻译后的 query 供状态追踪
        try:
            lang = detect_language(query)
            translated_query = translate_query(query, lang, _llm)
        except Exception:
            pass
    else:
        raw_docs = hybrid_retriever.invoke(query)

    if state.get("reranker_enabled", True):
        scored = reranker.rerank(query, raw_docs)
        docs = [doc for _, doc in scored]
        scores = [round(s, 3) for s, _ in scored]
    else:
        docs = raw_docs
        scores = [1.0] * len(raw_docs)

    # ── 父子检索：命中 child → 替换为 parent（context 更完整）────────────
    # 无 parent_chunk_id 的旧库 chunk 直接跳过，不报错
    if state.get("parent_child_enabled", False) and chunk_lookup:
        seen_pids: set = set()
        new_docs: List[Document] = []
        new_scores: List[float] = []
        for doc, score in zip(docs, scores):
            pid = doc.metadata.get("parent_chunk_id", "")
            if pid:
                if pid not in seen_pids:
                    parent = chunk_lookup.get(pid)
                    if parent:
                        new_docs.append(parent)
                        new_scores.append(score)
                        seen_pids.add(pid)
                    else:
                        # parent 未在 lookup 中（异常情况），保留 child
                        new_docs.append(doc)
                        new_scores.append(score)
            else:
                # 旧库 chunk（无 parent_chunk_id），直接保留
                new_docs.append(doc)
                new_scores.append(score)
        if new_docs:
            docs, scores = new_docs, new_scores

    # ── 邻居扩展：用 prev/next_chunk_id 补充上下文 ────────────────────────
    # 父子模式已替换为 parent，邻居扩展在 parent 模式下跳过（避免噪声）
    elif chunk_lookup:
        seen_ids = {d.metadata.get("chunk_id") for d in docs if d.metadata.get("chunk_id")}
        extra = []
        for doc in docs:
            for id_key in ("prev_chunk_id", "next_chunk_id"):
                neighbor_id = doc.metadata.get(id_key, "")
                if neighbor_id and neighbor_id not in seen_ids:
                    neighbor = chunk_lookup.get(neighbor_id)
                    if neighbor:
                        extra.append(neighbor)
                        seen_ids.add(neighbor_id)
        docs = docs + extra
        scores = scores + [0.0] * len(extra)

    return {"docs": docs, "scores": scores, "translated_query": translated_query}


def rewrite_node(state: RAGState, llm=None) -> dict:
    """改写查询，换角度重新检索"""
    _llm = llm or get_llm()
    chain = REWRITE_PROMPT | _llm | StrOutputParser()
    top_score = state["scores"][0] if state.get("scores") else 0.0
    new_query = chain.invoke({
        "question": state["question"],
        "query": state["query"],
        "top_score": top_score,
    }).strip()
    return {
        "query": new_query,
        "retry": state.get("retry", 0) + 1,
    }


def _stream_or_invoke(prompt, llm, inputs: dict, callback) -> str:
    """统一的流式/批量调用辅助函数。callback 非 None 时逐 token 回调并返回完整文本。"""
    if callback is None:
        return (prompt | llm | StrOutputParser()).invoke(inputs)
    full = ""
    for chunk in (prompt | llm).stream(inputs):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            callback(token)
            full += token
    return full


def generate_node(state: RAGState, llm=None) -> dict:
    """用精排后的文档生成最终答案，有对话历史时带入上下文"""
    _llm = llm or get_llm()
    cb = state.get("token_callback")
    history = state.get("history") or []
    if history:
        answer = _stream_or_invoke(
            GENERATE_WITH_HISTORY_PROMPT, _llm,
            {
                "context": format_docs(state["docs"]),
                "question": state["question"],
                "chat_history": _format_history(history),
            },
            cb,
        )
    else:
        answer = _stream_or_invoke(
            RAG_PROMPT, _llm,
            {"context": format_docs(state["docs"]), "question": state["question"]},
            cb,
        )
    return {"answer": answer}


def direct_node(state: RAGState, llm=None) -> dict:
    """无需检索，直接回答，有对话历史时带入上下文"""
    _llm = llm or get_llm()
    cb = state.get("token_callback")
    history = state.get("history") or []
    if history:
        history_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个智能助手，请直接、简洁地回答用户的问题。\n\n"
             "注意：本系统基于 RAG（检索增强生成），每次只检索少量相关文档片段（Top-K）。"
             "如果用户询问「共有多少条记录」「列出所有XX」等需要遍历全库的问题，"
             "请明确告知系统无法统计全量数据，只能回答关于具体内容的问题。\n\n"
             "【对话历史】\n{chat_history}"),
            ("human", "{question}"),
        ])
        answer = _stream_or_invoke(
            history_prompt, _llm,
            {"question": state["question"], "chat_history": _format_history(history)},
            cb,
        )
    else:
        answer = _stream_or_invoke(DIRECT_PROMPT, _llm, {"question": state["question"]}, cb)
    return {"answer": answer}


def fallback_node(state: RAGState, llm=None) -> dict:
    """检索质量不足且重试耗尽：优先走 Tavily 网络搜索，失败则用 LLM 自身知识"""
    _llm = llm or get_llm()
    question = state["question"]

    method = state.get("fallback_method", "auto")

    cb = state.get("token_callback")

    # ── 方案B：Tavily 网络搜索 ──────────────────────────────────────────
    if method in ("auto", "web"):
        try:
            from tavily import TavilyClient
            from config import TAVILY_API_KEY
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(query=question, max_results=3, search_depth="basic")
            results = resp.get("results", [])
            if results:
                context = "\n\n".join(
                    f"[来源{i+1}] {r.get('title','')}\n{r.get('content','')}"
                    for i, r in enumerate(results)
                )
                web_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "你是一个专业助手。以下是通过网络搜索获取的相关资料，"
                     "请基于这些资料回答用户的问题，保持客观准确。"),
                    ("human", "参考资料：\n{context}\n\n问题：{question}"),
                ])
                answer = _stream_or_invoke(web_prompt, _llm, {"context": context, "question": question}, cb)
                web_sources = [
                    {"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in results
                ]
                return {"answer": answer, "fallback_type": "web", "web_sources": web_sources}
        except Exception:
            pass  # Tavily 失败，降级到方案A

    # ── 方案A：LLM 自身知识 ─────────────────────────────────────────────
    answer = _stream_or_invoke(FALLBACK_PROMPT, _llm, {"question": question}, cb)
    return {"answer": answer, "fallback_type": "llm", "web_sources": []}


def tool_executor_node(state: RAGState, llm=None) -> dict:
    """
    执行 analyze_process_data 或 web_search 工具调用。
    结果直接作为 answer 返回，不再走检索流程。
    """
    _llm = llm or get_llm()
    tool_name = state.get("tool_name", "")
    tool_args = state.get("tool_args", {})
    question = state["question"]

    if tool_name == "analyze_process_data":
        from tools.data_analyzer import analyze_process_data
        result = analyze_process_data(tool_args.get("query", question))
        return {"answer": result, "tool_result": result}

    if tool_name == "web_search":
        try:
            from tavily import TavilyClient
            from config import TAVILY_API_KEY
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(query=tool_args.get("query", question), max_results=3, search_depth="basic")
            results = resp.get("results", [])
            if results:
                context = "\n\n".join(
                    f"[来源{i+1}] {r.get('title', '')}\n{r.get('content', '')}"
                    for i, r in enumerate(results)
                )
                web_sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]
                web_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是专业助手。基于以下网络搜索结果回答问题，保持客观准确。"),
                    ("human", "参考资料：\n{context}\n\n问题：{question}"),
                ])
                answer = (web_prompt | _llm | StrOutputParser()).invoke(
                    {"context": context, "question": question}
                )
                return {"answer": answer, "fallback_type": "web", "web_sources": web_sources, "tool_result": answer}
        except Exception as e:
            return {"answer": f"网络搜索失败：{e}", "tool_result": ""}

    return {"answer": f"未知工具：{tool_name}", "tool_result": ""}


# ── Query 拆解节点 ─────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "判断以下问题是否是需要分解成多个子问题才能完整回答的复杂问题。\n\n"
     "【复杂问题特征】满足任意一条即判定为复杂：\n"
     "- 涉及两个或以上独立话题/实体，需分别查找再对比综合\n"
     "  （例：'A和B有什么区别'、'X规定和Y操作是否矛盾'）\n"
     "- 需要先获取某个结论，再用该结论查询另一件事（多跳推理）\n"
     "  （例：'投诉最多的问题，和产品路线有没有冲突'）\n"
     "- 问题中含有明确的对比/关联词：对比、区别、矛盾、冲突、关系、影响、差异\n\n"
     "【简单问题特征】下列情况不要拆解：\n"
     "- 单一事实查询（某参数是多少、某规程是什么）\n"
     "- 虽然较长但只围绕一个核心问题\n\n"
     "输出严格 JSON，不要任何解释：\n"
     "不需拆解：{{\"is_complex\": false, \"sub_queries\": [\"原问题\"]}}\n"
     "需要拆解：{{\"is_complex\": true, \"sub_queries\": [\"子问题1\", \"子问题2\"]}}"),
    ("human", "问题：{question}"),
])


def decompose_node(state: RAGState, hybrid_retriever=None, reranker=None, llm=None) -> dict:
    """
    Query 拆解节点。

    decompose_enabled=False 或判断为简单问题时：原样传递，后续走 retrieve_node。
    判断为复杂问题时：拆分为子问题分别检索，RRF 合并后精排，直接填充 docs/scores，
    后续跳过 retrieve_node 直接进入 evaluate_decision。
    """
    if not state.get("decompose_enabled", True):
        return {"sub_queries": [], "decompose_route": "retrieve"}

    _llm = llm or get_llm()
    question = state.get("query") or state["question"]

    # ── LLM 判断是否需要拆解 ───────────────────────────────────────────────
    try:
        raw = (_llm | StrOutputParser()).invoke(
            DECOMPOSE_PROMPT.format_messages(question=question)
        )
        # 提取 JSON（有时 LLM 会在 JSON 外包一段说明文字）
        import re as _re
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        is_complex = parsed.get("is_complex", False)
        sub_queries = parsed.get("sub_queries", [question])
    except Exception as e:
        print(f"[decompose] 判断失败，降级为单 query：{e}")
        return {"sub_queries": [], "decompose_route": "retrieve"}

    if not is_complex or len(sub_queries) <= 1:
        return {"sub_queries": [], "decompose_route": "retrieve"}

    print(f"[decompose] 拆解为 {len(sub_queries)} 个子问题：{sub_queries}")

    # ── 多路检索 ───────────────────────────────────────────────────────────
    all_docs: List[Document] = []
    seen_keys: set = set()
    for sq in sub_queries:
        try:
            docs = hybrid_retriever.invoke(sq)
            for d in docs:
                key = d.page_content[:80]
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_docs.append(d)
        except Exception as e:
            print(f"[decompose] 子 query '{sq}' 检索失败：{e}")

    if not all_docs:
        return {"sub_queries": sub_queries, "decompose_route": "retrieve"}

    # ── Reranker 精排（用原始问题打分）────────────────────────────────────
    if state.get("reranker_enabled", True) and reranker and all_docs:
        scored = reranker.rerank(question, all_docs)
        docs = [doc for _, doc in scored]
        scores = [round(s, 3) for s, _ in scored]
    else:
        docs = all_docs
        scores = [1.0] * len(all_docs)

    return {
        "docs": docs,
        "scores": scores,
        "sub_queries": sub_queries,
        "decompose_route": "generate",   # 已有 docs，跳过 retrieve_node
    }


def decompose_decision(state: RAGState) -> str:
    """decompose_node 之后的路由：已拆解 → generate；未拆解 → retrieve"""
    return state.get("decompose_route", "retrieve")


# ── 条件边函数 ─────────────────────────────────────────────────────────────

def route_decision(state: RAGState) -> str:
    """返回 retrieve / direct / tool"""
    return state.get("route", "retrieve")


def evaluate_decision(state: RAGState) -> str:
    """检索结果充分 → generate；不充分且可重试 → rewrite；否则强制 generate"""
    scores = state.get("scores", [])
    top_score = scores[0] if scores else 0.0
    retry = state.get("retry", 0)
    threshold = state.get("score_threshold") or SCORE_THRESHOLD
    max_r = state.get("max_retry_limit") or MAX_RETRY

    if top_score >= threshold:
        return "generate"
    if retry < max_r:
        return "rewrite"
    # 重试耗尽：检查是否启用 fallback
    if state.get("fallback_enabled", True):
        return "fallback"
    return "generate"


# ── 图构建 ─────────────────────────────────────────────────────────────────

def build_rag_graph(hybrid_retriever, reranker, llm=None, chunk_lookup: dict = None):
    """
    构建并编译 LangGraph RAG 图。

    Args:
        hybrid_retriever: Level 2 的 EnsembleRetriever 实例
        reranker:         Level 2 的 BGEReranker 实例
        llm:              LLM 实例，None 时各节点自动初始化
        chunk_lookup:     {chunk_id: Document} 字典，用于 retrieve_node 邻居扩展

    Returns:
        编译好的 CompiledGraph，调用方式：
            graph.invoke({"question": "你的问题"})
    """
    _llm = llm or get_llm()

    # 把外部依赖绑定到节点函数
    from functools import partial
    _condense      = partial(condense_node,      llm=_llm)
    _router        = partial(router_node,        llm=_llm)
    _decompose     = partial(decompose_node,     hybrid_retriever=hybrid_retriever, reranker=reranker, llm=_llm)
    _retrieve      = partial(retrieve_node,      hybrid_retriever=hybrid_retriever, reranker=reranker, llm=_llm, chunk_lookup=chunk_lookup)
    _rewrite       = partial(rewrite_node,       llm=_llm)
    _generate      = partial(generate_node,      llm=_llm)
    _direct        = partial(direct_node,        llm=_llm)
    _fallback      = partial(fallback_node,      llm=_llm)
    _tool_executor = partial(tool_executor_node, llm=_llm)

    builder = StateGraph(RAGState)

    builder.add_node("condense",      _condense)
    builder.add_node("router",        _router)
    builder.add_node("decompose",     _decompose)
    builder.add_node("retrieve",      _retrieve)
    builder.add_node("rewrite",       _rewrite)
    builder.add_node("generate",      _generate)
    builder.add_node("direct",        _direct)
    builder.add_node("fallback",      _fallback)
    builder.add_node("tool_executor", _tool_executor)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "router")
    builder.add_conditional_edges(
        "router", route_decision,
        # retrieve 路由先经过 decompose，其他路由不变
        {"retrieve": "decompose", "direct": "direct", "tool": "tool_executor"},
    )
    builder.add_conditional_edges(
        "decompose", decompose_decision,
        # 未拆解 → retrieve（正常流程）；已拆解 → generate（docs 已填充）
        {"retrieve": "retrieve", "generate": "generate"},
    )
    builder.add_conditional_edges(
        "retrieve", evaluate_decision,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    builder.add_edge("rewrite",       "retrieve")
    builder.add_edge("generate",      END)
    builder.add_edge("direct",        END)
    builder.add_edge("fallback",      END)
    builder.add_edge("tool_executor", END)

    return builder.compile()
