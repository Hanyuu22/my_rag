"""
LangChain 工具定义

供 LangGraph router 的 bind_tools 调用，以及 MCP Server 直接使用。

工具列表：
  search_knowledge_base   — 检索本地知识库（专业文档 / 投资信息）
  analyze_process_data    — Pandas 分析工艺 DCS 时序数据
  web_search              — Tavily 互联网搜索

注意：在 LangGraph 中，search_knowledge_base 被选中后会路由到现有
      retrieve_node，而非直接执行此处的函数体。
      在 MCP Server 中，三个工具函数均会实际执行。
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ── 输入 Schema ────────────────────────────────────────────────────────────

class SearchKBInput(BaseModel):
    query: str = Field(description="检索查询语句，尽量具体")
    collection: str = Field(description="知识库名称（如 hydro_manual / investment_db）")


class AnalyzeDataInput(BaseModel):
    query: str = Field(description="数据分析问题，如：R101一床层温度均值、总进料流量趋势")


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词或问题")


# ── 工具函数（MCP / 直接调用时执行） ──────────────────────────────────────

def _search_kb_func(query: str, collection: str) -> str:
    """向量 + BM25 混合检索，返回 Top-5 文档片段"""
    try:
        from langchain_chroma import Chroma
        from chains.rag_chain import get_embeddings
        from config import CHROMA_PERSIST_DIR
        vectorstore = Chroma(
            collection_name=collection,
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=get_embeddings(),
        )
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return f"知识库 '{collection}' 中未找到相关内容"
        parts = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "?")
            parts.append(f"[片段{i+1}] 页:{page}\n{doc.page_content[:300]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"检索失败：{e}"


def _analyze_data_func(query: str) -> str:
    """调用 data_analyzer 对工艺 DCS 数据做 pandas 分析"""
    from tools.data_analyzer import analyze_process_data
    return analyze_process_data(query)


def _web_search_func(query: str) -> str:
    """Tavily 互联网搜索，返回 Top-3 结果摘要"""
    try:
        from tavily import TavilyClient
        from config import TAVILY_API_KEY
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(query=query, max_results=3, search_depth="basic")
        results = resp.get("results", [])
        if not results:
            return "未找到相关网络搜索结果"
        parts = [
            f"[{r.get('title', '')}]\n{r.get('content', '')[:400]}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"网络搜索失败：{e}"


# ── StructuredTool 实例 ────────────────────────────────────────────────────

TOOL_SEARCH_KB = StructuredTool.from_function(
    func=_search_kb_func,
    name="search_knowledge_base",
    description=(
        "在本地知识库中检索专业文档。适用于：加氢裂化工艺规程、操作步骤、"
        "技术参数、投资机构/项目信息等需要查阅资料才能回答的问题。"
        "遇到专业性问题优先选此工具。"
    ),
    args_schema=SearchKBInput,
)

TOOL_ANALYZE_DATA = StructuredTool.from_function(
    func=_analyze_data_func,
    name="analyze_process_data",
    description=(
        "对加氢裂化装置 DCS 时序数据进行数值分析。"
        "适用于：统计均值/极值、时段筛选、温度压力趋势等数值计算类问题。"
        "不适合用于文字描述类知识查询。"
    ),
    args_schema=AnalyzeDataInput,
)

TOOL_WEB_SEARCH = StructuredTool.from_function(
    func=_web_search_func,
    name="web_search",
    description=(
        "在互联网搜索最新信息。适用于：知识库未覆盖的外部信息、"
        "市场动态、最新政策等需要网络资料的问题。"
    ),
    args_schema=WebSearchInput,
)

# 供 bind_tools 和 MCP 使用的完整工具列表
ALL_TOOLS = [TOOL_SEARCH_KB, TOOL_ANALYZE_DATA, TOOL_WEB_SEARCH]
