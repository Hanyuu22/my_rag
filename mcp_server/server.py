"""
RAG Knowledge Base — MCP Server

把本项目的 RAG 系统暴露为 MCP 工具，供 Claude Desktop 等 MCP 客户端调用。

工具列表：
  list_collections        — 列出所有可用知识库
  search_knowledge_base   — 向量检索知识库文档
  analyze_process_data    — Pandas 分析工艺 DCS 时序数据

启动方式（stdio，Claude Desktop 使用）：
  conda activate mineru_2.5
  python ~/rag_project/mcp_server/server.py

Claude Desktop 配置（~/.config/claude/claude_desktop_config.json）：
  {
    "mcpServers": {
      "rag-knowledge-base": {
        "command": "/path/to/conda/envs/mineru_2.5/bin/python",
        "args": ["/home/hanyuu/rag_project/mcp_server/server.py"]
      }
    }
  }

HTTP/SSE 模式（可选，供其他客户端使用）：
  mcp.run(transport="sse")  # 默认端口 8001
"""

import sys
import asyncio

sys.path.insert(0, "/home/hanyuu/rag_project")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "rag-knowledge-base",
    instructions=(
        "这是一个基于 MinerU + LangChain 的工业文档 RAG 知识库系统。\n"
        "支持查询加氢裂化工艺规程、投资数据库，以及对 DCS 时序数据做数值分析。\n"
        "使用 list_collections 获取可用知识库列表，再用 search_knowledge_base 检索。"
    ),
)


# ── 工具定义 ───────────────────────────────────────────────────────────────

@mcp.tool()
def list_collections() -> str:
    """列出所有可用的知识库名称及文档片段数量"""
    try:
        import chromadb
        from config import CHROMA_PERSIST_DIR
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collections = client.list_collections()
        if not collections:
            return "暂无可用知识库"
        lines = [f"- {c.name}（{c.count()} 个文档片段）" for c in collections]
        return "可用知识库：\n" + "\n".join(lines)
    except Exception as e:
        return f"获取知识库列表失败：{e}"


@mcp.tool()
def search_knowledge_base(query: str, collection: str, top_k: int = 5) -> str:
    """
    在指定知识库中检索相关文档片段。

    Args:
        query:      检索查询语句，尽量具体
        collection: 知识库名称（通过 list_collections 获取）
        top_k:      返回文档片段数量，默认 5
    """
    from tools.tool_definitions import _search_kb_func
    return _search_kb_func(query=query, collection=collection)


@mcp.tool()
def analyze_process_data(query: str) -> str:
    """
    对加氢裂化装置 DCS 时序数据进行数值分析。

    适用于：统计温度/压力/流量的均值和极值、查询时段数据、分析趋势等。
    不适用于：文字描述类知识查询（请使用 search_knowledge_base）。

    Args:
        query: 数据分析问题，如"R101一床层入口温度的均值是多少"
    """
    from tools.data_analyzer import analyze_process_data as _analyze
    return _analyze(query)


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 默认使用 stdio（Claude Desktop）
    # 改为 mcp.run(transport="sse") 可切换到 HTTP 模式
    mcp.run(transport="stdio")
