"""
全局配置文件
所有模块从这里读取配置，不要在各文件里硬编码。
"""

import os
import subprocess


def check_gpu() -> None:
    """
    检查 GPU 显存占用情况。
    若有残留进程，打印 PID 列表供手动处理（不自动 kill）。
    手动清理：kill <PID>
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    procs = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]

    if not procs:
        print("GPU 状态：无占用进程 ✓")
        return

    print(f"⚠️  GPU 有 {len(procs)} 个占用进程，若为残留请手动 kill：")
    for p in procs:
        pid = p.split(",")[0].strip()
        print(f"  {p}    →  kill {pid}")

# ── LLM（DashScope / OpenAI-compatible） ──────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = "qwen-plus"          # 生成用，省钱；换 claude 见下方注释
LLM_TEMPERATURE = 0.1

# 若要用 Claude（需要设置 ANTHROPIC_API_KEY 环境变量）：
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-sonnet-4-6")

# ── Embedding ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"   # 本地，中文 MTEB 强
EMBEDDING_DEVICE = "cuda"                           # 没 GPU 改 "cpu"

# ── Reranker ──────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"   # 本地，中英双语
RERANKER_TOP_N = 3                                  # 精排后保留几个

# ── 向量数据库（Chroma） ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.expanduser("~/rag_project/data/chroma_db")
CHROMA_COLLECTION_NAME = "rag_docs"

# ── 检索参数 ──────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = 10      # 初检 Top-K（BM25 和向量各取这么多，再 RRF 融合）
RERANKER_TOP_N = 3        # 精排后保留

# ── Tavily 网络搜索（Fallback 方案B） ────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

# ── 切分参数 ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
