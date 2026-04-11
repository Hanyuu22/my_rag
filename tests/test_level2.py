"""
Level 2 集成测试

对比 Level 1（纯 Dense）vs Level 2（Hybrid + Rerank）的检索质量和速度。

评估指标（轻量级，无需 LLM 调用）：
- retrieval_time:  检索耗时（秒）
- rerank_time:     精排耗时（秒）
- generation_time: LLM 生成耗时（秒）
- total_time:      全链路耗时（秒）
- top1_score:      Reranker 给最高分文档的置信分（0~1，越高越相关）
- retrieved_count: 检索到的文档数量
"""

import sys, warnings, os, time
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hanyuu/rag_project")

from langchain_core.output_parsers import StrOutputParser

from config import check_gpu
check_gpu()  # 有残留进程时打印警告；check_gpu(auto_kill=True) 可自动清理

from loaders.word_loader import WordLoader
from splitters.markdown_splitter import split_documents
from chains.rag_chain import (
    build_vectorstore, get_llm, format_docs, RAG_PROMPT
)
from retrievers.hybrid_retriever import build_hybrid_retriever
from retrievers.reranker import BGEReranker
from utils.result_logger import ResultLogger

QUESTIONS = [
    "加氢裂化装置的主要反应条件是什么？",
    "装置开工前需要哪些准备工作？",
    "反应器的操作温度和压力范围是多少？",
]

DOC_PATH = "/home/hanyuu/rag_project/data/raw/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节.doc"
COLLECTION = "hydro_manual"


def run_with_metrics(query, hybrid_retriever, reranker, llm):
    """运行完整 Level 2 链路，返回结果 + 各阶段耗时"""
    metrics = {}
    t_total = time.time()

    t0 = time.time()
    retrieved_docs = hybrid_retriever.invoke(query)
    metrics["retrieval_time"] = time.time() - t0
    metrics["retrieved_count"] = len(retrieved_docs)

    t0 = time.time()
    scored_docs = reranker.rerank(query, retrieved_docs)
    metrics["rerank_time"] = time.time() - t0
    metrics["top1_score"] = scored_docs[0][0] if scored_docs else 0.0
    metrics["rerank_scores"] = [round(s, 3) for s, _ in scored_docs]
    final_docs = [doc for _, doc in scored_docs]

    t0 = time.time()
    answer = (RAG_PROMPT | llm | StrOutputParser()).invoke({
        "context": format_docs(final_docs),
        "question": query,
    })
    metrics["generation_time"] = time.time() - t0
    metrics["total_time"] = time.time() - t_total

    return answer, final_docs, scored_docs, metrics


# ── 主流程 ────────────────────────────────────────────────────────────────

with ResultLogger("test_level2") as log:

    log.section("准备数据")
    docs = WordLoader(DOC_PATH, source_name="hydro_manual").load()
    chunks = split_documents(docs)
    log(f"chunks: {len(chunks)}")

    vectorstore = build_vectorstore(chunks, collection_name=COLLECTION)

    log.section("初始化 Level 2 组件")
    hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
    reranker = BGEReranker()
    llm = get_llm()
    log("BM25 + Dense Hybrid Retriever ✓")
    log("BGE Reranker ✓")

    all_metrics = []

    for q in QUESTIONS:
        log.section(f"问：{q}")

        answer, final_docs, scored_docs, metrics = run_with_metrics(
            q, hybrid_retriever, reranker, llm
        )
        all_metrics.append(metrics)

        log(f"答：{answer}")

        log(f"\n来源（Reranker 精排后 Top-{len(final_docs)}）：")
        for i, (score, doc) in enumerate(scored_docs, 1):
            log(f"  [{i}] score={score:.3f}  "
                f"p.{doc.metadata.get('page')}  "
                f"{doc.page_content[:80]}...")

        log.metrics({
            "检索耗时(s)":    metrics["retrieval_time"],
            "召回文档数":      metrics["retrieved_count"],
            "精排耗时(s)":    metrics["rerank_time"],
            "Top1置信分":     metrics["top1_score"],
            "精排分数列表":    metrics["rerank_scores"],
            "生成耗时(s)":    metrics["generation_time"],
            "全链路耗时(s)":  metrics["total_time"],
        })

    # 汇总
    log.section("整体指标汇总")

    def avg(key):
        return sum(m[key] for m in all_metrics) / len(all_metrics)

    log.metrics({
        "平均检索耗时(s)":   avg("retrieval_time"),
        "平均精排耗时(s)":   avg("rerank_time"),
        "平均生成耗时(s)":   avg("generation_time"),
        "平均全链路(s)":     avg("total_time"),
        "平均Top1置信分":    avg("top1_score"),
    }, title="📊 平均指标")

    log()
    log("参考：top1_score > 0.9 → 检索结果高度相关")
    log("      top1_score < 0.5 → 建议检查 chunk 切分或扩充知识库")
