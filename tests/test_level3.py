"""
Level 3 集成测试

测试 LangGraph 多步 RAG 图，验证：
1. 路由节点是否正确区分"需检索"vs"直接回答"
2. 检索质量不足时是否触发 query rewriting
3. 改写后检索结果是否有提升
4. 整体答案质量 vs Level 2
"""

import sys, warnings, os, time
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/hanyuu/rag_project")

from config import check_gpu
check_gpu()

from loaders.word_loader import WordLoader
from splitters.markdown_splitter import split_documents
from chains.rag_chain import build_vectorstore, get_llm
from retrievers.hybrid_retriever import build_hybrid_retriever
from retrievers.reranker import BGEReranker
from graphs.rag_graph import build_rag_graph, SCORE_THRESHOLD
from utils.result_logger import ResultLogger

DOC_PATH = "/home/hanyuu/rag_project/data/raw/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节.doc"
COLLECTION = "hydro_manual"

# 测试问题：前3个和 Level 2 相同（方便对比），后2个是新场景
QUESTIONS = [
    # Level 2 同款，用于对比
    ("retrieve", "加氢裂化装置的主要反应条件是什么？"),
    ("retrieve", "装置开工前需要哪些准备工作？"),
    ("retrieve", "反应器的操作温度和压力范围是多少？"),   # 预期触发 rewrite
    # 路由测试
    ("direct",   "你好，帮我介绍一下你自己"),             # 预期走 direct
    ("retrieve", "冷氢的作用是什么？"),                   # 专业术语，预期走 retrieve
]


def run_and_log(graph, question, expected_route, log):
    log.section(f"问：{question}  [预期路由: {expected_route}]")
    t0 = time.time()

    result = graph.invoke({
        "question": question,
        "query": "",
        "docs": [],
        "scores": [],
        "answer": "",
        "retry": 0,
        "route": "",
    })

    elapsed = time.time() - t0
    actual_route = result.get("route", "?")
    retry = result.get("retry", 0)
    scores = result.get("scores", [])
    top_score = scores[0] if scores else None

    route_ok = "✅" if actual_route == expected_route else "❌"
    log(f"路由：{actual_route}  {route_ok}（预期 {expected_route}）")
    if retry > 0:
        log(f"Query 改写：触发 {retry} 次  最终 query：{result.get('query', '')}")
    if scores:
        log(f"精排分数：{scores}  Top1: {top_score:.3f}  "
            f"{'≥阈值' if top_score >= SCORE_THRESHOLD else '< 阈值（已改写）'}")

    log(f"\n答：{result['answer']}")

    if result.get("docs"):
        log(f"\n来源（Top-{len(result['docs'])}）：")
        for i, doc in enumerate(result["docs"], 1):
            score_str = f"score={scores[i-1]:.3f}  " if i <= len(scores) else ""
            log(f"  [{i}] {score_str}p.{doc.metadata.get('page')}  "
                f"{doc.page_content[:80]}...")

    log.metrics({
        "路由结果":     actual_route,
        "改写次数":     retry,
        "Top1置信分":  top_score if top_score is not None else "N/A",
        "全链路耗时(s)": round(elapsed, 2),
    })

    return {
        "route_correct": actual_route == expected_route,
        "retry": retry,
        "top_score": top_score,
        "elapsed": elapsed,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────

with ResultLogger("test_level3") as log:

    log.section("准备数据")
    docs = WordLoader(DOC_PATH, source_name="hydro_manual").load()
    chunks = split_documents(docs)
    log(f"chunks: {len(chunks)}")
    vectorstore = build_vectorstore(chunks, collection_name=COLLECTION)

    log.section("初始化组件")
    hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
    reranker = BGEReranker()
    llm = get_llm()
    log("Hybrid Retriever ✓")
    log("BGE Reranker ✓")
    log("LLM ✓")

    log.section("构建 LangGraph")
    graph = build_rag_graph(hybrid_retriever, reranker, llm)
    log("RAG Graph 编译完成 ✓")
    log(f"节点：router → retrieve/direct → evaluate → generate/rewrite")

    all_results = []
    for expected_route, question in QUESTIONS:
        r = run_and_log(graph, question, expected_route, log)
        all_results.append(r)

    # 汇总
    log.section("整体指标汇总")
    route_acc = sum(r["route_correct"] for r in all_results) / len(all_results)
    rewrite_triggered = sum(1 for r in all_results if r["retry"] > 0)
    avg_elapsed = sum(r["elapsed"] for r in all_results) / len(all_results)
    retrieve_results = [r for r in all_results if r["top_score"] is not None]
    avg_top_score = sum(r["top_score"] for r in retrieve_results) / len(retrieve_results) if retrieve_results else 0

    log.metrics({
        "路由准确率":        f"{route_acc:.0%}",
        "触发改写的问题数":   rewrite_triggered,
        "平均Top1置信分":    round(avg_top_score, 3),
        "平均全链路耗时(s)":  round(avg_elapsed, 2),
    }, title="📊 汇总")

    log()
    log("路由阈值说明：top1_score < 0.5 → 触发 query rewriting（最多 2 次）")
