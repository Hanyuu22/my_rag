"""
Level 4 — RAGAS 多级对比评估

对 Level 1 / Level 2 / Level 3 用同一批问题分别跑，
对比四个 RAGAS 指标，量化每级改动的效果。

用法：
    conda activate mineru_2.5
    cd ~/rag_project
    python tests/test_level4.py
"""

import sys
import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, "/home/hanyuu/rag_project")

from chains.rag_chain import build_vectorstore, get_llm, format_docs, RAG_PROMPT
from langchain_core.output_parsers import StrOutputParser

# ── 测试问题 + Ground Truth ────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "加氢裂化装置的主要反应条件是什么？",
        "reference": (
            "加氢裂化装置的主要反应条件包括：反应温度（最重要的工艺参数，需严格控制，"
            "通过各床层注入冷氢控制床层温升，确保每段温升不超过10~20°C）、"
            "氢分压（高压有利于加氢反应进行，抑制结焦）、"
            "空速LHSV（影响反应深度和催化剂寿命）、"
            "氢油比（影响反应效果）。"
            "温度升高加快裂化速度，但过高导致加氢平衡转化率下降。"
        ),
    },
    {
        "question": "装置开工前需要哪些准备工作？",
        "reference": (
            "装置开工前准备工作包括：（1）生产主管部门组织各专业全面确认验收，"
            "现场做到工完、料净、场地清；（2）设备与系统检查：人孔、法兰、螺丝、垫片完好，"
            "机泵润滑油、密封、冷却水正常，电机绝缘接地合格，盲板管理到位；"
            "（3）安全设施检查：通风、通讯、消防、防毒器材、可燃气体报警仪、照明全面检查；"
            "（4）试压试漏：压力容器、储罐、管线按规定进行试压、试漏和气密试验；"
            "（5）传动设备单体试车、安全装置调试复位；"
            "（6）严格执行开工方案中的安全技术措施和环境保护措施。"
        ),
    },
    {
        "question": "反应器的操作温度和压力范围是多少？",
        "reference": (
            "加氢裂化反应器操作条件：全循环末期各床层出口温度374~392°C，"
            "平均反应温度约374~392°C；一次通过末期各床层入口温度357~384°C，"
            "出口温度384~392°C，平均反应温度为379°C和387°C。"
            "操作氢分压约9.8MPa，空速1.0~15.0 h⁻¹。"
        ),
    },
    {
        "question": "冷氢的作用是什么？",
        "reference": (
            "冷氢（急冷氢）用于控制催化剂床层温度，防止加氢裂化强放热反应导致"
            "床层温度过高而损坏催化剂。在各床层之间注入冷氢，调节下部床层入口温度，"
            "确保每段床层温升不大于10~20°C。床层入口温度通过调节急冷氢量来控制。"
        ),
    },
]

# ── 数据准备（与 test_level2/3 保持一致）──────────────────────────────────

CONTENT_LIST_PATH = Path(
    "/home/hanyuu/rag_project/data/mineru_output"
    "/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto"
    "/content_list_with_tables.json"
)


def prepare_chunks():
    from loaders.mineru_loader import MinerULoader
    from splitters.markdown_splitter import split_documents
    docs = MinerULoader(str(CONTENT_LIST_PATH)).load()
    return split_documents(docs)


# ── Level 1 Runner ─────────────────────────────────────────────────────────

def run_level1(question: str, vectorstore) -> Tuple[str, List[str]]:
    """Level 1：纯 Dense 检索 + 基础 LCEL 链"""
    from config import RETRIEVER_TOP_K
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )
    docs = retriever.invoke(question)
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": question,
    })
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


# ── Level 2 Runner ─────────────────────────────────────────────────────────

def run_level2(question: str, hybrid_retriever, reranker) -> Tuple[str, List[str]]:
    """Level 2：Hybrid Retriever + Reranker（无 LangGraph）"""
    raw_docs = hybrid_retriever.invoke(question)
    scored = reranker.rerank(question, raw_docs)
    docs = [doc for _, doc in scored]
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": question,
    })
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


# ── Level 3 Runner ─────────────────────────────────────────────────────────

def run_level3(question: str, graph) -> Tuple[str, List[str]]:
    """Level 3：完整 LangGraph（路由 / 改写 / 生成）"""
    result = graph.invoke({"question": question})
    answer = result.get("answer", "")
    docs = result.get("docs", [])
    contexts = [doc.page_content for doc in docs] if docs else []
    return answer, contexts


# ── 评估单级 ───────────────────────────────────────────────────────────────

def collect_samples(
    test_cases: List[Dict],
    runner_fn,
    level_name: str,
) -> List[Dict[str, Any]]:
    """对一批问题跑 runner，收集 RAGAS 所需的样本"""
    samples = []
    for tc in test_cases:
        q = tc["question"]
        print(f"  [{level_name}] 问：{q[:25]}...", end="  ", flush=True)
        t0 = time.time()
        answer, contexts = runner_fn(q)
        elapsed = time.time() - t0
        print(f"✓ {elapsed:.1f}s")
        samples.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "reference": tc["reference"],
        })
    return samples


# ── 打印结果 ───────────────────────────────────────────────────────────────

def print_comparison(results: Dict[str, Dict[str, float]]):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    header = f"{'指标':<20}" + "".join(f"{'Level'+str(i+1):>12}" for i in range(len(results)))
    sep = "─" * len(header)

    print(f"\n{'📊 RAGAS 多级对比':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for m in metrics:
        row = f"{m:<20}"
        for level_scores in results.values():
            val = level_scores.get(m)
            row += f"{'N/A':>12}" if val is None else f"{val:>12.4f}"
        print(row)
    print(sep)

    # 找出每个指标最高的 level
    print("\n最优 Level（各指标）：")
    for m in metrics:
        scores = {lv: s.get(m) for lv, s in results.items() if s.get(m) is not None}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {m:<20} → {best}  ({scores[best]:.4f})")


# ── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("/home/hanyuu/rag_project/data/results") / f"test_level4_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("测试：test_level4 — RAGAS 多级对比评估")
    print(f"时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 准备数据 ──
    print("\n准备数据...")
    chunks = prepare_chunks()
    print(f"  chunks: {len(chunks)}")

    # ── 初始化组件 ──
    print("\n初始化组件...")
    vectorstore = build_vectorstore(chunks)
    # 若向量库文档数和 chunks 数差距过大，说明是旧数据，强制重建
    actual_count = vectorstore._collection.count()
    if actual_count < len(chunks) * 0.5:
        print(f"  ⚠️  向量库只有 {actual_count} 条（chunks={len(chunks)}），强制重建...")
        vectorstore = build_vectorstore(chunks, force_rebuild=True)
    print("  Vectorstore ✓")

    from retrievers.hybrid_retriever import build_hybrid_retriever
    from retrievers.reranker import BGEReranker
    hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
    print("  Hybrid Retriever ✓")
    reranker = BGEReranker()
    print("  BGE Reranker ✓")

    from graphs.rag_graph import build_rag_graph
    llm = get_llm()
    graph = build_rag_graph(hybrid_retriever, reranker, llm)
    print("  LangGraph ✓")

    # ── 收集各 Level 答案 ──
    print("\n─── 收集 Level 1 答案 ───")
    samples_l1 = collect_samples(
        TEST_CASES,
        lambda q: run_level1(q, vectorstore),
        "L1",
    )

    print("\n─── 收集 Level 2 答案 ───")
    samples_l2 = collect_samples(
        TEST_CASES,
        lambda q: run_level2(q, hybrid_retriever, reranker),
        "L2",
    )

    print("\n─── 收集 Level 3 答案 ───")
    samples_l3 = collect_samples(
        TEST_CASES,
        lambda q: run_level3(q, graph),
        "L3",
    )

    # ── RAGAS 评估 ──
    from evaluation.ragas_eval import run_ragas

    print("\n─── RAGAS 评估 Level 1 ───")
    scores_l1 = run_ragas(samples_l1)
    print(f"  {scores_l1}")

    print("\n─── RAGAS 评估 Level 2 ───")
    scores_l2 = run_ragas(samples_l2)
    print(f"  {scores_l2}")

    print("\n─── RAGAS 评估 Level 3 ───")
    scores_l3 = run_ragas(samples_l3)
    print(f"  {scores_l3}")

    # ── 对比输出 ──
    results = {
        "Level1": scores_l1,
        "Level2": scores_l2,
        "Level3": scores_l3,
    }
    print_comparison(results)

    # ── 保存日志 ──
    log_data = {
        "timestamp": ts,
        "chunks": len(chunks),
        "test_cases": [tc["question"] for tc in TEST_CASES],
        "scores": results,
        "samples": {
            "level1": samples_l1,
            "level2": samples_l2,
            "level3": samples_l3,
        },
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n日志已保存到：{log_path}")


if __name__ == "__main__":
    main()
