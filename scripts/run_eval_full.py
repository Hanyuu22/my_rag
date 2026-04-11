"""
分类评估脚本：对 QA 数据集跑 RAGAS 四指标，按题型/难度/主题分组统计

流程：
  1. 读取 QA 数据集（generate_qa_v2.py 的输出）
  2. 对每道题调用 RAG graph，收集 answer + retrieved contexts
  3. 用 RAGAS 评估 Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall
  4. 输出整体均值 + 分类细分报告

用法：
  conda activate mineru_2.5
  cd ~/rag_project

  # 先用小样本验证（每类各取 5 题）
  python scripts/run_eval_full.py \\
      --qa_file  data/qa_dataset/gb_qa_2000.json \\
      --collection gb_standards \\
      --sample_per_type 5

  # 完整评估（耗时较长，建议 --max 100 先跑一批）
  python scripts/run_eval_full.py \\
      --qa_file  data/qa_dataset/gb_qa_2000.json \\
      --collection gb_standards \\
      --max 100
"""

import sys
import json
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/hanyuu/rag_project")


# ── RAG 调用 ──────────────────────────────────────────────────────────────────

def call_rag(question: str, graph) -> dict:
    """
    直接调用 RAG graph，返回 answer 和 retrieved contexts。
    不走 HTTP，直接 graph.invoke，避免网络开销。
    """
    try:
        result = graph.invoke({
            "question": question,
            "score_threshold": 0.3,     # 评估时适当放宽，确保有内容返回
            "max_retry_limit": 1,
            "reranker_enabled": True,
            "fallback_enabled": False,   # 评估时关 fallback，确保考察的是 RAG 本身
            "tools_enabled": False,
            "cross_lingual_enabled": False,
            "token_callback": None,
        })
        answer = result.get("answer", "")
        docs   = result.get("docs", [])
        contexts = [d.page_content for d in docs[:3]]
        return {"answer": answer, "contexts": contexts, "ok": True}
    except Exception as e:
        return {"answer": "", "contexts": [], "ok": False, "error": str(e)}


# ── 采样逻辑 ──────────────────────────────────────────────────────────────────

def sample_questions(qa_list: list[dict], sample_per_type: int | None,
                     max_total: int | None, seed: int = 42) -> list[dict]:
    random.seed(seed)

    if sample_per_type:
        # 按题型均匀采样
        by_type = defaultdict(list)
        for qa in qa_list:
            by_type[qa.get("question_type", "unknown")].append(qa)
        sampled = []
        for qtype, items in by_type.items():
            n = min(sample_per_type, len(items))
            sampled.extend(random.sample(items, n))
        return sampled

    if max_total and len(qa_list) > max_total:
        return random.sample(qa_list, max_total)

    return qa_list


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run(qa_file: str, collection: str, sample_per_type: int | None,
        max_total: int | None, output: str):

    # 加载 QA 数据集
    with open(qa_file, encoding="utf-8") as f:
        qa_list = json.load(f)
    print(f"加载 QA 数据集：{len(qa_list)} 条")

    questions = sample_questions(qa_list, sample_per_type, max_total)
    print(f"本次评估：{len(questions)} 条\n")

    # 构建 RAG graph
    print("初始化 RAG graph...")
    from backend.routers.chat import _build_graph
    graph = _build_graph(collection, top_k=10, reranker_top_n=3)
    print("初始化完成\n")

    # 对每道题调用 RAG
    samples_for_ragas = []
    failed = 0

    for idx, qa in enumerate(questions):
        q    = qa["question"]
        ref  = qa["ground_truth"]
        qtype = qa.get("question_type", "unknown")
        diff  = qa.get("difficulty",    "unknown")
        topic = qa.get("topic_tag",     "其他")

        print(f"[{idx+1}/{len(questions)}] [{qtype}] {q[:50]}...", end="  ")
        t0 = time.time()
        ret = call_rag(q, graph)
        elapsed = time.time() - t0

        if not ret["ok"] or not ret["answer"]:
            print(f"❌ ({elapsed:.1f}s)")
            failed += 1
            continue

        print(f"✅ ({elapsed:.1f}s)")
        samples_for_ragas.append({
            "question":      q,
            "answer":        ret["answer"],
            "contexts":      ret["contexts"],
            "reference":     ref,
            "question_type": qtype,
            "difficulty":    diff,
            "topic_tag":     topic,
        })

    print(f"\nRAG 调用完成：成功 {len(samples_for_ragas)}，失败 {failed}")

    if not samples_for_ragas:
        print("无有效样本，退出。")
        return

    # RAGAS 整体评估
    print("\n运行 RAGAS 评估...")
    from evaluation.ragas_eval import run_ragas
    overall = run_ragas(samples_for_ragas, show_progress=True)

    print("\n" + "="*55)
    print("【整体指标】")
    for k, v in overall.items():
        print(f"  {k:<25} {v:.4f}")

    # 按题型分组统计
    print("\n【按题型细分】")
    _print_breakdown(samples_for_ragas, "question_type")

    print("\n【按难度细分】")
    _print_breakdown(samples_for_ragas, "difficulty")

    print("\n【按主题细分】")
    _print_breakdown(samples_for_ragas, "topic_tag")

    # 保存结果
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "overall":        overall,
        "n_evaluated":    len(samples_for_ragas),
        "n_failed":       failed,
        "by_type":        _group_stats(samples_for_ragas, "question_type"),
        "by_difficulty":  _group_stats(samples_for_ragas, "difficulty"),
        "by_topic":       _group_stats(samples_for_ragas, "topic_tag"),
        "samples":        samples_for_ragas,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n结果已保存：{output_path}")


def _group_stats(samples: list[dict], group_key: str) -> dict:
    """对样本按某字段分组，各组独立跑 RAGAS"""
    from evaluation.ragas_eval import run_ragas

    groups = defaultdict(list)
    for s in samples:
        groups[s.get(group_key, "unknown")].append(s)

    result = {}
    for gname, gsamples in sorted(groups.items()):
        try:
            metrics = run_ragas(gsamples, show_progress=False)
            result[gname] = {"n": len(gsamples), **metrics}
        except Exception as e:
            result[gname] = {"n": len(gsamples), "error": str(e)}
    return result


def _print_breakdown(samples: list[dict], group_key: str):
    groups = defaultdict(list)
    for s in samples:
        groups[s.get(group_key, "unknown")].append(s)

    from evaluation.ragas_eval import run_ragas
    for gname, gsamps in sorted(groups.items()):
        try:
            m = run_ragas(gsamps, show_progress=False)
            cp = m.get("context_precision", 0)
            cr = m.get("context_recall",    0)
            f  = m.get("faithfulness",      0)
            ar = m.get("answer_relevancy",  0)
            print(f"  {gname:<15} n={len(gsamps):>3}  "
                  f"CP={cp:.3f}  CR={cr:.3f}  Faith={f:.3f}  AR={ar:.3f}")
        except Exception as e:
            print(f"  {gname:<15} n={len(gsamps):>3}  ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 分类评估")
    parser.add_argument("--qa_file",         default="data/qa_dataset/gb_qa_2000.json")
    parser.add_argument("--collection",      default="gb_standards")
    parser.add_argument("--sample_per_type", type=int, default=None,
                        help="每种题型取多少题（用于快速验证，默认不限）")
    parser.add_argument("--max",             type=int, default=None,
                        help="最多评估多少题（不指定则全部）")
    parser.add_argument("--output",          default="evaluation/results/eval_report.json")
    args = parser.parse_args()

    run(args.qa_file, args.collection, args.sample_per_type, args.max, args.output)
