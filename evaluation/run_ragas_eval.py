"""
Step 2 — RAGAS 批量评估 + 报告生成

读取 draft_golden_set.json（或指定文件），对每道题计算 RAGAS 四指标，
输出详细报告到 evaluation/results/。

用法：
    conda activate mineru_2.5
    cd ~/rag_project
    python evaluation/run_ragas_eval.py
    python evaluation/run_ragas_eval.py --input evaluation/draft_golden_set.json
    python evaluation/run_ragas_eval.py --skip-fallback   # 跳过 Fallback/路由类题目

注意：
  - Fallback 测试题（route="direct"，或 category 含"Fallback"/"路由"）
    不参与 RAGAS 评分（它们的 context 为空，会污染指标）
  - reference 为空的题目会跳过 context_precision / context_recall（不需要 ground truth）
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAGAS 批量评估")
    parser.add_argument("--input",  default="evaluation/draft_golden_set.json",
                        help="草稿文件路径（默认：evaluation/draft_golden_set.json）")
    parser.add_argument("--output-dir", default="evaluation/results",
                        help="结果输出目录（默认：evaluation/results）")
    parser.add_argument("--skip-fallback", action="store_true",
                        help="跳过 Fallback 测试和路由测试类题目（默认已自动过滤）")
    return parser.parse_args()


# ── 过滤逻辑 ──────────────────────────────────────────────────────────────────

SKIP_CATEGORIES = {"Fallback测试（知识库外）", "路由测试（闲聊）"}

def should_skip(item: dict) -> bool:
    """Fallback/路由测试题、answer 为空的题目不参与 RAGAS 评分"""
    if item["category"] in SKIP_CATEGORIES:
        return True
    if not item.get("answer"):
        return True
    if item.get("route") == "error":
        return True
    return False


# ── RAGAS 评估 ────────────────────────────────────────────────────────────────

def run_ragas(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对一批样本跑 RAGAS，返回 {per_sample: [...], aggregate: {...}}。

    对有 reference 的样本跑四指标；
    对 reference 为空的样本只跑 faithfulness + answer_relevancy。
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from chains.rag_chain import get_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())
    run_config = RunConfig(timeout=120, max_retries=2, max_wait=30)

    # 分组：有 reference 的用四指标，无 reference 的只用两指标
    with_ref    = [s for s in samples if s.get("reference")]
    without_ref = [s for s in samples if not s.get("reference")]

    all_results: Dict[int, dict] = {}

    # ── 四指标批次 ──
    if with_ref:
        print(f"\n[RAGAS] 有 reference 的题目：{len(with_ref)} 道（跑四指标）")
        eval_samples = [
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["reference"],
            )
            for s in with_ref
        ]
        dataset = EvaluationDataset(samples=eval_samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm, embeddings=ragas_emb,
            show_progress=True, raise_exceptions=False,
            run_config=run_config,
        )
        for i, s in enumerate(with_ref):
            all_results[s["id"]] = {
                "faithfulness":       _safe_score(result, "faithfulness",     i),
                "answer_relevancy":   _safe_score(result, "answer_relevancy", i),
                "context_precision":  _safe_score(result, "context_precision", i),
                "context_recall":     _safe_score(result, "context_recall",    i),
            }

    # ── 两指标批次 ──
    if without_ref:
        print(f"\n[RAGAS] 无 reference 的题目：{len(without_ref)} 道（跑两指标）")
        eval_samples = [
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
            )
            for s in without_ref
        ]
        dataset = EvaluationDataset(samples=eval_samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm, embeddings=ragas_emb,
            show_progress=True, raise_exceptions=False,
            run_config=run_config,
        )
        for i, s in enumerate(without_ref):
            all_results[s["id"]] = {
                "faithfulness":      _safe_score(result, "faithfulness",     i),
                "answer_relevancy":  _safe_score(result, "answer_relevancy", i),
                "context_precision": None,
                "context_recall":    None,
            }

    return all_results


def _safe_score(result, key: str, idx: int):
    try:
        v = result[key][idx]
        return round(float(v), 4) if not np.isnan(v) else None
    except Exception:
        return None


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def build_report(items: List[dict], ragas_scores: Dict[int, dict]) -> dict:
    """合并原始数据与 RAGAS 分数，生成完整报告"""

    per_question = []
    for item in items:
        scores = ragas_scores.get(item["id"], {})
        per_question.append({
            "id":               item["id"],
            "category":         item["category"],
            "question":         item["question"],
            "answer":           item["answer"],
            "reference":        item.get("reference", ""),
            "route":            item.get("route", ""),
            "top1_score":       item["scores"][0] if item.get("scores") else None,
            "elapsed":          item.get("elapsed"),
            "faithfulness":     scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision":scores.get("context_precision"),
            "context_recall":   scores.get("context_recall"),
        })

    # 按类别聚合
    by_category: Dict[str, list] = {}
    for row in per_question:
        by_category.setdefault(row["category"], []).append(row)

    category_summary = {}
    for cat, rows in by_category.items():
        category_summary[cat] = {
            "count":            len(rows),
            "faithfulness":     _nanmean([r["faithfulness"]     for r in rows]),
            "answer_relevancy": _nanmean([r["answer_relevancy"] for r in rows]),
            "context_precision":_nanmean([r["context_precision"]for r in rows]),
            "context_recall":   _nanmean([r["context_recall"]   for r in rows]),
            "avg_top1_score":   _nanmean([r["top1_score"]       for r in rows]),
            "avg_elapsed":      _nanmean([r["elapsed"]          for r in rows]),
        }

    # 全局聚合（不含 Fallback/路由题）
    rag_rows = [r for r in per_question if r["category"] not in SKIP_CATEGORIES]
    aggregate = {
        "total_evaluated":  len(rag_rows),
        "faithfulness":     _nanmean([r["faithfulness"]     for r in rag_rows]),
        "answer_relevancy": _nanmean([r["answer_relevancy"] for r in rag_rows]),
        "context_precision":_nanmean([r["context_precision"]for r in rag_rows]),
        "context_recall":   _nanmean([r["context_recall"]   for r in rag_rows]),
        "avg_top1_score":   _nanmean([r["top1_score"]       for r in rag_rows]),
        "avg_elapsed":      _nanmean([r["elapsed"]          for r in rag_rows]),
    }

    return {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate":      aggregate,
        "by_category":    category_summary,
        "per_question":   per_question,
    }


def _nanmean(vals: list):
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return round(float(np.mean(clean)), 4)


# ── 控制台打印 ────────────────────────────────────────────────────────────────

def print_report(report: dict):
    agg = report["aggregate"]
    print("\n" + "="*60)
    print("  RAGAS 评估结果汇总")
    print("="*60)
    print(f"  评估题数:       {agg['total_evaluated']}")
    print(f"  Faithfulness:   {_fmt(agg['faithfulness'])}")
    print(f"  AnswerRelevancy:{_fmt(agg['answer_relevancy'])}")
    print(f"  ContextPrec:    {_fmt(agg['context_precision'])}")
    print(f"  ContextRecall:  {_fmt(agg['context_recall'])}")
    print(f"  平均Top1得分:   {_fmt(agg['avg_top1_score'])}")
    print(f"  平均耗时:       {_fmt(agg['avg_elapsed'])} s")
    print()

    print("─"*60)
    print("  各类别明细")
    print("─"*60)
    for cat, s in report["by_category"].items():
        print(f"  [{cat}] ({s['count']} 题)")
        print(f"    Faith={_fmt(s['faithfulness'])}  "
              f"Relev={_fmt(s['answer_relevancy'])}  "
              f"Prec={_fmt(s['context_precision'])}  "
              f"Rec={_fmt(s['context_recall'])}")
    print()

    print("─"*60)
    print("  逐题得分")
    print("─"*60)
    for row in report["per_question"]:
        if row["category"] in SKIP_CATEGORIES:
            status = f"[{row['route']}]"
            print(f"  #{row['id']:02d} {status} {row['question'][:40]}")
        else:
            print(f"  #{row['id']:02d} "
                  f"Faith={_fmt(row['faithfulness'])}  "
                  f"Relev={_fmt(row['answer_relevancy'])}  "
                  f"Prec={_fmt(row['context_precision'])}  "
                  f"Rec={_fmt(row['context_recall'])}  "
                  f"| {row['question'][:35]}")
    print("="*60)


def _fmt(v):
    if v is None:
        return " N/A "
    return f"{v:.4f}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[错误] 输入文件不存在：{in_path}")
        print("请先运行：python evaluation/run_generate.py --collection <集合名称>")
        sys.exit(1)

    with open(in_path, encoding="utf-8") as f:
        all_items = json.load(f)

    # 过滤出需要 RAGAS 评估的题目
    eval_items = [item for item in all_items if not should_skip(item)]
    skip_items = [item for item in all_items if should_skip(item)]

    print(f"[加载] 共 {len(all_items)} 条，参与 RAGAS 评估：{len(eval_items)} 条，跳过：{len(skip_items)} 条")
    print(f"  跳过原因：Fallback/路由测试题 或 answer 为空")

    if not eval_items:
        print("[警告] 没有可评估的题目，请先运行 run_generate.py 生成答案。")
        return

    # 运行 RAGAS
    ragas_scores = run_ragas(eval_items)

    # 构建报告（包含所有题目，Fallback 题标注为跳过）
    report = build_report(all_items, ragas_scores)

    # 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 同时更新一份 latest.json 方便直接查看
    latest_path = out_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"[保存] 完整报告：{out_path}")
    print(f"[保存] 最新结果：{latest_path}")


if __name__ == "__main__":
    main()
