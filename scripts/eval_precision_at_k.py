"""
Precision@K 评估脚本

原理：
  对每条 QA，用 Hybrid Retriever 检索 Top-K 个 chunk，
  检查 ground_truth_chunk_id 是否出现在结果中。

指标：
  Hit@1  ：Top-1 命中率（最严格）
  Hit@3  ：Top-3 命中率
  Hit@5  ：Top-5 命中率
  MRR    ：Mean Reciprocal Rank，衡量 ground truth 排名位置
  按题型/难度/主题分组输出，找出薄弱环节

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python scripts/eval_precision_at_k.py \
      --input      data/qa_dataset/gb_qa_1000_annotated.json \
      --collection gb_standards \
      --top_k      5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from retrievers.hybrid_retriever import build_hybrid_retriever
from chains.rag_chain import get_embeddings, get_collection_embedding_model

# ── 主流程 ────────────────────────────────────────────────────────────────────

def evaluate(input_path: str, collection_name: str, top_k: int):
    input_path = Path(input_path)

    with open(input_path, encoding="utf-8") as f:
        qa_list = json.load(f)

    # 只评估有 chunk_id 标注的
    valid = [qa for qa in qa_list if qa.get("ground_truth_chunk_id")]
    print(f"总题数：{len(qa_list)}，有效标注：{len(valid)}\n")

    # 构建检索器
    print("构建 Hybrid Retriever...")
    from chains.rag_chain import load_vectorstore
    vectorstore = load_vectorstore(collection_name)
    # 先从 Chroma 取所有 chunks 供 BM25 使用
    results = vectorstore._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document as LCDoc
    chunks = [
        LCDoc(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)
    print(f"检索器就绪，top_k={top_k}\n")

    # 评估
    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum  = 0.0

    # 分组统计
    group_stats = defaultdict(lambda: {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0})

    for i, qa in enumerate(valid):
        question    = qa["question"]
        gt_id       = qa["ground_truth_chunk_id"]
        q_type      = qa.get("question_type", "unknown")
        difficulty  = qa.get("difficulty",    "unknown")
        topic       = qa.get("topic_tag",     "其他")

        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"  ⚠️ [{i+1}] 检索失败：{e}")
            continue

        gt_ids = set(qa.get("ground_truth_chunk_ids", [gt_id]))
        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]
        rank = None
        for pos, cid in enumerate(retrieved_ids, 1):
            if cid and cid in gt_ids:
                rank = pos
                break

        # 更新指标
        if rank is not None:
            if rank <= 1: hits_at[1] += 1
            if rank <= 3: hits_at[3] += 1
            if rank <= 5: hits_at[5] += 1
            rr_sum += 1.0 / rank

        # 分组统计（按题型）
        for group_key in [q_type, difficulty, topic]:
            group_stats[group_key]["total"] += 1
            if rank and rank <= 1: group_stats[group_key]["hit1"] += 1
            if rank and rank <= 3: group_stats[group_key]["hit3"] += 1
            if rank and rank <= 5: group_stats[group_key]["hit5"] += 1

        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(valid)}  "
                  f"Hit@1={hits_at[1]/(i+1):.3f}  "
                  f"Hit@3={hits_at[3]/(i+1):.3f}")

    n = len(valid)
    print(f"\n{'='*55}")
    print(f"评估完成（共 {n} 条）\n")
    print(f"{'指标':<12} {'值':>8}")
    print(f"{'-'*22}")
    print(f"{'Hit@1':<12} {hits_at[1]/n:>8.3f}  ({hits_at[1]}/{n})")
    print(f"{'Hit@3':<12} {hits_at[3]/n:>8.3f}  ({hits_at[3]}/{n})")
    print(f"{'Hit@5':<12} {hits_at[5]/n:>8.3f}  ({hits_at[5]}/{n})")
    print(f"{'MRR':<12} {rr_sum/n:>8.3f}")

    print(f"\n{'--- 分组 Hit@3 ---'}")
    for group, stat in sorted(group_stats.items()):
        t = stat["total"]
        if t == 0: continue
        print(f"  {group:<18} Hit@1={stat['hit1']/t:.2f}  "
              f"Hit@3={stat['hit3']/t:.2f}  "
              f"Hit@5={stat['hit5']/t:.2f}  (n={t})")

    # 保存结果
    result = {
        "collection": collection_name,
        "n_evaluated": n,
        "hit_at_1": round(hits_at[1]/n, 4),
        "hit_at_3": round(hits_at[3]/n, 4),
        "hit_at_5": round(hits_at[5]/n, 4),
        "mrr":      round(rr_sum/n, 4),
        "group_stats": {
            k: {
                "total": v["total"],
                "hit1":  round(v["hit1"]/v["total"], 3) if v["total"] else 0,
                "hit3":  round(v["hit3"]/v["total"], 3) if v["total"] else 0,
                "hit5":  round(v["hit5"]/v["total"], 3) if v["total"] else 0,
            }
            for k, v in group_stats.items()
        }
    }
    out = Path("data/results") / f"precision_at_k_{collection_name}_top{top_k}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至：{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/qa_dataset/gb_qa_1000_annotated.json")
    parser.add_argument("--collection", default="gb_standards")
    parser.add_argument("--top_k",      type=int, default=5)
    args = parser.parse_args()

    evaluate(args.input, args.collection, args.top_k)
